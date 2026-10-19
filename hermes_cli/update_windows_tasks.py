"""Preserve verified SYSTEM gateway tasks across updates, without guessing from an exe.

Only gateways already established by strict profile PID/lock discovery enter here.
Task names alone are insufficient: the exported SYSTEM task must launch that home's
generated gateway script. The same definition is checked again before Stop or Run.
"""

from __future__ import annotations

import base64
import hashlib
import json
import ntpath
from pathlib import Path
import subprocess
import tempfile
from xml.etree import ElementTree

from hermes_cli.gateway_windows_task_control import process_is_system


_TASK_SCRIPT = r"""$ErrorActionPreference = 'Stop'
$r = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('__REQUEST__')) | ConvertFrom-Json
try {
    $s = New-Object -ComObject 'Schedule.Service'
    $s.Connect()
    $t = $s.GetFolder('\').GetTask($r.name)
    $xml = $t.Xml
    $hash = [Security.Cryptography.SHA256]::Create()
    try { $digest = ([BitConverter]::ToString($hash.ComputeHash([Text.Encoding]::UTF8.GetBytes($xml)))).Replace('-', '').ToLowerInvariant() }
    finally { $hash.Dispose() }
    if ($r.expected -and $r.expected -ne $digest) { exit 13 }
    switch ($r.operation) {
        'stop' { $t.Stop(0) }
        'run' { $null = $t.Run($null) }
        'query' { }
        default { exit 87 }
    }
    [IO.File]::WriteAllText($r.output, $xml, (New-Object Text.UTF8Encoding($false)))
    exit 0
} catch {
    [IO.File]::WriteAllText($r.output, $_.Exception.GetBaseException().Message, (New-Object Text.UTF8Encoding($false)))
    $code = $_.Exception.GetBaseException().HResult -band 65535
    if ($code -eq 5) { exit 5 }
    if ($code -eq 2 -or $code -eq 3) { exit 2 }
    exit 1
}
"""


class TaskOperationError(RuntimeError):
    """The Scheduler result distinguishes rejection from an uncertain stop request."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.not_started = code in {2, 5, 13, 87, 127, 1223}


def task_operation(name: str, operation: str, expected: str = "") -> str | None:
    """Run a fixed Task Scheduler operation, elevating only on Access Denied.

    A cancelled prompt, timeout or changed registration fails the update. Querying
    an absent task returns None. No repository Python is executed with elevation.
    """
    from hermes_cli.gateway_windows import _assert_windows
    from hermes_cli._subprocess_compat import windows_hide_flags
    from hermes_cli.gateway_windows_task_control import _shell_execute_elevated_and_wait
    from hermes_platform.resolver import locate_command

    _assert_windows()
    if operation not in {"query", "stop", "run"}:
        raise TaskOperationError(87, "Unsupported task operation")
    if not (name == "Hermes_Gateway" or name.startswith("Hermes_Gateway_")) or any(
        c in name for c in "\\/\r\n\0"
    ):
        raise TaskOperationError(87, "Expected a Hermes gateway task name")
    if operation != "query" and (
        len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected)
    ):
        raise TaskOperationError(
            87, "Task control requires the inspected definition fingerprint"
        )
    command = locate_command("powershell").command
    if not command:
        raise TaskOperationError(
            127, "PowerShell is required to inspect protected gateway tasks"
        )
    with tempfile.TemporaryDirectory(prefix="hermes-gateway-task-") as scratch:
        output = Path(scratch) / "task.xml"
        request = {
            "name": name,
            "operation": operation,
            "expected": expected,
            "output": str(output),
        }
        data = base64.b64encode(json.dumps(request).encode("utf-8")).decode("ascii")
        script = _TASK_SCRIPT.replace("__REQUEST__", data)
        encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        args = ["-NoProfile", "-NonInteractive", "-EncodedCommand", encoded]
        result = subprocess.run(
            [*command, *args],
            timeout=20,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=windows_hide_flags(),
        )
        code = result.returncode
        detail = ""
        if code == 5:
            code, detail = _shell_execute_elevated_and_wait(
                command[0], subprocess.list2cmdline(args), cwd=scratch, timeout_s=20
            )
        if code == 2 and operation == "query":
            return None
        if code:
            detail = detail or {
                2: "task registration is missing",
                5: "access was denied",
                13: "task registration changed after inspection",
                124: "task operation timed out",
                1223: "Windows elevation was cancelled",
            }.get(code, "Task Scheduler returned an error")
            if detail == "Task Scheduler returned an error" and output.is_file():
                with output.open("rb") as stream:
                    detail = stream.read(4096).decode("utf-8-sig", errors="replace")
            raise TaskOperationError(
                code, f"Gateway task {name} {operation} failed ({code}): {detail}"
            )
        with output.open("rb") as stream:
            raw = stream.read(262145)
        if len(raw) > 262144:
            raise RuntimeError("Gateway task definition exceeds the inspection limit")
        return raw.decode("utf-8-sig")


def definition_matches_home(xml: str, home: Path, task_name: str) -> bool:
    """Accept only a single SYSTEM action pointing at this home's exact launcher."""
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError:
        return False
    for element in root.iter():
        element.tag = element.tag.rsplit("}", 1)[-1]
    principal = root.findtext("Principals/Principal/UserId", "").strip().upper()
    if principal not in {"SYSTEM", "NT AUTHORITY\\SYSTEM", "S-1-5-18"}:
        return False
    actions = root.find("Actions")
    if actions is None or len(actions) != 1 or actions[0].tag != "Exec":
        return False
    action = actions[0]
    command = ntpath.normcase(
        ntpath.normpath(action.findtext("Command", "").strip('"'))
    )
    arguments = action.findtext("Arguments", "").strip().casefold().replace("/", "\\")
    directory = home / "gateway-service"
    vbs = ntpath.normcase(str(directory / f"{task_name}.vbs"))
    cmd = ntpath.normcase(str(directory / f"{task_name}.cmd"))
    if ntpath.basename(command) in {"wscript.exe", "cscript.exe"}:
        return arguments in {f'\\\\b \\\\nologo "{vbs}"', f'"{vbs}"'}
    if ntpath.basename(command) == "cmd.exe":
        return arguments in {f'\\c "{cmd}"', f'\\d \\s \\c "{cmd}"'}
    return command == cmd and not arguments


def discover_profile_tasks(profile_processes: dict, service_pids: set[int]) -> dict:
    """Map only strict, live SYSTEM gateway identities to inspected profile tasks."""
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from hermes_cli import gateway_windows
    import psutil
    from hermes_platform.host.facts import os_family

    if os_family() != "win32":
        return {}
    tasks = {}
    for pid, profile in profile_processes.items():
        if pid in service_pids:
            continue
        try:
            process = psutil.Process(pid)
            born = process.create_time()
        except psutil.NoSuchProcess:
            continue
        if not process_is_system(pid):
            continue
        if born != profile.create_time:
            raise RuntimeError("Gateway identity changed before task discovery")
        scope = set_hermes_home_override(profile.path)
        try:
            names = dict.fromkeys((gateway_windows.get_task_name(), "Hermes_Gateway"))
        finally:
            reset_hermes_home_override(scope)
        for name in names:
            xml = task_operation(name, "query")
            if xml is not None and definition_matches_home(
                xml, Path(profile.path), name
            ):
                tasks[str(profile.profile)] = {
                    "name": name,
                    "pid": pid,
                    "home": str(profile.path),
                    "created": born,
                    "definition": hashlib.sha256(xml.encode("utf-8")).hexdigest(),
                }
                break
        if str(profile.profile) not in tasks:
            raise RuntimeError(
                f"SYSTEM gateway {pid} has no verified Scheduled Task; nothing was stopped"
            )
    return tasks


def stop_profile_tasks(tasks: dict, survivors: set[int], on_request=None) -> None:
    """Stop only still-live recorded gateways, refusing identity or definition drift."""
    import psutil

    for task in tasks.values():
        if task["pid"] not in survivors:
            continue
        try:
            born = psutil.Process(task["pid"]).create_time()
        except psutil.NoSuchProcess:
            continue
        if born != task["created"]:
            raise RuntimeError("Gateway identity changed before task stop")
        # Stop is asynchronous. Persist the request on success or uncertainty,
        # even when the profile's marker/socket could not accept a request.
        requested = True
        try:
            task_operation(task["name"], "stop", task["definition"])
        except TaskOperationError as exc:
            requested = not exc.not_started
            raise
        finally:
            if requested and on_request is not None:
                on_request(task["pid"])


def relaunch_profile(profile: str, old_pid: int, tasks: dict) -> bool:
    """A recorded SYSTEM task must never fall back to a current-user argv replay."""
    from hermes_cli.gateway import launch_detached_profile_gateway_restart
    from hermes_cli.profiles import get_profile_dir, profile_exists

    task = tasks.get(profile)
    if task is None:
        return launch_detached_profile_gateway_restart(profile, old_pid)
    if int(task["pid"]) != old_pid:
        raise RuntimeError("Task resume identity differs from the paused gateway")
    if not profile_exists(profile):
        raise RuntimeError("The paused gateway profile no longer exists")
    if Path(task["home"]).resolve() != Path(get_profile_dir(profile)).resolve():
        raise RuntimeError("Task resume home differs from the current profile")
    if task_operation(task["name"], "query", task["definition"]) is None:
        raise RuntimeError("The paused gateway's task registration is missing")
    refresh_task_launcher(task)
    task_operation(task["name"], "run", task["definition"])
    return True


def refresh_task_launcher(task: dict) -> None:
    """Retarget the exact legacy launcher to current Python without changing its task principal."""
    from hermes_cli import gateway_windows as windows
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    home = Path(task["home"])
    scope = set_hermes_home_override(home)
    try:
        settings = windows._launcher_settings(home)
        script = home / "gateway-service" / f"{task['name']}.cmd"
        windows._atomic_write(
            script,
            windows._build_gateway_cmd_script(*settings),
            script.with_suffix(".tmp"),
        )
        vbs = script.with_suffix(".vbs")
        windows._atomic_write(
            vbs,
            windows._build_gateway_vbs_script(*settings),
            vbs.with_suffix(".vbs.tmp"),
        )
    finally:
        reset_hermes_home_override(scope)
