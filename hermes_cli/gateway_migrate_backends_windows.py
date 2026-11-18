"""Windows facts for ``hermes gateway migrate --backends``: find the old backend, name what starts it,
and the steps that move it to the gateway by hand. Read-only.

Hermes never starts ``hermes dashboard`` / ``hermes serve`` at sign-in itself, so a Windows user who
keeps one running made a Task Scheduler task (``At log on``) or put a shortcut/script in the Startup
folder. The process scan has to work without ``wmic`` (Windows 11 24H2 and later ship without it):
psutil reads the same command lines. The ``hermes.exe`` launcher runs the real backend as a child with
the same arguments; only the child (the one that listens) is the backend.

Automatic migration is not offered on Windows (the plan exits 6); :func:`manual_steps` prints one
command per line that runs as typed in PowerShell or Command Prompt, without administrator rights.
"""

from __future__ import annotations

import contextlib
import functools
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree

_TASK_NS = "{http://schemas.microsoft.com/windows/2004/02/mit/task}"
_TASK_HEADER = re.compile(r"<!--\s*(\\[^>]*?)\s*-->")
_XML_DECL = re.compile(r"<\?xml[^>]*\?>")
_STARTUP_SUFFIXES = (".lnk", ".cmd", ".bat", ".vbs", ".ps1")


@functools.lru_cache(maxsize=1)
def wmic_missing() -> bool:
    """True on a Windows without ``wmic`` (11 24H2 and later), the only place this module's scan and
    ownership readers are used; where wmic works the report stays exactly main's."""
    return shutil.which("wmic") is None


def process_rows() -> list[tuple[int, str]]:
    """``(pid, command line)`` of every process this user can read, via psutil."""
    import psutil
    rows: list[tuple[int, str]] = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        with contextlib.suppress(Exception):
            argv = proc.info.get("cmdline") or []
            if argv:
                rows.append((int(proc.info["pid"]), subprocess.list2cmdline(argv)))
    return rows


def _ppid(pid: int) -> Optional[int]:
    import psutil
    with contextlib.suppress(Exception):
        return psutil.Process(pid).ppid()
    return None


def drop_launchers(rows: list[tuple[int, str]], ppid_of=_ppid) -> list[tuple[int, str]]:
    """*rows* without a backend whose own child is a backend too (the ``hermes.exe`` launcher)."""
    pids = {pid for pid, _ in rows}
    parents = {ppid_of(pid) for pid in pids} & pids
    return [(pid, cmd) for pid, cmd in rows if pid not in parents]


def argv_of(pid: int) -> Optional[list[str]]:
    import psutil
    with contextlib.suppress(Exception):
        return list(psutil.Process(pid).cmdline()) or None
    return None


def _ancestors(pid: int) -> list[tuple[int, str]]:
    import psutil
    with contextlib.suppress(Exception):
        return [(p.pid, (p.name() or "").lower()) for p in psutil.Process(pid).parents()]
    return []


def _schedule_service_pid() -> Optional[int]:
    import psutil
    with contextlib.suppress(Exception):
        return psutil.win_service_get("Schedule").pid()
    return None


def _hermes_args(argv: list[str]) -> list[str]:
    """The arguments after the ``hermes`` / ``hermes.exe`` / ``hermes_cli.main`` entry token."""
    for i, token in enumerate(argv):
        low = token.strip("\"'").replace("\\", "/").lower()
        if low.rsplit("/", 1)[-1] in ("hermes", "hermes.exe") or low.endswith("hermes_cli.main") \
                or low.endswith("hermes_cli/main.py"):
            return argv[i + 1:]
    return []


def tasks_running(xml_text: str, argv: list[str]) -> list[tuple[str, bool]]:
    """``(task path, enabled)`` of every task in a ``schtasks /Query /XML`` dump whose action runs
    ``hermes`` with exactly the backend's arguments."""
    want = _hermes_args(argv)
    if not want:
        return []
    found: list[tuple[str, bool]] = []
    parts = _TASK_HEADER.split(xml_text or "")
    for path, body in zip(parts[1::2], parts[2::2]):
        body = body[:body.rfind("</Task>") + len("</Task>")]  # the dump's closing </Tasks> follows the last one
        try:
            root = ElementTree.fromstring(_XML_DECL.sub("", body, count=1).strip())
        except ElementTree.ParseError:
            continue
        enabled = (root.findtext(f"{_TASK_NS}Settings/{_TASK_NS}Enabled") or "true").strip().lower() != "false"
        for action in root.iter(f"{_TASK_NS}Exec"):
            command = (action.findtext(f"{_TASK_NS}Command") or "").strip()
            args = (action.findtext(f"{_TASK_NS}Arguments") or "").strip()
            with contextlib.suppress(ValueError):
                if _hermes_args([command, *_split_windows_args(args)]) == want:
                    found.append((path.lstrip("\\"), enabled))
                    break
    return found


def _split_windows_args(text: str) -> list[str]:
    import shlex
    return [t.strip('"') for t in shlex.split(text, posix=False)]


def _query_tasks_xml() -> str:
    from hermes_cli.gateway_windows import _exec_schtasks
    code, out, _err = _exec_schtasks(["/Query", "/XML"], timeout=30)
    return out if code == 0 else ""


def startup_dirs() -> list[Path]:
    dirs = []
    for var, tail in (("APPDATA", r"Microsoft\Windows\Start Menu\Programs\Startup"),
                      ("PROGRAMDATA", r"Microsoft\Windows\Start Menu\Programs\StartUp")):
        if os.environ.get(var):
            dirs.append(Path(os.environ[var]) / tail)
    return dirs


def lnk_command(raw: bytes) -> Optional[list[str]]:
    """``[target, *arguments]`` of a Shell Link (``.lnk``) file's bytes, or None when they are not one.
    Reads the header's flags, the LinkInfo's local base path and the StringData (MS-SHLLINK 2.1-2.4)."""
    import struct
    if len(raw) < 0x4C or raw[:4] != b"\x4c\x00\x00\x00":
        return None
    try:
        flags = struct.unpack_from("<I", raw, 0x14)[0]
        pos, target = 0x4C, None
        if flags & 0x01:  # HasLinkTargetIDList
            pos += 2 + struct.unpack_from("<H", raw, pos)[0]
        if flags & 0x02:  # HasLinkInfo: LocalBasePath (ANSI, NUL-terminated) when VolumeIDAndLocalBasePath
            size, _hsize, info_flags, _vol, base = struct.unpack_from("<IIIII", raw, pos)
            if info_flags & 0x01:
                end = raw.index(b"\x00", pos + base)
                target = raw[pos + base:end].decode("mbcs" if os.name == "nt" else "latin-1", "replace")
            pos += size
        unicode, strings = bool(flags & 0x80), {}
        for bit, name in ((0x04, "name"), (0x08, "relative"), (0x10, "workdir"), (0x20, "args"), (0x40, "icon")):
            if flags & bit:
                count = struct.unpack_from("<H", raw, pos)[0]
                width = 2 if unicode else 1
                chunk = raw[pos + 2:pos + 2 + count * width]
                strings[name] = chunk.decode("utf-16-le" if unicode else "latin-1", "replace")
                pos += 2 + count * width
    except (struct.error, ValueError):
        return None
    target = target or strings.get("relative")
    if not target:
        return None
    try:
        return [target, *_split_windows_args(strings.get("args", ""))]
    except ValueError:
        return None


def _script_commands(text: str) -> list[list[str]]:
    """The command lines a ``.cmd`` / ``.bat`` / ``.ps1`` / ``.vbs`` script may run: each line, and each
    double-quoted string in it (``WScript.Shell``'s ``Run "hermes dashboard", 0``)."""
    found = []
    for line in text.splitlines():
        for candidate in (line, *re.findall(r'"([^"]*)"', line)):
            with contextlib.suppress(ValueError):
                found.append(_split_windows_args(candidate))
    return found


def startup_entries_running(argv: list[str], dirs: Optional[list[Path]] = None) -> list[Path]:
    """Startup-folder entries that start ``hermes`` with exactly the backend's arguments: a ``.lnk`` by its
    target and arguments, a script by a command line in it. Same rule as for tasks: every argument after
    the ``hermes`` entry token must match, so ``dashboard --port 9200`` is not ``dashboard``."""
    want = _hermes_args(argv)
    if not want:
        return []
    hits: list[Path] = []
    for folder in (startup_dirs() if dirs is None else dirs):
        with contextlib.suppress(OSError):
            for entry in sorted(folder.iterdir()):
                if entry.suffix.lower() not in _STARTUP_SUFFIXES or not entry.is_file():
                    continue
                with contextlib.suppress(OSError):
                    raw = entry.read_bytes()[:65536]
                    if entry.suffix.lower() == ".lnk":
                        commands = [c for c in [lnk_command(raw)] if c]
                    else:
                        commands = _script_commands(raw.decode("utf-8", "replace"))
                    if any(_hermes_args(c) == want for c in commands):
                        hits.append(entry)
    return hits


def supervisor(pid: int, argv: list[str]) -> tuple[str, Optional[str], Optional[str]]:
    """``(supervisor, unit, unit_path)`` of a running backend: ``scheduled-task`` when the Task Scheduler
    service is its ancestor (unit = the task path, None when no single task names its arguments),
    ``startup-folder`` when Startup-folder entries start exactly its command (unit/unit_path = the entry
    when there is one, None when several match: the operator must say which), else ``manual``. A
    Startup-folder match is a same-command fact, not proof that the entry started this process."""
    schedule = _schedule_service_pid()
    if schedule is not None and any(p == schedule for p, _name in _ancestors(pid)):
        tasks = tasks_running(_query_tasks_xml(), argv)
        return "scheduled-task", (tasks[0][0] if len(tasks) == 1 else None), None
    entries = startup_entries_running(argv)
    if len(entries) == 1:
        return "startup-folder", entries[0].name, str(entries[0])
    if entries:
        return "startup-folder", None, None
    return "manual", None, None


def _hermes_command(home: Optional[str]) -> tuple[list[str], str]:
    """``(lines to run first, how to type hermes)`` for *home* in PowerShell and Command Prompt."""
    from hermes_constants import profile_name_for_home
    name = profile_name_for_home(home) if home else "default"
    if name:  # also `-p default`: a bare `hermes` would follow a sticky `hermes profile use` choice
        return [], f"hermes -p {name}"
    return [f'# in this window first: $env:HERMES_HOME = "{home}" (PowerShell) or set HERMES_HOME={home} '
            "(Command Prompt)"], "hermes"


def manual_steps(old_host: Optional[str], old_port: Optional[int], item, *, env_keys=(), home: Optional[str] = None,
                 public_url=None) -> list[str]:
    """One command per line, in order, each valid as typed in PowerShell and in Command Prompt (prose
    lines start with ``#``). The old backend is stopped and kept from starting at sign-in before the
    gateway is installed, started or restarted (by what ``hermes gateway status`` says), so the gateway
    can bind the port; ``hermes webapp --status`` checks the result. *public_url*: ``(url, problem)``."""
    from hermes_cli.gateway_migrate_backends_plan import gateway_tail, public_url_steps
    env = str(Path(home) / ".env") if home else "the Hermes .env file"
    first, hermes = _hermes_command(home)
    steps = [*first, f"{hermes} webapp --build-only", f"{hermes} config set webapp.enabled true"]
    if old_host is not None and old_port is not None:
        steps += [f"{hermes} config set webapp.host {old_host}", f"{hermes} config set webapp.port {old_port}"]
    else:
        steps.append(f"# set the old backend's address: {hermes} config set webapp.host HOST, then webapp.port PORT")
    url, problem = public_url or (None, None)
    if url:  # never on a command line: PowerShell expands $name and $(...) inside double quotes
        steps.append("# copy the old backend's public URL (dashboard.public_url, or HERMES_DASHBOARD_PUBLIC_URL in "
                     f"its environment) into webapp.public_url: open config.yaml with {hermes} config edit")
    elif problem:
        steps += public_url_steps(hermes, public_url)  # prose only: the refused value is never printed
    if env_keys:
        steps.append(f"# copy {', '.join(sorted(env_keys))} from the old backend's environment into {env} "
                     "(values are never printed here)")
    sup, unit, path, pid = (getattr(item, "supervisor", None), getattr(item, "unit", None),
                            getattr(item, "unit_path", None), getattr(item, "pid", None))
    kill = [f"taskkill /PID {pid} /T /F"] if pid else []
    if sup == "scheduled-task" and unit:
        steps += [f'schtasks /End /TN "{unit}"', f'schtasks /Change /TN "{unit}" /DISABLE']
    elif sup == "scheduled-task":  # no single task's action names its arguments (a wrapper script runs it)
        steps += (kill or ["# in Task Scheduler, End the task that starts it"]) \
            + ["# and Disable that task in Task Scheduler"]
    elif sup == "startup-folder" and path:
        steps += kill + ["# this Startup-folder entry starts the same command; if it is what starts this backend "
                         "at sign-in, set it aside:", f'move "{path}" "{path}.disabled"']
    elif sup == "startup-folder":
        steps += kill + ["# more than one Startup-folder entry starts this exact command: find the one that starts "
                         "this backend (shell:startup) and move it aside"]
    else:
        steps += [kill[0] if kill else "# stop the old backend (Ctrl+C in its window)",
                  "# and remove whatever starts it at sign-in"]
    return steps + gateway_tail(hermes)
