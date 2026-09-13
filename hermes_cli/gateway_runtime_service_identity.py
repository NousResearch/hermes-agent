"""Read-only binding checks; unknown supervisor syntax is not profile absence.

These checks recognize the installed Hermes launch formats, not arbitrary shell
programs. Manager-loaded values take precedence over filenames and disk copies.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
import stat
from typing import NoReturn


SYSTEMD_IDENTITY_PROPERTIES = (
    "User", "DynamicUser", "ExecStart", "Environment", "EnvironmentFiles",
    "PassEnvironment", "UnsetEnvironment", "PAMName", "RootDirectory", "RootImage",
    "ExecStartPre", "ExecCondition", "DropInPaths",
)
# `systemctl show --all` (systemd 259) prints no line at all for these when the
# list is empty, so their absence carries no identity uncertainty.
SYSTEMD_OMITTED_WHEN_EMPTY = ("EnvironmentFiles", "ExecStartPre", "ExecCondition")


def _unverified() -> NoReturn:
    raise ValueError("service_identity_unverified")


def _absolute(value: str) -> Path:
    if not value or not Path(value).is_absolute():
        _unverified()
    return Path(value).resolve()


def verify_home(home: Path, configured: str) -> None:
    if os.path.normcase(str(_absolute(configured))) != os.path.normcase(str(home.resolve())):
        raise ValueError("profile_mismatch")


def verify_gateway_argv(argv: list[str], home: Path) -> None:
    from gateway.status import looks_like_gateway_command_line

    # The installer wraps launchd stderr with this fixed, non-shell module.
    if len(argv) > 7 and argv[1:4] == ["-m", "hermes_cli.stderr_timestamp", "--error-log"] and argv[5] == "--":
        if not re.fullmatch(r"python(?:w|\d+(?:\.\d+)*)?(?:\.exe)?", Path(argv[0]).name.lower()):
            _unverified()
        argv = argv[6:]
    if not argv or not looks_like_gateway_command_line(subprocess_command(argv)):
        _unverified()
    executable = Path(argv[0]).name.lower()
    if re.fullmatch(r"python(?:w|\d+(?:\.\d+)*)?(?:\.exe)?", executable):
        if argv[1:3] != ["-m", "hermes_cli.main"]:
            _unverified()
        args = argv[3:]
    elif executable in {"hermes", "hermes.exe"}:
        args = argv[1:]
    else:
        _unverified()
    profiles = []
    filtered = []
    index = 0
    while index < len(args):
        token = args[index]
        if token in {"--profile", "-p"}:
            index += 1
            if index == len(args):
                _unverified()
            profiles.append(args[index])
        elif token.startswith("--profile="):
            profiles.append(token.split("=", 1)[1])
        else:
            filtered.append(token)
        index += 1
    if len(profiles) > 1:
        _unverified()
    if profiles:
        # An explicit selector must agree with the pinned environment. This
        # preserves custom-root profiles without borrowing the caller's root.
        from hermes_constants import profile_name_for_home
        expected = profile_name_for_home(home) or (home.name if home.parent.name == "profiles" else "default")
        if profiles[0] != expected:
            raise ValueError("profile_mismatch")
    if filtered[:2] != ["gateway", "run"] or any(
        arg not in {"--quiet", "--external-supervisor"} for arg in filtered[2:]
    ):
        _unverified()


def subprocess_command(argv: list[str]) -> str:
    # The canonical matcher is quote-aware on both hosts.
    import subprocess
    return subprocess.list2cmdline(argv)


def environment_pairs(value: str) -> dict[str, str]:
    # systemctl uses C escapes for unusual bytes. Do not silently reinterpret
    # those with shell escaping (not the same grammar).
    if "\\" in value:
        _unverified()
    result = {}
    for item in shlex.split(value):
        key, separator, val = item.partition("=")
        if not separator:
            _unverified()
        result[key] = val
    return result


def verify_systemd(props: dict[str, str], manager_env: str, home: Path, *, system: bool) -> None:
    import pwd

    # systemd 259 omits empty exec-command lists and EnvironmentFiles from
    # `show --all` output entirely; an omitted list is empty, not unknown.
    props = {**{key: "" for key in SYSTEMD_OMITTED_WHEN_EMPTY}, **props}
    if not all(key in props for key in SYSTEMD_IDENTITY_PROPERTIES):
        _unverified()
    # These can change the execution account, environment, mount namespace, or
    # definitions at start time; show does not expose their resulting identity.
    if props["DynamicUser"] != "no" or any(props[key] for key in (
        "EnvironmentFiles", "PAMName", "RootDirectory", "RootImage", "ExecStartPre", "ExecCondition",
    )):
        _unverified()
    user = props["User"]
    uid = os.getuid()  # windows-footgun: ok — native systemd only
    if user:
        try:
            service_uid = int(user) if user.isdecimal() else pwd.getpwnam(user).pw_uid
        except KeyError:
            _unverified()
    else:
        service_uid = 0 if system else uid
    if service_uid != uid:
        raise ValueError("service_account_mismatch")
    inherited = environment_pairs(manager_env)
    env = inherited if not system else {k: v for k, v in inherited.items() if k in props["PassEnvironment"].split()}
    env.update(environment_pairs(props["Environment"]))
    for item in shlex.split(props["UnsetEnvironment"]):
        key, sep, value = item.partition("=")
        if not sep or env.get(key) == value:
            env.pop(key, None)
    configured = env.get("HERMES_HOME", "").strip()
    if not configured:
        configured = str(_absolute(env.get("HOME") or pwd.getpwuid(uid).pw_dir) / ".hermes")
    verify_home(home, configured)
    match = re.fullmatch(r"\{ path=(.*?) ; argv\[\]=(.*?) ; ignore_errors=(?:yes|no) ;(?:.*?)\}", props["ExecStart"])
    if not match or any(char in match[2] for char in ("$", "\\", "{", "}")):
        _unverified()
    argv = shlex.split(match[2])
    if not argv or match[1] != argv[0]:
        _unverified()
    verify_gateway_argv(argv, home)


def verify_launchd_loaded(output: str, home: Path, *, uid: int, username: str) -> None:
    # Only the job's own blocks, never inherited/default environments or a
    # different job embedded in launchctl diagnostic text.
    def block(name):
        found = re.findall(r"^\t" + name + r" = \{\n(.*?)^\t\}", output, re.M | re.S)
        if len(found) != 1:
            _unverified()
        return found[0].splitlines()

    for key, expected in (("uid", str(uid)), ("username", username)):
        values = re.findall(r"^\t" + key + r" = (.+)$", output, re.M)
        if values and values != [expected]:
            raise ValueError("service_account_mismatch")
    env = {}
    for line in block("environment"):
        if not line.strip():
            continue
        key, sep, value = line.strip().partition(" => ")
        if not sep or key in env:
            _unverified()
        env[key] = value
    verify_home(home, env.get("HERMES_HOME", ""))
    if home.parent.name != "profiles" and not env.get("HERMES_SUPERVISED_CHILD"):
        _unverified()
    argv = [line.strip() for line in block("arguments")]
    programs = re.findall(r"^\tprogram = (.+)$", output, re.M)
    if len(programs) != 1 or not argv or programs[0] != argv[0]:
        _unverified()
    verify_gateway_argv(argv, home)


def verify_launchd_plist(definition: dict, label: str, home: Path) -> None:
    if definition.get("Label") != label:
        _unverified()
    if definition.get("Disabled"):
        raise ValueError("service_disabled")
    if definition.get("UserName") or definition.get("GroupName") or definition.get("RootDirectory"):
        _unverified()
    env = definition.get("EnvironmentVariables", {})
    if not isinstance(env, dict):
        _unverified()
    verify_home(home, env.get("HERMES_HOME", ""))
    argv = definition.get("ProgramArguments")
    if not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
        _unverified()
    if definition.get("Program", argv[0] if argv else None) != (argv[0] if argv else None):
        _unverified()
    verify_gateway_argv(argv, home)
    if home.parent.name != "profiles" and not env.get("HERMES_SUPERVISED_CHILD"):
        _unverified()


def read_definition(path: Path) -> bytes:
    # Nonblocking open precedes fstat: a FIFO replacement must not consume the
    # entire startup deadline waiting for a writer. Symlinked installs work.
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0))
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > 1024 * 1024:
            _unverified()
        value = stream.read(1024 * 1024 + 1)
        if len(value) > 1024 * 1024:
            _unverified()
        return value


def _vbs_identity(script: str) -> tuple[str, list[str]]:
    # Recognize the vendor's complete template, not a HERMES_HOME substring
    # inside an arbitrary executable script. Doubled quotes are VB literals.
    literal = r'"(?:[^"\r\n]|"")*"'
    lines = script.splitlines()
    if len(lines) < 16 or not lines[0].startswith("' "):
        _unverified()
    fixed = ["Option Explicit", "Dim sh, env, existing_pp",
             'Set sh = CreateObject("WScript.Shell")', 'Set env = sh.Environment("PROCESS")']
    if lines[1:5] != fixed:
        _unverified()
    env = {}
    index = 5
    while index < len(lines):
        match = re.fullmatch(r'env.Item\("([A-Z_]+)"\) = (' + literal + ')', lines[index])
        if not match:
            break
        if match[1] not in {"HERMES_HOME", "HERMES_SUPERVISED_CHILD", "HERMES_GATEWAY_DETACHED", "PYTHONIOENCODING", "VIRTUAL_ENV"} or match[1] in env:
            _unverified()
        env[match[1]] = match[2][1:-1].replace('""', '"')
        index += 1
    tail = lines[index:]
    if len(tail) != 8 or tail[:2] != ['existing_pp = env.Item("PYTHONPATH")', 'If Len(existing_pp) > 0 Then'] or tail[3] != 'Else' or tail[5] != 'End If':
        _unverified()
    if not re.fullmatch(r'  env.Item\("PYTHONPATH"\) = ' + literal + r' & existing_pp', tail[2]) or not re.fullmatch(r'  env.Item\("PYTHONPATH"\) = ' + literal, tail[4]):
        _unverified()
    if not re.fullmatch(r'sh.CurrentDirectory = ' + literal, tail[6]):
        _unverified()
    run = re.fullmatch(r'sh.Run (' + literal + r'), 0, False', tail[7])
    if not run or not env.get("HERMES_SUPERVISED_CHILD"):
        _unverified()
    command = run[1][1:-1].replace('""', '"')
    if "%" in command or "%" in env.get("HERMES_HOME", ""):
        _unverified()
    # The template uses list2cmdline, with no embedded executable quote escapes.
    argv = [part.strip('"') for part in shlex.split(command, posix=False)]
    return env.get("HERMES_HOME", ""), argv


def verify_windows_task(xml: str | bytes, home: Path, account: str, sid: str) -> None:
    import xml.etree.ElementTree as ET

    try:
        task = ET.fromstring(xml)
    except ET.ParseError:
        _unverified()
    ns = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
    principals = task.findall("t:Principals/t:Principal", ns)
    actions = task.find("t:Actions", ns)
    if len(principals) != 1 or actions is None or len(actions) != 1:
        _unverified()
    principal = principals[0]
    user = principal.findtext("t:UserId", "", ns)
    if not user or not account or not sid or principal.find("t:GroupId", ns) is not None:
        _unverified()
    if user.casefold() not in {account.casefold(), sid.casefold()}:
        raise ValueError("service_account_mismatch")
    if principal.findtext("t:RunLevel", "", ns) != "LeastPrivilege" or actions.get("Context") != principal.get("id"):
        _unverified()
    if task.findtext("t:Settings/t:Enabled", "", ns) != "true":
        raise ValueError("service_disabled")
    action = actions[0]
    if action.tag != '{' + ns['t'] + '}Exec' or action.findtext('t:Command', '', ns).lower() not in {"wscript.exe", r"c:\windows\system32\wscript.exe"}:
        _unverified()
    args = action.findtext("t:Arguments", "", ns)
    match = re.fullmatch(r'//B //Nologo "([^"\r\n]+)"', args)
    if not match:
        _unverified()
    script = _absolute(match[1])
    if script.suffix.lower() != '.vbs':
        _unverified()
    configured, argv = _vbs_identity(read_definition(script).decode("utf-8"))
    verify_home(home, configured)
    verify_gateway_argv(argv, home)
