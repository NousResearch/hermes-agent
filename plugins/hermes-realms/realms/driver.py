"""OS containment for the trusted realm driver, not for arbitrary terminal code.

Only the private compositor/buses/runtime and render nodes are exposed. Host
home, session sockets, input devices, PID list and network namespace stay out.
"""

import os
from pathlib import Path
import shutil

from .lifecycle import validate_live


def sandbox_command(record, executable, args=()):
    validate_live(record)
    runtime = Path(record["runtime_dir"])
    binary = Path(executable).resolve(strict=True)
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise RuntimeError("bubblewrap is required for realm driver isolation")
    command = [
        bwrap,
        "--die-with-parent",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-net",
        "--unshare-ipc",
        "--unshare-uts",
        "--new-session",
        "--cap-drop",
        "ALL",
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind",
        "/etc",
        "/etc",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--dir",
        "/run",
        "--dir",
        "/run/user",
        "--dir",
        str(runtime.parent),
        "--bind",
        str(runtime),
        str(runtime),
    ]
    for path in ("/bin", "/sbin", "/lib", "/lib64"):
        target = Path(path)
        if target.is_symlink():
            command += ["--symlink", os.readlink(path), path]
        elif target.exists():
            command += ["--ro-bind", path, path]
    # EGL can render but no DRM card, uinput or physical event device is exposed.
    for node in sorted(Path("/dev/dri").glob("renderD*")):
        command += ["--dev-bind", str(node), str(node)]
    # Driver overlay may use Xwayland; expose only this realm's X socket.
    import json
    import re

    env = json.loads((runtime / "ready.json").read_text(encoding="utf-8"))["env"]
    display = env.get("DISPLAY", "")
    if not re.fullmatch(r":[0-9]+", display):
        raise ValueError("Invalid private Xwayland display")
    xsocket = Path("/tmp/.X11-unix") / ("X" + display[1:])
    if xsocket.exists():
        command += ["--ro-bind", str(xsocket), str(xsocket)]
    if not binary.is_relative_to("/usr"):
        command += ["--ro-bind", str(binary), "/opt/realm-driver"]
        binary = Path("/opt/realm-driver")
    # Bind only the exact host-approved bounded manifest, read-only. Its
    # contents/approval flags remain owned by Hermes, never rewritten here.
    args = list(args)
    for index, argument in enumerate(args):
        if argument in ("--capability-manifest", "--session-policy"):
            if index + 1 >= len(args):
                raise ValueError("Missing capability manifest path")
            manifest = Path(args[index + 1]).resolve(strict=True)
            if not manifest.is_file():
                raise ValueError("Capability manifest must be a regular file")
            command += ["--ro-bind", str(manifest), str(manifest)]
            args[index + 1] = str(manifest)
    command += ["--chdir", str(runtime), "--", str(binary), *args]
    return command


def create_driver_launcher(manager, realm_id, executable):
    """Return a private executable for Hermes' existing driver_command contract."""
    import sys

    manager.env(realm_id)
    record = next(r for r in manager.list() if r["id"] == realm_id)
    binary = str(Path(executable).resolve(strict=True))
    script = Path(record["runtime_dir"]) / "cua-contained"
    binding = str(Path(__file__).resolve().with_name("_binding.py"))
    content = (
        f"#!{sys.executable}\nimport runpy\n"
        f"driver_main = runpy.run_path({binding!r})['load_runtime']('driver').driver_main\n"
        f"driver_main({str(manager.home)!r},{realm_id!r},{binary!r})\n"
    )
    import stat

    with manager.registry.lock():
        if script.exists() or script.is_symlink():
            info = script.lstat()
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()  # windows-footgun: ok — runtime package rejects non-Linux hosts
                or stat.S_IMODE(info.st_mode) != 0o700
            ):
                raise ValueError("Private driver launcher ownership changed")
            if script.read_text(encoding="utf-8") != content:
                raise ValueError("Private driver launcher contents changed")
        else:
            descriptor = os.open(
                script, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o700
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(content)
    return str(script)


def driver_main(home, realm_id, binary):
    import sys
    from .manager import Manager

    manager = Manager(home)
    manager.env(realm_id)
    record = next(r for r in manager.list() if r["id"] == realm_id)
    command = sandbox_command(record, binary, sys.argv[1:])
    # Preserve the host's authoritative permission-mode sanitization. Never
    # restore manager.env over the caller's environment or add approval flags.
    os.execv(command[0], command)
