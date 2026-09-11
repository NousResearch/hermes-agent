"""Run one argv inside this conversation's Omarchy VM realm.

Usage: python -m realms.vm_launch HERMES_HOME VM_ID -- PROGRAM [ARG ...]

This is the ``command_prefix`` half of the VM realm, the counterpart to
``realms.launch`` for labwc realms. The terminal tool hands us a fully built
``bash -c <script>`` argv; we hand it to sshd in the guest and relay the
process's streams and exit status back. stdin/stdout/stderr are inherited by
ssh, so byte-for-byte piping, heredocs and the terminal tool's CWD markers work
unchanged.

The guest's HOME is not the host's. A host path in the command therefore simply
does not exist in there, which is the boundary doing its job — use
``realm(action=push)`` to copy work in.
"""

import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys


if __package__ in (None, ""):
    import runpy

    __package__ = runpy.run_path(
        str(Path(__file__).resolve().with_name("_binding.py"))
    )["load_runtime"]().__name__

from .lifecycle import RealmError
from .vm_manager import FORWARDED_ENV, VmManager


def guest_script(argv, cwd):
    """Wrap the host argv so it runs in the guest user's session, not as root.

    A shell plugin is built and tested as the desktop user against its running
    Hyprland, so the command needs that session's bus and display; the guest's
    own ``omarchy vm run`` convention is reproduced here rather than reimplemented
    differently. A missing working directory falls back to the guest user's home
    instead of failing: a host cwd is usually meaningless in there.
    """
    user = "$(getent passwd 1000 | cut -d: -f1)"
    prelude = (
        "export XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-1; "
        "export HYPRLAND_INSTANCE_SIGNATURE="
        "$(ls /run/user/1000/hypr 2>/dev/null | head -1); "
    )
    inner = "cd " + shlex.quote(cwd) + " 2>/dev/null || cd ~; " + shlex.join(argv)
    forwarded = " ".join(
        f"{name}={shlex.quote(os.environ[name])}"
        for name in FORWARDED_ENV
        if os.environ.get(name)
    )
    return (
        prelude
        + f"runuser -u {user} -w XDG_RUNTIME_DIR,WAYLAND_DISPLAY,"
        f"HYPRLAND_INSTANCE_SIGNATURE -- env {forwarded} "
        f"bash -lc {shlex.quote(inner)}"
    )


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 4 or args[2] != "--":
        print(
            "usage: python -m realms.vm_launch HOME VM_ID -- PROGRAM [ARG ...]",
            file=sys.stderr,
        )
        return 2
    home, vm_id = args[:2]
    try:
        manager = VmManager(home, vm_id=vm_id)
        record = manager.validate(vm_id)
    except (RealmError, OSError, ValueError) as exc:
        print("realm launch: " + str(exc), file=sys.stderr)
        return 125

    command = [
        *manager.ssh_argv(record, tty=os.isatty(0)),
        "--",
        guest_script(args[3:], os.getcwd()),
    ]
    # Signals are relayed rather than inherited: ssh in its own process group
    # would otherwise miss the terminal tool's group kill on timeout, leaving
    # the guest-side process running with nothing reading it.
    # stdin is inherited on purpose so piped/heredoc input reaches the guest
    # command; muzzling it with DEVNULL would feed every `cmd < file` an EOF.
    # noqa: subprocess-stdin
    child = subprocess.Popen(command, start_new_session=True)
    forwarded = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT)
    previous = {number: signal.getsignal(number) for number in forwarded}

    def relay(received, _frame):
        try:
            os.killpg(child.pid, received)
        except ProcessLookupError:
            pass

    for number in forwarded:
        signal.signal(number, relay)
    try:
        return child.wait()
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


if __name__ == "__main__":
    raise SystemExit(main())
