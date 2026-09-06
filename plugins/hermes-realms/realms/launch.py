"""Real stdin/stdout/stderr FD handoff to the realm-owned worker.

Usage: python -m realms.launch HERMES_HOME REALM_ID -- PROGRAM [ARG ...]
A script-path invocation also works without modifying a caller's PYTHONPATH.
"""

import os
import signal
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    import runpy

    __package__ = runpy.run_path(str(Path(__file__).resolve().with_name("_binding.py")))["load_runtime"]().__name__
from .manager import Manager
from .lifecycle import RealmError


class PtyRelay:
    """Give a tty-owning caller a separate child controlling PTY.

    Pipes are passed directly. An existing terminal cannot be stolen from its
    caller's session, so only the interactive case requires byte forwarding.
    """

    def __init__(self):
        import pty
        import termios
        import tty

        self.master, self.slave = pty.openpty()
        self.original = termios.tcgetattr(0)
        termios.tcsetattr(self.slave, termios.TCSANOW, self.original)
        self.resize()
        tty.setraw(0)
        self.read_input = True
        self.ended = False

    def resize(self):
        import fcntl
        import termios

        size = fcntl.ioctl(0, termios.TIOCGWINSZ, b"\x00" * 8)
        fcntl.ioctl(self.master, termios.TIOCSWINSZ, size)

    def fds(self):
        return (
            self.slave,
            self.slave if os.isatty(1) else 1,
            self.slave if os.isatty(2) else 2,
        )

    def after_launch(self):
        os.close(self.slave)
        self.slave = None

    def pump(self, timeout=0.05):
        import select

        readers = ([] if self.ended else [self.master]) + (
            [0] if self.read_input else []
        )
        if not readers:
            return
        ready, _, _ = select.select(readers, [], [], timeout)
        for source in ready:
            try:
                data = os.read(source, 65536)
            except OSError:
                data = b""
            if not data:
                if source == self.master:
                    self.ended = True
                else:
                    self.read_input = False
                continue
            target = 1 if source == self.master else self.master
            while data:
                data = data[os.write(target, data) :]

    def close(self):
        import termios

        termios.tcsetattr(0, termios.TCSANOW, self.original)
        if self.slave is not None:
            os.close(self.slave)
        os.close(self.master)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 4 or args[2] != "--":
        print(
            "usage: python -m realms.launch HOME REALM_ID -- PROGRAM [ARG ...]",
            file=sys.stderr,
        )
        return 2
    home, realm_id = args[:2]
    manager = Manager(home)
    pending_signals = []
    forwarded = (
        signal.SIGINT,
        signal.SIGTERM,
        signal.SIGHUP,  # windows-footgun: ok — runtime package rejects non-Linux hosts
        signal.SIGQUIT,  # windows-footgun: ok — runtime package rejects non-Linux hosts
        signal.SIGWINCH,
    )
    previous = {number: signal.getsignal(number) for number in forwarded}
    for number in forwarded:
        signal.signal(number, lambda received, frame: pending_signals.append(received))
    relay = None
    try:
        # Pass the caller's already-sanitized env EXACTLY. Never merge worker or
        # backend env here: doing so resurrects secrets stripped by the caller.
        if os.isatty(0):
            relay = PtyRelay()
        job = manager._rpc(
            realm_id,
            op="launch",
            command=args[3:],
            cwd=os.getcwd(),
            env=dict(os.environ),
            fds=relay.fds() if relay else (0, 1, 2),
        )
        if relay:
            relay.after_launch()
        while True:
            while pending_signals:
                number = pending_signals.pop(0)
                if number == signal.SIGWINCH and relay:
                    relay.resize()
                manager._rpc(realm_id, op="signal", job_id=job["job_id"], signal=number)
            result = manager._rpc(realm_id, op="job", job_id=job["job_id"])
            if result["returncode"] is not None:
                if relay:
                    deadline = time.monotonic() + 1
                    while not relay.ended and time.monotonic() < deadline:
                        relay.pump()
                code = result["returncode"]
                return code if code >= 0 else 128 - code
            if relay:
                relay.pump()
            else:
                time.sleep(0.1)
    except (RealmError, OSError, ValueError) as exc:
        print("realm launch: " + str(exc), file=sys.stderr)
        return 125
    finally:
        if relay:
            relay.close()
        for number, handler in previous.items():
            signal.signal(number, handler)


if __name__ == "__main__":
    raise SystemExit(main())
