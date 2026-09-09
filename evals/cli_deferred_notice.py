"""Drive the real CLI with a FIFO update-cache result under a Linux PTY.

Run from the checkout with the project Python. No provider requests are sent.
The FIFO releases the real cached update result only after the prompt renders.
"""
import argparse
import errno
import json
import os
from pathlib import Path
import pty
import select
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import time
import fcntl


def run_case(root, output, name, behind, early=False, cancel=False):
    with tempfile.TemporaryDirectory(prefix="hermes_test_notice_") as home:
        hh = Path(home) / ".hermes"
        hh.mkdir()
        (hh / "config.yaml").write_text(
            "model:\n  default: test-model\n  provider: custom\n"
            "  base_url: http://127.0.0.1:9/v1\n"
            "display:\n  interface: cli\n  skip_banner: false\n"
            "memory:\n  provider: ''\n", encoding="utf-8")
        cache = hh / ".update_check"
        # The version comes from the checkout, not an invented cache identity.
        from hermes_cli.banner import VERSION
        payload = json.dumps({"ts": time.time(), "behind": behind,
                              "rev": None, "ver": VERSION}).encode()
        if early:
            cache.write_bytes(payload)
        else:
            os.mkfifo(cache)
        env = {"PATH": os.environ["PATH"], "HOME": home, "HERMES_HOME": str(hh),
               "PYTHONPATH": str(root), "PYTHONUNBUFFERED": "1",
               "TERM": "xterm-256color", "LANG": "C.UTF-8",
               "OPENAI_API_KEY": "local-not-used", "PROMPT_TOOLKIT_NO_CPR": "1"}
        master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
        bootstrap = ("import hermes_cli.main as m; import hermes_cli.banner as b; "
                     "print('LOADED', m.__file__, b.__file__, flush=True); m.main()")
        proc = subprocess.Popen([sys.executable, "-c", bootstrap, "chat"], cwd=root,
                                env=env, stdin=slave, stdout=slave, stderr=slave,
                                start_new_session=True)
        os.close(slave)
        data = bytearray()

        def pump_until(predicate, timeout=30):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if predicate(bytes(data)):
                    return True
                if select.select([master], [], [], 0.1)[0]:
                    try:
                        chunk = os.read(master, 65536)
                    except OSError as exc:
                        if exc.errno == errno.EIO:
                            return predicate(bytes(data))
                        raise
                    if not chunk:
                        return predicate(bytes(data))
                    data.extend(chunk)
            return predicate(bytes(data))

        try:
            ready = pump_until(lambda b: b"\x1b[?2004h" in b)
            assert ready, f"{name}: prompt did not render: {data[-1500:]!r}"
            offset = len(data)
            if not early and not cancel:
                fd = os.open(cache, os.O_WRONLY | os.O_NONBLOCK)
                os.write(fd, payload)
                os.close(fd)
                pump_until(lambda b: b"to update" in b[offset:] or b"update available" in b[offset:], 3)
                # prompt_toolkit prints above the prompt via run_in_terminal (input detached,
                # cooked mode); type only after the prompt is redrawn, like a user would.
                redraw = len(data)
                pump_until(lambda b: b"\x1b[?2004h" in b[redraw:], 3)
            elif cancel:
                os.write(master, b"\x03")
            os.write(master, b"/quit\r")
            exited = pump_until(lambda b: proc.poll() is not None, 60)
            proc.wait(timeout=30)
            text = bytes(data).decode(errors="replace")
            (output / f"{name}.pty").write_bytes(data)
            result = {"case": name, "ready": ready, "exited": exited,
                      "returncode": proc.returncode, "garbled": "?[1;33m" in text,
                      "notice": "commits behind" in text or "update available" in text,
                      "loaded_worktree": str(root / "hermes_cli/banner.py") in text,
                      "raw_path": str(output / f"{name}.pty")}
            assert result["loaded_worktree"] and proc.returncode == 0, result
            return result
        finally:
            (output / f"{name}.pty").write_bytes(data)
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=30)
            os.close(master)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--expect", choices=("clean", "garbled"), required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [("late-count", 3, False, False), ("late-current", 0, False, False),
             ("late-unknown-count", -1, False, False), ("early-count", 3, True, False),
             ("cancel-pending", 3, False, True)]
    results = [run_case(root, args.output, *case) for case in cases]
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    for row in results:
        expected_notice = row["case"] not in ("late-current", "cancel-pending")
        assert row["notice"] == expected_notice, row
        expected_garble = args.expect == "garbled" and row["case"].startswith("late-") and expected_notice
        assert row["garbled"] == expected_garble, row


if __name__ == "__main__":
    main()
