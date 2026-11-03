"""Drive the real ``hermes fallback add`` under a Linux PTY into an ordinary picker error.

Run from the checkout with the project Python. No provider requests are sent: the picker
target is a saved custom provider with ``discover_models: false``. The auth store is made
unreadable AFTER the provider menu renders (the pre-picker snapshot has already happened), so
the canonical picker writes the temporary primary route to config.yaml and then fails inside
``deactivate_provider`` with a plain ``PermissionError`` -- not ``SystemExit``.

The invariant under test: config.yaml ``model`` must equal the pre-picker primary afterwards.
"""
import argparse
import errno
import json
import os
from pathlib import Path
import pty
import re
import select
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import time
import fcntl

PRIMARY = {"provider": "openrouter", "default": "primary/model-a",
           "base_url": "https://openrouter.ai/api/v1", "api_mode": "chat_completions"}
CONFIG = (
    "model:\n  provider: openrouter\n  default: primary/model-a\n"
    "  base_url: https://openrouter.ai/api/v1\n  api_mode: chat_completions\n"
    "custom_providers:\n  - name: LocalLab\n    base_url: http://127.0.0.1:9/v1\n"
    "    model: lab-model\n    discover_models: false\n    models:\n      - lab-model\n"
    "memory:\n  provider: ''\n")


def _persisted_model(root: Path, env: dict) -> dict:
    """``config.yaml`` ``model`` section as the CLI itself reads it (owner module, same env)."""
    out = subprocess.run(
        [sys.executable, "-c", "import json; from hermes_cli.config import load_config; "
         "print(json.dumps(load_config().get('model')))"],
        cwd=root, env=env, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def run(root: Path, output: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="hermes_test_fallback_") as home:
        hh = Path(home) / ".hermes"
        hh.mkdir()
        (hh / "config.yaml").write_text(CONFIG, encoding="utf-8")
        (hh / ".env").write_text("OPENROUTER_API_KEY=local-not-used\n", encoding="utf-8")
        auth = hh / "auth.json"
        auth.write_text(json.dumps({"version": 1, "providers": {}, "active_provider": "nous"}))
        # A stub ``curses`` package forces every menu onto its numbered fallback so the PTY
        # exchange is line-oriented (the curses UI is not what is under test here).
        shim = Path(home) / "shim" / "curses"
        shim.mkdir(parents=True)
        (shim / "__init__.py").write_text("raise ImportError('curses disabled for PTY harness')\n")
        env = {"PATH": os.environ["PATH"], "HOME": home, "HERMES_HOME": str(hh),
               "PYTHONPATH": f"{shim.parent}{os.pathsep}{root}", "PYTHONUNBUFFERED": "1",
               "TERM": "dumb", "LANG": "C.UTF-8"}
        master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 50, 120, 0, 0))
        proc = subprocess.Popen([sys.executable, "-m", "hermes_cli.main", "fallback", "add"],
                                cwd=root, env=env, stdin=slave, stdout=slave, stderr=slave,
                                start_new_session=True)
        os.close(slave)
        data = bytearray()

        def pump_until(predicate, timeout=60):
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
            assert pump_until(lambda b: b"Choice [default" in b), data[-2000:]
            text = bytes(data).decode(errors="replace")
            row = re.search(r"(\d+)\. LocalLab", text)
            assert row, text[-3000:]
            # Snapshot is done (menu is up); now make the auth store unreadable so the picker's
            # own deactivate_provider() fails with an ordinary OSError after writing the model.
            auth.chmod(0)
            os.write(master, f"{row.group(1)}\r".encode())
            offset = len(data)
            assert pump_until(lambda b: b"Choice [" in b[offset:]), data[-2000:]
            os.write(master, b"1\r")
            exited = pump_until(lambda b: proc.poll() is not None, 90)
            proc.wait(timeout=30)
            auth.chmod(0o600)
            model_after = _persisted_model(root, env)
            text = bytes(data).decode(errors="replace")
            return {"exited": exited, "returncode": proc.returncode,
                    "picker_error_surfaced": "PermissionError" in text,
                    "model_after": model_after, "primary_restored": model_after == PRIMARY,
                    "auth_active_provider": json.loads(auth.read_text()).get("active_provider"),
                    "restore_note": "Could not fully restore" in text,
                    "raw_path": str(output / "fallback-add-picker-error.pty")}
        finally:
            output.mkdir(parents=True, exist_ok=True)
            (output / "fallback-add-picker-error.pty").write_bytes(data)
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=30)
            os.close(master)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect", choices=("stranded", "restored"), required=True)
    args = parser.parse_args()
    result = run(Path(args.root).resolve(), args.output)
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    assert result["exited"] and result["returncode"] != 0, result
    assert result["picker_error_surfaced"], result
    assert result["primary_restored"] == (args.expect == "restored"), result


if __name__ == "__main__":
    main()
