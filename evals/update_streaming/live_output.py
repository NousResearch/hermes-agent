"""Disposable subprocess/tee probe; never invokes an actual update.

Run with the checkout's Python. --step drives the native Windows hand-off
watchdog fixture; otherwise print a live before/after JSON receipt.
"""
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def run_child(mode):
    if mode == "silent":
        time.sleep(12)
    else:
        for _ in range(24):
            os.write(1, b"build-progress ")
            time.sleep(.5)
    return 7


def step(mode):
    from hermes_cli.main_dashboard import _install_hangup_protection, _finalize_update_output
    from hermes_cli.main import _run_logged_subprocess
    state = _install_hangup_protection(gateway_mode=True)
    try:
        return _run_logged_subprocess([sys.executable, __file__, "--child", mode]).returncode
    finally:
        _finalize_update_output(state)


def probe():
    from hermes_cli.main_dashboard import _install_hangup_protection, _finalize_update_output
    from hermes_cli.main import _run_logged_subprocess
    receipts = []
    for gateway in (False, True):
        with tempfile.TemporaryDirectory(prefix="update-output-") as temp:
            os.environ["HERMES_HOME"] = temp
            os.environ["HOME"] = temp
            os.environ["USERPROFILE"] = temp
            screen, original = io.StringIO(), sys.stdout
            sys.stdout = screen
            state = _install_hangup_protection(gateway_mode=gateway)
            result = []
            started = time.monotonic()
            thread = threading.Thread(target=lambda: result.append(_run_logged_subprocess(
                [sys.executable, __file__, "--child", "progress"])))
            thread.start()
            log = Path(temp) / "logs" / "update.log"
            while time.monotonic() - started < 5:
                text = log.read_text(encoding="utf-8") if log.exists() else ""
                if "build-progress" in text:
                    break
                time.sleep(.02)
            early = "build-progress" in text
            observed = time.monotonic() - started
            alive = thread.is_alive()
            thread.join(20)
            _finalize_update_output(state)
            sys.stdout = original
            assert not thread.is_alive()
            receipts.append(dict(gateway=gateway, progress_before_exit=early,
                observed_seconds=round(observed, 3), alive_at_observation=alive,
                exit_code=result[0].returncode, captured_progress=result[0].stdout.count("build-progress"),
                screen_bytes=len(screen.getvalue()), log_bytes=log.stat().st_size if log.exists() else 0))
    print(json.dumps({"platform":sys.platform,"rows":receipts}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", choices=("silent", "progress"))
    parser.add_argument("--step", choices=("silent", "progress"))
    args = parser.parse_args()
    if args.child:
        raise SystemExit(run_child(args.child))
    if args.step:
        raise SystemExit(step(args.step))
    probe()
