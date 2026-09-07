"""Real disposable user-systemd transaction, never a production gateway."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

repo, tag, output = sys.argv[1:]
if not tag.isalnum():
    raise SystemExit("tag must be alphanumeric")
home = Path(tempfile.mkdtemp(prefix=f"updater-{tag}-"))
allowed = {key: os.environ[key] for key in ("PATH", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS") if key in os.environ}
os.environ.clear()
os.environ.update(allowed, HOME=str(home), HERMES_HOME=str(home / "hermes"))
sys.path.insert(0, repo)
from hermes_cli.update_cmd_fleet import _systemctl_reset_and_restart
unit = f"hermes-serve-audit-089c35aa-{tag}"
cmd = ["systemctl", "--user"]
def run(args):
    return subprocess.run(args, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=65)
def pid():
    return run(cmd + ["show", unit, "--property=MainPID", "--value"]).stdout.strip()
record: dict[str, object] = {"repo": repo, "tag": tag, "unit": unit, "home": str(home), "tier": "native disposable systemd service"}
try:
    start = run(["systemd-run", "--user", "--unit", unit, "--property=ExecStop=/bin/sleep 16", "--property=TimeoutStopSec=30", "--property=TimeoutStartSec=30", "/bin/sleep", "infinity"])
    assert start.returncode == 0, start.stderr
    old = pid()
    began = time.monotonic()
    try:
        result = _systemctl_reset_and_restart(cmd, unit)
        record.update(returncode=result.returncode, stderr=result.stderr)
    except subprocess.TimeoutExpired as exc:
        record.update(timeout=exc.timeout)
    record["elapsed"] = time.monotonic() - began
    deadline = time.monotonic() + 20
    while (pid() in (old, "0", "")) and time.monotonic() < deadline:
        time.sleep(.1)
    record.update(old_pid=old, new_pid=pid(), active=run(cmd + ["is-active", unit]).stdout.strip())
    missing = _systemctl_reset_and_restart(cmd, unit + "-missing")
    record["missing_unit_error_preserved"] = missing.returncode != 0 and bool(missing.stderr)
finally:
    record["cleanup_stop_rc"] = run(cmd + ["stop", unit]).returncode
    run(cmd + ["reset-failed", unit])
    record["inactive_after_cleanup"] = run(cmd + ["is-active", unit]).returncode != 0
    Path(output).write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record, indent=2))
