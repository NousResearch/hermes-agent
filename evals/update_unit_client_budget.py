"""Real disposable user-systemd transaction, never a production gateway."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

repo, tag, output = sys.argv[1:4]
catchup = sys.argv[4:] == ["--catchup"]
if not tag.isalnum():
    raise SystemExit("tag must be alphanumeric")
home = Path(tempfile.mkdtemp(prefix=f"updater-{tag}-"))
allowed = {key: os.environ[key] for key in ("PATH", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS") if key in os.environ}
os.environ.clear()
os.environ.update(allowed, HOME=str(home), HERMES_HOME=str(home / "hermes"))
sys.path.insert(0, repo)
from hermes_cli import update_cmd_fleet as fleet
_systemctl_reset_and_restart = fleet._systemctl_reset_and_restart
actual_systemctl = fleet._systemctl
def scoped_systemctl(argv, *, timeout):
    if "list-units" in argv:
        if "--user" not in argv:
            return subprocess.CompletedProcess(argv, 0, "", "")
        argv = [arg for arg in argv if arg not in ("hermes-gateway*", "hermes-serve*")] + [unit + ".service"]
    return actual_systemctl(argv, timeout=timeout)
if catchup:
    # Bound discovery to our transient unit; never enumerate user services.
    fleet._systemctl = scoped_systemctl
unit = f"hermes-serve-audit-089c35aa-{tag}"
cmd = ["systemctl", "--user"]
def run(args):
    return subprocess.run(args, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=65)
def pid():
    return run(cmd + ["show", unit, "--property=MainPID", "--value"]).stdout.strip()
record: dict[str, object] = {"repo": repo, "tag": tag, "unit": unit, "home": str(home), "tier": "native disposable systemd service"}
try:
    start = run(["systemd-run", "--user", "--unit", unit, f"--property=ExecStop=/bin/sleep {31 if catchup else 16}", "--property=TimeoutStopSec=45", "--property=TimeoutStartSec=30", "/bin/sleep", "infinity"])
    assert start.returncode == 0, start.stderr
    old = pid()
    began = time.monotonic()
    try:
        if catchup:
            failed = []
            fleet._restart_systemd_gateway_units_best_effort(failed, list(fleet._systemd_gateway_unit_listings()))
            record.update(failed_units=failed)
        else:
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
