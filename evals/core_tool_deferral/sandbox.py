"""Keep an eval agent's host-level side effects inside the run's sandbox.

HERMES_HOME isolation covers Hermes state only. Tasks like "schedule it nightly at 2am" let
the agent reach the operator's real crontab, systemd user manager and home directory: a
past run of ``longrange_backup_pipeline`` left three nightly cron jobs and two scripts on the
operator's machine, pointing into run dirs that no longer existed.

``isolate_host`` uses the product's own contract for this (``TERMINAL_HOME_MODE=profile``
pins every terminal child's HOME to ``{HERMES_HOME}/home``), points the worker's own HOME
there too so in-process ``~`` expansion follows, puts a file-backed ``crontab`` first on
PATH, and drops the user-bus variables so ``systemctl --user`` cannot reach the real manager.
"""
import os
import stat

CRONTAB_FILE_ENV = "EVAL_CRONTAB_FILE"

_CRONTAB_SHIM = """#!/bin/sh
# Eval sandbox crontab: the usual CLI, backed by a file inside the run dir.
f="${EVAL_CRONTAB_FILE:?}"
case "$1" in
  -l) if [ -s "$f" ]; then cat "$f"; else echo "no crontab for $(id -un)" >&2; exit 1; fi ;;
  -r) rm -f "$f" ;;
  -e) "${VISUAL:-${EDITOR:-vi}}" "$f" ;;
  ""|-) cat > "$f" ;;
  *) cat "$1" > "$f" ;;
esac
"""


def isolate_host(run_root: str, hermes_home: str) -> str:
    """Sandbox HOME, crontab and the systemd user bus for this process and its children.

    Returns the path of the file the sandboxed ``crontab`` reads and writes.
    """
    home = os.path.join(hermes_home, "home")
    bin_dir = os.path.join(run_root, "bin")
    os.makedirs(home, exist_ok=True)
    os.makedirs(bin_dir, exist_ok=True)
    shim = os.path.join(bin_dir, "crontab")
    with open(shim, "w", encoding="utf-8") as f:
        f.write(_CRONTAB_SHIM)
    os.chmod(shim, os.stat(shim).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    crontab_file = os.path.join(run_root, "crontab")
    os.environ[CRONTAB_FILE_ENV] = crontab_file
    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    os.environ["HOME"] = home
    os.environ["TERMINAL_HOME_MODE"] = "profile"
    for var in ("DBUS_SESSION_BUS_ADDRESS", "XDG_RUNTIME_DIR"):
        os.environ.pop(var, None)
    return crontab_file
