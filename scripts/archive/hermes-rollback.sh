#!/usr/bin/env bash
#
# hermes-rollback — Restore Hermes state from a pre-update backup (or any backup zip).
#
# Usage:
#   hermes-rollback           — Interactive: list backups and pick one
#   hermes-rollback --latest  — Use the most recent pre-update backup
#   hermes-rollback --backup /path/to/zip  — Restore from a specific zip
#   hermes-rollback --list    — Just list available backups
#   hermes-rollback --help    — Show usage
#
# What it does:
#   1. Stops the gateway service (if running)
#   2. Creates a pre-rollback snapshot of current state
#   3. Unzips the chosen backup over ~/.hermes/
#   4. Warns if hermes-agent/ git repo may need manual rollback
#   5. Restarts the gateway service
#   6. Opens the logs for verification
#
# Exit codes:
#   0 — success
#   1 — usage error or user abort
#   2 — backup not found or unreadable
#   3 — restore failed
#   4 — gateway restart failed
#

set -euo pipefail

HERMES_HOME="${HOME}/.hermes"
BACKUPS_DIR="${HERMES_HOME}/backups"
PRE_ROLLBACK_DIR="${HERMES_HOME}/backups"

# Colours (disable if not a tty)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'
if [[ ! -t 1 ]]; then
  RED=''; GREEN=''; YELLOW=''; CYAN=''; NC=''
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log_info()  { printf "${CYAN}[INFO]${NC}  %s\n" "$*"; }
log_ok()    { printf "${GREEN}[OK]${NC}    %s\n" "$*"; }
log_warn()  { printf "${YELLOW}[WARN]${NC}  %s\n" "$*" >&2; }
log_err()   { printf "${RED}[ERR]${NC}   %s\n" "$*" >&2; }

die() {
  log_err "$1"
  exit "${2:-1}"
}

list_backups() {
  if [[ ! -d "$BACKUPS_DIR" ]]; then
    log_warn "Backups directory does not exist: $BACKUPS_DIR"
    return
  fi
  local i=1
  while IFS= read -r -d '' f; do
    local sz
    sz=$(du -h "$f" 2>/dev/null | cut -f1)
    printf "  %2d) %s  (%s)\n" "$i" "$(basename "$f")" "$sz"
    ((i++)) || true
  done < <(find "$BACKUPS_DIR" -maxdepth 1 -type f -name "pre-update-*.zip" -print0 | sort -rz)

  if [[ $i -eq 1 ]]; then
    log_warn "No pre-update backups found in $BACKUPS_DIR"
  fi
}

latest_backup() {
  find "$BACKUPS_DIR" -maxdepth 1 -type f -name "pre-update-*.zip" -print0 2>/dev/null \
    | sort -rz \
    | head -z -n 1 \
    | tr -d '\0'
}

gateway_running() {
  systemctl --user is-active hermes-gateway &>/dev/null || \
    systemctl is-active hermes-gateway &>/dev/null || \
    pgrep -f "hermes.*gateway" >/dev/null 2>&1
}

stop_gateway() {
  log_info "Stopping gateway..."
  if systemctl --user is-active hermes-gateway &>/dev/null; then
    systemctl --user stop hermes-gateway || true
  elif systemctl is-active hermes-gateway &>/dev/null; then
    sudo systemctl stop hermes-gateway || true
  fi
  # Wait for process to actually exit
  local n=0
  while gateway_running && [[ $n -lt 30 ]]; do
    sleep 0.5
    ((n++)) || true
  done
  if gateway_running; then
    log_warn "Gateway still running after 15s — forcing kill"
    pkill -f "hermes.*gateway" || true
    sleep 1
  fi
}

start_gateway() {
  log_info "Starting gateway..."
  if systemctl --user is-active hermes-gateway &>/dev/null || \
     systemctl --user start hermes-gateway 2>/dev/null; then
    log_ok "Gateway started (user service)"
  elif systemctl is-active hermes-gateway &>/dev/null || \
       sudo systemctl start hermes-gateway 2>/dev/null; then
    log_ok "Gateway started (system service)"
  else
    log_err "Could not start gateway. Check: hermes gateway status"
    return 1
  fi
}

# Create an emergency snapshot of current state before we overwrite it.
snapshot_current() {
  local stamp
  stamp=$(date +%Y%m%d-%H%M%S)
  local snap="${PRE_ROLLBACK_DIR}/pre-rollback-${stamp}.zip"
  log_info "Creating pre-rollback snapshot: $(basename "$snap") ..."

  # Use python to call Hermes' own backup logic so we get SQLite-safe copies
  python3 - "$snap" "$HERMES_HOME" <<'PY' 2>/dev/null || true
import sys, zipfile, os, sqlite3, tempfile, shutil
from pathlib import Path

out_path = Path(sys.argv[1])
hermes_root = Path(sys.argv[2])

EXCLUDED_DIRS = {"hermes-agent", "__pycache__", ".git", "node_modules",
                  "backups", "checkpoints"}
EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".db-wal", ".db-shm", ".db-journal")
EXCLUDED_NAMES = {"gateway.pid", "cron.pid"}

def should_exclude(rel_path):
    for part in rel_path.parts:
        if part in EXCLUDED_DIRS:
            return True
    if rel_path.name in EXCLUDED_NAMES:
        return True
    if str(rel_path).endswith(EXCLUDED_SUFFIXES):
        return True
    return False

def safe_copy_db(src, dst):
    try:
        conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
        backup_conn = sqlite3.connect(str(dst))
        conn.backup(backup_conn)
        backup_conn.close(); conn.close()
        return True
    except Exception:
        try:
            shutil.copy2(src, dst)
            return True
        except Exception:
            return False

files_to_add = []
for dirpath, dirnames, filenames in os.walk(hermes_root, followlinks=False):
    dp = Path(dirpath)
    dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
    for fname in filenames:
        fpath = dp / fname
        try:
            rel = fpath.relative_to(hermes_root)
        except ValueError:
            continue
        if should_exclude(rel):
            continue
        if fpath.resolve() == out_path.resolve():
            continue
        files_to_add.append((fpath, rel))

if not files_to_add:
    sys.exit(0)

with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for abs_path, rel_path in files_to_add:
        try:
            if abs_path.suffix == ".db":
                with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
                    tmp_db = Path(tmp.name)
                try:
                    if safe_copy_db(abs_path, tmp_db):
                        zf.write(tmp_db, arcname=str(rel_path))
                finally:
                    tmp_db.unlink(missing_ok=True)
            else:
                zf.write(abs_path, arcname=str(rel_path))
        except Exception:
            continue
PY

  if [[ -f "$snap" ]]; then
    local sz
    sz=$(du -h "$snap" | cut -f1)
    log_ok "Pre-rollback snapshot saved: $(basename "$snap") ($sz)"
    echo "$snap"
  else
    log_warn "Could not create automatic pre-rollback snapshot"
    echo ""
  fi
}

# ---------------------------------------------------------------------------
# Restore logic
# ---------------------------------------------------------------------------

restore_backup() {
  local zipfile="$1"

  if [[ ! -f "$zipfile" ]]; then
    die "Backup not found: $zipfile" 2
  fi

  log_info "Selected backup: $(basename "$zipfile")"
  log_info "Hermes home:  $HERMES_HOME"
  echo

  # Safety check
  if [[ ! -f "$HERMES_HOME/config.yaml" ]]; then
    log_warn "$HERMES_HOME/config.yaml not found — are you sure this is the right HERMES_HOME?"
    read -rp "Continue anyway? [y/N] " ans
    [[ "${ans,,}" =~ ^y(es)?$ ]] || die "Aborted." 1
  fi

  # Snapshot current state first
  snapshot_current
  echo

  # Stop gateway
  stop_gateway
  echo

  # Also kill any orphaned hermes processes (per skill cleanup recipe)
  log_info "Cleaning up orphaned hermes processes..."
  pkill -f "hermes.*gateway" 2>/dev/null || true
  pkill -f "python.*hermes_cli" 2>/dev/null || true
  sleep 1

  # Unzip backup
  log_info "Restoring backup..."
  if unzip -o "$zipfile" -d "$HERMES_HOME"; then
    log_ok "Backup restored to $HERMES_HOME"
  else
    die "Failed to unzip backup. Your Hermes home may be in a bad state." 3
  fi
  echo

  # Warn about git repo
  log_warn "Backups do NOT include the hermes-agent/ source code."
  log_warn "If the update broke the source code itself, you must roll back"
  log_warn "the git repo manually:"
  echo "    cd ~/repos/KenseiAgent"
  echo "    git log --oneline -10"
  echo "    git reset --hard <good-commit>"
  echo

  # Set permissions
  chmod 600 "$HERMES_HOME/.env" 2>/dev/null || true
  chmod 600 "$HERMES_HOME/config.yaml" 2>/dev/null || true

  # Restart gateway
  echo
  start_gateway

  # Summary
  echo
  log_ok "Rollback complete."
  log_info "Gateway status:"
  hermes gateway status 2>/dev/null || systemctl --user status hermes-gateway --no-pager 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

usage() {
  cat <<EOF
Usage: hermes-rollback [OPTION]

Restore Hermes state from a pre-update backup zip.

Options:
  --latest              Use the most recent pre-update backup
  --backup FILE         Restore from a specific zip file
  --list                List available backups and exit
  --help                Show this help

Backups are stored in: ${BACKUPS_DIR}
Hermes home is:         ${HERMES_HOME}

Examples:
  hermes-rollback              # Interactive menu
  hermes-rollback --latest     # Non-interactive, restore latest
  hermes-rollback --list       # Just peek at what you've got
EOF
}

main() {
  local backup=""
  local do_list=false

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --latest)
        backup=$(latest_backup)
        if [[ -z "$backup" ]]; then
          die "No pre-update backups found in $BACKUPS_DIR" 2
        fi
        ;;
      --backup)
        shift
        backup="${1:-}"
        [[ -n "$backup" ]] || die "--backup requires a file path" 1
        ;;
      --list)
        do_list=true
        ;;
      --help|-h)
        usage
        exit 0
        ;;
      *)
        die "Unknown option: $1 (try --help)" 1
        ;;
    esac
    shift
  done

  if $do_list; then
    echo "Available pre-update backups:"
    list_backups
    exit 0
  fi

  # Interactive mode
  if [[ -z "$backup" ]]; then
    echo "Hermes pre-update backups:"
    list_backups
    echo

    if [[ ! -d "$BACKUPS_DIR" ]] || [[ -z "$(ls -A "$BACKUPS_DIR"/*.zip 2>/dev/null)" ]]; then
      die "No backups available. Run 'hermes update' to create one." 2
    fi

    read -rp "Enter number to restore, or 'q' to quit: " choice
    [[ "$choice" == "q" ]] && die "Aborted." 1

    # Map choice to file
    local i=1
    while IFS= read -r -d '' f; do
      if [[ "$i" == "$choice" ]]; then
        backup="$f"
        break
      fi
      ((i++)) || true
    done < <(find "$BACKUPS_DIR" -maxdepth 1 -type f -name "pre-update-*.zip" -print0 | sort -rz)

    if [[ -z "$backup" ]]; then
      die "Invalid choice: $choice" 2
    fi
  fi

  restore_backup "$backup"
}

main "$@"
