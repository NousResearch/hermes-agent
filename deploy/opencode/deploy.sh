#!/usr/bin/env bash
# Deploy opencode_task (hermes-agent feat/opencode-task-tool) plus the hardened
# opencode wrapper. Dry run by default; --apply writes; --restart also restarts
# the agent services (only when no chat is running, see the WebUI watchdog note).
#
#   deploy.sh                 show what would change
#   deploy.sh --apply         merge the branch into the live checkout, install
#                             wrapper + key file, tighten opencode.jsonc,
#                             add gate check g (warning only), run tests + gate
#   deploy.sh --apply --restart   same, then restart hermes-mux-gateway and the WebUI
#   deploy.sh --wrapper-only  (re)install only the wrapper (after `npm i -g @opencode/cli@<version>`)
#   deploy.sh --rollback      undo using the state file written by --apply
set -euo pipefail

HOME_DIR="$(getent passwd "$(id -un)" | cut -d: -f6)"
LIVE="$HOME_DIR/.hermes/hermes-agent"
BRANCH="feat/opencode-task-tool"
HERE="$(cd "$(dirname "$0")" && pwd)"
WRAPPER_DST="$HOME_DIR/.npm-global/bin/opencode"
REAL_BIN="$HOME_DIR/.npm-global/lib/node_modules/@opencode/cli/bin/opencode.exe"  # OpenCode 2 (npm @opencode/cli)
SLOT_DIR="$HOME_DIR/.hermes/run/opencode-slots"
KEY_FILE="$HOME_DIR/.config/opencode/omniroute.key"
OC_CONFIG="$HOME_DIR/.config/opencode/opencode.jsonc"
GATE="$HOME_DIR/.hermes/scripts/verify_security_controls.sh"
STAMP="$(date +%Y%m%d-%H%M%S)"
STATE="$HOME_DIR/.hermes/backups/opencode-task-deploy.state"

MODE="dry"; RESTART=0
for arg in "$@"; do
  case "$arg" in
    --apply) MODE="apply" ;;
    --restart) RESTART=1 ;;
    --wrapper-only) MODE="wrapper" ;;
    --rollback) MODE="rollback" ;;
    *) echo "unknown argument: $arg" >&2; exit 64 ;;
  esac
done

say() { echo "[$MODE] $*"; }

install_wrapper() {
  [ -x "$REAL_BIN" ] || { echo "opencode binary missing: $REAL_BIN" >&2; exit 1; }
  if [ -e "$WRAPPER_DST" ] || [ -L "$WRAPPER_DST" ]; then
    cp -P "$WRAPPER_DST" "$WRAPPER_DST.bak-$STAMP"
    echo "WRAPPER_BACKUP=$WRAPPER_DST.bak-$STAMP" >> "$STATE"
    rm -f "$WRAPPER_DST"
  fi
  sed -e "s#__OPENCODE_REAL_BIN__#$REAL_BIN#" -e "s#__OPENCODE_SLOT_DIR__#$SLOT_DIR#" \
      -e "s#__OPENCODE_KEY_FILE__#$KEY_FILE#" "$HERE/opencode-wrapper.sh" > "$WRAPPER_DST"
  chmod 755 "$WRAPPER_DST"
  mkdir -p "$SLOT_DIR" && chmod 700 "$SLOT_DIR"
}

if [ "$MODE" = "rollback" ]; then
  [ -f "$STATE" ] || { echo "no state file $STATE" >&2; exit 1; }
  # shellcheck disable=SC1090
  . "$STATE"
  if [ -n "${PREV_HEAD:-}" ]; then
    git -C "$LIVE" reset --hard "$PREV_HEAD"
    say "live checkout back at $PREV_HEAD"
  fi
  if [ -n "${WRAPPER_BACKUP:-}" ] && { [ -e "$WRAPPER_BACKUP" ] || [ -L "$WRAPPER_BACKUP" ]; }; then
    rm -f "$WRAPPER_DST"; cp -P "$WRAPPER_BACKUP" "$WRAPPER_DST"; say "wrapper restored"
  fi
  if [ -n "${GATE_BACKUP:-}" ] && [ -f "$GATE_BACKUP" ]; then cp -p "$GATE_BACKUP" "$GATE"; say "gate restored"; fi
  if [ "${KEY_CREATED:-0}" = 1 ]; then rm -f "$KEY_FILE"; say "key file removed"; fi
  if [ -n "${OC_CONFIG_MODE:-}" ]; then chmod "$OC_CONFIG_MODE" "$OC_CONFIG"; say "opencode.jsonc mode back to $OC_CONFIG_MODE"; fi
  say "restart hermes-mux-gateway and the WebUI to unload the tool"
  exit 0
fi

if [ "$MODE" = "wrapper" ]; then
  mkdir -p "$(dirname "$STATE")"; install_wrapper; say "wrapper installed at $WRAPPER_DST"; exit 0
fi

command -v bwrap >/dev/null || { echo "bubblewrap (bwrap) is required for opencode_task" >&2; exit 1; }
PREV_HEAD="$(git -C "$LIVE" rev-parse HEAD)"
say "live checkout: $(git -C "$LIVE" branch --show-current) @ ${PREV_HEAD:0:12}"
if [ -n "$(git -C "$LIVE" status --porcelain --untracked-files=no)" ]; then
  echo "live checkout has uncommitted changes; refusing (another session may be working there)" >&2; exit 1
fi
git -C "$LIVE" merge-base --is-ancestor "$PREV_HEAD" "$BRANCH" || {
  echo "$BRANCH does not contain the live HEAD; rebase it first" >&2; exit 1; }
say "would fast-forward to $(git -C "$LIVE" rev-parse --short "$BRANCH") ($BRANCH)"
say "would install wrapper at $WRAPPER_DST (backup kept), slot dir $SLOT_DIR"
say "would write $KEY_FILE (600) from OMNIROUTE_API_KEY in ~/.hermes/.env when absent"
say "would chmod 600 $OC_CONFIG (currently $(stat -c %a "$OC_CONFIG"); it holds a literal API key)"
say "would add gate check g (warning only) to $GATE"
[ "$MODE" = "apply" ] || { echo "dry run: re-run with --apply"; exit 0; }

mkdir -p "$(dirname "$STATE")"
: > "$STATE"
echo "PREV_HEAD=$PREV_HEAD" >> "$STATE"
git -C "$LIVE" merge --ff-only "$BRANCH"
install_wrapper
if [ ! -f "$KEY_FILE" ]; then
  KEY="$(grep -m1 '^OMNIROUTE_API_KEY=' "$HOME_DIR/.hermes/.env" | cut -d= -f2- | tr -d '"')"
  if [ -n "$KEY" ]; then
    umask 077; printf '%s\n' "$KEY" > "$KEY_FILE"; chmod 600 "$KEY_FILE"; echo "KEY_CREATED=1" >> "$STATE"
  fi
fi
echo "OC_CONFIG_MODE=$(stat -c %a "$OC_CONFIG")" >> "$STATE"
chmod 600 "$OC_CONFIG"
if ! grep -q "check_g:" "$GATE"; then
  cp -p "$GATE" "$GATE.bak-opencode-$STAMP"; echo "GATE_BACKUP=$GATE.bak-opencode-$STAMP" >> "$STATE"
  patch -s "$GATE" < "$HERE/verify_security_controls.check_g.patch"
fi
(cd "$LIVE" && ./venv/bin/python -m pytest tests/tools/test_opencode_task_tool.py tests/tools/test_profile_scope_wiring.py -q -p no:cacheprovider)
bash "$GATE"
if [ "$RESTART" = 1 ]; then
  systemctl --user restart hermes-mux-gateway.service
  say "gateway restarted; restart the WebUI when no chat is running"
else
  say "done; restart hermes-mux-gateway and the WebUI (when idle) to load opencode_task"
fi
