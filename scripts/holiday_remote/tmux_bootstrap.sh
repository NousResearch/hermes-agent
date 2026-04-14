#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  tmux_bootstrap.sh [session_name] [workdir] [--force-reset] [--attach]

Examples:
  ./scripts/holiday_remote/tmux_bootstrap.sh
  ./scripts/holiday_remote/tmux_bootstrap.sh holiday /home/yeqiuqiu/clawd-architect
  ./scripts/holiday_remote/tmux_bootstrap.sh holiday /home/yeqiuqiu/clawd-architect --force-reset --attach
EOF
}

have() { command -v "$1" >/dev/null 2>&1; }

SESSION="holiday"
WORKDIR="/home/yeqiuqiu/clawd-architect"
FORCE_RESET="false"
AUTO_ATTACH="false"

POSITIONAL=()
for arg in "$@"; do
  case "$arg" in
    --force-reset) FORCE_RESET="true" ;;
    --attach) AUTO_ATTACH="true" ;;
    -h|--help)
      usage
      exit 0
      ;;
    *) POSITIONAL+=("$arg") ;;
  esac
done

if [[ "${#POSITIONAL[@]}" -ge 1 ]]; then
  SESSION="${POSITIONAL[0]}"
fi
if [[ "${#POSITIONAL[@]}" -ge 2 ]]; then
  WORKDIR="${POSITIONAL[1]}"
fi

if ! have tmux; then
  echo "tmux command not found"
  exit 1
fi

if [[ ! -d "$WORKDIR" ]]; then
  echo "workdir does not exist: $WORKDIR"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_OPS="$SCRIPT_DIR/remote_ops.sh"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  if [[ "$FORCE_RESET" == "true" ]]; then
    tmux kill-session -t "$SESSION"
  else
    echo "tmux session already exists: $SESSION"
    echo "Attach: tmux attach -t $SESSION"
    [[ "$AUTO_ATTACH" == "true" ]] && exec tmux attach -t "$SESSION"
    exit 0
  fi
fi

# Window 1: main shell
_tmux_start_cmd="cd '$WORKDIR' && exec bash"
tmux new-session -d -s "$SESSION" -n main -c "$WORKDIR" "bash -lc \"$_tmux_start_cmd\""

# Window 2: status shell with immediate snapshot.
if [[ -x "$REMOTE_OPS" ]]; then
  tmux new-window -t "$SESSION:" -n status -c "$WORKDIR" \
    "bash -lc '$REMOTE_OPS status; echo; echo \"Tip: run watch -n 20 ./scripts/holiday_remote/remote_ops.sh status --compact\"; exec bash'"
else
  tmux new-window -t "$SESSION:" -n status -c "$WORKDIR" "bash"
fi

# Window 3: logs shell for quick tail commands.
tmux new-window -t "$SESSION:" -n logs -c "$WORKDIR" \
  "bash -lc 'echo \"Tip: tail -n 80 state/handover/latest.md\"; echo \"Tip: tail -n 80 reports/handover_context_latest.md\"; exec bash'"

# Keep operator landing in main window.
tmux select-window -t "$SESSION:main"

echo "tmux holiday session ready: $SESSION"
echo "Attach: tmux attach -t $SESSION"

if [[ "$AUTO_ATTACH" == "true" ]]; then
  exec tmux attach -t "$SESSION"
fi
