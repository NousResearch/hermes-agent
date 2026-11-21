#!/usr/bin/env bash
# Full A/B eval: N models x 2 arms x 9 tasks x R reps.
#
# Usage:
#   ./run_all.sh <baseline-tree> <fixes-tree> [reps] [model ...]
#
#   baseline-tree  checkout of the code WITHOUT the changes (e.g. a worktree
#                  of origin/main)
#   fixes-tree     checkout WITH the changes (e.g. your integration branch)
#   reps           repetitions per cell (default 3)
#   model ...      models to test (default: the Aug 2026 pair)
#
# Requires: ABEVAL_HOME pointing at a configured Hermes home (see README.md),
# and this script run with the python that has hermes-agent's deps installed.
set -euo pipefail

BASE=${1:?usage: run_all.sh <baseline-tree> <fixes-tree> [reps] [model ...]}
FIXES=${2:?usage: run_all.sh <baseline-tree> <fixes-tree> [reps] [model ...]}
REPS=${3:-3}
shift $(( $# >= 3 ? 3 : 2 ))
MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=("anthropic/claude-sonnet-4.5" "qwen/qwen3-coder-30b-a3b-instruct")
fi

PY=${PYTHON:-python3}
EVAL="$(cd "$(dirname "$0")" && pwd)/ab_eval.py"
export ABEVAL_ROOT=${ABEVAL_ROOT:-$PWD/abeval-workspace}

status=0
run_batch() {
  local batch_status=0
  "$PY" "$EVAL" run "$@" || batch_status=$?
  case "$batch_status" in
    0) ;;
    3) status=3 ;; # Missing measurements, but the gateway shut down cleanly.
    *) return "$batch_status" ;; # Interrupts and lifecycle failures stop the battery.
  esac
}

for model in "${MODELS[@]}"; do
  echo "=== $model / baseline ==="
  run_batch --arm baseline --model "$model" --reps "$REPS" --pythonpath "$BASE"
  echo "=== $model / fixes ==="
  run_batch --arm fixes --model "$model" --reps "$REPS" --pythonpath "$FIXES"
done
echo "=== ALL RUNS DONE ==="
"$PY" "$EVAL" report --models "$(IFS=,; echo "${MODELS[*]}")"
exit "$status"
