#!/usr/bin/env bash
# Full A/B eval: N models x 2 code arms x guidance arms x tasks x R reps.
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
GUIDANCE_ARMS=${ABEVAL_GUIDANCE_ARMS:-all_on,tool_use_off,execution_off,all_off}
IFS=, read -r -a GUIDANCE_ARM_LIST <<< "$GUIDANCE_ARMS"

for model in "${MODELS[@]}"; do
  for guidance_arm in "${GUIDANCE_ARM_LIST[@]}"; do
    echo "=== $model / baseline-$guidance_arm ==="
    "$PY" "$EVAL" run --arm "baseline-$guidance_arm" --guidance-arm "$guidance_arm" --model "$model" --reps "$REPS" --pythonpath "$BASE"
    echo "=== $model / fixes-$guidance_arm ==="
    "$PY" "$EVAL" run --arm "fixes-$guidance_arm" --guidance-arm "$guidance_arm" --model "$model" --reps "$REPS" --pythonpath "$FIXES"
  done
done
echo "=== ALL RUNS DONE ==="
REPORT_ARMS=()
for guidance_arm in "${GUIDANCE_ARM_LIST[@]}"; do
  REPORT_ARMS+=("baseline-$guidance_arm" "fixes-$guidance_arm")
done
"$PY" "$EVAL" report --models "$(IFS=,; echo "${MODELS[*]}")" --arms "$(IFS=,; echo "${REPORT_ARMS[*]}")"
