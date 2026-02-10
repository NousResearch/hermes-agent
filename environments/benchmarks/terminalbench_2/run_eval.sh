#!/bin/bash

# Terminal-Bench 2.0 Evaluation
#
# Run from repo root:
#   bash environments/benchmarks/terminalbench_2/run_eval.sh
#
# Override model:
#   bash environments/benchmarks/terminalbench_2/run_eval.sh \
#       --openai.model_name anthropic/claude-sonnet-4
#
# Run a subset:
#   bash environments/benchmarks/terminalbench_2/run_eval.sh \
#       --env.task_filter fix-git,git-multibranch

mkdir -p logs evals/terminal-bench-2
LOG_FILE="logs/terminalbench2_$(date +%Y%m%d_%H%M%S).log"

echo "Terminal-Bench 2.0 Evaluation"
echo "Log: $LOG_FILE"
echo ""

export TERMINAL_ENV=modal
export TERMINAL_TIMEOUT=300

python environments/benchmarks/terminalbench_2/terminalbench2_env.py evaluate \
  --config environments/benchmarks/terminalbench_2/default.yaml \
  "$@" \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "Log saved to: $LOG_FILE"
