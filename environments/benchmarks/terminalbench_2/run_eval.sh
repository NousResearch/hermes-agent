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
#
# All terminal settings (backend, timeout, lifetime, pool size) are
# configured via env config fields -- no env vars needed.

set -euo pipefail

mkdir -p logs evals/terminal-bench-2
LOG_FILE="logs/terminalbench2_$(date +%Y%m%d_%H%M%S).log"

echo "Terminal-Bench 2.0 Evaluation"
echo "Log file: $LOG_FILE"
echo ""

# Unbuffered python output so logs are written in real-time
export PYTHONUNBUFFERED=1

# Show INFO-level agent loop timing (api/tool durations per turn)
# These go to the log file; tqdm + [START]/[PASS]/[FAIL] go to terminal
export LOGLEVEL=INFO

python terminalbench2_env.py evaluate \
  --config default.yaml \
  "$@" \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "Log saved to: $LOG_FILE"
echo "Eval results: evals/terminal-bench-2/"
