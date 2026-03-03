#!/bin/bash

# YC-Bench Evaluation
#
# Run from repo root:
#   bash environments/benchmarks/yc_bench/run_eval.sh
#
# Override model:
#   bash environments/benchmarks/yc_bench/run_eval.sh \
#       --openai.model_name anthropic/claude-opus-4-6
#
# Run a single preset:
#   bash environments/benchmarks/yc_bench/run_eval.sh \
#       --env.presets '["fast_test"]' --env.seeds '[1]'

mkdir -p logs evals/yc-bench
LOG_FILE="logs/yc_bench_$(date +%Y%m%d_%H%M%S).log"

echo "YC-Bench Evaluation"
echo "Log: $LOG_FILE"
echo ""

python environments/benchmarks/yc_bench/yc_bench_env.py evaluate \
  --config environments/benchmarks/yc_bench/default.yaml \
  "$@" \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "Log saved to: $LOG_FILE"
