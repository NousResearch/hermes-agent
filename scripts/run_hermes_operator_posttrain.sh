#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
CONFIG="${HERMES_OPERATOR_CONFIG:-training/local_qlora_config.yaml}"
SFT_JSONL="${HERMES_OPERATOR_SFT:-training/corpora/hermes_operator_sft.jsonl}"
DPO_JSONL="${HERMES_OPERATOR_DPO:-training/corpora/hermes_operator_dpo.seed.jsonl}"
MERGED_DIR="${HERMES_OPERATOR_MERGED_DIR:-training/runs/hermes-operator-qlora/merged}"
LORA_DIR="${HERMES_OPERATOR_LORA_DIR:-training/runs/hermes-operator-qlora}"
OUTPUT_GGUF="${HERMES_OPERATOR_OUTPUT_GGUF:-training/runs/hermes-operator-q8_0.gguf}"
QUANTIZATION="${HERMES_OPERATOR_QUANTIZATION:-Q8_0}"

need_file() {
  if [[ ! -f "$1" ]]; then
    echo "missing file: $1" >&2
    exit 1
  fi
}

need_dir() {
  if [[ ! -d "$1" ]]; then
    echo "missing directory: $1" >&2
    exit 1
  fi
}

run_readiness() {
  need_file "$CONFIG"
  need_file "$SFT_JSONL"
  python scripts/check_training_ready.py \
    --sft "$SFT_JSONL" \
    --dpo "$DPO_JSONL" \
    --qlora-config "$CONFIG"
}

run_preprocess() {
  run_readiness
  axolotl preprocess "$CONFIG" --debug --debug-num-examples 3
}

run_train() {
  run_readiness
  axolotl train "$CONFIG"
}

run_merge() {
  run_readiness
  need_dir "$LORA_DIR"
  axolotl merge-lora "$CONFIG" --lora-model-dir="$LORA_DIR"
  need_dir "$MERGED_DIR"
}

run_gguf() {
  need_dir "$MERGED_DIR"
  if [[ -z "${LLAMA_CPP_ROOT:-}" ]]; then
    echo "LLAMA_CPP_ROOT is required for GGUF export" >&2
    exit 1
  fi
  need_file "$LLAMA_CPP_ROOT/convert_hf_to_gguf.py"
  if [[ -x "$LLAMA_CPP_ROOT/build/bin/llama-quantize" ]]; then
    QUANTIZE="$LLAMA_CPP_ROOT/build/bin/llama-quantize"
  elif [[ -x "$LLAMA_CPP_ROOT/build/bin/Release/llama-quantize" ]]; then
    QUANTIZE="$LLAMA_CPP_ROOT/build/bin/Release/llama-quantize"
  elif [[ -x "$LLAMA_CPP_ROOT/llama-quantize" ]]; then
    QUANTIZE="$LLAMA_CPP_ROOT/llama-quantize"
  else
    echo "llama-quantize not found under LLAMA_CPP_ROOT" >&2
    exit 1
  fi

  mkdir -p "$(dirname "$OUTPUT_GGUF")"
  F16_GGUF="${OUTPUT_GGUF%.gguf}.f16.gguf"
  python "$LLAMA_CPP_ROOT/convert_hf_to_gguf.py" --outfile "$F16_GGUF" --outtype f16 "$MERGED_DIR"
  "$QUANTIZE" "$F16_GGUF" "$OUTPUT_GGUF" "$QUANTIZATION"
  echo "wrote GGUF: $OUTPUT_GGUF"
}

case "$MODE" in
  ready) run_readiness ;;
  preprocess) run_preprocess ;;
  train) run_train ;;
  merge) run_merge ;;
  gguf) run_gguf ;;
  all)
    run_preprocess
    run_train
    run_merge
    run_gguf
    ;;
  *)
    echo "usage: $0 [ready|preprocess|train|merge|gguf|all]" >&2
    exit 2
    ;;
esac
