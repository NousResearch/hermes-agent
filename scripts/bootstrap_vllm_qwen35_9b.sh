#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/dachen/.hermes/hermes-agent"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
VLLM_VENV="${HERMES_VLLM_VENV:-$HERMES_HOME/vllm-venv}"
BOOTSTRAP_LOG="${HERMES_VLLM_BOOTSTRAP_LOG:-$HERMES_HOME/logs/vllm-bootstrap.log}"

mkdir -p "$(dirname "$BOOTSTRAP_LOG")"
exec >>"$BOOTSTRAP_LOG" 2>&1

echo "[$(date -Is)] bootstrap start"

if [[ ! -f "$VLLM_VENV/bin/activate" ]]; then
  python3.11 -m venv "$VLLM_VENV"
fi

source "$VLLM_VENV/bin/activate"

python - <<'PY' >/dev/null 2>&1 || pip install -U pip setuptools wheel
import vllm
PY

python - <<'PY' >/dev/null 2>&1 || pip install vllm --extra-index-url https://wheels.vllm.ai/nightly
import vllm
PY

python - <<'PY' >/dev/null 2>&1 || pip install bitsandbytes huggingface_hub
import bitsandbytes, huggingface_hub
PY

cd "$REPO_DIR"
./scripts/start_vllm_qwen35_9b.sh || true

echo "[$(date -Is)] waiting for vLLM health"
for _ in $(seq 1 720); do
  if curl -fsS http://127.0.0.1:8010/health >/dev/null 2>&1; then
    echo "[$(date -Is)] vLLM health ok"
    break
  fi
  sleep 10
done

if ! curl -fsS http://127.0.0.1:8010/health >/dev/null 2>&1; then
  echo "[$(date -Is)] vLLM did not become healthy in time"
  exit 1
fi

source "$REPO_DIR/venv/bin/activate"
python - <<'PY'
from pathlib import Path
import yaml

cfg_path = Path("/home/dachen/.hermes/config.yaml")
cfg = yaml.safe_load(cfg_path.read_text()) or {}

cfg.setdefault("model", {})
cfg["model"]["provider"] = "vllm-local"
cfg["model"]["default"] = "qwen3.5:9b"
cfg["model"]["base_url"] = "http://127.0.0.1:8010/v1"
cfg["model"]["api_mode"] = "chat_completions"
cfg["model"]["context_length"] = 65536

cfg.setdefault("auxiliary", {})
cfg["auxiliary"].setdefault("compression", {})
cfg["auxiliary"]["compression"]["provider"] = "vllm-local"
cfg["auxiliary"]["compression"]["model"] = "qwen3.5:9b"
cfg["auxiliary"]["compression"]["base_url"] = "http://127.0.0.1:8010/v1"
cfg["auxiliary"]["compression"]["context_length"] = 65536

cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
PY

pkill -f "hermes gateway run" || true
setsid bash -lc 'source /home/dachen/.hermes/hermes-agent/venv/bin/activate && hermes gateway run --replace >> /home/dachen/.hermes/logs/gateway.log 2>&1' >/dev/null 2>&1 < /dev/null &

echo "[$(date -Is)] gateway restarted on vLLM default"
