#!/data/data/com.termux/files/usr/bin/bash
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="${HOME}/.local/bin"
HOME_DIR="${HERMES_HOME:-$HOME/.hermes}"
MODEL_DIR="$HOME_DIR/models"
TENSOR_DIR="$HOME_DIR/tensor"
mkdir -p "$BIN_DIR" "$MODEL_DIR/litertlm" "$MODEL_DIR/gguf" "$TENSOR_DIR/config" "$TENSOR_DIR/logs" "$TENSOR_DIR/runtime" "$TENSOR_DIR/state" "$TENSOR_DIR/cache"
for f in "$ROOT"/bin/*; do
  [ -f "$f" ] || continue
  name="$(basename "$f")"
  install -m 755 "$f" "$BIN_DIR/$name"
done
cat > "$TENSOR_DIR/config/tensor.conf" <<EOF
TENSOR_HOST=127.0.0.1
TENSOR_PORT=9379
TENSOR_MODEL_DIR=$MODEL_DIR
TENSOR_THERMAL_MODE=balanced
TENSOR_LOG_LEVEL=INFO
EOF
if ! printf '%s' "$PATH" | tr ':' '\n' | grep -qx "$BIN_DIR"; then
  echo "Add this directory to PATH: $BIN_DIR"
fi
echo "Hermes Tensor Termux profile installed."
echo "Run: tensor doctor"
echo "Run: tensor status"