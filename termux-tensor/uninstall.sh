#!/data/data/com.termux/files/usr/bin/bash
set -u
BIN_DIR="$HOME/.local/bin"
for name in tensor tensor-doctor tensor-lm tensor-model tensor-server tensor-status tensor-thermal tensor-setup hermes-tensor tensor-bridge; do
  rm -f "$BIN_DIR/$name"
done
echo "Tensor command wrappers removed."
echo "Model and state data under ~/.hermes were preserved."