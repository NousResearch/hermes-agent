#!/usr/bin/env bash
set -euo pipefail

PREFIX="${1:-${PREFIX:-$HOME/.local/bin}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEMD_BIN="${MEMD_BIN:-memd}"

mkdir -p "$PREFIX"
install -m 0755 "$SCRIPT_DIR/memd-context.sh" "$PREFIX/memd-context"
install -m 0755 "$SCRIPT_DIR/memd-capture.sh" "$PREFIX/memd-capture"
install -m 0755 "$SCRIPT_DIR/memd-spill.sh" "$PREFIX/memd-spill"
install -m 0755 "$SCRIPT_DIR/memd-stop-save.sh" "$PREFIX/memd-stop-save"
install -m 0755 "$SCRIPT_DIR/memd-precompact-save.sh" "$PREFIX/memd-precompact-save"

cat > "$PREFIX/memd-hook-context" <<EOF
#!/usr/bin/env bash
exec "$PREFIX/memd-context" "\$@"
EOF
chmod +x "$PREFIX/memd-hook-context"

cat > "$PREFIX/memd-hook-spill" <<EOF
#!/usr/bin/env bash
exec "$MEMD_BIN" hook spill "\$@"
EOF
chmod +x "$PREFIX/memd-hook-spill"

cat > "$PREFIX/memd-hook-capture" <<EOF
#!/usr/bin/env bash
exec "$PREFIX/memd-capture" "\$@"
EOF
chmod +x "$PREFIX/memd-hook-capture"

cat > "$PREFIX/memd-hook-stop-save" <<EOF
#!/usr/bin/env bash
exec "$PREFIX/memd-stop-save" "\$@"
EOF
chmod +x "$PREFIX/memd-hook-stop-save"

cat > "$PREFIX/memd-hook-precompact-save" <<EOF
#!/usr/bin/env bash
exec "$PREFIX/memd-precompact-save" "\$@"
EOF
chmod +x "$PREFIX/memd-hook-precompact-save"

echo "Installed memd hooks to $PREFIX"
echo "Add $PREFIX to PATH if needed."
echo "Set MEMD_BIN if the memd CLI is not already on PATH."
