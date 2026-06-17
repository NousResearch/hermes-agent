#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
HOOK="$ROOT/.git/hooks/pre-commit"

mkdir -p "$(dirname "$HOOK")"

if [[ -f "$HOOK" ]] && ! grep -q "xNova secret scan" "$HOOK"; then
  cp "$HOOK" "$HOOK.before-xnova-secret-scan"
fi

cat > "$HOOK" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

# xNova secret scan
ROOT="$(git rev-parse --show-toplevel)"
exec "$ROOT/scripts/check-secrets.sh" --staged
EOF

chmod +x "$HOOK"
echo "Installed xNova secret pre-commit hook: $HOOK"
