#!/usr/bin/env bash
# Config validator — checks all Hermes YAML configs for syntax and structure
# Uses $HERMES_HOME (defaults to ~/.hermes)
set -euo pipefail

RED='\033[0;31m'; YEL='\033[1;33m'; GRN='\033[0;32m'; NC='\033[0m'
errors=0; warnings=0
ok()   { echo -e "${GRN}OK${NC} $1"; }
warn() { echo -e "${YEL}WARN${NC} $1"; warnings=$((warnings+1)); }
err()  { echo -e "${RED}ERR${NC} $1"; errors=$((errors+1)); }

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
PY="python3 -c"

# Prefer uv for isolated YAML parsing
if command -v uv &>/dev/null; then
  PY="uv run --with pyyaml python3 -c"
elif python3 -c "import yaml" 2>/dev/null; then
  PY="python3 -c"
else
  echo "ERROR: pyyaml not installed. Run: pip install pyyaml"
  echo "       Or install uv for auto-resolved execution."
  exit 2
fi

validate_yaml() {
  local path="$1" label="$2"
  if [ ! -f "$path" ]; then
    warn "${label}: file not found"
    return 1
  fi
  if $PY "import yaml; yaml.safe_load(open('$path')); print('valid')" 2>/dev/null; then
    ok "${label}: valid YAML"
    return 0
  else
    err "${label}: invalid YAML"
    return 1
  fi
}

check_field() {
  local path="$1" field="$2" label="$3"
  if $PY "
import yaml; c=yaml.safe_load(open('$path'))
keys='$field'.split('.'); v=c
for k in keys: v=v.get(k,{})
print('found' if v else 'missing')
" 2>/dev/null | grep -q found; then
    ok "${label}: $field present"
  else
    err "${label}: $field missing"
  fi
}

echo "=== Config Validation ==="
echo ""

echo "--- Main config ---"
if validate_yaml "${HERMES_HOME}/config.yaml" "config.yaml"; then
  check_field "${HERMES_HOME}/config.yaml" "model.default" "config.yaml"
  check_field "${HERMES_HOME}/config.yaml" "model.provider" "config.yaml"
fi
echo ""

echo "--- Profile configs ---"
for pdir in "${HERMES_HOME}"/profiles/*/; do
  [ -d "$pdir" ] || continue
  name=$(basename "$pdir")
  if validate_yaml "${pdir}config.yaml" "${name}/config.yaml"; then
    check_field "${pdir}config.yaml" "model.default" "${name}/config.yaml"
    check_field "${pdir}config.yaml" "model.provider" "${name}/config.yaml"
  fi
  for f in "profile.yaml" "SOUL.md" "gateway_state.json"; do
    [ -f "${pdir}${f}" ] && ok "${name}: $f exists" || warn "${name}: $f missing"
  done
done
echo ""

echo "--- Gateway states ---"
for pdir in "${HERMES_HOME}"/profiles/*/; do
  [ -d "$pdir" ] || continue
  name=$(basename "$pdir"); sf="${pdir}gateway_state.json"
  if [ -f "$sf" ]; then
    state=$(python3 -c "import json; print(json.load(open('$sf')).get('gateway_state','unknown'))" 2>/dev/null)
    ok "${name}: gateway_state=$state"
  else
    warn "${name}: no gateway_state.json"
  fi
done
echo ""

echo "--- Skills ---"
total=$(ls -d "${HERMES_HOME}"/skills/*/*/SKILL.md 2>/dev/null | wc -l)
ok "${total} skills found across all categories"
echo ""

echo "=== Summary ==="
[ $errors -gt 0 ] && [ $warnings -gt 0 ] && echo -e "${RED}${errors} errors, ${warnings} warnings${NC}" && exit 2
[ $errors -gt 0 ] && echo -e "${RED}${errors} errors${NC}" && exit 2
[ $warnings -gt 0 ] && echo -e "${YEL}${warnings} warnings${NC}" && exit 1
echo -e "${GRN}All clean${NC}" && exit 0
