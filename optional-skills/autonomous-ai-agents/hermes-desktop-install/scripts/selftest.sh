#!/usr/bin/env bash
# Offline self-test: script syntax, template rendering, placeholder hygiene.
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
fails=0
pass() { echo "PASS  $*"; }
fail() { echo "FAIL  $*"; fails=$((fails+1)); }

for s in "$HERE"/scripts/*.sh; do
  if bash -n "$s"; then pass "syntax $(basename "$s")"; else fail "syntax $(basename "$s")"; fi
done

# The Webapp is served by the gateway; a shipped systemd unit for it is the bug
# this repo exists to stop people repeating.
if ls "$HERE"/templates/*.service >/dev/null 2>&1; then
  fail "templates/ still ships a systemd unit (the gateway owns the listener)"
else
  pass "no systemd unit templates"
fi

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
site="$HERE/templates/Caddyfile.site"
if [ ! -f "$site" ]; then
  fail "missing templates/Caddyfile.site"
else
  sed -e 's|<DESK_HOST>|desk.example.com|g' -e 's|<PORT>|9120|g' "$site" > "$TMP/Caddyfile"
  if grep -qE '<[A-Z_]+>' "$TMP/Caddyfile"; then
    fail "Caddyfile.site has an unrendered placeholder"
  else
    pass "Caddyfile.site fully rendered"
  fi
  if command -v caddy >/dev/null 2>&1; then
    if caddy validate --config "$TMP/Caddyfile" --adapter caddyfile >"$TMP/out" 2>&1; then
      pass "Caddyfile.site validates"
    else
      fail "Caddyfile.site invalid: $(tail -2 "$TMP/out")"
    fi
  else
    echo "SKIP  caddy not installed"
  fi
fi

if command -v shellcheck >/dev/null 2>&1; then
  for s in "$HERE"/scripts/*.sh; do
    if shellcheck -S warning "$s"; then pass "shellcheck $(basename "$s")"; else fail "shellcheck $(basename "$s")"; fi
  done
else
  echo "SKIP  shellcheck not installed"
fi

echo
echo "failures: $fails"
exit $(( fails > 0 ? 1 : 0 ))
