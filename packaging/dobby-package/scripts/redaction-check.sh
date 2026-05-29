#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: %s PATH [PATH ...]\n' "$(basename "$0")" >&2
}

if [ "$#" -lt 1 ]; then
  usage
  exit 2
fi

tmp_findings="$(mktemp "${TMPDIR:-/tmp}/dobby-redaction-findings.XXXXXX")"
trap 'rm -f "$tmp_findings"' EXIT

fake_value_re='example|placeholder|replace|dummy|fake|canary|not[-_ ]?a[-_ ]?secret|your[-_ ]'

scan_pattern() {
  label="$1"
  regex="$2"
  file="$3"

  matches="$(LC_ALL=C grep -EInIH "$regex" "$file" 2>/dev/null || true)"
  if [ -z "$matches" ]; then
    return 0
  fi

  printf '%s\n' "$matches" | while IFS=: read -r path line rest; do
    if printf '%s\n' "$rest" | LC_ALL=C grep -Eiq "$fake_value_re"; then
      continue
    fi
    printf '%s:%s: %s\n' "$path" "$line" "$label" >>"$tmp_findings"
  done
}

scan_file() {
  file="$1"

  scan_pattern "aws access key" '(^|[^A-Z0-9])(AKIA|ASIA)[A-Z0-9]{16}([^A-Z0-9]|$)' "$file"
  scan_pattern "github token" '(^|[^A-Za-z0-9_])gh[pousr]_[A-Za-z0-9_]{20,}([^A-Za-z0-9_]|$)' "$file"
  scan_pattern "openai-style key" '(^|[^A-Za-z0-9_])sk-[A-Za-z0-9][A-Za-z0-9_-]{20,}([^A-Za-z0-9_-]|$)' "$file"
  scan_pattern "anthropic-style key" '(^|[^A-Za-z0-9_])sk-ant-[A-Za-z0-9_-]{20,}([^A-Za-z0-9_-]|$)' "$file"
  scan_pattern "slack token" '(^|[^A-Za-z0-9_])xox[baprs]-[A-Za-z0-9-]{20,}([^A-Za-z0-9-]|$)' "$file"
  scan_pattern "discord bot token" '(^|[^A-Za-z0-9_])[MN][A-Za-z0-9_-]{20,30}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{20,}([^A-Za-z0-9_-]|$)' "$file"
  scan_pattern "jwt" '(^|[^A-Za-z0-9_])eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}([^A-Za-z0-9_-]|$)' "$file"
  scan_pattern "private key block" '-----BEGIN (RSA |DSA |EC |OPENSSH |PGP )?PRIVATE KEY-----' "$file"
  scan_pattern "long secret assignment" '(^|[^A-Za-z0-9_])(api[_-]?key|token|secret|password|bot[_-]?token)[[:space:]]*[:=][[:space:]]*["'\'']?[A-Za-z0-9_./+=-]{24,}' "$file"
}

scan_path() {
  target="$1"
  if [ ! -e "$target" ]; then
    printf 'redaction-check: missing path: %s\n' "$target" >&2
    exit 2
  fi

  if [ -f "$target" ]; then
    scan_file "$target"
    return 0
  fi

  find "$target" \
    \( -name .git -o -name node_modules -o -name .venv -o -name __pycache__ \) -prune \
    -o -type f -print0 |
    while IFS= read -r -d '' file; do
      scan_file "$file"
    done
}

for target in "$@"; do
  scan_path "$target"
done

if [ -s "$tmp_findings" ]; then
  printf 'Secret-shaped values found:\n' >&2
  sort -u "$tmp_findings" >&2
  exit 1
fi

printf 'redaction-check: no secret-shaped values found\n'
