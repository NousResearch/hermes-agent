#!/usr/bin/env bash
# Rate-limited recon scan wrapper for the web-pentest skill.
# Wraps nmap + whatweb + curl headers; enforces scope.txt.
#
# Usage: recon-scan.sh <engagement-dir> <target-url>
#
# Example:
#   recon-scan.sh engagement-20260525-031415 http://127.0.0.1:9119
set -euo pipefail

ENGAGEMENT_DIR="${1:-}"
TARGET_URL="${2:-}"

if [[ -z "$ENGAGEMENT_DIR" || -z "$TARGET_URL" ]]; then
  echo "usage: $0 <engagement-dir> <target-url>" >&2
  exit 2
fi

if [[ ! -d "$ENGAGEMENT_DIR" ]]; then
  echo "Engagement directory $ENGAGEMENT_DIR does not exist." >&2
  echo "Run Phase 0 (engagement setup) first." >&2
  exit 2
fi

SCOPE_FILE="$ENGAGEMENT_DIR/scope.txt"
AUTH_FILE="$ENGAGEMENT_DIR/authorization.md"
EVIDENCE_DIR="$ENGAGEMENT_DIR/evidence"
LOG_FILE="$ENGAGEMENT_DIR/request-log.jsonl"

if [[ ! -f "$AUTH_FILE" ]]; then
  echo "Missing $AUTH_FILE — no engagement authorization on file." >&2
  echo "Fill out templates/authorization.md before running." >&2
  exit 3
fi

if [[ ! -f "$SCOPE_FILE" ]]; then
  echo "Missing $SCOPE_FILE — no scope allowlist on file." >&2
  exit 3
fi

mkdir -p "$EVIDENCE_DIR"

# Extract host from URL.
HOST="$(python3 -c "import sys, urllib.parse as u; print(u.urlparse(sys.argv[1]).hostname or '')" "$TARGET_URL")"
if [[ -z "$HOST" ]]; then
  echo "Could not parse host from URL: $TARGET_URL" >&2
  exit 4
fi

# Scope check: hostname must appear literally in scope.txt, OR the
# resolved IP must fall inside a CIDR listed there.
in_scope() {
  local host="$1"
  while IFS= read -r line; do
    # strip comments + whitespace
    local entry
    entry="$(printf '%s' "$line" | sed 's/#.*//' | tr -d '[:space:]')"
    [[ -z "$entry" ]] && continue
    if [[ "$entry" == "$host" ]]; then
      return 0
    fi
    # If entry is CIDR, check via python
    if [[ "$entry" == */* ]]; then
      python3 - "$host" "$entry" <<'PY' && return 0
import sys, socket, ipaddress
host, cidr = sys.argv[1], sys.argv[2]
try:
    ip = socket.gethostbyname(host)
    if ipaddress.ip_address(ip) in ipaddress.ip_network(cidr, strict=False):
        sys.exit(0)
except Exception:
    pass
sys.exit(1)
PY
    fi
  done < "$SCOPE_FILE"
  return 1
}

if ! in_scope "$HOST"; then
  echo "Host '$HOST' is NOT in $SCOPE_FILE. Refusing to scan." >&2
  echo "Add it to scope.txt only if it is genuinely authorized." >&2
  exit 5
fi

# Resolve URL for logging
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[recon-scan] target=$TARGET_URL host=$HOST ts=$TS"

# --- headers ---
echo "[recon-scan] fetching headers..."
HEADERS_FILE="$EVIDENCE_DIR/headers.txt"
curl -sSIk --max-time 15 -A "hermes-pentest/recon" "$TARGET_URL" > "$HEADERS_FILE" || true
sleep 0.2

# --- whatweb ---
if command -v whatweb >/dev/null 2>&1; then
  echo "[recon-scan] running whatweb..."
  whatweb -v --no-errors "$TARGET_URL" > "$EVIDENCE_DIR/whatweb.txt" 2>&1 || true
  sleep 0.2
else
  echo "[recon-scan] whatweb not installed — skipping. Install with: apt install whatweb"
fi

# --- robots / sitemap / .well-known ---
echo "[recon-scan] checking robots/sitemap/.well-known..."
for path in robots.txt sitemap.xml .well-known/security.txt; do
  outfile="$EVIDENCE_DIR/$(echo "$path" | tr / _).txt"
  curl -sSk --max-time 10 -A "hermes-pentest/recon" -o "$outfile" -w "%{http_code}\n" "$TARGET_URL/$path" \
       > "$outfile.status" || true
  sleep 0.2
done

# --- nmap (top 100 ports, default scripts off, scope-bounded) ---
if command -v nmap >/dev/null 2>&1; then
  echo "[recon-scan] running nmap (top 100 ports, T3, no NSE)..."
  nmap -sT -T3 --top-ports 100 -Pn -oN "$EVIDENCE_DIR/nmap.txt" "$HOST" >/dev/null 2>&1 || true
else
  echo "[recon-scan] nmap not installed — skipping. Install with: apt install nmap"
fi

# Log entry
printf '{"ts":"%s","phase":"recon","url":"%s","host":"%s","in_scope":true,"evidence_ref":"evidence/"}\n' \
  "$TS" "$TARGET_URL" "$HOST" >> "$LOG_FILE"

echo "[recon-scan] done. Evidence in $EVIDENCE_DIR/"
