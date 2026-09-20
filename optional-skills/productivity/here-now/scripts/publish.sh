#!/usr/bin/env bash
set -euo pipefail

BASE_URL="https://here.now"
CREDENTIALS_FILE="$HOME/.herenow/credentials"
API_KEY="${HERENOW_API_KEY:-}"
API_KEY_SOURCE="none"
if [[ -n "${HERENOW_API_KEY:-}" ]]; then
  API_KEY_SOURCE="env"
fi
ALLOW_NON_HERENOW_BASE_URL=0
SLUG=""
CLAIM_TOKEN=""
TITLE=""
DESCRIPTION=""
TTL=""
CLIENT=""
TARGET=""
FORKABLE=""
SPA_MODE=""
FROM_DRIVE=""
DRIVE_VERSION=""

usage() {
  cat <<'USAGE'
Usage: publish.sh <file-or-dir> [options]

Options:
  --api-key <key>         API key (or set $HERENOW_API_KEY)
  --slug <slug>           Update existing publish
  --claim-token <token>   Claim token for anonymous updates
  --title <text>          Viewer title
  --description <text>    Viewer description
  --ttl <seconds>         Expiry (authenticated only)
  --client <name>         Agent name for attribution (e.g. cursor, claude-code)
  --forkable              Allow others to fork this site
  --spa                   Enable SPA routing
  --from-drive <drv_...>  Publish a Drive snapshot instead of local files
  --version <dv_...>      Drive version for --from-drive (default: current head)
  --base-url <url>        API base (default: https://here.now)
  --allow-nonherenow-base-url
                         Allow auth requests to non-default API base URL
USAGE
  exit 1
}

die() { echo "error: $1" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUNDLED_JQ="${SKILL_DIR}/bin/jq"

if [[ -x "$BUNDLED_JQ" ]]; then
  JQ_BIN="$BUNDLED_JQ"
elif command -v jq >/dev/null 2>&1; then
  JQ_BIN="$(command -v jq)"
else
  die "requires jq"
fi

for cmd in curl file; do
  command -v "$cmd" >/dev/null 2>&1 || die "requires $cmd"
done

while [[ $# -gt 0 ]]; do
  case "$1" in
    --api-key)      API_KEY="$2"; API_KEY_SOURCE="flag"; shift 2 ;;
    --slug)         SLUG="$2"; shift 2 ;;
    --claim-token)  CLAIM_TOKEN="$2"; shift 2 ;;
    --title)        TITLE="$2"; shift 2 ;;
    --description)  DESCRIPTION="$2"; shift 2 ;;
    --ttl)          TTL="$2"; shift 2 ;;
    --client)       CLIENT="$2"; shift 2 ;;
    --base-url)     BASE_URL="$2"; shift 2 ;;
    --allow-nonherenow-base-url) ALLOW_NON_HERENOW_BASE_URL=1; shift ;;
    --forkable)     FORKABLE="true"; shift ;;
    --spa)          SPA_MODE="true"; shift ;;
    --from-drive)   FROM_DRIVE="$2"; shift 2 ;;
    --version)      DRIVE_VERSION="$2"; shift 2 ;;
    --help|-h)      usage ;;
    -*)             die "unknown option: $1" ;;
    *)              [[ -z "$TARGET" ]] && TARGET="$1" || die "unexpected argument: $1"; shift ;;
  esac
done

if [[ -n "$FROM_DRIVE" ]]; then
  [[ -z "$TARGET" ]] || die "--from-drive does not accept a local file-or-dir argument"
else
  [[ -n "$TARGET" ]] || usage
  [[ -e "$TARGET" ]] || die "path does not exist: $TARGET"
fi

# Load API key from credentials file if not provided via flag or env
if [[ -z "$API_KEY" && -f "$CREDENTIALS_FILE" ]]; then
  API_KEY=$(cat "$CREDENTIALS_FILE" | tr -d '[:space:]')
  [[ -n "$API_KEY" ]] && API_KEY_SOURCE="credentials"
fi

BASE_URL="${BASE_URL%/}"
STATE_DIR=".herenow"
STATE_FILE="$STATE_DIR/state.json"

# Safety guard: avoid accidentally sending bearer auth to arbitrary endpoints.
if [[ -n "$API_KEY" && "$BASE_URL" != "https://here.now" && "$ALLOW_NON_HERENOW_BASE_URL" -ne 1 ]]; then
  die "refusing to send API key to non-default base URL; pass --allow-nonherenow-base-url to override"
fi

# Auto-load claim token from state file for anonymous updates
if [[ -n "$SLUG" && -z "$CLAIM_TOKEN" && -z "$API_KEY" && -f "$STATE_FILE" ]]; then
  CLAIM_TOKEN=$("$JQ_BIN" -r --arg s "$SLUG" '.publishes[$s].claimToken // empty' "$STATE_FILE" 2>/dev/null || true)
fi

if [[ -n "$FROM_DRIVE" ]]; then
  [[ -n "$API_KEY" ]] || die "--from-drive requires an account API key"
  BODY=$("$JQ_BIN" -n --arg d "$FROM_DRIVE" '{driveId:$d}')
  [[ -n "$DRIVE_VERSION" ]] && BODY=$(echo "$BODY" | "$JQ_BIN" --arg v "$DRIVE_VERSION" '.versionId = $v')
  [[ -n "$SLUG" ]] && BODY=$(echo "$BODY" | "$JQ_BIN" --arg s "$SLUG" '.slug = $s')
  if [[ -n "$TITLE" || -n "$DESCRIPTION" ]]; then
    viewer="{}"
    [[ -n "$TITLE" ]] && viewer=$(echo "$viewer" | "$JQ_BIN" --arg t "$TITLE" '.title = $t')
    [[ -n "$DESCRIPTION" ]] && viewer=$(echo "$viewer" | "$JQ_BIN" --arg d "$DESCRIPTION" '.description = $d')
    BODY=$(echo "$BODY" | "$JQ_BIN" --argjson v "$viewer" '.viewer = $v')
  fi
  [[ "$FORKABLE" == "true" ]] && BODY=$(echo "$BODY" | "$JQ_BIN" '.forkable = true')
  [[ "$SPA_MODE" == "true" ]] && BODY=$(echo "$BODY" | "$JQ_BIN" '.spaMode = true')
  CLIENT_HEADER_VALUE="here-now-publish-sh"
  if [[ -n "$CLIENT" ]]; then
    normalized_client=$(echo "$CLIENT" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9._-' '-')
    normalized_client="${normalized_client#-}"
    normalized_client="${normalized_client%-}"
    if [[ -n "$normalized_client" ]]; then
      CLIENT_HEADER_VALUE="${normalized_client}/publish-sh"
    fi
  fi

  echo "publishing from Drive..." >&2
  RESPONSE=$(curl -sS -X POST "$BASE_URL/api/v1/publish/from-drive" \
    -H "authorization: Bearer $API_KEY" \
    -H "x-herenow-client: $CLIENT_HEADER_VALUE" \
    -H "content-type: application/json" \
    -d "$BODY")
  if echo "$RESPONSE" | "$JQ_BIN" -e '.error' >/dev/null 2>&1; then
    err=$(echo "$RESPONSE" | "$JQ_BIN" -r '.error')
    die "$err"
  fi
  SITE_URL=$(echo "$RESPONSE" | "$JQ_BIN" -r '.siteUrl')
  OUT_SLUG=$(echo "$RESPONSE" | "$JQ_BIN" -r '.slug')
  CURRENT_VERSION=$(echo "$RESPONSE" | "$JQ_BIN" -r '.currentVersionId')
  DRIVE_VERSION_OUT=$(echo "$RESPONSE" | "$JQ_BIN" -r '.driveVersionId')
  echo "$SITE_URL"
  echo "" >&2
  echo "publish_result.site_url=$SITE_URL" >&2
  echo "publish_result.slug=$OUT_SLUG" >&2
  echo "publish_result.action=from_drive" >&2
  echo "publish_result.auth_mode=authenticated" >&2
  echo "publish_result.api_key_source=$API_KEY_SOURCE" >&2
  echo "publish_result.persistence=permanent" >&2
  echo "publish_result.drive_id=$FROM_DRIVE" >&2
  echo "publish_result.drive_version_id=$DRIVE_VERSION_OUT" >&2
  echo "publish_result.current_version_id=$CURRENT_VERSION" >&2
  exit 0
fi

compute_sha256() {
  local f="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$f" | cut -d' ' -f1
  else
    shasum -a 256 "$f" | cut -d' ' -f1
  fi
}

guess_content_type() {
  local f="$1"
  case "${f##*.}" in
    html|htm) echo "text/html; charset=utf-8" ;;
    css)      echo "text/css; charset=utf-8" ;;
    js|mjs)   echo "text/javascript; charset=utf-8" ;;
    json)     echo "application/json; charset=utf-8" ;;
    md|txt)   echo "text/plain; charset=utf-8" ;;
    svg)      echo "image/svg+xml" ;;
    png)      echo "image/png" ;;
    jpg|jpeg) echo "image/jpeg" ;;
    gif)      echo "image/gif" ;;
    webp)     echo "image/webp" ;;
    pdf)      echo "application/pdf" ;;
    mp4)      echo "video/mp4" ;;
    mov)      echo "video/quicktime" ;;
    mp3)      echo "audio/mpeg" ;;
    wav)      echo "audio/wav" ;;
    xml)      echo "application/xml" ;;
    woff2)    echo "font/woff2" ;;
    woff)     echo "font/woff" ;;
    ttf)      echo "font/ttf" ;;
    ico)      echo "image/x-icon" ;;
    *)
      local detected
      detected=$(file --brief --mime-type "$f" 2>/dev/null || echo "application/octet-stream")
      echo "$detected"
      ;;
  esac
}

# Build file manifest as JSON array
FILES_JSON="[]"

if [[ -f "$TARGET" ]]; then
  sz=$(wc -c < "$TARGET" | tr -d ' ')
  ct=$(guess_content_type "$TARGET")
  bn=$(basename "$TARGET")
  h=$(compute_sha256 "$TARGET")
  FILES_JSON=$("$JQ_BIN" -n --arg p "$bn" --argjson s "$sz" --arg c "$ct" --arg h "$h" \
    '[{"path":$p,"size":$s,"contentType":$c,"hash":$h}]')
  FILE_MAP=$("$JQ_BIN" -n --arg p "$bn" --arg a "$(cd "$(dirname "$TARGET")" && pwd)/$(basename "$TARGET")" \
    '{($p):$a}')
elif [[ -d "$TARGET" ]]; then
  FILE_MAP="{}"
  while IFS= read -r -d '' f; do
    rel="${f#$TARGET/}"
    [[ "$rel" == ".DS_Store" ]] && continue
    [[ "$(basename "$rel")" == ".DS_Store" ]] && continue
    [[ "$rel" == ".herenow/fork-meta.json" ]] && continue
    sz=$(wc -c < "$f" | tr -d ' ')
    ct=$(guess_content_type "$f")
    h=$(compute_sha256 "$f")
    abs=$(cd "$(dirname "$f")" && pwd)/$(basename "$f")
    FILES_JSON=$(echo "$FILES_JSON" | "$JQ_BIN" --arg p "$rel" --argjson s "$sz" --arg c "$ct" --arg h "$h" \
      '. + [{"path":$p,"size":$s,"contentType":$c,"hash":$h}]')
    FILE_MAP=$(echo "$FILE_MAP" | "$JQ_BIN" --arg p "$rel" --arg a "$abs" '. + {($p):$a}')
  done < <(find "$TARGET" -type f -print0 | sort -z)
else
  die "not a file or directory: $TARGET"
fi

file_count=$(echo "$FILES_JSON" | "$JQ_BIN" 'length')
[[ "$file_count" -gt 0 ]] || die "no files found"

# Read fork-meta.json defaults if present and no explicit flags given
FORK_META=""
if [[ -d "$TARGET" ]]; then
  FORK_META_PATH="$TARGET/.herenow/fork-meta.json"
  if [[ -f "$FORK_META_PATH" ]]; then
    FORK_META=$(cat "$FORK_META_PATH")
    if [[ -z "$FORKABLE" ]]; then
      FORKABLE=$("$JQ_BIN" -r '.forkable // empty' <<< "$FORK_META" 2>/dev/null || true)
    fi
  fi
fi

# Build request body
BODY=$(echo "$FILES_JSON" | "$JQ_BIN" '{files: .}')

if [[ -n "$TTL" ]]; then
  BODY=$(echo "$BODY" | "$JQ_BIN" --argjson t "$TTL" '.ttlSeconds = $t')
fi

if [[ -n "$TITLE" || -n "$DESCRIPTION" ]]; then
  viewer="{}"
  [[ -n "$TITLE" ]] && viewer=$(echo "$viewer" | "$JQ_BIN" --arg t "$TITLE" '.title = $t')
  [[ -n "$DESCRIPTION" ]] && viewer=$(echo "$viewer" | "$JQ_BIN" --arg d "$DESCRIPTION" '.description = $d')
  BODY=$(echo "$BODY" | "$JQ_BIN" --argjson v "$viewer" '.viewer = $v')
fi

if [[ -n "$CLAIM_TOKEN" && -n "$SLUG" && -z "$API_KEY" ]]; then
  BODY=$(echo "$BODY" | "$JQ_BIN" --arg ct "$CLAIM_TOKEN" '.claimToken = $ct')
fi

if [[ "$FORKABLE" == "true" ]]; then
  BODY=$(echo "$BODY" | "$JQ_BIN" '.forkable = true')
fi

if [[ "$SPA_MODE" == "true" ]]; then
  BODY=$(echo "$BODY" | "$JQ_BIN" '.spaMode = true')
fi

# Determine endpoint and method
if [[ -n "$SLUG" ]]; then
  URL="$BASE_URL/api/v1/publish/$SLUG"
  METHOD="PUT"
else
  URL="$BASE_URL/api/v1/publish"
  METHOD="POST"
fi

# Build auth header
AUTH_ARGS=()
if [[ -n "$API_KEY" ]]; then
  AUTH_ARGS=(-H "authorization: Bearer $API_KEY")
fi

AUTH_MODE="anonymous"
if [[ -n "$API_KEY" ]]; then
  AUTH_MODE="authenticated"
fi

CLIENT_HEADER_VALUE="here-now-publish-sh"
if [[ -n "$CLIENT" ]]; then
  normalized_client=$(echo "$CLIENT" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9._-' '-')
  normalized_client="${normalized_client#-}"
  normalized_client="${normalized_client%-}"
  if [[ -n "$normalized_client" ]]; then
    CLIENT_HEADER_VALUE="${normalized_client}/publish-sh"
  fi
fi
CLIENT_ARGS=(-H "x-herenow-client: $CLIENT_HEADER_VALUE")

# Step 1: Create/update publish
echo "creating publish ($file_count files)..." >&2
RESPONSE=$(curl -sS -X "$METHOD" "$URL" \
  "${AUTH_ARGS[@]+"${AUTH_ARGS[@]}"}" \
  "${CLIENT_ARGS[@]+"${CLIENT_ARGS[@]}"}" \
  -H "content-type: application/json" \
  -d "$BODY")

# Check for errors
if echo "$RESPONSE" | "$JQ_BIN" -e '.error' >/dev/null 2>&1; then
  err=$(echo "$RESPONSE" | "$JQ_BIN" -r '.error')
  details=$(echo "$RESPONSE" | "$JQ_BIN" -r '.details // empty')
  die "$err${details:+ ($details)}"
fi

OUT_SLUG=$(echo "$RESPONSE" | "$JQ_BIN" -r '.slug')
VERSION_ID=$(echo "$RESPONSE" | "$JQ_BIN" -r '.upload.versionId')
FINALIZE_URL=$(echo "$RESPONSE" | "$JQ_BIN" -r '.upload.finalizeUrl')
SITE_URL=$(echo "$RESPONSE" | "$JQ_BIN" -r '.siteUrl')
UPLOAD_COUNT=$(echo "$RESPONSE" | "$JQ_BIN" '.upload.uploads | length')
SKIPPED_COUNT=$(echo "$RESPONSE" | "$JQ_BIN" '.upload.skipped // [] | length')

[[ "$OUT_SLUG" != "null" ]] || die "unexpected response: $RESPONSE"

# Step 2: Upload files (skipped files are unchanged from previous version)
if [[ "$SKIPPED_COUNT" -gt 0 ]]; then
  echo "uploading $UPLOAD_COUNT files ($SKIPPED_COUNT unchanged, skipped)..." >&2
else
  echo "uploading $UPLOAD_COUNT files..." >&2
fi
upload_errors=0

for i in $(seq 0 $((UPLOAD_COUNT - 1))); do
  upload_path=$(echo "$RESPONSE" | "$JQ_BIN" -r ".upload.uploads[$i].path")
  upload_url=$(echo "$RESPONSE" | "$JQ_BIN" -r ".upload.uploads[$i].url")
  upload_ct=$(echo "$RESPONSE" | "$JQ_BIN" -r ".upload.uploads[$i].headers[\"Content-Type\"] // empty")

  if [[ -f "$TARGET" && ! -d "$TARGET" ]]; then
    local_file="$TARGET"
  else
    local_file=$(echo "$FILE_MAP" | "$JQ_BIN" -r --arg p "$upload_path" '.[$p]')
  fi

  if [[ ! -f "$local_file" ]]; then
    echo "warning: missing local file for $upload_path" >&2
    upload_errors=$((upload_errors + 1))
    continue
  fi

  ct_args=()
  [[ -n "$upload_ct" ]] && ct_args=(-H "Content-Type: $upload_ct")

  http_code=$(curl -sS -o /dev/null -w "%{http_code}" -X PUT "$upload_url" \
    "${ct_args[@]+"${ct_args[@]}"}" \
    --data-binary "@$local_file")

  if [[ "$http_code" -lt 200 || "$http_code" -ge 300 ]]; then
    echo "warning: upload failed for $upload_path (HTTP $http_code)" >&2
    upload_errors=$((upload_errors + 1))
  fi
done

[[ "$upload_errors" -eq 0 ]] || die "$upload_errors file(s) failed to upload"

# Step 3: Finalize
echo "finalizing..." >&2
FIN_RESPONSE=$(curl -sS -X POST "$FINALIZE_URL" \
  "${AUTH_ARGS[@]+"${AUTH_ARGS[@]}"}" \
  "${CLIENT_ARGS[@]+"${CLIENT_ARGS[@]}"}" \
  -H "content-type: application/json" \
  -d "{\"versionId\":\"$VERSION_ID\"}")

if echo "$FIN_RESPONSE" | "$JQ_BIN" -e '.error' >/dev/null 2>&1; then
  err=$(echo "$FIN_RESPONSE" | "$JQ_BIN" -r '.error')
  die "finalize failed: $err"
fi

# Save state
mkdir -p "$STATE_DIR"
if [[ -f "$STATE_FILE" ]]; then
  STATE=$(cat "$STATE_FILE")
else
  STATE='{"publishes":{}}'
fi

entry=$("$JQ_BIN" -n --arg s "$SITE_URL" '{siteUrl: $s}')

RESPONSE_CLAIM_TOKEN=$(echo "$RESPONSE" | "$JQ_BIN" -r '.claimToken // empty')
RESPONSE_CLAIM_URL=$(echo "$RESPONSE" | "$JQ_BIN" -r '.claimUrl // empty')
RESPONSE_EXPIRES=$(echo "$RESPONSE" | "$JQ_BIN" -r '.expiresAt // empty')

[[ -n "$RESPONSE_CLAIM_TOKEN" ]] && entry=$(echo "$entry" | "$JQ_BIN" --arg v "$RESPONSE_CLAIM_TOKEN" '.claimToken = $v')
[[ -n "$RESPONSE_CLAIM_URL" ]] && entry=$(echo "$entry" | "$JQ_BIN" --arg v "$RESPONSE_CLAIM_URL" '.claimUrl = $v')
[[ -n "$RESPONSE_EXPIRES" ]] && entry=$(echo "$entry" | "$JQ_BIN" --arg v "$RESPONSE_EXPIRES" '.expiresAt = $v')

STATE=$(echo "$STATE" | "$JQ_BIN" --arg slug "$OUT_SLUG" --argjson e "$entry" '.publishes[$slug] = $e')
echo "$STATE" | "$JQ_BIN" '.' > "$STATE_FILE"

# Output
echo "$SITE_URL"

PERSISTENCE="permanent"
if [[ "$AUTH_MODE" == "anonymous" ]]; then
  PERSISTENCE="expires_24h"
elif [[ -n "$RESPONSE_EXPIRES" ]]; then
  PERSISTENCE="expires_at"
fi

SAFE_CLAIM_URL=""
if [[ -n "$RESPONSE_CLAIM_URL" && "$RESPONSE_CLAIM_URL" == https://* ]]; then
  SAFE_CLAIM_URL="$RESPONSE_CLAIM_URL"
fi

ACTION="create"
if [[ -n "$SLUG" ]]; then
  ACTION="update"
fi

echo "" >&2
echo "publish_result.site_url=$SITE_URL" >&2
echo "publish_result.slug=$OUT_SLUG" >&2
echo "publish_result.action=$ACTION" >&2
echo "publish_result.auth_mode=$AUTH_MODE" >&2
echo "publish_result.api_key_source=$API_KEY_SOURCE" >&2
echo "publish_result.persistence=$PERSISTENCE" >&2
echo "publish_result.expires_at=$RESPONSE_EXPIRES" >&2
echo "publish_result.claim_url=$SAFE_CLAIM_URL" >&2

if [[ "$AUTH_MODE" == "authenticated" ]]; then
  echo "authenticated publish (permanent, saved to your account)" >&2
else
  echo "anonymous publish (expires in 24h)" >&2
  if [[ -n "$SAFE_CLAIM_URL" ]]; then
    echo "claim URL: $SAFE_CLAIM_URL" >&2
  fi
  if [[ -n "$RESPONSE_CLAIM_TOKEN" ]]; then
    echo "claim token saved to $STATE_FILE" >&2
  fi
fi
