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
WORKSPACE=""
CLAIM_TOKEN=""
TITLE=""
DESCRIPTION=""
TTL=""
CLIENT=""
TARGET=""
SPA_MODE=""
CROSS_ORIGIN_ISOLATION=""
FROM_DRIVE=""
DRIVE_VERSION=""
OVERWRITE=0
BASE_VERSION_ID=""
FOLDER=""

usage() {
  cat <<'USAGE'
Usage: publish.sh <file-or-dir> [options]

Options:
  --api-key <key>         API key (or set $HERENOW_API_KEY)
  --slug <slug>           Update existing publish
  --workspace <subdomain> Publish into a workspace (team account) you belong to
  --claim-token <token>   Claim token for anonymous updates
  --title <text>          Viewer title
  --description <text>    Viewer description
  --folder <name|id>      File the Site in a dashboard folder (a name is created if missing; authenticated only)
  --ttl <seconds>         Expiry (authenticated only)
  --client <name>         Agent name for attribution (e.g. cursor, claude-code)
  --overwrite             Skip the stale-base check when updating (see below)
  --spa                   Enable SPA routing
  --cross-origin-isolation
                         Send COOP/COEP headers (threaded WebAssembly needs SharedArrayBuffer)
  --from-drive <drv_...>  Publish a Drive snapshot instead of local files
  --version <dv_...>      Drive version for --from-drive (default: current head)
  --base-url <url>        API base (default: https://here.now)
  --allow-nonherenow-base-url
                         Allow auth requests to non-default API base URL
USAGE
  exit 1
}

die() { echo "error: $1" >&2; exit 1; }

# Prints an actionable version_conflict report and exits. $1 = response JSON.
die_version_conflict() {
  local resp="$1"
  local msg cur src at
  msg=$(echo "$resp" | "$JQ_BIN" -r '.message // .error')
  cur=$(echo "$resp" | "$JQ_BIN" -r '.details.currentVersionId // empty')
  src=$(echo "$resp" | "$JQ_BIN" -r '.details.currentVersionSource // empty')
  at=$(echo "$resp" | "$JQ_BIN" -r '.details.currentVersionCreatedAt // empty')
  echo "error: $msg" >&2
  [[ -n "$cur" ]] && echo "live version: $cur${src:+ (created by $src)}${at:+ at $at}" >&2
  echo "publish_result.conflict=version_conflict" >&2
  echo "The live Site changed since this directory last published it." >&2
  echo "Options: read the live files and reconcile your local copy first" >&2
  echo "  (GET ${BASE_URL}/api/v1/publish/${SLUG}/files lists them; GET .../files/{path} returns each," >&2
  echo "  with your API key, no visitor password needed), then republish;" >&2
  echo "or re-run with --overwrite to replace the live version anyway." >&2
  exit 1
}

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
    --workspace)    WORKSPACE="$2"; shift 2 ;;
    --claim-token)  CLAIM_TOKEN="$2"; shift 2 ;;
    --title)        TITLE="$2"; shift 2 ;;
    --description)  DESCRIPTION="$2"; shift 2 ;;
    --folder)       FOLDER="$2"; shift 2 ;;
    --ttl)          TTL="$2"; shift 2 ;;
    --client)       CLIENT="$2"; shift 2 ;;
    --base-url)     BASE_URL="$2"; shift 2 ;;
    --allow-nonherenow-base-url) ALLOW_NON_HERENOW_BASE_URL=1; shift ;;
    --overwrite)    OVERWRITE=1; shift ;;
    --spa)          SPA_MODE="true"; shift ;;
    --cross-origin-isolation) CROSS_ORIGIN_ISOLATION="true"; shift ;;
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

# Workspace publishing requires an account API key and is not supported for
# --from-drive in this script (Drives are personal; use the API directly).
if [[ -n "$WORKSPACE" ]]; then
  [[ -n "$API_KEY" ]] || die "--workspace requires an account API key"
  [[ -z "$FROM_DRIVE" ]] || die "--workspace cannot be combined with --from-drive"
fi

# Folders belong to an account (docs: https://here.now/docs#folders), so an
# anonymous Site has nowhere to be filed.
if [[ -n "$FOLDER" ]]; then
  [[ -n "$API_KEY" ]] || die "--folder requires an account API key (anonymous Sites cannot be filed in folders)"
fi

# Safety guard: avoid accidentally sending bearer auth to arbitrary endpoints.
# https://api.here.now is the same API reached through Cloudflare, for
# networks that cannot connect to https://here.now directly; it is the only
# other host the key may go to without the override.
case "$BASE_URL" in
  https://here.now|https://api.here.now) ;;
  *)
    if [[ -n "$API_KEY" && "$ALLOW_NON_HERENOW_BASE_URL" -ne 1 ]]; then
      die "refusing to send API key to non-default base URL; pass --allow-nonherenow-base-url to override"
    fi
    ;;
esac

# Auto-load claim token from state file for slug updates (server uses it only for
# anonymous sites; harmless when an API key is also present).
if [[ -n "$SLUG" && -z "$CLAIM_TOKEN" && -f "$STATE_FILE" ]]; then
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
  [[ "$SPA_MODE" == "true" ]] && BODY=$(echo "$BODY" | "$JQ_BIN" '.spaMode = true')
  [[ "$CROSS_ORIGIN_ISOLATION" == "true" ]] && BODY=$(echo "$BODY" | "$JQ_BIN" '.crossOriginIsolation = true')
  [[ -n "$FOLDER" ]] && BODY=$(echo "$BODY" | "$JQ_BIN" --arg f "$FOLDER" '.folder = $f')
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
  FOLDER_OUT=$(echo "$RESPONSE" | "$JQ_BIN" -r '.folder.name // empty')
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
  echo "publish_result.folder=$FOLDER_OUT" >&2
  exit 0
fi

# Absolute source path: scopes the saved base version to slug + source path,
# so publishing a different directory to the same slug never falsely claims
# its files are current.
if [[ -f "$TARGET" ]]; then
  TARGET_ABS="$(cd "$(dirname "$TARGET")" && pwd)/$(basename "$TARGET")"
else
  TARGET_ABS="$(cd "$TARGET" && pwd)"
fi

# Optimistic concurrency (https://here.now/docs#update): when updating a slug
# this state file has published before from this same source path, declare
# that version as the base. The server rejects the update (version_conflict)
# if the live site moved past it — e.g. it was edited from another tool.
# --overwrite skips the check (an unchecked full replacement).
if [[ -n "$SLUG" && "$OVERWRITE" -ne 1 && -f "$STATE_FILE" ]]; then
  stored_version=$("$JQ_BIN" -r --arg s "$SLUG" '.publishes[$s].versionId // empty' "$STATE_FILE" 2>/dev/null || true)
  stored_path=$("$JQ_BIN" -r --arg s "$SLUG" '.publishes[$s].path // empty' "$STATE_FILE" 2>/dev/null || true)
  if [[ -n "$stored_version" && -n "$stored_path" && "$stored_path" == "$TARGET_ABS" ]]; then
    BASE_VERSION_ID="$stored_version"
  fi
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
    # Local publish state (slug/claim token cache) — never site content.
    [[ "$rel" == ".herenow/state.json" ]] && continue
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

if [[ -n "$CLAIM_TOKEN" && -n "$SLUG" ]]; then
  BODY=$(echo "$BODY" | "$JQ_BIN" --arg ct "$CLAIM_TOKEN" '.claimToken = $ct')
fi

if [[ "$SPA_MODE" == "true" ]]; then
  BODY=$(echo "$BODY" | "$JQ_BIN" '.spaMode = true')
fi

if [[ "$CROSS_ORIGIN_ISOLATION" == "true" ]]; then
  BODY=$(echo "$BODY" | "$JQ_BIN" '.crossOriginIsolation = true')
fi

if [[ -n "$FOLDER" ]]; then
  BODY=$(echo "$BODY" | "$JQ_BIN" --arg f "$FOLDER" '.folder = $f')
fi

if [[ -n "$BASE_VERSION_ID" ]]; then
  BODY=$(echo "$BODY" | "$JQ_BIN" --arg v "$BASE_VERSION_ID" '.baseVersionId = $v')
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

# Workspace account selector: sent on create/update and finalize so the Site
# is owned by the workspace (see https://here.now/docs#workspaces).
ACCOUNT_ARGS=()
if [[ -n "$WORKSPACE" ]]; then
  ACCOUNT_ARGS=(-H "x-herenow-account: $WORKSPACE")
fi

# Step 1: Create/update publish
echo "creating publish ($file_count files)..." >&2
RESPONSE=$(curl -sS -X "$METHOD" "$URL" \
  "${AUTH_ARGS[@]+"${AUTH_ARGS[@]}"}" \
  "${CLIENT_ARGS[@]+"${CLIENT_ARGS[@]}"}" \
  "${ACCOUNT_ARGS[@]+"${ACCOUNT_ARGS[@]}"}" \
  -H "content-type: application/json" \
  -d "$BODY")

# Check for errors
if echo "$RESPONSE" | "$JQ_BIN" -e '.error' >/dev/null 2>&1; then
  if [[ "$(echo "$RESPONSE" | "$JQ_BIN" -r '.code // empty')" == "version_conflict" ]]; then
    die_version_conflict "$RESPONSE"
  fi
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
upload_network_failures=0

# C-style loop: BSD seq counts DOWN for `seq 0 -1`, so a zero-upload
# republish (all files unchanged) used to iterate twice with null paths
# and die between create and finalize, stranding the site in pending.
for ((i = 0; i < UPLOAD_COUNT; i++)); do
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

  # `|| http_code=000`: a connection-level failure (proxy refusing CONNECT,
  # DNS, TLS) exits curl non-zero, which under set -e would kill the script
  # here with only curl's own message. Fall through so the count and the
  # hint below are reached.
  http_code=$(curl -sS -o /dev/null -w "%{http_code}" -X PUT "$upload_url" \
    "${ct_args[@]+"${ct_args[@]}"}" \
    --data-binary "@$local_file") || http_code="000"

  if [[ "$http_code" -lt 200 || "$http_code" -ge 300 ]]; then
    echo "warning: upload failed for $upload_path (HTTP $http_code)" >&2
    upload_errors=$((upload_errors + 1))
    upload_network_failures=$((upload_network_failures + 1))
  fi
done

if [[ "$upload_errors" -gt 0 ]]; then
  # Every PUT failed while the create call succeeded: the usual cause is an
  # egress allowlist that permits here.now but not the storage host.
  if [[ "$UPLOAD_COUNT" -gt 0 && "$upload_network_failures" -eq "$UPLOAD_COUNT" ]]; then
    echo "hint: every upload failed. Upload URLs PUT directly to *.r2.cloudflarestorage.com, not to here.now;" >&2
    echo "      if this environment restricts outbound network access, allow that host as well as here.now." >&2
  fi
  die "$upload_errors file(s) failed to upload"
fi

# Step 3: Finalize. The server makes this call idempotent: while the first
# attempt is still running, a repeat answers 409 finalize_in_flight with
# retryAfter; once it has completed, a repeat with the same versionId
# answers the live result (replayed:true). So when no real answer arrives
# (a proxy's gateway timeout, e.g. a 524 with an HTML body after about two
# minutes behind Cloudflare, which a large Site's copy-forward can exceed;
# or a dropped connection), or the server asks for a retry, repeat the same
# call until it answers or the wait budget ends. Running out of budget does
# not mean the finalize failed: it keeps running on the server.
FINALIZE_WAIT_SECONDS="${HERENOW_FINALIZE_WAIT_SECONDS:-900}"
finalize_deadline=$((SECONDS + FINALIZE_WAIT_SECONDS))
echo "finalizing..." >&2
while :; do
  # `|| ...000`: a connection-level failure exits curl non-zero, which under
  # set -e would end the script here; treat it like a missing answer.
  fin_out=$(curl -sS -w $'\n%{http_code}' -X POST "$FINALIZE_URL" \
    "${AUTH_ARGS[@]+"${AUTH_ARGS[@]}"}" \
    "${CLIENT_ARGS[@]+"${CLIENT_ARGS[@]}"}" \
    "${ACCOUNT_ARGS[@]+"${ACCOUNT_ARGS[@]}"}" \
    -H "content-type: application/json" \
    -d "{\"versionId\":\"$VERSION_ID\"}") || fin_out=$'\n000'
  fin_http_code="${fin_out##*$'\n'}"
  FIN_RESPONSE="${fin_out%$'\n'*}"

  fin_is_json=0
  echo "$FIN_RESPONSE" | "$JQ_BIN" -e 'type == "object"' >/dev/null 2>&1 && fin_is_json=1
  fin_code=""
  fin_retry_after=""
  if [[ "$fin_is_json" -eq 1 ]]; then
    fin_code=$(echo "$FIN_RESPONSE" | "$JQ_BIN" -r '.code // empty')
    fin_retry_after=$(echo "$FIN_RESPONSE" | "$JQ_BIN" -r '.retry_after // empty')
  fi

  if [[ "$fin_is_json" -eq 1 && "$fin_http_code" == 2* ]]; then
    break
  elif [[ "$fin_code" == "finalize_in_flight" ]]; then
    fin_reason="still running on the server"
  elif [[ "$fin_is_json" -eq 1 && "$fin_http_code" == 5* && -n "$fin_retry_after" ]]; then
    fin_reason="server asked for a retry (HTTP $fin_http_code)"
  elif [[ "$fin_is_json" -eq 0 && ( "$fin_http_code" == 5* || "$fin_http_code" == "000" ) ]]; then
    fin_reason="no answer (HTTP $fin_http_code: a proxy timeout or dropped connection; the finalize keeps running on the server)"
  else
    break
  fi

  [[ "$fin_retry_after" =~ ^[0-9]+$ ]] || fin_retry_after=10
  if (( SECONDS + fin_retry_after > finalize_deadline )); then
    echo "error: finalize did not answer within ${FINALIZE_WAIT_SECONDS}s ($fin_reason)." >&2
    echo "It may still complete on the server. To check, repeat the same finalize call:" >&2
    echo "  curl -sS -X POST $FINALIZE_URL -H 'content-type: application/json' -d '{\"versionId\":\"$VERSION_ID\"}'" >&2
    echo "  (with your Authorization header for an account Site). A JSON answer with currentVersionId means it is live." >&2
    echo "Set HERENOW_FINALIZE_WAIT_SECONDS to wait longer next time." >&2
    echo "publish_result.finalize=timeout" >&2
    exit 1
  fi
  echo "finalize: $fin_reason; retrying in ${fin_retry_after}s..." >&2
  sleep "$fin_retry_after"
done

if [[ "$fin_is_json" -eq 0 ]]; then
  die "finalize failed: unexpected response (HTTP $fin_http_code): $(echo "$FIN_RESPONSE" | tr -d '\r' | head -c 200)"
fi

if echo "$FIN_RESPONSE" | "$JQ_BIN" -e '.error' >/dev/null 2>&1; then
  if [[ "$(echo "$FIN_RESPONSE" | "$JQ_BIN" -r '.code // empty')" == "version_conflict" ]]; then
    die_version_conflict "$FIN_RESPONSE"
  fi
  err=$(echo "$FIN_RESPONSE" | "$JQ_BIN" -r '.error')
  die "finalize failed: $err"
fi

# Non-fatal finalize warnings (an invalid .herenow/proxy.json, a .wasm
# module published without crossOriginIsolation, ...). One line each, in
# the publish_result.* channel agents already read; the Site is live.
while IFS= read -r warning_line; do
  [[ -n "$warning_line" ]] && echo "publish_result.warning=$warning_line" >&2
done < <(echo "$FIN_RESPONSE" | "$JQ_BIN" -r '(.warnings // [])[]? | tostring' 2>/dev/null || true)

# Save state. Merge into the existing entry (never replace it wholesale) so
# a previously saved claimToken survives authenticated republishes.
mkdir -p "$STATE_DIR"
if [[ -f "$STATE_FILE" ]]; then
  STATE=$(cat "$STATE_FILE")
else
  STATE='{"publishes":{}}'
fi

entry=$(echo "$STATE" | "$JQ_BIN" --arg s "$OUT_SLUG" '.publishes[$s] // {}')
entry=$(echo "$entry" | "$JQ_BIN" --arg v "$SITE_URL" '.siteUrl = $v')

# The live version after this publish comes from the finalize response's
# currentVersionId — a byte-identical republish keeps the previous live
# version (unchanged:true), so the staged upload versionId must never be
# saved as the base.
LIVE_VERSION_ID=$(echo "$FIN_RESPONSE" | "$JQ_BIN" -r '.currentVersionId // empty')
[[ -n "$LIVE_VERSION_ID" ]] && entry=$(echo "$entry" | "$JQ_BIN" --arg v "$LIVE_VERSION_ID" '.versionId = $v')
entry=$(echo "$entry" | "$JQ_BIN" --arg v "$TARGET_ABS" '.path = $v')

RESPONSE_CLAIM_TOKEN=$(echo "$RESPONSE" | "$JQ_BIN" -r '.claimToken // empty')
RESPONSE_CLAIM_URL=$(echo "$RESPONSE" | "$JQ_BIN" -r '.claimUrl // empty')
RESPONSE_EXPIRES=$(echo "$RESPONSE" | "$JQ_BIN" -r '.expiresAt // empty')

# A token passed with --claim-token (a Site the user created at here.new
# and handed over, or one from another directory) is saved too, so the next
# update of the same slug from this directory needs no flag.
[[ -n "$CLAIM_TOKEN" ]] && entry=$(echo "$entry" | "$JQ_BIN" --arg v "$CLAIM_TOKEN" '.claimToken = $v')
[[ -n "$RESPONSE_CLAIM_TOKEN" ]] && entry=$(echo "$entry" | "$JQ_BIN" --arg v "$RESPONSE_CLAIM_TOKEN" '.claimToken = $v')
[[ -n "$RESPONSE_CLAIM_URL" ]] && entry=$(echo "$entry" | "$JQ_BIN" --arg v "$RESPONSE_CLAIM_URL" '.claimUrl = $v')
[[ -n "$RESPONSE_EXPIRES" ]] && entry=$(echo "$entry" | "$JQ_BIN" --arg v "$RESPONSE_EXPIRES" '.expiresAt = $v')

STATE=$(echo "$STATE" | "$JQ_BIN" --arg slug "$OUT_SLUG" --argjson e "$entry" '.publishes[$slug] = $e')
echo "$STATE" | "$JQ_BIN" '.' > "$STATE_FILE"

# Workspace label URL (finalize response preferred; create response fallback)
ACCOUNT_URL=$(echo "$FIN_RESPONSE" | "$JQ_BIN" -r '.accountUrl // empty')
if [[ -z "$ACCOUNT_URL" ]]; then
  ACCOUNT_URL=$(echo "$RESPONSE" | "$JQ_BIN" -r '.accountUrl // empty')
fi

# Preferred address (finalize response preferred; create response fallback).
# Present for Sites served on a partner harness's domain; empty otherwise.
PRIMARY_URL=$(echo "$FIN_RESPONSE" | "$JQ_BIN" -r '.primaryUrl // empty')
if [[ -z "$PRIMARY_URL" ]]; then
  PRIMARY_URL=$(echo "$RESPONSE" | "$JQ_BIN" -r '.primaryUrl // empty')
fi

# The Site's dashboard folder after this publish (create/update response);
# empty at the root.
FOLDER_OUT=$(echo "$RESPONSE" | "$JQ_BIN" -r '.folder.name // empty')

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
echo "publish_result.account_url=$ACCOUNT_URL" >&2
echo "publish_result.primary_url=$PRIMARY_URL" >&2
echo "publish_result.live_version_id=$LIVE_VERSION_ID" >&2
echo "publish_result.folder=$FOLDER_OUT" >&2

if [[ "$AUTH_MODE" == "authenticated" ]]; then
  echo "authenticated publish (permanent, saved to your account)" >&2
  if [[ -n "$ACCOUNT_URL" ]]; then
    echo "workspace URL: $ACCOUNT_URL" >&2
  fi
else
  echo "anonymous publish (expires in 24h)" >&2
  if [[ -n "$SAFE_CLAIM_URL" ]]; then
    echo "claim URL: $SAFE_CLAIM_URL" >&2
  fi
  if [[ -n "$RESPONSE_CLAIM_TOKEN" ]]; then
    echo "claim token saved to $STATE_FILE" >&2
  fi
fi
