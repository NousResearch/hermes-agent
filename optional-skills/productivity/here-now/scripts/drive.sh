#!/usr/bin/env bash
set -euo pipefail

BASE_URL="https://here.now"
CREDENTIALS_FILE="$HOME/.herenow/credentials"
API_KEY="${HERENOW_API_KEY:-}"
DRIVE_TOKEN="${HERENOW_DRIVE_TOKEN:-}"
ALLOW_NON_HERENOW_BASE_URL=0
MAX_FILE_BYTES=$((500 * 1024 * 1024))

usage() {
  cat <<'USAGE'
Usage: drive.sh [global options] <command> [args]

Global options:
  --api-key <key>        Account API key (or $HERENOW_API_KEY / ~/.herenow/credentials)
  --token <drv_live_...> Drive token (or $HERENOW_DRIVE_TOKEN)
  --base-url <url>       API base (default: https://here.now)
  --allow-nonherenow-base-url

Commands:
  create [name] [--default]
  default
  ls
  ls <drive> [prefix]
  cat <drive> <path>
  put <drive> <path> --from <local-file>
  import <drive> <prefix> --from <local-folder> [--dry-run]
  export <drive> <prefix> --to <local-folder> [--dry-run]
  rm <drive> <path> [--recursive --confirm <path>]
  share <drive> --perms read|write [--prefix notes/] [--ttl 30d] [--label text] [--manage-tokens]
  tokens <drive>
  revoke <drive> <tokenId>
  delete <drive> --confirm "<drive name>"
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
    --api-key) API_KEY="$2"; shift 2 ;;
    --token) DRIVE_TOKEN="$2"; shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --allow-nonherenow-base-url) ALLOW_NON_HERENOW_BASE_URL=1; shift ;;
    --help|-h) usage ;;
    --*) die "unknown global option: $1" ;;
    *) break ;;
  esac
done

CMD="${1:-}"
[[ -n "$CMD" ]] || usage
shift || true

if [[ -z "$API_KEY" && -z "$DRIVE_TOKEN" && -f "$CREDENTIALS_FILE" ]]; then
  API_KEY=$(tr -d '[:space:]' < "$CREDENTIALS_FILE")
fi

BASE_URL="${BASE_URL%/}"
if [[ "$BASE_URL" != "https://here.now" && "$ALLOW_NON_HERENOW_BASE_URL" -ne 1 ]]; then
  if [[ -n "$API_KEY" || -n "$DRIVE_TOKEN" ]]; then
    die "refusing to send credentials to non-default base URL; pass --allow-nonherenow-base-url to override"
  fi
fi

auth_header=()
if [[ -n "$DRIVE_TOKEN" ]]; then
  auth_header=(-H "authorization: Bearer $DRIVE_TOKEN")
elif [[ -n "$API_KEY" ]]; then
  auth_header=(-H "authorization: Bearer $API_KEY")
else
  die "missing credentials; set HERENOW_API_KEY, HERENOW_DRIVE_TOKEN, or ~/.herenow/credentials"
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
    css) echo "text/css; charset=utf-8" ;;
    js|mjs) echo "text/javascript; charset=utf-8" ;;
    json) echo "application/json; charset=utf-8" ;;
    md|txt) echo "text/plain; charset=utf-8" ;;
    svg) echo "image/svg+xml" ;;
    png) echo "image/png" ;;
    jpg|jpeg) echo "image/jpeg" ;;
    gif) echo "image/gif" ;;
    webp) echo "image/webp" ;;
    pdf) echo "application/pdf" ;;
    *) file --brief --mime-type "$f" 2>/dev/null || echo "application/octet-stream" ;;
  esac
}

api_json() {
  local method="$1"; shift
  local url="$1"; shift
  local body="${1:-}"
  local tmp
  tmp=$(mktemp)
  local code
  if [[ -n "$body" ]]; then
    code=$(curl -sS -o "$tmp" -w "%{http_code}" -X "$method" "$url" "${auth_header[@]}" -H "content-type: application/json" -d "$body")
  else
    code=$(curl -sS -o "$tmp" -w "%{http_code}" -X "$method" "$url" "${auth_header[@]}")
  fi
  if [[ "$code" -lt 200 || "$code" -ge 300 ]]; then
    local err
    err=$("$JQ_BIN" -r '.error // empty' "$tmp" 2>/dev/null || true)
    [[ -n "$err" ]] || err="$(cat "$tmp")"
    rm -f "$tmp"
    die "HTTP $code: $err"
  fi
  cat "$tmp"
  rm -f "$tmp"
}

urlenc() {
  "$JQ_BIN" -nr --arg v "$1" '$v|@uri'
}

urlenc_path() {
  local path="$1"
  local out=""
  local part
  IFS='/' read -r -a parts <<< "$path"
  for part in "${parts[@]}"; do
    [[ -n "$out" ]] && out="$out/"
    out="$out$(urlenc "$part")"
  done
  echo "$out"
}

resolve_drive() {
  local name="$1"
  if [[ "$name" == drv_* ]]; then
    echo "$name"
    return
  fi
  if [[ -n "$DRIVE_TOKEN" ]]; then
    die "drive tokens must reference drives by drv_ id; use account credentials to resolve drive names"
  fi
  if [[ "$name" == "default" || "$name" == "my-drive" || "$name" == "My Drive" ]]; then
    api_json GET "$BASE_URL/api/v1/drives/default" | "$JQ_BIN" -r '.drive.id'
    return
  fi
  local rows count
  rows=$(api_json GET "$BASE_URL/api/v1/drives" | "$JQ_BIN" --arg n "$name" '[.drives[] | select(.name == $n)]')
  count=$(echo "$rows" | "$JQ_BIN" 'length')
  [[ "$count" -eq 1 ]] || die "drive name '$name' matched $count drives; use a drv_ id"
  echo "$rows" | "$JQ_BIN" -r '.[0].id'
}

drive_head() {
  local id="$1"
  api_json GET "$BASE_URL/api/v1/drives/$id" | "$JQ_BIN" -r '.drive.headVersionId // .headVersionId // empty'
}

file_meta() {
  local id="$1"
  local path="$2"
  local prefix
  prefix=$(urlenc "$path")
  api_json GET "$BASE_URL/api/v1/drives/$id/files?prefix=$prefix&limit=200" | "$JQ_BIN" -c --arg p "$path" '.files[]? | select(.path == $p)' | head -n 1
}

put_file() {
  local drive="$1"; shift
  local path="$1"; shift
  local local_file=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --from) local_file="$2"; shift 2 ;;
      *) die "unexpected put argument: $1" ;;
    esac
  done
  [[ -f "$local_file" ]] || die "--from must be a file"
  local id sz ct sha meta body upload upload_url upload_id http_code
  id=$(resolve_drive "$drive")
  sz=$(wc -c < "$local_file" | tr -d ' ')
  [[ "$sz" -le "$MAX_FILE_BYTES" ]] || die "$path exceeds the $MAX_FILE_BYTES byte Drive file limit"
  ct=$(guess_content_type "$local_file")
  sha=$(compute_sha256 "$local_file")
  meta=$(file_meta "$id" "$path" || true)
  body=$("$JQ_BIN" -n --arg p "$path" --argjson s "$sz" --arg c "$ct" --arg sha "$sha" \
    '{path:$p,size:$s,contentType:$c,sha256:$sha}')
  if [[ -n "$meta" ]]; then
    etag=$(echo "$meta" | "$JQ_BIN" -r '.etag')
    body=$(echo "$body" | "$JQ_BIN" --arg e "$etag" '.ifMatch = $e')
  else
    body=$(echo "$body" | "$JQ_BIN" '.ifNoneMatch = "*"')
  fi
  upload=$(api_json POST "$BASE_URL/api/v1/drives/$id/files/uploads" "$body")
  upload_url=$(echo "$upload" | "$JQ_BIN" -r '.uploadUrl')
  upload_id=$(echo "$upload" | "$JQ_BIN" -r '.uploadId')
  http_code=$(curl -sS -o /dev/null -w "%{http_code}" -X PUT "$upload_url" -H "Content-Type: $ct" --data-binary "@$local_file")
  [[ "$http_code" -ge 200 && "$http_code" -lt 300 ]] || die "upload failed for $path (HTTP $http_code)"
  api_json POST "$BASE_URL/api/v1/drives/$id/files/finalize" "$("$JQ_BIN" -n --arg u "$upload_id" '{uploadId:$u}')" | "$JQ_BIN" .
}

case "$CMD" in
  create)
    name=""
    is_default="false"
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --default) is_default="true"; shift ;;
        *) [[ -z "$name" ]] && name="$1" || die "unexpected argument: $1"; shift ;;
      esac
    done
    body=$("$JQ_BIN" -n --arg n "$name" --argjson d "$is_default" '{isDefault:$d} + (if $n == "" then {} else {name:$n} end)')
    api_json POST "$BASE_URL/api/v1/drives" "$body" | "$JQ_BIN" .
    ;;
  default)
    api_json GET "$BASE_URL/api/v1/drives/default" | "$JQ_BIN" .
    ;;
  ls)
    if [[ $# -eq 0 ]]; then
      [[ -z "$DRIVE_TOKEN" ]] || die "drive tokens cannot list drives; pass a drv_ id"
      api_json GET "$BASE_URL/api/v1/drives" | "$JQ_BIN" .
    else
      id=$(resolve_drive "$1")
      prefix="${2:-}"
      api_json GET "$BASE_URL/api/v1/drives/$id/files?prefix=$(urlenc "$prefix")" | "$JQ_BIN" .
    fi
    ;;
  cat)
    [[ $# -eq 2 ]] || die "usage: drive.sh cat <drive> <path>"
    id=$(resolve_drive "$1")
    curl -fsS "$BASE_URL/api/v1/drives/$id/files/$(urlenc_path "$2")" "${auth_header[@]}"
    ;;
  put)
    [[ $# -ge 2 ]] || die "usage: drive.sh put <drive> <path> --from <local-file>"
    put_file "$@"
    ;;
  import)
    [[ $# -ge 2 ]] || die "usage: drive.sh import <drive> <prefix> --from <local-folder> [--dry-run]"
    drive="$1"; prefix="${2%/}"; shift 2
    from=""; dry=0
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --from) from="$2"; shift 2 ;;
        --dry-run) dry=1; shift ;;
        *) die "unexpected import argument: $1" ;;
      esac
    done
    [[ -d "$from" ]] || die "--from must be a folder"
    uploaded=0
    skipped=0
    failed=0
    planned=0
    while IFS= read -r -d '' f; do
      rel="${f#$from/}"
      [[ "$rel" == .git/* || "$rel" == node_modules/* || "$rel" == ".DS_Store" || "$rel" == */.DS_Store ]] && continue
      planned=$((planned + 1))
      sz=$(wc -c < "$f" | tr -d ' ')
      if [[ "$sz" -gt "$MAX_FILE_BYTES" ]]; then
        echo "skip oversized $f ($sz bytes > $MAX_FILE_BYTES)" >&2
        skipped=$((skipped + 1))
        continue
      fi
      dest="$rel"
      [[ -n "$prefix" ]] && dest="$prefix/$rel"
      if [[ "$dry" -eq 1 ]]; then
        echo "upload $f -> $dest"
        skipped=$((skipped + 1))
      else
        if (put_file "$drive" "$dest" --from "$f" >/dev/null); then
          uploaded=$((uploaded + 1))
        else
          failed=$((failed + 1))
        fi
      fi
    done < <(find "$from" -type f -print0 | sort -z)
    echo "planned=$planned uploaded=$uploaded skipped=$skipped failed=$failed"
    [[ "$failed" -eq 0 ]] || exit 1
    ;;
  export)
    [[ $# -ge 2 ]] || die "usage: drive.sh export <drive> <prefix> --to <local-folder> [--dry-run]"
    id=$(resolve_drive "$1"); prefix="${2%/}"; shift 2
    to=""; dry=0
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --to) to="$2"; shift 2 ;;
        --dry-run) dry=1; shift ;;
        *) die "unexpected export argument: $1" ;;
      esac
    done
    [[ -n "$to" ]] || die "--to is required"
    cursor=""
    total=0
    while true; do
      url="$BASE_URL/api/v1/drives/$id/files?prefix=$(urlenc "$prefix")&limit=200"
      [[ -n "$cursor" ]] && url="$url&cursor=$(urlenc "$cursor")"
      files=$(api_json GET "$url")
      while IFS= read -r p; do
        [[ -n "$p" ]] || continue
        rel="$p"
        [[ -n "$prefix" ]] && rel="${p#$prefix/}"
        out="$to/$rel"
        if [[ "$dry" -eq 1 ]]; then
          echo "download $p -> $out"
        else
          mkdir -p "$(dirname "$out")"
          curl -fsS "$BASE_URL/api/v1/drives/$id/files/$(urlenc_path "$p")" "${auth_header[@]}" -o "$out"
        fi
        total=$((total + 1))
      done < <(echo "$files" | "$JQ_BIN" -r '.files[].path')
      cursor=$(echo "$files" | "$JQ_BIN" -r '.nextCursor // empty')
      [[ -n "$cursor" ]] || break
    done
    echo "files=$total"
    ;;
  rm)
    [[ $# -ge 2 ]] || die "usage: drive.sh rm <drive> <path> [--recursive --confirm <path>]"
    id=$(resolve_drive "$1"); path="$2"; shift 2
    recursive=0; confirm=""
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --recursive) recursive=1; shift ;;
        --confirm) confirm="$2"; shift 2 ;;
        *) die "unexpected rm argument: $1" ;;
      esac
    done
    if [[ "$recursive" -eq 1 ]]; then
      [[ "$confirm" == "$path" ]] || die "recursive delete requires --confirm '$path'"
      head=$(drive_head "$id")
      api_json DELETE "$BASE_URL/api/v1/drives/$id/files/$(urlenc_path "$path")?recursive=true&baseVersionId=$(urlenc "$head")" | "$JQ_BIN" .
    else
      meta=$(file_meta "$id" "$path")
      etag=$(echo "$meta" | "$JQ_BIN" -r '.etag')
      curl -fsS -X DELETE "$BASE_URL/api/v1/drives/$id/files/$(urlenc_path "$path")" "${auth_header[@]}" -H "If-Match: $etag" | "$JQ_BIN" .
    fi
    ;;
  share)
    [[ $# -ge 1 ]] || die "usage: drive.sh share <drive> --perms read|write [--prefix notes/] [--ttl 30d] [--label text] [--manage-tokens]"
    id=$(resolve_drive "$1"); shift
    perms="write"; prefix=""; ttl=""; label=""; manage_tokens="false"
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --perms) perms="$2"; shift 2 ;;
        --prefix) prefix="$2"; shift 2 ;;
        --ttl) ttl="$2"; shift 2 ;;
        --label) label="$2"; shift 2 ;;
        --manage-tokens) manage_tokens="true"; shift ;;
        *) die "unexpected share argument: $1" ;;
      esac
    done
    body=$("$JQ_BIN" -n --arg p "$perms" --arg pp "$prefix" --arg ttl "$ttl" --arg label "$label" --argjson mt "$manage_tokens" \
      '{perms:$p} + (if $mt then {manageTokens:true} else {} end) + (if $ttl == "" then {} else {ttl:$ttl} end) + (if $pp == "" then {} else {pathPrefix:$pp} end) + (if $label == "" then {} else {label:$label} end)')
    api_json POST "$BASE_URL/api/v1/drives/$id/tokens" "$body" | "$JQ_BIN" -r '.shareBlock'
    ;;
  tokens)
    [[ $# -eq 1 ]] || die "usage: drive.sh tokens <drive>"
    id=$(resolve_drive "$1")
    api_json GET "$BASE_URL/api/v1/drives/$id/tokens" | "$JQ_BIN" .
    ;;
  revoke)
    [[ $# -eq 2 ]] || die "usage: drive.sh revoke <drive> <tokenId>"
    id=$(resolve_drive "$1")
    api_json DELETE "$BASE_URL/api/v1/drives/$id/tokens/$2" | "$JQ_BIN" .
    ;;
  delete)
    [[ $# -ge 1 ]] || die "usage: drive.sh delete <drive> --confirm <drive name>"
    id=$(resolve_drive "$1"); shift
    confirm=""
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --confirm) confirm="$2"; shift 2 ;;
        *) die "unexpected delete argument: $1" ;;
      esac
    done
    drive=$(api_json GET "$BASE_URL/api/v1/drives/$id")
    name=$(echo "$drive" | "$JQ_BIN" -r '.drive.name')
    [[ "$confirm" == "$name" ]] || die "delete requires --confirm '$name'"
    api_json DELETE "$BASE_URL/api/v1/drives/$id" | "$JQ_BIN" .
    ;;
  *)
    die "unknown command: $CMD"
    ;;
esac
