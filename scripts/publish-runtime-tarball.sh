#!/bin/bash
# ============================================================================
# publish-runtime-tarball.sh — ApexNodes China install mirror publisher
# ============================================================================
# Publishes the artifacts that the packaged ApexNodes desktop downloads from our
# public-read COS bucket when installing on a mainland-China machine (where
# github.com / pypi.org / registry.npmjs.org are blocked). See install.sh's
# "ApexNodes China mirror mode" block (HERMES_CN_MIRRORS) and
# apps/desktop/electron/bootstrap-runner.cjs.
#
# It produces (and, with --upload, pushes to COS):
#   1. hermes-agent-<full-sha>.tar.gz   — `git archive` of the pinned runtime
#                                          commit (matches the desktop build
#                                          stamp; clean source tree, no .git).
#   2. uv-<triple>.tar.gz               — astral uv binaries, fetched from
#                                          GitHub releases and re-hosted so CN
#                                          installs never touch github.com.
#                                          (npmmirror does NOT mirror uv.)
#
# This script runs on a release machine WITH GitHub access (it fetches uv from
# GitHub). The COS bucket must be configured public-read so install.sh can pull
# over plain HTTPS with no credentials. Default commit matches
# apps/desktop/scripts/write-build-stamp.cjs (fromApexNodesPin).
#
# ── Upload prerequisites (--upload) ─────────────────────────────────────────
#   coscli (https://cloud.tencent.com/document/product/436/63143), configured
#   with the COS credentials from the hermes-cloud master env, e.g.:
#     coscli config init   # or ~/.cos.yaml with secretId/secretKey/region
#   The bucket should already exist and have a public-read access policy.
#
# ── After publishing ────────────────────────────────────────────────────────
#   Point the desktop at the bucket by baking the base URL into
#   apps/desktop/electron/main.cjs (RUNTIME_COS_BASE) or setting it at pack time:
#     HERMES_RUNTIME_COS_BASE=https://<bucket>.cos.<region>.myqcloud.com/<prefix> \
#       npm run dist:mac:dmg
#   install.sh then fetches  <base>/hermes-agent-<sha>.tar.gz  and
#   <base>/uv-<triple>.tar.gz.
#
# Usage:
#   scripts/publish-runtime-tarball.sh [options]
#     --commit <sha|ref>   Runtime commit to archive (default: the pinned SHA)
#     --out-dir <dir>      Where to write artifacts (default: ./dist/cos-runtime)
#     --bucket <name>      COS bucket name (or set COS_BUCKET); required for --upload
#     --region <region>    COS region, e.g. ap-shanghai (or set COS_REGION)
#     --prefix <path>      Key prefix inside the bucket (default: runtime)
#     --uv-version <ver>   Pin a uv release (default: latest)
#     --targets "<a b..>"  uv target triples (default: the 4 mac/linux triples)
#     --no-uv              Skip the uv binaries (publish only the source tarball)
#     --upload             Actually upload to COS (default: build artifacts only)
#     -h, --help           Show this help
# ============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Keep in sync with apps/desktop/scripts/write-build-stamp.cjs (fromApexNodesPin).
DEFAULT_COMMIT="87740e8021390455962caa3ad2c16d522c0d306a"
DEFAULT_TARGETS="aarch64-apple-darwin x86_64-apple-darwin x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu x86_64-pc-windows-msvc aarch64-pc-windows-msvc"

COMMIT="$DEFAULT_COMMIT"
OUT_DIR="$REPO_ROOT/dist/cos-runtime"
BUCKET="${COS_BUCKET:-}"
REGION="${COS_REGION:-}"
PREFIX="runtime"
UV_VERSION=""
TARGETS="$DEFAULT_TARGETS"
WITH_UV=true
WITH_SOURCE=true
DO_UPLOAD=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --commit) COMMIT="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --bucket) BUCKET="$2"; shift 2 ;;
        --region) REGION="$2"; shift 2 ;;
        --prefix) PREFIX="$2"; shift 2 ;;
        --uv-version) UV_VERSION="$2"; shift 2 ;;
        --targets) TARGETS="$2"; shift 2 ;;
        --no-uv) WITH_UV=false; shift ;;
        --no-source) WITH_SOURCE=false; shift ;;
        --upload) DO_UPLOAD=true; shift ;;
        -h|--help) sed -n '2,200p' "$0" | sed -n '/^# Usage:/,/^# ====/p' | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

log() { printf '→ %s\n' "$1"; }
die() { printf '✗ %s\n' "$1" >&2; exit 1; }

# Resolve to the full SHA so the object name matches what the desktop build
# stamp pins (bootstrap-runner passes the full commit to install.sh --commit).
FULL_SHA="$(git -C "$REPO_ROOT" rev-parse "${COMMIT}^{commit}" 2>/dev/null)" \
    || die "commit not found in $REPO_ROOT: $COMMIT"

mkdir -p "$OUT_DIR"
SRC_TARBALL="$OUT_DIR/hermes-agent-${FULL_SHA}.tar.gz"

# ── 1. Source tarball ───────────────────────────────────────────────────────
# --prefix=hermes-agent/ so install.sh extracts with --strip-components=1.
if [ "$WITH_SOURCE" = true ]; then
    log "Archiving runtime source ${FULL_SHA:0:12} -> $(basename "$SRC_TARBALL")"
    git -C "$REPO_ROOT" archive --format=tar.gz --prefix=hermes-agent/ "$FULL_SHA" -o "$SRC_TARBALL"
    log "  $(du -h "$SRC_TARBALL" | cut -f1)  $SRC_TARBALL"
else
    log "Skipping source tarball (--no-source)."
fi

# ── 2. uv binaries (fetched from GitHub; re-hosted because npmmirror lacks uv) ─
UV_ARTIFACTS=()
if [ "$WITH_UV" = true ]; then
    if [ -n "$UV_VERSION" ]; then
        UV_BASE="https://github.com/astral-sh/uv/releases/download/${UV_VERSION}"
    else
        # GitHub serves the newest release's asset under /releases/latest/download/.
        UV_BASE="https://github.com/astral-sh/uv/releases/latest/download"
    fi
    for triple in $TARGETS; do
        # uv ships Windows as a .zip and every other platform as .tar.gz.
        case "$triple" in
            *windows*) uv_ext="zip" ;;
            *)         uv_ext="tar.gz" ;;
        esac
        out="$OUT_DIR/uv-${triple}.${uv_ext}"
        url="${UV_BASE}/uv-${triple}.${uv_ext}"
        if [ -s "$out" ]; then
            log "Reusing cached uv ($triple) at $out"
        else
            log "Fetching uv ($triple) from $url"
            # --http1.1: GitHub release downloads intermittently stall on HTTP/2
            # framing errors from some networks; forcing HTTP/1.1 + more retries
            # makes the fetch reliable (the asset itself is not blocked).
            curl -fSL --http1.1 --retry 5 "$url" -o "$out" || die "uv download failed: $url"
        fi
        UV_ARTIFACTS+=("$out")
    done
fi

# ── 3. Upload to COS (public-read bucket) ───────────────────────────────────
upload_one() {
    local local_path="$1" key="$2"
    local dest="cos://${BUCKET}/${PREFIX%/}/${key}"
    log "Uploading $(basename "$local_path") -> $dest"
    # Region/endpoint come from coscli's own config (~/.cos.yaml); `coscli cp`
    # has no --region flag (it resolves the bucket alias from config).
    coscli cp "$local_path" "$dest" \
        || die "coscli upload failed for $key (is coscli configured + bucket public-read?)"
}

if [ "$DO_UPLOAD" = true ]; then
    command -v coscli >/dev/null 2>&1 || die "coscli not found — install + configure it, or drop --upload to only build artifacts"
    [ -n "$BUCKET" ] || die "--bucket (or COS_BUCKET) is required for --upload"
    if [ "$WITH_SOURCE" = true ]; then
        upload_one "$SRC_TARBALL" "hermes-agent-${FULL_SHA}.tar.gz"
    fi
    for a in "${UV_ARTIFACTS[@]:-}"; do
        [ -n "$a" ] && upload_one "$a" "$(basename "$a")"
    done
    BASE_HINT="https://${BUCKET}.cos.${REGION:-<region>}.myqcloud.com/${PREFIX%/}"
    log "Done. Set HERMES_RUNTIME_COS_BASE=${BASE_HINT}"
else
    log "Built artifacts in $OUT_DIR (re-run with --upload to push to COS):"
    ls -1 "$OUT_DIR"
    log "Skipped upload (--upload not set)."
fi
