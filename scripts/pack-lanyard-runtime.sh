#!/usr/bin/env bash
# pack-lanyard-runtime.sh — Build a relocatable Hermes Lanyard runtime tarball.
#
# Consumer contract (company-brain-deploy bootstrap):
#   - Object shape: s3://$BUCKET/hermes/<version>/<filename>.tar.gz
#   - Archive may have a single top-level directory (flattened on extract)
#   - After flatten into /opt/hermes-lanyard, expects bin/hermes executable
#
# Versions must be immutable labels (e.g. v0.1.0). Never main/master/HEAD/latest.
#
# Local dry-run (no AWS):
#   ./scripts/pack-lanyard-runtime.sh --version v0.0.0-test
#
# Env / flags:
#   --version VER          Required immutable version label
#   --output-dir DIR       Where to write tarball + .sha256 (default: dist/lanyard)
#   --skip-frontend        Skip web/TUI npm builds (faster local smoke)
#   --python VER           Python for the venv (default: 3.12)
#   HERMES_PACK_EXTRAS     Space-separated uv --extra names
#                          (default: "all messaging")
#
# Offline for AWS only — still needs network for uv/npm unless caches are warm.
set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
VERSION=""
OUTPUT_DIR="${ROOT}/dist/lanyard"
SKIP_FRONTEND=0
PYTHON_VER="${HERMES_PACK_PYTHON:-3.12}"
# shellcheck disable=SC2206
EXTRAS=(${HERMES_PACK_EXTRAS:-all messaging})

usage() {
  sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
  echo "Usage: $0 --version <vX.Y.Z> [--output-dir DIR] [--skip-frontend] [--python VER]"
}

log() { echo "[pack-lanyard] $*" >&2; }
die() { echo "[pack-lanyard] ERROR: $*" >&2; exit 1; }

# Reject live git branches / mutable refs used as "versions".
is_forbidden_version_ref() {
  local v
  v="$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')"
  case "$v" in
    main|master|head|develop|development|trunk|latest|origin/*|refs/*|"")
      return 0
      ;;
  esac
  return 1
}

while [ $# -gt 0 ]; do
  case "$1" in
    --version) VERSION="${2:-}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:-}"; shift 2 ;;
    --skip-frontend) SKIP_FRONTEND=1; shift ;;
    --python) PYTHON_VER="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

[ -n "$VERSION" ] || die "--version is required (e.g. v0.1.0)"
if is_forbidden_version_ref "$VERSION"; then
  die "Refusing mutable/forbidden version label: '${VERSION}'. Use an immutable tag-style label (e.g. v0.1.0)."
fi
# Soft shape check: prefer v-prefixed labels; allow pre-release suffixes.
if ! echo "$VERSION" | grep -Eq '^[A-Za-z0-9][A-Za-z0-9._+-]*$'; then
  die "Version contains unsafe characters: '${VERSION}'"
fi

command -v uv >/dev/null 2>&1 || die "uv is required on PATH (https://docs.astral.sh/uv/)"
command -v tar >/dev/null 2>&1 || die "tar is required"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required"

cd "$ROOT"

STAGE_PARENT="$(mktemp -d "${TMPDIR:-/tmp}/hermes-lanyard-pack.XXXXXX")"
cleanup() { rm -rf "$STAGE_PARENT"; }
trap cleanup EXIT

# Single top-level directory so bootstrap flatten_extract is a no-op or clean.
TOP_NAME="hermes-${VERSION}"
PREFIX="${STAGE_PARENT}/${TOP_NAME}"
mkdir -p "$PREFIX/bin" "$PREFIX/share" "$PREFIX/.seed"

log "Staging prefix: $PREFIX"
log "Version: $VERSION"
log "Extras: ${EXTRAS[*]}"

# ---------- Frontend assets (package-data for hermes_cli) ----------
if [ "$SKIP_FRONTEND" -eq 0 ]; then
  if command -v npm >/dev/null 2>&1; then
    log "Building web dashboard + TUI (set --skip-frontend to skip)"
    if [ -f package-lock.json ]; then
      npm ci --no-audit --no-fund
    else
      npm install --no-audit --no-fund
    fi
    (cd web && npm ci --no-audit --no-fund && npm run build)
    (cd ui-tui && npm ci --no-audit --no-fund && npm run build)
    mkdir -p hermes_cli/tui_dist
    if [ -f ui-tui/dist/entry.js ]; then
      cp -f ui-tui/dist/entry.js hermes_cli/tui_dist/entry.js
    fi
  else
    log "WARN: npm not found — packing without rebuilt frontend assets"
  fi
else
  log "Skipping frontend build (--skip-frontend)"
fi

# ---------- Relocatable venv + project install ----------
log "Creating relocatable venv (python ${PYTHON_VER})"
uv venv --relocatable --python "$PYTHON_VER" --clear "$PREFIX/.venv"

EXTRA_ARGS=()
for extra in "${EXTRAS[@]}"; do
  EXTRA_ARGS+=(--extra "$extra")
done

log "Installing hermes-agent into prefix (uv sync --frozen --no-editable)"
# Point the project environment at the stage venv so the install is self-contained.
export UV_PROJECT_ENVIRONMENT="$PREFIX/.venv"
uv sync --frozen --no-editable --no-dev "${EXTRA_ARGS[@]}"

# ---------- Bundled skills / optional-skills (Homebrew-style share layout) ----------
log "Copying skills into share/"
if [ -d skills ]; then
  cp -a skills "$PREFIX/share/skills"
fi
if [ -d optional-skills ]; then
  cp -a optional-skills "$PREFIX/share/optional-skills"
fi
if [ -d optional-mcps ]; then
  cp -a optional-mcps "$PREFIX/share/optional-mcps"
fi
if [ -d locales ]; then
  cp -a locales "$PREFIX/share/locales"
fi

# ---------- bin/hermes relocatable wrapper ----------
# Resolve ROOT from this script's location so the tree works after bootstrap
# flatten into /opt/hermes-lanyard (or any install path).
cat >"$PREFIX/bin/hermes" <<'WRAPPER'
#!/bin/sh
# Hermes Lanyard runtime entrypoint — relocatable.
set -eu
ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
VENV_PY="${ROOT}/.venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo "hermes: missing interpreter at $VENV_PY" >&2
  exit 127
fi
# Bundled skill trees (same contract as Homebrew formula).
if [ -d "${ROOT}/share/skills" ]; then
  export HERMES_BUNDLED_SKILLS="${HERMES_BUNDLED_SKILLS:-${ROOT}/share/skills}"
fi
if [ -d "${ROOT}/share/optional-skills" ]; then
  export HERMES_OPTIONAL_SKILLS="${HERMES_OPTIONAL_SKILLS:-${ROOT}/share/optional-skills}"
fi
export HERMES_MANAGED="${HERMES_MANAGED:-lanyard-artifact}"
# Prefer the venv console script when present; fall back to module entry.
if [ -x "${ROOT}/.venv/bin/hermes" ]; then
  exec "${ROOT}/.venv/bin/hermes" "$@"
fi
exec "$VENV_PY" -m hermes_cli.main "$@"
WRAPPER
chmod 0755 "$PREFIX/bin/hermes"

# Install-method stamp next to the code tree (not under HERMES_HOME).
printf 'lanyard-artifact\n' >"$PREFIX/.install_method"
printf '%s\n' "$VERSION" >"$PREFIX/VERSION"
{
  echo "version=${VERSION}"
  echo "product=hermes-lanyard"
  echo "built_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if command -v git >/dev/null 2>&1 && git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "git_sha=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  fi
} >"$PREFIX/.seed/build.env"

# ---------- Archive + SHA-256 ----------
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(CDPATH= cd -- "$OUTPUT_DIR" && pwd)"
TARBALL_NAME="hermes-${VERSION}.tar.gz"
TARBALL_PATH="${OUTPUT_DIR}/${TARBALL_NAME}"
SHA_PATH="${OUTPUT_DIR}/hermes-${VERSION}.sha256"

log "Creating ${TARBALL_PATH}"
# Deterministic-ish: sort names; do not store full uid/gid when supported.
tar \
  --sort=name \
  --owner=0 --group=0 --numeric-owner \
  -C "$STAGE_PARENT" \
  -czf "$TARBALL_PATH" \
  "$TOP_NAME" 2>/dev/null \
  || tar -C "$STAGE_PARENT" -czf "$TARBALL_PATH" "$TOP_NAME"

SHA="$(sha256sum "$TARBALL_PATH" | awk '{print $1}')"
# sha256sum-compatible line (hash  filename) for easy verification
printf '%s  %s\n' "$SHA" "$TARBALL_NAME" >"$SHA_PATH"

# Also write a bare hash file some pin tools prefer
printf '%s\n' "$SHA" >"${OUTPUT_DIR}/hermes-${VERSION}.sha256.hex"

log "Tarball: $TARBALL_PATH"
log "SHA-256: $SHA"
log "SHA file: $SHA_PATH"

# Layout smoke: extract and verify single top-level dir + bin/hermes (pre- and post-flatten).
SMOKE="$(mktemp -d "${TMPDIR:-/tmp}/hermes-lanyard-smoke.XXXXXX")"
tar -xzf "$TARBALL_PATH" -C "$SMOKE"
mapfile -t _entries < <(find "$SMOKE" -mindepth 1 -maxdepth 1 ! -name '.*' | sort)
if [ "${#_entries[@]}" -ne 1 ] || [ ! -d "${_entries[0]}" ]; then
  rm -rf "$SMOKE"
  die "Smoke check failed: expected exactly one top-level directory in tarball"
fi
TOP_EXTRACTED="${_entries[0]}"
if [ ! -x "${TOP_EXTRACTED}/bin/hermes" ]; then
  rm -rf "$SMOKE"
  die "Smoke check failed: ${TOP_NAME}/bin/hermes missing or not executable"
fi
if [ ! -x "${TOP_EXTRACTED}/.venv/bin/python" ]; then
  rm -rf "$SMOKE"
  die "Smoke check failed: .venv/bin/python missing under top-level dir"
fi
# Simulate bootstrap flatten_extract (single top-level dir → install root)
FLAT="$(mktemp -d "${TMPDIR:-/tmp}/hermes-lanyard-flat.XXXXXX")"
# Prefer rsync if present; else tar-pipe to preserve hidden files
if command -v rsync >/dev/null 2>&1; then
  rsync -a "${TOP_EXTRACTED}/" "$FLAT/"
else
  tar -C "$TOP_EXTRACTED" -cf - . | tar -C "$FLAT" -xf -
fi
if [ ! -x "$FLAT/bin/hermes" ] || [ ! -x "$FLAT/.venv/bin/python" ]; then
  rm -rf "$SMOKE" "$FLAT"
  die "Smoke check failed: layout incomplete after flatten simulation"
fi
log "Smoke layout OK (top-level ${TOP_NAME}/ + flattened bin/hermes + .venv)"
rm -rf "$SMOKE" "$FLAT"

# Machine-readable summary for CI
cat <<EOF
PACK_VERSION=${VERSION}
PACK_TARBALL=${TARBALL_PATH}
PACK_TARBALL_NAME=${TARBALL_NAME}
PACK_SHA256=${SHA}
PACK_SHA_FILE=${SHA_PATH}
PACK_S3_KEY=hermes/${VERSION}/${TARBALL_NAME}
EOF
