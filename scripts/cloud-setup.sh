#!/usr/bin/env bash
# Bootstrap a fresh Linux environment for working on hermes-agent.
#
# Built for the Claude Code for Web sandbox (paste this into the cloud
# environment's setup-script field) and reused by .devcontainer so local /
# Codespaces dev matches. Idempotent — safe to re-run.
#
# What it provisions:
#   * uv (pinned) — Python toolchain + package manager
#   * Python 3.14 + locked deps in .venv. `uv sync --locked` asserts uv.lock is
#     in sync with pyproject.toml (it fails loudly if a pin was bumped without
#     re-locking) and installs exactly that locked, reproducible set — the same
#     .[all,dev] package footprint CI installs (note: CI runs the suite on
#     3.11; 3.14 is newer than the CI interpreter). The platform-bound extras
#     (matrix, voice) are intentionally excluded — they don't build cleanly on
#     every arch.
#   * optional: Node workspace deps (ui-tui, web, website) via --with-node
#
# Deliberately NOT installed: ripgrep. The only tests that shell out to it skip
# gracefully when it's absent (tests/tools/test_search_hidden_dirs.py), and the
# Claude Code sandbox already ships `rg` on PATH.
#
# Network: works as-is under Claude Code for Web "Trusted" egress. For "Custom"
# egress, allowlist: github.com, objects.githubusercontent.com, pypi.org,
# files.pythonhosted.org (and registry.npmjs.org for --with-node). uv itself is
# fetched from GitHub Releases — astral.sh does NOT need to be allowlisted (some
# restricted sandboxes return HTTP 403 for it).
#
# Usage:
#   scripts/cloud-setup.sh              # Python env only (fast)
#   scripts/cloud-setup.sh --with-node  # also install Node workspaces

set -euo pipefail

# Pinned uv version + sha256 of each GitHub release tarball. Bump deliberately
# (and re-test) rather than tracking latest — consistent with this repo's
# exact-pin supply-chain policy. Regenerate the digests when bumping UV_VERSION:
#   for t in x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu; do
#     curl -sSL "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-$t.tar.gz.sha256"
#   done
UV_VERSION="0.11.13"
UV_SHA256_x86_64="f830ea3d38ae1492acf53cb7f2cd0f81d6ae22b42d2d7310a6c7d42c451e1a43"
UV_SHA256_aarch64="12366407dc1fdba5179b10bd69c11ebfc2eff25791366089c0b2f5701056efc5"

WITH_NODE=0
for arg in "$@"; do
  case "$arg" in
    --with-node) WITH_NODE=1 ;;
    -h | --help)
      echo "usage: scripts/cloud-setup.sh [--with-node]"
      exit 0
      ;;
    *)
      echo "error: unknown argument: $arg" >&2
      echo "usage: scripts/cloud-setup.sh [--with-node]" >&2
      exit 2
      ;;
  esac
done

# Locate the hermes-agent checkout. This script runs in three shapes and must
# find the repo in all of them:
#   1. in-place     scripts/cloud-setup.sh — trust the script's own location
#   2. pasted body  cloud setup-script field — runs from a temp path, so
#                   BASH_SOURCE is useless. The sandbox's CWD is $HOME and the
#                   repo is cloned into a SUBDIR (e.g. /home/user/hermes-agent),
#                   so walking *up* from CWD won't find it — we must look down.
#   3. devcontainer CWD is the repo root
# A dir is the repo root iff it has BOTH pyproject.toml and run_agent.py.
# run_agent.py is unique to the root, so this won't match the vendored
# tinker-atropos/ sub-project (which ships its own pyproject.toml).
_is_hermes_root() { [ -f "$1/pyproject.toml" ] && [ -f "$1/run_agent.py" ]; }

resolve_repo_root() {
  local d c p
  # 1) walk up from $PWD (CWD is the repo root or a subdir of it)
  d="$PWD"
  while :; do
    _is_hermes_root "$d" && {
      printf '%s\n' "$d"
      return 0
    }
    [ "$d" = "/" ] && break
    d="$(dirname "$d")"
  done
  # 2) script-relative (genuine in-place run from outside the repo)
  d="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." 2> /dev/null && pwd || true)"
  [ -n "$d" ] && _is_hermes_root "$d" && {
    printf '%s\n' "$d"
    return 0
  }
  # 3) the sandbox clones the repo into a subdir, not into CWD — scan the
  #    immediate children of CWD and common checkout roots (any dir name works)
  for c in "$PWD"/*/ "$HOME"/*/ /workspace/*/ /workspaces/*/ /repo /repos/*/ /src/*/; do
    c="${c%/}"
    _is_hermes_root "$c" && {
      printf '%s\n' "$c"
      return 0
    }
  done
  # 4) bounded downward scan as a last resort (GNU find; the sandbox is Linux)
  while IFS= read -r p; do
    d="$(dirname "$p")"
    _is_hermes_root "$d" && {
      printf '%s\n' "$d"
      return 0
    }
  done < <(find "$PWD" "$HOME" /workspace /workspaces -maxdepth 3 -name pyproject.toml 2> /dev/null)
  return 1
}

if ! REPO_ROOT="$(resolve_repo_root)"; then
  {
    echo "error: could not locate the hermes-agent checkout."
    echo "  CWD=$PWD  HOME=$HOME"
    echo "  A repo root is a dir containing BOTH pyproject.toml and run_agent.py."
    echo "  Searched: up from CWD, the script dir, immediate children of"
    echo "  \$PWD/\$HOME/workspace(s), and a depth-3 scan — found nothing."
    echo "  --- contents of likely roots (so we can see where it was cloned):"
    seen=""
    for d in "$PWD" "$HOME" /workspace /workspaces /repo /src; do
      case " $seen " in *" $d "*) continue ;; esac
      [ -d "$d" ] || continue
      seen="$seen $d"
      echo "  [$d]"
      ls -la "$d" 2> /dev/null | sed 's/^/    /'
    done
    echo "  If hermes-agent is NOT listed above, the sandbox ran setup BEFORE"
    echo "  cloning (or no repo is attached): attach the repo to the environment,"
    echo "  or move 'uv sync' to a post-clone step. Otherwise paste the path above."
  } >&2
  exit 2
fi
cd "$REPO_ROOT"
echo "▶ repo root: $REPO_ROOT"

# ── uv ────────────────────────────────────────────────────────────────────────
# Ensure the *pinned* uv runs — do NOT just reuse whatever uv is already on PATH.
# The Claude Code sandbox ships uv 0.8.17, which is too old to parse this repo's
# `exclude-newer = "<N> days"` (the relative supply-chain cutoff under [tool.uv]
# in pyproject.toml). An old uv silently drops that cutoff, re-resolves a
# different package set, and then `uv sync --locked` aborts with "lockfile needs
# to be updated". The pinned uv understands the cutoff (it also records
# `exclude-newer-span` in uv.lock), so the lock validates and the locked sync
# succeeds. The standalone installer drops uv in ~/.local/bin, so put that on
# PATH first and pin the install dir there so the freshly installed uv wins.
export PATH="$HOME/.local/bin:$PATH"
uv_current=""
if command -v uv > /dev/null 2>&1; then
  read -r _ uv_current _ < <(uv --version 2> /dev/null) || true
fi
if [ "$uv_current" != "$UV_VERSION" ]; then
  if [ -n "$uv_current" ]; then
    echo "▶ found uv ${uv_current}, but this repo pins ${UV_VERSION} — installing pinned uv"
  else
    echo "▶ installing uv ${UV_VERSION}"
  fi
  # Pull the standalone binary straight from GitHub Releases. We deliberately do
  # NOT use the astral.sh installer: restricted sandboxes (e.g. Claude Code for
  # Web on "Custom" egress) block astral.sh with HTTP 403, whereas github.com +
  # objects.githubusercontent.com are already required (python-build-standalone,
  # git deps), so this adds no new allowlist host.
  case "$(uname -m)" in
    x86_64 | amd64) uv_target="x86_64-unknown-linux-gnu" uv_sha="$UV_SHA256_x86_64" ;;
    aarch64 | arm64) uv_target="aarch64-unknown-linux-gnu" uv_sha="$UV_SHA256_aarch64" ;;
    *)
      echo "error: no pinned uv build for arch '$(uname -m)'" >&2
      exit 1
      ;;
  esac
  uv_tarball="uv-${uv_target}.tar.gz"
  uv_tmp="$(mktemp -d)"
  curl -fsSL "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/${uv_tarball}" -o "$uv_tmp/$uv_tarball"
  echo "${uv_sha}  $uv_tmp/$uv_tarball" | sha256sum -c - > /dev/null
  tar -xzf "$uv_tmp/$uv_tarball" -C "$uv_tmp"
  mkdir -p "$HOME/.local/bin"
  install -m 0755 "$uv_tmp/uv-${uv_target}/uv" "$HOME/.local/bin/uv"
  install -m 0755 "$uv_tmp/uv-${uv_target}/uvx" "$HOME/.local/bin/uvx"
  rm -rf "$uv_tmp"
  hash -r 2> /dev/null || true
fi
uv --version

# ── Python + locked deps ──────────────────────────────────────────────────────
# Creates .venv at the repo root, which scripts/run_tests.sh probes for.
echo "▶ syncing Python 3.14 environment from uv.lock (.[all,dev])"
uv sync --locked --python 3.14 --extra all --extra dev

# ── Node workspaces (optional) ────────────────────────────────────────────────
if [ "$WITH_NODE" -eq 1 ]; then
  if ! command -v npm > /dev/null 2>&1; then
    echo "error: --with-node given but npm is not on PATH (need Node 24+)" >&2
    exit 1
  fi
  for dir in ui-tui web website; do
    if [ -f "$dir/package.json" ]; then
      echo "▶ npm ci in $dir"
      (cd "$dir" && npm ci)
    fi
  done
fi

echo "✓ setup complete. Run the test suite with: scripts/run_tests.sh"
