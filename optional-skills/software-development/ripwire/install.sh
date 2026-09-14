#!/usr/bin/env bash
#
# install.sh - install the ripwire binary on POSIX systems (macOS, Linux, WSL, Git Bash).
#
# Priority: already installed -> Homebrew -> GitHub pinned release tarball -> local build.
#
# Flags:
#   --version=<v>    Pin a specific version for the GitHub download (default 0.6.0)
#   --no-fallback    Don't build from source; fail if download misses
#   --quiet, -q      Suppress non-error output
#
# Exit codes:
#   0  Installed (or already present)
#   1  Argument error
#   2  All install methods failed
#   3  Network failure during download

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_ROOT="$SCRIPT_DIR"
CACHE_BIN_DIR="${RIPWIRE_BIN_DIR:-$SKILL_ROOT/bin}"

PINNED_VERSION="0.6.0"
USE_FALLBACK=1
QUIET=0

log() { [ "$QUIET" -eq 0 ] && printf '[install.sh] %s\n' "$*" >&2 || true; }
err()  { printf '[install.sh] error: %s\n' "$*" >&2; }

while [ "$#" -gt 0 ]; do
  case "$1" in
    --version=*) PINNED_VERSION="${1#*=}" ;;
    --no-fallback) USE_FALLBACK=0 ;;
    --quiet|-q) QUIET=1 ;;
    *) err "unknown argument: $1"; exit 1 ;;
  esac
  shift
done

OS="$(case "$(uname -s)" in Darwin) echo darwin ;; Linux) echo linux ;; MINGW*|MSYS*|CYGWIN*) echo windows ;; *) echo unknown ;; esac)"
ARCH="$(case "$(uname -m)" in arm64|aarch64) echo arm64 ;; x86_64|amd64) echo x64 ;; *) echo unknown ;; esac)"

ripwire_present() { command -v ripwire >/dev/null 2>&1 || [ -x "$CACHE_BIN_DIR/ripwire" ]; }

if ripwire_present; then
  log "ripwire already installed: $(command -v ripwire 2>/dev/null || echo "$CACHE_BIN_DIR/ripwire")"
  exit 0
fi

# --- GitHub release tarball (primary: zero-dep static binary, sha256 provided upstream) ---
try_github() {
  [ "$OS-$ARCH" != "unknown" ] || { err "no release asset for $(uname -s)-$(uname -m)"; return 1; }
  command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1 || { err "need curl or wget"; return 1; }

  local asset="ripwire-${PINNED_VERSION}-${OS}-${ARCH}.tar.gz"
  local url="https://github.com/redhat-et/ripwire/releases/download/v${PINNED_VERSION}/${asset}"
  local tmp; tmp="$(mktemp -d -t ripwire-install-XXXXXX)"

  log "downloading $url"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$url" -o "$tmp/$asset" || { rm -rf "$tmp"; return 3; }
    curl -fsSL "$url.sha256" -o "$tmp/$asset.sha256" 2>/dev/null || true
  else
    wget -q "$url" -O "$tmp/$asset" || { rm -rf "$tmp"; return 3; }
    wget -q "$url.sha256" -O "$tmp/$asset.sha256" 2>/dev/null || true
  fi

  # verify checksum when the sidecar exists (upstream publishes .sha256 for every asset)
  if [ -s "$tmp/$asset.sha256" ]; then
    ( cd "$tmp" && sha256sum -c "$asset.sha256" >/dev/null 2>&1 ) \
      || { err "sha256 mismatch for $asset"; rm -rf "$tmp"; return 3; }
    log "sha256 verified"
  fi

  mkdir -p "$CACHE_BIN_DIR"
  tar xzf "$tmp/$asset" -C "$tmp"
  local bin; bin="$(find "$tmp" -type f -name ripwire -perm -u+x | head -1)"
  [ -n "$bin" ] || { err "no ripwire binary inside $asset"; rm -rf "$tmp"; return 1; }
  mv "$bin" "$CACHE_BIN_DIR/ripwire"
  rm -rf "$tmp"
  log "installed: $CACHE_BIN_DIR/ripwire ($($CACHE_BIN_DIR/ripwire --version))"
  log "Add to PATH: export PATH=\"$CACHE_BIN_DIR:\$PATH\""
  return 0
}

# --- Homebrew (macOS) ---
try_brew() {
  command -v brew >/dev/null 2>&1 || return 1
  log "trying: brew install ripwire"
  brew install ripwire && return 0 || return 1
}

# --- build from source (C++23: GCC 13+ / Clang 17+) ---
try_build() {
  command -v cmake >/dev/null 2>&1 || { log "cmake missing; skip build"; return 1; }
  local cxx_ok=0
  if command -v g++ >/dev/null 2>&1 && g++ --version | head -1 | grep -oE '[0-9]+' | head -1 | grep -qE '1[3-9]|[2-9][0-9]'; then cxx_ok=1; fi
  [ "$cxx_ok" = 1 ] || { log "no C++23 compiler (need GCC 13+); skip build"; return 1; }
  local tmp; tmp="$(mktemp -d -t ripwire-build-XXXXXX)"
  log "building from source in $tmp (~4 min on 4 vCPU)"
  git clone --depth 1 https://github.com/redhat-et/ripwire.git "$tmp/src" >/dev/null 2>&1 || { rm -rf "$tmp"; return 3; }
  cmake -S "$tmp/src" -B "$tmp/build" -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1 \
    && cmake --build "$tmp/build" -j >/dev/null 2>&1 \
    || { rm -rf "$tmp"; return 1; }
  mkdir -p "$CACHE_BIN_DIR"
  mv "$tmp/build/ripwire" "$CACHE_BIN_DIR/ripwire"
  rm -rf "$tmp"
  log "built and installed: $CACHE_BIN_DIR/ripwire"
  return 0
}

if [ "$OS" = "darwin" ] && try_brew; then exit 0; fi
if try_github; then exit 0; fi

if [ "$USE_FALLBACK" -eq 1 ]; then
  if try_build; then exit 0; fi
fi

err "all install methods failed."
err "Manual: https://github.com/redhat-et/ripwire/releases (binary) or cmake build from source."
exit 2
