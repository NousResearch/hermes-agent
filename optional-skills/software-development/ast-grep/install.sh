#!/usr/bin/env bash
#
# install.sh - install the ast-grep binary on POSIX systems (macOS, Linux, WSL, Git Bash).
#
# Tries package managers in priority order, then falls back to downloading a
# pinned release binary from GitHub into <skill_root>/bin/sg.
#
# Order:
#   1. Already installed?  -> nothing to do
#   2. Homebrew (brew)
#   3. npm (@ast-grep/cli)
#   4. cargo binstall (faster) or cargo install (slower)
#   5. pip (ast-grep-cli)
#   6. nix-env (NixOS / Nix users)
#   7. mise (asdf successor)
#   8. GitHub release tarball -> <skill_root>/bin/sg
#
# Flags:
#   --method=<m>     Force one method: brew | npm | cargo | pip | nix | mise | github
#   --version=<v>    Pin a specific version when downloading from GitHub
#   --no-fallback    Don't fall back to GitHub tarball; fail if all package managers miss
#   --quiet, -q      Suppress non-error output
#
# Exit codes:
#   0  Installed (or already present)
#   1  Argument error
#   2  All install methods failed
#   3  Network failure during GitHub fallback

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_ROOT="$SCRIPT_DIR"
CACHE_BIN_DIR="${OMO_AST_GREP_BIN_DIR:-$SKILL_ROOT/bin}"

PINNED_VERSION="0.45.0"
FORCED_METHOD=""
USE_FALLBACK=1
QUIET=0

log() {
  if [ "$QUIET" -eq 0 ]; then
    printf '[install.sh] %s\n' "$*" >&2
  fi
}

err() {
  printf '[install.sh] error: %s\n' "$*" >&2
}

usage() {
  sed -n '2,/^set -/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//;/^set -/d'
  exit "${1:-0}"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --method=*) FORCED_METHOD="${1#*=}" ;;
    --version=*) PINNED_VERSION="${1#*=}" ;;
    --no-fallback) USE_FALLBACK=0 ;;
    --quiet|-q) QUIET=1 ;;
    --help|-h) usage 0 ;;
    *) err "unknown argument: $1"; usage 1 ;;
  esac
  shift
done

# --- detect platform -----------------------------------------------------

detect_os() {
  case "$(uname -s)" in
    Darwin) echo "darwin" ;;
    Linux) echo "linux" ;;
    MINGW*|MSYS*|CYGWIN*) echo "windows" ;;
    *) echo "unknown" ;;
  esac
}

detect_arch() {
  case "$(uname -m)" in
    arm64|aarch64) echo "aarch64" ;;
    x86_64|amd64) echo "x86_64" ;;
    *) echo "unknown" ;;
  esac
}

OS="$(detect_os)"
ARCH="$(detect_arch)"

# --- already installed? --------------------------------------------------

ast_grep_present() {
  if command -v ast-grep >/dev/null 2>&1; then
    return 0
  fi
  if command -v sg >/dev/null 2>&1; then
    if [ "$OS" = "linux" ]; then
      if "$(command -v sg)" --version 2>/dev/null | grep -qi 'ast-grep'; then
        return 0
      fi
      return 1
    fi
    return 0
  fi
  if [ -x "$CACHE_BIN_DIR/sg" ] || [ -x "$CACHE_BIN_DIR/ast-grep" ]; then
    return 0
  fi
  return 1
}

if [ -z "$FORCED_METHOD" ] && ast_grep_present; then
  log "ast-grep already installed: $(command -v ast-grep 2>/dev/null || command -v sg)"
  exit 0
fi

# --- per-method installers -----------------------------------------------

try_brew() {
  command -v brew >/dev/null 2>&1 || return 1
  log "trying: brew install ast-grep"
  brew install ast-grep && return 0 || return 1
}

try_npm() {
  command -v npm >/dev/null 2>&1 || return 1
  log "trying: npm install -g @ast-grep/cli"
  npm install -g @ast-grep/cli && return 0 || return 1
}

try_cargo() {
  if command -v cargo-binstall >/dev/null 2>&1; then
    log "trying: cargo binstall ast-grep"
    cargo binstall -y ast-grep && return 0 || true
  fi
  if command -v cargo >/dev/null 2>&1; then
    log "trying: cargo install ast-grep --locked"
    cargo install ast-grep --locked && return 0 || return 1
  fi
  return 1
}

try_pip() {
  if command -v pipx >/dev/null 2>&1; then
    log "trying: pipx install ast-grep-cli"
    pipx install ast-grep-cli && return 0 || true
  fi
  command -v pip3 >/dev/null 2>&1 || command -v pip >/dev/null 2>&1 || return 1
  PIP="$(command -v pip3 || command -v pip)"
  log "trying: $PIP install --user ast-grep-cli"
  $PIP install --user ast-grep-cli && return 0 || return 1
}

try_nix() {
  command -v nix-env >/dev/null 2>&1 || return 1
  log "trying: nix-env -iA nixpkgs.ast-grep"
  nix-env -iA nixpkgs.ast-grep && return 0 || return 1
}

try_mise() {
  command -v mise >/dev/null 2>&1 || return 1
  log "trying: mise use -g ast-grep"
  mise use -g ast-grep && return 0 || return 1
}

# Tarball assets are named like:
#   app-aarch64-apple-darwin.zip
#   app-x86_64-apple-darwin.zip
#   app-aarch64-unknown-linux-gnu.zip
#   app-x86_64-unknown-linux-gnu.zip
#   app-x86_64-pc-windows-msvc.zip   (.zip only on windows)

triple_for() {
  case "$OS-$ARCH" in
    darwin-aarch64) echo "aarch64-apple-darwin" ;;
    darwin-x86_64)  echo "x86_64-apple-darwin" ;;
    linux-aarch64)  echo "aarch64-unknown-linux-gnu" ;;
    linux-x86_64)   echo "x86_64-unknown-linux-gnu" ;;
    *) echo "" ;;
  esac
}

try_github() {
  TRIPLE="$(triple_for)"
  if [ -z "$TRIPLE" ]; then
    err "no GitHub release asset for $OS-$ARCH; install via package manager or build from source."
    return 1
  fi

  command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1 || {
    err "need curl or wget for GitHub fallback"
    return 1
  }

  ASSET="app-${TRIPLE}.zip"
  URL="https://github.com/ast-grep/ast-grep/releases/download/${PINNED_VERSION}/${ASSET}"
  TMP="$(mktemp -d -t ast-grep-install-XXXXXX)"
  trap 'rm -rf "$TMP"' RETURN

  log "downloading $URL"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$URL" -o "$TMP/$ASSET" || return 3
  else
    wget -q "$URL" -O "$TMP/$ASSET" || return 3
  fi

  command -v unzip >/dev/null 2>&1 || {
    err "need 'unzip' to extract GitHub release archives"
    return 1
  }

  unzip -q "$TMP/$ASSET" -d "$TMP/extract"
  mkdir -p "$CACHE_BIN_DIR"
  if [ -f "$TMP/extract/ast-grep" ]; then
    mv "$TMP/extract/ast-grep" "$CACHE_BIN_DIR/sg"
  elif [ -f "$TMP/extract/sg" ]; then
    mv "$TMP/extract/sg" "$CACHE_BIN_DIR/sg"
  else
    err "no ast-grep or sg binary found inside $ASSET"
    return 1
  fi
  chmod +x "$CACHE_BIN_DIR/sg"

  log "installed cached binary: $CACHE_BIN_DIR/sg"
  log "verify: $CACHE_BIN_DIR/sg --version"
  log ""
  log "Add to PATH for direct sg use:"
  log "  export PATH=\"$CACHE_BIN_DIR:\$PATH\""
  return 0
}

# --- dispatch ------------------------------------------------------------

run_method() {
  case "$1" in
    brew) try_brew ;;
    npm) try_npm ;;
    cargo) try_cargo ;;
    pip) try_pip ;;
    nix) try_nix ;;
    mise) try_mise ;;
    github) try_github ;;
    *) err "unknown method: $1"; return 1 ;;
  esac
}

if [ -n "$FORCED_METHOD" ]; then
  if run_method "$FORCED_METHOD"; then
    exit 0
  else
    err "method '$FORCED_METHOD' failed"
    exit 2
  fi
fi

# Try methods in OS-aware priority order.
case "$OS" in
  darwin) METHODS=(brew npm cargo pip mise) ;;
  linux)  METHODS=(npm cargo pip nix mise brew) ;;
  windows) METHODS=(npm cargo pip mise) ;;
  *) METHODS=(npm cargo pip) ;;
esac

for m in "${METHODS[@]}"; do
  if run_method "$m"; then
    log "installed via $m"
    exit 0
  fi
  log "$m unavailable or failed; trying next"
done

if [ "$USE_FALLBACK" -eq 1 ]; then
  log "all package managers failed; falling back to GitHub release"
  if try_github; then
    exit 0
  fi
fi

err "all install methods failed."
err ""
err "Manual options:"
err "  brew install ast-grep                 # macOS / linuxbrew"
err "  npm install -g @ast-grep/cli          # any OS with Node"
err "  cargo install ast-grep --locked       # any OS with Rust"
err "  pip install ast-grep-cli              # any OS with Python"
err "  https://github.com/ast-grep/ast-grep/releases   # manual binary"
exit 2
