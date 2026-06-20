#!/usr/bin/env bash
# ============================================================================
# scripts/lib/node-bootstrap.sh
# ----------------------------------------------------------------------------
# Sourceable helper: ensure Node.js satisfies ^20.19 || >=22.12 for the TUI
# (React + Ink), browser tools, and the WhatsApp bridge.
#
# Strategy (first hit wins — respects the user's existing tooling):
#   1. modern `node` already on PATH
#   2. ~/.hermes/node/ from a prior Hermes-managed install
#   3. fnm, proto, nvm (in that order) if the user already uses a version manager
#   4. Termux `pkg`, macOS Homebrew
#   5. pinned nodejs.org tarball into ~/.hermes/node/ (always works, zero shell rc edits)
#
# Usage:
#   source scripts/lib/node-bootstrap.sh
#   ensure_node   # returns 0 on success, non-zero on failure
#   if [ "$HERMES_NODE_AVAILABLE" = true ]; then ...; fi
#
# Env inputs (set before sourcing to override defaults):
#   HERMES_NODE_TARGET_MAJOR  (default: 22)   — installed when we install
#   HERMES_HOME               (default: $HOME/.hermes)
# ============================================================================

HERMES_NODE_TARGET_MAJOR="${HERMES_NODE_TARGET_MAJOR:-22}"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
HERMES_NODE_AVAILABLE=false

# ---------------------------------------------------------------------------
# Logging — prefer the host script's log_* helpers when present
# ---------------------------------------------------------------------------

_nb_log()  { declare -F log_info    >/dev/null 2>&1 && log_info    "$*" || printf '→ %s\n' "$*" >&2; }
_nb_ok()   { declare -F log_success >/dev/null 2>&1 && log_success "$*" || printf '✓ %s\n' "$*" >&2; }
_nb_warn() { declare -F log_warn    >/dev/null 2>&1 && log_warn    "$*" || printf '⚠ %s\n' "$*" >&2; }

# ---------------------------------------------------------------------------
# Platform + version helpers
# ---------------------------------------------------------------------------

_nb_is_termux() {
    [ -n "${TERMUX_VERSION:-}" ] || [[ "${PREFIX:-}" == *"com.termux/files/usr"* ]]
}

# Point a Hermes-managed Node's `npm install -g` at $HERMES_HOME/node. Global
# bins then land in $HERMES_HOME/node/bin, which Hermes can add to its own
# subprocess PATH without placing node/npm/npx in ~/.local/bin or another
# user-global command directory. Scoped to the managed Node via its prefix-local
# global npmrc; the user's other Node installs / ~/.npmrc are untouched.
# Idempotent no-op when there's no managed npm.
_nb_configure_npm_prefix() {
    [ -x "$HERMES_HOME/node/bin/npm" ] || return 0
    mkdir -p "$HERMES_HOME/node/etc"
    printf 'prefix=%s\n' "$HERMES_HOME/node" > "$HERMES_HOME/node/etc/npmrc"
}

_nb_link_points_into_hermes_node() {
    local link="$1"
    [ -L "$link" ] || return 1

    local target target_abs node_dir
    target="$(readlink "$link" 2>/dev/null)" || return 1
    case "$target" in
        /*) target_abs="$target" ;;
        *)  target_abs="$(dirname "$link")/$target" ;;
    esac

    node_dir="$HERMES_HOME/node"
    case "$target_abs" in
        "$node_dir"|"$node_dir"/*) return 0 ;;
    esac

    if [ -d "$node_dir" ] && [ -e "$target_abs" ]; then
        local resolved_target resolved_node
        resolved_target="$(cd "$(dirname "$target_abs")" 2>/dev/null && pwd -P)/$(basename "$target_abs")"
        resolved_node="$(cd "$node_dir" 2>/dev/null && pwd -P)"
        case "$resolved_target" in
            "$resolved_node"|"$resolved_node"/*) return 0 ;;
        esac
    fi
    return 1
}

_nb_remove_legacy_node_links() {
    local dirs=("$HOME/.local/bin" "/usr/local/bin")
    if _nb_is_termux && [ -n "${PREFIX:-}" ]; then
        dirs+=("$PREFIX/bin")
    fi

    local dir name link
    for dir in "${dirs[@]}"; do
        [ -d "$dir" ] || continue
        for name in node npm npx; do
            link="$dir/$name"
            if _nb_link_points_into_hermes_node "$link"; then
                rm -f "$link" 2>/dev/null || true
            fi
        done
    done
}

_nb_have_modern_node() {
    command -v node >/dev/null 2>&1 || return 1
    local ver major minor
    ver=$(node --version 2>/dev/null | sed 's/^v//')
    major="${ver%%.*}"
    minor="${ver#*.}"; minor="${minor%%.*}"
    [[ "$major" =~ ^[0-9]+$ ]] || return 1
    [[ "$minor" =~ ^[0-9]+$ ]] || minor=0
    if [ "$major" -eq 20 ] && [ "$minor" -ge 19 ]; then return 0; fi
    if [ "$major" -eq 22 ] && [ "$minor" -ge 12 ]; then return 0; fi
    [ "$major" -gt 22 ]
}

# ---------------------------------------------------------------------------
# Version-manager paths — respect what the user already uses
# ---------------------------------------------------------------------------

_nb_try_fnm() {
    command -v fnm >/dev/null 2>&1 || return 1
    _nb_log "fnm detected — installing Node $HERMES_NODE_TARGET_MAJOR..."
    eval "$(fnm env 2>/dev/null)" || true
    fnm install "$HERMES_NODE_TARGET_MAJOR" >/dev/null 2>&1 || return 1
    fnm use     "$HERMES_NODE_TARGET_MAJOR" >/dev/null 2>&1 || return 1
    _nb_have_modern_node || return 1
    _nb_ok "Node $(node --version) activated via fnm"
    return 0
}

_nb_try_proto() {
    command -v proto >/dev/null 2>&1 || return 1
    _nb_log "proto detected — installing Node $HERMES_NODE_TARGET_MAJOR..."
    proto install node "$HERMES_NODE_TARGET_MAJOR" >/dev/null 2>&1 || return 1
    _nb_have_modern_node || return 1
    _nb_ok "Node $(node --version) activated via proto"
    return 0
}

_nb_try_nvm() {
    local nvm_sh="${NVM_DIR:-$HOME/.nvm}/nvm.sh"
    [ -s "$nvm_sh" ] || return 1
    # shellcheck source=/dev/null
    \. "$nvm_sh" >/dev/null 2>&1 || return 1
    _nb_log "nvm detected — installing Node $HERMES_NODE_TARGET_MAJOR..."
    nvm install "$HERMES_NODE_TARGET_MAJOR" >/dev/null 2>&1 || return 1
    nvm use     "$HERMES_NODE_TARGET_MAJOR" >/dev/null 2>&1 || return 1
    _nb_have_modern_node || return 1
    _nb_ok "Node $(node --version) activated via nvm"
    return 0
}

# ---------------------------------------------------------------------------
# Platform package managers
# ---------------------------------------------------------------------------

_nb_try_termux_pkg() {
    _nb_is_termux || return 1
    _nb_log "Installing Node.js via pkg..."
    pkg install -y nodejs >/dev/null 2>&1 || return 1
    _nb_have_modern_node || return 1
    _nb_ok "Node $(node --version) installed via pkg"
    return 0
}

_nb_try_brew() {
    [ "$(uname -s)" = "Darwin" ] || return 1
    command -v brew >/dev/null 2>&1 || return 1
    _nb_log "Installing Node via Homebrew..."
    brew install "node@${HERMES_NODE_TARGET_MAJOR}" >/dev/null 2>&1 \
        || brew install node >/dev/null 2>&1 \
        || return 1
    brew link --overwrite --force "node@${HERMES_NODE_TARGET_MAJOR}" >/dev/null 2>&1 || true
    _nb_have_modern_node || return 1
    _nb_ok "Node $(node --version) installed via Homebrew"
    return 0
}

# ---------------------------------------------------------------------------
# Bundled binary fallback — always works, no shell rc edits
# ---------------------------------------------------------------------------

_nb_install_bundled_node() {
    local arch node_arch os_name node_os
    arch=$(uname -m)
    case "$arch" in
        x86_64)        node_arch="x64"    ;;
        aarch64|arm64) node_arch="arm64"  ;;
        armv7l)        node_arch="armv7l" ;;
        *)
            _nb_warn "Unsupported arch ($arch) — install Node.js manually: https://nodejs.org/"
            return 1
            ;;
    esac

    os_name=$(uname -s)
    case "$os_name" in
        Linux*)  node_os="linux"  ;;
        Darwin*) node_os="darwin" ;;
        *)
            _nb_warn "Unsupported OS ($os_name) — install Node.js manually: https://nodejs.org/"
            return 1
            ;;
    esac

    local index_url="https://nodejs.org/dist/latest-v${HERMES_NODE_TARGET_MAJOR}.x/"
    local tarball
    tarball=$(curl -fsSL "$index_url" \
        | grep -oE "node-v${HERMES_NODE_TARGET_MAJOR}\.[0-9]+\.[0-9]+-${node_os}-${node_arch}\.tar\.xz" \
        | head -1)
    if [ -z "$tarball" ]; then
        tarball=$(curl -fsSL "$index_url" \
            | grep -oE "node-v${HERMES_NODE_TARGET_MAJOR}\.[0-9]+\.[0-9]+-${node_os}-${node_arch}\.tar\.gz" \
            | head -1)
    fi
    if [ -z "$tarball" ]; then
        _nb_warn "Could not resolve Node $HERMES_NODE_TARGET_MAJOR binary for $node_os-$node_arch"
        return 1
    fi

    local tmp
    tmp=$(mktemp -d)
    _nb_log "Downloading $tarball..."
    curl -fsSL "${index_url}${tarball}" -o "$tmp/$tarball" || {
        _nb_warn "Download failed"; rm -rf "$tmp"; return 1
    }

    _nb_log "Extracting to $HERMES_HOME/node/..."
    if [[ "$tarball" == *.tar.xz ]]; then
        tar xf  "$tmp/$tarball" -C "$tmp" || { rm -rf "$tmp"; return 1; }
    else
        tar xzf "$tmp/$tarball" -C "$tmp" || { rm -rf "$tmp"; return 1; }
    fi

    local extracted
    extracted=$(find "$tmp" -maxdepth 1 -type d -name 'node-v*' 2>/dev/null | head -1)
    if [ ! -d "$extracted" ]; then
        _nb_warn "Extraction produced no node-v* directory"
        rm -rf "$tmp"
        return 1
    fi

    mkdir -p "$HERMES_HOME"
    rm -rf "$HERMES_HOME/node"
    mv "$extracted" "$HERMES_HOME/node"
    rm -rf "$tmp"

    _nb_remove_legacy_node_links
    _nb_configure_npm_prefix

    export PATH="$HERMES_HOME/node/bin:$PATH"

    _nb_have_modern_node || return 1
    _nb_ok "Node $(node --version) installed to $HERMES_HOME/node/"
    return 0
}

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

ensure_node() {
    HERMES_NODE_AVAILABLE=false

    # Repair pre-existing managed installs. The cleanup only removes
    # node/npm/npx symlinks that still point into this Hermes-managed Node tree.
    _nb_remove_legacy_node_links
    _nb_configure_npm_prefix

    if _nb_have_modern_node; then
        _nb_ok "Node $(node --version) found"
        HERMES_NODE_AVAILABLE=true
        return 0
    fi

    if [ -x "$HERMES_HOME/node/bin/node" ]; then
        export PATH="$HERMES_HOME/node/bin:$PATH"
        if _nb_have_modern_node; then
            _nb_ok "Node $(node --version) found (Hermes-managed)"
            HERMES_NODE_AVAILABLE=true
            return 0
        fi
    fi

    # Version managers first — respect the user's existing setup.
    _nb_try_fnm   && { HERMES_NODE_AVAILABLE=true; return 0; }
    _nb_try_proto && { HERMES_NODE_AVAILABLE=true; return 0; }
    _nb_try_nvm   && { HERMES_NODE_AVAILABLE=true; return 0; }

    # Platform package managers.
    _nb_try_termux_pkg && { HERMES_NODE_AVAILABLE=true; return 0; }
    _nb_try_brew       && { HERMES_NODE_AVAILABLE=true; return 0; }

    # Last resort: pinned nodejs.org tarball.
    _nb_install_bundled_node && { HERMES_NODE_AVAILABLE=true; return 0; }

    _nb_warn "Node.js install failed — TUI and browser tools will be unavailable."
    _nb_warn "Install manually: https://nodejs.org/en/download/  (or: \`brew install node\`, \`fnm install $HERMES_NODE_TARGET_MAJOR\`, etc.)"
    return 1
}
