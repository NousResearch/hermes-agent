#!/usr/bin/env bash
# ============================================================================
# scripts/lib/node-bootstrap.sh
# ----------------------------------------------------------------------------
# Sourceable helper: ensure Node.js matching .nvmrc (or default 26.3.0) is
# available for the TUI, browser tools, and the WhatsApp bridge.
#
# Strategy (strict order):
#   1. ~/.hermes/node/ (Hermes-managed, auto-upgrades if .nvmrc changes)
#   2. `node` already on PATH (fallback, validated against package.json engines)
#   3. Termux `pkg install nodejs` (fallback for Termux)
#
# Usage:
#   source scripts/lib/node-bootstrap.sh
#   ensure_node   # returns 0 on success, non-zero on failure
#   if [ "$HERMES_NODE_AVAILABLE" = true ]; then ...; fi
#
# Env inputs (set before sourcing to override defaults):
#   HERMES_NODE_TARGET_VERSION  (default: read from .nvmrc, fallback 26.3.0)
#   HERMES_HOME                 (default: $HOME/.hermes)
# ============================================================================

# Read target version from .nvmrc if present, otherwise default to 26.3.0
if [ -f ".nvmrc" ]; then
    HERMES_NODE_TARGET_VERSION=$(cat .nvmrc | tr -d '[:space:]')
else
    HERMES_NODE_TARGET_VERSION="${HERMES_NODE_TARGET_VERSION:-26.3.0}"
fi
# Extract major version for fallback compatibility checks
HERMES_NODE_TARGET_MAJOR="${HERMES_NODE_TARGET_VERSION%%.*}"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
HERMES_NODE_AVAILABLE=false

# ---------------------------------------------------------------------------
# Logging
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

_nb_node_major() {
    local v
    v=$(node --version 2>/dev/null | sed 's/^v//' | cut -d. -f1)
    [[ "$v" =~ ^[0-9]+$ ]] && echo "$v" || echo 0
}

# Dynamically read the minimum required Node major version from package.json's engines field.
# Falls back to HERMES_NODE_TARGET_MAJOR if package.json is missing or unreadable.
_nb_get_engines_min_major() {
    local pkg_json="package.json"
    if [ ! -f "$pkg_json" ] && [ -f "../package.json" ]; then
        pkg_json="../package.json"
    fi
    
    if [ -f "$pkg_json" ]; then
        # Extract the first number from the "node" engine string (e.g., ">=26.0.0" -> 26)
        # This avoids matching "node_modules" or other fields by requiring the specific JSON structure.
        local min_major
        min_major=$(grep -E '"node"[[:space:]]*:[[:space:]]*"[^"]*"' "$pkg_json" 2>/dev/null | grep -o '[0-9]\+' | head -1)
        if [[ "$min_major" =~ ^[0-9]+$ ]]; then
            echo "$min_major"
            return 0
        fi
    fi
    
    # Fallback to target major version if parsing fails
    echo "${HERMES_NODE_TARGET_MAJOR}"
}

_nb_have_modern_node() {
    command -v node >/dev/null 2>&1 || return 1
    local current_major
    current_major=$(_nb_node_major)
    local required_major
    required_major=$(_nb_get_engines_min_major)
    
    [ "$current_major" -ge "$required_major" ]
}

# ---------------------------------------------------------------------------
# Hermes-managed Node.js installation (fallback to nodejs.org tarball)
# ---------------------------------------------------------------------------
_nb_install_hermes_node() {
    local arch node_arch os_name node_os
    arch=$(uname -m)
    case "$arch" in
        x86_64)        node_arch="x64"    ;;
        aarch64|arm64) node_arch="arm64"  ;;
        armv7l)        node_arch="armv7l" ;;
        *)
            _nb_warn "Unsupported arch ($arch) for Hermes-managed Node.js"
            return 1
            ;;
    esac

    os_name=$(uname -s)
    case "$os_name" in
        Linux*)  node_os="linux"  ;;
        Darwin*) node_os="darwin" ;;
        *)
            _nb_warn "Unsupported OS ($os_name) for Hermes-managed Node.js"
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
        _nb_warn "Could not resolve Node $HERMES_NODE_TARGET_VERSION binary for $node_os-$node_arch"
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

    # Symlink to standard bin dir so it's on PATH for other tools
    local _link_dir="$HOME/.local/bin"
    if _nb_is_termux && [ -n "${PREFIX:-}" ]; then
        _link_dir="$PREFIX/bin"
    elif [ "$(id -u)" = 0 ] && [ "$os_name" = "Linux" ]; then
        _link_dir="/usr/local/bin"
    fi
    
    mkdir -p "$_link_dir"
    ln -sf "$HERMES_HOME/node/bin/node" "$_link_dir/node"
    ln -sf "$HERMES_HOME/node/bin/npm"  "$_link_dir/npm"
    ln -sf "$HERMES_HOME/node/bin/npx"  "$_link_dir/npx"
    
    export PATH="$HERMES_HOME/node/bin:$PATH"

    if ! _nb_have_modern_node; then
        _nb_warn "Installed Node.js version check failed"
        return 1
    fi
    
    _nb_ok "Node $(node --version) installed to $HERMES_HOME/node/"
    return 0
}

# ---------------------------------------------------------------------------
# Termux pkg fallback
# ---------------------------------------------------------------------------
_nb_try_termux_pkg() {
    _nb_is_termux || return 1
    _nb_log "Installing Node.js via pkg..."
    pkg install -y nodejs >/dev/null 2>&1 || return 1
    if _nb_have_modern_node; then
        _nb_ok "Node $(node --version) installed via pkg"
        return 0
    fi
    return 1
}

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
ensure_node() {
    HERMES_NODE_AVAILABLE=false

    # 1. Hermes-managed Node.js (~/.hermes/node/)
    if [ -x "$HERMES_HOME/node/bin/node" ]; then
        local managed_ver
        managed_ver=$("$HERMES_HOME/node/bin/node" --version 2>/dev/null | sed 's/^v//')
        if [ "$managed_ver" = "$HERMES_NODE_TARGET_VERSION" ]; then
            export PATH="$HERMES_HOME/node/bin:$PATH"
            _nb_ok "Node $(node --version) found (Hermes-managed)"
            HERMES_NODE_AVAILABLE=true
            return 0
        else
            _nb_log "Managed Node.js is $managed_ver, but .nvmrc requires $HERMES_NODE_TARGET_VERSION. Updating..."
            if _nb_install_hermes_node; then
                HERMES_NODE_AVAILABLE=true
                return 0
            fi
        fi
    else
        # Not installed yet, try to install it first
        if _nb_install_hermes_node; then
            HERMES_NODE_AVAILABLE=true
            return 0
        fi
    fi

    # 2. Node already on PATH (validated against package.json engines)
    if _nb_have_modern_node; then
        _nb_ok "Node $(node --version) found on PATH (meets package.json engines requirement)"
        HERMES_NODE_AVAILABLE=true
        return 0
    fi

    # 3. Termux pkg (fallback)
    if _nb_try_termux_pkg; then
        HERMES_NODE_AVAILABLE=true
        return 0
    fi

    _nb_warn "Node.js install failed — TUI and browser tools will be unavailable."
    _nb_warn "Install manually: https://nodejs.org/en/download/"
    return 1
}
