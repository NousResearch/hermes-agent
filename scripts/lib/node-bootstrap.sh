#!/usr/bin/env bash
# ============================================================================
# scripts/lib/node-bootstrap.sh
# ----------------------------------------------------------------------------
# Sourceable helper: ensure Node.js >= MIN_VERSION is available for the TUI
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
#   HERMES_NODE_MIN_VERSION   (default: 20)   — accepted on PATH
#   HERMES_NODE_TARGET_MAJOR  (default: 22)   — installed when we install
#   HERMES_HOME               (default: $HOME/.hermes)
# ============================================================================

HERMES_NODE_MIN_VERSION="${HERMES_NODE_MIN_VERSION:-20}"
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

# Redirect a Hermes-managed Node's `npm install -g` into $HERMES_HOME/node so
# global bins stay self-contained under $HERMES_HOME/node/bin, which Hermes
# already uses for its own subprocess PATH, instead of shadowing the user's
# node/npm/npx in ~/.local/bin. Scoped to the managed Node via its prefix-local
# global npmrc; the user's other Node installs / ~/.npmrc are untouched.
# Idempotent no-op when there's no managed npm.
_nb_configure_npm_prefix() {
    [ -x "$HERMES_HOME/node/bin/npm" ] || return 0
    mkdir -p "$HERMES_HOME/node/etc"
    printf 'prefix=%s\n' "$HERMES_HOME/node" > "$HERMES_HOME/node/etc/npmrc"
}

_nb_link_points_into_hermes_node() {
    local _link="$1" _target _target_abs _node_dir
    [ -L "$_link" ] || return 1
    _target="$(readlink "$_link" 2>/dev/null)" || return 1
    case "$_target" in
        /*) _target_abs="$_target" ;;
        *)  _target_abs="$(dirname "$_link")/$_target" ;;
    esac

    _node_dir="$HERMES_HOME/node"
    case "$_target_abs" in
        "$_node_dir"|"$_node_dir"/*) return 0 ;;
    esac

    if [ -d "$_node_dir" ] && [ -e "$_target_abs" ]; then
        local _resolved_target _resolved_node
        _resolved_target="$(cd "$(dirname "$_target_abs")" 2>/dev/null && pwd -P)/$(basename "$_target_abs")"
        _resolved_node="$(cd "$_node_dir" 2>/dev/null && pwd -P)"
        case "$_resolved_target" in
            "$_resolved_node"|"$_resolved_node"/*) return 0 ;;
        esac
    fi
    return 1
}

_nb_remove_legacy_node_links() {
    local _dirs=() _dir _name _link
    _dirs+=("$HOME/.local/bin")
    [ -n "${PREFIX:-}" ] && _dirs+=("$PREFIX/bin")
    _dirs+=("/usr/local/bin")

    for _dir in "${_dirs[@]}"; do
        [ -d "$_dir" ] || continue
        for _name in node npm npx; do
            _link="$_dir/$_name"
            if _nb_link_points_into_hermes_node "$_link"; then
                rm -f "$_link" || true
            fi
        done
    done
}

_nb_node_major() {
    local v
    v=$(node --version 2>/dev/null | sed 's/^v//' | cut -d. -f1)
    [[ "$v" =~ ^[0-9]+$ ]] && echo "$v" || echo 0
}

_nb_have_modern_node() {
    command -v node >/dev/null 2>&1 || return 1
    [ "$(_nb_node_major)" -ge "$HERMES_NODE_MIN_VERSION" ]
}

_nb_node_bin_major() {
    local node_bin="$1" v
    [ -x "$node_bin" ] || return 1
    v=$("$node_bin" --version 2>/dev/null | sed 's/^v//' | cut -d. -f1)
    [[ "$v" =~ ^[0-9]+$ ]] && echo "$v" || echo 0
}

_nb_activate_node_bin() {
    local node_bin="$1" label="$2" major node_dir
    major="$(_nb_node_bin_major "$node_bin")" || return 1
    [ "$major" -ge "$HERMES_NODE_MIN_VERSION" ] || return 1
    node_dir="$(dirname "$node_bin")"
    case ":$PATH:" in
        *":$node_dir:"*) ;;
        *) export PATH="$node_dir:$PATH" ;;
    esac
    _nb_ok "Node $("$node_bin" --version) found ($label)"
    return 0
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

_nb_try_existing_nvm() {
    local nvm_root="${NVM_DIR:-$HOME/.nvm}" node_bin
    for node_bin in "$nvm_root"/versions/node/*/bin/node; do
        [ -e "$node_bin" ] || continue
        _nb_activate_node_bin "$node_bin" "nvm" && return 0
    done
    return 1
}

_nb_try_existing_fnm() {
    local fnm_root node_bin
    for fnm_root in "${FNM_DIR:-$HOME/.local/share/fnm}" "$HOME/.fnm"; do
        for node_bin in "$fnm_root"/node-versions/*/installation/bin/node "$fnm_root"/node-versions/*/bin/node; do
            [ -e "$node_bin" ] || continue
            _nb_activate_node_bin "$node_bin" "fnm" && return 0
        done
    done
    return 1
}

_nb_try_existing_volta() {
    local volta_root="${VOLTA_HOME:-$HOME/.volta}"
    _nb_activate_node_bin "$volta_root/bin/node" "Volta"
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

    # Remove legacy node/npm/npx shims that older Hermes installers placed in
    # ~/.local/bin or other command dirs. Only links into this HERMES_HOME are
    # removed; user-managed nvm/fnm/Volta binaries are left untouched.
    _nb_remove_legacy_node_links

    # Keep pre-existing managed installs self-contained. No-op when there's no
    # managed Node, so it's safe to run first.
    _nb_configure_npm_prefix

    if _nb_have_modern_node; then
        _nb_ok "Node $(node --version) found"
        HERMES_NODE_AVAILABLE=true
        return 0
    fi

    # Prefer explicit user version-manager installs over Hermes's private copy.
    _nb_try_existing_fnm   && { HERMES_NODE_AVAILABLE=true; return 0; }
    _nb_try_existing_volta && { HERMES_NODE_AVAILABLE=true; return 0; }
    _nb_try_existing_nvm   && { HERMES_NODE_AVAILABLE=true; return 0; }

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
