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
#   HERMES_NODE_MIN_VERSION        (default: 20)       — accepted on PATH
#   HERMES_NODE_TARGET_MAJOR       (default: 22)       — installed when we install
#   HERMES_NODE_TARGET_MIN_VERSION (default: 22.22.0)  — floor a managed tree
#                                                        must clear; keep in
#                                                        sync with the root
#                                                        package.json's
#                                                        engines.node
#   HERMES_HOME                    (default: $HOME/.hermes)
# ============================================================================

HERMES_NODE_MIN_VERSION="${HERMES_NODE_MIN_VERSION:-20}"
HERMES_NODE_TARGET_MAJOR="${HERMES_NODE_TARGET_MAJOR:-22}"
HERMES_NODE_TARGET_MIN_VERSION="${HERMES_NODE_TARGET_MIN_VERSION:-22.22.0}"
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

# Echo "<major> <minor> <patch>" for a dotted version ("v22.21.1", "22.22.0",
# "22"). Drops the `v` prefix and any prerelease suffix, pads missing
# components with 0. Returns non-zero (and echoes nothing) when a component is
# not a plain integer.
_nb_version_triple() {
    local ver rest major minor patch part
    ver="${1#v}"
    ver="${ver%%-*}"

    major="${ver%%.*}"
    case "$ver"  in *.*) rest="${ver#*.}"  ;; *) rest="" ;; esac
    minor="${rest%%.*}"
    case "$rest" in *.*) rest="${rest#*.}" ;; *) rest="" ;; esac
    patch="${rest%%.*}"

    [ -n "$minor" ] || minor=0
    [ -n "$patch" ] || patch=0
    for part in "$major" "$minor" "$patch"; do
        case "$part" in ''|*[!0-9]*) return 1 ;; esac
    done
    printf '%s %s %s\n' "$major" "$minor" "$patch"
}

# True when version $1 is strictly older than version $2. Fails closed: an
# unreadable version is never reported as older, so we don't redownload a tree
# just because we couldn't parse its `--version`.
_nb_version_lt() {
    local lhs rhs
    lhs="$(_nb_version_triple "$1")" || return 1
    rhs="$(_nb_version_triple "$2")" || return 1
    # Intentional word splitting: each side is exactly three integer fields.
    # shellcheck disable=SC2086
    set -- $lhs $rhs
    if [ "$1" -ne "$4" ]; then [ "$1" -lt "$4" ]; return; fi
    if [ "$2" -ne "$5" ]; then [ "$2" -lt "$5" ]; return; fi
    [ "$3" -lt "$6" ]
}

# Where to symlink node/npm/npx so they land on PATH.
# Mirrors get_command_link_dir() from install.sh: root FHS → /usr/local/bin,
# Termux → $PREFIX/bin, otherwise ~/.local/bin.
_nb_get_link_dir() {
    if _nb_is_termux && [ -n "${PREFIX:-}" ]; then
        echo "$PREFIX/bin"
    elif [ "$(id -u)" = 0 ] && [ "$(uname -s)" = "Linux" ]; then
        echo "/usr/local/bin"
    else
        echo "$HOME/.local/bin"
    fi
}

# Redirect a Hermes-managed Node's `npm install -g` to the command link dir
# (already on PATH) instead of the default $HERMES_HOME/node/bin, which is off
# PATH and wiped on every Node upgrade. Scoped to the managed Node via its
# prefix-local global npmrc; the user's other Node installs / ~/.npmrc are
# untouched. Idempotent no-op when there's no managed npm.
_nb_configure_npm_prefix() {
    [ -x "$HERMES_HOME/node/bin/npm" ] || return 0
    local _link_dir
    _link_dir="$(_nb_get_link_dir)"
    mkdir -p "$HERMES_HOME/node/etc"
    printf 'prefix=%s\n' "$(dirname "$_link_dir")" > "$HERMES_HOME/node/etc/npmrc"
}

_nb_node_major() {
    local v
    v=$(node --version 2>/dev/null | sed 's/^v//' | cut -d. -f1)
    [[ "$v" =~ ^[0-9]+$ ]] && echo "$v" || echo 0
}

# The npm range the checkout's root package.json demands. Read from the
# manifest rather than duplicated here so the two can never drift; falls back
# to the current floor when the manifest is unreadable (vendored copy of this
# script, stripped install tree).
_nb_npm_range() {
    if [ -n "${HERMES_NPM_TARGET_RANGE:-}" ]; then
        printf '%s\n' "$HERMES_NPM_TARGET_RANGE"
        return 0
    fi
    local repo_root manifest range
    repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)"
    manifest="$repo_root/package.json"
    if [ -r "$manifest" ]; then
        # sed, not node: this runs before a usable node is guaranteed.
        range=$(sed -n '/"engines"/,/}/p' "$manifest" \
            | sed -n 's/.*"npm"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
            | head -1)
        if [ -n "$range" ]; then
            printf '%s\n' "$range"
            return 0
        fi
    fi
    printf '>=12.0.0\n'
}

# Upgrade the managed tree's bundled npm into the checkout's engines.npm range.
#
# The nodejs.org tarball ships whatever npm that Node major bundles — Node
# 26.5.1 bundles npm 11.17.0, one minor below our own `engines.npm` floor of
# >=12. With `engine-strict=true` in the repo .npmrc that is fatal, not a
# warning, so a brand-new install died at the first `npm ci` with EBADENGINE.
# The Python side recovers through hermes_cli/npm_engine.py; the installer path
# had no such rung, so provision the right npm here instead of reacting later.
#
# Three details are load-bearing, all mirroring upgrade_managed_npm():
#   - a temp cwd, so the checkout's own .npmrc (engine-strict, min-release-age)
#     does not gate the very upgrade meant to satisfy it;
#   - npm_config_min_release_age=0, which also neutralises a user ~/.npmrc;
#   - an explicit --prefix at the managed tree, because
#     _nb_configure_npm_prefix wrote prefix=~/.local into its etc/npmrc, and
#     without the override this installs a second npm elsewhere while the
#     managed tree stays stale.
#
# Best-effort: a failure here leaves a working Node with an old npm, which is
# strictly better than no Node at all, and npm_engine.py still covers the
# EBADENGINE that follows.
_nb_ensure_bundled_npm_range() {
    local npm_bin="$HERMES_HOME/node/bin/npm"
    [ -x "$npm_bin" ] || return 0

    local range have want
    range="$(_nb_npm_range)"
    [ -n "$range" ] || return 0

    # Skip the network round-trip when the bundled npm already satisfies the
    # range. Only the ">=N" shape we actually author is checked; anything more
    # exotic falls through to letting npm itself decide.
    if [[ "$range" =~ ^\>=([0-9]+) ]]; then
        want="${BASH_REMATCH[1]}"
        have=$("$npm_bin" --version 2>/dev/null | cut -d. -f1)
        if [[ "$have" =~ ^[0-9]+$ ]] && [ "$have" -ge "$want" ]; then
            return 0
        fi
    fi

    _nb_log "Upgrading bundled npm to satisfy $range..."
    local tmp_cwd
    tmp_cwd=$(mktemp -d)
    if (
        cd "$tmp_cwd" || exit 1
        CI=1 npm_config_min_release_age=0 \
            "$npm_bin" install --global \
                --prefix "$HERMES_HOME/node" \
                "npm@$range" \
                --no-fund --no-audit --progress=false >/dev/null 2>&1
    ); then
        rm -rf "$tmp_cwd"
        _nb_ok "npm $("$npm_bin" --version 2>/dev/null) installed"
        return 0
    fi

    rm -rf "$tmp_cwd"
    _nb_warn "Could not upgrade bundled npm to $range — \`npm ci\` may fail with EBADENGINE."
    _nb_warn "Fix manually: npm install -g --prefix \"$HERMES_HOME/node\" npm@\"$range\""
    return 1
}

_nb_have_modern_node() {
    command -v node >/dev/null 2>&1 || return 1
    [ "$(_nb_node_major)" -ge "$HERMES_NODE_MIN_VERSION" ]
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

    local _link_dir
    _link_dir="$(_nb_get_link_dir)"
    # HERMES_NODE_SKIP_LINKS=1: the caller only wants the private managed tree
    # (e.g. the EBADENGINE recovery provisioning a runtime alongside a working
    # system Node). Skipping the links keeps the user's own node/npm first on
    # PATH instead of shadowing them with ours.
    if [ "${HERMES_NODE_SKIP_LINKS:-0}" != "1" ]; then
        mkdir -p "$_link_dir"
        ln -sf "$HERMES_HOME/node/bin/node" "$_link_dir/node"
        ln -sf "$HERMES_HOME/node/bin/npm"  "$_link_dir/npm"
        ln -sf "$HERMES_HOME/node/bin/npx"  "$_link_dir/npx"
    fi

    _nb_configure_npm_prefix

    export PATH="$HERMES_HOME/node/bin:$PATH"

    _nb_have_modern_node || return 1
    _nb_ok "Node $(node --version) installed to $HERMES_HOME/node/"
    # The tarball's bundled npm is usually below the repo's engines.npm floor.
    # Best-effort: an old npm still beats no Node.
    _nb_ensure_bundled_npm_range || true
    return 0
}

# ---------------------------------------------------------------------------
# Heal a broken Hermes-managed Node tree (partial upgrade / missing lib/)
# ---------------------------------------------------------------------------

_nb_managed_tool_broken() {
    local tool="$1"
    local probe
    for probe in \
        "$HERMES_HOME/node/bin/$tool" \
        "$HERMES_HOME/node/${tool}.exe" \
        "$HERMES_HOME/node/$tool"; do
        if [ -x "$probe" ] || [ -f "$probe" ]; then
            if ! "$probe" --version >/dev/null 2>&1; then
                return 0
            fi
        fi
    done
    return 1
}

# The managed node runs but is below the target — either below
# HERMES_NODE_TARGET_MAJOR (an old tree from a previous major) or below the
# full HERMES_NODE_TARGET_MIN_VERSION floor within the current major. The
# second case is the one that bites: a 22.21.x tree looks current by major but
# `.npmrc`'s engine-strict=true makes every `npm ci` reject it with
# EBADENGINE, so `hermes update` dies while the heal declines to fire.
# Outdated heals the same way broken does, so existing users get upgraded on
# the next heal probe, not just on a full installer re-run. Mirrors
# _managed_node_tree_outdated() in hermes_constants.py.
_nb_managed_node_outdated() {
    local probe ver major
    for probe in "$HERMES_HOME/node/bin/node" "$HERMES_HOME/node/node"; do
        [ -x "$probe" ] || continue
        ver="$("$probe" --version 2>/dev/null)" || return 1
        major="${ver#v}"; major="${major%%.*}"
        case "$major" in ''|*[!0-9]*) return 1 ;; esac
        [ "$major" -lt "$HERMES_NODE_TARGET_MAJOR" ] && return 0
        _nb_version_lt "$ver" "$HERMES_NODE_TARGET_MIN_VERSION" && return 0
        return 1
    done
    return 1
}

_nb_managed_node_needs_heal() {
    local tool
    for tool in node npm npx; do
        if _nb_managed_tool_broken "$tool"; then
            return 0
        fi
    done
    _nb_managed_node_outdated
}

# Redownload the pinned nodejs.org tarball when a managed tree exists but
# node/npm/npx fail a --version probe. No-op when the tree is healthy or
# absent. Used by hermes_constants.find_hermes_node_executable() and safe
# to call from install reruns.
heal_managed_node() {
    [ -d "$HERMES_HOME/node" ] || return 1
    if ! _nb_managed_node_needs_heal; then
        return 0
    fi
    _nb_log "Hermes-managed Node is broken — redownloading to $HERMES_HOME/node/..."
    _nb_install_bundled_node
}

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

ensure_node() {
    HERMES_NODE_AVAILABLE=false

    # Repair pre-existing managed installs where `npm install -g` lands off
    # PATH. No-op when there's no managed Node, so it's safe to run first.
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
            # A tree from an older install still carries that Node major's
            # bundled npm, and the upgrade in _nb_install_bundled_node is
            # best-effort — one offline install leaves an at-target tree
            # stranded below engines.npm forever, since heal only fires for a
            # *broken* tree. Mirrors Update-ManagedNpm's reuse-path call in
            # install.ps1. No-ops on a probe when the npm is already in range.
            _nb_ensure_bundled_npm_range || true
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
