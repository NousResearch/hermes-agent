#!/usr/bin/env bash
# ============================================================================
# scripts/lib/apexnodes-region-detect.sh
# ----------------------------------------------------------------------------
# Sourceable ApexNodes overlay: region self-detection + China mirror downgrade.
#
# This is the SINGLE SOURCE OF TRUTH for the "should this install use domestic
# (mainland-China) mirrors?" decision and, when yes, the mirror/COS env it sets.
# It is an ApexNodes overlay seam (see apex_overlay/README.md): the file lives in
# OUR namespace under scripts/lib/, which upstream Hermes never creates, so it
# has a zero-conflict merge surface. The upstream installer (scripts/install.sh)
# stays byte-for-byte upstream apart from one self-locating `source` of this lib
# plus a handful of tiny call sites — drastically shrinking the conflict surface
# on the hot install.sh file (6 upstream churn commits) when bumping the runtime.
#
# Mirrors scripts/lib/apexnodes-region-detect.ps1 for Windows. Keep the two in
# step: the region heuristic, precedence rules, env-var names, and COS layout
# are identical by design.
#
# Usage (from install.sh, which sources this only when the file is present):
#   source scripts/lib/apexnodes-region-detect.sh
#   apexnodes_resolve_region          # sets HERMES_CN_MIRRORS from region
#   apexnodes_apply_cn_mirror_env     # exports CN mirror env iff CN
# Then the COS download helpers (also defined here) are available:
#   apexnodes_install_uv_from_cos     # sets UV_CMD on success (CN only)
#   apexnodes_download_runtime_tarball# populates INSTALL_DIR (CN only)
#
# OFF by default: with HERMES_CN_MIRRORS unset/0 every function here is a no-op
# and the installer behaves byte-for-byte like upstream (curl|bash, CI, etc).
#
# Inputs read from the environment (set before sourcing / before calling):
#   HERMES_CN_MIRRORS         authoritative override (rule 1): 1=on, 0=off
#   APEXNODES_REGION          explicit region knob (rule 2): cn|global
#   HERMES_RUNTIME_COS_BASE   public-read COS base for the runtime tarball + uv
#   HERMES_HOME               install root (default $HOME/.hermes)
#   OS / INSTALL_COMMIT / BRANCH / INSTALL_DIR / UV_CMD / UV_VERSION
#                             provided by install.sh; the COS helpers read/write
#                             them so behavior matches the previous in-line code.
#
# The installer provides log_info / log_warn / log_success. When this lib is
# sourced standalone (e.g. a unit test) those may be absent, so we define
# no-op-safe fallbacks below — never let a missing logger break detection.
# ============================================================================

# Logger fallbacks: only define if the installer has not already provided them.
# `command -v` keeps these from clobbering install.sh's real loggers.
if ! command -v log_info >/dev/null 2>&1;    then log_info()    { echo "$*"; }      fi
if ! command -v log_warn >/dev/null 2>&1;    then log_warn()    { echo "$*" >&2; }  fi
if ! command -v log_success >/dev/null 2>&1; then log_success() { echo "$*"; }      fi

# ============================================================================
# ApexNodes region detection (decides whether CN mirror mode turns on)
# ============================================================================
# Goal: a fresh mainland-China machine should auto-pick domestic mirrors for any
# MISSING dependency, while everyone else keeps the upstream defaults byte-for-
# byte. This block only ever *decides a region* and, from it, sets
# HERMES_CN_MIRRORS — the "ApexNodes China mirror mode" function below (and the
# COS download helpers) remain the single source of truth for the actual mirror
# URLs. Reuse, not a second mirror table.
#
# Precedence (highest first):
#   1. HERMES_CN_MIRRORS already set in the env  -> respect it verbatim, skip
#      detection entirely. The packaged desktop (bootstrap-runner.cjs) and ops
#      overrides set this directly, so this keeps current behavior unchanged and
#      every existing test green.
#   2. APEXNODES_REGION=cn|global                -> explicit operator/user knob
#      (escape hatch + testability). cn => mirrors on, global => mirrors off.
#   3. neither set                              -> auto-detect mainland China
#      with the heuristic below; default to "global" (no mirrors) on any doubt,
#      because picking the wrong region only ever slows a download — it must
#      never break the first-run install path.
#
# The detection result is cached in $HERMES_HOME/.apexnodes-region so the
# per-stage bootstrap (each --stage runs in a fresh process) probes the network
# at most once instead of once per stage. Delete that file (or set
# APEXNODES_REGION) to re-decide.

# Lowercase helper that works on bash 3.2 (macOS /bin/bash lacks ${var,,}).
_an_lower() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }

# Region diagnostics go to STDERR, never stdout. stdout is the structured
# channel the bootstrap runner parses for --manifest / --stage --json frames;
# the runner forwards stderr as ordinary log lines, so the user still sees this.
_an_log() { echo "$1" >&2; }

# Cheap, offline first pass: is the machine's timezone plausibly mainland China?
# CST (UTC+8) covers all of China, but also HK / Taiwan / Singapore / Perth, so
# this is only a *gate* for the network probe below — never decisive on its own.
# Checked via the IANA zone name first (Asia/Shanghai, Asia/Urumqi) and then the
# numeric UTC offset as a fallback for minimal images with no zoneinfo names.
_an_timezone_suggests_cn() {
    local tz="${TZ:-}"
    if [ -z "$tz" ] && [ -r /etc/timezone ]; then
        tz="$(cat /etc/timezone 2>/dev/null)"
    fi
    if [ -z "$tz" ] && [ -L /etc/localtime ]; then
        # e.g. /usr/share/zoneinfo/Asia/Shanghai -> Asia/Shanghai
        tz="$(readlink /etc/localtime 2>/dev/null | sed 's#.*/zoneinfo/##')"
    fi
    case "$tz" in
        Asia/Shanghai|Asia/Urumqi|Asia/Chongqing|Asia/Harbin|Asia/Kashgar|PRC)
            return 0 ;;
    esac
    # Fallback: numeric offset. date %z is +0800 across CST; combined with the
    # network probe this still isolates the mainland from non-CN +0800 regions.
    local off
    off="$(date +%z 2>/dev/null)"
    [ "$off" = "+0800" ] && return 0
    return 1
}

# Decisive pass: race a domestic endpoint against a foreign one with short
# timeouts. In mainland China the foreign endpoint is typically blocked or very
# slow while the domestic one answers immediately; elsewhere both answer (or the
# foreign one answers and the domestic one may be slower). We only classify CN
# when the domestic probe SUCCEEDS *and* the foreign probe FAILS — a deliberately
# conservative AND so transient flakiness biases toward "global" (defaults).
# Uses HEAD requests; no payload is downloaded. Returns 0 for CN, 1 otherwise.
_an_network_suggests_cn() {
    command -v curl >/dev/null 2>&1 || return 1
    # Domestic anchor: npmmirror is one of the mirrors we'd actually use and is
    # reachable nationwide. Foreign anchor: the npm registry we'd otherwise hit.
    local cn_url="https://registry.npmmirror.com/"
    local foreign_url="https://registry.npmjs.org/"
    local cn_ok=1 foreign_ok=1
    curl -fsS -I --max-time 4 "$cn_url" >/dev/null 2>&1 && cn_ok=0
    curl -fsS -I --max-time 4 "$foreign_url" >/dev/null 2>&1 && foreign_ok=0
    # CN iff domestic reachable AND foreign unreachable.
    [ "$cn_ok" -eq 0 ] && [ "$foreign_ok" -ne 0 ] && return 0
    return 1
}

# Resolve the region into HERMES_CN_MIRRORS. No-op when HERMES_CN_MIRRORS is
# already set (precedence rule 1). Writes/reads the cache file otherwise.
apexnodes_resolve_region() {
    # Rule 1: explicit HERMES_CN_MIRRORS wins; do not touch it, do not probe.
    if [ -n "${HERMES_CN_MIRRORS:-}" ]; then
        return 0
    fi

    # Rule 2: explicit region knob.
    local region
    region="$(_an_lower "${APEXNODES_REGION:-}")"
    case "$region" in
        cn|china|mainland)
            export HERMES_CN_MIRRORS=1
            _an_log "→ ApexNodes region: cn (from APEXNODES_REGION) — using China mirrors"
            return 0 ;;
        global|intl|international|foreign|row)
            export HERMES_CN_MIRRORS=0
            _an_log "→ ApexNodes region: global (from APEXNODES_REGION) — using default sources"
            return 0 ;;
        "") : ;;  # fall through to auto-detect
        *)
            _an_log "⚠ Unknown APEXNODES_REGION='${APEXNODES_REGION}' (expected cn|global) — auto-detecting" ;;
    esac

    # Rule 3: auto-detect. Reuse a cached decision from an earlier stage process.
    local cache="${HERMES_HOME:-$HOME/.hermes}/.apexnodes-region"
    if [ -r "$cache" ]; then
        local cached
        cached="$(_an_lower "$(cat "$cache" 2>/dev/null)")"
        case "$cached" in
            cn)     export HERMES_CN_MIRRORS=1; return 0 ;;
            global) export HERMES_CN_MIRRORS=0; return 0 ;;
        esac
    fi

    # No cache: decide now. Gate the network probe on the cheap timezone hint so
    # the common (non-CN) case usually skips the probe entirely, but still probe
    # when the timezone is unknown so headless/UTC images aren't misclassified.
    local detected="global"
    if _an_timezone_suggests_cn; then
        if _an_network_suggests_cn; then
            detected="cn"
        fi
    elif [ -z "${TZ:-}" ] && [ ! -e /etc/timezone ] && [ ! -L /etc/localtime ]; then
        # Timezone genuinely undeterminable (bare container) — fall back to the
        # network probe alone rather than assuming global.
        if _an_network_suggests_cn; then
            detected="cn"
        fi
    fi

    # Persist for sibling stage processes (best-effort; never fail the install).
    mkdir -p "${HERMES_HOME:-$HOME/.hermes}" 2>/dev/null || true
    printf '%s\n' "$detected" > "$cache" 2>/dev/null || true

    if [ "$detected" = "cn" ]; then
        export HERMES_CN_MIRRORS=1
        _an_log "→ ApexNodes region: cn (auto-detected) — using China mirrors"
        _an_log "  (override with APEXNODES_REGION=global if this is wrong)"
    else
        export HERMES_CN_MIRRORS=0
        # Quiet on the global path to keep upstream/CI output byte-clean.
    fi
}

# ============================================================================
# ApexNodes China mirror mode (opt-in via HERMES_CN_MIRRORS=1)
# ============================================================================
# OFF by default: with the flag unset apexnodes_apply_cn_mirror_env does nothing
# and the installer behaves byte-for-byte like upstream (curl|bash one-liner,
# CI, etc).
#
# The packaged ApexNodes desktop sets HERMES_CN_MIRRORS=1 (and, once provisioned,
# HERMES_RUNTIME_COS_BASE) from electron/bootstrap-runner.cjs so a fresh mainland-
# China machine can install without reaching github.com / pypi.org /
# registry.npmjs.org directly. The split is deliberate:
#   * Our runtime source + uv (no public CN mirror exists) come from our own
#     public-read COS bucket — see apexnodes_download_runtime_tarball /
#     apexnodes_install_uv_from_cos.
#   * Every public third-party dependency uses an established CN mirror below.
# Each value uses ${VAR:-default} so an operator can override any single mirror
# via the real environment without editing this script.
apexnodes_cn_enabled() { [ "${HERMES_CN_MIRRORS:-0}" = "1" ]; }

# Export the CN mirror env (no-op unless CN mode is on). Idempotent: every value
# uses ${VAR:-default} so a pre-set operator override is preserved.
apexnodes_apply_cn_mirror_env() {
    apexnodes_cn_enabled || return 0
    # Python package index → Tsinghua TUNA (PyPI mirror).
    export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
    export UV_INDEX_URL="${UV_INDEX_URL:-$UV_DEFAULT_INDEX}"
    export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
    # uv-managed CPython (astral python-build-standalone) → npmmirror binary mirror.
    export UV_PYTHON_INSTALL_MIRROR="${UV_PYTHON_INSTALL_MIRROR:-https://registry.npmmirror.com/-/binary/python-build-standalone}"
    # npm registry + Electron binaries → npmmirror.
    export npm_config_registry="${npm_config_registry:-https://registry.npmmirror.com}"
    export ELECTRON_MIRROR="${ELECTRON_MIRROR:-https://npmmirror.com/mirrors/electron/}"
    # Node.js dist tarballs → npmmirror binary mirror (consumed by install_node).
    export HERMES_NODE_DIST_BASE="${HERMES_NODE_DIST_BASE:-https://registry.npmmirror.com/-/binary/node}"
}

# ============================================================================
# CN-mode COS download helpers
# ============================================================================

# Map the current OS/arch to an astral uv release target triple (the names uv
# publishes its tarballs under, e.g. uv-aarch64-apple-darwin.tar.gz). Used to
# build the COS mirror download URL in CN mode. Returns non-zero (no output)
# for unsupported platforms so the caller falls back to the astral installer.
_uv_target_triple() {
    local arch
    arch="$(uname -m)"
    case "$OS" in
        macos)
            case "$arch" in
                arm64|aarch64) echo "aarch64-apple-darwin" ;;
                x86_64) echo "x86_64-apple-darwin" ;;
                *) return 1 ;;
            esac
            ;;
        linux)
            case "$arch" in
                x86_64) echo "x86_64-unknown-linux-gnu" ;;
                aarch64|arm64) echo "aarch64-unknown-linux-gnu" ;;
                *) return 1 ;;
            esac
            ;;
        *) return 1 ;;
    esac
}

# CN mode: fetch a prebuilt uv from our public-read COS bucket instead of the
# astral.sh installer, which downloads the binary from github.com (blocked in
# mainland China). The publish script (scripts/publish-runtime-tarball.sh)
# mirrors uv-<triple>.tar.gz next to the runtime tarball. On any failure we
# return non-zero so install_uv falls through to the astral path. Sets UV_CMD.
apexnodes_install_uv_from_cos() {
    apexnodes_cn_enabled || return 1
    [ -n "${HERMES_RUNTIME_COS_BASE:-}" ] || return 1
    command -v curl >/dev/null 2>&1 || return 1

    local triple
    triple="$(_uv_target_triple)" || return 1
    [ -n "$triple" ] || return 1

    local url="${HERMES_RUNTIME_COS_BASE%/}/uv-${triple}.tar.gz"
    local tmp
    tmp="$(mktemp -d 2>/dev/null || echo "/tmp/hermes-uv.$$")"
    mkdir -p "$tmp" "$HERMES_HOME/bin"

    log_info "Fetching uv from COS mirror: $url"
    if ! curl -fsSL --max-time 120 "$url" -o "$tmp/uv.tar.gz"; then
        log_warn "COS uv download failed ($url) — will try the astral.sh installer"
        rm -rf "$tmp"
        return 1
    fi
    if ! tar xzf "$tmp/uv.tar.gz" -C "$tmp" 2>/dev/null; then
        log_warn "COS uv tarball could not be extracted — will try the astral.sh installer"
        rm -rf "$tmp"
        return 1
    fi

    # uv tarballs unpack into uv-<triple>/{uv,uvx}; locate the binaries by name.
    local found foundx
    found="$(find "$tmp" -type f -name uv 2>/dev/null | head -1)"
    if [ -z "$found" ]; then
        log_warn "uv binary not found inside COS tarball — will try the astral.sh installer"
        rm -rf "$tmp"
        return 1
    fi
    cp "$found" "$HERMES_HOME/bin/uv"
    chmod +x "$HERMES_HOME/bin/uv"
    foundx="$(find "$tmp" -type f -name uvx 2>/dev/null | head -1)"
    if [ -n "$foundx" ]; then
        cp "$foundx" "$HERMES_HOME/bin/uvx"
        chmod +x "$HERMES_HOME/bin/uvx"
    fi
    rm -rf "$tmp"

    if [ ! -x "$HERMES_HOME/bin/uv" ]; then
        return 1
    fi
    UV_CMD="$HERMES_HOME/bin/uv"
    UV_VERSION=$($UV_CMD --version 2>/dev/null)
    log_success "Managed uv installed from COS mirror ($UV_VERSION)"
    return 0
}

# CN mode: download the pinned runtime source as a tarball from our public-read
# COS bucket instead of git-cloning github.com (blocked/slow in mainland China).
# The tarball is `git archive --prefix=hermes-agent/` of the pinned upstream
# commit — a clean source tree with NO .git directory. Keyed by the pinned
# commit (preferred) or branch so the COS object matches the build stamp.
# Returns 0 with INSTALL_DIR populated on success; non-zero (and INSTALL_DIR
# removed) on any failure so clone_repo falls back to a normal git clone.
apexnodes_download_runtime_tarball() {
    apexnodes_cn_enabled || return 1
    [ -n "${HERMES_RUNTIME_COS_BASE:-}" ] || return 1
    command -v curl >/dev/null 2>&1 || return 1

    local key="${INSTALL_COMMIT:-$BRANCH}"
    [ -n "$key" ] || return 1

    local url="${HERMES_RUNTIME_COS_BASE%/}/hermes-agent-${key}.tar.gz"
    local tmp
    tmp="$(mktemp -d 2>/dev/null || echo "/tmp/hermes-src.$$")"
    mkdir -p "$tmp"

    log_info "Downloading runtime source from COS mirror: $url"
    if ! curl -fsSL --max-time 300 "$url" -o "$tmp/runtime.tar.gz"; then
        log_warn "COS runtime download failed ($url) — falling back to git clone"
        rm -rf "$tmp"
        return 1
    fi

    mkdir -p "$INSTALL_DIR"
    # Archive is built with --prefix=hermes-agent/, so strip the leading dir.
    if ! tar xzf "$tmp/runtime.tar.gz" -C "$INSTALL_DIR" --strip-components=1; then
        log_warn "COS runtime tarball could not be extracted — falling back to git clone"
        rm -rf "$tmp" "$INSTALL_DIR"
        return 1
    fi
    rm -rf "$tmp"

    if [ ! -f "$INSTALL_DIR/pyproject.toml" ]; then
        log_warn "COS runtime tarball missing pyproject.toml — falling back to git clone"
        rm -rf "$INSTALL_DIR"
        return 1
    fi

    log_success "Runtime source ready from COS mirror ($key)"
    return 0
}
