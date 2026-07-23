#!/data/data/com.termux/files/usr/bin/bash
# Native Android/Termux installer for Hermes Agent.
# This path intentionally uses Termux packages + uv directly; it does not use
# proot, Ubuntu, pip, or a global third-party Python package index.
set -euo pipefail

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
INSTALL_DIR="${HERMES_INSTALL_DIR:-$HERMES_HOME/hermes-agent}"
BRANCH="main"
INSTALL_COMMIT=""
RUN_SETUP=true
SKIP_BROWSER=false
NO_SKILLS=false
NON_INTERACTIVE=false
USE_VENV=true
PYTHON_BIN=""
ANDROID_BUILD_API=24
REPO_URL="${HERMES_REPO_URL:-https://github.com/NousResearch/hermes-agent.git}"
# The fallback is locked by both release coordinates and content digest. A
# replaced or corrupted release asset is rejected before package extraction.
readonly PINNED_PYTHON_VERSION="3.13.14"
readonly PINNED_PYTHON_RELEASE="termux-aarch64-20260719.9.1"
readonly PINNED_PYTHON_ASSET="python_3.13.14_aarch64.deb"
readonly PINNED_PYTHON_SHA256="42376a2a47e50048cb7eca2d0f442fc1895fbca2aee2dee3d2fd82728ea1bd80"
readonly PINNED_PYTHON_URL="https://github.com/adybag14-cyber/termux-python/releases/download/${PINNED_PYTHON_RELEASE}/${PINNED_PYTHON_ASSET}"

log() { printf '\033[0;36m→\033[0m %s\n' "$*"; }
ok() { printf '\033[0;32m✓\033[0m %s\n' "$*"; }
warn() { printf '\033[0;33m⚠\033[0m %s\n' "$*" >&2; }
die() { printf '\033[0;31m✗\033[0m %s\n' "$*" >&2; exit 1; }

is_termux() {
    [ -n "${TERMUX_VERSION:-}" ] || [[ "${PREFIX:-}" == *"com.termux/files/usr"* ]]
}

usage() {
    cat <<'USAGE'
Hermes Agent native Termux installer

Usage: install-termux.sh [OPTIONS]
  --branch NAME          Git branch to install (default: main)
  --commit SHA           Pin the checkout to an exact commit
  --dir PATH             Installation directory
  --hermes-home PATH     Hermes data directory
  --python PATH          CPython 3.11, 3.12, or 3.13 interpreter
  --android-api-level N  Android wheel build target (default: 24)
  --skip-setup           Skip API/provider setup
  --skip-browser         Skip optional Node browser dependency install
  --no-skills            Do not seed bundled skills
  --non-interactive      Never prompt
  --no-venv              Unsupported on native Termux (isolated venv required)
  -h, --help             Show this help
USAGE
}

parse_args() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --branch|-Branch)
                [ "$#" -ge 2 ] || die "$1 requires a value"
                BRANCH="$2"
                shift 2
                ;;
            --commit|-Commit)
                [ "$#" -ge 2 ] || die "$1 requires a value"
                INSTALL_COMMIT="$2"
                shift 2
                ;;
            --dir)
                [ "$#" -ge 2 ] || die "$1 requires a value"
                INSTALL_DIR="$2"
                shift 2
                ;;
            --hermes-home)
                [ "$#" -ge 2 ] || die "$1 requires a value"
                HERMES_HOME="$2"
                shift 2
                ;;
            --python)
                [ "$#" -ge 2 ] || die "$1 requires a value"
                PYTHON_BIN="$2"
                shift 2
                ;;
            --android-api-level)
                [ "$#" -ge 2 ] || die "$1 requires a value"
                ANDROID_BUILD_API="$2"
                shift 2
                ;;
            --skip-setup) RUN_SETUP=false; shift ;;
            --skip-browser|--no-playwright) SKIP_BROWSER=true; shift ;;
            --no-skills) NO_SKILLS=true; shift ;;
            --non-interactive|-NonInteractive) NON_INTERACTIVE=true; shift ;;
            --no-venv) USE_VENV=false; shift ;;
            -h|--help) usage; exit 0 ;;
            --manifest|-Manifest|--stage|-Stage|--json|-Json|--include-desktop|-IncludeDesktop|--ensure|--postinstall)
                die "$1 is not supported by the native Termux installer"
                ;;
            *) die "Unknown option: $1" ;;
        esac
    done
    [ "$USE_VENV" = true ] || die \
        "Native Termux installs require an isolated venv; remove --no-venv"
}

python_is_supported() {
    local candidate="$1"
    [ -x "$candidate" ] || return 1
    "$candidate" -c \
        'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info < (3, 14) else 1)' \
        >/dev/null 2>&1
}

python_is_immutable_target() {
    local candidate="$1"
    [ -x "$candidate" ] || return 1
    "$candidate" -c \
        'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 13) else 1)' \
        >/dev/null 2>&1
}

find_immutable_python() {
    local candidate resolved
    for candidate in python3.13 python; do
        resolved="$(command -v "$candidate" 2>/dev/null || true)"
        if [ -n "$resolved" ] && python_is_immutable_target "$resolved"; then
            printf '%s\n' "$resolved"
            return 0
        fi
    done
    return 1
}

find_supported_python() {
    local candidate resolved
    if [ -n "$PYTHON_BIN" ]; then
        resolved="$PYTHON_BIN"
        command -v "$resolved" >/dev/null 2>&1 && resolved="$(command -v "$resolved")"
        python_is_supported "$resolved" || die \
            "--python must point to CPython 3.11, 3.12, or 3.13: $resolved"
        printf '%s\n' "$resolved"
        return 0
    fi

    for candidate in python3.13 python3.12 python3.11 python; do
        resolved="$(command -v "$candidate" 2>/dev/null || true)"
        if [ -n "$resolved" ] && python_is_supported "$resolved"; then
            printf '%s\n' "$resolved"
            return 0
        fi
    done
    return 1
}

termux_architecture() {
    dpkg --print-architecture 2>/dev/null || uname -m
}

install_pinned_python313() {
    command -v curl >/dev/null 2>&1 || die "curl is required to download Python"
    command -v dpkg >/dev/null 2>&1 || die "dpkg is required to inspect Python"
    command -v dpkg-deb >/dev/null 2>&1 || die "dpkg-deb is required to extract Python"
    command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required to verify Python"

    local architecture
    architecture="$(termux_architecture)"
    case "$architecture" in
        aarch64|arm64) ;;
        *) die "The pinned Termux Python build supports aarch64 only, not $architecture" ;;
    esac

    log "No CPython 3.13 interpreter is available for the immutable wheel target"
    log "Installing pinned Python $PINNED_PYTHON_VERSION side-by-side"

    local tmp deb staged_prefix package version package_arch
    tmp="$(mktemp -d "${TMPDIR:-$PREFIX/tmp}/hermes-python313.XXXXXX")"
    deb="$tmp/$PINNED_PYTHON_ASSET"
    if ! curl -fL --retry 3 --retry-all-errors \
        "$PINNED_PYTHON_URL" -o "$deb"; then
        rm -rf "$tmp"
        die "Failed to download pinned Python $PINNED_PYTHON_VERSION"
    fi

    if ! printf '%s  %s\n' "$PINNED_PYTHON_SHA256" "$deb" | sha256sum -c -; then
        rm -rf "$tmp"
        die "Pinned Python checksum verification failed"
    fi

    package="$(dpkg-deb -f "$deb" Package 2>/dev/null || true)"
    version="$(dpkg-deb -f "$deb" Version 2>/dev/null || true)"
    package_arch="$(dpkg-deb -f "$deb" Architecture 2>/dev/null || true)"
    if [ "$package" != "python" ] || \
       [ "$version" != "$PINNED_PYTHON_VERSION" ] || \
       [ "$package_arch" != "aarch64" ]; then
        rm -rf "$tmp"
        die "Pinned Python package metadata did not match the locked release"
    fi

    dpkg-deb -x "$deb" "$tmp/root"
    staged_prefix="$tmp/root$PREFIX"
    [ -x "$staged_prefix/bin/python3.13" ] || {
        rm -rf "$tmp"
        die "Pinned package did not contain $PREFIX/bin/python3.13"
    }

    # Install only the versioned payload. The package also contains aliases for
    # python/python3 and related tools; removing them before the copy keeps the
    # Termux system interpreter untouched.
    rm -f \
        "$staged_prefix/bin/python" "$staged_prefix/bin/python3" \
        "$staged_prefix/bin/python-config" "$staged_prefix/bin/python3-config" \
        "$staged_prefix/bin/pip" "$staged_prefix/bin/pip3" \
        "$staged_prefix/bin/idle" "$staged_prefix/bin/idle3" \
        "$staged_prefix/bin/pydoc" "$staged_prefix/bin/pydoc3" \
        "$staged_prefix/bin/2to3" \
        "$staged_prefix/lib/pkgconfig/python3.pc" \
        "$staged_prefix/lib/pkgconfig/python3-embed.pc" \
        "$staged_prefix/share/man/man1/python.1.gz" \
        "$staged_prefix/share/man/man1/python3.1.gz"
    cp -a "$staged_prefix/." "$PREFIX/"
    rm -rf "$tmp"
    hash -r
    python_is_supported "$PREFIX/bin/python3.13" || die \
        "Pinned Python $PINNED_PYTHON_VERSION failed its launch check"
    ok "$($PREFIX/bin/python3.13 --version) installed side-by-side"
}

install_system_packages() {
    log "Installing native Termux runtime packages"
    pkg install -y \
        git curl ca-certificates coreutils dpkg uv \
        gdbm libandroid-posix-semaphore libandroid-support libbz2 libcrypt \
        libexpat libffi liblzma libsqlite ncurses ncurses-ui-libs openssl \
        readline zlib libjpeg-turbo libpng freetype libwebp openjpeg \
        littlecms libtiff libyaml nodejs ripgrep ffmpeg >/dev/null
    ok "Termux runtime and uv are ready"
}

install_native_build_packages() {
    log "Installing native compiler fallback for this unsupported wheel target"
    pkg install -y clang rust make pkg-config binutils patchelf >/dev/null
    ok "Termux native build toolchain is ready"
}

prepare_repository() {
    mkdir -p "$(dirname "$INSTALL_DIR")"
    if [ -d "$INSTALL_DIR/.git" ]; then
        log "Updating existing Hermes checkout"
        local stash_ref=""
        if [ -n "$(git -C "$INSTALL_DIR" status --porcelain 2>/dev/null)" ]; then
            git -C "$INSTALL_DIR" stash push --include-untracked -m \
                "hermes-termux-installer-$(date -u +%Y%m%dT%H%M%SZ)" >/dev/null
            stash_ref="$(git -C "$INSTALL_DIR" stash list -1 --format='%gd')"
            warn "Local changes were stashed as $stash_ref"
        fi
        git -C "$INSTALL_DIR" fetch origin "$BRANCH"
        git -C "$INSTALL_DIR" checkout "$BRANCH" 2>/dev/null || \
            git -C "$INSTALL_DIR" checkout -B "$BRANCH" "origin/$BRANCH"
        git -C "$INSTALL_DIR" merge --ff-only "origin/$BRANCH"
        if [ -n "$stash_ref" ]; then
            if git -C "$INSTALL_DIR" stash pop >/dev/null; then
                ok "Restored local checkout changes"
            else
                warn "Could not automatically restore $stash_ref; it remains in git stash"
            fi
        fi
    elif [ -e "$INSTALL_DIR" ]; then
        local backup="${INSTALL_DIR}.broken-$(date -u +%Y%m%d-%H%M%S)"
        warn "Moving non-git install aside to $backup"
        mv "$INSTALL_DIR" "$backup"
        git clone --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
    else
        log "Cloning Hermes Agent"
        git clone --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
    fi

    if [ -n "$INSTALL_COMMIT" ]; then
        git -C "$INSTALL_DIR" fetch origin "$INSTALL_COMMIT" || true
        git -C "$INSTALL_DIR" checkout --detach "$INSTALL_COMMIT"
    fi
    ok "Repository ready at $INSTALL_DIR"
}

create_venv() {
    local python="$1"
    local backup=""
    cd "$INSTALL_DIR"
    if [ -d venv ]; then
        backup="venv.pre-native-termux-$(date -u +%Y%m%d-%H%M%S)"
        mv venv "$backup"
        warn "Preserved previous venv as $INSTALL_DIR/$backup until validation succeeds"
    fi

    if ! uv venv --python "$python" venv; then
        [ -n "$backup" ] && [ -d "$backup" ] && mv "$backup" venv
        die "uv failed to create the Hermes virtual environment"
    fi
    VENV_BACKUP="$backup"
    export VENV_BACKUP
    ok "Virtual environment ready: $(venv/bin/python --version)"
}

install_python_dependencies() {
    cd "$INSTALL_DIR"
    local venv_python="$INSTALL_DIR/venv/bin/python"
    local work="$HERMES_HOME/cache/termux-install"
    local direct="$work/direct.in"
    local lock_constraints="$work/lock-constraints.txt"
    local resolved="$work/resolved.txt"
    local wheelhouse_cache="$HERMES_HOME/cache/termux-wheelhouse"
    local cargo_home="$HERMES_HOME/cache/cargo-termux"
    mkdir -p "$work" "$cargo_home" "$HERMES_HOME/cache/uv-termux" "$wheelhouse_cache"

    export UV_NO_CONFIG=1
    export UV_LINK_MODE=copy
    export UV_PYTHON="$venv_python"
    export UV_CACHE_DIR="$HERMES_HOME/cache/uv-termux"
    export UV_CONCURRENT_BUILDS=1
    export UV_CONCURRENT_INSTALLS=1
    export UV_CONCURRENT_DOWNLOADS=4
    export CARGO_BUILD_JOBS=1
    export CARGO_HOME="$cargo_home"
    export ANDROID_API_LEVEL="$ANDROID_BUILD_API"
    export UV_DEFAULT_INDEX="https://pypi.org/simple"
    export UV_INDEX_STRATEGY=first-index
    unset UV_INDEX UV_EXTRA_INDEX_URL PIP_INDEX_URL PIP_EXTRA_INDEX_URL \
        _PYTHON_HOST_PLATFORM

    log "Installing the dependency resolver prerequisite"
    uv pip install --python "$venv_python" 'packaging==26.0'

    "$venv_python" scripts/termux_requirements.py \
        --pyproject pyproject.toml \
        --lock uv.lock \
        --requirements "$direct" \
        --constraints "$lock_constraints" \
        --python-version "$($venv_python -c \
            'import platform; print(platform.python_version())')"

    log "Resolving the Android-safe Termux dependency graph"
    uv pip compile "$direct" \
        --python "$venv_python" \
        --constraint "$lock_constraints" \
        --output-file "$resolved" \
        --no-annotate

    local wheelhouse wheelhouse_rc
    set +e
    wheelhouse="$("$venv_python" scripts/prepare_termux_wheelhouse.py \
        --cache-root "$wheelhouse_cache" \
        --requirements "$resolved" \
        --uv-lock uv.lock \
        --curl "$(command -v curl)")"
    wheelhouse_rc=$?
    set -e

    if [ "$wheelhouse_rc" -eq 0 ]; then
        log "Installing with immutable CPython 3.13 Android ARM64 wheels"
        uv pip install --python "$venv_python" \
            --requirements "$resolved" \
            --constraint constraints-termux.txt \
            --find-links "$wheelhouse" \
            --only-binary :all:
        ok "Verified immutable wheelhouse $wheelhouse"
    elif [ "$wheelhouse_rc" -eq 2 ]; then
        warn "This Python/architecture is outside the published immutable wheel target"
        warn "Falling back to one-time native builds for compatibility"
        install_native_build_packages
        uv pip install --python "$venv_python" \
            'setuptools>=77,<83' wheel packaging cython pycparser

        log "Prebuilding setuptools extensions with Android wheel tags"
        "$venv_python" scripts/install_android_wheels.py \
            --uv "$(command -v uv)" \
            --python "$venv_python" \
            --requirements "$resolved" \
            --android-api-level "$ANDROID_BUILD_API"

        install_resolved_from_source() {
            uv pip install --python "$venv_python" --requirements "$resolved" \
                --constraint constraints-termux.txt
        }

        log "Installing resolved dependencies with serial native builds"
        if ! install_resolved_from_source; then
            local broken_cargo="${cargo_home}.broken-$(date -u +%Y%m%d-%H%M%S)"
            warn "Native dependency build failed; retrying with a fresh Cargo registry"
            [ -d "$cargo_home" ] && mv "$cargo_home" "$broken_cargo"
            mkdir -p "$cargo_home"
            install_resolved_from_source || die \
                "Hermes dependency installation failed on the clean Cargo retry"
            rm -rf "$broken_cargo"
        fi
    else
        die "The pinned immutable Termux wheelhouse could not be verified or downloaded"
    fi

    log "Installing Hermes editable entrypoints with uv"
    uv pip install --python "$venv_python" --no-deps --editable .
    uv pip check --python "$venv_python"

    "$venv_python" - <<'PY'
import importlib

required = [
    "hermes_cli",
    "psutil",
    "yaml",
    "cffi",
    "PIL",
    "pydantic_core",
    "cryptography",
    "jiter",
    "rpds",
]
for module in required:
    importlib.import_module(module)
print("Hermes native dependency smoke test passed")
PY
    ok "Python dependencies installed"
}

install_launcher() {
    local launcher="$PREFIX/bin/hermes"
    local venv_python="$INSTALL_DIR/venv/bin/python"
    local venv_hermes="$INSTALL_DIR/venv/bin/hermes"
    [ -x "$venv_hermes" ] || die \
        "Hermes console entrypoint was not generated: $venv_hermes"
    cat > "$launcher" <<EOF_LAUNCHER
#!$PREFIX/bin/bash
unset PYTHONPATH
unset PYTHONHOME
export UV_LINK_MODE=copy
case "\${1:-}" in
    --version|-V|version)
        exec "$venv_python" -c 'from hermes_cli import __release_date__, __version__; import sys; print(f"Hermes Agent v{__version__} ({__release_date__})"); print(f"Install directory: $INSTALL_DIR"); print(f"Python: {sys.version.split()[0]}")'
        ;;
esac
exec "$venv_hermes" "\$@"
EOF_LAUNCHER
    chmod 755 "$launcher"
    hash -r
    ok "Installed launcher at $launcher"
}

configure_hermes() {
    mkdir -p "$HERMES_HOME"/{cron,sessions,logs,pairing,hooks,image_cache,audio_cache,memories,skills}
    if [ ! -f "$HERMES_HOME/.env" ]; then
        if [ -f "$INSTALL_DIR/.env.example" ]; then
            cp "$INSTALL_DIR/.env.example" "$HERMES_HOME/.env"
        else
            touch "$HERMES_HOME/.env"
        fi
    fi
    chmod 600 "$HERMES_HOME/.env"
    if [ ! -f "$HERMES_HOME/config.yaml" ] && \
       [ -f "$INSTALL_DIR/cli-config.yaml.example" ]; then
        cp "$INSTALL_DIR/cli-config.yaml.example" "$HERMES_HOME/config.yaml"
    fi
    if [ ! -f "$HERMES_HOME/SOUL.md" ]; then
        printf '%s\n' \
            'You are Hermes Agent, an intelligent AI assistant created by Nous Research.' \
            > "$HERMES_HOME/SOUL.md"
    fi

    if [ "$NO_SKILLS" = true ]; then
        printf '%s\n' 'Bundled skill seeding disabled by --no-skills.' \
            > "$HERMES_HOME/.no-bundled-skills"
    elif [ -f "$INSTALL_DIR/tools/skills_sync.py" ]; then
        "$INSTALL_DIR/venv/bin/python" "$INSTALL_DIR/tools/skills_sync.py" || \
            warn "Bundled skill sync failed"
    fi
    ok "Hermes configuration directory is ready"
}

install_optional_node_tools() {
    [ "$SKIP_BROWSER" = false ] || return 0
    command -v npm >/dev/null 2>&1 || {
        warn "npm unavailable; skipping browser tools"
        return 0
    }
    log "Installing optional browser command dependencies"
    if (
        cd "$INSTALL_DIR"
        npm install --workspaces=false --silent --no-fund --no-audit --progress=false
    ); then
        ok "Browser command dependencies installed"
    else
        warn "Browser command dependency install failed; core Hermes remains usable"
    fi
}

run_setup() {
    [ "$RUN_SETUP" = true ] || return 0
    [ "$NON_INTERACTIVE" = false ] || return 0
    if [ -r /dev/tty ] && [ -w /dev/tty ]; then
        "$PREFIX/bin/hermes" setup </dev/tty >/dev/tty || \
            warn "Setup wizard did not complete"
    else
        warn "No interactive terminal is available; run 'hermes setup' later"
    fi
}

main() {
    parse_args "$@"
    is_termux || die "This installer is only for native Termux/Android"
    [ -n "${PREFIX:-}" ] || die "PREFIX is not set; launch this from Termux"
    case "$ANDROID_BUILD_API" in
        ''|*[!0-9]*) die "--android-api-level must be numeric" ;;
    esac
    [ "$ANDROID_BUILD_API" -ge 21 ] && [ "$ANDROID_BUILD_API" -le 99 ] || \
        die "Android build API must be between 21 and 99"

    unset PYTHONPATH PYTHONHOME
    install_system_packages

    local python architecture
    if [ -n "$PYTHON_BIN" ]; then
        # An explicit 3.11/3.12 request is supported through the source-build
        # compatibility path. Automatic installs prefer the immutable cp313 set.
        python="$(find_supported_python || true)"
    else
        python="$(find_immutable_python || true)"
        if [ -z "$python" ]; then
            architecture="$(termux_architecture)"
            case "$architecture" in
                aarch64|arm64)
                    install_pinned_python313
                    python="$PREFIX/bin/python3.13"
                    ;;
                *)
                    python="$(find_supported_python || true)"
                    ;;
            esac
        fi
    fi
    [ -n "$python" ] || die \
        "No supported CPython 3.11-3.13 interpreter is available for this architecture"
    ok "Using $($python --version) at $python"

    prepare_repository
    create_venv "$python"
    install_python_dependencies
    install_launcher
    configure_hermes
    install_optional_node_tools

    "$PREFIX/bin/hermes" --version
    echo "git" > "$INSTALL_DIR/.install_method"
    if [ -n "${VENV_BACKUP:-}" ] && [ -d "$INSTALL_DIR/$VENV_BACKUP" ]; then
        rm -rf "$INSTALL_DIR/$VENV_BACKUP"
    fi
    run_setup
    ok "Hermes Agent is installed natively in Termux"
    printf '%s\n' "Run: hermes"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
