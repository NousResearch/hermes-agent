#!/usr/bin/env bash
# Hermes Agent bootstrap: git checkout + venv + hermes command on PATH.
# Heavy dependencies (tool binaries, browsers, node) are pm's job after
# this: `hermes pm install`. Stage protocol kept for Hermes-Setup:
#   --manifest            print the stage list as JSON
#   --stage NAME [--json] run one stage
#   --non-interactive     skip stages that need input
#   --include-desktop     add the desktop build stage
set -u

# Prevent uv from discovering config files (uv.toml, pyproject.toml) from the
# wrong user's home directory when running under sudo -u <user>.  See #21269.
# pm's own venv sync re-isolates (pm/packages.py::uv_env), so this bootstrap
# hygiene can't break the locked sync the way it used to before pm owned it.
export UV_NO_CONFIG=1

REPO_URL="${HERMES_REPO_URL:-https://github.com/NousResearch/hermes-agent.git}"
BRANCH="main"
INSTALL_COMMIT=""
INSTALL_DIR="${HERMES_INSTALL_DIR:-$HOME/.hermes/hermes-agent}"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
STAGE=""
WANT_MANIFEST=false
JSON=false
NON_INTERACTIVE=false
INCLUDE_DESKTOP=false

while [ $# -gt 0 ]; do
    case "$1" in
        --branch|-Branch) BRANCH="$2"; shift 2 ;;
        --commit|-Commit) INSTALL_COMMIT="$2"; shift 2 ;;
        --dir) INSTALL_DIR="$2"; shift 2 ;;
        --manifest|-Manifest) WANT_MANIFEST=true; shift ;;
        --stage|-Stage) STAGE="$2"; shift 2 ;;
        --json|-Json) JSON=true; shift ;;
        --non-interactive|-NonInteractive) NON_INTERACTIVE=true; shift ;;
        --skip-setup|--skip-browser) NON_INTERACTIVE=true; shift ;;
        --include-desktop|-IncludeDesktop) INCLUDE_DESKTOP=true; shift ;;
        -h|--help)
            echo "Usage: install.sh [--branch NAME] [--commit SHA] [--dir PATH]"
            echo "                  [--manifest] [--stage NAME] [--json]"
            echo "                  [--non-interactive] [--include-desktop]"
            exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

log() { printf "\033[1;34m[hermes]\033[0m %s\n" "$1"; }
fail() { printf "\033[1;31m[hermes]\033[0m %s\n" "$1" >&2; exit 1; }

# --- BEGIN GENERATED: bootstrap pins (scripts/gen-bootstrap-pins.py) ---
# Derived from pm/lock.json. DO NOT EDIT BY HAND:
# run scripts/gen-bootstrap-pins.py after a pin bump.
UV_PIN_VERSION="0.12.3"

# Sets UV_PIN_URL + UV_PIN_SHA256 for a <os>-<arch> target key.
uv_bootstrap_pin() {
    case "$1" in
        linux-x64)
            UV_PIN_URL="https://github.com/astral-sh/uv/releases/download/0.12.3/uv-x86_64-unknown-linux-gnu.tar.gz"
            UV_PIN_SHA256="600cf9a742aca00d292673b16b5acffaa7b8c269a364ad0c2e79498dcb1fe101"
            ;;
        linux-arm64)
            UV_PIN_URL="https://github.com/astral-sh/uv/releases/download/0.12.3/uv-aarch64-unknown-linux-gnu.tar.gz"
            UV_PIN_SHA256="bb66cb52e7b1823aed1183630d8d8e5c958840d584a4c55ec10a4cfc168dcca2"
            ;;
        darwin-x64)
            UV_PIN_URL="https://github.com/astral-sh/uv/releases/download/0.12.3/uv-x86_64-apple-darwin.tar.gz"
            UV_PIN_SHA256="4c9f52262a14da336e4a42ed24992d12d0c956acde87619e4611d321dffa602b"
            ;;
        darwin-arm64)
            UV_PIN_URL="https://github.com/astral-sh/uv/releases/download/0.12.3/uv-aarch64-apple-darwin.tar.gz"
            UV_PIN_SHA256="546f7f8a6c70ff13a3a9d2bc958db3427298cebf3e0cb756f9177133b7068843"
            ;;
        *)
            UV_PIN_URL=""
            UV_PIN_SHA256=""
            return 1
            ;;
    esac
}
# --- END GENERATED: bootstrap pins ---

uv_bootstrap_target() {
    # Map this host to a pm/lock.json target key (<os>-<arch>).
    local _arch
    case "$(uname -m)" in
        arm64|aarch64) _arch="arm64" ;;
        x86_64|amd64)  _arch="x64" ;;
        *) return 1 ;;
    esac
    case "$(uname -s)" in
        Linux)  echo "linux-$_arch" ;;
        Darwin) echo "darwin-$_arch" ;;
        *) return 1 ;;
    esac
}

# Provision uv for this host from the pinned pm/lock.json artifact. Stages
# the EXACT artifact pm itself uses into the same store slot
# (~/.hermes/tools/uv-<version>-<target>/), sha256-verified, so the byte
# authority is pm/lock.json - no astral-latest, no curl|sh.
UV_CMD=""
ensure_uv() {
    [ -n "$UV_CMD" ] && return 0
    if command -v uv >/dev/null 2>&1; then
        # Developer shortcut: an existing uv on PATH is fine to use; this
        # branch fetches nothing.
        UV_CMD="uv"
        return 0
    fi
    local _target
    if ! _target="$(uv_bootstrap_target)"; then
        fail "no pinned uv build for this platform ($(uname -s) $(uname -m)); install uv manually: https://docs.astral.sh/uv/"
    fi
    if ! uv_bootstrap_pin "$_target"; then
        fail "no pinned uv artifact for $_target; install uv manually: https://docs.astral.sh/uv/"
    fi
    local _store="${HERMES_RUNTIME_DIR:-$HOME/.hermes/tools}"
    local _entry="$_store/uv-$UV_PIN_VERSION-$_target"
    UV_CMD="$_entry/uv"
    if [ ! -x "$UV_CMD" ]; then
        log "staging pinned uv $UV_PIN_VERSION ($_target) into the pm store"
        local _tmp
        _tmp="$(mktemp -d 2>/dev/null || echo "/tmp/hermes-uv-bootstrap.$$")"
        mkdir -p "$_tmp"
        if ! curl -LsSf "$UV_PIN_URL" -o "$_tmp/uv.tar.gz"; then
            rm -rf "$_tmp"
            fail "failed to download pinned uv from $UV_PIN_URL"
        fi
        local _digest
        if command -v sha256sum >/dev/null 2>&1; then
            _digest="$(sha256sum "$_tmp/uv.tar.gz" | cut -d' ' -f1)"
        else
            _digest="$(shasum -a 256 "$_tmp/uv.tar.gz" | cut -d' ' -f1)"
        fi
        if [ "$_digest" != "$UV_PIN_SHA256" ]; then
            rm -rf "$_tmp"
            fail "uv download digest mismatch (expected $UV_PIN_SHA256, got $_digest)"
        fi
        if ! tar -xzf "$_tmp/uv.tar.gz" -C "$_tmp"; then
            rm -rf "$_tmp"
            fail "failed to extract pinned uv archive"
        fi
        local _unpacked
        _unpacked="$(find "$_tmp" -mindepth 1 -maxdepth 2 -name uv -type f | head -n1)"
        if [ -z "$_unpacked" ]; then
            rm -rf "$_tmp"
            fail "uv binary not found in the downloaded archive"
        fi
        mkdir -p "$_entry"
        mv "$_unpacked" "$UV_CMD"
        [ -f "$(dirname "$_unpacked")/uvx" ] && mv "$(dirname "$_unpacked")/uvx" "$_entry/uvx"
        chmod +x "$UV_CMD"
        chmod +x "$_entry/uvx" 2>/dev/null || true
        rm -rf "$_tmp"
    fi
    # Make the staged (or found) uv available to bare `uv` invocations.
    export PATH="$(dirname "$UV_CMD"):$PATH"
    if ! "$UV_CMD" --version >/dev/null 2>&1; then
        fail "pinned uv staged but does not run on this host"
    fi
    log "uv ready ($("$UV_CMD" --version 2>/dev/null))"
}

check_platform() {
    case "$(uname -s 2>/dev/null)" in
        Linux*) : ;;
        Darwin*) : ;;
        *) fail "unsupported platform: $(uname -s). On Windows use install.ps1." ;;
    esac
}

json_frame() {
    # $1 ok, $2 stage, $3 skipped, $4 reason
    if [ -n "${4:-}" ]; then
        printf '{"ok":%s,"stage":"%s","skipped":%s,"reason":"%s"}\n' "$1" "$2" "$3" "$4"
    else
        printf '{"ok":%s,"stage":"%s","skipped":%s}\n' "$1" "$2" "$3"
    fi
}

emit_manifest() {
    local desktop=""
    if [ "$INCLUDE_DESKTOP" = true ]; then
        desktop='{"name":"desktop","title":"Build desktop app","category":"runtime","needs_user_input":false},'
    fi
    printf '%s' '{"protocol_version":1,"stages":['
    printf '%s' '{"name":"prerequisites","title":"System prerequisites","category":"runtime","needs_user_input":false},'
    printf '%s' '{"name":"repository","title":"Download Hermes Agent","category":"runtime","needs_user_input":false},'
    printf '%s' '{"name":"venv","title":"Create Python environment","category":"runtime","needs_user_input":false},'
    printf '%s' '{"name":"python-deps","title":"Install Python dependencies","category":"runtime","needs_user_input":false},'
    printf '%s' '{"name":"node-deps","title":"Install tool dependencies","category":"runtime","needs_user_input":false},'
    printf '%s' '{"name":"path","title":"Install hermes command","category":"runtime","needs_user_input":false},'
    printf '%s' '{"name":"config","title":"Prepare config and skills","category":"configuration","needs_user_input":false},'
    printf '%s' '{"name":"setup","title":"Configure API keys and settings","category":"configuration","needs_user_input":true},'
    printf '%s' '{"name":"gateway","title":"Configure gateway service","category":"configuration","needs_user_input":true},'
    printf '%s' "$desktop"
    printf '%s\n' '{"name":"complete","title":"Finish install","category":"runtime","needs_user_input":false}]}'
}

stage_prerequisites() {
    command -v git >/dev/null 2>&1 || fail "git is required. Install it with your system package manager."
    command -v curl >/dev/null 2>&1 || fail "curl is required. Install it with your system package manager."
    log "prerequisites ok (git, curl)"
}

stage_repository() {
    if [ -d "$INSTALL_DIR/.git" ]; then
        log "updating $INSTALL_DIR"
        git -C "$INSTALL_DIR" fetch origin "$BRANCH" || fail "git fetch failed"
        git -C "$INSTALL_DIR" checkout "$BRANCH" || fail "git checkout failed"
        git -C "$INSTALL_DIR" pull --ff-only origin "$BRANCH" || log "not fast-forwardable; keeping local state"
    else
        log "cloning $REPO_URL ($BRANCH) into $INSTALL_DIR"
        mkdir -p "$(dirname "$INSTALL_DIR")"
        git clone --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR" || fail "git clone failed"
    fi
    if [ -n "$INSTALL_COMMIT" ]; then
        git -C "$INSTALL_DIR" checkout "$INSTALL_COMMIT" || fail "could not pin commit $INSTALL_COMMIT"
    fi
}

stage_venv() {
    ensure_uv
    log "creating venv"
    (cd "$INSTALL_DIR" && "$UV_CMD" venv --allow-existing venv) || fail "uv venv failed"
}

# Delegate the whole python+venv+tools install to pm: stage the pinned uv,
# let uv run pm.cli, and pm provisions the interpreter, the venv (default
# extras = [all], so it matches what `hermes update` force-syncs), and the
# tool store — all hash-verified against pm/lock.json + uv.lock. install.sh
# no longer runs `uv sync` directly; pm is the single install authority
# (the run_locked_uv_sync contract moved into pm/packages.py::uv_env).
bootstrap_pm() {
    ensure_uv
    local _py
    _py="$(awk '/^    "python": \{/ { in_py = 1 }
        in_py && /^      "version":/ { gsub(/.*: "|"$|",$/, ""); print; exit }' \
        "$INSTALL_DIR/pm/lock.json" | cut -d+ -f1 | cut -d. -f1,2)"
    [ -n "$_py" ] || _py="3.11"
    log "delegating python + venv + tools to pm (hash-verified via uv.lock)"
    (cd "$INSTALL_DIR" && "$UV_CMD" run --no-project --python "$_py" python -m pm.cli install) \
        || fail "pm install failed"
}

stage_python_deps() {
    bootstrap_pm
}

stage_node_deps() {
    # Tool binaries, node, browsers: pm packages, installed on demand or
    # via `hermes pm install`. Nothing to do at bootstrap time.
    log "tool dependencies are managed by pm (hermes pm install)"
}

stage_path() {
    local link_dir="$HOME/.local/bin"
    mkdir -p "$link_dir"
    rm -f "$link_dir/hermes"
    # Boot wrapper: exec the pm STORE python with PYTHONPATH=repo:venv-
    # site-packages (repo first). Never boots through venv/bin/python —
    # the venv is only a uv sync target and pyvenv.cfg is inert dead
    # config (pm work item 3). Store root + python entry mirror
    # pm/paths.py / pm facts.json; resolved at BOOT so the wrapper picks
    # up `hermes pm install` whenever it materializes the store.
    cat > "$link_dir/hermes" <<WRAPPER
#!/usr/bin/env bash
# Hermes boot wrapper: store python + PYTHONPATH (never the venv python).
set -u
repo="\${HERMES_INSTALL_DIR:-$INSTALL_DIR}"
unset PYTHONHOME
runtime="\${HERMES_RUNTIME_DIR:-}"
if [ -z "\$runtime" ] && [ -f "\$repo/install-stamp.json" ]; then
    runtime="\$(sed -n 's/.*"runtimeDir"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "\$repo/install-stamp.json" | head -n 1)"
fi
[ -n "\$runtime" ] || runtime="\$HOME/.hermes/tools"
store_py=""
if [ -f "\$runtime/facts.json" ]; then
    entry="\$(sed -n 's/.*"entry"[[:space:]]*:[[:space:]]*"\(python-[^"]*\)".*/\1/p' "\$runtime/facts.json" | head -n 1)"
    if [ -n "\$entry" ] && [ -x "\$runtime/\$entry/bin/python3" ]; then
        store_py="\$runtime/\$entry/bin/python3"
    fi
fi
if [ -z "\$store_py" ]; then
    for d in "\$runtime"/python-*/bin/python3; do
        [ -x "\$d" ] && store_py="\$d"
    done
fi
if [ -z "\$store_py" ]; then
    echo "hermes: no pm store interpreter under \$runtime - run 'hermes pm install' first" >&2
    exit 1
fi
site=""
for d in "\$repo"/venv/lib/python3.*/site-packages "\$repo"/.venv/lib/python3.*/site-packages; do
    [ -d "\$d" ] && site="\$d"
done
export PYTHONPATH="\$repo\${site:+:\$site}"
exec "\$store_py" "\$repo/hermes" "\$@"
WRAPPER
    chmod +x "$link_dir/hermes"
    case ":$PATH:" in
        *":$link_dir:"*) : ;;
        *) log "add $link_dir to your PATH to use the hermes command" ;;
    esac
    log "hermes command installed at $link_dir/hermes"
}

stage_config() {
    mkdir -p "$HERMES_HOME"/cron "$HERMES_HOME"/sessions "$HERMES_HOME"/logs \
        "$HERMES_HOME"/pairing "$HERMES_HOME"/hooks "$HERMES_HOME"/image_cache \
        "$HERMES_HOME"/audio_cache "$HERMES_HOME"/memories "$HERMES_HOME"/skills
    if [ ! -f "$HERMES_HOME/.env" ]; then
        cp "$INSTALL_DIR/.env.example" "$HERMES_HOME/.env" 2>/dev/null || touch "$HERMES_HOME/.env"
    fi
    chmod 600 "$HERMES_HOME/.env"
    if [ ! -f "$HERMES_HOME/config.yaml" ] && [ -f "$INSTALL_DIR/cli-config.yaml.example" ]; then
        cp "$INSTALL_DIR/cli-config.yaml.example" "$HERMES_HOME/config.yaml"
    fi
    log "config prepared in $HERMES_HOME"
}

stage_setup() {
    if [ "$NON_INTERACTIVE" = true ]; then return 0; fi
    "$INSTALL_DIR/venv/bin/python" "$INSTALL_DIR/hermes" setup || true
}

stage_gateway() {
    if [ "$NON_INTERACTIVE" = true ]; then return 0; fi
    "$INSTALL_DIR/venv/bin/python" "$INSTALL_DIR/hermes" gateway install || true
}

stage_desktop() {
    "$INSTALL_DIR/venv/bin/python" "$INSTALL_DIR/hermes" desktop build || fail "desktop build failed"
}

stage_complete() {
    local commit
    commit="$INSTALL_COMMIT"
    [ -n "$commit" ] || commit=$(git -C "$INSTALL_DIR" rev-parse HEAD 2>/dev/null) || commit=""
    if [ -n "$commit" ]; then
        printf '{\n  "schemaVersion": 1,\n  "pinnedCommit": "%s",\n  "pinnedBranch": "%s",\n  "completedAt": "%s"\n}\n' \
            "$commit" "$BRANCH" "$(date -u +%Y-%m-%dT%H:%M:%S.000Z)" > "$INSTALL_DIR/.hermes-bootstrap-complete.tmp"
        mv -f "$INSTALL_DIR/.hermes-bootstrap-complete.tmp" "$INSTALL_DIR/.hermes-bootstrap-complete"
    fi
    log "install complete. Run: hermes"
}

run_stage() {
    case "$1" in
        prerequisites) stage_prerequisites ;;
        repository) stage_repository ;;
        venv) stage_venv ;;
        python-deps) stage_python_deps ;;
        node-deps) stage_node_deps ;;
        path) stage_path ;;
        config) stage_config ;;
        setup) stage_setup ;;
        gateway) stage_gateway ;;
        desktop) stage_desktop ;;
        complete) stage_complete ;;
        *) echo "unknown stage: $1" >&2; exit 2 ;;
    esac
}

if [ "$WANT_MANIFEST" = true ]; then
    emit_manifest
    exit 0
fi

check_platform

if [ -n "$STAGE" ]; then
    if [ "$NON_INTERACTIVE" = true ] && { [ "$STAGE" = setup ] || [ "$STAGE" = gateway ]; }; then
        [ "$JSON" = true ] && json_frame true "$STAGE" true "needs user input"
        exit 0
    fi
    if run_stage "$STAGE"; then
        [ "$JSON" = true ] && json_frame true "$STAGE" false
        exit 0
    else
        rc=$?
        [ "$JSON" = true ] && json_frame false "$STAGE" false "stage failed"
        exit "$rc"
    fi
fi

# No --stage: run the whole ladder.
for s in prerequisites repository venv python-deps node-deps path config setup gateway complete; do
    run_stage "$s"
done
