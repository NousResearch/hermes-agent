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
INSTALL_DIR="${HERMES_INSTALL_DIR:-}"
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
        --hermes-home|-HermesHome) HERMES_HOME="$2"; shift 2 ;;
        --manifest|-Manifest) WANT_MANIFEST=true; shift ;;
        --stage|-Stage) STAGE="$2"; shift 2 ;;
        --json|-Json) JSON=true; shift ;;
        --non-interactive|-NonInteractive) NON_INTERACTIVE=true; shift ;;
        --skip-setup|--skip-browser) NON_INTERACTIVE=true; shift ;;
        --include-desktop|-IncludeDesktop) INCLUDE_DESKTOP=true; shift ;;
        -h|--help)
            echo "Usage: install.sh [--branch NAME] [--commit SHA] [--dir PATH]"
            echo "                  [--hermes-home PATH]"
            echo "                  [--manifest] [--stage NAME] [--json]"
            echo "                  [--non-interactive] [--include-desktop]"
            exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

INSTALL_DIR="${INSTALL_DIR:-$HERMES_HOME/hermes-agent}"
export HERMES_HOME

log() { printf "\033[1;34m[hermes]\033[0m %s\n" "$1"; }
fail() { STAGE_REASON="$1"; printf "\033[1;31m[hermes]\033[0m %s\n" "$1" >&2; exit 1; }

# --- BEGIN GENERATED: bootstrap pins (scripts/gen-bootstrap-pins.py) ---
# Derived from pm/lock.json. DO NOT EDIT BY HAND:
# run scripts/gen-bootstrap-pins.py after a pin bump.
UV_PIN_VERSION="0.12.3"

# Sets UV_PIN_URL + UV_PIN_SHA256 for a <os>-<arch> target key.
uv_bootstrap_pin() {
    case "$1" in
        linux-x64)
            UV_PIN_URL="https://github.com/astral-sh/uv/releases/download/0.12.3/uv-x86_64-unknown-linux-gnu.tar.gz"
            UV_PIN_MIRROR="https://hermes-assets.nousresearch.com/upstream/sha256/600cf9a742aca00d292673b16b5acffaa7b8c269a364ad0c2e79498dcb1fe101"
            UV_PIN_SHA256="600cf9a742aca00d292673b16b5acffaa7b8c269a364ad0c2e79498dcb1fe101"
            ;;
        linux-arm64)
            UV_PIN_URL="https://github.com/astral-sh/uv/releases/download/0.12.3/uv-aarch64-unknown-linux-gnu.tar.gz"
            UV_PIN_MIRROR="https://hermes-assets.nousresearch.com/upstream/sha256/bb66cb52e7b1823aed1183630d8d8e5c958840d584a4c55ec10a4cfc168dcca2"
            UV_PIN_SHA256="bb66cb52e7b1823aed1183630d8d8e5c958840d584a4c55ec10a4cfc168dcca2"
            ;;
        darwin-x64)
            UV_PIN_URL="https://github.com/astral-sh/uv/releases/download/0.12.3/uv-x86_64-apple-darwin.tar.gz"
            UV_PIN_MIRROR="https://hermes-assets.nousresearch.com/upstream/sha256/4c9f52262a14da336e4a42ed24992d12d0c956acde87619e4611d321dffa602b"
            UV_PIN_SHA256="4c9f52262a14da336e4a42ed24992d12d0c956acde87619e4611d321dffa602b"
            ;;
        darwin-arm64)
            UV_PIN_URL="https://github.com/astral-sh/uv/releases/download/0.12.3/uv-aarch64-apple-darwin.tar.gz"
            UV_PIN_MIRROR="https://hermes-assets.nousresearch.com/upstream/sha256/546f7f8a6c70ff13a3a9d2bc958db3427298cebf3e0cb756f9177133b7068843"
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
        local _fetched_from="$UV_PIN_URL"
        # Only network availability failures permit trying identical mirrored bytes.
        if curl -LsSf "$UV_PIN_URL" -o "$_tmp/uv.tar.gz"; then
            :
        else
            local _curl_status=$?
            case "$_curl_status" in
                5|6|7|18|22|28|52|55|56) ;;
                *) rm -rf "$_tmp"; fail "failed to download pinned uv from $UV_PIN_URL (curl $_curl_status)" ;;
            esac
            if [ -n "${UV_PIN_MIRROR:-}" ] && curl -LsSf "$UV_PIN_MIRROR" -o "$_tmp/uv.tar.gz"; then
                _fetched_from="$UV_PIN_MIRROR"
            else
                rm -rf "$_tmp"
                fail "failed to download pinned uv from $UV_PIN_URL or ${UV_PIN_MIRROR:-no mirror}"
            fi
        fi
        local _digest
        if command -v sha256sum >/dev/null 2>&1; then
            _digest="$(sha256sum "$_tmp/uv.tar.gz" | cut -d' ' -f1)"
        else
            _digest="$(shasum -a 256 "$_tmp/uv.tar.gz" | cut -d' ' -f1)"
        fi
        if [ "$_digest" != "$UV_PIN_SHA256" ]; then
            rm -rf "$_tmp"
            fail "uv download digest mismatch from $_fetched_from (expected $UV_PIN_SHA256, got $_digest)"
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

json_string() {
    local value="$1" code char escaped
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    for ((code = 1; code < 32; code++)); do
        printf -v char '\\%03o' "$code"
        printf -v char '%b' "$char"
        printf -v escaped '\\u%04x' "$code"
        value="${value//"$char"/$escaped}"
    done
    printf '"%s"' "$value"
}

json_frame() {
    # $1 ok, $2 stage, $3 skipped, $4 reason
    if [ -n "${4:-}" ]; then
        printf '{"ok":%s,"stage":%s,"skipped":%s,"reason":%s}\n' "$1" "$(json_string "$2")" "$3" "$(json_string "$4")"
    else
        printf '{"ok":%s,"stage":%s,"skipped":%s}\n' "$1" "$(json_string "$2")" "$3"
    fi
}

stage_result() {
    local code="$1" ok=false reason="${STAGE_REASON:-}"
    if [ "$code" -eq 0 ]; then
        ok=true
    else
        reason="${reason:-stage failed (exit $code)}"
    fi
    if [ "$JSON" = true ]; then
        json_frame "$ok" "$STAGE" "${STAGE_SKIPPED:-false}" "$reason"
    fi
}

# The single authoritative stage list: emit_manifest prints it AND the
# no-flag ladder runs it, so --include-desktop affects the real run
# exactly as the manifest advertises.
stage_names() {
    printf '%s\n' prerequisites repository venv python-deps node-deps path config setup gateway
    [ "$INCLUDE_DESKTOP" = true ] && printf '%s\n' desktop
    printf '%s\n' complete
}

# "$1" stage name -> its manifest record fields (title|category|needs_user_input).
stage_record() {
    case "$1" in
        prerequisites) echo "System prerequisites|runtime|false" ;;
        repository)    echo "Download Hermes Agent|runtime|false" ;;
        venv)          echo "Create Python environment|runtime|false" ;;
        python-deps)   echo "Install Python dependencies|runtime|false" ;;
        node-deps)     echo "Install tool dependencies|runtime|false" ;;
        path)          echo "Install hermes command|runtime|false" ;;
        config)        echo "Prepare config and skills|configuration|false" ;;
        setup)         echo "Configure API keys and settings|configuration|true" ;;
        gateway)       echo "Configure gateway service|configuration|true" ;;
        desktop)       echo "Build desktop app|runtime|false" ;;
        complete)      echo "Finish install|runtime|false" ;;
    esac
}

emit_manifest() {
    printf '%s' '{"protocol_version":1,"stages":['
    _sep=""
    for _s in $(stage_names); do
        IFS='|' read -r _title _category _needs <<< "$(stage_record "$_s")"
        printf '%s{"name":"%s","title":"%s","category":"%s","needs_user_input":%s}' \
            "$_sep" "$_s" "$_title" "$_category" "$_needs"
        _sep=","
    done
    printf '%s\n' ']}'
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
        local staged attempt cloned=false
        staged="$(mktemp -d "$(dirname "$INSTALL_DIR")/.hermes-clone-XXXXXX")" || fail "cannot stage clone"
        for attempt in 1 2 3; do
            if git clone --branch "$BRANCH" "$REPO_URL" "$staged/tree"; then
                cloned=true
                break
            fi
            rm -rf "$staged/tree"
            [ "$attempt" = 3 ] || sleep "$((attempt * 5))"
        done
        if [ "$cloned" = false ]; then
            log "direct clone failed; trying deferred blob download"
            if git clone --depth 1 --single-branch --filter=blob:none --no-checkout \
                --branch "$BRANCH" "$REPO_URL" "$staged/tree"; then
                for attempt in 1 2; do
                    if git -C "$staged/tree" reset --hard HEAD; then
                        cloned=true
                        break
                    fi
                    [ "$attempt" = 2 ] || sleep 5
                done
            fi
        fi
        if [ "$cloned" = false ]; then
            rm -rf "$staged"
            fail "git clone failed; no checkout published"
        fi
        if ! mv "$staged/tree" "$INSTALL_DIR"; then
            rm -rf "$staged"
            fail "cannot publish cloned checkout"
        fi
        rmdir "$staged"
    fi
    if [ -n "$INSTALL_COMMIT" ]; then
        git -C "$INSTALL_DIR" checkout "$INSTALL_COMMIT" || fail "could not pin commit $INSTALL_COMMIT"
    fi
}

stage_venv() {
    # Keep the installer stage protocol; PM alone creates dependency environments.
    local boot_py
    bootstrap_python
    log "bootstrap Python ready; PM prepares the dependency environment"
}

# Tool-only bootstrap: acquire uv and Python before PM's own dependencies exist.
# The application dependency graph is never installed in this interpreter.
bootstrap_python() {
    ensure_uv
    local _py
    # Read packages.python.version by following object names and braces, not
    # indentation — same pre-Python reader contract as setup-hermes.sh's pin().
    _py="$(awk -F '"' '
        /^[[:space:]]*("[^"]+"[[:space:]]*:[[:space:]]*)?\{/ { path[++depth] = $2; next }
        /^[[:space:]]*\}[[:space:]]*,?[[:space:]]*$/ { delete path[depth--]; next }
        path[2] == "packages" && path[3] == "python" && $2 == "version" && depth == 3 { print $4; exit }
    ' "$INSTALL_DIR/pm/lock.json" | cut -d+ -f1 | cut -d. -f1,2)"
    [ -n "$_py" ] || _py="3.14"
    "$UV_CMD" python install --no-bin "$_py" || fail "bootstrap Python installation failed"
    boot_py="$("$UV_CMD" python find --managed-python "$_py")" || fail "bootstrap Python lookup failed"
    boot_py="${boot_py%$'\r'}"
}

# uv exits before PM can replace its tool entry. pm.cli then prepares and
# enters its independently locked runtime before mutating application deps.
bootstrap_pm() {
    local boot_py
    bootstrap_python
    log "delegating python + venv + tools to pm (hash-verified via uv.lock)"
    (cd "$INSTALL_DIR" && "$boot_py" -m pm.cli install) || fail "pm install failed"
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
    local boot_py
    bootstrap_python
    (cd "$INSTALL_DIR" && "$boot_py" -I -X utf8 hermes_cli/_launchers.py "$link_dir") || fail "launcher publication failed"
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
    # `hermes desktop --build-only` is the current authority (same path as
    # `hermes gui` / the update flow); no installer-local node/electron code.
    "$INSTALL_DIR/venv/bin/python" "$INSTALL_DIR/hermes" desktop --build-only || fail "desktop build failed"
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

run_stage() (
    # Keep failure handling out of conditional calls, which disable errexit.
    set -e
    STAGE="$1"
    STAGE_REASON=""
    STAGE_SKIPPED=false
    trap 'stage_result "$?"' EXIT
    if [ "$NON_INTERACTIVE" = true ] && { [ "$STAGE" = setup ] || [ "$STAGE" = gateway ]; }; then
        STAGE_SKIPPED=true
        STAGE_REASON="needs user input"
        exit 0
    fi
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
        *) STAGE_REASON="unknown stage: $1"; printf '%s\n' "$STAGE_REASON" >&2; exit 2 ;;
    esac
)

# Main. Guarded so the script can be SOURCED for its functions (the
# installer-test harness sources it with --manifest, which must define
# the functions and stop before main).
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    if [ "$WANT_MANIFEST" = true ]; then
        emit_manifest
        exit 0
    fi

    if [ -n "$STAGE" ] && [ "$JSON" = true ]; then
        trap 'stage_result "$?"' EXIT
    fi
    check_platform
    trap - EXIT

    if [ -n "$STAGE" ]; then
        run_stage "$STAGE"
        exit "$?"
    fi

    # No --stage: run the whole ladder — the same authoritative list the
    # manifest prints, so --include-desktop inserts desktop here too.
    for s in $(stage_names); do
        run_stage "$s"
        rc=$?
        [ "$rc" -eq 0 ] || exit "$rc"
    done
fi
