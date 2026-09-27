#!/usr/bin/env bash
# ============================================================================
# Hermes Agent Setup Script — THE dev-environment entry point.
# ============================================================================
# Sets up the pm-managed development environment from a fresh clone:
#   1. Stage the pinned uv from pm/lock.json (sha256-verified, into the pm
#      store slot) — pm needs uv to bootstrap, so it cannot stage uv itself.
#   2. Use uv to install and locate bootstrap Python, then let uv exit.
#      Run `python -m pm.cli install` directly so PM can safely replace uv.
#      PM owns the final interpreter, tool store, and dependency generation.
#   3. Point you at `source ./activate` — the venv-style way to put the pm
#      env (PATH + tool vars) into your current shell.
# There is no pip fallback tier here on purpose.
# ============================================================================

set -e

# Setup and activation prepare the isolated test environment. Installers call
# pm.cli directly and never select it.
runtime_only=false
test_environment="--test-environment"
for option in "$@"; do
    case "$option" in
        --runtime-only) runtime_only=true ;;
        --test-environment|--test-environment=*) test_environment="$option" ;;
        *) printf 'Unknown setup option: %s\n' "$option" >&2; exit 2 ;;
    esac
done
# Colors
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Prevent uv from discovering config files (uv.toml, pyproject.toml) from the
# wrong user's home directory when running under sudo -u <user>.  See #21269.
export UV_NO_CONFIG=1

echo ""
echo -e "${CYAN}☤ Hermes Agent Setup${NC}"
echo ""

# ============================================================================
# Install / locate uv — staged from pm/lock.json (the lockfile is the only
# authority; no astral-latest, no curl|sh).
# ============================================================================

echo -e "${CYAN}→${NC} Checking for uv..."

lock="$SCRIPT_DIR/pm/lock.json"
[ -f "$lock" ] || { echo -e "${RED}✗${NC} pm/lock.json not found" >&2; exit 1; }

# Read the shared mirror location before Python is available. Match object
# keys, not indentation — same rule as pin() below.
mirror_origin="$(awk -F '"' '$2 == "origin" { print $4; exit }' "$SCRIPT_DIR/pm/artifact-mirror.json")"
mirror_prefix="$(awk -F '"' '$2 == "prefix" { print $4; exit }' "$SCRIPT_DIR/pm/artifact-mirror.json")"
mirror_url_for() { # $1 = lowercase sha256
  [ -n "$mirror_origin" ] && [ -n "$mirror_prefix" ] || return 1
  printf '%s/%s%s' "$mirror_origin" "$mirror_prefix" "$1"
}

case "$(uname -s)" in
  Linux) os=linux ;;
  Darwin) os=darwin ;;
  MINGW*|MSYS*|CYGWIN*) os=win32 ;;
  *) echo -e "${RED}✗${NC} unsupported OS $(uname -s)" >&2; exit 1 ;;
esac
if [ "$os" = win32 ]; then
  # PROCESSOR_ARCHITECTURE lies under an emulated shell (x64 msys on a
  # WoA box reports AMD64); the registry carries the machine's truth.
  winarch="$(MSYS2_ARG_CONV_EXCL='*' reg.exe query 'HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment' /v PROCESSOR_ARCHITECTURE 2>/dev/null | tr -d '\r' | awk '/PROCESSOR_ARCHITECTURE/ {print $NF}')"
  case "${winarch:-${PROCESSOR_ARCHITECTURE:-}}" in
    ARM64) arch=arm64 ;;
    *) arch=x64 ;;
  esac
else
  case "$(uname -m)" in
    arm64|aarch64) arch=arm64 ;;
    x86_64|amd64) arch=x64 ;;
    *) echo -e "${RED}✗${NC} unsupported arch $(uname -m)" >&2; exit 1 ;;
  esac
fi
target="$os-$arch"

# The machine-written lock has one member per line. Follow object names and
# braces, not indentation, to read pins before Python is available.
pin() { # $1 = field (url | sha256 | version), $2 = package (default: uv)
  awk -F '"' -v package="${2:-uv}" -v target="$target" -v field="$1" '
    /^[[:space:]]*("[^"]+"[[:space:]]*:[[:space:]]*)?\{[[:space:]]*$/ {
      path[++depth] = $2; next
    }
    /^[[:space:]]*}[[:space:]]*,?[[:space:]]*$/ {
      delete path[depth--]; next
    }
    path[2] == "packages" && path[3] == package && $2 == field &&
      ((field == "version" && depth == 3) ||
       (depth == 5 && path[4] == "artifacts" && path[5] == target)) {
      print $4; exit
    }' "$lock"
}
uv_version="$(pin version)"
py_version="$(pin version python | cut -d+ -f1 | cut -d. -f1,2)"
[ -n "$uv_version" ] || { echo -e "${RED}✗${NC} no uv pin in pm/lock.json" >&2; exit 1; }

# The bootstrap store follows the machine root folded from HERMES_HOME, never a
# hardcoded ~/.hermes; byte-identical to install.sh's copy (the mirror-body test
# compares both with pm's answer). A stamped ``runtimeDir`` is not readable here
# yet (no Python); HERMES_RUNTIME_DIR remains the explicit override.
# --- BEGIN store-root resolver (mirrored in scripts/install.sh) ---
_hermes_user_home() {
    # '~user' -> the account's home, or nothing when unknown (getent / dscacheutil).
    local _name="$1" _home=""
    case "$_name" in
        ""|*[!A-Za-z0-9_.-]*) return 1 ;;
    esac
    if command -v getent >/dev/null 2>&1; then
        _home="$(getent passwd "$_name" 2>/dev/null | cut -d: -f6)"
    fi
    if [ -z "$_home" ] && command -v dscacheutil >/dev/null 2>&1; then
        _home="$(dscacheutil -q user -a name "$_name" 2>/dev/null | awk '/^dir:/{print $2}')"
    fi
    [ -n "$_home" ] || return 1
    printf '%s' "$_home"
}
_hermes_expand() {
    # os.path.expandvars+expanduser without eval: an UNSET name keeps its source
    # text (braces included), anything else malformed stays literal, and printenv
    # reads the exported table (unlike ${!name}).
    local _s="$1" _out="" _name _v
    while [ -n "$_s" ]; do
        case "$_s" in
            '$'*)
                _s="${_s#\$}"
                case "$_s" in
                    '{'*)
                        case "$_s" in
                            *'}'*)
                                _name="${_s#\{}"
                                _name="${_name%%\}*}"
                                _s="${_s#*\}}"
                                if [ -z "$_name" ]; then
                                    _out="$_out\${}"
                                elif _v="$(printenv "$_name")"; then
                                    _out="$_out$_v"
                                else
                                    _out="$_out\${$_name}"
                                fi
                                ;;
                            *)
                                _out="$_out\$"
                                ;;
                        esac
                        ;;
                    [A-Za-z0-9_]*)
                        _name="${_s%%[!A-Za-z0-9_]*}"
                        _s="${_s#"$_name"}"
                        if _v="$(printenv "$_name")"; then
                            _out="$_out$_v"
                        else
                            _out="$_out\$$_name"
                        fi
                        ;;
                    *)
                        _out="$_out\$"
                        ;;
                esac
                ;;
            *)
                _out="$_out${_s%%\$*}"
                _s="${_s#"${_s%%\$*}"}"
                ;;
        esac
    done
    case "$_out" in
        "~"|"~/"*)
            # posixpath.expanduser: HOME when it is set — an EMPTY HOME answers
            # "/" — and the account database when HOME is unset (expanduser's pwd
            # fallback). '~user' below always resolves through the account database.
            local _tail=""
            case "$_out" in "~/"*) _tail="${_out#"~"}" ;; esac
            if [ "${HOME+x}" = x ]; then
                if [ -n "$HOME" ]; then
                    _out="$HOME$_tail"
                else
                    _out="/${_tail#/}"
                fi
            elif _v="$(_hermes_user_home "$(id -un 2>/dev/null)")"; then
                _out="$_v$_tail"
            fi
            ;;
        "~"?*)
            # '~user'/'~user/rest': the account home replaces the prefix; unknown stays literal.
            local _user="${_out#"~"}" _tail=""
            case "$_user" in
                */*) _tail="/${_user#*/}"; _user="${_user%%/*}" ;;
            esac
            if _v="$(_hermes_user_home "$_user")"; then
                _out="$_v$_tail"
            fi
            ;;
    esac
    printf '%s' "$_out"
}
_hermes_norm() {
    # Lexical normalization like Path: collapse repeated separators and drop
    # "."; ".." stays VERBATIM (the fold runs on the RESOLVED path).
    local _rest="$1" _out="" _comp
    case "$_rest" in
        "//"[!/]*) _out="//" ;;
        /*)        _out="/" ;;
    esac
    _rest="${_rest#/}"
    case "$_rest" in
        "//"*) _rest="${_rest#//}" ;;
    esac
    while [ -n "$_rest" ]; do
        _comp="${_rest%%/*}"
        case "$_rest" in
            */*) _rest="${_rest#*/}" ;;
            *)   _rest="" ;;
        esac
        [ -z "$_comp" ] && continue
        [ "$_comp" = "." ] && continue
        if [ -z "$_out" ]; then _out="$_comp"
        elif [ "$_out" = "/" ] || [ "$_out" = "//" ]; then _out="$_out$_comp"
        else _out="$_out/$_comp"; fi
    done
    printf '%s' "$_out"
}
_hermes_resolve() {
    # Path.resolve(strict=False): follow every symlink along the chain (not only
    # the longest existing prefix), keep missing components, pop ".." off the
    # resolved prefix (never the anchor).
    local _p="$1"
    case "$_p" in
        /*) ;;
        *) _p="$PWD/$_p" ;;
    esac
    local _body="${_p#/}" _rest _c _cand _link _res="" _depth=0
    _rest="${_body#/}"
    while [ -n "$_rest" ]; do
        _c="${_rest%%/*}"
        case "$_rest" in
            */*) _rest="${_rest#*/}" ;;
            *)   _rest="" ;;
        esac
        [ -z "$_c" ] && continue
        [ "$_c" = "." ] && continue
        if [ "$_c" = ".." ]; then
            # A one-component pop empties the chain ("%/*" alone keeps the component).
            case "$_res" in
                */*) _res="${_res%/*}" ;;
                *)   _res="" ;;
            esac
            continue
        fi
        if [ -z "$_res" ]; then _cand="/$_c"; else _cand="/$_res/$_c"; fi
        if [ -L "$_cand" ]; then
            _depth=$((_depth + 1))
            if [ "$_depth" -gt 40 ]; then
                # Symlink loop (realpath's cap): answer the lexical input.
                _hermes_norm "$1"
                return
            fi
            _link="$(readlink "$_cand")"
            case "$_link" in
                /*)
                    _res=""
                    _rest="${_link#/}/$_rest"
                    _rest="${_rest#/}"
                    ;;
                *)
                    _rest="$_link/$_rest"
                    ;;
            esac
            continue
        fi
        if [ -z "$_res" ]; then _res="$_c"; else _res="$_res/$_c"; fi
    done
    printf '/%s' "$_res"
}
hermes_default_home() {
    # The home used when HERMES_HOME is unset: the platform account home with
    # the data-dir suffix appended LITERALLY (_get_platform_default_hermes_home).
    # HOME is used verbatim when SET and non-empty; an EMPTY HOME makes
    # posixpath.expanduser("~") answer "/", so the default is "/.hermes<suffix>";
    # an unset HOME resolves through the account database; a missing account
    # keeps the tilde the way expanduser does.
    local _suffix="${HERMES_DATA_DIR_SUFFIX:-}" _account_home
    if [ "${HOME+x}" = x ]; then
        if [ -n "$HOME" ]; then
            printf '%s' "$HOME/.hermes$_suffix"
        else
            printf '%s' "/.hermes$_suffix"
        fi
        return
    fi
    if _account_home="$(_hermes_user_home "$(id -un 2>/dev/null)" 2>/dev/null)"; then
        printf '%s' "$_account_home/.hermes$_suffix"
        return
    fi
    printf '%s' "~/.hermes$_suffix"
}
hermes_root_of() {
    # Containment is decided on RESOLVED paths but the lexical form is returned,
    # mirroring get_default_hermes_root() (its non-fold answer keeps "..").
    local _default_lex _default_res _lex _res _raw
    # The default is the platform account home + LITERAL suffix; never run it
    # through _hermes_expand (a $ in the suffix would expand; pm keeps it raw).
    _default_lex="$(_hermes_norm "$(hermes_default_home)")"
    _default_res="$(_hermes_resolve "$_default_lex")"
    # pm strips the RAW env home before expanding it; whitespace-only folds to default.
    _raw="${HERMES_HOME-}"
    _raw="${_raw#"${_raw%%[![:space:]]*}"}"
    _raw="${_raw%"${_raw##*[![:space:]]}"}"
    if [ -n "$_raw" ]; then
        _lex="$(_hermes_norm "$(_hermes_expand "$_raw")")"
    else
        _lex="$_default_lex"
    fi
    _res="$(_hermes_resolve "$_lex")"
    case "$_res/" in
        "$_default_res"/*) _lex="$_default_lex" ;;
        *)
            local _parent="${_lex%/*}"
            if [ "${_parent##*/}" = profiles ] && [ "$_parent" != "$_lex" ]; then
                if [ "$_parent" = profiles ]; then _lex=".";
                elif [ "$_parent" = /profiles ]; then _lex=/;
                else _lex="${_parent%/profiles}"; fi
            fi
            ;;
    esac
    # "." like Path("."): an empty root would make "${HERMES_ROOT}/tools" the fs root.
    [ -z "$_lex" ] && _lex="."
    printf '%s' "$_lex"
}
# --- END store-root resolver ---


hermes_root="$(hermes_root_of)"
store="${HERMES_RUNTIME_DIR:-$hermes_root/tools}"
entry="$store/uv-$uv_version-$target"
uv="$entry/uv"; [ "$os" = win32 ] && uv="$entry/uv.exe"

if [ -x "$uv" ]; then
  echo -e "${GREEN}✓${NC} pinned uv found ($("$uv" --version 2>/dev/null))"
else
  url="$(pin url)"; sha="$(pin sha256)"
  [ -n "$url" ] && [ -n "$sha" ] || { echo -e "${RED}✗${NC} no uv artifact for $target" >&2; exit 1; }
  echo -e "${CYAN}→${NC} Staging pinned uv $uv_version ($target) into the pm store..."
  mkdir -p "$store"
  tmp="$(mktemp -d "$store/.bootstrap-XXXXXX")"; trap 'rm -rf "$tmp"' EXIT
  archive="$tmp/${url##*/}"
  fetch_pinned() {
    if curl -fsSL -o "$2" "$1"; then return 0; else _curl_status=$?; fi
    case "$_curl_status" in 5|6|7|18|22|28|52|55|56) ;; *) return "$_curl_status" ;; esac
    local mirror; mirror="$(mirror_url_for "$sha")" || return 1
    curl -fsSL -o "$2" "$mirror" || return 1
  }
  if ! fetch_pinned "$url" "$archive"; then
    _tried="$url"
    if _m="$(mirror_url_for "$sha")"; then _tried="$_tried or $_m"; fi
    echo -e "${RED}✗${NC} failed to download pinned uv from $_tried" >&2
    exit 1
  fi
  got="$( (sha256sum "$archive" 2>/dev/null || shasum -a 256 "$archive") | cut -d' ' -f1 | tr -d '\\')"
  [ "$got" = "$sha" ] || { echo -e "${RED}✗${NC} sha256 mismatch for uv (got $got, pinned $sha)" >&2; exit 1; }
  mkdir -p "$tmp/tree"
  case "$archive" in
    *.zip) unzip -q "$archive" -d "$tmp/tree" ;;
    *) tar -xzf "$archive" -C "$tmp/tree" ;;
  esac
  # flatten a single wrapping dir (uv tarballs ship uv-<triple>/uv).
  # BSD find lacks GNU's -mindepth/-maxdepth flags, so enumerate children in
  # the shell; the two dot globs include hidden entries without matching . or ..
  inner= inner_count=0
  for child in "$tmp/tree"/* "$tmp/tree"/.[!.]* "$tmp/tree"/..?*; do
    [ -e "$child" ] || [ -L "$child" ] || continue
    inner="$child"
    inner_count=$((inner_count + 1))
  done
  if [ "$inner_count" = 1 ] && [ -d "$inner" ]; then
    mv "$inner" "$tmp/entry"
  else
    mv "$tmp/tree" "$tmp/entry"
  fi
  rm -rf "$entry"
  mv "$tmp/entry" "$entry"
  echo -e "${GREEN}✓${NC} uv installed ($("$uv" --version 2>/dev/null))"
fi

# ============================================================================
# Delegate to pm: python + venv + tool store + hash-verified venv sync
# ============================================================================

echo -e "${CYAN}→${NC} Installing python + tools + dependencies via pm (hash-verified via uv.lock)..."
echo -e "${CYAN}→${NC} (first run on a fresh checkout can take 1-5 minutes)"
# PM can replace its uv entry only after the bootstrap uv has exited.
# A bare version lets uv pick emulated x86_64 on Windows-on-ARM.
py_request="$py_version"
if [ "$os" = win32 ]; then
  case "$arch" in arm64) py_request="cpython-$py_version-windows-aarch64-none" ;;
                  *) py_request="cpython-$py_version-windows-x86_64-none" ;; esac
fi
# uv's default state (~/.cache/uv, ~/.local/share/uv) belongs to the USER's
# uv (#101269); pin both to the Hermes root. The cache keeps its OWN slot -
# pm seeds <root>/cache/uv once, skipping entries that already exist, so
# bootstrap bytes there first would mark a partial seed done - while nothing
# seeds the python dir the `python find` below reads back.
export UV_CACHE_DIR="$hermes_root/cache/uv-bootstrap"
export UV_PYTHON_INSTALL_DIR="$hermes_root/cache/uv-python"
"$uv" python install --no-bin --no-registry "$py_request"
boot_py="$("$uv" python find --managed-python "$py_request")"
boot_py="${boot_py%$'\r'}"
# Activation trusts the recorded tool digest; a direct setup re-checks it
# (setup-hermes.ps1 draws the same line).
pm_args=("$test_environment")
[ "$runtime_only" = true ] && pm_args+=(--trust-recorded)
if ! "$boot_py" -m pm.cli install "${pm_args[@]}"; then
    echo -e "${RED}✗${NC} pm install failed — see output above."
    exit 1
fi
echo -e "${GREEN}✓${NC} Tools + dependencies installed (hash-verified via pm + uv.lock)"

if [ "$runtime_only" = true ]; then
    exit 0
fi

# ============================================================================
# Environment file
# ============================================================================

if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        # .env holds API keys — restrict to owner-only access (matches
        # scripts/install.sh which already chmods 600 after creation).
        chmod 600 .env 2>/dev/null || true
        echo -e "${GREEN}✓${NC} Created .env from template"
    fi
else
    # Tighten an existing .env's perms in case it was created elsewhere
    # under a permissive umask.
    chmod 600 .env 2>/dev/null || true
    echo -e "${GREEN}✓${NC} .env exists"
fi

# ============================================================================
# Publish user-facing launchers
# ============================================================================

echo -e "${CYAN}→${NC} Setting up hermes command..."

# Reuse the bootstrap interpreter only to run the shared launcher writer.
bin_dir="$HOME/.local/bin"
if [ "$os" = win32 ]; then
    bin_dir="$(cygpath -am "${HERMES_HOME:-${LOCALAPPDATA:-$HOME/AppData/Local}/hermes}/bin")"
fi
if ! "$boot_py" -I -X utf8 hermes_cli/_launchers.py "$bin_dir"; then
    echo -e "${RED}✗${NC} launcher publication failed" >&2
    exit 1
fi
echo -e "${GREEN}✓${NC} Published Hermes commands in $bin_dir"

if [ "$os" != win32 ]; then
    # Determine the appropriate shell config file
    SHELL_CONFIG=""
    if [[ "$SHELL" == *"zsh"* ]]; then
            SHELL_CONFIG="$HOME/.zshrc"
        elif [[ "$SHELL" == *"bash"* ]]; then
            SHELL_CONFIG="$HOME/.bashrc"
            [ ! -f "$SHELL_CONFIG" ] && SHELL_CONFIG="$HOME/.bash_profile"
        else
            # Fallback to checking existing files
            if [ -f "$HOME/.zshrc" ]; then
                SHELL_CONFIG="$HOME/.zshrc"
            elif [ -f "$HOME/.bashrc" ]; then
                SHELL_CONFIG="$HOME/.bashrc"
            elif [ -f "$HOME/.bash_profile" ]; then
                SHELL_CONFIG="$HOME/.bash_profile"
            fi
        fi

        if [ -n "$SHELL_CONFIG" ]; then
            # Touch the file just in case it doesn't exist yet but was selected
            touch "$SHELL_CONFIG" 2>/dev/null || true

            if ! echo "$PATH" | tr ':' '\n' | grep -q "^$HOME/.local/bin$"; then
                if ! grep -q '\.local/bin' "$SHELL_CONFIG" 2>/dev/null; then
                    echo "" >> "$SHELL_CONFIG"
                    echo "# Hermes Agent — ensure ~/.local/bin is on PATH" >> "$SHELL_CONFIG"
                    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_CONFIG"
                    echo -e "${GREEN}✓${NC} Added ~/.local/bin to PATH in $SHELL_CONFIG"
                else
                    echo -e "${GREEN}✓${NC} ~/.local/bin already in $SHELL_CONFIG"
                fi
            else
                echo -e "${GREEN}✓${NC} ~/.local/bin already on PATH"
            fi
        fi
fi

# ============================================================================
# Seed bundled skills into ~/.hermes/skills/
# ============================================================================

HERMES_SKILLS_DIR="${HERMES_HOME:-$HOME/.hermes}/skills"
mkdir -p "$HERMES_SKILLS_DIR"

echo ""
echo "Syncing bundled skills to ~/.hermes/skills/ ..."
if "$boot_py" -m tools.skills_sync 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Skills synced"
else
    # Fallback: copy if sync script fails (missing deps, etc.)
    if [ -d "$SCRIPT_DIR/skills" ]; then
        cp -rn "$SCRIPT_DIR/skills/"* "$HERMES_SKILLS_DIR/" 2>/dev/null || true
        echo -e "${GREEN}✓${NC} Skills copied"
    fi
fi

# ============================================================================
# Done
# ============================================================================

echo ""
echo -e "${GREEN}✓ Setup complete!${NC}"
echo ""
echo "Next steps:"
echo ""
echo "  1. Activate the dev environment (venv-style, in THIS shell):"
echo "     source ./activate"
echo ""
echo "  2. Run the setup wizard to configure API keys:"
echo "     hermes setup"
echo ""
echo "  3. Start chatting:"
echo "     hermes"
echo ""
echo "Other commands:"
echo "  hermes pm install     # Re-run the tool + dependency install"
echo "  hermes status         # Check configuration"
echo "  hermes doctor         # Diagnose issues"
echo "  deactivate            # Undo the activation (restore PATH etc.)"
echo ""
