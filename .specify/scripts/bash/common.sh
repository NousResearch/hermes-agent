#!/usr/bin/env bash
# Common functions and variables for all scripts

# Find repository root by searching upward for .specify directory
# This is the primary marker for spec-kit projects
find_specify_root() {
    local dir="${1:-$(pwd)}"
    # Normalize to absolute path to prevent infinite loop with relative paths
    # Use -- to handle paths starting with - (e.g., -P, -L)
    dir="$(cd -- "$dir" 2>/dev/null && pwd)" || return 1
    local prev_dir=""
    while true; do
        if [ -d "$dir/.specify" ]; then
            echo "$dir"
            return 0
        fi
        # Stop if we've reached filesystem root or dirname stops changing
        if [ "$dir" = "/" ] || [ "$dir" = "$prev_dir" ]; then
            break
        fi
        prev_dir="$dir"
        dir="$(dirname "$dir")"
    done
    return 1
}

# Resolve an explicit SPECIFY_INIT_DIR project override (the directory that
# *contains* .specify/), for non-interactive / CI use — e.g. running a Spec Kit
# command against a member project from a monorepo root without cd.
#
# Precondition: SPECIFY_INIT_DIR is non-empty. Echoes the validated absolute
# project root, or prints an error and returns 1. Strict by design: the path
# must exist and contain .specify/, with no silent fallback to cwd or the
# script-location default (which would silently write to the wrong project).
#
# This is the single resolver: bundled extensions inherit it by sourcing core
# (e.g. the git extension's create-new-feature-branch) rather than duplicating it.
resolve_specify_init_dir() {
    local init_root
    # Normalize: relative paths resolve against $(pwd); a trailing slash collapses.
    # CDPATH="" so a relative value cannot be resolved against the caller's CDPATH
    # (which would also echo to stdout and corrupt the captured path).
    if ! init_root="$(CDPATH="" cd -- "$SPECIFY_INIT_DIR" 2>/dev/null && pwd)"; then
        echo "ERROR: SPECIFY_INIT_DIR does not point to an existing directory: $SPECIFY_INIT_DIR" >&2
        return 1
    fi
    if [[ ! -d "$init_root/.specify" ]]; then
        echo "ERROR: SPECIFY_INIT_DIR is not a Spec Kit project (no .specify/ directory): $init_root" >&2
        return 1
    fi
    printf '%s\n' "$init_root"
}

# Get repository root, prioritizing .specify directory
# This prevents using a parent repository when spec-kit is initialized in a subdirectory
get_repo_root() {
    # Explicit project override wins (see resolve_specify_init_dir).
    if [[ -n "${SPECIFY_INIT_DIR:-}" ]]; then
        resolve_specify_init_dir
        return
    fi

    # First, look for .specify directory (spec-kit's own marker)
    local specify_root
    if specify_root=$(find_specify_root); then
        echo "$specify_root"
        return
    fi

    # Final fallback to script location
    local script_dir="$(CDPATH="" cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    (cd "$script_dir/../../.." && pwd)
}

# Get current feature name from explicit state only.
# Returns the feature identifier or empty string if none is set.
# Feature state is set by SPECIFY_FEATURE (from create-new-feature or
# the git extension) or implicitly via .specify/feature.json.
get_current_branch() {
    if [[ -n "${SPECIFY_FEATURE:-}" ]]; then
        echo "$SPECIFY_FEATURE"
        return
    fi

    # No explicit feature set — caller must handle this via feature.json
    # in get_feature_paths(). Return empty to signal "unknown".
    echo ""
}

# Safely read .specify/feature.json's "feature_directory" value.
# Prints the raw value (possibly relative) to stdout, or empty string if the file
# is missing, unparseable, or does not contain the key. Always returns 0 so callers
# under `set -e` cannot be aborted by parser failure.
# Parser order mirrors the historical get_feature_paths behavior: jq -> python3 -> grep/sed.
read_feature_json_feature_directory() {
    local repo_root="$1"
    local fj="$repo_root/.specify/feature.json"
    [[ -f "$fj" ]] || { printf '%s' ''; return 0; }

    # Try parsers in order (jq -> python3 -> grep/sed), falling through on
    # failure. Selection is by *parse success*, not mere availability: on
    # Windows `python3` commonly resolves to the Microsoft Store App Execution
    # Alias stub, which passes `command -v` but fails at runtime (exit 49), so
    # an availability-gated `elif` would pick python3, swallow its failure, and
    # never reach the grep/sed fallback -- leaving feature.json unreadable even
    # though it is valid (issue #3304).
    local _fd=''
    if command -v jq >/dev/null 2>&1; then
        if ! _fd=$(jq -r '.feature_directory // empty' "$fj" 2>/dev/null); then
            _fd=''
        fi
    fi
    if [[ -z "$_fd" ]] && command -v python3 >/dev/null 2>&1; then
        # Use Python so pretty-printed/multi-line JSON still parses correctly.
        if ! _fd=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); v=d.get('feature_directory'); print(v if v else '')" "$fj" 2>/dev/null); then
            _fd=''
        fi
    fi
    if [[ -z "$_fd" ]]; then
        # Last-resort single-line grep/sed fallback. The `|| true` guards against
        # grep returning 1 (no match) aborting under `set -e` / `pipefail`.
        _fd=$( { grep -E '"feature_directory"[[:space:]]*:' "$fj" 2>/dev/null || true; } \
            | head -n 1 \
            | sed -E 's/^[^:]*:[[:space:]]*"([^"]*)".*$/\1/' )
    fi

    printf '%s' "$_fd"
    return 0
}

# Persist a feature_directory value to .specify/feature.json.
# Writes only when the file is missing or the value differs from what's stored.
# Accepts the raw (possibly relative) path — callers should pass the original
# user-supplied value, not the normalized absolute path.
_persist_feature_json() {
    local repo_root="$1"
    local feature_dir_value="$2"
    local fj="$repo_root/.specify/feature.json"

    # Strip repo_root prefix if the value is absolute and under repo_root
    if [[ "$feature_dir_value" == "$repo_root/"* ]]; then
        feature_dir_value="${feature_dir_value#"$repo_root/"}"
    fi

    # Read current value (if any) and skip write when unchanged
    local current_val
    current_val=$(read_feature_json_feature_directory "$repo_root")
    if [[ "$current_val" == "$feature_dir_value" ]]; then
        return 0
    fi

    # Ensure .specify/ directory exists
    mkdir -p "$repo_root/.specify"

    # Write feature.json — prefer jq for safe JSON, fall back to printf
    if command -v jq >/dev/null 2>&1; then
        jq -cn --arg fd "$feature_dir_value" '{feature_directory:$fd}' > "$fj"
    else
        printf '{"feature_directory":"%s"}\n' "$(json_escape "$feature_dir_value")" > "$fj"
    fi
}

get_feature_paths() {
    # Read-only callers (e.g. check-prerequisites.sh --paths-only) pass
    # --no-persist so pure path resolution never writes .specify/feature.json,
    # which would dirty the working tree or overwrite a pinned value (issue #3025).
    local no_persist=false
    if [[ "${1:-}" == "--no-persist" ]]; then
        no_persist=true
        shift
    fi

    # Split decl/assignment so a SPECIFY_INIT_DIR validation failure in
    # get_repo_root propagates as a hard error instead of being masked by `local`.
    local repo_root
    repo_root=$(get_repo_root) || return 1
    local current_branch
    current_branch=$(get_current_branch)

    # Resolve feature directory.  Priority:
    #   1. SPECIFY_FEATURE_DIRECTORY env var (explicit override)
    #   2. .specify/feature.json "feature_directory" key (persisted by specify command)
    #   3. Error — no feature context available
    local feature_dir
    if [[ -n "${SPECIFY_FEATURE_DIRECTORY:-}" ]]; then
        feature_dir="$SPECIFY_FEATURE_DIRECTORY"
        # Normalize relative paths to absolute under repo root
        [[ "$feature_dir" != /* ]] && feature_dir="$repo_root/$feature_dir"
        # Persist to feature.json so future sessions without the env var still
        # work — unless the caller opted out for read-only resolution (#3025).
        if [[ "$no_persist" != true ]]; then
            _persist_feature_json "$repo_root" "$SPECIFY_FEATURE_DIRECTORY"
        fi
    elif [[ -f "$repo_root/.specify/feature.json" ]]; then
        local _fd
        _fd=$(read_feature_json_feature_directory "$repo_root")
        if [[ -n "$_fd" ]]; then
            feature_dir="$_fd"
            # Normalize relative paths to absolute under repo root
            [[ "$feature_dir" != /* ]] && feature_dir="$repo_root/$feature_dir"
        else
            echo "ERROR: Feature directory not found. Set SPECIFY_FEATURE_DIRECTORY or ensure .specify/feature.json contains feature_directory." >&2
            return 1
        fi
    else
        echo "ERROR: Feature directory not found. Set SPECIFY_FEATURE_DIRECTORY or run the specify command to create .specify/feature.json." >&2
        return 1
    fi

    # When no branch context exists (no SPECIFY_FEATURE, feature resolved via
    # SPECIFY_FEATURE_DIRECTORY or feature.json), fall back to the feature
    # directory basename so CURRENT_BRANCH is a usable identifier rather than
    # an empty, misleading value (issue #3026).
    if [[ -z "$current_branch" ]]; then
        local feature_dir_trimmed="${feature_dir%/}"
        current_branch="${feature_dir_trimmed##*/}"
    fi

    # Use printf '%q' to safely quote values, preventing shell injection
    # via crafted branch names or paths containing special characters
    printf 'REPO_ROOT=%q\n' "$repo_root"
    printf 'CURRENT_BRANCH=%q\n' "$current_branch"
    printf 'FEATURE_DIR=%q\n' "$feature_dir"
    printf 'FEATURE_SPEC=%q\n' "$feature_dir/spec.md"
    printf 'IMPL_PLAN=%q\n' "$feature_dir/plan.md"
    printf 'TASKS=%q\n' "$feature_dir/tasks.md"
    printf 'RESEARCH=%q\n' "$feature_dir/research.md"
    printf 'DATA_MODEL=%q\n' "$feature_dir/data-model.md"
    printf 'QUICKSTART=%q\n' "$feature_dir/quickstart.md"
    printf 'CONTRACTS_DIR=%q\n' "$feature_dir/contracts"
}

# Check if jq is available for safe JSON construction
has_jq() {
    command -v jq >/dev/null 2>&1
}

get_invoke_separator() {
    local repo_root="${1:-$(get_repo_root)}"
    if [[ "${_SPECIFY_INVOKE_SEPARATOR_CACHE_REPO_ROOT:-}" == "$repo_root" && -n "${_SPECIFY_INVOKE_SEPARATOR_CACHE_VALUE:-}" ]]; then
        printf '%s\n' "$_SPECIFY_INVOKE_SEPARATOR_CACHE_VALUE"
        return 0
    fi

    local integration_json="$repo_root/.specify/integration.json"
    local separator="."
    local parsed=0

    if [[ -f "$integration_json" ]]; then
        # Try parsers in order (jq -> python3 -> awk), falling through on
        # failure. Selection is by *parse success*, not mere availability: on
        # Windows `python3` commonly resolves to the Microsoft Store App
        # Execution Alias stub, which passes `command -v` but fails at runtime
        # (exit 49). An availability-gated branch would pick python3, swallow
        # its failure, and — because this function historically had no text
        # fallback — silently return "." even for `-`-separator integrations
        # (e.g. forge, cline), yielding wrong command hints (issue #3304).
        if command -v jq >/dev/null 2>&1; then
            local jq_separator
            if jq_separator=$(jq -r '(.default_integration // .integration // "") as $k | if $k == "" then "." else (.integration_settings[$k].invoke_separator // ".") end' "$integration_json" 2>/dev/null); then
                case "$jq_separator" in
                    "."|"-") separator="$jq_separator"; parsed=1 ;;
                esac
            fi
        fi

        if [[ "$parsed" -eq 0 ]] && command -v python3 >/dev/null 2>&1; then
            local py_separator
            if py_separator=$(python3 - "$integration_json" <<'PY' 2>/dev/null
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as fh:
        state = json.load(fh)
    key = state.get("default_integration") or state.get("integration") or ""
    settings = state.get("integration_settings")
    separator = "."
    if isinstance(key, str) and isinstance(settings, dict):
        entry = settings.get(key)
        if isinstance(entry, dict) and entry.get("invoke_separator") in {".", "-"}:
            separator = entry["invoke_separator"]
    print(separator)
except Exception:
    sys.exit(1)
PY
); then
                case "$py_separator" in
                    "."|"-") separator="$py_separator"; parsed=1 ;;
                esac
            fi
        fi

        if [[ "$parsed" -eq 0 ]]; then
            # Last-resort text fallback for environments with neither jq nor a
            # working python3 (e.g. stock Windows + Git Bash). Reads the active
            # integration key (default_integration, else integration) and its
            # invoke_separator from within the integration_settings object.
            # Handles both pretty-printed (the written form) and compact JSON.
            # Accumulate all lines into one buffer in END rather than using
            # gawk-only whole-file slurp (RS="^$"), so this stays portable to
            # the BSD awk on macOS.
            local awk_separator
            awk_separator=$(awk '
                function keyval(d, name,   v) {
                    if (match(d, "\"" name "\"[ \t\r\n]*:[ \t\r\n]*\"[^\"]*\"")) {
                        v=substr(d,RSTART,RLENGTH); sub(/^.*:[ \t\r\n]*"/,"",v); sub(/"$/,"",v); return v
                    }
                    return ""
                }
                { doc = doc $0 "\n" }
                END {
                    key=keyval(doc,"default_integration"); if (key=="") key=keyval(doc,"integration")
                    sep="."
                    if (key!="") {
                        settings=doc
                        if (match(doc, /"integration_settings"[ \t\r\n]*:[ \t\r\n]*[{]/)) {
                            settings=substr(doc, RSTART+RLENGTH-1)
                        }
                        if (match(settings, "\"" key "\"[ \t\r\n]*:[ \t\r\n]*[{]")) {
                            start=RSTART+RLENGTH-1
                            depth=0
                            obj=""
                            for (i=start; i<=length(settings); i++) {
                                c=substr(settings,i,1)
                                obj=obj c
                                if (c=="{") depth++
                                else if (c=="}") { depth--; if (depth==0) break }
                            }
                            if (match(obj, /"invoke_separator"[ \t\r\n]*:[ \t\r\n]*"[-.]"/)) {
                                tok=substr(obj,RSTART,RLENGTH); s=substr(tok,length(tok)-1,1)
                                if (s=="." || s=="-") sep=s
                            }
                        }
                    }
                    print sep
                }
            ' "$integration_json" 2>/dev/null)
            case "$awk_separator" in
                "."|"-") separator="$awk_separator" ;;
            esac
        fi
    fi

    _SPECIFY_INVOKE_SEPARATOR_CACHE_REPO_ROOT="$repo_root"
    _SPECIFY_INVOKE_SEPARATOR_CACHE_VALUE="$separator"
    printf '%s\n' "$separator"
}

format_speckit_command() {
    local command_name="$1"
    local repo_root="${2:-$(get_repo_root)}"
    local separator
    if [[ "${_SPECIFY_INVOKE_SEPARATOR_CACHE_REPO_ROOT:-}" == "$repo_root" && -n "${_SPECIFY_INVOKE_SEPARATOR_CACHE_VALUE:-}" ]]; then
        separator="$_SPECIFY_INVOKE_SEPARATOR_CACHE_VALUE"
    else
        separator=$(get_invoke_separator "$repo_root")
        _SPECIFY_INVOKE_SEPARATOR_CACHE_REPO_ROOT="$repo_root"
        _SPECIFY_INVOKE_SEPARATOR_CACHE_VALUE="$separator"
    fi

    command_name="${command_name#/}"
    command_name="${command_name#speckit.}"
    command_name="${command_name#speckit-}"
    command_name="${command_name//./$separator}"

    printf '/speckit%s%s\n' "$separator" "$command_name"
}

# Escape a string for safe embedding in a JSON value (fallback when jq is unavailable).
# Handles backslash, double-quote, and JSON-required control character escapes (RFC 8259).
json_escape() {
    local s="$1"
    s="${s//\\/\\\\}"
    s="${s//\"/\\\"}"
    s="${s//$'\n'/\\n}"
    s="${s//$'\t'/\\t}"
    s="${s//$'\r'/\\r}"
    s="${s//$'\b'/\\b}"
    s="${s//$'\f'/\\f}"
    # Escape any remaining U+0001-U+001F control characters as \uXXXX.
    # (U+0000/NUL cannot appear in bash strings and is excluded.)
    # LC_ALL=C ensures ${#s} counts bytes and ${s:$i:1} yields single bytes,
    # so multi-byte UTF-8 sequences (first byte >= 0xC0) pass through intact.
    local LC_ALL=C
    local i char code
    for (( i=0; i<${#s}; i++ )); do
        char="${s:$i:1}"
        printf -v code '%d' "'$char" 2>/dev/null || code=256
        if (( code >= 1 && code <= 31 )); then
            printf '\\u%04x' "$code"
        else
            printf '%s' "$char"
        fi
    done
}

check_file() { [[ -f "$1" ]] && echo "  ✓ $2" || echo "  ✗ $2"; }
check_dir() { [[ -d "$1" && -n $(ls -A "$1" 2>/dev/null) ]] && echo "  ✓ $2" || echo "  ✗ $2"; }

_python3_command() {
    if command -v python3 >/dev/null 2>&1 &&
        python3 -c 'import sys; raise SystemExit(sys.version_info.major != 3)' >/dev/null 2>&1; then
        printf '%s\n' "python3"
    elif command -v python >/dev/null 2>&1 &&
        python -c 'import sys; raise SystemExit(sys.version_info.major != 3)' >/dev/null 2>&1; then
        printf '%s\n' "python"
    elif command -v py >/dev/null 2>&1 &&
        py -3 -c 'import sys' >/dev/null 2>&1; then
        printf '%s\n' "py -3"
    else
        return 1
    fi
}

_sorted_extension_ids() {
    local ext_dir="$1"
    local python_spec
    if python_spec=$(_python3_command); then
        local -a python_cmd
        read -r -a python_cmd <<< "$python_spec"
        local py_stderr sorted_ids
        py_stderr=$(mktemp)
        if sorted_ids=$(SPECKIT_EXTENSIONS="$ext_dir" "${python_cmd[@]}" -c "
import json, os, re, sys
from pathlib import Path

root = Path(os.environ['SPECKIT_EXTENSIONS'])
registered = {}
registry = root / '.registry'
if os.path.lexists(registry):
    if not registry.is_file():
        print('registry_invalid: not a regular file', file=sys.stderr)
        sys.exit(1)
    try:
        data = json.loads(registry.read_text(encoding='utf-8'))
    except Exception as exc:
        print('registry_invalid: ' + str(exc), file=sys.stderr)
        sys.exit(1)
    if not isinstance(data, dict):
        print('registry_invalid: root must be a mapping', file=sys.stderr)
        sys.exit(1)
    raw_extensions = data.get('extensions', {})
    if not isinstance(raw_extensions, dict):
        print('registry_invalid: extensions must be a mapping', file=sys.stderr)
        sys.exit(1)
    registered = raw_extensions

def priority(value):
    if isinstance(value, bool):
        return 10
    try:
        parsed = int(value)
        return parsed if parsed >= 1 else 10
    except (TypeError, ValueError, OverflowError):
        return 10

ranked = []
for ext_id, meta in registered.items():
    if isinstance(ext_id, str) and re.fullmatch(r'[a-z0-9-]+', ext_id) and isinstance(meta, dict) and bool(meta.get('enabled', True)):
        ranked.append((priority(meta.get('priority')), ext_id))
for path in root.iterdir():
    if path.is_dir() and re.fullmatch(r'[a-z0-9-]+', path.name) and path.name not in registered:
        ranked.append((10, path.name))
for _, ext_id in sorted(ranked):
    print(ext_id)
" 2>"$py_stderr"); then
            rm -f "$py_stderr"
            printf '%s\n' "$sorted_ids"
            return 0
        else
            echo "Error: invalid extension registry $ext_dir/.registry" >&2
            rm -f "$py_stderr"
            return 1
        fi
    fi

    if [ -e "$ext_dir/.registry" ] || [ -L "$ext_dir/.registry" ]; then
        if [ ! -f "$ext_dir/.registry" ] || [ ! -r "$ext_dir/.registry" ]; then
            echo "Error: invalid extension registry $ext_dir/.registry" >&2
            return 1
        fi
        echo "Error: Python 3 is required to honor the extension registry" >&2
        return 2
    fi

    local ext extension_id
    for ext in "$ext_dir"/*/; do
        [ -d "$ext" ] || continue
        extension_id=$(basename "$ext")
        case "$extension_id" in *[!a-z0-9-]*) continue ;; esac
        printf '%s\n' "$extension_id"
    done
}

# Resolve a template name to a file path using the priority stack:
#   1. .specify/templates/overrides/
#   2. .specify/presets/<preset-id>/templates/ (sorted by priority from .registry)
#   3. .specify/extensions/<ext-id>/templates/
#   4. .specify/templates/ (core)
resolve_template() {
    local template_name="$1"
    local repo_root="$2"
    local base="$repo_root/.specify/templates"

    case "$template_name" in ""|*[!a-z0-9-]*) return 1 ;; esac

    # Priority 1: Project overrides
    local override="$base/overrides/${template_name}.md"
    [ -f "$override" ] && echo "$override" && return 0

    # Priority 2: Installed presets (sorted by priority from .registry)
    local presets_dir="$repo_root/.specify/presets"
    if [ -d "$presets_dir" ]; then
        local registry_file="$presets_dir/.registry"
        local python_spec=""
        local -a python_cmd=()
        if python_spec=$(_python3_command); then
            read -r -a python_cmd <<< "$python_spec"
        fi
        if [ -f "$registry_file" ] && [ "${#python_cmd[@]}" -gt 0 ]; then
            # Read preset IDs sorted by priority (lower number = higher precedence).
            # The python3 call is wrapped in an if-condition so that set -e does not
            # abort the function when python3 exits non-zero (e.g. invalid JSON).
            local sorted_presets=""
            if sorted_presets=$(SPECKIT_REGISTRY="$registry_file" "${python_cmd[@]}" -c "
import json, re, sys, os
try:
    with open(os.environ['SPECKIT_REGISTRY'], encoding='utf-8') as f:
        data = json.load(f)
    presets = data.get('presets', {})
    def priority(meta):
        if not isinstance(meta, dict) or isinstance(meta.get('priority'), bool):
            return 10
        try:
            value = int(meta.get('priority', 10))
            return value if value >= 1 else 10
        except (TypeError, ValueError, OverflowError):
            return 10
    for pid, meta in sorted(presets.items(), key=lambda x: (priority(x[1]), x[0])):
        if isinstance(meta, dict) and bool(meta.get('enabled', True)) and re.fullmatch(r'[a-z0-9-]+', pid):
            print(pid)
except Exception:
    sys.exit(1)
" 2>/dev/null); then
                if [ -n "$sorted_presets" ]; then
                    # python3 succeeded and returned preset IDs — search in priority order
                    while IFS= read -r preset_id; do
                        local candidate="$presets_dir/$preset_id/templates/${template_name}.md"
                        [ -f "$candidate" ] && echo "$candidate" && return 0
                        candidate="$presets_dir/$preset_id/${template_name}.md"
                        [ -f "$candidate" ] && echo "$candidate" && return 0
                    done <<< "$sorted_presets"
                fi
                # python3 succeeded but registry has no presets — nothing to search
            else
                # python3 failed (missing, or registry parse error) — fall back to unordered directory scan
                for preset in "$presets_dir"/*/; do
                    [ -d "$preset" ] || continue
                    local candidate="$preset/templates/${template_name}.md"
                    [ -f "$candidate" ] && echo "$candidate" && return 0
                    candidate="$preset/${template_name}.md"
                    [ -f "$candidate" ] && echo "$candidate" && return 0
                done
            fi
        else
            # Fallback: alphabetical directory order (no python3 available)
            for preset in "$presets_dir"/*/; do
                [ -d "$preset" ] || continue
                local candidate="$preset/templates/${template_name}.md"
                [ -f "$candidate" ] && echo "$candidate" && return 0
                candidate="$preset/${template_name}.md"
                [ -f "$candidate" ] && echo "$candidate" && return 0
            done
        fi
    fi

    # Priority 3: Extension-provided templates
    local ext_dir="$repo_root/.specify/extensions"
    if [ -d "$ext_dir" ]; then
        local sorted_extensions=""
        if ! sorted_extensions=$(_sorted_extension_ids "$ext_dir"); then
            return 2
        fi
        while IFS= read -r extension_id; do
            [ -n "$extension_id" ] || continue
            local ext="$ext_dir/$extension_id"
            local candidate="$ext/templates/${template_name}.md"
            [ -f "$candidate" ] || candidate="$ext/${template_name}.md"
            [ -f "$candidate" ] && echo "$candidate" && return 0
        done <<< "$sorted_extensions"
    fi

    # Priority 4: Core templates
    local core="$base/${template_name}.md"
    [ -f "$core" ] && echo "$core" && return 0

    # Template not found in any location.
    # Return 1 so callers can distinguish "not found" from "found".
    # Callers running under set -e should use: TEMPLATE=$(resolve_template ...) || true
    return 1
}

# Resolve a template name to composed content using composition strategies.
# Reads strategy metadata from preset manifests and composes content
# from multiple layers using prepend, append, or wrap strategies.
#
# Usage: CONTENT=$(resolve_template_content "template-name" "$REPO_ROOT")
# Returns composed content string on stdout; exit code 1 if not found.
resolve_template_content() {
    local template_name="$1"
    local repo_root="$2"
    local base="$repo_root/.specify/templates"

    case "$template_name" in ""|*[!a-z0-9-]*) return 1 ;; esac

    # Collect all layers (highest priority first)
    local -a layer_paths=()
    local -a layer_strategies=()

    # Priority 1: Project overrides (always "replace")
    local override="$base/overrides/${template_name}.md"
    if [ -f "$override" ]; then
        if ! cat "$override"; then
            echo "Error: failed to read template layer $override" >&2
            return 2
        fi
        return 0
    fi

    local effective_base_found=false

    # Priority 2: Installed presets (sorted by priority from .registry)
    local presets_dir="$repo_root/.specify/presets"
    if [ -d "$presets_dir" ]; then
        local registry_file="$presets_dir/.registry"
        local sorted_presets=""
        local registry_parsed=false
        local python_spec=""
        local -a python_cmd=()
        if python_spec=$(_python3_command); then
            read -r -a python_cmd <<< "$python_spec"
        fi
        if [ -f "$registry_file" ] && [ "${#python_cmd[@]}" -gt 0 ]; then
            if sorted_presets=$(SPECKIT_REGISTRY="$registry_file" "${python_cmd[@]}" -c "
import json, re, sys, os
try:
    with open(os.environ['SPECKIT_REGISTRY'], encoding='utf-8') as f:
        data = json.load(f)
    presets = data.get('presets', {})
    def priority(meta):
        if not isinstance(meta, dict) or isinstance(meta.get('priority'), bool):
            return 10
        try:
            value = int(meta.get('priority', 10))
            return value if value >= 1 else 10
        except (TypeError, ValueError, OverflowError):
            return 10
    for pid, meta in sorted(presets.items(), key=lambda x: (priority(x[1]), x[0])):
        if isinstance(meta, dict) and bool(meta.get('enabled', True)) and re.fullmatch(r'[a-z0-9-]+', pid):
            print(pid)
except Exception:
    sys.exit(1)
" 2>/dev/null); then
                registry_parsed=true
            fi
        fi
        if [ "$registry_parsed" = false ]; then
            for preset in "$presets_dir"/*/; do
                [ -d "$preset" ] || continue
                local fallback_id
                fallback_id=$(basename "$preset")
                case "$fallback_id" in *[!a-z0-9-]*) continue ;; esac
                sorted_presets+="${sorted_presets:+$'\n'}$fallback_id"
            done
        fi

        if [ -n "$sorted_presets" ]; then
            while IFS= read -r preset_id; do
                local strategy="replace"
                local manifest_file=""
                local manifest="$presets_dir/$preset_id/preset.yml"
                local manifest_declared=false
                if [ -f "$manifest" ]; then
                    if [ "${#python_cmd[@]}" -eq 0 ]; then
                        echo "Error: Python 3 and PyYAML are required to resolve preset template composition" >&2
                        return 2
                    fi
                    local result
                    local py_stderr
                    local parse_status
                    py_stderr=$(mktemp)
                    if result=$(SPECKIT_MANIFEST="$manifest" SPECKIT_TMPL="$template_name" "${python_cmd[@]}" -c "
import sys, os
try:
    import yaml
except ImportError:
    print('yaml_missing', file=sys.stderr)
    sys.exit(2)
try:
    with open(os.environ['SPECKIT_MANIFEST'], encoding='utf-8') as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError('manifest root must be a mapping')
    if 'provides' not in data:
        raise ValueError('manifest missing provides section')
    provides = data['provides']
    if not isinstance(provides, dict):
        raise ValueError('manifest provides must be a mapping')
    if 'templates' not in provides:
        raise ValueError('manifest provides missing templates')
    templates = provides['templates']
    if not isinstance(templates, list):
        raise ValueError('manifest templates must be a list')
    if not templates:
        raise ValueError('manifest must provide at least one template')
    valid_types = ('template', 'command', 'script')
    valid_strategies = ('replace', 'prepend', 'append', 'wrap')
    for t in templates:
        if not isinstance(t, dict):
            raise ValueError('manifest template entries must be mappings')
        if 'type' not in t or 'name' not in t or 'file' not in t:
            raise ValueError('manifest template entry missing type, name, or file')
        for field in ('type', 'name', 'file'):
            if not isinstance(t[field], str):
                raise ValueError('manifest template ' + field + ' must be a string')
        if t['type'] not in valid_types:
            raise ValueError('invalid manifest template type')
        strategy = t.get('strategy', 'replace')
        if not isinstance(strategy, str):
            raise ValueError('manifest template strategy must be a string')
        strategy = strategy.lower()
        if strategy not in valid_strategies:
            raise ValueError('invalid manifest template strategy')
        if t['type'] == 'script' and strategy not in ('replace', 'wrap'):
            raise ValueError('invalid manifest script strategy')
    for t in templates:
        if t.get('name') == os.environ['SPECKIT_TMPL'] and t.get('type', 'template') == 'template':
            file_value = t.get('file', '')
            strategy = t.get('strategy', 'replace')
            print('found\t' + strategy + '\t' + file_value)
            sys.exit(0)
    print('absent\treplace\t')
except Exception as exc:
    print(f'manifest_invalid: {exc}', file=sys.stderr)
    sys.exit(3)
" 2>"$py_stderr"); then
                        parse_status=0
                    else
                        parse_status=$?
                    fi
                    if [ "$parse_status" -ne 0 ]; then
                        if [ "$parse_status" -eq 2 ]; then
                            echo "Error: PyYAML is required to resolve preset template composition" >&2
                        else
                            echo "Error: invalid preset manifest $manifest" >&2
                        fi
                        rm -f "$py_stderr"
                        return 2
                    fi
                    if [ -n "$result" ]; then
                        local declaration
                        IFS=$'\t' read -r declaration strategy manifest_file <<< "$result"
                        [ "$declaration" = "found" ] && manifest_declared=true
                        strategy=$(printf '%s' "$strategy" | tr '[:upper:]' '[:lower:]')
                    fi
                    rm -f "$py_stderr"
                fi

                local candidate=""
                if [ -n "$manifest_file" ]; then
                    case "$manifest_file" in
                        /*|*../*|../*) manifest_file="" ;;
                    esac
                fi
                if [ -n "$manifest_file" ]; then
                    local mf="$presets_dir/$preset_id/$manifest_file"
                    [ -f "$mf" ] && candidate="$mf"
                fi
                if [ -z "$candidate" ] && [ "$manifest_declared" = false ]; then
                    local cf="$presets_dir/$preset_id/templates/${template_name}.md"
                    [ -f "$cf" ] && candidate="$cf"
                    if [ -z "$candidate" ]; then
                        cf="$presets_dir/$preset_id/${template_name}.md"
                        [ -f "$cf" ] && candidate="$cf"
                    fi
                fi
                if [ -n "$candidate" ]; then
                    layer_paths+=("$candidate")
                    layer_strategies+=("$strategy")
                    if [ "$strategy" = "replace" ]; then
                        effective_base_found=true
                        break
                    fi
                fi
            done <<< "$sorted_presets"
        fi
    fi

    # Priority 3: Extension-provided templates (always "replace")
    local ext_dir="$repo_root/.specify/extensions"
    if [ "$effective_base_found" = false ] && [ -d "$ext_dir" ]; then
        local sorted_extensions=""
        if ! sorted_extensions=$(_sorted_extension_ids "$ext_dir"); then
            return 2
        fi
        while IFS= read -r extension_id; do
            [ -n "$extension_id" ] || continue
            local ext="$ext_dir/$extension_id"
            local candidate="$ext/templates/${template_name}.md"
            [ -f "$candidate" ] || candidate="$ext/${template_name}.md"
            if [ -f "$candidate" ]; then
                layer_paths+=("$candidate")
                layer_strategies+=("replace")
                effective_base_found=true
                break
            fi
        done <<< "$sorted_extensions"
    fi

    # Priority 4: Core templates (always "replace")
    local core="$base/${template_name}.md"
    if [ "$effective_base_found" = false ] && [ -f "$core" ]; then
        layer_paths+=("$core")
        layer_strategies+=("replace")
    fi

    local count=${#layer_paths[@]}
    [ "$count" -eq 0 ] && return 1

    # Check if any layer uses a non-replace strategy
    local has_composition=false
    for s in "${layer_strategies[@]}"; do
        [ "$s" != "replace" ] && has_composition=true && break
    done

    # If the top (highest-priority) layer is replace, it wins entirely —
    # lower layers are irrelevant regardless of their strategies.
    if [ "${layer_strategies[0]}" = "replace" ]; then
        if ! cat "${layer_paths[0]}"; then
            echo "Error: failed to read template layer ${layer_paths[0]}" >&2
            return 2
        fi
        return 0
    fi

    if [ "$has_composition" = false ]; then
        if ! cat "${layer_paths[0]}"; then
            echo "Error: failed to read template layer ${layer_paths[0]}" >&2
            return 2
        fi
        return 0
    fi

    # Find the effective base: scan from highest priority (index 0) downward
    # to find the nearest replace layer. Only compose layers above that base.
    local base_idx=-1
    local i
    for (( i=0; i<count; i++ )); do
        if [ "${layer_strategies[$i]}" = "replace" ]; then
            base_idx=$i
            break
        fi
    done

    if [ $base_idx -lt 0 ]; then
        echo "Error: template '$template_name' has composing layers but no replace base" >&2
        return 2
    fi

    # Read the base content; compose layers above the base (higher priority)
    local content
    if ! content=$(cat "${layer_paths[$base_idx]}"; status=$?; printf x; exit "$status"); then
        echo "Error: failed to read template layer ${layer_paths[$base_idx]}" >&2
        return 2
    fi
    content="${content%x}"

    for (( i=base_idx-1; i>=0; i-- )); do
        local path="${layer_paths[$i]}"
        local strat="${layer_strategies[$i]}"
        local layer_content
        # Preserve trailing newlines
        if ! layer_content=$(cat "$path"; status=$?; printf x; exit "$status"); then
            echo "Error: failed to read template layer $path" >&2
            return 2
        fi
        layer_content="${layer_content%x}"

        case "$strat" in
            replace) content="$layer_content" ;;
            prepend)
                content=$(printf '%s\n\n%s' "$layer_content" "$content"; printf x)
                content="${content%x}"
                ;;
            append)
                content=$(printf '%s\n\n%s' "$content" "$layer_content"; printf x)
                content="${content%x}"
                ;;
            wrap)
                case "$layer_content" in
                    *'{CORE_TEMPLATE}'*) ;;
                    *) echo "Error: wrap strategy missing {CORE_TEMPLATE} placeholder" >&2; return 2 ;;
                esac
                while [[ "$layer_content" == *'{CORE_TEMPLATE}'* ]]; do
                    local before="${layer_content%%\{CORE_TEMPLATE\}*}"
                    local after="${layer_content#*\{CORE_TEMPLATE\}}"
                    layer_content="${before}${content}${after}"
                done
                content="$layer_content"
                ;;
            *) echo "Error: unknown strategy '$strat'" >&2; return 2 ;;
        esac
    done

    printf '%s' "$content"
    return 0
}
