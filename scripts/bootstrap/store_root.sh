# Canonical store-root resolver for the bootstraps that speak POSIX shell.
#
# Sourceable AND spliceable: pure function definitions, so
# scripts/gen-bootstrap-resolver.py inlines it byte-for-byte between markers in
# install.sh (piped to bash, no sibling to source), while post-checkout callers
# — setup-hermes.sh and scripts/_activation.sh — source it directly and carry
# no copy. The bytes are generated, the truth lives once.
#
# Mirrors hermes_constants.get_default_hermes_root(), which the installers
# cannot import (they run before Python exists). The platform is detected HERE,
# not by the caller: setup-hermes.sh and scripts/_activation.sh source this file
# while install.sh takes it spliced, and setup-hermes.sh still sets `$os` for
# its own reasons — a caller-supplied gate is exactly the fragile shape this
# single source replaces.

# --- BEGIN canonical store-root resolver (scripts/gen-bootstrap-resolver.py) ---
# _hermes_os -> win32 | posix. Same precedence setup-hermes.sh used: uname,
# so a Cygwin/Git Bash shell reads pm's win32 default like native Windows does.
_hermes_os() {
    case "$(uname -s 2>/dev/null)" in
        MINGW*|MSYS*|CYGWIN*) printf 'win32' ;;
        *) printf 'posix' ;;
    esac
}

# _hermes_user_home NAME -> the account's home, or failure when unknown.

_hermes_user_home() {
    local _name="$1" _home=""
    case "$_name" in ""|*[!A-Za-z0-9_.-]*) return 1 ;; esac
    if command -v getent >/dev/null 2>&1; then
        _home="$(getent passwd "$_name" 2>/dev/null | cut -d: -f6)"
    fi
    if [ -z "$_home" ] && command -v dscacheutil >/dev/null 2>&1; then
        _home="$(dscacheutil -q user -a name "$_name" 2>/dev/null | awk '/^dir:/{print $2}')"
    fi
    [ -n "$_home" ] || return 1
    printf '%s' "$_home"
}

# _hermes_env_ci NAME -> the value of *name* read case-insensitively, as
# ntpath.expandvars reads os.environ. An exact printenv hit is one fork and
# keeps multi-line values verbatim; the awk fallback scans the exported names
# because printenv cannot fold case.
_hermes_env_ci() {
    local _name="$1" _v
    if _v="$(printenv "$_name")"; then
        printf '%s' "$_v"
        return 0
    fi
    env | awk -F= -v w="$_name" '
        tolower($1) == tolower(w) { sub(/^[^=]*=/, ""); printf "%s", $0; found=1; exit }
        END { exit !found }
    '
}

# _hermes_env_name NAME -> an exported name's value; present-and-empty counts
# as set (os.environ maps it to ''), absent fails.
_hermes_env_name() { printenv "$1" 2>/dev/null; }

# _hermes_vars VALUE -> os.path.expandvars(VALUE), ported from the platform's
# own _varpattern. posixpath has one form ($VAR / ${VAR}) with no escapes or
# quoting; ntpath (Windows) also folds '%%' and '$$' and copies '...' spans
# verbatim, and reads names case-insensitively because os.environ folds case
# there. An UNSET name keeps its source text, braces included.
_hermes_vars() {
    local _s="$1" _out="" _win _pat _i _n _m _name _val _lookup
    _win="$(_hermes_os)"
    if [ "$_win" = win32 ]; then
        # Same alternation as ntpath._varpattern: quoted span, %VAR%/escape,
        # $VAR/$$/braces.
        _pat="'[^']*'?|%(%|[^%]*%?)|[$]([$]|[-a-zA-Z0-9_]+|[{][^}]*[}]?)"
        _lookup=_hermes_env_ci
    else
        _pat="[$]([a-zA-Z0-9_]+|[{][^}]*[}]?)"
        _lookup=_hermes_env_name
    fi
    _ere="(${_pat})"
    while [ -n "$_s" ]; do
        _i=0; _n=${#_s}; _m=""
        while [ "$_i" -lt "$_n" ]; do
            if [[ "${_s:$_i}" =~ ^$_ere ]] && [ -n "${BASH_REMATCH[1]}" ]; then
                _m="${BASH_REMATCH[1]}"
                break
            fi
            _i=$((_i + 1))
        done
        if [ -z "$_m" ]; then _out="$_out$_s"; break; fi
        _out="$_out${_s:0:$_i}"
        _s="${_s:$((_i + ${#_m}))}"
        case "$_m" in
            "'"*) _out="$_out$_m" ;;                       # quoted span, verbatim
            '%'*)
                # ntpath: '%' + ('%' | [^%]* '%'?). Only a body that ENDS with
                # '%' expands; a bare '%' folds, an unclosed one stays source.
                _body="${_m:1}"
                if [ "$_body" = "%" ]; then _out="$_out%"
                elif [ "${_body%\%}" = "$_body" ]; then _out="$_out$_m"
                else
                    _name="${_body%\%}"
                    if [ -z "$_name" ]; then _out="$_out$_m"
                    elif _val="$(_hermes_env_ci "$_name")"; then _out="$_out$_val"
                    else _out="$_out%$_name%"
                    fi
                fi ;;
            '$$') _out="$_out\$" ;;                        # ntpath's '$' escape
            '${'*'}')
                _name="${_m:2:${#_m}-3}"
                if [ -z "$_name" ]; then _out="$_out$_m"
                elif _val="$("$_lookup" "$_name")"; then _out="$_out$_val"
                else _out="$_out$_m"
                fi ;;
            '$'*)
                _name="${_m:1}"
                if _val="$("$_lookup" "$_name")"; then _out="$_out$_val"
                else _out="$_out$_m"
                fi ;;
        esac
    done
    printf '%s' "$_out"
}

# _hermes_tilde VALUE -> the host os.path.expanduser(VALUE). win32 follows
# ntpath's ladder (USERPROFILE, else the HOMEDRIVE+HOMEPATH pair, else literal)
# and its ~user guess; POSIX follows posixpath's (HOME when SET — even empty —
# else the account database), with trailing separators stripped.
_hermes_tilde() {
    local _v="$1" _win _i _n _c _user _tail _home _drive _h _leaf _parent _cur _seps
    case "$_v" in '~'*) ;; *) printf '%s' "$_v"; return ;; esac
    _win="$(_hermes_os)"
    _n="${#_v}"; _i=1
    while [ "$_i" -lt "$_n" ]; do
        _c="${_v:$_i:1}"
        if [ "$_win" = win32 ]; then
            case "$_c" in /|\\) break ;; esac
        else
            [ "$_c" = / ] && break
        fi
        _i=$((_i + 1))
    done
    _user="${_v:1:$((_i - 1))}"
    _tail="${_v:$_i}"
    if [ "$_win" = win32 ]; then
        if [ "${USERPROFILE+x}" = x ]; then _home="$USERPROFILE"
        elif [ "${HOMEPATH+x}" = x ]; then
            _drive="${HOMEDRIVE-}"; _h="$HOMEPATH"
            case "$_h" in
                /|\\*) _drive="${_drive%"${_drive##*[!/\\]}"}"; _home="$_drive$_h" ;;
                *) _home="$_drive$_h" ;;
            esac
        else
            printf '%s' "$_v"; return
        fi
        if [ -n "$_user" ]; then
            _leaf="${_home##*[\\/]}"
            _parent="${_home%"${_home##*[\\/]}"}"
            _cur="${USERNAME-}"
            if [ "$_user" != "$_cur" ]; then
                [ "$_cur" = "$_leaf" ] || { printf '%s' "$_v"; return; }
                case "$_parent" in
                    '') _home="$_user" ;;
                    *:|*[\\/]) _home="${_parent}${_user}" ;;
                    *) _home="${_parent}\\${_user}" ;;
                esac
            fi
        fi
        printf '%s' "$_home$_tail"
        return
    fi
    if [ -n "$_user" ]; then
        if _home="$(_hermes_user_home "$_user")"; then printf '%s' "${_home%/}$_tail"
        else printf '%s' "$_v"
        fi
        return
    fi
    if [ "${HOME+x}" = x ]; then _home="${HOME%/}"
    elif ! _home="$(_hermes_user_home "$(id -un 2>/dev/null)")"; then printf '%s' "$_v"; return
    fi
    _home="${_home%/}"
    if [ -n "$_home" ]; then printf '%s' "$_home$_tail"
    else printf '%s' "${_tail:-/}"
    fi
}
# _hermes_expand VALUE -> os.path.expanduser(os.path.expandvars(VALUE)).
_hermes_expand() { _hermes_tilde "$(_hermes_vars "$1")"; }

# _hermes_norm VALUE -> str(Path(VALUE)): collapse separator runs, drop ".";
# ".." is KEPT, because pm returns the lexical path a custom home spelled.
# On win32 a backslash is a separator, exactly as Path treats it.
_hermes_norm() {
    local _rest="$1" _out="" _comp
    if [ "$(_hermes_os)" = win32 ]; then
        _rest="${_rest//\\//}"
    fi
    case "$_rest" in "//"[!/]*) _out="//" ;; /*) _out="/" ;; esac
    _rest="${_rest#/}"
    case "$_rest" in "//"*) _rest="${_rest#//}" ;; esac
    while [ -n "$_rest" ]; do
        _comp="${_rest%%/*}"
        case "$_rest" in */*) _rest="${_rest#*/}" ;; *) _rest="" ;; esac
        [ -z "$_comp" ] && continue
        [ "$_comp" = "." ] && continue
        case "$_out" in
            "") _out="$_comp" ;;
            "/") _out="/$_comp" ;;
            "//") _out="//$_comp" ;;
            *) _out="$_out/$_comp" ;;
        esac
    done
    printf '%s' "$_out"
}

# _hermes_resolve VALUE -> Path(VALUE).resolve(strict=False): follow every
# symlink along the chain (not only the longest existing prefix), keep missing
# components, pop ".." off the resolved prefix (never the anchor). One portable
# walk on purpose: `realpath -m` does not exist on BSD/macOS, so a host-tool
# fast path plus a shell fallback would put a second, never-executed resolver
# behind every macOS run — this is the only implementation every POSIX host runs.
_hermes_resolve() {
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

# hermes_default_home -> _get_platform_default_hermes_home(): win32 answers
# %LOCALAPPDATA%\hermes<suffix> first, else HOME under AppData/Local, else the
# literal "~/AppData/Local/..." form; POSIX answers the account home. The
# suffix is appended LITERALLY, never expanded.
hermes_default_home() {
    local _suffix="${HERMES_DATA_DIR_SUFFIX:-}" _acct _lad
    if [ "$(_hermes_os)" = win32 ]; then
        _lad="${LOCALAPPDATA-}"
        _lad="${_lad#"${_lad%%[![:space:]]*}"}"
        _lad="${_lad%"${_lad##*[![:space:]]}"}"
        if [ -n "$_lad" ]; then printf '%s' "$_lad/hermes$_suffix"
        elif [ -n "${HOME:-}" ]; then printf '%s' "$HOME/AppData/Local/hermes$_suffix"
        else printf '%s' "~/AppData/Local/hermes$_suffix"
        fi
        return
    fi
    if [ "${HOME+x}" = x ]; then
        # Path.home() collapses every root form (/, //, ///) to '/' before the
        # join, so the answer is '/.hermes'. A raw "$HOME/.hermes" keeps the
        # '//' anchor, which Path treats as a different path.
        _home="${HOME%/}"
        while [ "${_home#/}" != "$_home" ]; do _home="${_home#/}"; done
        [ -n "$_home" ] && { printf '%s' "/$_home/.hermes$_suffix"; return; }
        printf '%s' "/.hermes$_suffix"; return
    fi
    if _acct="$(_hermes_user_home "$(id -un 2>/dev/null)" 2>/dev/null)"; then
        printf '%s' "$_acct/.hermes$_suffix"; return
    fi
    printf '%s' "~/.hermes$_suffix"
}

# hermes_root_of -> get_default_hermes_root(). Containment is decided on
# RESOLVED paths, folded case-insensitively on win32 (WindowsPath.relative_to
# normcases); the returned form is lexical (it keeps ".."), as pm returns.
hermes_root_of() {
    local _def_lex _def_res _cmp_res _cmp_def _raw _lex _res _parent
    _def_lex="$(_hermes_norm "$(hermes_default_home)")"
    _def_res="$(_hermes_resolve "$_def_lex")"
    _raw="${HERMES_HOME-}"
    _raw="${_raw#"${_raw%%[![:space:]]*}"}"
    _raw="${_raw%"${_raw##*[![:space:]]}"}"
    if [ -n "$_raw" ]; then
        _lex="$(_hermes_norm "$(_hermes_expand "$_raw")")"
        _res="$(_hermes_resolve "$_lex")"
        _cmp_res="$_res"
        _cmp_def="$_def_res"
        if [ "$(_hermes_os)" = win32 ]; then
            _cmp_res="$(printf '%s' "$_res" | tr '[:upper:]' '[:lower:]')"
            _cmp_def="$(printf '%s' "$_def_res" | tr '[:upper:]' '[:lower:]')"
        fi
        case "$_cmp_res/" in
            "$_cmp_def"/*) _lex="$_def_lex" ;;
            *)
                _parent="${_lex%/*}"
                if [ "${_parent##*/}" = profiles ] && [ "$_parent" != "$_lex" ]; then
                    case "$_parent" in
                        profiles) _lex="." ;;
                        /profiles) _lex="/" ;;
                        *) _lex="${_parent%/profiles}" ;;
                    esac
                fi ;;
        esac
    else
        _lex="$_def_lex"
    fi
    [ -n "$_lex" ] || _lex="."
    printf '%s' "$_lex"
}
# --- END canonical store-root resolver ---
