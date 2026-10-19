#!/usr/bin/env bash
# GitHub environment detection helper for Hermes Agent skills.
#
# Usage (via terminal tool) — source from anywhere; helper scripts are
# resolved relative to this file's own directory:
#   source "${HERMES_HOME:-$HOME/.hermes}/skills/software-development/github/scripts/gh-env.sh"
#
# After sourcing, these variables are set:
#   GH_AUTH_METHOD   - "gh", "curl" or "none" -- what authorises the work
#   GH_TOKEN_SOURCE  - where GITHUB_TOKEN actually came from ("gh auth token",
#                      "$GITHUB_TOKEN", a .env path, a store path, or empty)
#   GITHUB_TOKEN     - the token (set when GH_AUTH_METHOD is "gh" or "curl")
#   GH_USER          - GitHub username AS RETURNED BY THE API, never asserted
#                      ("unverified ..." if a token exists but the API refused it)
#   GH_GH_STATUS     - what 'gh auth status' / 'command -v gh' said
#   GH_CRED_SOURCE   - the credential path git itself will consult in $PWD, named
#   GH_CRED_NOTE     - anything a human should decide about what was found
#   GH_AUTH_EVIDENCE - the command whose output produced the verdict
#   GH_OWNER         - repo owner  (only if inside a git repo with a github remote)
#   GH_REPO          - repo name   (only if inside a git repo with a github remote)
#   GH_OWNER_REPO    - owner/repo  (only if inside a git repo with a github remote)
#
# Truthfulness rule (t_a05196f0; Brandon ruling 2026-10-19: "we should be using gh
# authentication"): this reporter may only print a verdict a probe produced. 'gh' is
# the source of truth where gh is the mechanism. Where something else authorises git,
# the reporter NAMES that mechanism and where it is configured, so a human can decide
# whether it is intended -- it never silently tolerates a second mechanism, and it
# never prints "none" while a working credential path exists (the false negative this
# card was opened for). Nothing here changes auth policy or removes a credential path.

# --- Credential-store extraction (shared by the $HOME store and repo stores) --

# Print the first GitHub token in the credential-store file given as $1; print
# nothing if there is none. POSIX-sh clean: no &>, no [[, no bash-only arrays.
_hermes_store_token() {
    _hst_py=""
    # Resolve the extractor. $0 is the CALLER's argv0 when a POSIX shell sources this
    # file, so it cannot be trusted; BASH_SOURCE covers bash, and the known skill
    # roots cover everything else (the installed copies are byte-identical by design).
    if [ -n "$_h_script_dir" ] && [ -f "$_h_script_dir/git-credential-token.py" ]; then
        _hst_py="$_h_script_dir/git-credential-token.py"
    fi
    if [ -z "$_hst_py" ]; then
        for _hst_root in "${HERMES_HOME:-$HOME/.hermes}/skills" /opt/data/skills \
                         /opt/data/profiles/*/skills "$PWD"; do
            for _hst_dir in "$_hst_root"/*/github/scripts "$_hst_root"/*/*/github/scripts; do
                if [ -f "$_hst_dir/git-credential-token.py" ]; then
                    _hst_py="$_hst_dir/git-credential-token.py"
                    break
                fi
            done
            if [ -n "$_hst_py" ]; then break; fi
        done
    fi
    # Bound the read: on the repo path the file name comes from the repo's own git
    # config, so refuse anything implausibly large for a credential store. The
    # regular-file test excludes devices like /dev/zero.
    _hst_size=0
    if [ -n "$_hst_py" ] && [ -f "$1" ] && [ ! -d "$1" ]; then
        _hst_size=$(wc -c < "$1" 2>/dev/null || echo 0)
    fi
    if [ -n "$_hst_py" ] && [ -f "$1" ] && [ ! -d "$1" ] && [ "$_hst_size" -le 262144 ]; then
        python3 "$_hst_py" "$1" 2>/dev/null || :
    fi
    unset _hst_py _hst_size _hst_root _hst_dir
}

# Where this file itself lives, for the extractor lookup above.
_h_script_dir=""
_h_self=${BASH_SOURCE:-}
case "$_h_self" in
    */gh-env.sh) ;;
    *) case "$0" in */gh-env.sh) _h_self=$0 ;; *) _h_self="" ;; esac ;;
esac
case "$_h_self" in */*) _h_script_dir=${_h_self%/*} ;; esac
unset _h_self

# --- Auth detection ---

GH_AUTH_METHOD="none"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
GH_USER=""
GH_GH_STATUS=""
GH_CRED_SOURCE=""
GH_CRED_NOTE=""
GH_AUTH_EVIDENCE=""
GH_TOKEN_SOURCE=""
_h_named=""
_h_store_note=""
_h_gh_won=0

_h_redact() {
    # Evidence lines must not carry secrets: 'gh auth status' prints a token field.
    printf '%s' "$1" \
        | sed -E 's#(gh[pousr]_|github_pat_)[A-Za-z0-9_-]+#\1[redacted]#g' \
              | tr '\n\r' '  ' | cut -c1-180
}

# --- Probe 1: is gh the mechanism? Ask gh, and honour its exit code. ---------
if command -v gh >/dev/null 2>&1; then
    _h_ghrc=0
    _h_ghout=$(gh auth status 2>&1) || _h_ghrc=$?
    if [ "$_h_ghrc" -eq 0 ]; then
        GH_AUTH_METHOD="gh"
        _h_gh_won=1
        GH_AUTH_EVIDENCE="gh auth status -> exit 0: $(_h_redact "$_h_ghout")"
        GH_USER=$(gh api user --jq '.login' 2>/dev/null) || :
    else
        GH_GH_STATUS="installed but NOT authenticated: 'gh auth status' -> exit $_h_ghrc: $(_h_redact "$_h_ghout")"
    fi
    unset _h_ghout _h_ghrc
else
    GH_GH_STATUS="not installed ('command -v gh' -> not found)"
fi

# Where gh authenticated it is the single source of truth: hand callers the SAME token
# gh holds instead of a second copy pulled out of a store.
if [ "$_h_gh_won" -eq 1 ] && [ -z "$GITHUB_TOKEN" ]; then
    _h_gt=""
    _h_gt=$(gh auth token 2>/dev/null) || :
    if [ -n "$_h_gt" ]; then
        GITHUB_TOKEN="$_h_gt"
        GH_TOKEN_SOURCE="gh auth token"
    fi
    unset _h_gt
fi

if [ "$_h_gh_won" -ne 1 ]; then
    # --- Probe 2: a token already in the environment, or in the Hermes .env ----
    if [ -n "$GITHUB_TOKEN" ]; then
        GH_TOKEN_SOURCE='$GITHUB_TOKEN (already in the environment)'
    else
        _h_envf="${HERMES_HOME:-$HOME/.hermes}/.env"
        if [ -f "$_h_envf" ] && grep -q "^GITHUB_TOKEN=" "$_h_envf" 2>/dev/null; then
            _h_grep=""
            _h_grep=$(grep "^GITHUB_TOKEN=" "$_h_envf" | head -1 | cut -d= -f2 | tr -d '\n\r') || :
            if [ -n "$_h_grep" ]; then
                GITHUB_TOKEN="$_h_grep"
                GH_TOKEN_SOURCE="$_h_envf GITHUB_TOKEN="
            fi
            unset _h_grep
        fi
        unset _h_envf
    fi

    # --- Probe 3: git's own credential chain, in this directory ----------------
    # The t_a05196f0 fix. $HOME/.git-credentials used to be the last word, so in a
    # profile with no store the reporter printed "none" even where git auth
    # demonstrably works: the fleet repos (obsidian-notes, meal-planner,
    # backup-repos/*) carry an ABSOLUTE repo-local 'store --file=<path>', which is
    # independent of $HOME. We ask git what IT would use here, and we say so.
    # Only 'store' is ever read; any other helper is NAMED, never executed.
    _h_origin=""
    _h_helper=""
    _h_repo=""
    _h_url=""
    _h_origin_file=""
    _h_helper=$(git config --show-origin --get-all credential.helper 2>/dev/null) || :
    _h_origin=$(printf '%s\n' "$_h_helper" | cut -f1) || :
    _h_helper=$(git config --get-all credential.helper 2>/dev/null) || :
    _h_repo=$(git rev-parse --show-toplevel 2>/dev/null) || :
    _h_url=$(git remote get-url origin 2>/dev/null) || :
    _h_kind=""
    _h_name=""
    _h_store=""
    _h_store_active=""
    if [ -n "$_h_helper" ]; then
        # git's own semantics: an EMPTY helper value resets the list, so anything
        # configured earlier (e.g. a global 'store') is off unless re-listed after it.
        # Count, do not match: a matched BLANK line yields empty output, which is
        # indistinguishable from "no reset present".
        _h_reset=0
        _h_reset=$(printf '%s\n' "$_h_helper" | grep -c '^[[:space:]]*$') || :
        _h_kind=$(printf '%s\n' "$_h_helper" | sed -n '/[^[:space:]]/{s/^[[:space:]]*//; s/[[:space:]]*$//; p}' | head -1) || :
        _h_name=${_h_kind%%[[:space:]]*}
        _h_origin_file=$(printf '%s\n' "$_h_origin" | sed -n '/[^[:space:]]/{p; q}' | cut -c1-120) || :
        _h_named="credential.helper='$_h_kind'"
        if [ "$_h_reset" -gt 0 ]; then _h_named="$_h_named (list reset by an empty value first)"; fi
        if [ -n "$_h_origin_file" ]; then _h_named="$_h_named, configured in $_h_origin_file"; fi
        case "$_h_name" in
            store)
                _h_store=$(printf '%s\n' "$_h_helper" \
                    | sed -n 's/^[[:space:]]*store[[:space:]][[:space:]]*--file=//p' | head -1) || :
                if [ -z "$_h_store" ]; then _h_store="$HOME/.git-credentials"; fi
                case "$_h_store" in
                    /*) ;;
                    *) if [ -n "$_h_repo" ]; then _h_store="$_h_repo/$_h_store"; fi ;;
                esac
                _h_store_active="$_h_store"
                ;;
            \!*)
                _h_named="$_h_named -- a CUSTOM helper command: named by this reporter, never executed"
                ;;
            "")
                ;;
            *)
                _h_named="$_h_named -- not a plain 'store' helper, so this reporter does not read it"
                ;;
        esac
        GH_CRED_SOURCE="$_h_named"
    elif [ -n "$_h_repo" ]; then
        GH_CRED_SOURCE="no credential.helper configured for this repo (git would prompt for credentials)"
    fi
    unset _h_helper _h_origin _h_reset
    case "$_h_url" in
        # [!...] is the POSIX negated class. [^...] is a bashism that dash reads
        # as the literal set {^,/,@}, so it silently never matched (case H caught it).
        *://[!/@]*:*@*)
            GH_CRED_NOTE="the remote origin URL embeds a credential in cleartext -- rotate and scrub it (t_a438d3ac)"
            ;;
    esac

    # Deliberate deviation from git's precedence, stated out loud: the profile's own
    # $HOME store is consulted BEFORE a repo-named store, so a repo's config can never
    # shadow (or swap out) the store the operator installed for this profile.
    if [ -z "$GITHUB_TOKEN" ] && [ -f "$HOME/.git-credentials" ]; then
        _h_tok=""
        _h_tok=$(_hermes_store_token "$HOME/.git-credentials") || :
        if [ -n "$_h_tok" ]; then
            GITHUB_TOKEN="$_h_tok"
            GH_TOKEN_SOURCE="$HOME/.git-credentials"
            GH_AUTH_EVIDENCE="token read from \$HOME/.git-credentials by git-credential-token.py"
            if [ "$_h_store_active" != "$HOME/.git-credentials" ]; then
                _h_store_note="\$HOME/.git-credentials holds a token, but git will not read it in this directory (active store helper: ${_h_store_active:-none}); git would prompt instead"
            fi
        fi
        unset _h_tok
    fi
    if [ -z "$GITHUB_TOKEN" ] && [ -n "$_h_store" ]; then
        _h_tok=""
        _h_tok=$(_hermes_store_token "$_h_store") || :
        if [ -n "$_h_tok" ]; then
            GITHUB_TOKEN="$_h_tok"
            GH_TOKEN_SOURCE="$_h_store"
            GH_AUTH_EVIDENCE="git config --get-all credential.helper -> 'store --file=$_h_store'; token read from that file by git-credential-token.py"
        fi
        unset _h_tok
    fi
    unset _h_repo _h_url _h_kind _h_name _h_store _h_store_active _h_origin_file
fi

# --- Verdict ----------------------------------------------------------------
if [ "$GH_AUTH_METHOD" != "gh" ] && [ -n "$GITHUB_TOKEN" ]; then
    GH_AUTH_METHOD="curl"
fi
if [ "$GH_AUTH_METHOD" = "none" ]; then GH_USER=""; fi


# Resolve username for curl method
if [ "$GH_AUTH_METHOD" = "curl" ] && [ -z "$GH_USER" ]; then
    GH_USER=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
        https://api.github.com/user 2>/dev/null \
        | python3 -c "import sys,json; print(json.load(sys.stdin).get('login',''))" 2>/dev/null) || :
fi

# A token the API refuses is not "authenticated as <someone>" -- say so, instead of
# leaving the reader to infer a username that was never returned.
if [ "$GH_AUTH_METHOD" = "curl" ] && [ -z "$GH_USER" ]; then
    GH_USER="unverified (a token exists; GET https://api.github.com/user returned no login)"
fi

# --- Repo detection (if inside a git repo with a GitHub remote) ---

GH_OWNER=""
GH_REPO=""
GH_OWNER_REPO=""

_remote_url=""
_remote_url=$(git remote get-url origin 2>/dev/null) || :
if [ -n "$_remote_url" ] && echo "$_remote_url" | grep -q "github.com"; then
    GH_OWNER_REPO=$(echo "$_remote_url" | sed -E 's|.*github\.com[:/]||; s|\.git$||')
    GH_OWNER=$(echo "$GH_OWNER_REPO" | cut -d/ -f1)
    GH_REPO=$(echo "$GH_OWNER_REPO" | cut -d/ -f2)
fi
unset _remote_url

# --- Summary ----------------------------------------------------------------
# Each line names the measurement that produced it. Keep in sync with the contract
# at the top of this file.

echo "GitHub Auth: $GH_AUTH_METHOD"
if [ -n "$GH_AUTH_EVIDENCE" ]; then echo "  evidence: $GH_AUTH_EVIDENCE"; fi
if [ -n "$GH_TOKEN_SOURCE" ]; then echo "  token from: $GH_TOKEN_SOURCE"; fi
if [ -n "$GH_USER" ]; then echo "User: $GH_USER"; fi
if [ -n "$GH_CRED_SOURCE" ]; then echo "Credential source: $GH_CRED_SOURCE"; fi
if [ -n "$GH_CRED_NOTE" ]; then echo "Note: $GH_CRED_NOTE"; fi
if [ -n "$_h_store_note" ]; then echo "Note: $_h_store_note"; fi
if [ -n "$GH_GH_STATUS" ]; then echo "gh: $GH_GH_STATUS"; fi
if [ -n "$GH_OWNER_REPO" ]; then echo "Repo: $GH_OWNER_REPO"; fi
if [ "$GH_AUTH_METHOD" = "none" ]; then
    echo "WARNING not authenticated -- set up auth first (see references/auth.md in this skill)"
    echo "  probed, all refused:"
    echo "    gh                -> ${GH_GH_STATUS:-not checked}"
    echo "    credential.helper -> ${_h_named:-not configured for this directory}"
    if [ -f "$HOME/.git-credentials" ]; then
        echo "    \$HOME/.git-credentials -> present, but yielded no usable token"
    else
        echo "    \$HOME/.git-credentials -> absent"
    fi
    echo "  If git pushes DO work from here, this verdict is wrong: report it (task"
    echo "  devsecops) instead of working around it. Prefer 'gh auth login' (t_a05196f0)."
fi
unset _h_named _h_store_note _h_gh_won _h_script_dir
unset -f _hermes_store_token 2>/dev/null || unset _hermes_store_token 2>/dev/null || :
