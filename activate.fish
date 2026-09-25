# ============================================================================
# venv-style activation for the Hermes dev environment (pm-managed tools).
#
#   source ./activate.fish   (fish, repo root)
#   source ./activate        (bash / zsh)
#   .\activate.ps1            (PowerShell)
#
# Emits the composed pm env (PATH + tool vars) into the CURRENT shell, with
# save/restore: `deactivate` undoes exactly what activation changed.
#
# `hermes` becomes a shell function for this checkout. It runs only while the
# shell is inside this worktree and refuses outside it, so a sibling worktree
# keeps its own command. A prompt prefix names the worktree.
#
# Sync through setup before selecting the environment. PM owns freshness;
# activation does not maintain a second dependency stamp. It trusts the
# recorded tool digest instead of re-hashing every entry. `hermes pm
# install` and `hermes update` keep that check.
#
# Options: `--test-extras a,b` selects the test environment's runtime extras
# (default: [all]). Extra argv after `--` is ignored. Unlike bash `source`,
# fish `source file` does not inherit the caller's $argv unless you pass them.
# ============================================================================

set -l _HERMES_REPO (builtin realpath -- (dirname (status -f)))

set -l _hermes_test_env --test-environment
set -l _hermes_expect_extras 0
set -l _hermes_i 1
while test $_hermes_i -le (count $argv)
    set -l _hermes_arg $argv[$_hermes_i]
    if test $_hermes_expect_extras -eq 1
        set _hermes_test_env "--test-environment=$_hermes_arg"
        set _hermes_expect_extras 0
        set _hermes_i (math $_hermes_i + 1)
        continue
    end
    switch $_hermes_arg
        case --
            break
        case --test-extras
            set _hermes_expect_extras 1
        case '--test-extras=*'
            set _hermes_test_env "--test-environment="(string replace -- '--test-extras=' '' -- $_hermes_arg)
        case '*'
            true
    end
    set _hermes_i (math $_hermes_i + 1)
end
if test $_hermes_expect_extras -eq 1
    printf '%s\n' 'activate: --test-extras needs a comma-separated list' >&2
    return 2
end

# Setup runs in a child: failures must not exit or partially activate the
# caller's shell, and activation must not republish launchers or shell config.
if not env -u PYTHONHOME -u PYTHONPATH -u VIRTUAL_ENV \
        bash "$_HERMES_REPO/setup-hermes.sh" --runtime-only $_hermes_test_env >&2
    printf '%s\n' 'activate: setup failed; shell environment unchanged' >&2
    return 1
end

# Guard against double-sourcing: deactivate first when a previous activation
# left its function in this shell. Child shells inherit __HERMES_ACTIVATED but
# not the function, and there is nothing to undo there.
if functions -q deactivate
    deactivate
end

set -l _hermes_repo $_HERMES_REPO
set -l _hermes_py ''
for _hermes_candidate in \
        "$_hermes_repo/.venv/bin/python" \
        "$_hermes_repo/.venv/Scripts/python.exe" \
        "$_hermes_repo/venv/bin/python" \
        "$_hermes_repo/venv/Scripts/python.exe"
    if test -x "$_hermes_candidate"
        set _hermes_py $_hermes_candidate
        break
    end
end
if test -z "$_hermes_py"
    set -l _hermes_stores
    if set -q HERMES_RUNTIME_DIR; and test -n "$HERMES_RUNTIME_DIR"
        set -a _hermes_stores $HERMES_RUNTIME_DIR
    end
    set -a _hermes_stores "$_hermes_repo/../tools"
    if set -q HERMES_HOME; and test -n "$HERMES_HOME"
        set -a _hermes_stores "$HERMES_HOME/tools"
    else
        set -a _hermes_stores "$HOME/.hermes/tools"
    end
    for _hermes_store in $_hermes_stores
        for _hermes_candidate in \
                $_hermes_store/python-*/bin/python3 \
                $_hermes_store/python-*/python.exe \
                $_hermes_store/python-*/bin/python \
                $_hermes_store/python-*/bin/python.exe
            if test -x "$_hermes_candidate"
                set _hermes_py $_hermes_candidate
                break
            end
        end
        if test -n "$_hermes_py"
            break
        end
    end
end
if test -z "$_hermes_py"
    echo "activate: no bootstrap Python found; run setup-hermes.sh" >&2
    return 1
end

# --- emit export lines from `pm env` ---
set -l _hermes_json (
    env PYTHONHOME= PYTHONPATH=$_hermes_repo $_hermes_py -m pm.environments | string collect
)
if test -z "$_hermes_json"; or not string match -q '{*' -- "$_hermes_json"
    echo "activate: could not read pm env (run ./setup-hermes.sh first)" >&2
    return 1
end

set -l _hermes_keys (
    printf '%s' "$_hermes_json" | $_hermes_py -c '
import json, sys, re
readonly = {
    "PWD", "SHLVL", "status", "pipestatus", "history", "version",
    "fish_pid", "fish_kill_signal", "hostname", "_", "argv",
}
print("\n".join(
    k for k in json.load(sys.stdin)
    if k not in readonly and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k)
))
'
)
if test -z "$_hermes_keys"
    set _hermes_keys PATH
end

# --- snapshot what we are about to change (deactivate restores this) ---
# __HERMES_ACTIVATED is part of the composed env below, so it is snapshotted
# and exported with every other key — no separate assignment here.
set -g __HERMES_KEY_LIST $_hermes_keys
for _hermes_k in $__HERMES_KEY_LIST
    if set -q $_hermes_k
        set -g __HERMES_SAVED_$_hermes_k set
        # Indirect: value of the variable named by $_hermes_k (list-safe for PATH).
        set -g __HERMES_PRIOR_$_hermes_k $$_hermes_k
    else
        set -g __HERMES_SAVED_$_hermes_k unset
        set -e __HERMES_PRIOR_$_hermes_k
    end
end

# Apply composed env. PATH is a fish path list — split on the native separator.
# Other keys stay single strings (PYTHONPATH keeps os.pathsep joins).
# Python source stays free of single quotes so it fits in a fish single-quoted -c.
printf '%s' "$_hermes_json" | $_hermes_py -c '
import json, sys, re, os
sep = os.pathsep
# fish rejects writes to these; skip even if pm env echoes the process env.
readonly = {
    "PWD", "SHLVL", "status", "pipestatus", "history", "version",
    "fish_pid", "fish_kill_signal", "hostname", "_", "argv",
}
def q(s):
    return chr(39) + s.replace(chr(92), chr(92) * 2).replace(chr(39), chr(92) + chr(39)) + chr(39)
for k, v in json.load(sys.stdin).items():
    if k in readonly or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k):
        continue
    s = str(v)
    if k == "PATH":
        parts = [p for p in s.split(sep) if p]
        print("set -gx PATH " + " ".join(q(p) for p in parts))
    else:
        print("set -gx %s %s" % (k, q(s)))
' | source

# The MSYS/Cygwin runtime hands a native Windows Python PATH in Windows form.
# Convert with cygpath when present (no-op on normal Linux/macOS).
set -l _hermes_cygpath (command -s cygpath 2>/dev/null)
if test -n "$_hermes_cygpath"
    set -l _win_path (string join ';' $PATH)
    set -gx PATH (string split : -- ($_hermes_cygpath -u -p $_win_path))
    for _hermes_k in HOME TMPDIR TMP TEMP
        if set -q $_hermes_k; and test -n "$$_hermes_k"
            set -gx $_hermes_k ($_hermes_cygpath -u $$_hermes_k)
        end
    end
end

# This checkout, not whichever `hermes` PATH finds. A function beats PATH and
# an alias. It runs only while the shell is inside this worktree.
set -g __HERMES_WORKTREE $_hermes_repo
# The branch names the worktree. A checkout cannot share a branch with another.
set -g __HERMES_WORKTREE_NAME (git -C $_hermes_repo rev-parse --abbrev-ref HEAD 2>/dev/null)
if test -z "$__HERMES_WORKTREE_NAME"; or test "$__HERMES_WORKTREE_NAME" = HEAD
    set -g __HERMES_WORKTREE_NAME (basename $_hermes_repo)
end

function _hermes_worktree_here --description 'True when cwd is inside the activated Hermes worktree'
    set -l top (git rev-parse --show-toplevel 2>/dev/null)
    or return 1
    set -l here (builtin realpath -- $top)
    set -l root (builtin realpath -- $__HERMES_WORKTREE)
    test "$here" = "$root"
end

function hermes --description 'Run this worktree\'s hermes CLI'
    if not _hermes_worktree_here
        printf '%s\n' "hermes: "(pwd)" is outside $__HERMES_WORKTREE; refusing (the installed command is hidden while this checkout is active)" >&2
        return 1
    end
    set -l _oldpwd $PWD
    builtin cd $__HERMES_WORKTREE
    or return 1
    set -l _code 0
    if set -q PYTHON; and test -n "$PYTHON"
        "$PYTHON" hermes $argv
        set _code $status
    else if test -x .venv/Scripts/python.exe
        .venv/Scripts/python.exe hermes $argv
        set _code $status
    else if test -x .venv/bin/python
        .venv/bin/python hermes $argv
        set _code $status
    else if test -x venv/Scripts/python.exe
        venv/Scripts/python.exe hermes $argv
        set _code $status
    else if test -x venv/bin/python
        venv/bin/python hermes $argv
        set _code $status
    else
        python hermes $argv
        set _code $status
    end
    builtin cd $_oldpwd
    return $_code
end

# Prompt prefix: (branch) while inside the worktree. Save any existing fish_prompt.
if functions -q fish_prompt
    if functions -q __hermes_saved_fish_prompt
        functions -e __hermes_saved_fish_prompt
    end
    functions -c fish_prompt __hermes_saved_fish_prompt
    set -g __HERMES_SAVED_FISH_PROMPT 1
else
    set -g __HERMES_SAVED_FISH_PROMPT 0
end

function fish_prompt
    if _hermes_worktree_here
        printf '(%s) ' $__HERMES_WORKTREE_NAME
    end
    if functions -q __hermes_saved_fish_prompt
        __hermes_saved_fish_prompt
    else
        printf '%s@%s %s%s%s> ' $USER (prompt_hostname) (set_color $fish_color_cwd) (prompt_pwd) (set_color normal)
    end
end

function deactivate --description 'Undo Hermes activate.fish'
    for _hermes_k in $__HERMES_KEY_LIST
        set -l _hermes_was __HERMES_SAVED_$_hermes_k
        set -l _hermes_flag $$_hermes_was
        if test "$_hermes_flag" = set
            set -l _hermes_prior __HERMES_PRIOR_$_hermes_k
            set -gx $_hermes_k $$_hermes_prior
        else
            set -e $_hermes_k
        end
        set -e __HERMES_SAVED_$_hermes_k
        set -e __HERMES_PRIOR_$_hermes_k
    end
    if test "$__HERMES_SAVED_FISH_PROMPT" = 1; and functions -q __hermes_saved_fish_prompt
        functions -e fish_prompt
        functions -c __hermes_saved_fish_prompt fish_prompt
        functions -e __hermes_saved_fish_prompt
    else
        functions -e fish_prompt
    end
    set -e __HERMES_KEY_LIST
    set -e __HERMES_ACTIVATED
    set -e __HERMES_WORKTREE
    set -e __HERMES_WORKTREE_NAME
    set -e __HERMES_SAVED_FISH_PROMPT
    functions -e deactivate hermes _hermes_worktree_here
end
