# venv-style activation of the Hermes dev environment for fish.
#
#   source ./activate.fish
#
# The fish counterpart of ./activate: applies the composed pm env to the
# CURRENT shell, with save/restore (`deactivate` undoes exactly what activation
# changed), makes `hermes` a function for this checkout, and prefixes the prompt
# with the worktree name. Scripts and other processes run under
# `scripts/run-in-hermes-env` instead.
#
# Options: `--test-extras a,b` selects the test environment's runtime extras
# (default: [all]).
#
# Composition (setup sync, bootstrap Python, pm env) is shared with ./activate
# through scripts/_activation.sh, so this file only applies its output.

set -l repo (path resolve (path dirname (status filename)))

argparse 'test-extras=' -- $argv
or return 2
set -l test_environment --test-environment
if set -q _flag_test_extras
    set test_environment --test-environment=$_flag_test_extras
end

set -l library '. "$1/scripts/_activation.sh"'
bash -c "$library; hermes_sync \"\$1\" \"\$2\"" _ $repo $test_environment >&2
or begin
    echo 'activate.fish: setup failed; shell environment unchanged' >&2
    return 1
end

# Re-activating deactivates first, as ./activate does.
functions -q deactivate
and deactivate

set -l composed (bash -c "$library; hermes_compose_env \"\$1\" fish" _ $repo)
or begin
    echo 'activate.fish: could not compose the environment (see above)' >&2
    return 1
end

# --- snapshot what we are about to change (deactivate restores this) ---
set -g __hermes_keys (string replace -rf '^set -gx (\S+) .*' '$1' -- $composed)
for name in $__hermes_keys
    if set -q $name
        set -g __hermes_had_$name 1
        set -g __hermes_prior_$name $$name
    end
end
string join \n -- $composed | source

# This checkout, not whichever `hermes` PATH finds. A function beats PATH.
set -g __hermes_worktree $repo
# The branch names the worktree. A checkout cannot share a branch with another.
set -g __hermes_worktree_name (git -C $repo rev-parse --abbrev-ref HEAD 2>/dev/null)
if test -z "$__hermes_worktree_name" -o "$__hermes_worktree_name" = HEAD
    set __hermes_worktree_name (path basename $repo)
end

function __hermes_worktree_here
    set -l top (git rev-parse --show-toplevel 2>/dev/null)
    or return 1
    test (path resolve $top) = (path resolve $__hermes_worktree)
end

function hermes --description 'hermes of the activated checkout'
    if not __hermes_worktree_here
        echo "hermes: $PWD is outside $__hermes_worktree; refusing (the installed command is hidden while this checkout is active)" >&2
        return 1
    end
    set -l python python
    if set -q PYTHON
        set python $PYTHON
    else if test -x $__hermes_worktree/.venv/bin/python
        set python $__hermes_worktree/.venv/bin/python
    else if test -x $__hermes_worktree/venv/bin/python
        set python $__hermes_worktree/venv/bin/python
    end
    pushd $__hermes_worktree >/dev/null
    $python hermes $argv
    set -l code $status
    popd >/dev/null
    return $code
end

functions -c fish_prompt __hermes_saved_fish_prompt
function fish_prompt
    if __hermes_worktree_here
        printf '(%s) ' $__hermes_worktree_name
    end
    __hermes_saved_fish_prompt
end

function deactivate --description 'undo activate.fish'
    for name in $__hermes_keys
        set -l had __hermes_had_$name
        set -l prior __hermes_prior_$name
        if set -q $had
            set -gx $name $$prior
        else
            set -eg $name
        end
        set -eg $had $prior
    end
    functions -e fish_prompt
    functions -c __hermes_saved_fish_prompt fish_prompt
    functions -e __hermes_saved_fish_prompt
    set -eg __hermes_keys __hermes_worktree __hermes_worktree_name
    functions -e deactivate hermes __hermes_worktree_here
end
