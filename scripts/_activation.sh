# Sourced, never executed: the activation staleness check shared by
# scripts/_hermes-python (the shebang prologue) and scripts/run_tests.sh.
#
# pm records, beside the installed-state file named by __HERMES_ACTIVATED, one
# stamp per dependency input carrying the exact mtime that input had when the
# install was last verified against it (pm.environments.record_activation_inputs).
# Any input whose mtime DIFFERS from its stamp (newer or older, since a branch
# switch can move it either way) means the inherited environment may not match
# its inputs. `-nt`/`-ot` are bash builtins, so the check costs no process
# spawn, and every successful activation re-records, so one re-activation
# settles it. Missing stamps or inputs, a dangling sentinel or a literal "1"
# all read as stale.

# hermes_activation_current REPO: status 0 when the inherited environment
# matches REPO's current inputs.
hermes_activation_current() {
    local repo="$1" sentinel stamps stamp input stamped=0
    [ -n "${__HERMES_ACTIVATED:-}" ] && [ -e "${__HERMES_ACTIVATED}" ] || return 1
    # activate.ps1 records a Windows path; Git Bash accepts it with slashes.
    sentinel="${__HERMES_ACTIVATED//\\//}"
    stamps="${sentinel%/*}/inputs"
    [ -f "$stamps/.project-root" ] || return 1
    local owner="$repo" recorded
    recorded="$(< "$stamps/.project-root")"
    if command -v cygpath >/dev/null 2>&1; then
        owner="$(cygpath -am "$repo")" || return 1
        recorded="${recorded//\\//}"
        [ "${owner,,}" = "${recorded,,}" ] || return 1
    else
        [ "$(cd "$owner" && pwd -P)" = "$recorded" ] || return 1
    fi
    for stamp in "$stamps"/* "$stamps"/*/*; do
        [ -f "$stamp" ] || continue
        stamped=1
        input="$repo/${stamp#"$stamps"/}"
        if [ ! -e "$input" ] || [ "$input" -nt "$stamp" ] || [ "$input" -ot "$stamp" ]; then
            return 1
        fi
    done
    [ "$stamped" = 1 ]
}

# hermes_ensure_activated REPO: source REPO/activate unless the inherited
# environment is current. `--` passes no options: `source` without arguments
# would hand activate the CALLER's positional parameters.
hermes_ensure_activated() {
    hermes_activation_current "$1" && return 0
    # shellcheck source=/dev/null
    . "$1/activate" --
}
