#!/command/with-contenv sh
# /opt/hermes/docker/main-wrapper.sh — wraps the container's CMD with
# the same argument-routing logic the pre-s6 entrypoint.sh used. Runs
# as /init's "main program" (Docker CMD) so it inherits stdin/stdout/
# stderr from the container.
#
# Routing:
#   no args                       → exec `hermes` (the default)
#   first arg is an executable    → exec it directly (sleep, bash, sh, …)
#   first arg is anything else    → exec `hermes <args>` (subcommand passthrough)
#
# We drop to the hermes user while preserving non-root supplementary groups
# injected by the container runtime (for example docker-compose `group_add`
# for /var/run/docker.sock access).
set -e

cd /opt/data
# shellcheck disable=SC1091
. /opt/hermes/.venv/bin/activate
if [ -f /opt/data/.env ]; then
    set -a
    # shellcheck disable=SC1091
    . /opt/data/.env
    set +a
fi
export HERMES_HOME=/opt/data
export HOME=/opt/data
if [ "${1:-}" = "gateway" ]; then
    export HERMES_GATEWAY_SESSION=1
fi

run_as_hermes() {
    primary_gid=$(id -g hermes)
    extra_groups=$(
        awk -v primary="$primary_gid" '
            $1 == "Groups:" {
                for (i = 2; i <= NF; i++) {
                    if ($i != "0" && $i != primary) {
                        printf "%s%s", sep, $i
                        sep = ","
                    }
                }
            }
        ' /proc/self/status
    )
    if [ -n "$extra_groups" ]; then
        exec setpriv --reuid hermes --regid hermes --groups "$extra_groups" "$@"
    fi
    exec setpriv --reuid hermes --regid hermes --init-groups "$@"
}

if [ $# -eq 0 ]; then
    run_as_hermes hermes
fi

if command -v "$1" >/dev/null 2>&1; then
    # Bare executable — pass through directly.
    run_as_hermes "$@"
fi

# Hermes subcommand pass-through.
run_as_hermes hermes "$@"
