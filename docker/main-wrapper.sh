#!/command/with-contenv sh
# shellcheck shell=sh
# /opt/hermes/docker/main-wrapper.sh — wraps the container's CMD with
# the same argument-routing logic the pre-s6 entrypoint.sh used. Runs
# as /init's "main program" (Docker CMD) so it inherits stdin/stdout/
# stderr from the container.
#
# Shebang note: /init scrubs env before invoking CMD, so a plain
# `#!/bin/sh` wrapper sees an empty environ and `ENV HERMES_HOME=/opt/data`
# from the Dockerfile never reaches `hermes`. with-contenv repopulates
# the env from /run/s6/container_environment before exec'ing, which is
# what s6-supervised services use too (see main-hermes/run).
#
# Routing:
#   no args                       → exec `hermes` (the default)
#   first arg is an executable    → exec it directly (sleep, bash, sh, …)
#   first arg is anything else    → exec `hermes <args>` (subcommand passthrough)
#
# Drop to hermes via s6-setuidgid, but skip it when already non-root.
set -e

drop() { [ "$(id -u)" = 0 ] && set -- s6-setuidgid hermes "$@"; exec "$@"; }

# HOME comes through with-contenv as /root (the /init context). Override
# to the hermes user's home before dropping privileges so libraries that
# resolve paths via $HOME (e.g. discord lockfile under XDG_STATE_HOME)
# don't try to write to /root.
export HOME=/opt/data

cd /opt/data
# shellcheck disable=SC1091
. /opt/hermes/.venv/bin/activate

if [ $# -eq 0 ]; then
    drop hermes
fi

if command -v "$1" >/dev/null 2>&1; then
    # Bare executable — pass through directly.
    drop "$@"
fi

# Hermes subcommand pass-through.
drop hermes "$@"
