#!/bin/sh
# NOVA control-plane container entrypoint.
#
# Thin on purpose. It resolves three things the container needs and the image cannot
# know — where the tenant bundle is, where state lives, what to bind — and then execs
# the Python process so it becomes PID 1 and receives SIGTERM directly. `nova serve`
# installs a SIGTERM handler for exactly this; wrapping it in a supervisor would only put
# something between the signal and the code that handles it.
#
# Nothing here weakens NOVA's own exposure guards. Binding a non-loopback interface
# still requires a principals file and TLS (or an explicit statement that a proxy
# terminates it) — this script passes the operator's choice through and lets
# nova/control/server.py refuse. It does not supply a default that would make the
# refusal go away.
#
# Environment (all optional; defaults shown):
#   NOVA_BUNDLE            /var/lib/nova/bundle   tenant declarations, on the volume
#   NOVA_HOME              /var/lib/nova/home     runtime + control-plane state
#   NOVA_BIND_HOST         127.0.0.1
#   NOVA_BIND_PORT         8787
#   NOVA_PRINCIPALS        <unset>                default: $NOVA_HOME/control-principals.yaml
#   NOVA_TLS_CERT          <unset>
#   NOVA_TLS_KEY           <unset>
#   NOVA_BEHIND_TLS_PROXY  <unset>                any non-empty value sets the flag
#   NOVA_APPLY_ON_START    <unset>                any non-empty value runs `nova apply` first
#   NOVA_LOG_LEVEL         <unset>                read by nova itself
#   NOVA_LOG_FORMAT        json                   read by nova itself
#
# Arguments: `serve` (the image's CMD), `apply`, `plan`, `validate`, or `--` followed by
# a raw `python -m nova` invocation for debugging over SSM.

set -eu

NOVA_BUNDLE="${NOVA_BUNDLE:-/var/lib/nova/bundle}"
NOVA_HOME="${NOVA_HOME:-/var/lib/nova/home}"
NOVA_BIND_HOST="${NOVA_BIND_HOST:-127.0.0.1}"
NOVA_BIND_PORT="${NOVA_BIND_PORT:-8787}"
export NOVA_HOME
# The runtime resolves its own home from HERMES_HOME. Pointing both at one directory is
# the documented alias chain (nova/runtime/hermes/paths.py), and means a `hermes_cli`
# command run inside this container for debugging reads the same state the control plane
# is serving, rather than a second, empty home under /home/nova.
export HERMES_HOME="${HERMES_HOME:-$NOVA_HOME}"

log() { printf 'nova-entrypoint: %s\n' "$*" >&2; }

die() { log "$*"; exit 1; }

ensure_state() {
    # The AWS unit mounts an empty XFS volume over /var/lib/nova, so the directory the
    # image created is gone by the time this runs. Create what is missing; never touch
    # what is already there.
    mkdir -p "$NOVA_HOME" 2>/dev/null || die "cannot create $NOVA_HOME. Is the state volume mounted and writable by uid $(id -u)?"
    [ -w "$NOVA_HOME" ] || die "$NOVA_HOME is not writable by uid $(id -u). The state volume must be owned by the container user."
}

require_bundle() {
    [ -d "$NOVA_BUNDLE" ] || die "no tenant bundle at $NOVA_BUNDLE. Place the tenant's declarations on the state volume, or set NOVA_BUNDLE. This image ships no tenant configuration."
    [ -f "$NOVA_BUNDLE/organization.yaml" ] || die "$NOVA_BUNDLE has no organization.yaml, so it is not a tenant bundle."
}

# TLS and principals are passed through exactly as the operator set them. An unset value
# means the flag is not passed at all, which is not the same as passing an empty one.
serve_flags() {
    set -- --host "$NOVA_BIND_HOST" --port "$NOVA_BIND_PORT"
    [ -n "${NOVA_PRINCIPALS:-}" ] && set -- "$@" --principals "$NOVA_PRINCIPALS"
    [ -n "${NOVA_TLS_CERT:-}" ] && set -- "$@" --tls-cert "$NOVA_TLS_CERT"
    [ -n "${NOVA_TLS_KEY:-}" ] && set -- "$@" --tls-key "$NOVA_TLS_KEY"
    [ -n "${NOVA_BEHIND_TLS_PROXY:-}" ] && set -- "$@" --behind-tls-proxy
    printf '%s\n' "$@"
}

command="${1:-serve}"
[ $# -gt 0 ] && shift

case "$command" in
    serve)
        ensure_state
        require_bundle
        if [ -n "${NOVA_APPLY_ON_START:-}" ]; then
            log "applying $NOVA_BUNDLE before serving"
            python -m nova apply "$NOVA_BUNDLE"
        fi
        log "serving $NOVA_BUNDLE on $NOVA_BIND_HOST:$NOVA_BIND_PORT (state: $NOVA_HOME)"
        # Read the flag list into positional parameters without a subshell swallowing the
        # exec: `set --` with command substitution, newline-separated so a path with a
        # space survives.
        OLDIFS=$IFS; IFS='
'
        set -- $(serve_flags)
        IFS=$OLDIFS
        exec python -m nova serve "$NOVA_BUNDLE" "$@"
        ;;
    apply|plan|validate)
        ensure_state
        require_bundle
        exec python -m nova "$command" "$NOVA_BUNDLE" "$@"
        ;;
    --)
        # Raw passthrough for debugging: `docker run ... -- nova status` and similar.
        ensure_state
        exec python -m nova "$@"
        ;;
    *)
        die "unknown command '$command'. Expected: serve, apply, plan, validate, or -- followed by nova arguments."
        ;;
esac
