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
#   NOVA_OIDC_ISSUER       <unset>                Cognito sign-in via the load balancer:
#   NOVA_OIDC_CLIENT_ID    <unset>                  the user pool issuer and app client id
#   NOVA_OIDC_ADMIN_GROUP  nova-admin             Cognito group that makes an admin
#   NOVA_OIDC_VIEWER_GROUP nova-viewer            Cognito group that makes a viewer
#   NOVA_OIDC_LOGOUT_URL   <unset>                where Sign out goes (the pool's /logout)
#   NOVA_APPLY_ON_START    <unset>                any non-empty value runs `nova apply` first
#   NOVA_PRUNE_ORPHANS     <unset>                with APPLY_ON_START, also `--prune`
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
    [ -n "${NOVA_OIDC_ISSUER:-}" ] && set -- "$@" --oidc-issuer "$NOVA_OIDC_ISSUER"
    [ -n "${NOVA_OIDC_CLIENT_ID:-}" ] && set -- "$@" --oidc-client-id "$NOVA_OIDC_CLIENT_ID"
    [ -n "${NOVA_OIDC_ADMIN_GROUP:-}" ] && set -- "$@" --oidc-admin-group "$NOVA_OIDC_ADMIN_GROUP"
    [ -n "${NOVA_OIDC_VIEWER_GROUP:-}" ] && set -- "$@" --oidc-viewer-group "$NOVA_OIDC_VIEWER_GROUP"
    [ -n "${NOVA_OIDC_LOGOUT_URL:-}" ] && set -- "$@" --oidc-logout-url "$NOVA_OIDC_LOGOUT_URL"
    printf '%s\n' "$@"
}

command="${1:-serve}"
[ $# -gt 0 ] && shift

case "$command" in
    serve)
        ensure_state
        require_bundle
        if [ -n "${NOVA_APPLY_ON_START:-}" ]; then
            # Reconcile the bundle into runtime profiles before serving. Without this the
            # declared agents exist only as YAML: the dispatcher asks
            # `profiles.profile_exists(assignee)` before spawning, and a name with no
            # profile directory is refused silently — the board fills with `ready` tasks
            # nothing will ever claim. That is not hypothetical; it is how the first field
            # deployment failed.
            #
            # A failing apply does NOT stop the serve. `set -e` would turn a malformed
            # bundle into a crash-looping container, and the control plane is the surface
            # an operator uses to see WHY it is malformed — taking it away is the one
            # response that removes the diagnosis along with the symptom. The failure is
            # logged loudly, the exit status is kept, and the tasks screen reports the
            # board as unattended on its own.
            # NOVA_PRUNE_ORPHANS opts this boot apply into reconciling NOVA-managed
            # profiles the bundle no longer declares (e.g. a channel-derived profile left
            # behind when a channel is removed). Off by default: an unattended apply on
            # every restart must not quietly delete state on a transient bad bundle, and
            # `nova apply --prune` already keeps any profile that still holds customer
            # state. Set it only when the bundle is meant to be the whole truth for this
            # host.
            prune_flag=""
            [ -n "${NOVA_PRUNE_ORPHANS:-}" ] && prune_flag="--prune"
            log "applying $NOVA_BUNDLE before serving (NOVA_APPLY_ON_START${prune_flag:+, $prune_flag})"
            if python -m nova apply "$NOVA_BUNDLE" $prune_flag; then
                log "apply finished; declared agents are materialized as runtime profiles"
            else
                status=$?
                log "APPLY FAILED (exit $status). Serving anyway so the Control API can say"
                log "why — but no new agent profiles were written, so work assigned to an"
                log "agent that has never been applied will sit in 'ready' unclaimed."
            fi
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
