#!/command/with-contenv sh
# shellcheck shell=sh
# hermes-role.sh -- print this container's role: "gateway" or "dashboard".
#
# SOURCE this file (`. /opt/hermes/docker/hermes-role.sh`); it only defines
# the `hermes_role` function and runs nothing at load time.
#
# Why this exists
# ---------------
# The standard deployment runs TWO containers from the SAME image with the
# SAME /init + cont-init.d hooks, both mounting the same ~/.hermes volume at
# /opt/data:
#
#     gateway   -> CMD `gateway run`            (serves the bots + :11435 proxy)
#     dashboard -> CMD `dashboard --host ...`   (serves the web UI on :9119)
#
# Only the GATEWAY container may seed and run the per-profile gateways and the
# Claude proxy (:11435). That proxy's s6-log writes to the SHARED
# /opt/data/logs/claude-proxy/ and takes an exclusive lock there. When the
# dashboard container also seeds the proxy (03-seed-data-services) and starts
# it, whichever container's s6-log wins the lock first starves the other:
# the loser dies with "s6-log: fatal: unable to lock
# /opt/data/logs/claude-proxy/lock: Resource busy" (exit 111) and flaps
# forever, taking the proxy (and the bots) down with it. This was observed in
# production -- the dashboard container held the lock and the gateway's
# logger never recovered. See .github/workflows/diagnose-contabo.yml for the
# read-only probe that confirmed it.
#
# Role resolution (first match wins)
# ----------------------------------
#   1. $HERMES_ROLE, when explicitly set to `gateway` or `dashboard`. The
#      dashboard service sets HERMES_ROLE=dashboard in docker-compose.yml;
#      with-contenv exposes the docker `environment:` to cont-init hooks.
#   2. Otherwise infer from the container CMD. s6-overlay's rc.init carries
#      the CMD after the main program:
#        ... /run/s6/basedir/scripts/rc.init top \
#                /opt/hermes/docker/main-wrapper.sh <CMD...>
#      A leading `dashboard` token means the dashboard container; anything
#      else (`gateway run`, a bare exec, or an empty CMD) is the gateway.
#   3. Default to `gateway`. Single-container / all-in-one deployments run the
#      whole stack, and a detection miss must fail safe toward "run the
#      gateway", never toward "silently run nothing".
hermes_role() {
    case "${HERMES_ROLE:-}" in
        dashboard) echo dashboard; return 0 ;;
        gateway)   echo gateway;   return 0 ;;
    esac

    # Infer from the CMD that s6-overlay's rc.init (and, once started, the
    # main-wrapper.sh "main program") carries in its argv. Both reference
    # main-wrapper.sh; the token immediately after it is the docker CMD's
    # first word -- the role discriminator. Our own hook's argv does NOT
    # contain main-wrapper.sh, so we never match ourselves.
    for cl in /proc/[0-9]*/cmdline; do
        args=$(tr '\0' ' ' < "$cl" 2>/dev/null) || continue
        case "$args" in
            *main-wrapper.sh\ *) ;;
            *) continue ;;
        esac
        cmd=${args#*main-wrapper.sh }
        # shellcheck disable=SC2086  # deliberate split into positional args
        set -- $cmd
        case "${1:-}" in
            dashboard) echo dashboard; return 0 ;;
        esac
    done

    echo gateway
}
