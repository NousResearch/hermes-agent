#!/bin/sh
# shellcheck shell=sh
# /usr/bin/tini — compatibility shim for legacy orchestration templates.
#
# Background
# ----------
# The published image used to ship real `tini` as PID 1. After the
# s6-overlay migration, PID 1 is `/init`. Downstream catalogs (Hostinger
# Hermes WebUI, NAS "update" flows that preserve an old entrypoint, etc.)
# still pin entrypoints like:
#
#   ["/usr/bin/tini", "-g", "--"]
#   ["/usr/bin/tini", "-g", "--", "gateway", "run"]
#
# A plain `ln -sf /init /usr/bin/tini` (#34192) made the binary exist, but
# forwarded tini's own flags into s6-overlay. `/init` then handed `-g` to
# `rc.init` as the container CMD, which fails with:
#
#   /run/s6/basedir/scripts/rc.init: 91: -g: not found
#
# …and with `restart: unless-stopped` the container boot-loops forever
# (#66679).
#
# This shim strips the tini CLI surface, then exec's `/init` with the
# image's main-wrapper so privilege drop + arg routing still apply.
#
# Tini flags we discard (see krallin/tini --help):
#   -g / --kill-process-group, -s / --subreaper, -w, -v (repeatable),
#   -h / --help, -l, --version, -p SIGNAL, -e EXIT_CODE, and `--`.
# s6-overlay already reaps zombies and forwards signals as PID 1; the
# flag semantics are intentionally no-ops here.
#
# Test hook: HERMES_TINI_SHIM_TARGET overrides the `/init` path (and
# HERMES_TINI_SHIM_WRAPPER the main-wrapper path) so unit tests can
# record argv without a real s6 tree.

set -e

INIT_TARGET="${HERMES_TINI_SHIM_TARGET:-/init}"
WRAPPER="${HERMES_TINI_SHIM_WRAPPER:-/opt/hermes/docker/main-wrapper.sh}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --)
            shift
            break
            ;;
        -g|--kill-process-group|-s|--subreaper|-w|-l|-h|--help|--version)
            shift
            ;;
        -v)
            # tini allows repeated -v; drop them all.
            shift
            ;;
        -p|-e)
            # These take a mandatory argument.
            shift
            if [ "$#" -gt 0 ]; then
                shift
            fi
            ;;
        -*)
            # Unknown short/long option from an older/newer tini — drop
            # rather than forwarding into /init (would recreate #66679).
            shift
            ;;
        *)
            break
            ;;
    esac
done

# No program left (entrypoint was just `tini -g --`) → same as the
# image's default ENTRYPOINT + empty CMD.
if [ "$#" -eq 0 ]; then
    exec "$INIT_TARGET" "$WRAPPER"
fi

# Caller already supplied the wrapper or /init — don't double-wrap.
case "$1" in
    /init|"$INIT_TARGET")
        exec "$@"
        ;;
    "$WRAPPER"|*/main-wrapper.sh)
        exec "$INIT_TARGET" "$@"
        ;;
esac

exec "$INIT_TARGET" "$WRAPPER" "$@"
