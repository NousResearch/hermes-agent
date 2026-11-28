#!/bin/sh
# shellcheck shell=sh
# Entry-point dispatcher for runtimes that may or may not give the image
# ownership of PID 1.
#
# Normal Docker / Podman path: this script is PID 1, so we delegate to
# s6-overlay's /init exactly as before and keep the full supervision tree.
#
# Wrapped-runtime path (Fly Machines, `docker run --init`, some Nomad/K8s
# setups): the platform's own init is already PID 1 and execs the image
# entrypoint as a child. s6-overlay aborts there with "can only run as pid 1",
# so we run the stage2 bootstrap directly and then exec the main wrapper
# without /init.

set -e

# No command and no terminal (`docker run -d <image>`): the interactive CLI would quit at once on
# a non-TTY stdin and take the container down in a --restart loop (#159590). Run the documented
# headless mode instead. Chosen HERE, before /init, so the boot reconciler reads the same
# `gateway run` argv the main program runs (one gateway, a saved `gateway stop` respected, the
# HERMES_GATEWAY_NO_SUPERVISE opt-out handled exactly as an explicit `gateway run`).
if [ $# -eq 0 ] && ! [ -t 0 ]; then
    echo "[hermes] no command and no terminal: running 'hermes gateway run' (use 'docker run -it' for the interactive CLI)" >&2
    set -- /opt/hermes/bin/hermes gateway run
fi

if [ "$$" -eq 1 ]; then
    # The interactive CMD owns the terminal. Supervised gateways and dashboard
    # output must go to s6's logger instead of overwriting the TUI or shell.
    # Resolve this before /init wires the service descriptors, leaving headless
    # docker logs and explicit s6 settings unchanged.
    if [ -t 0 ] && [ -t 1 ]; then
        export S6_LOGGING="${S6_LOGGING:-2}"
        export S6_CMD_USE_TERMINAL="${S6_CMD_USE_TERMINAL:-1}"
    fi
    exec /init /opt/hermes/docker/main-wrapper.sh "$@"
fi

echo "[hermes] WARNING: container entrypoint is not PID 1; skipping s6-overlay /init and falling back to direct bootstrap. Supervised services are unavailable in this runtime, but the requested command will still run." >&2
# /init normally seeds PATH with s6's helpers; the non-PID-1 fallback skips it.
export PATH="/command:/package/admin/s6/command:${PATH}"
/opt/hermes/docker/stage2-hook.sh
exec /opt/hermes/docker/main-wrapper.sh "$@"
