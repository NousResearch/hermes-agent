#!/bin/sh
# Back-compat shim for callers that hard-coded docker/entrypoint.sh as the
# container ENTRYPOINT (NAS App Center / UGOS packages, old compose).
# The image's real ENTRYPOINT is docker/entrypoint-dispatch.sh.
#
# This file must exec the dispatcher (NOT only stage2-hook.sh). stage2 is
# bootstrap-only and never execs CMD; /init already runs it via
# /etc/cont-init.d/01-hermes-setup. Forwarding to the dispatcher preserves
# PID-1 vs wrapped-runtime routing so CMD still runs:
#   PID 1     → /init + main-wrapper.sh (stage2 once via cont-init)
#   non-PID-1 → stage2 then exec main-wrapper.sh
#
# Deprecation: this shim is preserved so hard-coded ENTRYPOINT overrides
# still boot. Drop the override when you can — docker will use the image's
# default dispatcher. Surface a warning to stderr so anyone still invoking
# this path sees the migration notice in their logs.
echo "[hermes] WARNING: docker/entrypoint.sh is a deprecated shim under " \
    "s6-overlay. The container's real ENTRYPOINT is " \
    "entrypoint-dispatch.sh (which delegates to /init + main-wrapper.sh " \
    "when PID 1). This shim now execs the dispatcher so CMD still runs. " \
    "If you hard-coded docker/entrypoint.sh as your ENTRYPOINT, drop the " \
    "override — docker will use the image's default ENTRYPOINT dispatcher." >&2
exec /opt/hermes/docker/entrypoint-dispatch.sh "$@"
