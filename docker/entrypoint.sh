#!/bin/sh
# s6-overlay shim. The real logic lives in docker/stage2-hook.sh, invoked
# by /etc/cont-init.d/01-hermes-setup (installed by the Dockerfile). This
# file exists so external references to docker/entrypoint.sh still work,
# but it's no longer the ENTRYPOINT — entrypoint-dispatch.sh is.
#
# When called directly (e.g. by an old wrapper script that hard-coded
# docker/entrypoint.sh as the container ENTRYPOINT, or an image-upgrade
# package that carried the override forward), this is PID 1 and there is
# no /init to hand off to. Run the stage2 bootstrap (UID remap, chown,
# config seed, skills sync) and then exec the CMD directly, the same
# fallback entrypoint-dispatch.sh uses when it isn't PID 1 either. This
# keeps the pre-s6 contract "entrypoint.sh sets up state then execs
# hermes" intact instead of leaving the container exit after bootstrap
# with the CMD never run (#107263).
#
# Full s6 supervision (dashboard side-process, per-profile gateways) is
# unavailable on this path since /init never runs. Surface a warning to
# stderr so anyone still invoking this path sees the migration notice.
#
# Deprecation: this shim is preserved for one release cycle to give
# downstream users time to migrate their wrappers to the image's real
# ENTRYPOINT (entrypoint-dispatch.sh). It will be removed in a future
# major release.
#
# Test hook: HERMES_ENTRYPOINT_SHIM_STAGE2 / HERMES_ENTRYPOINT_SHIM_WRAPPER
# override the stage2-hook.sh / main-wrapper.sh paths so unit tests can
# record behavior without a real s6 tree or root privileges.

set -e

STAGE2="${HERMES_ENTRYPOINT_SHIM_STAGE2:-/opt/hermes/docker/stage2-hook.sh}"
WRAPPER="${HERMES_ENTRYPOINT_SHIM_WRAPPER:-/opt/hermes/docker/main-wrapper.sh}"

echo "[hermes] WARNING: docker/entrypoint.sh is a deprecated shim under " \
    "s6-overlay. The container's real ENTRYPOINT is " \
    "entrypoint-dispatch.sh. This path runs stage2 bootstrap and the " \
    "requested command, but supervised services (dashboard, per-profile " \
    "gateways) are unavailable since /init never runs. If you hard-coded " \
    "docker/entrypoint.sh as your ENTRYPOINT, drop the override — docker " \
    "will use the image's default ENTRYPOINT dispatcher instead." >&2

# /init normally seeds PATH with s6's helpers; this path skips it.
export PATH="/command:/package/admin/s6/command:${PATH}"
"$STAGE2"
exec "$WRAPPER" "$@"
