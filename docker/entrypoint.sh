#!/bin/sh
# s6-overlay shim. The real logic lives in docker/stage2-hook.sh, invoked
# by /etc/cont-init.d/01-hermes-setup (installed by the Dockerfile). This
# file exists so external references to docker/entrypoint.sh still work,
# but it's no longer the ENTRYPOINT — /init is.
#
# When called directly (e.g. by an old wrapper script that hard-coded
# docker/entrypoint.sh as the container ENTRYPOINT, or by an external
# orchestration script that invokes it inside the container), forward to
# the stage2 hook for parity with the pre-s6 entrypoint behavior. The
# stage2 hook only handles cont-init bootstrap (UID remap, chown, config
# seed, skills sync); it does NOT exec the CMD. Callers that depended
# on the pre-s6 contract "entrypoint.sh sets up state then execs hermes"
# will see the bootstrap happen but the CMD will not run from this shim.
#
# Deprecation: this shim is preserved for one release cycle to give
# downstream users time to migrate their wrappers to the image's real
# ENTRYPOINT (`/init`). It will be removed in a future major release.
# Surface a warning to stderr so anyone still invoking this path
# sees the migration notice in their logs.
echo "[hermes] WARNING: docker/entrypoint.sh is a deprecated shim under " \
    "s6-overlay. The container's real ENTRYPOINT is " \
    "entrypoint-dispatch.sh (which delegates to /init + main-wrapper.sh " \
    "when PID 1); this script only runs the stage2 cont-init hook " \
    "and does NOT exec the CMD. If you hard-coded docker/entrypoint.sh " \
    "as your ENTRYPOINT, drop the override — docker will use the image's " \
    "default ENTRYPOINT dispatcher, which handles bootstrap AND CMD." >&2
exec /opt/hermes/docker/stage2-hook.sh "$@"
