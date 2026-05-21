#!/bin/sh
# s6-overlay shim. The real logic lives in docker/stage2-hook.sh, invoked
# by /etc/cont-init.d/01-hermes-setup (installed by the Dockerfile). This
# file exists so external references to docker/entrypoint.sh still work,
# but it's no longer the ENTRYPOINT — /init is.
#
# When called directly (e.g. by an old wrapper script that hard-coded
# docker/entrypoint.sh), forward to the stage2 hook for parity with the
# pre-s6 entrypoint behavior.
exec /opt/hermes/docker/stage2-hook.sh "$@"
