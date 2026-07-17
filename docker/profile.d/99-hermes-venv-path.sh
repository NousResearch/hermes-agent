# Re-prepend the Hermes venv/bin directories onto PATH for login shells.
#
# Debian's /etc/profile hardcodes PATH for login shells, discarding the image's
# `ENV PATH` *before* it sources /etc/profile.d/*.sh. The terminal tool builds
# its environment snapshot through a login shell (`bash -l -c ... export -p`,
# see tools/environments/base.py:init_session), so without this drop-in the
# snapshot -- and every later terminal command that sources it -- loses the venv
# and a bare `python3` resolves to the system interpreter, which has none of
# Hermes' dependencies (not even the Pillow that hermes-agent itself imports).
#
# This file is sourced *after* the reset, so re-prepending here repairs it.
# Prepend only when the venv prefix isn't already leading PATH: that keeps the
# repair idempotent when a login shell re-sources this file, yet still fixes a
# PATH where the venv is merely present but not in front (a bare `python3` would
# otherwise resolve to /usr/bin/python3). Keep the directory list in sync with
# `ENV PATH` in the Dockerfile.
# See https://github.com/NousResearch/hermes-agent/issues/56634.
case "${PATH}" in
    /opt/hermes/bin:/opt/hermes/.venv/bin:/opt/data/.local/bin:*) ;;
    *) PATH="/opt/hermes/bin:/opt/hermes/.venv/bin:/opt/data/.local/bin:${PATH}" ;;
esac
export PATH
