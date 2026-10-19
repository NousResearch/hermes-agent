#!/usr/bin/env bash
# ts_prefix: prefix each stdin line with [+MM:SS] relative to the DRIVER's
# start (TS_BASE), not this pipe's start -- every log in a leg shares one
# time base, so the playback.html sync (one offset slider against the
# recording) aligns all files at once. The playback.html leg player uses
# these prefixes to sync a log file to the screen recording's timeline
# (the video starts a few seconds before the driver, hence the player's
# offset slider).
#
# Usage (sourced from a driver):
#   TS_BASE=$SECONDS        # once, at driver start
#   cmd 2>&1 | ts_prefix > "$LOG_DIR/x.log"
#
# Must be the LAST consumer in the pipe: with `set -o pipefail` the exit
# status stays the command's, while ts_prefix's own status is always 0.
# SECONDS is bash-specific, so this helper is bash-only.

ts_prefix() {
  local _base="${TS_BASE:-$SECONDS}"
  local _line _t
  while IFS= read -r _line; do
    _t=$((SECONDS - _base))
    printf '[+%02d:%02d] %s\n' $((_t / 60)) $((_t % 60)) "$_line"
  done
}
