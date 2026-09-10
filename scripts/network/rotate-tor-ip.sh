#!/bin/bash
set -euo pipefail
# Rotate Tor exit IP — send NEWNYM signal
# kensei is a real member of the debian-tor group (needed to read
# /run/tor/control.authcookie), so no group-switching wrapper is needed —
# `sg`/`newgrp` requires setgid() privilege, which NoNewPrivileges=true on
# the calling systemd service blocks even for a group we're already in.
# Usage: rotate-tor-ip.sh [--quiet]

QUIET=false
[ "${1:-}" = "--quiet" ] && QUIET=true

OUTPUT=$(/usr/bin/python3 /home/kensei/.hermes/scripts/tor-newnym.py 2>&1) || RC=$?

if [ "$QUIET" = true ]; then
    exit "${RC:-0}"
fi
echo "$OUTPUT"
exit "${RC:-0}"