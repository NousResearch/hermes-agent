#!/usr/bin/env bash
# Nous Discord archive digest — deterministic no_agent collector.
# Runs pull_and_diff.py, then assembles the clean digest message + dark HTML
# via build_digest.py and prints it to stdout with the MEDIA: line LAST.
# Empty stdout (NO_NEW_MESSAGES / FIRST_RUN) = cron [SILENT].
set -u
exec python3 /home/kensei/.hermes/nous-archive/nous_digest_run.py