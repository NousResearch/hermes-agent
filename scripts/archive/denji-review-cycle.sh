#!/usr/bin/env bash
# Wrapper for denji-review-cycle.py in the KenseiAgent repo.
# Called by the Hermes cron scheduler.
exec python3 /home/kensei/repos/KenseiAgent/scripts/denji-review-cycle.py "$@"
