#!/usr/bin/env bash
# Nous Discord archive digest — data collector for the LLM cron.
# Wraps ~/.hermes/nous-archive/pull_and_diff.py; its stdout is injected
# into the cron prompt as context.
set -u
exec python3 /home/kensei/.hermes/nous-archive/pull_and_diff.py
