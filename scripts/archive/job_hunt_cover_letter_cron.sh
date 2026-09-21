#!/usr/bin/env bash
set -euo pipefail

output=$(python3 /home/kensei/job-hunt/scripts/assemble_cover_letter.py 2>&1)
status=$?

if [ $status -ne 0 ]; then
  printf '%s\n' "$output"
  exit $status
fi

if [ -n "$output" ]; then
  printf '%s\n' "$output"
fi
