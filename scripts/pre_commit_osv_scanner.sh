#!/usr/bin/env bash
# Pre-commit: osv-scanner audit (non-blocking — matches upstream fail-on-vuln: false).
# Reports known CVEs in pinned dependencies but does not block commits.
set -euo pipefail

ARGS=(--lockfile=uv.lock)
[ -f package-lock.json ] && ARGS+=(--lockfile=package-lock.json)

osv-scanner "${ARGS[@]}" || true
