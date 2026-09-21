#!/usr/bin/env bash
set -euo pipefail

repo_root="${HERMES_AGENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
engine_root="$repo_root/content_engine"
builder=${REVIEW_BUNDLE_SCRIPT:-"$engine_root/scripts/build_review_bundle.py"}
python_bin=${REVIEW_BUNDLE_PYTHON:-python3}

if [[ ! -f "$builder" ]]; then
    printf 'review bundle builder missing: %s\n' "$builder" >&2
    exit 66
fi

cd "$engine_root"
export PYTHONPATH="$engine_root${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" "$builder"
