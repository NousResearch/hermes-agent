#!/usr/bin/env bash
# Sync Hub mirrors + skills/neon-genie package from distribution.yaml.
# Canonical: python scripts/distribution_spine.py write
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
exec python3 scripts/distribution_spine.py write
