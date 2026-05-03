#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/helixa-get.sh" "/api/v2/stats"
