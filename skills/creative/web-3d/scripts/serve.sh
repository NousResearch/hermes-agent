#!/bin/bash
# web-3d skill - local static server for demos/assets

set -euo pipefail

PORT="${1:-8080}"
DIR="${2:-.}"

echo "Serving $(cd "$DIR" && pwd) at http://localhost:$PORT"
cd "$DIR"
python3 -m http.server "$PORT"
