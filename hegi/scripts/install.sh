#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
hermes_home="${HERMES_HOME:-$HOME/.hermes}"
target="$hermes_home/hegi"

mkdir -p "$target/archive"
if [[ ! -e "$target/config.yaml" ]]; then
  cp "$repo_root/hegi/config/default.yaml" "$target/config.yaml"
  chmod 600 "$target/config.yaml"
  echo "HEGI config installed: $target/config.yaml"
else
  echo "HEGI config already exists; unchanged: $target/config.yaml"
fi
