#!/usr/bin/env bash
set -euo pipefail

mkdir -p /sandbox /agent/.hermes/skills /agent/.hermes/credentials /agent/.hermes/cache
cd /sandbox
if [ "$#" -eq 0 ]; then
  exec bash
fi
exec "$@"
