#!/bin/bash
# Build libfts5_cjk.so and install to ~/.hermes/lib/ (or $1).
set -euo pipefail
cd "$(dirname "$0")"
gcc -shared -fPIC -O2 -Wall -Wextra fts5_cjk.c -o libfts5_cjk.so
dest="${1:-$HOME/.hermes/lib}"
mkdir -p "$dest"
install -m 0644 libfts5_cjk.so "$dest/libfts5_cjk.so"
echo "installed: $dest/libfts5_cjk.so"
