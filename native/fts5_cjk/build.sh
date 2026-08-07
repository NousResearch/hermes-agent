#!/bin/bash
# Build libfts5_cjk.so and install to ~/.hermes/lib/ (or $1).
#
# Uses the vendored public-domain SQLite extension headers. Some platforms
# ship a sqlite3ext.h that exists but deliberately defines
# SQLITE_OMIT_LOAD_EXTENSION (notably the macOS SDK), which makes a simple
# header-presence probe produce a shared object with unresolved symbols.
set -euo pipefail
cd "$(dirname "$0")"

dest="${1:-$HOME/.hermes/lib}"
mkdir -p "$dest"
compiler="${CC:-gcc}"
tmp_output="$(mktemp "$dest/.libfts5_cjk.XXXXXX")"
trap 'rm -f "$tmp_output"' EXIT

"$compiler" -shared -fPIC -O2 -Wall -Wextra -Ivendor \
  fts5_cjk.c -o "$tmp_output"
install -m 0644 "$tmp_output" "$dest/libfts5_cjk.so"
echo "installed: $dest/libfts5_cjk.so"
