#!/bin/bash
# Build libfts5_cjk.so and install to ~/.hermes/lib/ (or $1).
#
# Uses the system sqlite3ext.h when present, else the vendored copy in
# vendor/ (public-domain SQLite amalgamation headers) so the build works
# without libsqlite3-dev installed.
set -euo pipefail
cd "$(dirname "$0")"

CFLAGS_EXTRA=""
if ! echo '#include <sqlite3ext.h>' | gcc -E -xc - >/dev/null 2>&1; then
  CFLAGS_EXTRA="-Ivendor"
fi

gcc -shared -fPIC -O2 -Wall -Wextra $CFLAGS_EXTRA fts5_cjk.c -o libfts5_cjk.so
dest="${1:-$HOME/.hermes/lib}"
mkdir -p "$dest"
install -m 0644 libfts5_cjk.so "$dest/libfts5_cjk.so"
echo "installed: $dest/libfts5_cjk.so"
