#!/usr/bin/env bash
# Purge macOS build-host metadata from a staged tree: AppleDouble `._*`
# sidecars and `.DS_Store` entries (#126097). Any `._*`-named file is
# build-host leakage by definition -- the Finder/xattr convention has no
# legitimate use in a Termux payload -- which keeps this purge symmetric
# with the name-based install gate in validate_installed.py.
#
# Usage: purge_macos_metadata.sh <root-dir>
# Prints the number of purged files; exits non-zero only on real errors.
set -Eeuo pipefail

root="${1:?usage: purge_macos_metadata.sh <root-dir>}"
count=0
while IFS= read -r -d '' stray; do
    rm -f -- "$stray"
    count=$((count + 1))
done < <(find "$root" \( -name '._*' -o -name '.DS_Store' \) -type f -print0)
printf '%s\n' "$count"
