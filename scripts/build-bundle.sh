#!/usr/bin/env bash
# Build the desktop bundle for the commit checked out here, on this machine.
#
#   scripts/build-bundle.sh [--variant bundled|light] [-- <electron-builder args>]
#
# macOS gets a DMG and ZIP, Linux an AppImage, in apps/desktop/release/.
# The commit does not need to be pushed. The build is unsigned.
# Windows has its own script: scripts/build-bundle.ps1.
set -euo pipefail

variant=bundled
while [ $# -gt 0 ]; do
  case "$1" in
    --variant) variant="${2:?--variant needs bundled or light}"; shift 2 ;;
    --) shift; break ;;
    -h|--help) sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "build-bundle: unknown argument $1 (see --help)" >&2; exit 2 ;;
  esac
done
case "$variant" in bundled|light) ;; *) echo "build-bundle: --variant must be bundled or light" >&2; exit 2 ;; esac

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"

# The driver needs a host Python 3.11+ only to bootstrap; PM installs the pinned one.
python=
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1 &&
     "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
    python="$candidate"; break
  fi
done
[ -n "$python" ] || { echo "build-bundle: needs Python 3.11+ on PATH" >&2; exit 1; }

# The driver refuses a dirty tree. Say so before any work starts.
if [ -n "$(git status --porcelain --untracked-files=all)" ]; then
  echo "build-bundle: the checkout has uncommitted changes. Commit them (the build packages HEAD) or stash them." >&2
  git status --short --untracked-files=all >&2
  exit 1
fi
commit="$(git rev-parse HEAD)"

# A rebuild in the same checkout needs the previous work and outputs gone.
# .cache stays: it holds downloaded tools and is safe to reuse.
for stale in .build/desktop-job apps/desktop/build apps/desktop/dist apps/desktop/release; do
  if [ -e "$stale" ]; then
    echo "build-bundle: removing previous output $stale"
    chmod -R u+w "$stale" 2>/dev/null || true
    rm -rf "$stale"
  fi
done

echo "build-bundle: building $commit ($variant)"
"$python" scripts/bundles/desktop.py --commit "$commit" --variant "$variant" ${1+-- "$@"}

echo "build-bundle: done. Artifacts in $repo/apps/desktop/release:"
ls -lh apps/desktop/release | sed 's/^/  /'
