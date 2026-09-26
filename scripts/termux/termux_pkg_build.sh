#!/usr/bin/env bash
# Stage the pinned bionic python/node .debs into the payload, via pm.
#
# pm owns everything here: the lock rows (pm/lock.json, linux-arm64-bionic
# targets of the python/node packages), the digest-verified download, the
# hardened ar+tar extraction (DebPackage.unpack), and the file-evidence
# verify (the staged binaries are bionic and cannot exec on the staging
# host). stage_only() publishes the store entry WITHOUT touching this
# host's installed facts -- the payload copy below is a pure consumer.
#
# Usage: termux_pkg_build.sh <package> <subdir> <payload-dir>
#   <package>     pm package name: python | node
#   <subdir>      payload subdir the staged tree lands in: python | node
#   <payload-dir> payload root
set -euo pipefail

PKG="$1"
SUBDIR="$2"
mkdir -p "$3"
PAYLOAD="$(cd "$3" && pwd)"
STAGED="$PAYLOAD/$SUBDIR"

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$REPO_ROOT"

TARGET="linux-arm64-bionic"

# Stage (idempotent: an already-published verifying entry is a no-op) and
# print the entry path -- pm's own resolvers, no layout guessing here.
# Resolve a real interpreter: some hosts alias python3 to a Store stub
# (Windows App Execution Aliases). The stdlib client bootstraps PM's separate
# locked runtime; staging never imports PM's dependencies into this process.
PM_PY="$(command -v python3 || true)"
if [ -n "$PM_PY" ] && "$PM_PY" -c "import sys; assert sys.version_info >= (3, 11)" 2>/dev/null; then
  :
else
  PM_PY="$(command -v python || true)"
fi
[ -n "$PM_PY" ] || { echo "termux_pkg_build: no usable python found" >&2; exit 1; }
ENTRY_DIR="$("$PM_PY" - "$PKG" "$TARGET" <<'PYEOF'
import sys
sys.path.insert(0, ".")
from pm.client import stage_only
# stage_only already returns the published entry path -- no re-derivation.
print(stage_only(sys.argv[1], sys.argv[2]))
PYEOF
)" || { echo "termux_pkg_build: pm stage failed" >&2; exit 1; }
[ -n "$ENTRY_DIR" ] && [ -d "$ENTRY_DIR" ] || { echo "termux_pkg_build: staged entry missing: $ENTRY_DIR" >&2; exit 1; }

rm -rf "$STAGED"
cp -a "$ENTRY_DIR" "$STAGED"
echo "termux_pkg_build($PKG): pm entry staged -> $STAGED"
