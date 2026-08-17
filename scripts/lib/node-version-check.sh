# node-version-check.sh -- sourceable Node version predicate for the desktop
# rebuild hand-off (scripts/desktop-update/posix.sh).
#
# The predicate is NOT hardcoded here: every call reads
# apps/desktop/package.json's `engines.node` fresh via
# scripts/lib/node-version-check.js and evaluates the given version against
# it. The official dependency declaration is the single source of truth, so
# when the desktop dependency tree's Node floor changes upstream this gate
# follows automatically -- no manual copy to keep in sync, no drift window.
#
# Semantics of the current declaration (`^22.22.0 || ^24.0.0 || >=26.0.0`):
#   - The dependency tree's real floor is >=22.22.0 (react-router 8.3.0,
#     engines.node).
#   - nanoid@6 declares `^22 || ^24 || >=26` -- which excludes the
#     odd-numbered releases 23 and 25 even though they clear the floor.
#   - The historical blanket `>=22.22` predicate (install.sh pre-#84397)
#     accepted 23/25, letting `npm ci` fail with EBADENGINE AFTER the gate
#     passed. This predicate is the intersection that actually builds.
#
# Returns 0 when the given `node --version` string (with or without the
# leading v) can build the desktop tree, 1 otherwise.
node_satisfies_build() {
  local ver="${1#v}"
  local node_bin="${2:-}"
  local lib_dir repo_root
  lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  repo_root="$(cd "$lib_dir/../.." && pwd)"
  if [ -z "$node_bin" ]; then
    command -v node >/dev/null 2>&1 || return 1
    node_bin="$(command -v node)"
  fi
  "$node_bin" "$lib_dir/node-version-check.js" "$ver" "$repo_root/apps/desktop/package.json" >/dev/null 2>&1
}
