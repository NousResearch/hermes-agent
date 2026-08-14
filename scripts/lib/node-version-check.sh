# node-version-check.sh -- sourceable Node version predicate shared by the
# installer (scripts/install.sh) and the desktop rebuild hand-off
# (scripts/desktop-update/posix.sh).
#
# Single source of truth for "which Node can build the desktop dependency
# tree". Every node gate in the repo should read from here so a toolchain
# floor change lands in one place instead of drifting across copies.
#
# Returns 0 when the given `node --version` string (with or without the
# leading v) can build the desktop tree, 1 otherwise.
#
# Why this boundary:
#   - The dependency tree's real floor is >=22.22.0 (react-router 8.3.0,
#     engines.node).
#   - nanoid@6 declares `^22 || ^24 || >=26` — which excludes the
#     odd-numbered releases 23 and 25 even though they clear the floor.
#   - The historical blanket `>=22.22` predicate (install.sh pre-#84397)
#     accepted 23/25, letting `npm ci` fail with EBADENGINE AFTER the gate
#     passed. This predicate is the intersection that actually builds.
#   - Keep in sync with install.sh's node_satisfies_build; if that function
#     moves here, update install.sh to source this file instead of defining
#     its own copy.
node_satisfies_build() {
  local ver="${1#v}"
  local major="${ver%%.*}"
  local minor="${ver#*.}"; minor="${minor%%.*}"
  case "$major" in ''|*[!0-9]*) return 1 ;; esac
  case "$minor" in ''|*[!0-9]*) minor=0 ;; esac
  if [ "$major" -eq 22 ] && [ "$minor" -ge 22 ]; then return 0; fi
  if [ "$major" -eq 24 ] || [ "$major" -ge 26 ]; then return 0; fi
  return 1
}
