#!/usr/bin/env bash
# Example cron wrapper showing coord_guard.sh as step 0.
#
# Pattern: your scheduler runs THIS wrapper; the wrapper consults the
# coordination board before doing any real work. If an interactive session
# holds what this job touches, the job defers (rc 75 from the guard) and
# simply exits 0 — silent skip; the job fires again on its own schedule.
#
# The guard is FAIL-OPEN: if the board/CLI is missing or broken, the job
# proceeds unguarded. Coordination protects work; it never blocks a backup.

set -euo pipefail

# --- step 0: coordination guard -------------------------------------------
# A full install puts both files in ~/.hermes/scripts. In a skill-bundle
# install, default to this example's sibling scripts directory.
EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COORD_SCRIPTS_DIR=${COORD_SCRIPTS_DIR:-"$(cd "$EXAMPLE_DIR/../scripts" && pwd)"}
# shellcheck source=/dev/null
source "$COORD_SCRIPTS_DIR/coord_guard.sh"

# args: <job-id-in-your-manifest> <policy skip|wait> <wait-timeout-s> <claim-ttl-min>
guard_rc=0
coord_guard "nightly-backup-example" wait 900 90 || guard_rc=$?
case "$guard_rc" in
  0) ;;            # claimed, or the guard deliberately failed open
  75) exit 0 ;;    # another actor holds the resource: skip this tick silently
  *) exit "$guard_rc" ;;  # wrapper/configuration error, not a board outage
esac
# rc 0: either claimed (COORD_GUARD_ID set, EXIT trap releases) or fail-open.

# --- the actual job --------------------------------------------------------
echo "backing up..." >&2
# tar -czf ... etc.

# Claims auto-release via the EXIT trap installed by coord_guard.
# If you define your own EXIT trap later in this script, include `coord_done`.
