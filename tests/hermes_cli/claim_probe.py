"""Disposable claim contender/crash probe, not an agent or task worker.

Run as a CHILD PROCESS by test_kanban_authority_history.py: two real processes
racing one claim, and a process killed either side of COMMIT. Those properties
cannot be observed from inside a single interpreter, which is why this is a
separate script rather than a fixture.

It lived untracked in .phase3-evidence/ until the current-main merge, so the
three tests that spawn it passed only on the machine that authored it and would
have failed in any fresh clone. It is a test dependency, so it belongs in tests/
and must be tracked.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Sandbox guards. The probe writes through the real kanban code, so it must
# never be pointed at a live Hermes home. HERMES_HOME is no longer placed under
# the repo (pytest puts it in its own tmp tree), so containment in ROOT is not
# the property to assert -- "is not the user's actual home" is.
home = os.environ.get("HERMES_HOME")
assert home, "HERMES_HOME must be set; refusing to run against a default home"
home_resolved = Path(home).resolve()
sys.path.insert(0, str(ROOT))
from hermes_constants import _get_platform_default_hermes_home  # noqa: E402

assert home_resolved != Path(_get_platform_default_hermes_home()).resolve(), (
    f"refusing to run against the real Hermes home: {home_resolved}"
)

from hermes_cli import kanban_db as kb  # noqa: E402
from hermes_cli import kanban_db_connect as kbc  # noqa: E402

# The code under test, not an installed copy that happens to be importable.
assert Path(kb.__file__).resolve() == ROOT / "hermes_cli" / "kanban_db.py"

path, task, mode = sys.argv[1:]
# The target database may be a SIBLING of HERMES_HOME (the tests put board.db
# next to it under one pytest tmp dir), so containment in the home is the wrong
# assertion. The property that matters is that it is not a live board: it must
# sit under the same sandbox tree, never under the real home.
db_resolved = Path(path).resolve()
assert db_resolved.is_relative_to(home_resolved.parent), (
    f"database {db_resolved} is outside the sandbox {home_resolved.parent}"
)

conn = kbc.connect(Path(path))
if mode in {"before", "after"}:
    # _execute_boundary_with_retry moved to kanban_db_connect, and write_txn
    # (also there) resolves it in THAT module's globals. Patching it on
    # kanban_db would no longer intercept the COMMIT.
    boundary = kbc._execute_boundary_with_retry

    def interrupted(db, sql):
        if sql == "COMMIT" and mode == "before":
            os._exit(23)
        result = boundary(db, sql)
        if sql == "COMMIT" and mode == "after":
            os._exit(23)
        return result

    kbc._execute_boundary_with_retry = interrupted
claimed = kb.claim_task(conn, task, claimer="private-bearer-token")
print("WON" if claimed is not None else "LOST", flush=True)
conn.close()
