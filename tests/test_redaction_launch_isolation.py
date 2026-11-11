"""The hermetic fixture must re-pin agent.redact's import-time snapshot (#138466).

``agent.redact._REDACT_ENABLED`` is evaluated at import, and test modules import it at
collection — before ``_hermetic_environment`` step 2 can delete a launch shell's
``HERMES_REDACT_SECRETS=false``. A worker shell (``chat -q`` of a profile with
``security.redact_secrets: false``) exports the variable, so a pytest launched from it
inherits redaction off for the whole run and redaction tests fail locally while CI
(launch snapshot already ``True``) stays green. Step 3d of the fixture re-pins the
secure default per test, the way 3b re-pins ``hermes_state.DEFAULT_DB_PATH``.
"""
from __future__ import annotations

import agent.redact as redact_mod

# Simulate the poisoned launch snapshot: HERMES_REDACT_SECRETS=false was read into
# _REDACT_ENABLED at import time. Set at module scope so it happens before the
# autouse fixture runs, exactly like the real collection-time import does.
redact_mod._REDACT_ENABLED = False


def test_hermetic_environment_repins_import_time_redaction_snapshot():
    # Without step 3d this stays False: the no-override branch of
    # _redact_enabled() returns the launch snapshot unchanged.
    assert redact_mod._REDACT_ENABLED is True
