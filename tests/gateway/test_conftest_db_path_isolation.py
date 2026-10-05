"""The test-isolation fixture must not defeat runtime HERMES_HOME redirection.

``tests/conftest.py`` re-pins ``hermes_state.DEFAULT_DB_PATH`` at the start of
every test so an argless ``SessionDB()`` cannot open the developer's real
``~/.hermes/state.db``. That protection is worth keeping.

But ``hermes_state._default_db_path()`` reads:

    if DEFAULT_DB_PATH != _IMPORT_DEFAULT_DB_PATH:
        return DEFAULT_DB_PATH          # "someone re-pointed it deliberately"
    return get_hermes_home() / "state.db"

so re-pinning ONLY the constant trips the first branch and nails every argless
``SessionDB()`` to the fixture's path for the rest of the test. A test can then
no longer redirect ``HERMES_HOME`` at runtime — which is exactly what
``test_session_store_default_db_uses_runtime_hermes_home`` exists to verify.

The re-pin is conditional on ``hermes_state`` already being in ``sys.modules``,
so the symptom is order-shaped without being an ordering bug: the test passes
ALONE (nothing imported it yet, the re-pin is skipped, live resolution runs)
and fails inside the suite (an earlier file imported it at collection, the
re-pin fires). It failed in file order and shuffled alike, on upstream main and
on feature branches — 1 failed / 767 passed, unchanged for as long as both
pieces have coexisted.

Fix: move ``_IMPORT_DEFAULT_DB_PATH`` with it, so the two stay EQUAL and
``_default_db_path()`` falls through to ``get_hermes_home()``. Same answer by
default — the fixture points HERMES_HOME at the same directory — but a test
that re-points HERMES_HOME is now followed.
"""
import os
import sys
from pathlib import Path

import pytest

# Imported at MODULE level on purpose, not inside the tests.
#
# The conftest re-pin is guarded by `sys.modules.get("hermes_state") is not
# None`, so it only fires when something imported the module before the fixture
# ran. A top-level import here happens at COLLECTION, which guarantees that
# condition for this file — otherwise these tests would pass vacuously when run
# alone (the re-pin skipped, nothing to defeat) and only mean anything inside a
# suite that happened to import it first. That order-dependence is the very
# thing under test; it must not also decide whether the test is meaningful.
import hermes_state  # noqa: E402


def test_hermes_state_is_imported_so_the_repin_actually_fires():
    """Guards the premise: without the module the fixture skips the re-pin and
    every assertion below would pass vacuously."""
    assert "hermes_state" in sys.modules, (
        "hermes_state not imported — this file cannot exercise the re-pin"
    )


def test_the_two_pins_are_kept_in_sync():
    """The fix itself. Unequal values re-arm the short-circuit."""
    assert hermes_state.DEFAULT_DB_PATH == hermes_state._IMPORT_DEFAULT_DB_PATH, (
        "conftest re-pinned DEFAULT_DB_PATH without moving "
        "_IMPORT_DEFAULT_DB_PATH; _default_db_path() will now short-circuit and "
        "ignore runtime HERMES_HOME"
    )


def test_default_db_path_still_points_inside_the_fake_home():
    """The safety property the re-pin exists for must survive the fix."""
    resolved = Path(hermes_state._default_db_path())
    home = Path(os.environ["HERMES_HOME"])
    assert resolved == home / "state.db"
    assert str(resolved).startswith(str(home)), (
        f"{resolved} escapes the per-test home {home} — an argless SessionDB() "
        "could reach the developer's real state.db"
    )


def test_a_test_can_redirect_hermes_home_at_runtime(tmp_path, monkeypatch):
    """THE regression. This is what the short-circuit made impossible."""
    alt = tmp_path / "alt_home"
    alt.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(alt))

    assert Path(hermes_state._default_db_path()) == alt / "state.db", (
        "_default_db_path() ignored a runtime HERMES_HOME change — the "
        "conftest re-pin is short-circuiting live resolution again"
    )


def test_redirection_survives_repeated_changes(tmp_path, monkeypatch):
    """Live resolution, not a one-shot re-read cached on first use."""
    for name in ("one", "two", "three"):
        d = tmp_path / name
        d.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(d))
        assert Path(hermes_state._default_db_path()) == d / "state.db"


def test_a_deliberate_repoint_is_still_honoured(monkeypatch, tmp_path):
    """Do not over-correct: the short-circuit is a real feature.

    Production code that assigns DEFAULT_DB_PATH on purpose must still win
    over the environment — only the fixture's bookkeeping was the problem.
    """
    deliberate = tmp_path / "explicit" / "state.db"
    deliberate.parent.mkdir()
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", deliberate)
    # _IMPORT_DEFAULT_DB_PATH deliberately NOT moved here — that asymmetry is
    # the signal "a caller re-pointed this on purpose".
    assert Path(hermes_state._default_db_path()) == deliberate


@pytest.mark.parametrize("attr", ["DEFAULT_DB_PATH", "_IMPORT_DEFAULT_DB_PATH"])
def test_both_attributes_still_exist(attr):
    """The fixture guards each with hasattr; a rename would silently skip it."""
    assert hasattr(hermes_state, attr), (
        f"hermes_state.{attr} is gone — tests/conftest.py guards the re-pin "
        f"with hasattr, so the isolation would silently stop applying"
    )
