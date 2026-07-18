"""Regression coverage for the test suite's process-state isolation."""

import os
from pathlib import Path

import hermes_state


def test_session_db_default_is_scoped_to_per_test_hermes_home():
    """Collection-time imports must not retain the developer's live DB path."""
    expected = Path(os.environ["HERMES_HOME"]) / "state.db"

    # Do not construct SessionDB here: if this isolation regresses, construction
    # would open and initialize the developer's live database before failing.
    assert hermes_state.DEFAULT_DB_PATH == expected
