"""The store holder and the repair module claim the one repair attempt per ``state.db`` from one set: a claim made
through either is seen by the other.
"""

import os
from pathlib import Path

import pytest

from persistence.sqlite._state.repair import _claim_repair_attempt, _repair_attempted_paths
from persistence.sqlite import _holder


@pytest.fixture
def home():
    return Path(os.environ["HERMES_HOME"])


@pytest.fixture
def path(home):
    return home / "state.db"


def test_the_one_repair_attempt_per_file_is_claimed_once_for_both_sides(path):
    try:
        assert _holder._claim_repair_attempt(path) is True
        assert _claim_repair_attempt(path) is False
    finally:
        _repair_attempted_paths.discard(str(path))
