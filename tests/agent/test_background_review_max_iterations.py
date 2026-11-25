"""``auxiliary.background_review.max_iterations`` — the review fork's iteration ceiling.

The fork used to run with the hard-coded ``_REVIEW_MAX_ITERATIONS`` (16) only, which truncated an
honest review of a large SKILL.md ("Reached maximum iterations (16)") before it finished reading.
The override mirrors ``max_input_tokens``: an explicit 1..64 wins, anything unset or malformed
keeps the historical constant.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent.background_review import (
    _REVIEW_MAX_ITERATIONS,
    _review_max_iterations,
)


@pytest.mark.parametrize(
    ("task_cfg", "expected"),
    [
        ({}, _REVIEW_MAX_ITERATIONS),
        ({"max_iterations": 1}, 1),
        ({"max_iterations": 40}, 40),
        ({"max_iterations": 64}, 64),
        ({"max_iterations": 0}, _REVIEW_MAX_ITERATIONS),
        ({"max_iterations": 65}, _REVIEW_MAX_ITERATIONS),
        ({"max_iterations": -3}, _REVIEW_MAX_ITERATIONS),
        ({"max_iterations": "40"}, _REVIEW_MAX_ITERATIONS),
        ({"max_iterations": 40.5}, _REVIEW_MAX_ITERATIONS),
        ({"max_iterations": True}, _REVIEW_MAX_ITERATIONS),
        ({"max_iterations": None}, _REVIEW_MAX_ITERATIONS),
        ({"max_iterations": [40]}, _REVIEW_MAX_ITERATIONS),
    ],
)
def test_review_max_iterations_resolution(task_cfg, expected):
    """Only a real integer in 1..64 overrides the ceiling; every other shape falls back."""
    assert _review_max_iterations(task_cfg) == expected


def test_review_max_iterations_survives_a_non_dict_config():
    """A malformed ``auxiliary.background_review`` never breaks the fork spawn."""
    bad_cfg: Any = "not-a-dict"
    assert _review_max_iterations(bad_cfg) == _REVIEW_MAX_ITERATIONS


def test_background_review_config_does_not_freeze_a_fixed_iteration_ceiling():
    """The default config must leave the resolver deciding, like ``max_input_tokens``."""
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert "max_iterations" not in DEFAULT_CONFIG["auxiliary"]["background_review"]
