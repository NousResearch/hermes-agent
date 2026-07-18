"""Reasoning-effort resolution for LM Studio.

Covers the contract that Hermes' generic effort ladder must stay monotonic
once it is mapped onto LM Studio's narrower vocabulary: a stronger requested
level may resolve to an equal-or-stronger LM Studio level, never a weaker one.
"""

from __future__ import annotations

import pytest

from agent.lmstudio_reasoning import resolve_lmstudio_effort
from hermes_constants import VALID_REASONING_EFFORTS

# Rank of each value LM Studio accepts, weakest to strongest. Used to assert
# the resolved ladder never inverts.
_LM_RANK = {"minimal": 0, "low": 1, "medium": 2, "high": 3, "xhigh": 4}


@pytest.mark.parametrize("effort", ["max", "ultra"])
def test_strong_efforts_clamp_to_lmstudio_ceiling(effort):
    """"max"/"ultra" exceed LM Studio's vocabulary and clamp to its ceiling.

    Without the clamp they miss the valid set, keep the "medium" default and
    resolve *below* "xhigh" -- more requested reasoning yielding less.
    """
    assert resolve_lmstudio_effort({"enabled": True, "effort": effort}, None) == "xhigh"


def test_effort_ladder_is_monotonic():
    """Resolving Hermes' canonical ladder never produces an inversion."""
    resolved = [
        resolve_lmstudio_effort({"enabled": True, "effort": effort}, None)
        for effort in VALID_REASONING_EFFORTS
    ]
    ranks = [_LM_RANK[value] for value in resolved]
    assert ranks == sorted(ranks), dict(zip(VALID_REASONING_EFFORTS, resolved))


@pytest.mark.parametrize(
    "effort,expected",
    [
        ("minimal", "minimal"),
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("xhigh", "xhigh"),
    ],
)
def test_levels_within_lmstudio_vocabulary_are_unchanged(effort, expected):
    """Negative control: the clamp must not disturb levels LM Studio knows."""
    assert resolve_lmstudio_effort({"enabled": True, "effort": effort}, None) == expected


def test_unparseable_effort_still_falls_back_to_medium():
    """Negative control: clamping must not change the unrecognized-input path.

    This is the behaviour "max"/"ultra" were previously conflated with.
    """
    assert resolve_lmstudio_effort({"enabled": True, "effort": "banana"}, None) == "medium"


def test_disabled_reasoning_still_resolves_to_none():
    """Negative control: the clamp sits after the enabled=False short-circuit."""
    assert resolve_lmstudio_effort({"enabled": False, "effort": "max"}, None) == "none"


@pytest.mark.parametrize("effort", ["max", "ultra"])
def test_clamped_effort_is_still_checked_against_allowed_options(effort):
    """A clamped value stays subject to the model's published allowed set.

    "max" resolves to "xhigh"; a model that does not publish "xhigh" gets the
    field omitted (``None``) so LM Studio applies the model's own default --
    exactly how a directly-requested "xhigh" already behaves.
    """
    assert (
        resolve_lmstudio_effort(
            {"enabled": True, "effort": effort}, ["off", "minimal", "low"]
        )
        is None
    )
    assert (
        resolve_lmstudio_effort(
            {"enabled": True, "effort": effort}, ["low", "medium", "high", "xhigh"]
        )
        == "xhigh"
    )


def test_clamp_does_not_rewrite_published_allowed_options():
    """The clamp must not leak into allowed_options normalization.

    A model publishing "max" is not claiming LM Studio's request vocabulary
    accepts it; allowed_options passes through untouched, so a resolved
    "xhigh" that the model does not publish is still omitted.
    """
    assert resolve_lmstudio_effort({"enabled": True, "effort": "max"}, ["max"]) is None
