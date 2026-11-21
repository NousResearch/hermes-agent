"""``batch_runner --list_distributions`` — the documented flag must list, not crash.

``main()`` declares a ``list_distributions: bool`` parameter that shadows the module-level
``list_distributions`` import, so the flag branch used to call the bool and die with
``TypeError: 'bool' object is not callable`` before the handled-error try block.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import batch_runner  # noqa: E402


def test_list_distributions_flag_prints_the_catalog_and_returns(capsys):
    """The flag lists the available distributions and returns instead of raising."""
    batch_runner.main(list_distributions=True)

    out = capsys.readouterr().out
    assert "Available Toolset Distributions" in out
    assert "Usage:" in out


def test_list_distributions_prints_every_distribution_name(capsys):
    """Every catalog entry is printed — a silent empty listing would pass the smoke test above."""
    from toolset_distributions import list_distributions

    batch_runner.main(list_distributions=True)

    out = capsys.readouterr().out
    for name in list_distributions():
        assert name in out
