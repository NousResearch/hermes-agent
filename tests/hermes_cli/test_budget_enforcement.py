"""Unit tests for the on_budget_check enforcement contract (PR1: contract + soft path)."""

from __future__ import annotations

from hermes_cli.plugins import VALID_HOOKS


def test_on_budget_check_is_a_valid_hook():
    assert "on_budget_check" in VALID_HOOKS
