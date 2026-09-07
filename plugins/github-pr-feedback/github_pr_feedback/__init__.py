"""Hermes plugin helpers for governed GitHub pull-request feedback intake."""

from .completion_guard import register_completion_guard
from .repair_completion_policy import register_repair_completion_policy


def register(ctx) -> None:
    """Register governed GitHub PR feedback completion policies."""
    register_completion_guard(ctx)
    register_repair_completion_policy(ctx)
