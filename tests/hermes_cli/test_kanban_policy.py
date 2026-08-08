"""Regression tests for the kanban policy resolvers extracted to
``hermes_cli/kanban_policy.py`` (wave-1 godfile decomposition, s1 c2).

Every moved resolver is exercised both through the new module (direct import)
and through the ``hermes_cli.kanban_db`` re-export surface (``kb.*``).
"""

from __future__ import annotations

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_policy as kp


def test_moved_functions_are_re_exported_from_kanban_db():
    for name in (
        "_resolve_claim_ttl_seconds",
        "_resolve_crash_grace_seconds",
        "_resolve_rate_limit_cooldown_seconds",
    ):
        assert getattr(kb, name) is getattr(kp, name), name


def test_claim_ttl_explicit_value_wins(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_TTL_SECONDS", "999")
    assert kp._resolve_claim_ttl_seconds(60) == 60
    assert kp._resolve_claim_ttl_seconds(0) == 1  # clamped to >= 1


def test_claim_ttl_env_and_default(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_CLAIM_TTL_SECONDS", raising=False)
    assert kp._resolve_claim_ttl_seconds() == kb.DEFAULT_CLAIM_TTL_SECONDS
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_TTL_SECONDS", "321")
    assert kp._resolve_claim_ttl_seconds() == 321
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_TTL_SECONDS", "not-an-int")
    assert kp._resolve_claim_ttl_seconds() == kb.DEFAULT_CLAIM_TTL_SECONDS
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_TTL_SECONDS", "-5")
    assert kp._resolve_claim_ttl_seconds() == kb.DEFAULT_CLAIM_TTL_SECONDS


def test_crash_grace_env_and_default(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", raising=False)
    assert kp._resolve_crash_grace_seconds() == kb.DEFAULT_CRASH_GRACE_SECONDS
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    assert kp._resolve_crash_grace_seconds() == 0  # immediate-reclaim mode
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "42")
    assert kp._resolve_crash_grace_seconds() == 42
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "junk")
    assert kp._resolve_crash_grace_seconds() == kb.DEFAULT_CRASH_GRACE_SECONDS


def test_rate_limit_cooldown_env_and_default(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", raising=False)
    assert kp._resolve_rate_limit_cooldown_seconds() == kb.DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "0")
    assert kp._resolve_rate_limit_cooldown_seconds() == 0  # re-spawn next tick
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "77")
    assert kp._resolve_rate_limit_cooldown_seconds() == 77
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "-1")
    assert kp._resolve_rate_limit_cooldown_seconds() == kb.DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS
