"""Unit tests for the shared agent isolation contract."""

import pytest


@pytest.mark.parametrize(
    ("ignore_rules", "safe_mode", "env", "expected"),
    [
        # Default path: no flags, no env.
        (None, None, {}, False),
        # Env path: each variable alone enables isolation; safe mode implies
        # ignore-rules at the flag level too.
        (None, None, {"HERMES_IGNORE_RULES": "1"}, True),
        (None, None, {"HERMES_SAFE_MODE": "1"}, True),
        (None, None, {"HERMES_IGNORE_RULES": "true"}, True),
        (None, None, {"HERMES_IGNORE_RULES": "0", "HERMES_SAFE_MODE": "0"}, False),
        # Explicit flags win over the environment.
        (True, None, {}, True),
        (False, None, {"HERMES_IGNORE_RULES": "1"}, False),
        (None, True, {}, True),
        (False, True, {}, True),
        (True, False, {}, True),
        (False, False, {"HERMES_SAFE_MODE": "1"}, False),
    ],
)
def test_resolve_agent_isolation(monkeypatch, ignore_rules, safe_mode, env, expected):
    from agent.isolation import resolve_agent_isolation

    monkeypatch.delenv("HERMES_IGNORE_RULES", raising=False)
    monkeypatch.delenv("HERMES_SAFE_MODE", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    skip_context_files, skip_memory = resolve_agent_isolation(
        ignore_rules=ignore_rules,
        safe_mode=safe_mode,
    )
    assert skip_context_files is expected
    assert skip_memory is expected


def test_resolve_agent_isolation_returns_pair_together(monkeypatch):
    """Both skip flags must always move together."""
    from agent.isolation import resolve_agent_isolation

    monkeypatch.setenv("HERMES_IGNORE_RULES", "1")
    assert resolve_agent_isolation() == (True, True)
    monkeypatch.delenv("HERMES_IGNORE_RULES", raising=False)
    assert resolve_agent_isolation() == (False, False)
