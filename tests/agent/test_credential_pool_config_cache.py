"""Regression coverage for the ``cached_config_loads`` context manager
introduced in #31556 to keep ``list_authenticated_providers()`` from
re-parsing ``config.yaml`` once per CredentialPool construction."""

from __future__ import annotations

from unittest.mock import patch

from agent.credential_pool import (
    CredentialPool,
    PooledCredential,
    _load_config_safe,
    cached_config_loads,
    get_pool_strategy,
)


def _make_entry(provider: str = "openai") -> PooledCredential:
    return PooledCredential(
        provider=provider,
        id="cred-1",
        label="key-1",
        auth_type="api_key",
        priority=0,
        source="manual",
        access_token="sk-test",
    )


def test_load_config_safe_called_each_time_outside_context():
    call_count = 0

    def _load() -> dict:
        nonlocal call_count
        call_count += 1
        return {"credential_pool_strategies": {"openai": "round_robin"}}

    with patch("hermes_cli.config.load_config", side_effect=_load):
        for _ in range(3):
            _load_config_safe()

    assert call_count == 3


def test_load_config_safe_memoized_within_context():
    call_count = 0

    def _load() -> dict:
        nonlocal call_count
        call_count += 1
        return {"credential_pool_strategies": {"openai": "round_robin"}}

    with patch("hermes_cli.config.load_config", side_effect=_load):
        with cached_config_loads():
            for _ in range(5):
                config = _load_config_safe()
                assert config == {"credential_pool_strategies": {"openai": "round_robin"}}

    assert call_count == 1


def test_cache_cleared_after_context_exits():
    call_count = 0

    def _load() -> dict:
        nonlocal call_count
        call_count += 1
        return {"credential_pool_strategies": {}}

    with patch("hermes_cli.config.load_config", side_effect=_load):
        with cached_config_loads():
            _load_config_safe()
            _load_config_safe()
        # Outside the context the cache is gone and every call re-loads.
        _load_config_safe()
        _load_config_safe()

    assert call_count == 3  # 1 inside the context + 2 outside


def test_nested_contexts_restore_outer_value():
    """The outer context's memoized config must survive an inner context."""

    loads: list[str] = []

    def _load() -> dict:
        loads.append("outer-or-inner")
        return {"_marker": len(loads)}

    with patch("hermes_cli.config.load_config", side_effect=_load):
        with cached_config_loads():
            outer = _load_config_safe()
            assert outer["_marker"] == 1
            with cached_config_loads():
                # Inner context starts fresh — first call inside re-loads.
                inner = _load_config_safe()
                assert inner["_marker"] == 2
                # ...and is then memoized for the rest of the inner scope.
                assert _load_config_safe() is inner
            # Back in the outer context: the original memoized value
            # is restored, no additional load.
            assert _load_config_safe() is outer

    assert len(loads) == 2


def test_failed_load_does_not_poison_cache():
    """A None / raising load_config must not freeze the cache at None."""

    sequence: list = [None, {"credential_pool_strategies": {"openai": "least_used"}}]

    def _load():
        return sequence.pop(0)

    with patch("hermes_cli.config.load_config", side_effect=_load):
        with cached_config_loads():
            assert _load_config_safe() is None  # first attempt returns None
            # Second attempt should retry, and now succeeds + memoizes.
            second = _load_config_safe()
            assert second == {"credential_pool_strategies": {"openai": "least_used"}}
            # Third call hits the memoized value, no further loads.
            assert _load_config_safe() is second

    assert sequence == []  # both staged returns consumed, no third load


def test_credential_pool_construction_loads_config_once_in_context():
    """Regression for #31556: building many CredentialPools in one
    discovery pass must parse config.yaml exactly once."""

    call_count = 0

    def _load() -> dict:
        nonlocal call_count
        call_count += 1
        return {"credential_pool_strategies": {"openai": "round_robin"}}

    with patch("hermes_cli.config.load_config", side_effect=_load):
        with cached_config_loads():
            pools = [
                CredentialPool("openai", [_make_entry("openai")])
                for _ in range(20)
            ]

    assert len(pools) == 20
    assert all(p._strategy == "round_robin" for p in pools)
    assert call_count == 1


def test_credential_pool_construction_outside_context_loads_each_time():
    """Without the context, every CredentialPool reparses config.yaml —
    this is the pre-fix behavior that #31556 documents."""

    call_count = 0

    def _load() -> dict:
        nonlocal call_count
        call_count += 1
        return {"credential_pool_strategies": {}}

    with patch("hermes_cli.config.load_config", side_effect=_load):
        for _ in range(5):
            CredentialPool("openai", [_make_entry("openai")])

    assert call_count == 5


def test_get_pool_strategy_uses_cached_config():
    """``get_pool_strategy`` is the call site that actually compounds in
    the picker — verify it observes the cache."""

    call_count = 0

    def _load() -> dict:
        nonlocal call_count
        call_count += 1
        return {"credential_pool_strategies": {"openai": "random"}}

    with patch("hermes_cli.config.load_config", side_effect=_load):
        with cached_config_loads():
            for _ in range(10):
                assert get_pool_strategy("openai") == "random"

    assert call_count == 1
