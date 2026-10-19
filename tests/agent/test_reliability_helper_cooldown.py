"""A broken local helper must not stampede on concurrent request paths."""
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

import agent.command_token_source as source


def test_failed_mint_is_shared_by_waiters_without_reexecuting(monkeypatch):
    mint = Mock(side_effect=source.CommandTokenError("fixture helper failure"))
    monkeypatch.setattr(source, "_mint", mint)
    provider = source.CommandTokenSource("fixture-command", "fixture")

    def request():
        with pytest.raises(source.CommandTokenError):
            provider()

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: request(), range(8)))
    assert mint.call_count == 1


def test_failure_cooldown_expires_and_success_restores_normal_cache(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(source.time, "monotonic", lambda: now[0])
    mint = Mock(side_effect=[source.CommandTokenError("fixture failure"), ("fixture-value", 300)])
    monkeypatch.setattr(source, "_mint", mint)
    provider = source.CommandTokenSource("fixture-command", "fixture")
    with pytest.raises(source.CommandTokenError):
        provider()
    now[0] = 101.0
    with pytest.raises(source.CommandTokenError):
        provider()
    assert mint.call_count == 1
    now[0] = 131.0
    assert provider() == "fixture-value"
    assert provider() == "fixture-value"
    assert mint.call_count == 2


def test_failure_state_never_retains_exception_or_helper_output(monkeypatch):
    marker = "fixture-private-helper-output"
    failure = source.CommandTokenError(marker)
    failure.__cause__ = RuntimeError("fixture-cause-output")
    monkeypatch.setattr(source, "_mint", Mock(side_effect=failure))
    provider = source.CommandTokenSource("fixture-command", "fixture")
    with pytest.raises(source.CommandTokenError):
        provider()
    with pytest.raises(source.CommandTokenError) as cached:
        provider()
    assert marker not in str(cached.value)
    assert cached.value.__cause__ is None
    assert all(not isinstance(value, BaseException) for value in vars(provider).values())


def test_different_sources_do_not_share_failures(monkeypatch):
    mint = Mock(side_effect=[source.CommandTokenError("fixture failure"), ("fixture-value", 300)])
    monkeypatch.setattr(source, "_mint", mint)
    broken = source.CommandTokenSource("fixture-command-a", "fixture-a")
    healthy = source.CommandTokenSource("fixture-command-b", "fixture-b")
    with pytest.raises(source.CommandTokenError):
        broken()
    assert healthy() == "fixture-value"
    assert mint.call_count == 2


def test_expired_cached_value_is_not_reused_after_failed_refresh(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(source.time, "monotonic", lambda: now[0])
    mint = Mock(side_effect=[("fixture-old-value", 120), source.CommandTokenError("fixture failure")])
    monkeypatch.setattr(source, "_mint", mint)
    provider = source.CommandTokenSource("fixture-command", "fixture")
    assert provider() == "fixture-old-value"
    now[0] = 161.0
    with pytest.raises(source.CommandTokenError):
        provider()
    with pytest.raises(source.CommandTokenError):
        provider()
    assert mint.call_count == 2
