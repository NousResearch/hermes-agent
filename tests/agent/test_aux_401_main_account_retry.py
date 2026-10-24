"""Auxiliary 401 recovery: retry once with the main agent's live credential.

Scenario: the compression auxiliary client gets a 401 "OAuth access token has been revoked"; the refresh
re-reads the same entry and the pool answers "no available entries", while the main agent has already
rotated to another credential -> the summary is lost. Fix: ONE immediate retry with the main agent's live
credential; a persistent failure is logged explicitly and the error propagates (compression aborts without
touching the history).
"""

import logging

import pytest

import agent.auxiliary_client as aux

OLD = "sk-ant-oat01-REVOKED-old-token"
NEW = "sk-ant-oat01-LIVE-main-thread-token"


class _ApiError(Exception):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def _revoked():
    return _ApiError("Error code: 401 - OAuth access token has been revoked", status_code=401)


class _Client:
    api_key = OLD
    base_url = "https://api.anthropic.com"


@pytest.fixture(autouse=True)
def _no_refresh_no_pool(monkeypatch):
    monkeypatch.setattr(aux, "_refresh_provider_credentials", lambda *a, **k: False)
    monkeypatch.setattr(aux, "_recoverable_pool_provider", lambda *a, **k: None)

    def _no_fallback(first_err, route):
        return None
        yield  # pragma: no cover  (generator)

    monkeypatch.setattr(aux, "_ladder_provider_fallback", _no_fallback)


def _ladder(main_runtime):
    return aux._aux_recovery_ladder(
        _revoked(), client=_Client(), kwargs={"model": "claude-sonnet-4-5", "messages": []},
        task="compression", async_mode=False, base_info="https://api.anthropic.com",
        resolved_provider="anthropic", resolved_model="claude-sonnet-4-5", resolved_base_url=None,
        resolved_api_key=None, resolved_api_mode=None, final_model="claude-sonnet-4-5", max_tokens=None,
        main_runtime=main_runtime, route_info={},
    )


def _drive(ladder, outcomes):
    """Run the ladder; ``outcomes`` = list of results/exceptions for successive steps. Returns (value, steps)."""
    steps = []

    def perform(step):
        steps.append(step)
        out = outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    return aux._drive_ladder(ladder, perform), steps


def test_revoked_401_retries_once_with_main_agent_credential():
    value, steps = _drive(_ladder({"provider": "anthropic", "live_api_key": lambda: NEW}), ["SUMMARY"])
    assert value == "SUMMARY"
    assert len(steps) == 1, "exactly ONE retry"
    assert steps[0].kind == "retry_same_provider"
    assert steps[0].args == ("anthropic", "claude-sonnet-4-5", NEW)
    req = type("R", (), {"resolved_provider": "anthropic", "resolved_api_mode": None})()
    kind, _args, call_kwargs = aux._ladder_step_call(
        steps[0], req, {"resolved_api_key": OLD, "resolved_base_url": "x", "main_runtime": {}}, {})
    assert kind == "retry" and call_kwargs["resolved_api_key"] == NEW and call_kwargs["resolved_base_url"] is None


def test_persistent_401_logs_explicitly_and_raises(caplog):
    caplog.set_level(logging.INFO)
    with pytest.raises(_ApiError):
        _drive(_ladder({"provider": "anthropic", "live_api_key": lambda: NEW}), [_revoked()])
    assert "retry with the main agent's credential failed too" in caplog.text
    assert "history is left untouched" in caplog.text


@pytest.mark.parametrize("main_runtime", [
    {"provider": "anthropic", "live_api_key": lambda: OLD},        # main agent on the same dead key
    {"provider": "anthropic"},                                      # no live reader (other callers)
    {"provider": "openai", "live_api_key": lambda: NEW},           # main agent not on Anthropic
    {"provider": "anthropic", "live_api_key": lambda: "not-a-key"},
    {"provider": "anthropic", "live_api_key": lambda: 1 / 0},      # reader error = native behaviour
])
def test_no_extra_retry_when_it_cannot_help(main_runtime):
    with pytest.raises(_ApiError):
        _drive(_ladder(main_runtime), [])  # any step would pop from an empty list → IndexError, not _ApiError


def test_normalize_main_runtime_keeps_live_reader_only_if_callable():
    fn = lambda: NEW
    assert aux._normalize_main_runtime({"provider": "anthropic", "live_api_key": fn})["live_api_key"] is fn
    assert "live_api_key" not in aux._normalize_main_runtime({"provider": "anthropic", "live_api_key": NEW})


def test_agent_binding_reads_live_rotated_key():
    from types import SimpleNamespace

    from agent.auxiliary_main_account import bind_live_main_credential

    class _Agent(SimpleNamespace):
        pass

    agent = _Agent(api_mode="anthropic_messages", _anthropic_api_key=OLD, api_key=OLD,
                   context_compressor=SimpleNamespace())
    bind_live_main_credential(agent)
    reader = agent.context_compressor._live_api_key_fn
    assert reader() == OLD
    agent._anthropic_api_key = NEW  # what _swap_credential does on a pool rotation
    assert reader() == NEW
