"""Regression tests for #111988 — fail-closed in-context bound on hygiene misses.

When the hygiene turn-hold budget expires without a committed summary,
``_hmwa_hygiene_on_turn_hold`` logs "proceeding without compression this turn"
and the turn ran on the FULL uncompressed transcript: no deterministic cap on
the model payload. The fix bounds the in-context transcript whenever hygiene
hasn't landed by turn start — leading system/setup rows + newest tail, total <=
``hygiene_hard_message_limit``. History on disk is untouched; availability is
unchanged.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.gateway.test_session_hygiene_turnhold_adoption import (
    _build_runner,
    _install_fakes,
    _make_event,
)


def _write_failclosed_config(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "compression:\n"
        "  enabled: true\n"
        "  hygiene_hard_message_limit: 50\n"
    )


def _long_history():
    head = [
        {"role": "session_meta", "content": "meta"},
        {"role": "system", "content": "setup-1"},
        {"role": "system", "content": "setup-2"},
    ]
    body = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}", "timestamp": f"t{i}"}
        for i in range(97)
    ]
    return head + body  # 100 rows


@pytest.mark.asyncio
async def test_turn_hold_miss_applies_failclosed_bound(monkeypatch, tmp_path):
    """A turn-hold release with no committed summary must not run on the full
    transcript: the in-context history is bounded to setup head + newest tail,
    total <= hygiene_hard_message_limit, and the input list is not mutated."""
    gateway_run = importlib.import_module("gateway.run")
    _write_failclosed_config(tmp_path)
    _install_fakes(monkeypatch, gateway_run, tmp_path, agent_cls=type("DummyAgent", (), {}))
    runner = _build_runner(gateway_run, MagicMock(), MagicMock())

    plan = SimpleNamespace(
        needs_compress=True, msg_count=100, approx_tokens=10**9, warn_token_threshold=10**9,
    )
    monkeypatch.setattr(runner, "_hmwa_hygiene_plan", AsyncMock(return_value=plan))

    async def _raise_turn_hold(*_args, **_kwargs):
        raise gateway_run.HygieneTurnHoldExceeded("turn-hold budget expired")

    monkeypatch.setattr(runner, "_hmwa_hygiene_detached_attempt", _raise_turn_hold)

    history = _long_history()
    event = _make_event()
    session_entry = runner.session_store.get_or_create_session.return_value
    out = await runner._hmwa_run_session_hygiene(
        event, event.source, session_entry, session_entry.session_key, history, "q", 1,
    )

    assert len(out) == 50
    assert [m["role"] for m in out[:3]] == ["session_meta", "system", "system"]
    assert out[3:] == history[-47:]
    assert out is not history
    assert len(history) == 100, "the on-disk transcript must not be mutated"


@pytest.mark.asyncio
async def test_failclosed_bound_pure_contract():
    """The bound itself: passthrough identity when it fits, setup head + newest
    tail when it doesn't, degenerate clamp, nonpositive limit."""
    gateway_run = importlib.import_module("gateway.run")
    bound = gateway_run.GatewayRunner._hmwa_hygiene_failclosed_bound

    history = _long_history()
    assert bound(history, 50) is not history
    assert len(bound(history, 50)) == 50
    assert bound(history, 1000) is history

    all_setup = [{"role": "system", "content": f"s{i}"} for i in range(10)]
    assert len(bound(all_setup, 5)) == 5

    assert len(bound(history, 0)) <= 1
