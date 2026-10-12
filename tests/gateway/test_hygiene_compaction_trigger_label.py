"""Each automatic compression entry point names itself in the attempt record.

Gateway session hygiene compacts through ``agent._compress_context(..., trigger=...)`` from two
places: the live codex app-server agent and the detached hygiene agent. Both must label the
attempt ``gateway_hygiene`` so the content-free attempt telemetry does not report it as plain
``auto``. These tests drive the real hygiene callers with an agent double that records the
label it received.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from gateway.run import GatewayRunner, run_codex_hygiene_compaction


def _recording_agent(labels: list) -> SimpleNamespace:
    def _record(messages, system_message, **kwargs):
        labels.append(kwargs.get("trigger"))
        return messages, system_message

    return SimpleNamespace(context_compressor=None, _codex_session=object(), _compress_context=_record)


def _history() -> list:
    return [{"role": "user", "content": f"m{i}"} for i in range(4)]


async def _drive_codex_hygiene(agent: SimpleNamespace) -> None:
    gateway = SimpleNamespace(_agent_cache={"tg:1": (agent, 0.0)}, _agent_cache_lock=None)
    await run_codex_hygiene_compaction(
        gateway, "tg:1", "sess-1", auto_mode="hermes", history=_history(),
        approx_tokens=100_000, timeout_seconds=5.0,
    )


async def _drive_detached_hygiene(agent: SimpleNamespace) -> None:
    async def _build_agent(*_args):
        return agent, None

    async def _wait_for_summary(attempt, *_args):
        return await attempt.future

    async def _apply_result(*_args, **_kwargs):
        return None

    async def _cleanup(*_args, **_kwargs):
        return None

    gateway = SimpleNamespace(
        _hmwa_hygiene_build_agent=_build_agent, _hmwa_hygiene_wait_for_summary=_wait_for_summary,
        _hmwa_hygiene_apply_result=_apply_result, _track_deferred_agent_worker=lambda *_a: None,
        _evict_cached_agent=lambda _key: None, _cleanup_agent_resources_off_loop=_cleanup,
    )
    attempt = GatewayRunner._HygieneAttempt(agent=None, meta=None)
    await GatewayRunner._hmwa_hygiene_detached_attempt(
        gateway, attempt, SimpleNamespace(total_ceiling_seconds=60.0),
        SimpleNamespace(approx_tokens=100_000), _history(), _history(), "m", {}, None,
        SimpleNamespace(session_id="sess-1"), "tg:1", "qk", 0,
    )


@pytest.mark.parametrize(
    "drive",
    [
        pytest.param(_drive_codex_hygiene, id="codex_live_agent"),
        pytest.param(_drive_detached_hygiene, id="detached_agent"),
    ],
)
def test_gateway_hygiene_caller_passes_its_trigger_label(drive):
    labels: list = []

    asyncio.run(drive(_recording_agent(labels)))

    assert labels == ["gateway_hygiene"]
