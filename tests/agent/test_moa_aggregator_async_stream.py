"""Regression #130132: production MoA teardown closes unread native streams."""

import asyncio
import threading
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

from agent import auxiliary_client, relay_runtime
from agent import chat_completion_helpers as helpers
from run_agent import AIAgent


class _AsyncStream:
    def __init__(self) -> None:
        self.owner = asyncio.get_running_loop()
        self.reads = 0
        self.closed = False

    def __aiter__(self) -> "_AsyncStream":
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        assert asyncio.get_running_loop() is self.owner
        if self.reads == 3:
            raise StopAsyncIteration
        self.reads += 1
        return ChatCompletionChunk(
            id="chunk",
            object="chat.completion.chunk",
            created=0,
            model="native-actor",
            choices=[
                Choice(index=0, delta=ChoiceDelta(content="chunk"), finish_reason=None)
            ],
        )

    async def aclose(self) -> None:
        assert asyncio.get_running_loop() is self.owner
        self.closed = True


@pytest.mark.parametrize("chunks_to_consume", [0, 1])
def test_streaming_call_closes_abandoned_aggregator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, chunks_to_consume: int
) -> None:
    def abandon(stream: Iterator[object], **_kwargs: object) -> Iterator[object]:
        for _ in range(chunks_to_consume):
            delivered.append(next(stream))
            yield delivered[-1]
        raise KeyboardInterrupt("abandoned aggregator")

    monkeypatch.setattr(helpers, "_iter_provider_stream_chunks", abandon)
    # Codex bypasses the inner Relay seam; the outer facade must adapt that result.
    for managed, codex_bypass in ((True, False), (False, False), (True, True)):
        delivered: list[object] = []
        opened: list[_AsyncStream] = []
        home = tmp_path / f"{managed}-{codex_bypass}"
        home.mkdir()
        # Exercise the raw Relay failure too, without a semaphore wrapper hiding it.
        limit = 0 if codex_bypass or (managed and chunks_to_consume) else 1
        (home / "config.yaml").write_text(
            "providers:\n  native-test:\n    base_url: http://127.0.0.1:1/v1\n"
            "    api_key: test-key\n"
            f"auxiliary:\n  moa_aggregator:\n    max_concurrency: {limit}\n"
            "moa:\n  presets:\n    cleanup:\n      enabled: false\n"
            "      aggregator:\n        provider: native-test\n        model: native-actor\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(home))

        async def create(**_kwargs: object) -> _AsyncStream:
            source = _AsyncStream()
            opened.append(source)
            return source

        client = (
            auxiliary_client.CodexAuxiliaryClient.__new__(
                auxiliary_client.CodexAuxiliaryClient
            )
            if codex_bypass
            else SimpleNamespace()
        )
        monkeypatch.setattr(
            client,
            "chat",
            SimpleNamespace(completions=SimpleNamespace(create=create)),
            raising=False,
        )
        monkeypatch.setattr(
            auxiliary_client,
            "_get_cached_client",
            lambda *a, **k: (client, "native-actor"),
        )
        agent = AIAgent(
            provider="moa",
            model="cleanup",
            api_key="test-key",
            base_url="moa://local",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            enabled_toolsets=[],
            max_iterations=1,
            session_id="moa-cleanup",
        )
        relay_runtime._reset_for_tests()
        lease = turn = None
        consumer = "test.moa-aggregator-cleanup"
        if managed:
            lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
                profile_key=relay_runtime.current_profile_key(),
                session_id="moa-cleanup",
                platform="cli",
            )
            turn = relay_runtime.SESSION_COORDINATOR.begin_turn(
                lease, turn_id="turn", task_id="task"
            )
            lease.host.retain_managed_execution(consumer)
        try:
            workers_before = set(threading.enumerate())
            call = helpers._StreamingCall(
                agent,
                {"model": "cleanup", "messages": [{"role": "user", "content": "hi"}]},
                None,
            )
            # Only the production consumer closes the stream, not this test.
            with pytest.raises(KeyboardInterrupt, match="abandoned aggregator"):
                call._call()
            assert len(opened) == 1
            assert len(delivered) == chunks_to_consume
            if not managed:  # Relay may read ahead; the direct path must stay lazy.
                assert opened[0].reads == chunks_to_consume
            assert opened[0].closed
            assert opened[0].owner.is_closed()
            assert not [
                t
                for t in set(threading.enumerate()) - workers_before
                if t.name == "moa-aggregator-async-stream"
            ]
            if limit:
                semaphore = auxiliary_client._acquire_sync_aux_semaphore(
                    "moa_aggregator"
                )
                assert semaphore is not None
                assert semaphore.acquire(blocking=False), (
                    "abandonment leaked the concurrency permit"
                )
                semaphore.release()
        finally:
            if lease is not None and turn is not None:
                lease.host.release_managed_execution(consumer)
                relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="cancelled")
                relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
            relay_runtime._reset_for_tests()
