from __future__ import annotations

import asyncio
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest


class _ComposedGatewayRunner:
    def __init__(self, store, async_store):
        from gateway.run import GatewayRunner

        self.session_store = store
        self.async_session_store = async_store
        self._async_session_store = async_store
        self._contextual_transcript_apply_lock = None
        self._contextual_transcript_applied_prefix = (
            GatewayRunner._contextual_transcript_applied_prefix
        )
        self._apply_contextual_cron_transcript_sync = MethodType(
            GatewayRunner._apply_contextual_cron_transcript_sync, self
        )
        self._apply_contextual_cron_transcript_async_fake = MethodType(
            GatewayRunner._apply_contextual_cron_transcript_async_fake, self
        )
        self._apply_contextual_cron_transcript = MethodType(
            GatewayRunner._apply_contextual_cron_transcript, self
        )

    def _is_user_authorized(self, _source) -> bool:
        return True

    def _session_key_for_source(self, source) -> str:
        from gateway.session import build_session_key

        return build_session_key(source)

    def _contextual_cron_session_busy(self, _session_key: str) -> bool:
        return False

    async def _run_contextual_cron_turn(self, item, entry, history):
        from gateway.contextual_cron import ContextualCronOutcome

        chosen = next(
            message["content"].split("chosen word is ", 1)[1].rstrip(".")
            for message in history
            if message.get("role") == "user"
            and "chosen word is " in str(message.get("content"))
        )
        item.transcript_session_id = entry.session_id
        assert isinstance(item.transcript_base_message_count, int)
        assert isinstance(item.transcript_base_revision, int)
        item.last_prompt_tokens = 23
        item.transcript_entries = [
            {
                "role": "user",
                "content": item.prompt,
                "display_kind": "hidden",
                "message_id": f"contextual-cron:{item.execution_id}:0",
            },
            {
                "role": "assistant",
                "content": f"Recovered chosen word: {chosen}",
                "message_id": f"contextual-cron:{item.execution_id}:1",
            },
        ]
        return ContextualCronOutcome.notify(f"Recovered chosen word: {chosen}")


@pytest.mark.asyncio
async def test_builtin_current_session_composes_scheduler_gateway_transcript_and_single_delivery(
    monkeypatch, tmp_path
):
    """Real V1 composition: built-in run -> live lane -> outbox -> one delivery."""
    import cron.executions as executions
    import cron.jobs as jobs
    import cron.scheduler as scheduler
    from gateway.config import GatewayConfig, Platform as ConfigPlatform
    from gateway.contextual_cron import ContextualCronGateway
    from gateway.session import AsyncSessionStore, Platform, SessionSource, SessionStore
    from gateway.session_context import clear_session_vars, set_session_vars

    profile_home = tmp_path / "profile"
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    cron_dir = profile_home / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", cron_dir / "executions.db")
    monkeypatch.setattr(
        "hermes_cli.config.load_config", lambda: {"cron": {"provider": "builtin"}}
    )

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="42",
        chat_type="dm",
        user_id="42",
    )
    store = SessionStore(profile_home / "sessions", GatewayConfig())
    entry = store.get_or_create_session(source, force_new=True)
    store.append_to_transcript(
        entry.session_id,
        {"role": "user", "content": "The chosen word is marzipan."},
        skip_db=False,
    )
    session_key = entry.session_key
    session_id = entry.session_id
    tokens = set_session_vars(
        platform="telegram",
        chat_id="42",
        chat_type="dm",
        user_id="42",
        session_key=session_key,
        session_id=session_id,
    )
    try:
        job = jobs.create_job(
            prompt="Continue the conversation and report the chosen word.",
            schedule="every 1h",
            deliver="origin",
            session_target="current",
        )
    finally:
        clear_session_vars(tokens)

    async_store = AsyncSessionStore(store)
    lane = ContextualCronGateway(
        _ComposedGatewayRunner(store, async_store), busy_poll_seconds=0.001
    )
    loop = asyncio.get_running_loop()

    platform_cfg = SimpleNamespace(enabled=True)
    monkeypatch.setattr(
        "gateway.config.load_gateway_config",
        lambda: SimpleNamespace(platforms={ConfigPlatform.TELEGRAM: platform_cfg}),
    )
    send = AsyncMock(return_value={"success": True})
    monkeypatch.setattr("tools.send_message_tool._send_to_platform", send)
    monkeypatch.setattr(
        scheduler,
        "_CONTEXTUAL_AUTHORIZER",
        lambda target: target.get("origin", {}).get("user_id") == "42",
    )

    def dispatch(bound_job, *, execution_id):
        return lane.dispatch_from_scheduler(
            bound_job,
            execution_id=execution_id,
            loop=loop,
            timeout=5,
        )

    assert await asyncio.to_thread(
        scheduler.run_one_job,
        job,
        loop=loop,
        contextual_dispatch=dispatch,
    )

    record = executions.latest_execution(job["id"])
    assert record is not None
    assert record["status"] == "completed"
    assert record["outcome"] == "notify"
    assert record["delivery_state"] == "sent"
    assert record["transcript_state"] == "applied"
    history = store.load_transcript_strict(entry.session_id)
    assert [item["message_id"] for item in history[-2:]] == [
        f"contextual-cron:{record['id']}:0",
        f"contextual-cron:{record['id']}:1",
    ]
    assert history[-2]["display_kind"] == "hidden"
    assert "marzipan" in history[-1]["content"]
    refreshed_entry = store.peek_session_entry(entry.session_key)
    assert refreshed_entry is not None
    assert refreshed_entry.last_prompt_tokens == 23
    assert send.await_count == 1
    assert "marzipan" in str(send.await_args)

    replay = dict(job, execution_id=record["id"])
    assert await asyncio.to_thread(
        scheduler.run_one_job,
        replay,
        loop=loop,
        contextual_dispatch=dispatch,
    )
    assert send.await_count == 1
    assert len(store.load_transcript_strict(entry.session_id)) == 3
