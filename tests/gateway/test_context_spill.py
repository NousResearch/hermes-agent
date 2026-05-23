import json
import os
import stat
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from typing import Any

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.context_spill import (
    ContextSpillConfig,
    ContextSpillWriteError,
    decide_context_spill,
    load_context_spill_config,
    request_pressure_from_api_kwargs,
    write_context_spill,
)
from gateway.session import SessionEntry, SessionSource, SessionStore, build_session_key


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="chat-1",
        thread_id="thread-1",
        user_id="user-1",
        chat_type="group",
        chat_name="Test Chat",
    )


def _entry(source=None, session_id="old-session") -> SessionEntry:
    source = source or _source()
    return SessionEntry(
        session_key=build_session_key(source),
        session_id=session_id,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=source.platform,
        chat_type=source.chat_type,
    )


def _config(tmp_path: Path, **overrides) -> ContextSpillConfig:
    values: dict[str, Any] = dict(
        wiki_dir=tmp_path / "wiki" / "outputs" / "gateway-context-spills",
        raw_state_dir=tmp_path / "state" / "context-spills",
        allow_unsafe_paths=True,
    )
    values.update(overrides)
    return ContextSpillConfig(**values)


def test_decision_trips_on_message_count(tmp_path):
    cfg = _config(tmp_path, hard_message_limit=180, hard_token_limit=999999)
    history = [{"role": "user", "content": "x"} for _ in range(181)]
    decision = decide_context_spill(history, "", context_length=256000, config=cfg, stage="history")
    assert decision.should_spill is True
    assert "message_limit" in decision.reason


def test_decision_trips_on_token_threshold(tmp_path):
    cfg = _config(tmp_path, hard_message_limit=999, token_threshold_ratio=0.10, hard_token_limit=20_000)
    history = [{"role": "user", "content": "x" * 20_000} for _ in range(20)]
    decision = decide_context_spill(history, "", context_length=256000, config=cfg, stage="history")
    assert decision.should_spill is True
    assert "token_limit" in decision.reason


def test_decision_trips_on_current_backfill_size(tmp_path):
    cfg = _config(tmp_path, hard_char_limit=50_000, hard_message_limit=999, hard_token_limit=999999)
    message = "[Recent channel messages]\n" + ("large backfill\n" * 5000)
    decision = decide_context_spill([], message, context_length=256000, config=cfg, stage="inbound")
    assert decision.should_spill is True
    assert "char_limit" in decision.reason or "token_limit" in decision.reason


def test_write_context_spill_writes_redacted_wiki_raw_manifest_permissions(tmp_path):
    cfg = _config(tmp_path)
    source = _source()
    entry = _entry(source)
    history = [
        {"role": "user", "content": "OPENAI_API_KEY=sk-testsecret000000000000000000000000"},
        {"role": "assistant", "content": "Authorization: Bearer ghp_secretsecretsecretsecretsecret"},
    ]
    decision = decide_context_spill(history, "current ask", context_length=256000, config=cfg, stage="history")
    result = write_context_spill(
        history=history,
        message_text="current ask",
        source=source,
        session_entry=entry,
        decision=decision,
        config=cfg,
    )

    assert result.wiki_path and result.wiki_path.exists()
    assert result.raw_path and result.raw_path.exists()
    assert result.manifest_path and result.manifest_path.exists()
    wiki_text = result.wiki_path.read_text()
    assert "sk-testsecret000000000000000000000000" not in wiki_text
    assert "ghp_secretsecretsecretsecretsecret" not in wiki_text
    raw_payload = json.loads(result.raw_path.read_text())
    assert raw_payload["history"][0]["content"].startswith("OPENAI_API_KEY=sk-test")
    if os.name != "nt":
        assert stat.S_IMODE(result.raw_path.stat().st_mode) == 0o600
        assert stat.S_IMODE((tmp_path / "state" / "context-spills").stat().st_mode) == 0o700
    assert len(result.recovery_message) < 4000
    assert "current ask" in result.recovery_message
    assert "OPENAI_API_KEY" not in result.recovery_message
    assert len(result.user_notice) <= 500


def test_write_context_spill_redaction_fail_closed_with_raw(monkeypatch, tmp_path):
    cfg = _config(tmp_path)
    source = _source()
    entry = _entry(source)
    decision = decide_context_spill([], "ask", context_length=1, config=cfg, stage="inbound")

    def boom(_text, *, force=False, code_file=False):
        raise RuntimeError("redaction down")

    monkeypatch.setattr("agent.redact.redact_sensitive_text", boom)
    result = write_context_spill(
        history=[{"role": "user", "content": "secret"}],
        message_text="ask",
        source=source,
        session_entry=entry,
        decision=decision,
        config=cfg,
    )
    assert result.raw_path and result.raw_path.exists()
    assert result.wiki_path is None
    assert "raw local bundle" in result.recovery_message


def test_raw_unavailable_fails_closed(monkeypatch, tmp_path):
    cfg = _config(tmp_path)
    source = _source()
    entry = _entry(source)
    decision = decide_context_spill([], "ask", context_length=1, config=cfg, stage="inbound")

    def fail_json(*args, **kwargs):
        raise PermissionError("no write")

    monkeypatch.setattr("gateway.context_spill._atomic_write_json", fail_json)
    with pytest.raises(ContextSpillWriteError):
        write_context_spill(
            history=[],
            message_text="ask",
            source=source,
            session_entry=entry,
            decision=decision,
            config=cfg,
        )


def test_session_store_spill_and_reset_quarantines_old_session(tmp_path):
    source = _source()
    gw_cfg = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")})
    gw_cfg.sessions_dir = tmp_path / "sessions"
    store = SessionStore(gw_cfg.sessions_dir, gw_cfg)
    store._db = None
    entry = store.get_or_create_session(source)
    old_sid = entry.session_id
    raw_path = tmp_path / "state" / "context-spills" / "spill.json"
    new_entry = store.spill_and_reset_session(
        entry.session_key,
        old_session_id=old_sid,
        spill_id="spill-1",
        wiki_path=str(tmp_path / "wiki" / "spill.md"),
        raw_path=str(raw_path),
        reason="message_limit",
    )
    assert new_entry is not None
    assert new_entry.session_id != old_sid
    assert store.is_session_quarantined(old_sid) is True
    assert store.switch_session(entry.session_key, old_sid) is None
    assert store.mark_resume_pending(entry.session_key) is True  # new lane is safe


def test_quarantined_active_entry_beats_resume_pending(tmp_path):
    source = _source()
    gw_cfg = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")})
    gw_cfg.sessions_dir = tmp_path / "sessions"
    store = SessionStore(gw_cfg.sessions_dir, gw_cfg)
    store._db = None
    entry = store.get_or_create_session(source)
    entry.quarantined = True
    entry.resume_pending = True
    store._save()
    fresh = store.get_or_create_session(source)
    assert fresh.session_id != entry.session_id
    assert fresh.auto_reset_reason == "quarantined"


def test_request_pressure_guard_blocks_large_payload():
    api_kwargs = {
        "model": "small",
        "messages": [{"role": "user", "content": "x" * 240_000}],
        "tools": [{"type": "function", "function": {"name": "t", "parameters": {"description": "y" * 120_000}}}],
    }
    pressure = request_pressure_from_api_kwargs(api_kwargs, context_length=64_000)
    assert pressure["too_large"] is True


def test_request_pressure_guard_uses_large_model_context_not_180k_cap():
    # Regression for Codex-routed GPT-5.5: ~183k estimated tokens is below
    # 90% of a 272k context window, so the final pre-API guard must not call it
    # "too large" just because the old gateway spill default was 180k.
    api_kwargs = {
        "model": "gpt-5.5",
        "messages": [{"role": "user", "content": "x" * 734_000}],
        "tools": [],
    }
    pressure = request_pressure_from_api_kwargs(api_kwargs, context_length=272_000)
    assert 180_000 <= pressure["approx_tokens"] < pressure["threshold_tokens"]
    assert pressure["threshold_tokens"] == 244_800
    assert pressure["too_large"] is False


def test_request_guard_failure_copy_does_not_claim_model_context_exceeded():
    from gateway.run import _normalize_empty_agent_response

    response = _normalize_empty_agent_response(
        {
            "failed": True,
            "partial": True,
            "error": "Hermes pre-API request-size guard blocked an oversized internal payload",
            "request_size_guard": {"approx_tokens": 183_599, "threshold_tokens": 244_800},
        },
        "",
        history_len=80,
    )

    assert "gateway/request-shaping guard" in response
    assert "not proof" in response
    assert "model context" in response
    assert "Session too large" not in response


def test_write_context_spill_collapses_nested_handoff_and_tool_output(tmp_path):
    cfg = _config(tmp_path, include_redacted_wiki_excerpt_chars=24_000)
    source = _source()
    entry = _entry(source)
    huge_skill_body = "SKILL_BODY_OMITTED_SHOULD_NOT_APPEAR " * 1000
    prior_handoff = "\n".join(
        [
            "[GATEWAY HANDOFF]",
            "The prior gateway session exceeded safe context limits and was spilled before model call.",
            "- Spill ID: prior-spill",
            f"- Wiki handoff: {tmp_path}/prior.md",
            "Current user ask, redacted excerpt:",
            "fix the app",
            "Instructions:",
            "1. Continue from this handoff.",
            huge_skill_body,
        ]
    )
    history = [
        {"role": "user", "content": prior_handoff},
        {"role": "tool", "content": json.dumps({"name": "skill_view", "content": huge_skill_body})},
        {"role": "assistant", "content": "continuing"},
    ]
    decision = decide_context_spill(history, "why is this happening", context_length=256000, config=cfg, stage="history")

    result = write_context_spill(
        history=history,
        message_text="why is this happening",
        source=source,
        session_entry=entry,
        decision=decision,
        config=cfg,
    )

    assert result.wiki_path and result.wiki_path.exists()
    wiki_text = result.wiki_path.read_text()
    assert "SKILL_BODY_OMITTED_SHOULD_NOT_APPEAR" not in wiki_text
    assert "prior gateway handoff body omitted" in wiki_text
    assert "skill_view output omitted" in wiki_text or "tool output omitted" in wiki_text
    assert len(wiki_text) < 30_000
    assert result.raw_path is not None
    raw_payload = json.loads(result.raw_path.read_text())
    assert "SKILL_BODY_OMITTED_SHOULD_NOT_APPEAR" in raw_payload["history"][1]["content"]


def test_write_context_spill_respects_notify_user_false(tmp_path):
    cfg = _config(tmp_path, notify_user=False)
    source = _source()
    entry = _entry(source)
    decision = decide_context_spill([], "ask", context_length=1, config=cfg, stage="history")

    result = write_context_spill(
        history=[],
        message_text="ask",
        source=source,
        session_entry=entry,
        decision=decision,
        config=cfg,
    )

    assert result.user_notice == ""
    assert result.recovery_message
    assert result.raw_path and result.raw_path.exists()


def test_load_context_spill_config_known_defaults_with_temp_paths(tmp_path):
    cfg = load_context_spill_config(
        {
            "gateway_context_spill": {
                "wiki_dir": str(tmp_path / "wiki" / "outputs" / "gateway-context-spills"),
                "raw_state_dir": str(tmp_path / "state" / "context-spills"),
            }
        },
        tmp_path,
        allow_unsafe_paths=True,
    )
    assert cfg.enabled is True
    assert cfg.mode == "enforce"
    assert cfg.hard_message_limit == 180


@pytest.mark.asyncio
async def test_pre_api_guard_spill_preserves_ask_and_quarantines_old_session(monkeypatch, tmp_path):
    from gateway.run import GatewayRunner
    import gateway.run as gateway_run

    source = _source()
    gw_cfg = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")})
    gw_cfg.sessions_dir = tmp_path / "sessions"
    store = SessionStore(gw_cfg.sessions_dir, gw_cfg)
    store._db = None
    entry = store.get_or_create_session(source)
    old_sid = entry.session_id

    runner = object.__new__(GatewayRunner)
    runner.config = gw_cfg
    runner.session_store = store
    runner.adapters = {}
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    runner._pending_model_notes = {}
    runner._agent_cache_lock = None

    raw_dir = tmp_path / "state" / "context-spills" / "custom-raw"
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {
            "gateway_context_spill": {
                "wiki_dir": str(tmp_path / "wiki" / "outputs" / "gateway-context-spills"),
                "raw_state_dir": str(raw_dir),
                "allow_unsafe_paths": True,
            }
        },
    )

    result, new_entry = await runner._spill_gateway_context_after_request_guard(
        history=[{"role": "user", "content": "old transcript"}],
        message_text="do not lose this current ask",
        source=source,
        session_entry=entry,
        session_key=entry.session_key,
        agent_result={
            "request_size_guard": {
                "approx_tokens": 190_000,
                "request_bytes": 2_000_000,
                "threshold_tokens": 180_000,
            }
        },
        context_prompt="system prompt that inflated the final request",
        channel_prompt="",
    )

    assert result.raw_path and result.raw_path.exists()
    assert result.wiki_path and result.wiki_path.exists()
    assert "do not lose this current ask" in result.recovery_message
    assert new_entry.session_id != old_sid
    assert store.is_session_quarantined(old_sid) is True
    assert store.switch_session(entry.session_key, old_sid) is None


@pytest.mark.asyncio
async def test_startup_auto_resume_skips_quarantined_session_ids(tmp_path):
    from gateway.run import GatewayRunner

    source = _source()
    gw_cfg = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")})
    gw_cfg.sessions_dir = tmp_path / "sessions"
    store = SessionStore(gw_cfg.sessions_dir, gw_cfg)
    store._db = None
    entry = store.get_or_create_session(source)
    entry.resume_pending = True
    entry.resume_reason = "restart_timeout"
    entry.last_resume_marked_at = datetime.now()
    entry.origin = source
    raw_dir = tmp_path / "state" / "context-spills"
    with store._lock:
        index = store._read_quarantine_index_locked()
        index[entry.session_id] = {
            "session_key": entry.session_key,
            "spill_id": "spill-1",
            "wiki_path": str(tmp_path / "wiki" / "spill.md"),
            "raw_path": str(raw_dir / "spill.json"),
            "reason": "pre_api_request_guard",
            "quarantined_at": datetime.now().isoformat(),
        }
        store._write_quarantine_index_locked(index)
        store._save()

    class RecordingAdapter:
        def __init__(self):
            self.events = []

        async def handle_message(self, event):
            self.events.append(event)

    adapter = RecordingAdapter()
    runner = object.__new__(GatewayRunner)
    runner.config = gw_cfg
    runner.session_store = store
    runner.adapters = {Platform.DISCORD: adapter}
    runner._background_tasks = set()

    scheduled = runner._schedule_resume_pending_sessions()

    assert scheduled == 0
    assert adapter.events == []
