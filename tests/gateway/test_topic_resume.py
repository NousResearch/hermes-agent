from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, TopicResumeConfig
from gateway.session import SessionEntry, SessionSource


UTC = timezone.utc


def _make_workspace(root: Path, workspace_id: str) -> Path:
    ws = root / ".hermes" / "topic-workspaces" / workspace_id
    (ws / "meta").mkdir(parents=True, exist_ok=True)
    (ws / "meta" / "topic.json").write_text(
        '{\n'
        '  "platform": "telegram",\n'
        '  "chat_id": "-1003748162924",\n'
        '  "thread_id": "853",\n'
        '  "topic_name": "Recorder Project",\n'
        f'  "workspace_id": "{workspace_id}"\n'
        '}\n',
        encoding="utf-8",
    )
    (ws / "state.md").write_text(
        "# Topic\n\n"
        "Topic Name: Recorder Project\n"
        f"Workspace ID: {workspace_id}\n"
        "Platform: Telegram\n"
        "Chat ID: -1003748162924\n"
        "Thread ID: 853\n"
        "Status: active\n"
        "Created: 2026-04-18T22:59:41+00:00\n"
        "Last Updated: 2026-04-23T13:50:00+00:00\n\n"
        "## Summary\n"
        "Build a lightweight local-first Windows game recorder.\n\n"
        "## Current State\n"
        "Latest checkpoint is commit `a4727a3`; black-screen capture remains unresolved.\n\n"
        "## Decisions\n"
        "- Keep Electron as the shell.\n"
        "- Prefer OBS-class capture backend.\n\n"
        "## Open Loops\n"
        "- Run the next Windows rerun on `a4727a3`.\n"
        "- Inspect the black-screen hook boundary.\n\n"
        "## Next Actions\n"
        "- Del: rerun on Windows and send logs.\n"
        "- Reno: inspect audio and black-screen behavior.\n\n"
        "## Operating Contract\n"
        "- Del gives the goal.\n"
        "- Reno writes the short practical plan.\n"
        "- Codex expands the plan and executes the code.\n"
        "- Reno supervises and updates Del.\n"
        "- Do not ask Del to manually shuttle prompts into Codex unless he explicitly asks for that mode.\n\n"
        "## Topic-Specific Instructions\n"
        "- Preserve the Recorder Project workflow on resume.\n",
        encoding="utf-8",
    )
    return ws


class _FakeSessionStore:
    def __init__(self, messages=None, messages_by_session_id=None):
        self._messages = messages or []
        self._messages_by_session_id = messages_by_session_id or {}

    def load_transcript(self, session_id: str):
        if session_id in self._messages_by_session_id:
            return list(self._messages_by_session_id[session_id])
        return list(self._messages)



def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1003748162924",
        chat_name="Recorder Project",
        chat_type="group",
        user_id="1352028621",
        user_name="Addel Hamoudhy",
        thread_id="853",
    )



def _make_session_entry() -> SessionEntry:
    now = datetime(2026, 4, 23, 14, 0, tzinfo=UTC)
    return SessionEntry(
        session_key="agent:main:telegram:group:-1003748162924:853",
        session_id="session-recorder",
        created_at=now,
        updated_at=now,
        platform=Platform.TELEGRAM,
        chat_type="group",
        was_auto_reset=True,
        auto_reset_reason="idle",
        reset_had_activity=True,
    )



def test_topic_resume_config_round_trip_in_gateway_config():
    cfg = GatewayConfig.from_dict(
        {
            "topic_resume": {
                "enabled": False,
                "recent_message_count": 5,
                "max_message_chars": 222,
                "include_recent_messages": False,
                "trigger_on_auto_reset": True,
            }
        }
    )

    assert isinstance(cfg.topic_resume, TopicResumeConfig)
    assert cfg.topic_resume.enabled is False
    assert cfg.topic_resume.recent_message_count == 5
    assert cfg.topic_resume.max_message_chars == 222
    assert cfg.topic_resume.include_recent_messages is False
    assert cfg.to_dict()["topic_resume"]["trigger_on_auto_reset"] is True



def test_resolve_topic_workspace_matches_platform_chat_and_thread(tmp_path):
    from gateway.topic_resume import resolve_topic_workspace

    ws = _make_workspace(tmp_path, "telegram--1003748162924-853-recorder-project")
    source = _make_source()

    resolved = resolve_topic_workspace(source, tmp_path / ".hermes")

    assert resolved == ws



def test_parse_workspace_state_extracts_operating_contract(tmp_path):
    from gateway.topic_resume import parse_workspace_state

    ws = _make_workspace(tmp_path, "telegram--1003748162924-853-recorder-project")
    parsed = parse_workspace_state(ws / "state.md")

    assert parsed["summary"].startswith("Build a lightweight")
    assert "a4727a3" in parsed["current_state"]
    assert parsed["open_loops"][0] == "Run the next Windows rerun on `a4727a3`."
    assert any("Codex expands the plan" in item for item in parsed["operating_contract"])
    assert any("manually shuttle prompts into Codex" in item for item in parsed["operating_contract"])



def test_load_recent_topic_messages_filters_to_user_and_assistant_and_truncates():
    from gateway.topic_resume import load_recent_topic_messages

    transcript = [
        {"role": "system", "content": "ignore me"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "tool", "content": "ignore me too"},
        {"role": "user", "content": "u2" * 300},
        {"role": "assistant", "content": "a2"},
    ]
    store = _FakeSessionStore(transcript)

    recent = load_recent_topic_messages(store, "session-recorder", limit=3, max_chars=40)

    assert [item["role"] for item in recent] == ["assistant", "user", "assistant"]
    assert recent[1]["content"].endswith("…")
    assert len(recent[1]["content"]) == 41



def test_build_topic_resume_context_combines_workspace_and_recent_messages(tmp_path):
    from gateway.topic_resume import build_topic_resume_context

    _make_workspace(tmp_path, "telegram--1003748162924-853-recorder-project")
    source = _make_source()
    session_entry = _make_session_entry()
    store = _FakeSessionStore(
        [
            {"role": "user", "content": "Can we continue from last night?"},
            {"role": "assistant", "content": "Yes, same recorder workflow as before."},
        ]
    )
    cfg = GatewayConfig()

    ctx = build_topic_resume_context(
        source=source,
        session_store=store,
        session_entry=session_entry,
        config=cfg.topic_resume,
        hermes_home=tmp_path / ".hermes",
        is_new_session=True,
    )

    assert ctx is not None
    assert ctx.workspace_id == "telegram--1003748162924-853-recorder-project"
    assert "a4727a3" in (ctx.current_state or "")
    assert any("Codex expands the plan" in item for item in ctx.operating_contract)
    assert ctx.recent_messages[0]["role"] == "user"
    assert ctx.was_auto_resume is True



def test_build_topic_resume_context_uses_previous_session_messages_after_auto_reset(tmp_path):
    from gateway.topic_resume import build_topic_resume_context

    _make_workspace(tmp_path, "telegram--1003748162924-853-recorder-project")
    source = _make_source()
    session_entry = _make_session_entry()
    session_entry.session_id = "session-new"
    session_entry.previous_session_id = "session-old"

    store = _FakeSessionStore(
        messages_by_session_id={
            "session-old": [
                {"role": "user", "content": "I sent fresh Windows logs late last night."},
                {"role": "assistant", "content": "Good, I am reviewing those logs now."},
            ],
            "session-new": [
                {"role": "user", "content": "where are we at here. what are we waiting on?"},
            ],
        }
    )

    ctx = build_topic_resume_context(
        source=source,
        session_store=store,
        session_entry=session_entry,
        config=GatewayConfig().topic_resume,
        hermes_home=tmp_path / ".hermes",
        is_new_session=True,
    )

    assert ctx is not None
    assert ctx.recent_messages == [
        {"role": "user", "content": "I sent fresh Windows logs late last night."},
        {"role": "assistant", "content": "Good, I am reviewing those logs now."},
    ]



def test_build_topic_resume_context_uses_previous_session_messages_for_manual_new_session(tmp_path):
    from gateway.topic_resume import build_topic_resume_context

    _make_workspace(tmp_path, "telegram--1003748162924-853-recorder-project")
    session_entry = _make_session_entry()
    session_entry.was_auto_reset = False
    session_entry.session_id = "session-new"
    session_entry.previous_session_id = "session-old"

    store = _FakeSessionStore(
        messages_by_session_id={
            "session-old": [
                {"role": "user", "content": "Last thing we did was verify the hydration bug."},
                {"role": "assistant", "content": "Next I should patch the resume path."},
            ],
            "session-new": [
                {"role": "user", "content": "brand new empty session"},
            ],
        }
    )

    ctx = build_topic_resume_context(
        source=_make_source(),
        session_store=store,
        session_entry=session_entry,
        config=GatewayConfig().topic_resume,
        hermes_home=tmp_path / ".hermes",
        is_new_session=True,
    )

    assert ctx is not None
    assert ctx.recent_messages == [
        {"role": "user", "content": "Last thing we did was verify the hydration bug."},
        {"role": "assistant", "content": "Next I should patch the resume path."},
    ]



def test_build_topic_resume_prompt_includes_contract_and_recent_messages(tmp_path):
    from gateway.topic_resume import build_topic_resume_context, build_topic_resume_prompt

    _make_workspace(tmp_path, "telegram--1003748162924-853-recorder-project")
    ctx = build_topic_resume_context(
        source=_make_source(),
        session_store=_FakeSessionStore(
            [
                {"role": "user", "content": "Please continue on the recorder project."},
                {"role": "assistant", "content": "I’ll keep the same Codex-supervised workflow."},
            ]
        ),
        session_entry=_make_session_entry(),
        config=GatewayConfig().topic_resume,
        hermes_home=tmp_path / ".hermes",
        is_new_session=True,
    )

    prompt = build_topic_resume_prompt(ctx)

    assert "## Topic Resume Context" in prompt
    assert "Operating Contract" in prompt
    assert "Do not ask Del to manually shuttle prompts into Codex" in prompt
    assert "Recent Topic Messages" in prompt
    assert "Please continue on the recorder project." in prompt



def test_build_topic_resume_context_skips_when_feature_disabled(tmp_path):
    from gateway.topic_resume import build_topic_resume_context

    _make_workspace(tmp_path, "telegram--1003748162924-853-recorder-project")
    cfg = TopicResumeConfig(enabled=False)

    ctx = build_topic_resume_context(
        source=_make_source(),
        session_store=_FakeSessionStore([]),
        session_entry=_make_session_entry(),
        config=cfg,
        hermes_home=tmp_path / ".hermes",
        is_new_session=True,
    )

    assert ctx is None



def test_build_topic_resume_context_respects_auto_reset_trigger_flag(tmp_path):
    from gateway.topic_resume import build_topic_resume_context

    _make_workspace(tmp_path, "telegram--1003748162924-853-recorder-project")
    session_entry = _make_session_entry()
    session_entry.was_auto_reset = True
    cfg = TopicResumeConfig(trigger_on_new_session=True, trigger_on_auto_reset=False)

    ctx = build_topic_resume_context(
        source=_make_source(),
        session_store=_FakeSessionStore(
            [{"role": "user", "content": "fresh but should be ignored"}]
        ),
        session_entry=session_entry,
        config=cfg,
        hermes_home=tmp_path / ".hermes",
        is_new_session=True,
    )

    assert ctx is None


@pytest.mark.asyncio
async def test_gateway_runner_passes_topic_resume_context_prompt_to_run_agent(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner

    _make_workspace(tmp_path, "telegram--1003748162924-853-recorder-project")

    now = datetime(2026, 4, 23, 14, 0, tzinfo=UTC)
    session_entry = SessionEntry(
        session_key="agent:main:telegram:group:-1003748162924:853",
        session_id="session-recorder",
        created_at=now,
        updated_at=now,
        platform=Platform.TELEGRAM,
        chat_type="group",
        was_auto_reset=True,
        auto_reset_reason="idle",
        reset_had_activity=False,
    )

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(topic_resume=TopicResumeConfig())
    runner.adapters = {Platform.TELEGRAM: MagicMock()}
    runner.adapters[Platform.TELEGRAM].send = AsyncMock()
    runner.hooks = SimpleNamespace(emit=MagicMock())
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = [
        {"role": "user", "content": "Please continue from where we left off."},
        {"role": "assistant", "content": "Same recorder workflow as before."},
    ]
    runner.session_store.config = runner.config
    runner._set_session_env = MagicMock(return_value=[])
    runner._clear_session_env = MagicMock()
    runner._bind_adapter_run_generation = MagicMock()
    runner._is_session_run_current = MagicMock(return_value=True)
    runner._format_session_info = MagicMock(return_value="")
    runner._update_runtime_status = MagicMock()
    runner._resolve_session_agent_runtime = MagicMock()
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner._active_event_loop = None
    runner._loop_thread = None
    runner.delivery_router = MagicMock()
    runner._pending_approvals = {}
    runner._voice_mode = {}
    runner._background_tasks = set()
    runner._should_send_voice_reply = MagicMock(return_value=False)

    captured = {}

    async def _fake_run_agent(**kwargs):
        captured["context_prompt"] = kwargs["context_prompt"]
        return {"final_response": "ok", "messages": [], "api_calls": 1}

    runner._run_agent = _fake_run_agent

    async def _fake_emit(*args, **kwargs):
        return None

    runner.hooks.emit = _fake_emit

    source = _make_source()
    event = SimpleNamespace(
        text="Continue the recorder project",
        message_id="m1",
        channel_prompt=None,
        metadata=None,
        auto_skill=None,
        media_urls=[],
        media_types=[],
        quoted_text=None,
        quoted_sender_name=None,
        quoted_media_urls=[],
        quoted_media_types=[],
        sender_name=None,
        sender_id=None,
        message_type=None,
        reply_to_message_id=None,
        source=source,
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path / ".hermes")
    monkeypatch.setattr("gateway.run._config_path", tmp_path / ".hermes" / "config.yaml")
    (tmp_path / ".hermes").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".hermes" / "config.yaml").write_text("{}\n", encoding="utf-8")

    await runner._handle_message_with_agent(event, source, session_entry.session_key, 1)

    assert "## Topic Resume Context" in captured["context_prompt"]
    assert "Recorder Project" in captured["context_prompt"]
    assert "Do not ask Del to manually shuttle prompts into Codex" in captured["context_prompt"]



def test_agent_config_signature_changes_when_session_id_changes():
    from gateway.run import GatewayRunner

    runtime = {
        "provider": "openai-codex",
        "api_key": "secret-token",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api_mode": "responses",
    }

    sig_a = GatewayRunner._agent_config_signature(
        "gpt-5.4",
        runtime,
        ["file", "terminal"],
        "topic prompt",
        "session-a",
    )
    sig_b = GatewayRunner._agent_config_signature(
        "gpt-5.4",
        runtime,
        ["file", "terminal"],
        "topic prompt",
        "session-b",
    )

    assert sig_a != sig_b
