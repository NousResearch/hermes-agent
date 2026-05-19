"""Regression coverage for the Nox v2 / Башня Telegram coordinator router.

The coordinator topic must not run heavy project tasks in the main Башня
session. Known project work is packeted to the matching `<Project> · Сотрудник`
topic and the main topic only receives a compact acknowledgment.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


BASHNYA_CHAT_ID = "-1003889439571"
BASHNYA_THREAD_ID = "25"


class CaptureAdapter:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(
            {
                "chat_id": str(chat_id),
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata or {},
            }
        )
        return SendResult(success=True, message_id=f"sent-{len(self.sent)}")

    async def stop_typing(self, chat_id):
        return None


class StateCheckingAdapter(CaptureAdapter):
    def __init__(self, phase_path) -> None:
        super().__init__()
        self.phase_path = phase_path

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        if "PROJECT EMPLOYEE TASK" in content:
            match = re.search(r"^run_id: (\S+)$", content, flags=re.MULTILINE)
            assert match, "employee packet must include run_id before send"
            run_id = match.group(1)
            payload = json.loads(self.phase_path.read_text(encoding="utf-8"))
            assert payload["runs"][run_id]["status"] == "planning"
            assert payload["runs"][run_id]["packet_message_id"] == ""
        return await super().send(chat_id, content, reply_to=reply_to, metadata=metadata)


class FakeSessionStore:
    def __init__(self, source: SessionSource) -> None:
        self.entry = SimpleNamespace(
            session_id="sess-bashnya",
            session_key=build_session_key(source),
            created_at=1,
            updated_at=2,
            was_auto_reset=False,
            is_fresh_reset=False,
            last_prompt_tokens=0,
        )
        self.config = SimpleNamespace(
            get_reset_policy=lambda **_kwargs: SimpleNamespace(
                notify=False,
                notify_exclude_platforms=[],
                idle_minutes=0,
                at_hour=0,
            )
        )

    def get_or_create_session(self, _source):
        return self.entry

    def load_transcript(self, _session_id):
        return []

    def has_any_sessions(self):
        return True


@pytest.fixture
def bashnya_routes_path(tmp_path, monkeypatch):
    path = tmp_path / "project-employees.json"
    path.write_text(
        json.dumps(
            {
                "projects": {
                    "metaauto": {
                        "project": "metaauto",
                        "display": "MetaAuto",
                        "topic_name": "MetaAuto · Сотрудник",
                        "chat_id": BASHNYA_CHAT_ID,
                        "message_thread_id": "31",
                    },
                    "ryadom": {
                        "project": "ryadom",
                        "display": "Ryadom",
                        "topic_name": "Ryadom · Сотрудник",
                        "chat_id": BASHNYA_CHAT_ID,
                        "message_thread_id": "32",
                    },
                    "nox-system": {
                        "project": "nox-system",
                        "display": "Nox OS",
                        "topic_name": "Nox OS · Сотрудник",
                        "chat_id": BASHNYA_CHAT_ID,
                        "message_thread_id": "33",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    phase_path = tmp_path / "phase-runs.json"
    monkeypatch.setenv("NOX_PROJECT_EMPLOYEE_ROUTING_PATH", str(path))
    monkeypatch.setenv("NOX_PHASE_RUNS_PATH", str(phase_path))
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", BASHNYA_CHAT_ID)
    return path


@pytest.fixture
def nox_phase_runs_path(bashnya_routes_path):
    return bashnya_routes_path.with_name("phase-runs.json")


def _make_bashnya_event(text: str) -> MessageEvent:
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=BASHNYA_CHAT_ID,
        thread_id=BASHNYA_THREAD_ID,
        chat_type="supergroup",
        user_id="8756671470",
        user_name="Danya",
    )
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id="source-msg-1",
    )


def _make_runner(source: SessionSource, adapter: CaptureAdapter) -> Any:
    runner: Any = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")},
        group_sessions_per_user=False,
        thread_sessions_per_user=False,
        stt_enabled=False,
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.session_store = FakeSessionStore(source)
    runner._session_db = None
    runner._voice_mode = {}
    runner._pending_native_image_paths_by_session = {}
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._session_run_generation = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_ack_ts = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._reasoning_config = None
    runner._service_tier = None
    runner._background_tasks = set()
    runner._nox_worker_agents_by_run_id = {}
    runner._nox_worker_agents_by_task_id = {}
    runner._active_profile_name = lambda: "nox-v2"
    runner._is_telegram_topic_lane = lambda _source: False
    runner._set_session_env = lambda _context: []
    runner._clear_session_env = lambda _tokens: None
    runner._bind_adapter_run_generation = lambda *_args, **_kwargs: None
    runner._reply_anchor_for_event = lambda _event: _event.message_id
    runner._thread_metadata_for_source = lambda source, reply_anchor=None: {
        "thread_id": source.thread_id,
        "reply_to_message_id": reply_anchor,
    }
    runner._resolve_session_reasoning_config = lambda **_kwargs: None
    runner._load_service_tier = lambda: None
    runner._is_session_run_current = lambda *_args, **_kwargs: True
    runner._start_nox_worker_plan_runner = lambda **_kwargs: "plan-task-id"
    runner._start_nox_worker_phase_runner = lambda **_kwargs: "phase-task-id"
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    return runner


@pytest.mark.asyncio
async def test_bashnya_project_task_routes_to_employee_lane_before_agent(monkeypatch, bashnya_routes_path):
    """Known heavy project task in Башня is packeted to employee lane, not run in main lane."""
    event = _make_bashnya_event("почини metaauto coordinator routing и прогони тесты")
    adapter = CaptureAdapter()
    runner = _make_runner(event.source, adapter)

    async def fail_if_agent_runs(**_kwargs):  # pragma: no cover - should not be called
        raise AssertionError("project task must be routed before _run_agent")

    runner._run_agent = fail_if_agent_runs

    ack = await runner._handle_message_with_agent(
        event,
        event.source,
        build_session_key(event.source),
        run_generation=1,
    )

    assert "Статус: взял в работу" in ack
    assert "Куда ушло: MetaAuto · Сотрудник" in ack
    assert "run_id:" in ack

    assert len(adapter.sent) == 1
    packet = adapter.sent[0]
    assert packet["chat_id"] == BASHNYA_CHAT_ID
    assert packet["metadata"] == {"thread_id": "31"}
    assert "PROJECT EMPLOYEE TASK" in packet["content"]
    assert "Project: MetaAuto" in packet["content"]
    assert "Mode: RUN" in packet["content"]
    assert "почини metaauto coordinator routing" in packet["content"]


@pytest.mark.asyncio
async def test_bashnya_cross_project_skill_task_routes_to_nox_system_lane_before_agent(bashnya_routes_path):
    """Cross-project skills/agents work belongs to Nox OS, not the main Башня session."""
    event = _make_bashnya_event(
        "внедри UI/design skills для всех проектов и самих агентов, "
        "чтобы любой агент мог ими пользоваться"
    )
    adapter = CaptureAdapter()
    runner = _make_runner(event.source, adapter)

    async def fail_if_agent_runs(**_kwargs):  # pragma: no cover - should not be called
        raise AssertionError("cross-project agent task must be routed before _run_agent")

    runner._run_agent = fail_if_agent_runs

    ack = await runner._handle_message_with_agent(
        event,
        event.source,
        build_session_key(event.source),
        run_generation=1,
    )

    assert "Статус: взял в работу" in ack
    assert "Куда ушло: Nox OS · Сотрудник" in ack
    assert "run_id:" in ack

    assert len(adapter.sent) == 1
    packet = adapter.sent[0]
    assert packet["chat_id"] == BASHNYA_CHAT_ID
    assert packet["metadata"] == {"thread_id": "33"}
    assert "PROJECT EMPLOYEE TASK" in packet["content"]
    assert "Project: Nox OS" in packet["content"]
    assert "Mode: PROJECT" in packet["content"]
    assert "UI/design skills" in packet["content"]


@pytest.mark.asyncio
async def test_bashnya_explicit_project_wins_over_system_skill_markers(bashnya_routes_path):
    """System-scope vocabulary must not steal an explicitly named product project."""
    event = _make_bashnya_event("внедри skills в metaauto агенте и прогони тесты")
    adapter = CaptureAdapter()
    runner = _make_runner(event.source, adapter)

    async def fail_if_agent_runs(**_kwargs):  # pragma: no cover - should not be called
        raise AssertionError("explicit project task must be routed before _run_agent")

    runner._run_agent = fail_if_agent_runs

    ack = await runner._handle_message_with_agent(
        event,
        event.source,
        build_session_key(event.source),
        run_generation=1,
    )

    assert "Статус: взял в работу" in ack
    assert "Куда ушло: MetaAuto · Сотрудник" in ack
    assert len(adapter.sent) == 1
    assert adapter.sent[0]["metadata"] == {"thread_id": "31"}
    assert "Project: MetaAuto" in adapter.sent[0]["content"]


@pytest.mark.asyncio
async def test_bashnya_unqualified_landing_followup_infers_ryadom_active_context(bashnya_routes_path):
    """Long landing follow-up in Башня must route to Ryadom, not continue main control chat."""
    event = _make_bashnya_event(
        "[The user sent a voice message~ Here's what they said: "
        "\"Что не хватает нашему лендингу, чтобы он был максимально продающим? "
        "Нужны хуки, вау hero, отзывы, ценник и макеты до кода.\"]"
    )
    adapter = CaptureAdapter()
    runner = _make_runner(event.source, adapter)
    runner._save_nox_phase_runs(
        {
            "meta-run": {
                "run_id": "meta-run",
                "status": "awaiting_approval",
                "project": "metaauto",
                "display": "MetaAuto",
                "bashnya_chat_id": BASHNYA_CHAT_ID,
                "bashnya_thread_id": BASHNYA_THREAD_ID,
                "employee_chat_id": BASHNYA_CHAT_ID,
                "employee_thread_id": "31",
                "topic_name": "MetaAuto · Сотрудник",
                "updated_at": "2026-05-18T19:43:31",
            },
            "ryadom-run": {
                "run_id": "ryadom-run",
                "status": "awaiting_approval",
                "project": "ryadom",
                "display": "Ryadom",
                "bashnya_chat_id": BASHNYA_CHAT_ID,
                "bashnya_thread_id": BASHNYA_THREAD_ID,
                "employee_chat_id": BASHNYA_CHAT_ID,
                "employee_thread_id": "32",
                "topic_name": "Ryadom · Сотрудник",
                "updated_at": "2026-05-18T23:37:03",
            },
        }
    )

    async def fail_if_agent_runs(**_kwargs):  # pragma: no cover - should not be called
        raise AssertionError("unqualified project follow-up must not run in Башня")

    runner._run_agent = fail_if_agent_runs

    ack = await runner._handle_message_with_agent(
        event,
        event.source,
        build_session_key(event.source),
        run_generation=1,
    )

    assert "Статус: взял в работу" in ack
    assert "Куда ушло: Ryadom · Сотрудник" in ack
    assert adapter.sent[0]["metadata"] == {"thread_id": "32"}
    assert "Project: Ryadom" in adapter.sent[0]["content"]


@pytest.mark.asyncio
async def test_bashnya_substantial_unqualified_task_blocks_instead_of_main_agent(bashnya_routes_path):
    """If project cannot be inferred, Башня asks for project and does not run the main agent."""
    event = _make_bashnya_event(
        "Собери максимально сильный рабочий план: сначала аудит, потом варианты, "
        "потом список рисков. Ничего не коммить и не пушить, "
        "просто разложи по фазам и покажи мне proof в правильном месте."
    )
    adapter = CaptureAdapter()
    runner = _make_runner(event.source, adapter)
    runner._save_nox_phase_runs(
        {
            "meta-run": {
                "run_id": "meta-run",
                "status": "awaiting_approval",
                "project": "metaauto",
                "bashnya_chat_id": BASHNYA_CHAT_ID,
                "bashnya_thread_id": BASHNYA_THREAD_ID,
                "employee_chat_id": BASHNYA_CHAT_ID,
                "employee_thread_id": "31",
                "topic_name": "MetaAuto · Сотрудник",
                "updated_at": "2026-05-18T19:43:31",
            },
            "ryadom-run": {
                "run_id": "ryadom-run",
                "status": "awaiting_approval",
                "project": "ryadom",
                "bashnya_chat_id": BASHNYA_CHAT_ID,
                "bashnya_thread_id": BASHNYA_THREAD_ID,
                "employee_chat_id": BASHNYA_CHAT_ID,
                "employee_thread_id": "32",
                "topic_name": "Ryadom · Сотрудник",
                "updated_at": "2026-05-18T23:37:03",
            },
        }
    )

    async def fail_if_agent_runs(**_kwargs):  # pragma: no cover - should not be called
        raise AssertionError("ambiguous project task must block before _run_agent")

    runner._run_agent = fail_if_agent_runs

    reply = await runner._handle_message_with_agent(
        event,
        event.source,
        build_session_key(event.source),
        run_generation=1,
    )

    assert "проект не указан" in reply
    assert "Main run в Башне не запускаю" in reply
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_bashnya_fast_question_stays_direct_and_can_be_capped(bashnya_routes_path):
    """Brief/non-project Башня messages are not packeted away and use the light turn cap."""
    event = _make_bashnya_event("почему так долго?")
    adapter = CaptureAdapter()
    runner = _make_runner(event.source, adapter)

    routed = await runner._maybe_route_nox_bashnya_task(event, event.text)

    assert routed is None
    assert adapter.sent == []
    assert runner._maybe_cap_nox_bashnya_fast_iterations(event.source, event.text, 90) == 8


def _make_employee_event(text: str) -> MessageEvent:
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=BASHNYA_CHAT_ID,
        thread_id="31",
        chat_type="supergroup",
        user_id="8756671470",
        user_name="Danya",
    )
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id="employee-msg-1",
    )


def _make_ryadom_employee_event(text: str) -> MessageEvent:
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=BASHNYA_CHAT_ID,
        thread_id="32",
        chat_type="supergroup",
        user_id="8756671470",
        user_name="Danya",
    )
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id="ryadom-employee-msg-1",
    )


@pytest.mark.asyncio
async def test_bashnya_project_task_starts_worker_plan_runner(monkeypatch, bashnya_routes_path):
    """Routing a project task should start a plan-only worker run, not just post a packet."""
    event = _make_bashnya_event("почини metaauto импорты и разбей работу на фазы")
    adapter = CaptureAdapter()
    runner = _make_runner(event.source, adapter)
    started: list[dict] = []

    def fake_start_plan_runner(*, run_id, route, state):
        started.append({"run_id": run_id, "route": route, "state": state})
        return "plan-task-id"

    runner._start_nox_worker_plan_runner = fake_start_plan_runner

    ack = await runner._handle_message_with_agent(
        event,
        event.source,
        build_session_key(event.source),
        run_generation=1,
    )

    assert "Режим: PROJECT" in ack
    assert "Планер: запущен" in ack
    assert started, "plan runner must start after routing"
    run_id = started[0]["run_id"]
    runs = runner._load_nox_phase_runs()
    assert runs[run_id]["status"] == "planning"
    assert runs[run_id]["goal"] == event.text
    assert runs[run_id]["employee_thread_id"] == "31"


@pytest.mark.asyncio
async def test_employee_approval_starts_next_phase(monkeypatch, bashnya_routes_path):
    """After the worker plan is ready, 'утвердить' in the employee topic starts phase 1 only."""
    event = _make_employee_event("утвердить")
    adapter = CaptureAdapter()
    runner = _make_runner(event.source, adapter)
    runner._save_nox_phase_runs(
        {
            "run-1": {
                "run_id": "run-1",
                "status": "awaiting_approval",
                "project": "metaauto",
                "display": "MetaAuto",
                "goal": "почини metaauto импорты",
                "plan": "Фаза 1: диагностика\nФаза 2: фикс и тесты",
                "phase_index": 0,
                "phase_count": 2,
                "bashnya_chat_id": BASHNYA_CHAT_ID,
                "bashnya_thread_id": BASHNYA_THREAD_ID,
                "employee_chat_id": BASHNYA_CHAT_ID,
                "employee_thread_id": "31",
                "topic_name": "MetaAuto · Сотрудник",
            }
        }
    )
    started: list[dict] = []

    def fake_start_phase_runner(*, run_id, route, state):
        started.append({"run_id": run_id, "route": route, "state": dict(state)})
        return "phase-task-id"

    runner._start_nox_worker_phase_runner = fake_start_phase_runner

    reply = await runner._maybe_handle_nox_worker_approval(event, event.text)

    assert "Фаза 1" in reply
    assert "запущена" in reply
    assert started, "phase runner must start only after approval"
    runs = runner._load_nox_phase_runs()
    assert runs["run-1"]["status"] == "running_phase"
    assert runs["run-1"]["phase_index"] == 1
    assert started[0]["state"]["phase_index"] == 1


@pytest.mark.asyncio
async def test_bare_plus_approval_in_employee_topic_starts_matching_phase(bashnya_routes_path):
    """Bare '+' is allowed only as a deterministic gate approval in the current worker topic."""
    event = _make_employee_event("+")
    adapter = CaptureAdapter()
    runner = _make_runner(event.source, adapter)
    runner._save_nox_phase_runs(
        {
            "run-plus": {
                "run_id": "run-plus",
                "status": "awaiting_approval",
                "project": "metaauto",
                "display": "MetaAuto",
                "goal": "safe next phase",
                "plan": "Фаза 1: диагностика",
                "phase_index": 0,
                "phase_count": 1,
                "bashnya_chat_id": BASHNYA_CHAT_ID,
                "bashnya_thread_id": BASHNYA_THREAD_ID,
                "employee_chat_id": BASHNYA_CHAT_ID,
                "employee_thread_id": "31",
                "topic_name": "MetaAuto · Сотрудник",
            }
        }
    )
    started: list[dict] = []

    def fake_start_phase_runner(*, run_id, route, state):
        started.append({"run_id": run_id, "route": route, "state": dict(state)})
        return "phase-task-id"

    runner._start_nox_worker_phase_runner = fake_start_phase_runner

    reply = await runner._maybe_handle_nox_worker_approval(event, "[Exzy] +")

    assert "Фаза 1" in reply
    assert started[0]["run_id"] == "run-plus"
    assert runner._load_nox_phase_runs()["run-plus"]["status"] == "running_phase"


@pytest.mark.asyncio
async def test_bare_plus_in_bashnya_without_waiting_gate_stops_before_agent(bashnya_routes_path):
    """Bare '+' in Башня must not fall through to the LLM and continue stale context."""
    event = _make_bashnya_event("+")
    adapter = CaptureAdapter()
    runner = _make_runner(event.source, adapter)

    async def fail_if_agent_runs(**_kwargs):  # pragma: no cover - should not be called
        raise AssertionError("bare plus must be handled deterministically before _run_agent")

    runner._run_agent = fail_if_agent_runs

    reply = await runner._handle_message_with_agent(
        event,
        event.source,
        build_session_key(event.source),
        run_generation=1,
    )

    assert "Нет worker-фазы" in reply
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_bare_plus_in_bashnya_requires_run_id_when_multiple_gates(bashnya_routes_path):
    """Bare '+' in Башня is rejected if more than one active gate could match."""
    event = _make_bashnya_event("+")
    adapter = CaptureAdapter()
    runner = _make_runner(event.source, adapter)
    base_state = {
        "status": "awaiting_approval",
        "goal": "safe next phase",
        "plan": "Фаза 1: диагностика",
        "phase_index": 0,
        "phase_count": 1,
        "bashnya_chat_id": BASHNYA_CHAT_ID,
        "bashnya_thread_id": BASHNYA_THREAD_ID,
    }
    runner._save_nox_phase_runs(
        {
            "meta-run": {
                "run_id": "meta-run",
                "project": "metaauto",
                "display": "MetaAuto",
                "employee_chat_id": BASHNYA_CHAT_ID,
                "employee_thread_id": "31",
                "topic_name": "MetaAuto · Сотрудник",
                **base_state,
            },
            "ryadom-run": {
                "run_id": "ryadom-run",
                "project": "ryadom",
                "display": "Ryadom",
                "employee_chat_id": BASHNYA_CHAT_ID,
                "employee_thread_id": "32",
                "topic_name": "Ryadom · Сотрудник",
                **base_state,
            },
        }
    )

    async def fail_if_agent_runs(**_kwargs):  # pragma: no cover - should not be called
        raise AssertionError("ambiguous plus must be handled before _run_agent")

    runner._run_agent = fail_if_agent_runs

    reply = await runner._handle_message_with_agent(
        event,
        event.source,
        build_session_key(event.source),
        run_generation=1,
    )

    assert "Нужен run_id" in reply
    assert "meta-run" in reply and "ryadom-run" in reply
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_prefixed_employee_approval_is_bound_to_exact_project_thread(bashnya_routes_path):
    """Gateway-prepared '[User] утвердить' must approve only the run waiting in this topic."""
    event = _make_ryadom_employee_event("утвердить")
    adapter = CaptureAdapter()
    runner = _make_runner(event.source, adapter)
    base_state = {
        "status": "awaiting_approval",
        "goal": "safe next phase",
        "plan": "Фаза 1: диагностика",
        "phase_index": 0,
        "phase_count": 1,
        "bashnya_chat_id": BASHNYA_CHAT_ID,
        "bashnya_thread_id": BASHNYA_THREAD_ID,
    }
    runner._save_nox_phase_runs(
        {
            "meta-run": {
                "run_id": "meta-run",
                "project": "metaauto",
                "display": "MetaAuto",
                "employee_chat_id": BASHNYA_CHAT_ID,
                "employee_thread_id": "31",
                "topic_name": "MetaAuto · Сотрудник",
                **base_state,
            },
            "ryadom-run": {
                "run_id": "ryadom-run",
                "project": "ryadom",
                "display": "Ryadom",
                "employee_chat_id": BASHNYA_CHAT_ID,
                "employee_thread_id": "32",
                "topic_name": "Ryadom · Сотрудник",
                **base_state,
            },
        }
    )
    started: list[dict] = []

    def fake_start_phase_runner(*, run_id, route, state):
        started.append({"run_id": run_id, "route": route, "state": dict(state)})
        return "phase-task-id"

    runner._start_nox_worker_phase_runner = fake_start_phase_runner

    reply = await runner._maybe_handle_nox_worker_approval(event, "[Exzy] утвердить")

    assert "Ryadom · Сотрудник" in reply
    assert started[0]["run_id"] == "ryadom-run"
    runs = runner._load_nox_phase_runs()
    assert runs["meta-run"]["status"] == "awaiting_approval"
    assert runs["ryadom-run"]["status"] == "running_phase"


@pytest.mark.asyncio
async def test_project_employee_approval_without_matching_run_stops_before_agent(bashnya_routes_path):
    """An approval word in a project employee topic must not fall through to the LLM."""
    event = _make_ryadom_employee_event("утвердить")
    runner = _make_runner(event.source, CaptureAdapter())
    runner._save_nox_phase_runs(
        {
            "meta-run": {
                "run_id": "meta-run",
                "status": "awaiting_approval",
                "project": "metaauto",
                "display": "MetaAuto",
                "goal": "safe next phase",
                "plan": "Фаза 1: диагностика",
                "phase_index": 0,
                "phase_count": 1,
                "bashnya_chat_id": BASHNYA_CHAT_ID,
                "bashnya_thread_id": BASHNYA_THREAD_ID,
                "employee_chat_id": BASHNYA_CHAT_ID,
                "employee_thread_id": "31",
                "topic_name": "MetaAuto · Сотрудник",
            }
        }
    )

    reply = await runner._maybe_handle_nox_worker_approval(event, "[Exzy] утвердить")

    assert reply is not None
    assert "Нет worker-фазы" in reply
    assert "Ryadom · Сотрудник" in reply
    assert runner._load_nox_phase_runs()["meta-run"]["status"] == "awaiting_approval"


class FakeSteerAgent:
    def __init__(self) -> None:
        self.steers: list[str] = []

    def steer(self, text: str) -> bool:
        self.steers.append(text)
        return True


@pytest.mark.asyncio
async def test_employee_live_steer_delivers_to_running_worker_agent_and_persists_amendment(bashnya_routes_path):
    """Plain text in `<Project> · Сотрудник` during an active worker run is live-steered, not run as a new chat."""
    event = _make_employee_event("усвой прямо сейчас: не коммить, только покажи report")
    runner = _make_runner(event.source, CaptureAdapter())
    runner._save_nox_phase_runs(
        {
            "run-live": {
                "run_id": "run-live",
                "status": "running_phase",
                "project": "metaauto",
                "display": "MetaAuto",
                "goal": "почини metaauto импорты",
                "plan": "Фаза 1: диагностика",
                "phase_index": 1,
                "phase_count": 1,
                "phase_task_id": "phase-task-id",
                "bashnya_chat_id": BASHNYA_CHAT_ID,
                "bashnya_thread_id": BASHNYA_THREAD_ID,
                "employee_chat_id": BASHNYA_CHAT_ID,
                "employee_thread_id": "31",
                "topic_name": "MetaAuto · Сотрудник",
            }
        }
    )
    fake_agent = FakeSteerAgent()
    runner._nox_worker_agents_by_run_id = {"run-live": fake_agent}

    reply = await runner._maybe_handle_nox_worker_live_steer(event, "[Danya] " + event.text)

    assert "Усвоил прямо сейчас" in reply
    assert "run-live" in reply
    assert fake_agent.steers, "active worker agent must receive a live steer"
    assert "не коммить" in fake_agent.steers[0]
    assert "run_id: run-live" in fake_agent.steers[0]
    runs = runner._load_nox_phase_runs()
    amendment = runs["run-live"]["amendments"][-1]
    assert amendment["delivery"] == "delivered_live"
    assert amendment["message_id"] == "employee-msg-1"
    assert "только покажи report" in amendment["text"]


@pytest.mark.asyncio
async def test_employee_live_steer_without_registered_worker_is_saved_and_blocks_main_agent(bashnya_routes_path):
    """If the hidden worker is starting/restarted, employee follow-up is still stored as run amendment and does not hit the main LLM."""
    event = _make_employee_event("добавь ограничение: без production deploy")
    adapter = CaptureAdapter()
    runner = _make_runner(event.source, adapter)
    runner._save_nox_phase_runs(
        {
            "run-planning": {
                "run_id": "run-planning",
                "status": "planning",
                "project": "metaauto",
                "display": "MetaAuto",
                "goal": "почини metaauto импорты",
                "phase_index": 0,
                "phase_count": 0,
                "bashnya_chat_id": BASHNYA_CHAT_ID,
                "bashnya_thread_id": BASHNYA_THREAD_ID,
                "employee_chat_id": BASHNYA_CHAT_ID,
                "employee_thread_id": "31",
                "topic_name": "MetaAuto · Сотрудник",
            }
        }
    )

    async def fail_if_agent_runs(**_kwargs):  # pragma: no cover - should not be called
        raise AssertionError("employee live steer must be handled before _run_agent")

    runner._run_agent = fail_if_agent_runs

    reply = await runner._handle_message_with_agent(
        event,
        event.source,
        build_session_key(event.source),
        run_generation=1,
    )

    assert "сохранил amendment" in reply
    assert "run-planning" in reply
    runs = runner._load_nox_phase_runs()
    assert runs["run-planning"]["amendments"][-1]["delivery"] == "saved_for_worker"
    assert "без production deploy" in runs["run-planning"]["amendments"][-1]["text"]
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_employee_live_steer_multiple_active_runs_requires_explicit_run_id(bashnya_routes_path):
    """Ambiguous employee-topic amendments stop until Danya names the target run_id."""
    event = _make_employee_event("добавь это ограничение в текущую фазу")
    runner = _make_runner(event.source, CaptureAdapter())
    base_state = {
        "status": "running_phase",
        "project": "metaauto",
        "display": "MetaAuto",
        "goal": "почини metaauto импорты",
        "phase_index": 1,
        "phase_count": 2,
        "bashnya_chat_id": BASHNYA_CHAT_ID,
        "bashnya_thread_id": BASHNYA_THREAD_ID,
        "employee_chat_id": BASHNYA_CHAT_ID,
        "employee_thread_id": "31",
        "topic_name": "MetaAuto · Сотрудник",
    }
    runner._save_nox_phase_runs(
        {
            "run-a": {**base_state, "run_id": "run-a", "updated_at": "2026-05-19T12:00:00"},
            "run-b": {**base_state, "run_id": "run-b", "updated_at": "2026-05-19T12:01:00"},
        }
    )

    ambiguous_reply = await runner._maybe_handle_nox_worker_live_steer(event, "[Danya] " + event.text)

    assert "Нужен Даня" in ambiguous_reply
    assert "несколько активных runs" in ambiguous_reply
    runs = runner._load_nox_phase_runs()
    assert "amendments" not in runs["run-a"]
    assert "amendments" not in runs["run-b"]

    explicit_event = _make_employee_event("run-a: применяй только к первой фазе")
    explicit_reply = await runner._maybe_handle_nox_worker_live_steer(
        explicit_event,
        "[Danya] " + explicit_event.text,
    )

    assert "сохранил amendment" in explicit_reply
    runs = runner._load_nox_phase_runs()
    assert runs["run-a"]["amendments"][-1]["text"] == "run-a: применяй только к первой фазе"
    assert "amendments" not in runs["run-b"]


def test_worker_phase_prompt_includes_saved_live_amendments(bashnya_routes_path):
    """Saved live-steer amendments are injected into later worker prompts after restarts/gates."""
    event = _make_employee_event("placeholder")
    runner = _make_runner(event.source, CaptureAdapter())
    state = {
        "run_id": "run-with-amendment",
        "status": "running_phase",
        "project": "metaauto",
        "display": "MetaAuto",
        "goal": "почини metaauto импорты",
        "plan": "Фаза 1: диагностика",
        "phase_index": 1,
        "phase_count": 1,
        "amendments": [
            {
                "id": "a1",
                "created_at": "2026-05-19T12:00:00",
                "text": "не коммить; только report",
                "delivery": "saved_for_worker",
            }
        ],
    }

    prompt = runner._build_nox_worker_phase_prompt(state, 1)

    assert "Live amendments / user steering" in prompt
    assert "не коммить; только report" in prompt


@pytest.mark.asyncio
async def test_worker_plan_safe_scope_is_auto_approved_by_nox_pm(monkeypatch, bashnya_routes_path):
    """Safe worker plans should not bounce back to Danya for phase approval."""
    event = _make_bashnya_event("почини metaauto импорты и прогони тесты")
    adapter = CaptureAdapter()
    runner = _make_runner(event.source, adapter)
    run_id = "run-safe"
    state = {
        "run_id": run_id,
        "status": "planning",
        "project": "metaauto",
        "display": "MetaAuto",
        "goal": event.text,
        "phase_index": 0,
        "phase_count": 0,
        "bashnya_chat_id": BASHNYA_CHAT_ID,
        "bashnya_thread_id": BASHNYA_THREAD_ID,
        "employee_chat_id": BASHNYA_CHAT_ID,
        "employee_thread_id": "31",
        "topic_name": "MetaAuto · Сотрудник",
        "source_user_id": "8756671470",
        "source_user_name": "Danya",
    }

    async def fake_worker_plan(**_kwargs):
        return "Цель: safe fix\nФаза 1: диагностика\nФаза 2: фикс и targeted tests\nHuman gate: none"

    started: list[dict] = []

    def fake_start_phase_runner(*, run_id, route, state):
        started.append({"run_id": run_id, "route": route, "state": dict(state)})
        return "phase-task-id"

    runner._run_nox_worker_agent_once = fake_worker_plan
    runner._start_nox_worker_phase_runner = fake_start_phase_runner
    runner._save_nox_phase_runs({run_id: state})

    await runner._run_nox_worker_plan(run_id, {}, state, "plan-task-id")

    assert started, "Nox PM must start phase 1 without Danya for safe plans"
    assert started[0]["state"]["phase_index"] == 1
    runs = runner._load_nox_phase_runs()
    assert runs[run_id]["status"] == "running_phase"
    assert runs[run_id]["phase_index"] == 1
    assert runs[run_id]["approval_mode"] == "nox_pm_auto"
    assert any("auto-approved" in item["content"] for item in adapter.sent)


@pytest.mark.asyncio
async def test_worker_plan_high_risk_keeps_human_gate(monkeypatch, bashnya_routes_path):
    """High-risk plans still require Danya even when Nox PM auto-approval is enabled."""
    event = _make_bashnya_event("подготовь metaauto deploy production")
    adapter = CaptureAdapter()
    runner = _make_runner(event.source, adapter)
    run_id = "run-risk"
    state = {
        "run_id": run_id,
        "status": "planning",
        "project": "metaauto",
        "display": "MetaAuto",
        "goal": event.text,
        "phase_index": 0,
        "phase_count": 0,
        "bashnya_chat_id": BASHNYA_CHAT_ID,
        "bashnya_thread_id": BASHNYA_THREAD_ID,
        "employee_chat_id": BASHNYA_CHAT_ID,
        "employee_thread_id": "31",
        "topic_name": "MetaAuto · Сотрудник",
        "source_user_id": "8756671470",
        "source_user_name": "Danya",
    }

    async def fake_worker_plan(**_kwargs):
        return "Цель: release\nФаза 1: проверить diff\nФаза 2: deploy production\nHuman gate: deploy production"

    started: list[dict] = []
    runner._run_nox_worker_agent_once = fake_worker_plan
    runner._start_nox_worker_phase_runner = lambda **kwargs: started.append(kwargs) or "phase-task-id"
    runner._save_nox_phase_runs({run_id: state})

    await runner._run_nox_worker_plan(run_id, {}, state, "plan-task-id")

    assert started == []
    runs = runner._load_nox_phase_runs()
    assert runs[run_id]["status"] == "awaiting_approval"
    assert runs[run_id]["approval_mode"] == "human"
    assert "deploy production" in runs[run_id]["approval_reason"]
    assert any("Нужен Даня" in item["content"] for item in adapter.sent)


@pytest.mark.asyncio
async def test_employee_packet_send_happens_after_durable_state(monkeypatch, nox_phase_runs_path):
    """A visible employee packet must not exist without durable run state for restart recovery."""
    event = _make_bashnya_event("почини metaauto импорты и разбей работу на фазы")
    adapter = StateCheckingAdapter(nox_phase_runs_path)
    runner = _make_runner(event.source, adapter)

    ack = await runner._handle_message_with_agent(
        event,
        event.source,
        build_session_key(event.source),
        run_generation=1,
    )

    assert "Статус: взял в работу" in ack
    match = re.search(r"run_id: `([^`]+)`", ack)
    assert match
    run_id = match.group(1)
    runs = runner._load_nox_phase_runs()
    assert runs[run_id]["packet_message_id"] == "sent-1"
    assert runs[run_id]["status"] == "planning"


@pytest.mark.asyncio
async def test_unrelated_topic_approval_text_falls_through(bashnya_routes_path):
    """Exact approval words in unrelated Telegram topics must not be hijacked by the Nox gate."""
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=BASHNYA_CHAT_ID,
        thread_id="99",
        chat_type="supergroup",
        user_id="8756671470",
        user_name="Danya",
    )
    event = MessageEvent(
        text="утвердить",
        message_type=MessageType.TEXT,
        source=source,
        message_id="other-topic-msg-1",
    )
    runner = _make_runner(source, CaptureAdapter())

    assert await runner._maybe_handle_nox_worker_approval(event, event.text) is None


@pytest.mark.asyncio
async def test_approval_requires_exact_command_and_supports_hyphenated_run_id(bashnya_routes_path):
    """Safety gate accepts exact approve/run_id commands, not ambiguous negated phrases."""
    event = _make_employee_event("утвердить не запускай")
    adapter = CaptureAdapter()
    runner = _make_runner(event.source, adapter)
    base_state = {
        "status": "awaiting_approval",
        "project": "metaauto",
        "display": "MetaAuto",
        "goal": "почини metaauto импорты",
        "plan": "Фаза 1: диагностика",
        "phase_index": 0,
        "phase_count": 1,
        "bashnya_chat_id": BASHNYA_CHAT_ID,
        "bashnya_thread_id": BASHNYA_THREAD_ID,
        "employee_chat_id": BASHNYA_CHAT_ID,
        "employee_thread_id": "31",
        "topic_name": "MetaAuto · Сотрудник",
    }
    runner._save_nox_phase_runs(
        {
            "bashnya-20260518-111111-aaaa": {"run_id": "bashnya-20260518-111111-aaaa", **base_state},
            "bashnya-20260518-222222-bbbb": {"run_id": "bashnya-20260518-222222-bbbb", **base_state},
        }
    )
    started: list[dict] = []

    def fake_start_phase_runner(*, run_id, route, state):
        started.append({"run_id": run_id, "route": route, "state": dict(state)})
        return "phase-task-id"

    runner._start_nox_worker_phase_runner = fake_start_phase_runner

    assert await runner._maybe_handle_nox_worker_approval(event, event.text) is None
    assert started == []

    event.text = "утвердить bashnya-20260518-222222-bbbb"
    reply = await runner._maybe_handle_nox_worker_approval(event, event.text)

    assert "Фаза 1" in reply
    assert started[0]["run_id"] == "bashnya-20260518-222222-bbbb"
    runs = runner._load_nox_phase_runs()
    assert runs["bashnya-20260518-111111-aaaa"]["status"] == "awaiting_approval"
    assert runs["bashnya-20260518-222222-bbbb"]["status"] == "running_phase"
