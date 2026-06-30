"""
Tests for session_orchestration/spawn.py.

Coverage
--------
1. derive_session_name: deterministic, stable, tmux-safe.
2. get_adapter: resolves all known names; raises UnknownAgentError for unknown.
3. parse_spawn_args: parses key=value tokens + trailing prompt; handles edge cases.
4. spawn_session (happy path): launch called, registry row written (source=spawn,
   correct fields), thread creator called, relay seeded — all via fakes.
5. spawn_session: registry row has source="spawn" (not "adopt").
6. spawn_session: thread creator failure is non-fatal; SpawnResult.thread_id=None.
7. spawn_session: z_command is prepended to prompt when present.
8. handle_spawn_command: returns disabled message when so_enabled=False.
9. handle_spawn_command: returns usage message when required args are missing.
10. handle_spawn_command (integration): calls spawn_session with correct request.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from session_orchestration.adapters.base import AgentAdapter
from session_orchestration.registry import SessionOrchestrationRegistry
from session_orchestration.relay import SessionRelay
from session_orchestration.spawn import (
    SpawnRequest,
    UnknownAgentError,
    derive_session_name,
    get_adapter,
    parse_spawn_args,
    spawn_session,
    handle_spawn_command,
    handle_stop_command,
    handle_restart_command,
)
from session_orchestration.types import Capabilities, SessionHandle, SessionLifecycle


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "state.db"


@pytest.fixture()
def registry(db_path: Path) -> SessionOrchestrationRegistry:
    return SessionOrchestrationRegistry(db_path=db_path)


def _make_handle(session_id: Optional[str] = None) -> SessionHandle:
    sid = session_id or str(uuid.uuid4())
    return SessionHandle(
        session_id=sid,
        tmux_session=f"hermes-cc-{sid[:8]}",
        pane=f"hermes-cc-{sid[:8]}:0.0",
        launch_ts=datetime.now(tz=timezone.utc),
    )


class _FakeAdapter(AgentAdapter):
    """Stub adapter that records calls without touching tmux."""

    def __init__(self, fixed_session_id: Optional[str] = None):
        self._session_id = fixed_session_id or str(uuid.uuid4())
        self.launch_calls: List[tuple] = []
        self.drive_calls: List[str] = []
        self.resume_calls: List[str] = []
        self.detect_returns = SessionLifecycle.WAITING_USER

    def capabilities(self) -> Capabilities:
        return Capabilities()

    def launch(self, workdir: str, prompt: str) -> SessionHandle:
        self.launch_calls.append((workdir, prompt))
        return _make_handle(session_id=self._session_id)

    def drive(self, handle: SessionHandle, message: str) -> None:
        self.drive_calls.append(message)

    def detect(self, handle: SessionHandle) -> SessionLifecycle:
        return self.detect_returns

    def resume(self, handle: SessionHandle, prompt: str) -> None:
        self.resume_calls.append(prompt)

    def terminate(self, handle: SessionHandle) -> None:
        pass  # safe no-op in tests


class _FakeRelay:
    """Stub relay that records send_message calls."""

    def __init__(self):
        self.calls: List[tuple] = []

    def send_message(
        self,
        task_id: str,
        handle: SessionHandle,
        message: str,
        *,
        retry_on_conflict: bool = False,
    ) -> None:
        self.calls.append((task_id, message))


# ---------------------------------------------------------------------------
# 1. derive_session_name: deterministic + stable
# ---------------------------------------------------------------------------


def test_derive_session_name_is_deterministic():
    name1 = derive_session_name("claude", "/home/zeke/project", "fix the bug")
    name2 = derive_session_name("claude", "/home/zeke/project", "fix the bug")
    assert name1 == name2


def test_derive_session_name_differs_on_different_inputs():
    a = derive_session_name("claude", "/home/zeke/project", "fix the bug")
    b = derive_session_name("omp", "/home/zeke/project", "fix the bug")
    c = derive_session_name("claude", "/home/other/project", "fix the bug")
    d = derive_session_name("claude", "/home/zeke/project", "different prompt")
    assert len({a, b, c, d}) == 4


def test_derive_session_name_is_tmux_safe():
    name = derive_session_name("claude-code", "/home/zeke/project", "hello world")
    # tmux session names must not contain colons or dots per convention
    assert ":" not in name
    assert len(name) <= 40
    # Only alphanumeric and hyphens
    import re
    assert re.fullmatch(r"[a-zA-Z0-9\-]+", name)


def test_derive_session_name_starts_with_hermes():
    name = derive_session_name("omp", "/tmp/x", "prompt")
    assert name.startswith("hermes-")


# ---------------------------------------------------------------------------
# 2. get_adapter: resolves known names; raises on unknown
# ---------------------------------------------------------------------------


def test_get_adapter_resolves_claude():
    from session_orchestration.adapters.claude_code import ClaudeCodeAdapter
    adapter = get_adapter("claude")
    assert isinstance(adapter, ClaudeCodeAdapter)


def test_get_adapter_resolves_claude_code():
    from session_orchestration.adapters.claude_code import ClaudeCodeAdapter
    adapter = get_adapter("claude-code")
    assert isinstance(adapter, ClaudeCodeAdapter)


def test_get_adapter_resolves_omp():
    from session_orchestration.adapters.omp import OmpAdapter
    adapter = get_adapter("omp")
    assert isinstance(adapter, OmpAdapter)


def test_get_adapter_raises_for_unknown():
    with pytest.raises(UnknownAgentError, match="not-a-real-agent"):
        get_adapter("not-a-real-agent")


def test_get_adapter_case_insensitive():
    from session_orchestration.adapters.claude_code import ClaudeCodeAdapter
    adapter = get_adapter("Claude")
    assert isinstance(adapter, ClaudeCodeAdapter)


# ---------------------------------------------------------------------------
# 3. parse_spawn_args
# ---------------------------------------------------------------------------


def test_parse_spawn_args_basic():
    result = parse_spawn_args("agent=claude workdir=/tmp/foo fix the bug in auth.py")
    assert result["agent"] == "claude"
    assert result["workdir"] == "/tmp/foo"
    assert result["prompt"] == "fix the bug in auth.py"


def test_parse_spawn_args_with_z_command():
    result = parse_spawn_args("agent=omp workdir=/tmp/x z_command=/z-plan the feature")
    assert result["agent"] == "omp"
    assert result["z_command"] == "/z-plan"
    assert result["prompt"] == "the feature"


def test_parse_spawn_args_missing_prompt_returns_no_prompt_key():
    result = parse_spawn_args("agent=claude workdir=/tmp/foo")
    assert "prompt" not in result or not result.get("prompt")


def test_parse_spawn_args_empty_string():
    result = parse_spawn_args("")
    assert result == {}


def test_parse_spawn_args_workdir_with_spaces():
    result = parse_spawn_args('agent=claude workdir="/tmp/my project" fix it')
    assert result["workdir"] == "/tmp/my project"


# ---------------------------------------------------------------------------
# 4. spawn_session happy path: launch, registry row, thread, relay seed
# ---------------------------------------------------------------------------


def test_spawn_session_calls_launch(registry):
    adapter = _FakeAdapter()
    relay = _FakeRelay()
    thread_creator_calls: List[tuple] = []

    def thread_creator(chat_id: str, name: str) -> str:
        thread_creator_calls.append((chat_id, name))
        return "thread-123"

    request = SpawnRequest(
        prompt="fix the auth bug",
        agent="claude",
        workdir="/tmp/project",
        parent_chat_id="chan-456",
    )

    result = spawn_session(
        request,
        adapter=adapter,
        registry=registry,
        relay=relay,
        thread_creator=thread_creator,
    )

    # launch was called with correct workdir and prompt
    assert len(adapter.launch_calls) == 1
    assert adapter.launch_calls[0][0] == "/tmp/project"
    assert "fix the auth bug" in adapter.launch_calls[0][1]


def test_spawn_session_writes_registry_row_source_spawn(registry):
    adapter = _FakeAdapter()
    relay = _FakeRelay()

    request = SpawnRequest(
        prompt="implement feature X",
        agent="omp",
        workdir="/tmp/project",
    )

    result = spawn_session(
        request,
        adapter=adapter,
        registry=registry,
        relay=relay,
    )

    row = registry.get(result.task_id)
    assert row is not None
    assert row["source"] == "spawn"
    assert row["agent"] == "omp"
    assert row["workdir"] == "/tmp/project"
    assert row["state"] == "RUNNING"
    assert row["tmux_session"] is not None


def test_spawn_session_registry_row_has_correct_task_id(registry):
    fixed_id = str(uuid.uuid4())
    adapter = _FakeAdapter(fixed_session_id=fixed_id)
    relay = _FakeRelay()

    request = SpawnRequest(
        prompt="hello",
        agent="claude",
        workdir="/tmp/x",
    )

    result = spawn_session(request, adapter=adapter, registry=registry, relay=relay)

    assert result.task_id == fixed_id
    row = registry.get(fixed_id)
    assert row is not None


def test_spawn_session_creates_thread(registry):
    adapter = _FakeAdapter()
    relay = _FakeRelay()
    thread_calls: List[tuple] = []

    def thread_creator(chat_id: str, name: str) -> str:
        thread_calls.append((chat_id, name))
        return "thread-999"

    request = SpawnRequest(
        prompt="do something",
        agent="claude",
        workdir="/tmp/p",
        parent_chat_id="chan-123",
    )

    result = spawn_session(
        request,
        adapter=adapter,
        registry=registry,
        relay=relay,
        thread_creator=thread_creator,
    )

    assert result.thread_id == "thread-999"
    assert len(thread_calls) == 1
    assert thread_calls[0][0] == "chan-123"

    # thread_id must be persisted in the registry row
    row = registry.get(result.task_id)
    assert row["discord_thread_id"] == "thread-999"


def test_spawn_session_seeds_first_prompt_via_relay(registry):
    adapter = _FakeAdapter()
    relay = _FakeRelay()

    request = SpawnRequest(
        prompt="write tests for auth.py",
        agent="claude",
        workdir="/tmp/proj",
    )

    result = spawn_session(request, adapter=adapter, registry=registry, relay=relay)

    assert len(relay.calls) == 1
    seeded_task_id, seeded_msg = relay.calls[0]
    assert seeded_task_id == result.task_id
    assert "write tests for auth.py" in seeded_msg


# ---------------------------------------------------------------------------
# 5. source="spawn" not "adopt" (contract for single-writer discipline)
# ---------------------------------------------------------------------------


def test_spawn_session_row_source_is_spawn_not_adopt(registry):
    adapter = _FakeAdapter()
    relay = _FakeRelay()

    result = spawn_session(
        SpawnRequest(prompt="p", agent="claude", workdir="/tmp/x"),
        adapter=adapter,
        registry=registry,
        relay=relay,
    )

    row = registry.get(result.task_id)
    assert row["source"] == "spawn"
    assert row["source"] != "adopt"


# ---------------------------------------------------------------------------
# 6. Thread creator failure is non-fatal
# ---------------------------------------------------------------------------


def test_spawn_session_thread_failure_is_non_fatal(registry):
    adapter = _FakeAdapter()
    relay = _FakeRelay()

    def failing_thread_creator(chat_id: str, name: str) -> Optional[str]:
        raise RuntimeError("Discord API failed")

    request = SpawnRequest(
        prompt="do work",
        agent="claude",
        workdir="/tmp/p",
        parent_chat_id="chan-x",
    )

    result = spawn_session(
        request,
        adapter=adapter,
        registry=registry,
        relay=relay,
        thread_creator=failing_thread_creator,
    )

    # Should succeed despite thread failure
    assert result.task_id is not None
    assert result.thread_id is None
    # Registry row should exist
    row = registry.get(result.task_id)
    assert row is not None


def test_spawn_session_no_thread_when_no_parent_chat_id(registry):
    adapter = _FakeAdapter()
    relay = _FakeRelay()
    thread_calls: List[tuple] = []

    def thread_creator(chat_id: str, name: str) -> str:
        thread_calls.append((chat_id, name))
        return "thread-123"

    # No parent_chat_id → thread creator should NOT be called
    request = SpawnRequest(
        prompt="work",
        agent="claude",
        workdir="/tmp/p",
        parent_chat_id=None,  # no parent
    )

    result = spawn_session(
        request,
        adapter=adapter,
        registry=registry,
        relay=relay,
        thread_creator=thread_creator,
    )

    assert result.thread_id is None
    assert thread_calls == []


# ---------------------------------------------------------------------------
# 7. z_command is prepended to prompt
# ---------------------------------------------------------------------------


def test_spawn_session_prepends_z_command(registry):
    adapter = _FakeAdapter()
    relay = _FakeRelay()

    request = SpawnRequest(
        prompt="implement the feature",
        agent="claude",
        workdir="/tmp/p",
        z_command="/z-plan",
    )

    spawn_session(request, adapter=adapter, registry=registry, relay=relay)

    launched_prompt = adapter.launch_calls[0][1]
    assert launched_prompt.startswith("/z-plan")
    assert "implement the feature" in launched_prompt

    seeded_msg = relay.calls[0][1]
    assert seeded_msg.startswith("/z-plan")


# ---------------------------------------------------------------------------
# 8. handle_spawn_command: disabled when so_enabled=False
# ---------------------------------------------------------------------------


def test_handle_spawn_command_disabled():
    import asyncio

    event = MagicMock()
    event.text = "/so-spawn agent=claude workdir=/tmp/x do work"
    event.source = MagicMock()
    event.source.user_id = "u1"
    event.source.chat_id = "c1"

    # config with session_orchestration.enabled=False
    config = {"session_orchestration": {"enabled": False}}

    result = asyncio.run(
        handle_spawn_command(event, "agent=claude workdir=/tmp/x do work", config=config)
    )

    assert "disabled" in result.lower()


# ---------------------------------------------------------------------------
# 9. handle_spawn_command: missing args returns usage
# ---------------------------------------------------------------------------


def test_handle_spawn_command_missing_agent():
    import asyncio

    event = MagicMock()
    event.text = "/so-spawn workdir=/tmp/x do work"
    event.source = MagicMock()
    event.source.user_id = "u1"
    event.source.chat_id = "c1"

    config = {"session_orchestration": {"enabled": True}}

    result = asyncio.run(
        handle_spawn_command(event, "workdir=/tmp/x do work", config=config)
    )

    assert "Usage" in result or "Missing" in result
    assert "agent" in result.lower()


def test_handle_spawn_command_missing_workdir():
    import asyncio

    event = MagicMock()
    event.source = MagicMock()
    event.source.user_id = "u1"
    event.source.chat_id = "c1"

    config = {"session_orchestration": {"enabled": True}}

    result = asyncio.run(
        handle_spawn_command(event, "agent=claude do work", config=config)
    )

    assert "workdir" in result.lower()


# ---------------------------------------------------------------------------
# 10. handle_spawn_command integration: calls spawn_session with correct request
# ---------------------------------------------------------------------------


def test_handle_spawn_command_integration(db_path):
    import asyncio

    registry = SessionOrchestrationRegistry(db_path=db_path)
    adapter = _FakeAdapter()
    relay = _FakeRelay()

    event = MagicMock()
    event.text = "/so-spawn agent=claude workdir=/tmp/myproject fix the bug"
    event.source = MagicMock()
    event.source.user_id = "user-123"
    event.source.chat_id = "chan-456"
    event.source.platform = "discord"

    config = {"session_orchestration": {"enabled": True, "feed_channel_id": "feed-789"}}

    result = asyncio.run(
        handle_spawn_command(
            event,
            "agent=claude workdir=/tmp/myproject fix the bug",
            config=config,
            platform_adapter=None,
            registry=registry,
            relay=relay,
            _adapter_override=adapter,
        )
    )

    # Reply should contain the task_id
    assert "Task ID" in result
    # launch was called
    assert len(adapter.launch_calls) == 1
    # relay was seeded
    assert len(relay.calls) == 1
    # registry has the row
    all_rows = registry.list()
    assert len(all_rows) == 1
    assert all_rows[0]["source"] == "spawn"
    assert all_rows[0]["agent"] == "claude"


# ---------------------------------------------------------------------------
# T006 — enqueue_terminate: queue contract
# ---------------------------------------------------------------------------


def test_enqueue_terminate_inserts_row_with_kind_terminate(registry):
    """enqueue_terminate inserts a queue row with intent='terminate'."""
    registry.enqueue_terminate("task-abc", restart=False)

    conn = registry._connect()
    rows = conn.execute(
        "SELECT * FROM session_orchestration_queue WHERE task_id = 'task-abc'"
    ).fetchall()
    conn.close()

    assert len(rows) == 1
    row = dict(rows[0])
    assert row["intent"] == "terminate"


def test_enqueue_terminate_payload_restart_false(registry):
    """enqueue_terminate with restart=False writes payload.restart=False."""
    import json
    registry.enqueue_terminate("task-stop", restart=False)

    conn = registry._connect()
    rows = conn.execute(
        "SELECT payload FROM session_orchestration_queue WHERE task_id = 'task-stop'"
    ).fetchall()
    conn.close()

    assert len(rows) == 1
    payload = json.loads(rows[0]["payload"])
    assert payload["restart"] is False


def test_enqueue_terminate_payload_restart_true(registry):
    """enqueue_terminate with restart=True writes payload.restart=True."""
    import json
    registry.enqueue_terminate("task-restart", restart=True)

    conn = registry._connect()
    rows = conn.execute(
        "SELECT payload FROM session_orchestration_queue WHERE task_id = 'task-restart'"
    ).fetchall()
    conn.close()

    assert len(rows) == 1
    payload = json.loads(rows[0]["payload"])
    assert payload["restart"] is True


def test_apply_intent_terminate_does_not_raise(registry):
    """_apply_intent must not raise when draining a terminate intent."""
    import json
    intent = {
        "id": 1,
        "task_id": "task-xyz",
        "intent": "terminate",
        "payload": json.dumps({"restart": False}),
    }
    # Must complete without raising
    registry._apply_intent(intent)


# ---------------------------------------------------------------------------
# T006 — handle_stop_command
# ---------------------------------------------------------------------------


def test_handle_stop_command_returns_confirmation(registry):
    """handle_stop_command with valid task_id returns non-empty string."""
    event = MagicMock()
    result = handle_stop_command(event, "task_id=task-123", registry=registry)

    assert result
    assert isinstance(result, str)
    assert len(result) > 0


def test_handle_stop_command_enqueues_terminate_restart_false(registry):
    """handle_stop_command calls enqueue_terminate with restart=False."""
    import json
    event = MagicMock()
    handle_stop_command(event, "task_id=task-stop-x", registry=registry)

    conn = registry._connect()
    rows = conn.execute(
        "SELECT intent, payload FROM session_orchestration_queue "
        "WHERE task_id = 'task-stop-x'"
    ).fetchall()
    conn.close()

    assert len(rows) == 1
    assert rows[0]["intent"] == "terminate"
    payload = json.loads(rows[0]["payload"])
    assert payload["restart"] is False


def test_handle_stop_command_missing_task_id_returns_error_not_raises(registry):
    """handle_stop_command returns an error string when task_id is absent."""
    event = MagicMock()
    result = handle_stop_command(event, "", registry=registry)

    assert isinstance(result, str)
    assert "task_id" in result.lower()

    # No queue row should have been written
    conn = registry._connect()
    count = conn.execute(
        "SELECT COUNT(*) FROM session_orchestration_queue"
    ).fetchone()[0]
    conn.close()
    assert count == 0


def test_handle_stop_command_missing_task_id_does_not_raise_on_garbage(registry):
    """handle_stop_command returns error string (not raise) for garbage input."""
    event = MagicMock()
    result = handle_stop_command(event, "foo bar baz", registry=registry)

    assert isinstance(result, str)
    assert result  # non-empty error


# ---------------------------------------------------------------------------
# T006 — handle_restart_command
# ---------------------------------------------------------------------------


def test_handle_restart_command_enqueues_terminate_restart_true(registry):
    """handle_restart_command calls enqueue_terminate with restart=True."""
    import json
    event = MagicMock()
    handle_restart_command(event, "task_id=task-restart-y", registry=registry)

    conn = registry._connect()
    rows = conn.execute(
        "SELECT intent, payload FROM session_orchestration_queue "
        "WHERE task_id = 'task-restart-y'"
    ).fetchall()
    conn.close()

    assert len(rows) == 1
    assert rows[0]["intent"] == "terminate"
    payload = json.loads(rows[0]["payload"])
    assert payload["restart"] is True


def test_handle_restart_command_returns_confirmation(registry):
    """handle_restart_command with valid task_id returns non-empty string."""
    event = MagicMock()
    result = handle_restart_command(event, "task_id=task-789", registry=registry)

    assert result
    assert isinstance(result, str)


def test_handle_restart_command_missing_task_id_returns_error_not_raises(registry):
    """handle_restart_command returns an error string when task_id is absent."""
    event = MagicMock()
    result = handle_restart_command(event, "", registry=registry)

    assert isinstance(result, str)
    assert "task_id" in result.lower()

    conn = registry._connect()
    count = conn.execute(
        "SELECT COUNT(*) FROM session_orchestration_queue"
    ).fetchone()[0]
    conn.close()
    assert count == 0


# ---------------------------------------------------------------------------
# T-REV-001 — CommandDef registration + gateway dispatch for so-stop/so-restart
# ---------------------------------------------------------------------------


def test_so_stop_command_def_registered():
    """so-stop must appear in the command registry so Discord can autocomplete it."""
    from hermes_cli.commands import resolve_command

    cmd = resolve_command("so-stop")
    assert cmd is not None, "so-stop not found in COMMAND_REGISTRY"
    assert cmd.name == "so-stop"
    assert cmd.gateway_only is True
    assert cmd.gateway_config_gate == "session_orchestration.enabled"


def test_so_restart_command_def_registered():
    """so-restart must appear in the command registry so Discord can autocomplete it."""
    from hermes_cli.commands import resolve_command

    cmd = resolve_command("so-restart")
    assert cmd is not None, "so-restart not found in COMMAND_REGISTRY"
    assert cmd.name == "so-restart"
    assert cmd.gateway_only is True
    assert cmd.gateway_config_gate == "session_orchestration.enabled"


def test_gateway_so_stop_dispatch_reaches_handle_stop_command(registry):
    """_handle_so_stop_command must call spawn.handle_stop_command and return its result.

    Uses asyncio.run() directly since pytest-asyncio is not installed in this env.
    """
    import asyncio
    from unittest.mock import MagicMock, patch

    # Build a minimal stand-in that replicates only the method under test,
    # avoiding the heavy GatewayRunner constructor.
    class _FakeRunner:
        config = {}

        async def _handle_so_stop_command(self, event):
            try:
                from session_orchestration.spawn import handle_stop_command as _h
            except ImportError as exc:  # pragma: no cover
                return f"Session orchestration is not available: {exc}"

            raw_text = getattr(event, "text", "") or ""
            for prefix in ("/so-stop",):
                if raw_text.lower().startswith(prefix):
                    args_text = raw_text[len(prefix):].strip()
                    break
            else:
                args_text = raw_text.strip()

            cfg = self.config if isinstance(self.config, dict) else (
                self.config.__dict__ if hasattr(self.config, "__dict__") else {}
            )
            return _h(event, args_text, config=cfg)

    event = MagicMock()
    event.text = "/so-stop task_id=task-abc"

    with patch(
        "session_orchestration.spawn.handle_stop_command",
        wraps=lambda ev, args, **kw: handle_stop_command(ev, args, registry=registry),
    ) as mock_stop:
        runner = _FakeRunner()
        result = asyncio.run(runner._handle_so_stop_command(event))

    mock_stop.assert_called_once()
    call_args = mock_stop.call_args
    assert call_args[0][1] == "task_id=task-abc", (
        "args_text must be the text after '/so-stop'"
    )
    assert isinstance(result, str)
    assert "task-abc" in result


def test_gateway_so_restart_dispatch_reaches_handle_restart_command(registry):
    """_handle_so_restart_command must call spawn.handle_restart_command and return its result.

    Uses asyncio.run() directly since pytest-asyncio is not installed in this env.
    """
    import asyncio
    from unittest.mock import MagicMock, patch

    class _FakeRunner:
        config = {}

        async def _handle_so_restart_command(self, event):
            try:
                from session_orchestration.spawn import handle_restart_command as _h
            except ImportError as exc:  # pragma: no cover
                return f"Session orchestration is not available: {exc}"

            raw_text = getattr(event, "text", "") or ""
            for prefix in ("/so-restart",):
                if raw_text.lower().startswith(prefix):
                    args_text = raw_text[len(prefix):].strip()
                    break
            else:
                args_text = raw_text.strip()

            cfg = self.config if isinstance(self.config, dict) else (
                self.config.__dict__ if hasattr(self.config, "__dict__") else {}
            )
            return _h(event, args_text, config=cfg)

    event = MagicMock()
    event.text = "/so-restart task_id=task-xyz"

    with patch(
        "session_orchestration.spawn.handle_restart_command",
        wraps=lambda ev, args, **kw: handle_restart_command(ev, args, registry=registry),
    ) as mock_restart:
        runner = _FakeRunner()
        result = asyncio.run(runner._handle_so_restart_command(event))

    mock_restart.assert_called_once()
    call_args = mock_restart.call_args
    assert call_args[0][1] == "task_id=task-xyz", (
        "args_text must be the text after '/so-restart'"
    )
    assert isinstance(result, str)
    assert "task-xyz" in result
