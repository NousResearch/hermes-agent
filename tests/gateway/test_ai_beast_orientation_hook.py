"""Tests for the opt-in AI Beast orientation gateway hook.

The hook must stay disabled by default, preserve Hermes-owned commands, and only
delegate read-only orientation commands through an explicit local AI Beast root.
"""

from __future__ import annotations

import builtins
import importlib
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.hooks import HookRegistry
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


HOOK_MODULE = "gateway.ai_beast_orientation_hook"



def _write_fake_ai_beast_adapter(project_root: Path) -> None:
    package = project_root / "ai_beast_registry"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "loader.py").write_text(
        "from pathlib import Path\n"
        "class RegistryError(ValueError):\n"
        "    pass\n"
        "def load_registry(workspaces_path, bindings_path):\n"
        "    workspaces = Path(workspaces_path)\n"
        "    bindings = Path(bindings_path)\n"
        "    if not workspaces.exists() or not bindings.exists():\n"
        "        raise RegistryError('registry files missing')\n"
        "    return {'workspaces': workspaces, 'bindings': bindings}\n",
        encoding="utf-8",
    )
    (package / "telegram_adapter.py").write_text(
        "from .loader import RegistryError\n"
        "CALLS = []\n"
        "def handle_telegram_orientation_command(text, registry, *, chat_id=None, thread_id=None, bot_username=None, enable_beast_namespace=False):\n"
        "    CALLS.append((text, registry['workspaces'].name, registry['bindings'].name, chat_id, thread_id, bot_username))\n"
        "    if text == '/projects':\n"
        "        return 'Projects (read-only registry):\\n- AI Beast [ai-beast]'\n"
        "    if text == '/whereami':\n"
        "        return f'Workspace: AI Beast\\nchat={chat_id} thread={thread_id} bot={bot_username}'\n"
        "    raise RegistryError(f'unsupported command: {text}')\n",
        encoding="utf-8",
    )
    (package / "beast_namespace.py").write_text(
        "from dataclasses import dataclass\n"
        "import shlex\n"
        "class BeastParseStatus:\n"
        "    RECOGNISED = 'recognised'\n"
        "class BeastCommandClass:\n"
        "    READ_ONLY_ORIENTATION = 'read_only_orientation'\n"
        "    PROPOSAL_APPROVAL_GATED = 'proposal_approval_gated'\n"
        "    STATE_CHANGING_APPROVAL_GATED = 'state_changing_approval_gated'\n"
        "    UNKNOWN = 'unknown'\n"
        "@dataclass(frozen=True)\n"
        "class Parse:\n"
        "    status: str\n"
        "    command_class: str\n"
        "    subcommand: str | None = None\n"
        "    args: tuple[str, ...] = ()\n"
        "    is_read_only: bool = False\n"
        "    is_proposal_only: bool = False\n"
        "    is_state_changing: bool = False\n"
        "    requires_approval: bool = False\n"
        "    fail_closed: bool = True\n"
        "    writes_memory: bool = False\n"
        "    mutates_kanban: bool = False\n"
        "    invokes_durable_continuation: bool = False\n"
        "    activates_routing: bool = False\n"
        "    creates_binding: bool = False\n"
        "def parse_beast_command(text):\n"
        "    try:\n"
        "        parts = shlex.split(text)\n"
        "    except ValueError:\n"
        "        return Parse('invalid_syntax', 'unknown')\n"
        "    if len(parts) < 2:\n"
        "        return Parse('missing_subcommand', 'unknown')\n"
        "    sub = parts[1]\n"
        "    args = tuple(parts[2:])\n"
        "    if sub in {'whereami', 'projects', 'topicstatus'} and not args:\n"
        "        return Parse('recognised', 'read_only_orientation', sub, args, True, False, False, False, False)\n"
        "    if sub == 'sessions' and len(args) == 1:\n"
        "        return Parse('recognised', 'read_only_orientation', sub, args, True, False, False, False, False)\n"
        "    if sub == 'bindtopic' and len(args) == 1:\n"
        "        return Parse('recognised', 'proposal_approval_gated', sub, args, False, True, False, True, False)\n"
        "    if sub in {'unbindtopic', 'inbox'} and not args:\n"
        "        return Parse('recognised', 'state_changing_approval_gated', sub, args, False, False, True, True, False)\n"
        "    if sub in {'pause', 'resume', 'cancel', 'open', 'switch'} and len(args) == 1:\n"
        "        return Parse('recognised', 'state_changing_approval_gated', sub, args, False, False, True, True, False)\n"
        "    if sub in {'task', 'steer', 'move', 'newsession'} and len(args) == 2:\n"
        "        return Parse('recognised', 'state_changing_approval_gated', sub, args, False, False, True, True, False)\n"
        "    return Parse('unknown_subcommand', 'unknown', sub, args)\n",
        encoding="utf-8",
    )


def _write_fake_registry(project_root: Path) -> None:
    registry_root = project_root / "docs" / "interaction-layer" / "registry"
    registry_root.mkdir(parents=True)
    (registry_root / "workspaces.json").write_text('{"workspaces": [], "projects": []}\n', encoding="utf-8")
    (registry_root / "bindings.json").write_text('{"bindings": []}\n', encoding="utf-8")


def _clear_fake_ai_beast_modules() -> None:
    for name in list(sys.modules):
        if name == "ai_beast_registry" or name.startswith("ai_beast_registry."):
            sys.modules.pop(name, None)


def _fake_ai_beast_adapter_calls() -> list:
    adapter = importlib.import_module("ai_beast_registry.telegram_adapter")
    return adapter.CALLS


def _load_hook_module():
    """Load the future hook module.

    RED expectation: this currently raises ModuleNotFoundError because the
    Hermes-side AI Beast orientation hook has not been implemented yet.
    """
    return importlib.import_module(HOOK_MODULE)


def _context(
    command: str,
    *,
    project_root: Path | None = None,
    raw_args: str = "",
    beast_namespace_enabled: bool | None = None,
) -> dict:
    config: dict[str, object] = {
        "enabled": project_root is not None,
    }
    if project_root is not None:
        config["project_root"] = str(project_root)
    if beast_namespace_enabled is not None:
        config["beast_namespace_enabled"] = beast_namespace_enabled
    return {
        "command": command,
        "raw_args": raw_args,
        "message": f"/{command} {raw_args}".strip(),
        "source": SimpleNamespace(
            platform="telegram",
            chat_id="chat-1",
            thread_id="thread-1",
            user_id="user-1",
        ),
        "gateway_config": SimpleNamespace(ai_beast_orientation=config),
    }


def test_ai_beast_orientation_hook_module_loads_for_green_slice():
    hook = _load_hook_module()

    assert hasattr(hook, "handle")


@pytest.mark.asyncio
async def test_ai_beast_orientation_whereami_requires_explicit_enabled_root(tmp_path):
    hook = _load_hook_module()
    fake_adapter = Mock(return_value="AI Beast: workspace orientation")

    result = await hook.handle(
        "command:whereami",
        _context("whereami", project_root=tmp_path),
        orientation_adapter=fake_adapter,
    )

    assert result == {
        "decision": "handled",
        "message": "AI Beast: workspace orientation",
    }
    fake_adapter.assert_called_once()




@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("whereami", "Workspace: AI Beast"),
        ("projects", "Projects (read-only registry):"),
    ],
)
async def test_ai_beast_orientation_lazily_wires_read_only_adapter_from_explicit_root(
    tmp_path, monkeypatch, command, expected
):
    hook = _load_hook_module()
    _write_fake_ai_beast_adapter(tmp_path)
    _write_fake_registry(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    _clear_fake_ai_beast_modules()

    result = await hook.handle(
        f"command:{command}",
        _context(command, project_root=tmp_path),
    )

    assert result["decision"] == "handled"
    assert expected in result["message"]


@pytest.mark.asyncio
async def test_ai_beast_orientation_lazy_adapter_uses_gateway_context_without_side_effects(tmp_path, monkeypatch):
    hook = _load_hook_module()
    _write_fake_ai_beast_adapter(tmp_path)
    _write_fake_registry(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    _clear_fake_ai_beast_modules()

    result = await hook.handle(
        "command:whereami",
        _context("whereami", project_root=tmp_path),
        side_effects={
            "memory_write": Mock(side_effect=AssertionError("memory write called")),
            "kanban_mutation": Mock(side_effect=AssertionError("Kanban mutation called")),
            "durable_continuation": Mock(side_effect=AssertionError("durable continuation called")),
            "binding_write": Mock(side_effect=AssertionError("binding write called")),
            "telegram_send": Mock(side_effect=AssertionError("live Telegram send called")),
        },
    )

    assert result["decision"] == "handled"
    assert "chat=chat-1 thread=thread-1" in result["message"]


@pytest.mark.asyncio
async def test_ai_beast_orientation_lazy_adapter_fails_closed_without_registry(tmp_path, monkeypatch):
    hook = _load_hook_module()
    _write_fake_ai_beast_adapter(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    _clear_fake_ai_beast_modules()

    result = await hook.handle(
        "command:whereami",
        _context("whereami", project_root=tmp_path),
    )

    assert result["decision"] == "deny"
    assert "AI Beast orientation adapter failed" in result["message"]


@pytest.mark.asyncio
async def test_ai_beast_orientation_lazy_import_does_not_pollute_sys_modules(tmp_path, monkeypatch):
    hook = _load_hook_module()
    _write_fake_ai_beast_adapter(tmp_path)
    _write_fake_registry(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    _clear_fake_ai_beast_modules()

    result = await hook.handle(
        "command:whereami",
        _context("whereami", project_root=tmp_path),
    )

    assert result["decision"] == "handled"
    assert not any(
        name == "ai_beast_registry" or name.startswith("ai_beast_registry.")
        for name in sys.modules
    )


@pytest.mark.asyncio
async def test_ai_beast_orientation_rejects_registry_file_symlink_escape(tmp_path, monkeypatch):
    hook = _load_hook_module()
    _write_fake_ai_beast_adapter(tmp_path)
    _write_fake_registry(tmp_path)
    outside_file = tmp_path.parent / "outside-workspaces.json"
    outside_file.write_text('{"workspaces": [], "projects": []}\n', encoding="utf-8")
    registry_root = tmp_path / "docs" / "interaction-layer" / "registry"
    (registry_root / "workspaces.json").unlink()
    (registry_root / "workspaces.json").symlink_to(outside_file)
    monkeypatch.syspath_prepend(str(tmp_path))
    _clear_fake_ai_beast_modules()

    result = await hook.handle(
        "command:whereami",
        _context("whereami", project_root=tmp_path),
    )

    assert result["decision"] == "deny"
    assert "AI Beast orientation adapter failed" in result["message"]


@pytest.mark.asyncio
async def test_ai_beast_orientation_is_disabled_by_default(tmp_path):
    hook = _load_hook_module()
    fake_adapter = Mock(side_effect=AssertionError("disabled hook called adapter"))

    result = await hook.handle(
        "command:whereami",
        _context("whereami", project_root=None),
        orientation_adapter=fake_adapter,
    )

    assert result is None
    fake_adapter.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["status", "sessions"])
async def test_ai_beast_orientation_cannot_hijack_hermes_owned_commands(tmp_path, command):
    hook = _load_hook_module()
    fake_adapter = Mock(side_effect=AssertionError(f"/{command} was hijacked"))

    result = await hook.handle(
        f"command:{command}",
        _context(command, project_root=tmp_path),
        orientation_adapter=fake_adapter,
    )

    assert result is None
    fake_adapter.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    ["task", "steer", "pause", "resume", "bindtopic", "switch", "open", "newsession"],
)
async def test_ai_beast_orientation_forbidden_commands_remain_unavailable(tmp_path, command):
    hook = _load_hook_module()
    fake_adapter = Mock(side_effect=AssertionError(f"forbidden /{command} dispatched"))

    result = await hook.handle(
        f"command:{command}",
        _context(command, project_root=tmp_path),
        orientation_adapter=fake_adapter,
    )

    assert result is None
    fake_adapter.assert_not_called()


@pytest.mark.asyncio
async def test_ai_beast_orientation_rejects_invalid_explicit_root(tmp_path):
    hook = _load_hook_module()
    missing_root = tmp_path / "missing-ai-beast-root"
    fake_adapter = Mock(side_effect=AssertionError("invalid root reached adapter"))

    result = await hook.handle(
        "command:whereami",
        _context("whereami", project_root=missing_root),
        orientation_adapter=fake_adapter,
    )

    assert result == {
        "decision": "deny",
        "message": "AI Beast orientation root is not available.",
    }
    fake_adapter.assert_not_called()


@pytest.mark.asyncio
async def test_ai_beast_orientation_hook_does_not_call_forbidden_side_effects(tmp_path):
    hook = _load_hook_module()
    side_effects = {
        "memory_write": Mock(side_effect=AssertionError("memory write called")),
        "kanban_mutation": Mock(side_effect=AssertionError("Kanban mutation called")),
        "durable_continuation": Mock(side_effect=AssertionError("durable continuation called")),
        "binding_write": Mock(side_effect=AssertionError("binding write called")),
        "smart_routing": Mock(side_effect=AssertionError("smart routing called")),
        "inbox_persistence": Mock(side_effect=AssertionError("inbox persistence called")),
        "telegram_send": Mock(side_effect=AssertionError("live Telegram send called")),
    }
    fake_adapter = Mock(return_value="AI Beast: read-only orientation")

    result = await hook.handle(
        "command:whereami",
        _context("whereami", project_root=tmp_path),
        orientation_adapter=fake_adapter,
        side_effects=side_effects,
    )

    assert result["decision"] == "handled"
    for forbidden_call in side_effects.values():
        forbidden_call.assert_not_called()


@pytest.mark.asyncio
async def test_ai_beast_beast_namespace_is_disabled_by_default(tmp_path):
    hook = _load_hook_module()
    _write_fake_ai_beast_adapter(tmp_path)
    _write_fake_registry(tmp_path)

    result = await hook.handle(
        "command:beast",
        _context("beast", project_root=tmp_path, raw_args="whereami"),
    )

    assert result is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw_args", "expected_fragments"),
    [
        ("whereami", ("/beast whereami", "read_only_orientation", "read-only")),
        ("projects", ("/beast projects", "read_only_orientation", "read-only")),
        (
            "sessions interaction-routing-layer",
            ("/beast sessions", "argument_count=1", "read_only_orientation", "read-only metadata"),
        ),
        (
            "bindtopic interaction-routing-layer",
            ("/beast bindtopic", "argument_count=1", "proposal_approval_gated", "approval-gated only"),
        ),
        (
            'task interaction-routing-layer "test task"',
            ("/beast task", "state_changing_approval_gated", "not executed"),
        ),
        (
            'steer card123 "test steer"',
            ("/beast steer", "state_changing_approval_gated", "not executed"),
        ),
    ],
)
async def test_ai_beast_beast_namespace_enabled_fixture_classifies_without_execution(
    tmp_path, monkeypatch, raw_args, expected_fragments
):
    hook = _load_hook_module()
    _write_fake_ai_beast_adapter(tmp_path)
    _write_fake_registry(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    _clear_fake_ai_beast_modules()
    side_effects = {
        "memory_write": Mock(side_effect=AssertionError("memory write called")),
        "kanban_mutation": Mock(side_effect=AssertionError("Kanban mutation called")),
        "durable_continuation": Mock(side_effect=AssertionError("durable continuation called")),
        "binding_write": Mock(side_effect=AssertionError("binding write called")),
        "smart_routing": Mock(side_effect=AssertionError("smart routing called")),
        "inbox_persistence": Mock(side_effect=AssertionError("inbox persistence called")),
        "audit_write": Mock(side_effect=AssertionError("audit write called")),
        "telegram_send": Mock(side_effect=AssertionError("live Telegram send called")),
    }

    result = await hook.handle(
        "command:beast",
        _context(
            "beast",
            project_root=tmp_path,
            raw_args=raw_args,
            beast_namespace_enabled=True,
        ),
        side_effects=side_effects,
    )

    assert result["decision"] == "handled"
    for expected in expected_fragments:
        assert expected in result["message"]
    for forbidden_call in side_effects.values():
        forbidden_call.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw_args",
    [
        'task interaction-routing-layer "test task"',
        'steer card123 "test steer"',
        "pause card123",
        "resume card123",
        "cancel card123",
        "unbindtopic",
        "move card123 lane2",
        "inbox",
        "open card123",
        "switch interaction-routing-layer",
        "newsession interaction-routing-layer focused-session",
    ],
)
async def test_ai_beast_beast_namespace_state_changing_approved_list_is_classified_only(
    tmp_path, monkeypatch, raw_args
):
    hook = _load_hook_module()
    _write_fake_ai_beast_adapter(tmp_path)
    _write_fake_registry(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    _clear_fake_ai_beast_modules()

    result = await hook.handle(
        "command:beast",
        _context(
            "beast",
            project_root=tmp_path,
            raw_args=raw_args,
            beast_namespace_enabled=True,
        ),
    )

    assert result["decision"] == "handled"
    assert "state_changing_approval_gated" in result["message"]
    assert "not executed" in result["message"]


@pytest.mark.asyncio
@pytest.mark.parametrize("raw_args", ["", "unknown", "topicstatus", 'whereami "unterminated'])
async def test_ai_beast_beast_namespace_enabled_fixture_fails_closed_for_bad_input(
    tmp_path, monkeypatch, raw_args
):
    hook = _load_hook_module()
    _write_fake_ai_beast_adapter(tmp_path)
    _write_fake_registry(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    _clear_fake_ai_beast_modules()

    result = await hook.handle(
        "command:beast",
        _context(
            "beast",
            project_root=tmp_path,
            raw_args=raw_args,
            beast_namespace_enabled=True,
        ),
    )

    assert result["decision"] == "deny"
    assert "fail-closed" in result["message"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw_args",
    [
        "sessions SECRET_TOKEN_123",
        "bindtopic SECRET_TOKEN_123",
        'task interaction-routing-layer "SECRET_TOKEN_123"',
    ],
)
async def test_ai_beast_beast_namespace_recognised_input_does_not_echo_raw_args(tmp_path, monkeypatch, raw_args):
    hook = _load_hook_module()
    _write_fake_ai_beast_adapter(tmp_path)
    _write_fake_registry(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    _clear_fake_ai_beast_modules()

    result = await hook.handle(
        "command:beast",
        _context(
            "beast",
            project_root=tmp_path,
            raw_args=raw_args,
            beast_namespace_enabled=True,
        ),
    )

    assert result["decision"] == "handled"
    assert "SECRET_TOKEN_123" not in result["message"]
    assert "secret_token_123" not in result["message"].lower()
    assert "argument_count=" in result["message"]


def test_ai_beast_beast_namespace_formatter_final_deny_uses_safe_label():
    hook = _load_hook_module()

    result = hook._format_beast_namespace_result(
        SimpleNamespace(
            status="recognised",
            command_class="read_only_orientation",
            subcommand="sessions",
            args=("SECRET_TOKEN_123",),
            is_read_only=False,
            is_proposal_only=False,
            is_state_changing=False,
        )
    )

    assert result["decision"] == "deny"
    assert "fail-closed" in result["message"]
    assert "SECRET_TOKEN_123" not in result["message"]
    assert "argument_count=1" in result["message"]


@pytest.mark.asyncio
@pytest.mark.parametrize("raw_args", ["", "task"])
async def test_ai_beast_beast_namespace_final_deny_fails_closed_without_crash(tmp_path, monkeypatch, raw_args):
    hook = _load_hook_module()
    _write_fake_ai_beast_adapter(tmp_path)
    _write_fake_registry(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    _clear_fake_ai_beast_modules()

    result = await hook.handle(
        "command:beast",
        _context(
            "beast",
            project_root=tmp_path,
            raw_args=raw_args,
            beast_namespace_enabled=True,
        ),
    )

    assert result["decision"] == "deny"
    assert "fail-closed" in result["message"]
    assert "No command behaviour executed" in result["message"]


@pytest.mark.asyncio
@pytest.mark.parametrize("raw_args", ["unknown SECRET_TOKEN_123", "SECRET_TOKEN_123"])
async def test_ai_beast_beast_namespace_unknown_input_does_not_echo_raw_text(tmp_path, monkeypatch, raw_args):
    hook = _load_hook_module()
    _write_fake_ai_beast_adapter(tmp_path)
    _write_fake_registry(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    _clear_fake_ai_beast_modules()

    result = await hook.handle(
        "command:beast",
        _context(
            "beast",
            project_root=tmp_path,
            raw_args=raw_args,
            beast_namespace_enabled=True,
        ),
    )

    assert result["decision"] == "deny"
    assert "fail-closed" in result["message"]
    assert "SECRET_TOKEN_123" not in result["message"]
    assert "secret_token_123" not in result["message"].lower()
    assert "/beast unknown SECRET_TOKEN_123" not in result["message"]
    assert "argument_count=" in result["message"]


def _make_gateway_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        user_id="user-1",
        user_name="tester",
        chat_type="dm",
        thread_id="thread-1",
    )


def _make_gateway_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_gateway_source(), message_id="message-1")


def _make_gateway_runner_with_real_orientation_hook(
    project_root: Path,
    *,
    command_hook_commands: dict[str, object] | None = None,
    beast_namespace_enabled: bool = False,
):
    from gateway.run import GatewayRunner

    orientation_config: dict[str, object] = {
        "enabled": True,
        "project_root": str(project_root),
        "bot_username": "hermes_test_bot",
    }
    if beast_namespace_enabled:
        orientation_config["beast_namespace_enabled"] = True
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")},
        command_hook_commands=command_hook_commands or {"whereami": {}, "projects": {}},
        ai_beast_orientation=orientation_config,
    )
    adapter = MagicMock()
    adapter.send = AsyncMock(side_effect=AssertionError("live platform send called"))
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = HookRegistry()
    runner.hooks.discover_and_load()

    session_entry = SessionEntry(
        session_key=build_session_key(_make_gateway_source()),
        session_id="session-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    runner._run_agent = AsyncMock(side_effect=AssertionError("AI Beast command leaked to agent"))
    return runner


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("/whereami", "Workspace: AI Beast"),
        ("/projects", "Projects (read-only registry):"),
    ],
)
async def test_ai_beast_orientation_gateway_command_path_uses_real_hook_without_agent_loop(
    tmp_path, monkeypatch, command, expected
):
    _write_fake_ai_beast_adapter(tmp_path)
    _write_fake_registry(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    _clear_fake_ai_beast_modules()
    runner = _make_gateway_runner_with_real_orientation_hook(tmp_path)

    result = await runner._handle_message(_make_gateway_event(command))

    assert result is not None
    assert expected in result
    if command == "/whereami":
        assert "chat=chat-1 thread=thread-1" in result
        assert "bot=hermes_test_bot" in result
    runner._run_agent.assert_not_called()
    runner.adapters[Platform.TELEGRAM].send.assert_not_called()


@pytest.mark.asyncio
async def test_ai_beast_beast_gateway_path_is_disabled_by_default_even_when_hook_command_configured(
    tmp_path, monkeypatch
):
    _write_fake_ai_beast_adapter(tmp_path)
    _write_fake_registry(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    _clear_fake_ai_beast_modules()
    runner = _make_gateway_runner_with_real_orientation_hook(
        tmp_path,
        command_hook_commands={"whereami": {}, "projects": {}, "beast": {}},
        beast_namespace_enabled=False,
    )

    result = await runner._handle_message(_make_gateway_event("/beast whereami"))

    assert result is not None
    assert "registered for command hooks" in result
    assert "read_only_orientation" not in result
    runner._run_agent.assert_not_called()
    runner.adapters[Platform.TELEGRAM].send.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("/beast whereami", "read_only_orientation"),
        ("/beast projects", "read_only_orientation"),
        ("/beast sessions interaction-routing-layer", "read-only metadata"),
        ("/beast unknown", "fail-closed"),
        ("/beast", "fail-closed"),
        ("/beast task interaction-routing-layer test-task", "state_changing_approval_gated"),
        ("/beast bindtopic interaction-routing-layer", "proposal_approval_gated"),
    ],
)
async def test_ai_beast_beast_gateway_fixture_enabled_path_handles_without_agent_or_live_send(
    tmp_path, monkeypatch, message, expected
):
    _write_fake_ai_beast_adapter(tmp_path)
    _write_fake_registry(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    _clear_fake_ai_beast_modules()
    runner = _make_gateway_runner_with_real_orientation_hook(
        tmp_path,
        command_hook_commands={"whereami": {}, "projects": {}, "beast": {}},
        beast_namespace_enabled=True,
    )

    result = await runner._handle_message(_make_gateway_event(message))

    assert result is not None
    assert expected in result
    runner._run_agent.assert_not_called()
    runner.adapters[Platform.TELEGRAM].send.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/status", "/sessions"])
async def test_ai_beast_orientation_gateway_fixture_preserves_hermes_owned_commands(
    tmp_path, monkeypatch, command
):
    _write_fake_ai_beast_adapter(tmp_path)
    _write_fake_registry(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    _clear_fake_ai_beast_modules()
    runner = _make_gateway_runner_with_real_orientation_hook(tmp_path)
    runner._running_agents[build_session_key(_make_gateway_source())] = MagicMock()

    result = await runner._handle_message(_make_gateway_event(command))

    assert result is not None
    assert "Workspace: AI Beast" not in result
    assert "Projects (read-only registry):" not in result
    assert _fake_ai_beast_adapter_calls() == []
    runner._run_agent.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        "/task",
        "/steer",
        "/pause",
        "/resume",
        "/bindtopic",
        "/switch",
        "/open",
        "/newsession",
        "/steer hi",
        "/task create something",
        "/bindtopic interaction-routing-layer",
        "/switch ai-beast",
        "/Steer hi",
        "/TASK create something",
    ],
)
async def test_ai_beast_orientation_gateway_fixture_keeps_forbidden_commands_unavailable(
    tmp_path, monkeypatch, command
):
    _write_fake_ai_beast_adapter(tmp_path)
    _write_fake_registry(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    _clear_fake_ai_beast_modules()
    runner = _make_gateway_runner_with_real_orientation_hook(tmp_path)

    result = await runner._handle_message(_make_gateway_event(command))

    assert result is not None
    assert "Workspace: AI Beast" not in result
    assert "Projects (read-only registry):" not in result
    assert "AI Beast" not in result
    assert _fake_ai_beast_adapter_calls() == []
    runner._run_agent.assert_not_called()
    runner.adapters[Platform.TELEGRAM].send.assert_not_called()


@pytest.mark.asyncio
async def test_ai_beast_orientation_forbidden_boundary_fails_closed_when_command_list_import_fails(
    tmp_path, monkeypatch
):
    _write_fake_ai_beast_adapter(tmp_path)
    _write_fake_registry(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    _clear_fake_ai_beast_modules()
    runner = _make_gateway_runner_with_real_orientation_hook(tmp_path)
    real_import = builtins.__import__

    def fail_forbidden_command_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "gateway.ai_beast_orientation_hook" and "FORBIDDEN_COMMANDS" in fromlist:
            raise ImportError("simulated forbidden command list import failure")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_forbidden_command_import)

    result = await runner._handle_message(_make_gateway_event("/steer hi"))

    assert result is not None
    assert "Workspace: AI Beast" not in result
    assert "Projects (read-only registry):" not in result
    assert "AI Beast" not in result
    assert _fake_ai_beast_adapter_calls() == []
    runner._run_agent.assert_not_called()
    runner.adapters[Platform.TELEGRAM].send.assert_not_called()
