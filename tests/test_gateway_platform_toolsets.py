from unittest.mock import AsyncMock, MagicMock, patch
import asyncio


def _make_runner():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {}
    runner._running_agents = {}
    runner._pending_messages = {}
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = MagicMock(
        session_id="sess-1234567890",
        created_at=MagicMock(strftime=lambda fmt: "2026-03-20 16:30"),
        updated_at=MagicMock(strftime=lambda fmt: "2026-03-20 16:31"),
        total_tokens=42,
        session_key="telegram:123",
    )
    runner._is_user_authorized = MagicMock(return_value=True)
    return runner


def _make_event():
    event = MagicMock()
    event.source.platform.value = "telegram"
    event.source.platform.name = "TELEGRAM"
    event.source.chat_type = "dm"
    event.source.chat_id = "123"
    event.source.thread_id = None
    event.source.user_id = "123"
    event.source.user_id_alt = None
    return event


def test_resolve_platform_toolsets_readds_required_core_toolset():
    from gateway.config import Platform

    runner = _make_runner()
    with patch.object(runner, "_load_platform_toolsets_config", return_value={"telegram": ["file", "terminal"]}):
        toolsets, meta = runner._resolve_platform_toolsets(Platform.TELEGRAM)

    assert toolsets[0] == "hermes-telegram"
    assert "file" in toolsets
    assert "terminal" in toolsets
    assert meta["forced"] == "hermes-telegram"


def test_resolve_platform_toolsets_ignores_invalid_entries():
    from gateway.config import Platform

    runner = _make_runner()
    with patch.object(runner, "_load_platform_toolsets_config", return_value={"telegram": ["bogus", "terminal"]}):
        toolsets, meta = runner._resolve_platform_toolsets(Platform.TELEGRAM)

    assert "bogus" not in toolsets
    assert meta["invalid"] == ["bogus"]
    assert "terminal" in toolsets


@patch(
    "gateway.status.read_runtime_status",
    return_value={
        "gateway_state": "running",
        "updated_at": "2026-03-20T21:30:00Z",
        "platforms": {"telegram": {"state": "connected"}},
    },
)
def test_health_command_reports_runtime_state(_mock_status):
    runner = _make_runner()
    event = _make_event()

    with patch.object(runner, "_resolve_platform_toolsets", return_value=(["hermes-telegram", "terminal"], {"invalid": [], "forced": None})), \
         patch.object(runner, "_get_available_tools_for_platform", return_value=["terminal", "read_file"]):
        result = asyncio.run(runner._handle_health_command(event))

    assert "gateway=running" in result
    assert "running" in result
    assert "connected" in result
    assert "2 tools" in result


def test_tools_command_lists_tool_surface():
    runner = _make_runner()
    event = _make_event()

    with patch.object(runner, "_resolve_platform_toolsets", return_value=(["hermes-telegram", "terminal"], {"invalid": ["bogus"], "forced": "hermes-telegram"})), \
         patch.object(runner, "_get_available_tools_for_platform", return_value=["terminal", "read_file", "write_file"]):
        result = asyncio.run(runner._handle_tools_command(event))

    assert "3 tools" in result
    assert "terminal, read_file, write_file" in result
    assert "invalid: bogus" in result


def test_write_runtime_status_clears_stale_errors_when_connected(tmp_path, monkeypatch):
    from gateway import status as status_mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    status_mod.write_runtime_status(
        gateway_state="starting",
        exit_reason="boom",
        platform="telegram",
        platform_state="error",
        error_code="telegram_connect_error",
        error_message="dns failed",
    )

    status_mod.write_runtime_status(
        gateway_state="running",
        platform="telegram",
        platform_state="connected",
    )

    payload = status_mod.read_runtime_status()
    assert payload["gateway_state"] == "running"
    assert payload["exit_reason"] == "boom"
    assert payload["platforms"]["telegram"]["state"] == "connected"
    assert payload["platforms"]["telegram"]["error_code"] == "telegram_connect_error"
    assert payload["platforms"]["telegram"]["error_message"] == "dns failed"


def test_cwd_command_reports_backend_and_directory(monkeypatch):
    runner = _make_runner()
    event = _make_event()
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", "/home/carlos/Projects")
    monkeypatch.setenv("TERMINAL_TIMEOUT", "180")
    monkeypatch.setenv("TERMINAL_LIFETIME_SECONDS", "300")

    result = asyncio.run(runner._handle_cwd_command(event))
    assert "📂" in result
    assert "/home/carlos/Projects" in result
    assert "local" in result


def test_limits_command_reports_session_limits(monkeypatch):
    runner = _make_runner()
    event = _make_event()
    monkeypatch.setenv("HERMES_MAX_ITERATIONS", "90")
    monkeypatch.setenv("TERMINAL_TIMEOUT", "180")

    with patch.object(runner, "_get_available_tools_for_platform", return_value=["terminal", "read_file"]), \
         patch.object(runner, "_load_reasoning_config", return_value={"enabled": True, "effort": "medium"}):
        result = asyncio.run(runner._handle_limits_command(event))

    assert "max_iterations=90" in result
    assert "reasoning=medium" in result


def test_models_command_shortcuts_to_model_listing():
    runner = _make_runner()
    event = _make_event()
    event.text = "/models"
    event.get_command.return_value = "models"
    event.get_command_args.return_value = ""
    event.source.user_id = "123"

    async def fake_model_handler(passed_event):
        return f"handled {passed_event.text}"

    runner._handle_model_command = fake_model_handler
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner._pending_messages = {}
    runner.adapters = {}

    async def _run():
        return await runner._handle_message(event)

    result = asyncio.run(_run())
    assert result == "handled /models"


def test_tail_trace_entries_filters_by_session(tmp_path):
    runner = _make_runner()
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True)
    trace_path = log_dir / "gateway-trace.jsonl"
    trace_path.write_text(
        "\n".join([
            '{"session_key":"telegram:dm:123","stage":"inbound_received","ts":"2026-03-20T21:00:00Z"}',
            '{"session_key":"other","stage":"ignored","ts":"2026-03-20T21:00:01Z"}',
            '{"session_key":"telegram:dm:123","stage":"agent_end","ts":"2026-03-20T21:00:02Z"}',
        ]),
        encoding="utf-8",
    )

    with patch("gateway.run._hermes_home", tmp_path):
        items = runner._tail_trace_entries("telegram:dm:123", count=10)

    assert [item["stage"] for item in items] == ["inbound_received", "agent_end"]


def test_trace_command_formats_recent_session_entries(tmp_path):
    runner = _make_runner()
    event = _make_event()
    event.get_command_args.return_value = "5"
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True)
    trace_path = log_dir / "gateway-trace.jsonl"
    trace_path.write_text(
        "\n".join([
            '{"session_key":"agent:main:telegram:dm:123","stage":"inbound_received","ts":"2026-03-20T21:00:00Z","preview":"hello"}',
            '{"session_key":"agent:main:telegram:dm:123","stage":"command_dispatch","ts":"2026-03-20T21:00:01Z","command":"trace"}',
        ]),
        encoding="utf-8",
    )

    with patch("gateway.run._hermes_home", tmp_path):
        result = asyncio.run(runner._handle_trace_command(event))

    assert "🪵" in result
    assert "inbound_received" in result
    assert "command_dispatch" in result


def test_resolve_game_profile_aliases():
    runner = _make_runner()
    assert runner._resolve_game_profile("pokemon")["port"] == 9878
    assert runner._resolve_game_profile("fr")["name"] == "pokemon"
    assert runner._resolve_game_profile("smw")["name"] == "mario"
    assert runner._resolve_game_profile("unknown") is None


def test_game_status_all_formats_summary():
    runner = _make_runner()
    event = _make_event()
    event.get_command_args.return_value = "status all"

    async def fake_is_running(profile):
        if profile["name"] == "pokemon":
            return True, {"screen": "240x160"}
        return False, None

    runner._game_is_running = fake_is_running
    result = asyncio.run(runner._handle_game_command(event))
    assert "pokemon" in result
    assert "mario" in result
    assert "240x160" in result
    assert "offline" in result


def test_game_command_rejects_unknown_target():
    runner = _make_runner()
    event = _make_event()
    event.get_command_args.return_value = "start badtarget"
    result = asyncio.run(runner._handle_game_command(event))
    assert "unknown game" in result
    assert "badtarget" in result


def test_game_start_pokemon_auto_runs_intro():
    runner = _make_runner()
    profile = runner._game_profiles()["pokemon"]

    states = iter([
        (False, None),
        (True, {"screen": "240x160"}),
    ])

    async def fake_is_running(_profile):
        return next(states)

    async def fake_http_json(_profile, path):
        assert path == "/macro/run_intro"
        return {"state": {"map_group": 4, "map_num": 1, "x": 13, "y": 13, "party": 0}}

    runner._game_is_running = fake_is_running
    runner._game_http_json = fake_http_json

    class DummyProc:
        async def communicate(self):
            return b"", b""

    with patch("asyncio.create_subprocess_shell", return_value=DummyProc()), \
         patch("asyncio.sleep", new=AsyncMock()):
        result = asyncio.run(runner._game_start_profile(profile))

    assert "intro done" in result
    assert "map 4:1" in result


def test_game_start_non_pokemon_skips_intro():
    runner = _make_runner()
    profile = runner._game_profiles()["mario"]

    states = iter([
        (False, None),
        (True, {"screen": "240x160"}),
    ])

    async def fake_is_running(_profile):
        return next(states)

    runner._game_is_running = fake_is_running

    class DummyProc:
        async def communicate(self):
            return b"", b""

    with patch("asyncio.create_subprocess_shell", return_value=DummyProc()), \
         patch("asyncio.sleep", new=AsyncMock()):
        result = asyncio.run(runner._game_start_profile(profile))

    assert "mario" in result
    assert "started" in result
    assert "intro done" not in result
