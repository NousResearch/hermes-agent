"""Tests for agent.brain_host — Phase 3 Brain Host seam.

Coverage:
  * parity     — BrainHost.build_agent forwards kwargs identically to direct AIAgent().
  * singleton  — BrainHost.get() always returns the same instance.
  * flag-gate  — HERMES_BRAIN_HOST=0/unset → brain_host never imported by _make_agent;
                  HERMES_BRAIN_HOST=1 → routes through BrainHost.build_agent.
"""

from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_recorder_class():
    """Return a fresh recorder class (captures __init__ kwargs) and a list
    that accumulates every instance created."""
    instances: list = []

    class RecorderAgent:
        def __init__(self, **kwargs):
            self._kwargs = kwargs
            instances.append(self)

    return RecorderAgent, instances


# ---------------------------------------------------------------------------
# Parity test
# ---------------------------------------------------------------------------

def test_brain_host_parity():
    """BrainHost.build_agent must pass exactly the same kwargs as a direct call.

    Monkeypatch run_agent.AIAgent with a recorder so we can compare the two
    construction paths without importing the real AIAgent.
    """
    from agent.brain_host import AgentSpec, BrainHost

    RecorderAgent, instances = _make_recorder_class()

    test_kwargs = {
        "model": "claude-opus-4-6",
        "provider": "anthropic",
        "api_key": "sk-test",
        "quiet_mode": True,
    }

    with patch("run_agent.AIAgent", RecorderAgent):
        # --- path 1: via BrainHost ---
        result_hosted = BrainHost.get().build_agent(
            AgentSpec(intent="test", kwargs=test_kwargs)
        )
        # --- path 2: direct ---
        from run_agent import AIAgent as _Direct  # noqa: PLC0415

        result_direct = _Direct(**test_kwargs)

    assert len(instances) == 2
    hosted_kwargs = instances[0]._kwargs
    direct_kwargs = instances[1]._kwargs
    assert hosted_kwargs == direct_kwargs, (
        f"kwargs mismatch:\nhosted={hosted_kwargs}\ndirect={direct_kwargs}"
    )
    # Both return the recorder instance (not None, not a mock wrapper).
    assert isinstance(result_hosted, RecorderAgent)
    assert isinstance(result_direct, RecorderAgent)


# ---------------------------------------------------------------------------
# Singleton test
# ---------------------------------------------------------------------------

def test_brain_host_singleton():
    """BrainHost.get() must return the same object on repeated calls."""
    from agent.brain_host import BrainHost

    a = BrainHost.get()
    b = BrainHost.get()
    assert a is b


# ---------------------------------------------------------------------------
# Flag-gate tests — default OFF
# ---------------------------------------------------------------------------

def _call_make_agent_with_env(env_patch: dict, monkeypatch_patches: dict):
    """Import tui_gateway.server._make_agent under a controlled set of
    patches, call it, and return (mock_ai_agent, sys_modules_snapshot)."""
    fake_runtime = {
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com",
        "api_key": "sk-test",
        "api_mode": "anthropic_messages",
        "command": None,
        "args": None,
        "credential_pool": None,
    }
    fake_cfg = {"agent": {"system_prompt": ""}, "model": {"default": "claude-opus-4-6"}}

    mock_ai_agent = MagicMock()

    # Remove brain_host from sys.modules to get a clean import-detection slate.
    sys.modules.pop("agent.brain_host", None)

    ctx_managers = [
        patch.dict(os.environ, env_patch, clear=False),
        patch("tui_gateway.server._load_cfg", return_value=fake_cfg),
        patch("tui_gateway.server._get_db", return_value=MagicMock()),
        patch("tui_gateway.server._load_reasoning_config", return_value=None),
        patch("tui_gateway.server._load_service_tier", return_value=None),
        patch("tui_gateway.server._load_enabled_toolsets", return_value=None),
        patch("tui_gateway.server._load_fallback_model", return_value=None),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value=fake_runtime,
        ),
        patch("run_agent.AIAgent", mock_ai_agent),
    ]
    # Apply any extra patches the caller wants.
    for target, new_val in monkeypatch_patches.items():
        ctx_managers.append(patch(target, new_val))

    # Enter all context managers.
    entered = []
    try:
        for cm in ctx_managers:
            entered.append(cm.__enter__())
        from tui_gateway.server import _make_agent  # noqa: PLC0415

        _make_agent("sid-gate", "key-gate")
        modules_after = set(sys.modules.keys())
    finally:
        for cm, result in zip(reversed(ctx_managers), reversed(entered)):
            cm.__exit__(None, None, None)

    return mock_ai_agent, modules_after


def test_flag_gate_off_by_default():
    """When HERMES_BRAIN_HOST is unset, agent.brain_host must NOT be imported."""
    env = {}
    os.environ.pop("HERMES_BRAIN_HOST", None)

    _, mods = _call_make_agent_with_env(env, {})
    assert "agent.brain_host" not in mods, (
        "agent.brain_host was imported even though HERMES_BRAIN_HOST is unset"
    )


def test_flag_gate_off_when_zero():
    """When HERMES_BRAIN_HOST=0, agent.brain_host must NOT be imported."""
    _, mods = _call_make_agent_with_env({"HERMES_BRAIN_HOST": "0"}, {})
    assert "agent.brain_host" not in mods, (
        "agent.brain_host was imported even though HERMES_BRAIN_HOST=0"
    )


def test_flag_gate_on_routes_through_brain_host():
    """When HERMES_BRAIN_HOST=1, _make_agent must call BrainHost.build_agent."""
    from agent.brain_host import BrainHost  # noqa: PLC0415 — ensure loaded

    build_agent_mock = MagicMock(return_value=MagicMock())

    with (
        patch.dict(os.environ, {"HERMES_BRAIN_HOST": "1"}),
        patch("agent.brain_host.BrainHost.build_agent", build_agent_mock),
    ):
        fake_runtime = {
            "provider": "anthropic",
            "base_url": "https://api.anthropic.com",
            "api_key": "sk-test",
            "api_mode": "anthropic_messages",
            "command": None,
            "args": None,
            "credential_pool": None,
        }
        fake_cfg = {"agent": {"system_prompt": ""}, "model": {"default": "claude-opus-4-6"}}

        with (
            patch("tui_gateway.server._load_cfg", return_value=fake_cfg),
            patch("tui_gateway.server._get_db", return_value=MagicMock()),
            patch("tui_gateway.server._load_reasoning_config", return_value=None),
            patch("tui_gateway.server._load_service_tier", return_value=None),
            patch("tui_gateway.server._load_enabled_toolsets", return_value=None),
            patch("tui_gateway.server._load_fallback_model", return_value=None),
            patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                return_value=fake_runtime,
            ),
            patch("run_agent.AIAgent", MagicMock()),
        ):
            from tui_gateway.server import _make_agent  # noqa: PLC0415

            _make_agent("sid-on", "key-on")

    build_agent_mock.assert_called_once()
    spec = build_agent_mock.call_args.args[0]
    assert spec.intent == "tui_gateway"
    assert isinstance(spec.kwargs, dict)
    # Spot-check a stable key that the gate path must forward.
    assert spec.kwargs.get("quiet_mode") is True


# ---------------------------------------------------------------------------
# Flag-gate tests — gateway/platforms/api_server._create_agent
#
# gateway.platforms.api_server._create_agent is importable and its
# AIAgent construction is refactored into agent_kwargs, so we can drive
# it directly with the same FakeAgent monkeypatching pattern used in
# test_api_server.py.
#
# gateway/run.py's gate lives deep inside GatewayRunner._run_agent (an
# async method with heavy I/O dependencies); that site is verified by
# import-compilation + grep-assert (see task notes) rather than a
# unit-driving test.
# ---------------------------------------------------------------------------

def _make_fake_runtime():
    return {
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com",
        "api_key": "sk-test",
        "api_mode": "anthropic_messages",
    }


def _patch_api_server_deps(extra_patches=None):
    """Return a list of context managers that stub the heavy dependencies of
    APIServerAdapter._create_agent so we can call it without real config."""
    from unittest.mock import patch, MagicMock

    patches = [
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value=_make_fake_runtime()),
        patch("gateway.run._resolve_gateway_model", return_value="claude-opus-4-6"),
        patch("gateway.run._load_gateway_config", return_value={}),
        patch(
            "gateway.run.GatewayRunner._load_reasoning_config",
            staticmethod(lambda: None),
        ),
        patch(
            "gateway.run.GatewayRunner._load_fallback_model",
            staticmethod(lambda: None),
        ),
        patch("hermes_cli.tools_config._get_platform_tools", lambda *_: set()),
    ]
    if extra_patches:
        patches.extend(extra_patches)
    return patches


def test_api_server_flag_gate_off_by_default():
    """When HERMES_BRAIN_HOST is unset, api_server._create_agent must NOT
    import agent.brain_host."""
    from unittest.mock import MagicMock, patch
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.config import PlatformConfig

    os.environ.pop("HERMES_BRAIN_HOST", None)
    sys.modules.pop("agent.brain_host", None)

    mock_ai_agent = MagicMock()
    adapter = APIServerAdapter(PlatformConfig(enabled=True))

    ctx_managers = _patch_api_server_deps([
        patch("run_agent.AIAgent", mock_ai_agent),
    ])
    entered = []
    try:
        for cm in ctx_managers:
            entered.append(cm.__enter__())
        with patch.object(adapter, "_ensure_session_db", return_value=None):
            with patch.dict(os.environ, {}, clear=False):
                adapter._create_agent(session_id="test-off")
        modules_after = set(sys.modules.keys())
    finally:
        for cm in reversed(ctx_managers):
            cm.__exit__(None, None, None)

    assert "agent.brain_host" not in modules_after, (
        "agent.brain_host was imported even though HERMES_BRAIN_HOST is unset"
    )
    mock_ai_agent.assert_called_once()


def test_api_server_flag_gate_off_when_zero():
    """When HERMES_BRAIN_HOST=0, api_server._create_agent must NOT import
    agent.brain_host."""
    from unittest.mock import MagicMock, patch
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.config import PlatformConfig

    sys.modules.pop("agent.brain_host", None)

    mock_ai_agent = MagicMock()
    adapter = APIServerAdapter(PlatformConfig(enabled=True))

    ctx_managers = _patch_api_server_deps([
        patch("run_agent.AIAgent", mock_ai_agent),
        patch.dict(os.environ, {"HERMES_BRAIN_HOST": "0"}, clear=False),
    ])
    entered = []
    try:
        for cm in ctx_managers:
            entered.append(cm.__enter__())
        with patch.object(adapter, "_ensure_session_db", return_value=None):
            adapter._create_agent(session_id="test-zero")
        modules_after = set(sys.modules.keys())
    finally:
        for cm in reversed(ctx_managers):
            cm.__exit__(None, None, None)

    assert "agent.brain_host" not in modules_after, (
        "agent.brain_host was imported even though HERMES_BRAIN_HOST=0"
    )
    mock_ai_agent.assert_called_once()


def test_api_server_flag_gate_on_routes_through_brain_host():
    """When HERMES_BRAIN_HOST=1, api_server._create_agent must call
    BrainHost.build_agent with intent='api-server'."""
    from unittest.mock import MagicMock, patch
    from agent.brain_host import BrainHost  # ensure loaded
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.config import PlatformConfig

    build_agent_mock = MagicMock(return_value=MagicMock())
    adapter = APIServerAdapter(PlatformConfig(enabled=True))

    ctx_managers = _patch_api_server_deps([
        patch("run_agent.AIAgent", MagicMock()),
        patch("agent.brain_host.BrainHost.build_agent", build_agent_mock),
        patch.dict(os.environ, {"HERMES_BRAIN_HOST": "1"}, clear=False),
    ])
    entered = []
    try:
        for cm in ctx_managers:
            entered.append(cm.__enter__())
        with patch.object(adapter, "_ensure_session_db", return_value=None):
            adapter._create_agent(session_id="test-on")
    finally:
        for cm in reversed(ctx_managers):
            cm.__exit__(None, None, None)

    build_agent_mock.assert_called_once()
    spec = build_agent_mock.call_args.args[0]
    assert spec.intent == "api-server"
    assert isinstance(spec.kwargs, dict)
    # Spot-check stable keys that the gate path must forward.
    assert spec.kwargs.get("quiet_mode") is True
    assert spec.kwargs.get("platform") == "api_server"


# ---------------------------------------------------------------------------
# brain_host_gate.build_agent — the one-line gate helper every migrated
# construction site calls.
# ---------------------------------------------------------------------------

def test_gate_helper_flag_off_constructs_directly_without_importing_brain_host():
    """With the flag unset, build_agent must construct via run_agent.AIAgent
    and must NOT import agent.brain_host (zero-footprint invariant)."""
    from agent.brain_host_gate import build_agent

    RecorderAgent, instances = _make_recorder_class()

    os.environ.pop("HERMES_BRAIN_HOST", None)
    sys.modules.pop("agent.brain_host", None)

    with patch("run_agent.AIAgent", RecorderAgent):
        result = build_agent("test-intent", model="m", quiet_mode=True)

    assert "agent.brain_host" not in sys.modules, (
        "agent.brain_host was imported even though HERMES_BRAIN_HOST is unset"
    )
    assert isinstance(result, RecorderAgent)
    assert instances[0]._kwargs == {"model": "m", "quiet_mode": True}


def test_gate_helper_flag_zero_constructs_directly():
    """HERMES_BRAIN_HOST=0 behaves identically to unset."""
    from agent.brain_host_gate import build_agent

    RecorderAgent, instances = _make_recorder_class()
    sys.modules.pop("agent.brain_host", None)

    with (
        patch.dict(os.environ, {"HERMES_BRAIN_HOST": "0"}, clear=False),
        patch("run_agent.AIAgent", RecorderAgent),
    ):
        result = build_agent("test-intent", model="m")

    assert "agent.brain_host" not in sys.modules
    assert isinstance(result, RecorderAgent)
    assert instances[0]._kwargs == {"model": "m"}


def test_gate_helper_flag_on_routes_through_brain_host_with_intent():
    """HERMES_BRAIN_HOST=1 routes through BrainHost.build_agent, forwarding the
    intent tag and the exact kwargs on the AgentSpec."""
    from agent.brain_host import BrainHost  # noqa: PLC0415 — ensure loaded
    from agent.brain_host_gate import build_agent

    build_agent_mock = MagicMock(return_value=MagicMock())

    with (
        patch.dict(os.environ, {"HERMES_BRAIN_HOST": "1"}, clear=False),
        patch("agent.brain_host.BrainHost.build_agent", build_agent_mock),
    ):
        build_agent("cron", model="m", quiet_mode=True)

    build_agent_mock.assert_called_once()
    spec = build_agent_mock.call_args.args[0]
    assert spec.intent == "cron"
    assert spec.kwargs == {"model": "m", "quiet_mode": True}


def test_gate_helper_off_on_kwargs_parity():
    """The kwargs that reach AIAgent must be identical on both gate paths."""
    from agent.brain_host_gate import build_agent

    RecorderAgent, instances = _make_recorder_class()
    test_kwargs = {"model": "claude-opus-4-6", "api_key": "sk-test", "quiet_mode": True}

    with patch("run_agent.AIAgent", RecorderAgent):
        os.environ.pop("HERMES_BRAIN_HOST", None)
        build_agent("parity", **test_kwargs)

        with patch.dict(os.environ, {"HERMES_BRAIN_HOST": "1"}, clear=False):
            build_agent("parity", **test_kwargs)

    assert len(instances) == 2
    assert instances[0]._kwargs == instances[1]._kwargs == test_kwargs


# ---------------------------------------------------------------------------
# Site-table source check — every migrated construction site must call
# build_agent with its registered intent.  This is the practical substitute
# for unit-driving call sites that live deep inside async/thread workers.
# ---------------------------------------------------------------------------

# (relative file path, expected intents constructed in that file)
_MIGRATED_SITES = [
    ("tui_gateway/server.py", ["tui_gateway", "tui-background", "preview-restart"]),
    ("gateway/run.py", ["gateway-run", "history-hygiene", "gateway-background"]),
    ("gateway/platforms/api_server.py", ["api-server"]),
    ("gateway/platforms/feishu_comment.py", ["feishu-comment"]),
    ("gateway/slash_commands.py", ["compress"]),
    ("hermes_cli/prompt_size.py", ["prompt-size"]),
    ("hermes_cli/oneshot.py", ["oneshot"]),
    ("hermes_cli/cli_commands_mixin.py", ["cli-background"]),
    ("hermes_cli/cli_agent_setup_mixin.py", ["cli"]),
    ("agent/background_review.py", ["background-review"]),
    ("agent/curator.py", ["curator"]),
    ("acp_adapter/session.py", ["acp"]),
    ("cron/scheduler.py", ["cron"]),
    ("batch_runner.py", ["batch"]),
    ("run_agent.py", ["run-agent-cli"]),
    ("tools/delegate_tool.py", ["delegate"]),
]


def test_all_migrated_sites_use_gate_helper():
    """Every registered construction site calls build_agent("<intent>", ...)."""
    import pathlib
    import re

    root = pathlib.Path(__file__).parents[2]
    missing = []
    for rel_path, intents in _MIGRATED_SITES:
        src = (root / rel_path).read_text()
        for intent in intents:
            # Matches both single-line build_agent("cron", ...) and the
            # multi-line call style build_agent(\n    "cron",\n    ...
            if not re.search(r'build_agent\(\s*"' + re.escape(intent) + r'"', src):
                missing.append(f'{rel_path}: build_agent("{intent}" …)')
    assert not missing, "sites not routed through brain_host_gate:\n" + "\n".join(missing)
