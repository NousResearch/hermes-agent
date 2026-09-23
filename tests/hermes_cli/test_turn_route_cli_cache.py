"""Turn-route choices must select the matching warm CLI agent."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch


def _import_cli():
    import hermes_cli.config as config_mod

    if not hasattr(config_mod, "save_env_value_secure"):
        config_mod.save_env_value_secure = lambda key, value: {
            "success": True,
            "stored_as": key,
            "validated": False,
        }

    import cli as cli_mod

    return cli_mod


def test_chat_rebuilds_and_reuses_agents_for_selected_model_and_provider_routes():
    cli_mod = _import_cli()
    shell = cli_mod.HermesCLI(model="alpha", compact=True, max_turns=1)
    shell.provider = "provider-a"
    shell.requested_provider = "provider-a"
    shell.api_mode = "chat_completions"
    shell.base_url = "https://provider-a.example/v1"
    shell.api_key = "key-a"
    shell.acp_command = None
    shell.acp_args = []
    shell._credential_pool = None
    shell.service_tier = None
    shell._skip_turn_routing = False
    shell._active_agent_route_signature = None
    shell.agent = None
    shell._session_db = object()
    shell._ensure_runtime_credentials = lambda: True
    shell.finalize_preloaded_skills = lambda: None
    shell._install_tool_callbacks = lambda: None
    shell._ensure_tirith_security = lambda: None

    selected = {"model": "alpha", "provider": "provider-a"}
    agents = []
    turn_agents = []

    class CapturingAgent:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self._print_fn = None
            self._pending_cli_user_message = {}
            self._notification_config = {}
            agents.append(self)

        def release_clients(self):
            pass

    def fake_apply(route, **_context):
        payload = {**route, **selected}
        changed = (payload["model"], payload["provider"]) != (
            route["model"], route["runtime"]["provider"])
        return SimpleNamespace(changed=changed, payload=payload, trace=[])

    # Keep chat() and _init_agent() real through route comparison and agent
    # construction. Stub only the unrelated turn execution/rendering work.
    shell._sync_fallback_chain_with_config = lambda _agent: None
    shell._chat_route_images = lambda message, _images: message
    shell._chat_expand_context_references = lambda message: (message, None)
    shell._chat_stage_user_message = lambda _agent, _message: None
    shell._reset_stream_state = lambda: None
    shell._chat_setup_turn_audio = lambda *_args: None
    shell._chat_run_agent = lambda *_args: None
    shell._chat_monitor_agent_thread = lambda _turn, _thread: None
    shell._chat_settle_turn = lambda _turn: None
    shell._chat_render_turn = lambda _turn, _thread, _interrupt: (
        turn_agents.append(shell.agent) or shell.agent.model
    )
    shell._chat_release_turn_audio = lambda _turn: None

    class Console:
        def print(self, *_args, **_kwargs):
            pass

    with (
        patch.object(cli_mod, "_prepare_deferred_agent_startup", lambda: None),
        patch.object(cli_mod, "set_secret_capture_callback", lambda _callback: None),
        patch.object(cli_mod, "ChatConsole", Console),
        patch.object(cli_mod, "_cprint", lambda *_args, **_kwargs: None),
        patch.object(cli_mod, "_accent_hex", lambda: "white"),
        patch("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build", lambda **_kwargs: None),
        patch("run_agent.AIAgent", CapturingAgent),
        patch("hermes_cli.middleware.apply_turn_route_middleware", fake_apply),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda *, requested, target_model: {
                "api_key": "key-b",
                "base_url": "https://provider-b.example/v1",
                "provider": requested,
                "requested_provider": requested,
                "api_mode": "chat_completions",
                "command": None,
                "args": [],
                "credential_pool": None,
            },
        ),
        patch("agent.notification_presentation.notification_config_snapshot", lambda: {}),
        patch("agent.notification_presentation.notification_policy_snapshot", lambda *_args: nullcontext()),
        patch("gateway.warning_notifications.diagnostic_turn_muted", lambda *_args: False),
    ):
        # Configured A, model-only A→B, repeated B, provider A→B, repeated B,
        # then back to configured A. The chat() cache gate owns reuse/rebuild.
        assert shell.chat("turn 1") == "alpha"
        selected.update(model="beta")
        assert shell.chat("turn 2") == "beta"
        assert shell.chat("turn 3") == "beta"
        selected.update(model="gamma", provider="provider-b")
        assert shell.chat("turn 4") == "gamma"
        assert shell.chat("turn 5") == "gamma"
        selected.update(model="alpha", provider="provider-a")
        assert shell.chat("turn 6") == "alpha"

    assert [agent.model for agent in agents] == ["alpha", "beta", "gamma", "alpha"]
    assert [agent.provider for agent in agents] == ["provider-a", "provider-a", "provider-b", "provider-a"]
    assert turn_agents[1] is turn_agents[2]
    assert turn_agents[3] is turn_agents[4]
    assert turn_agents[0] is not turn_agents[1]
    assert turn_agents[1] is not turn_agents[3]
    assert turn_agents[3] is not turn_agents[5]
