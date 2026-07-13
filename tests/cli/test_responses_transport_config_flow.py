from types import SimpleNamespace

import cli
from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin


class _CLISetupHarness(CLIAgentSetupMixin):
    def __init__(self):
        self.agent = None
        self._session_db = object()
        self._resumed = False
        self.conversation_history = []
        self.api_key = "token"
        self.base_url = "https://chatgpt.com/backend-api/codex"
        self.provider = "openai-codex"
        self.api_mode = "codex_responses"
        self.responses_transport = "websocket-cached"
        self.acp_command = None
        self.acp_args = []
        self.model = "gpt-5-codex"
        self.max_tokens = None
        self.max_turns = 1
        self.enabled_toolsets = []
        self.disabled_toolsets = []
        self.verbose = False
        self.system_prompt = None
        self.prefill_messages = []
        self.reasoning_config = None
        self.service_tier = None
        self._providers_only = None
        self._providers_ignore = None
        self._providers_order = None
        self._provider_sort = None
        self._provider_require_params = False
        self._provider_data_collection = None
        self._openrouter_min_coding_score = None
        self.session_id = "session"
        self._fallback_model = None
        self.checkpoints_enabled = False
        self.checkpoint_max_snapshots = 1
        self.checkpoint_max_total_size_mb = 1
        self.checkpoint_max_file_size_mb = 1
        self.pass_session_id = False
        self.ignore_rules = True
        self._inline_diffs_enabled = False
        self.streaming_enabled = False
        self._pending_title = None

    def _install_tool_callbacks(self):
        pass

    def _ensure_tirith_security(self):
        pass

    def _ensure_runtime_credentials(self):
        return True

    def _current_reasoning_callback(self):
        return None

    def _clarify_callback(self, *_args, **_kwargs):
        return None

    def _on_thinking(self, *_args, **_kwargs):
        pass

    def _on_tool_progress(self, *_args, **_kwargs):
        pass

    def _on_notice(self, *_args, **_kwargs):
        pass

    def _on_notice_clear(self, *_args, **_kwargs):
        pass

    def _on_reaction(self, *_args, **_kwargs):
        pass


def test_classic_cli_passes_resolved_responses_transport_to_agent(monkeypatch):
    captured = {}

    def fake_agent(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(_print_fn=None)

    monkeypatch.setattr(cli, "AIAgent", fake_agent)
    monkeypatch.setattr(cli, "_prepare_deferred_agent_startup", lambda: None)
    monkeypatch.setattr("hermes_cli.mcp_startup.wait_for_mcp_discovery", lambda: None)

    harness = _CLISetupHarness()
    assert harness._init_agent()
    assert captured["responses_transport"] == "websocket-cached"
    assert harness._active_agent_route_signature[4] == "websocket-cached"
