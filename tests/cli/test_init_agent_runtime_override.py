from types import SimpleNamespace

from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin


class _StubCLI(CLIAgentSetupMixin):
    def __init__(self):
        self.api_key = "stale-key"
        self.base_url = "https://stale.example/v1"
        self.provider = "stale-provider"
        self.api_mode = "chat_completions"
        self.acp_command = None
        self.acp_args = []
        self._credential_pool = None
        self.model = "gpt-test"
        self.max_tokens = 1024
        self.max_turns = 5
        self.enabled_toolsets = []
        self.disabled_toolsets = []
        self.verbose = False
        self.tool_progress_mode = "off"
        self.system_prompt = None
        self.prefill_messages = None
        self.reasoning_config = None
        self.service_tier = None
        self._providers_only = None
        self._providers_ignore = None
        self._providers_order = None
        self._provider_sort = None
        self._provider_require_params = None
        self._provider_data_collection = None
        self._openrouter_min_coding_score = None
        self.session_id = "test-session"
        self._session_db = None
        self._clarify_callback = None
        self._fallback_model = None
        self.checkpoints_enabled = False
        self.checkpoint_max_snapshots = None
        self.checkpoint_max_total_size_mb = None
        self.checkpoint_max_file_size_mb = None
        self.pass_session_id = False
        self.ignore_rules = False
        self._inline_diffs_enabled = False
        self.streaming_enabled = False
        self.agent = None
        self._active_agent_route_signature = None
        self._resumed = False
        self.conversation_history = []
        self._pending_title = None
        self.refresh_calls = 0

    def _ensure_runtime_credentials(self):
        self.refresh_calls += 1
        self.api_key = "fresh-codex-key"
        self.base_url = "https://chatgpt.com/backend-api/codex"
        self.provider = "openai-codex"
        self.api_mode = "codex_responses"
        self.acp_command = None
        self.acp_args = []
        self._credential_pool = object()
        return True

    def _current_reasoning_callback(self):
        return None

    def _install_tool_callbacks(self):
        return None

    def _ensure_tirith_security(self):
        return None

    def _on_thinking(self, *_args, **_kwargs):
        return None

    def _on_tool_progress(self, *_args, **_kwargs):
        return None

    def _on_tool_start(self, *_args, **_kwargs):
        return None

    def _on_tool_complete(self, *_args, **_kwargs):
        return None

    def _stream_delta(self, *_args, **_kwargs):
        return None

    def _on_tool_gen_start(self, *_args, **_kwargs):
        return None

    def _on_notice(self, *_args, **_kwargs):
        return None

    def _on_notice_clear(self, *_args, **_kwargs):
        return None


def _patch_aiagent_and_helpers(monkeypatch, captured):
    monkeypatch.setattr("cli._prepare_deferred_agent_startup", lambda: None)
    monkeypatch.setattr("cli._cprint", lambda *a, **k: None)
    monkeypatch.setattr("cli.ChatConsole", lambda: SimpleNamespace(print=lambda *a, **k: None))
    monkeypatch.setattr("hermes_cli.mcp_startup.wait_for_mcp_discovery", lambda: None)

    def fake_aiagent(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            session_id=kwargs.get("session_id"),
            _print_fn=None,
            _session_db_created=False,
            _pending_title=None,
            tool_progress_callback=None,
        )

    monkeypatch.setattr("cli.AIAgent", fake_aiagent)


def test_init_agent_uses_post_refresh_api_mode_over_stale_override(monkeypatch):
    captured = {}
    _patch_aiagent_and_helpers(monkeypatch, captured)

    cli = _StubCLI()
    stale_override = {
        "api_key": "stale-key",
        "base_url": "https://stale.example/v1",
        "provider": "stale-provider",
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": None,
    }

    ok = cli._init_agent(runtime_override=stale_override)
    assert ok is True
    assert cli.refresh_calls == 1
    assert captured["api_mode"] == "codex_responses"
    assert captured["api_key"] == "fresh-codex-key"
    assert captured["base_url"] == "https://chatgpt.com/backend-api/codex"
    assert captured["provider"] == "openai-codex"
    assert captured["credential_pool"] is cli._credential_pool


def test_init_agent_runtime_override_does_not_clobber_provider_or_base_url(monkeypatch):
    captured = {}
    _patch_aiagent_and_helpers(monkeypatch, captured)

    cli = _StubCLI()
    stale_override = {
        "api_key": "stale-key",
        "base_url": "https://stale.example/v1",
        "provider": "stale-provider",
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": None,
    }

    ok = cli._init_agent(runtime_override=stale_override)
    assert ok is True
    assert captured["provider"] == "openai-codex"
    assert captured["base_url"] == "https://chatgpt.com/backend-api/codex"
    assert captured["api_key"] == "fresh-codex-key"


def test_init_agent_works_with_no_runtime_override(monkeypatch):
    captured = {}
    _patch_aiagent_and_helpers(monkeypatch, captured)

    cli = _StubCLI()
    ok = cli._init_agent()
    assert ok is True
    assert captured["api_mode"] == "codex_responses"
    assert captured["provider"] == "openai-codex"
