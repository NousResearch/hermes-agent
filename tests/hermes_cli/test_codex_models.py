import json
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hermes_cli.codex_models import DEFAULT_CODEX_MODELS, get_codex_model_ids


def test_get_codex_model_ids_prioritizes_default_and_cache(tmp_path, monkeypatch):
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir(parents=True, exist_ok=True)
    (codex_home / "config.toml").write_text('model = "gpt-5.2-codex"\n')
    (codex_home / "models_cache.json").write_text(
        json.dumps(
            {
                "models": [
                    {"slug": "gpt-5.3-codex", "priority": 20, "supported_in_api": True},
                    {"slug": "gpt-5.1-codex", "priority": 5, "supported_in_api": True},
                    {"slug": "gpt-5.4", "priority": 1, "supported_in_api": True},
                    {"slug": "gpt-5-hidden-codex", "priority": 2, "visibility": "hidden"},
                ]
            }
        )
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    models = get_codex_model_ids()

    assert models[0] == "gpt-5.2-codex"
    assert "gpt-5.1-codex" in models
    assert "gpt-5.3-codex" in models
    # Non-codex-suffixed models are included when the cache says they're available
    assert "gpt-5.4" in models
    assert "gpt-5.4-mini" in models
    assert "gpt-5-hidden-codex" not in models


def test_setup_wizard_codex_import_resolves():
    """Regression test for #712: setup.py must import the correct function name."""
    # This mirrors the exact import used in hermes_cli/setup.py line 873.
    # A prior bug had 'get_codex_models' (wrong) instead of 'get_codex_model_ids'.
    from hermes_cli.codex_models import get_codex_model_ids as setup_import
    assert callable(setup_import)


def test_get_codex_model_ids_falls_back_to_curated_defaults(tmp_path, monkeypatch):
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    models = get_codex_model_ids()

    assert models[: len(DEFAULT_CODEX_MODELS)] == DEFAULT_CODEX_MODELS
    assert "gpt-5.4" in models
    assert "gpt-5.3-codex-spark" in models


def test_get_codex_model_ids_adds_forward_compat_models_from_templates(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.codex_models._fetch_models_from_api",
        lambda access_token: ["gpt-5.2-codex"],
    )

    models = get_codex_model_ids(access_token="codex-access-token")

    assert models == ["gpt-5.2-codex", "gpt-5.4-mini", "gpt-5.4", "gpt-5.3-codex", "gpt-5.3-codex-spark"]


def test_model_command_uses_runtime_access_token_for_codex_list(monkeypatch):
    from hermes_cli.main import _model_flow_openai_codex

    captured = {}

    monkeypatch.setattr(
        "hermes_cli.auth.get_codex_auth_status",
        lambda: {"logged_in": True},
    )
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_codex_runtime_credentials",
        lambda *args, **kwargs: {"api_key": "codex-access-token"},
    )

    def _fake_get_codex_model_ids(access_token=None):
        captured["access_token"] = access_token
        return ["gpt-5.2-codex", "gpt-5.2"]

    def _fake_prompt_model_selection(model_ids, current_model=""):
        captured["model_ids"] = list(model_ids)
        captured["current_model"] = current_model
        return None

    monkeypatch.setattr(
        "hermes_cli.codex_models.get_codex_model_ids",
        _fake_get_codex_model_ids,
    )
    monkeypatch.setattr(
        "hermes_cli.auth._prompt_model_selection",
        _fake_prompt_model_selection,
    )

    _model_flow_openai_codex({}, current_model="openai/gpt-5.4")

    assert captured["access_token"] == "codex-access-token"
    assert captured["model_ids"] == ["gpt-5.2-codex", "gpt-5.2"]
    assert captured["current_model"] == "openai/gpt-5.4"


# ── Tests for _normalize_model_for_provider ──────────────────────────


def _make_cli(model="anthropic/claude-opus-4.6", **kwargs):
    """Create a HermesCLI with minimal mocking."""
    import cli as _cli_mod
    from cli import HermesCLI

    _clean_config = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "auto",
        },
        "display": {"compact": False, "tool_progress": "all", "resume_display": "full"},
        "agent": {},
        "terminal": {"env_type": "local"},
    }
    clean_env = {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}
    with (
        patch("cli.get_tool_definitions", return_value=[]),
        patch.dict("os.environ", clean_env, clear=False),
        patch.dict(_cli_mod.__dict__, {"CLI_CONFIG": _clean_config}),
    ):
        cli = HermesCLI(model=model, **kwargs)
    return cli


class TestNormalizeModelForProvider:
    """_normalize_model_for_provider() trusts user-selected models.

    Only two things happen:
    1. Provider prefixes are stripped (API needs bare slugs)
    2. The *untouched default* model is swapped for a Codex model
    Everything else passes through — the API is the judge.
    """

    def test_non_codex_provider_is_noop(self):
        cli = _make_cli(model="gpt-5.4")
        changed = cli._normalize_model_for_provider("openrouter")
        assert changed is False
        assert cli.model == "gpt-5.4"

    def test_native_provider_prefix_is_stripped_before_agent_startup(self):
        cli = _make_cli(model="zai/glm-5.1")
        changed = cli._normalize_model_for_provider("zai")
        assert changed is True
        assert cli.model == "glm-5.1"

    def test_bare_codex_model_passes_through(self):
        cli = _make_cli(model="gpt-5.3-codex")
        changed = cli._normalize_model_for_provider("openai-codex")
        assert changed is False
        assert cli.model == "gpt-5.3-codex"

    def test_bare_non_codex_model_passes_through(self):
        """gpt-5.4 (no 'codex' suffix) passes through — user chose it."""
        cli = _make_cli(model="gpt-5.4")
        changed = cli._normalize_model_for_provider("openai-codex")
        assert changed is False
        assert cli.model == "gpt-5.4"

    def test_any_bare_model_trusted(self):
        """Even a non-OpenAI bare model passes through — user explicitly set it."""
        cli = _make_cli(model="claude-opus-4-6")
        changed = cli._normalize_model_for_provider("openai-codex")
        # User explicitly chose this model — we trust them, API will error if wrong
        assert changed is False
        assert cli.model == "claude-opus-4-6"

    def test_provider_prefix_stripped(self):
        """openai/gpt-5.4 → gpt-5.4 (strip prefix, keep model)."""
        cli = _make_cli(model="openai/gpt-5.4")
        changed = cli._normalize_model_for_provider("openai-codex")
        assert changed is True
        assert cli.model == "gpt-5.4"

    def test_any_provider_prefix_stripped(self):
        """anthropic/claude-opus-4.6 → claude-opus-4.6 (strip prefix only).
        User explicitly chose this — let the API decide if it works."""
        cli = _make_cli(model="anthropic/claude-opus-4.6")
        changed = cli._normalize_model_for_provider("openai-codex")
        assert changed is True
        assert cli.model == "claude-opus-4.6"

    def test_opencode_go_prefix_stripped(self):
        cli = _make_cli(model="opencode-go/kimi-k2.5")
        cli.api_mode = "chat_completions"
        changed = cli._normalize_model_for_provider("opencode-go")
        assert changed is True
        assert cli.model == "kimi-k2.5"
        assert cli.api_mode == "chat_completions"

    def test_opencode_zen_claude_sets_messages_mode(self):
        cli = _make_cli(model="opencode-zen/claude-sonnet-4-6")
        cli.api_mode = "chat_completions"
        changed = cli._normalize_model_for_provider("opencode-zen")
        assert changed is True
        assert cli.model == "claude-sonnet-4-6"
        assert cli.api_mode == "anthropic_messages"

    def test_default_model_replaced(self):
        """No model configured (empty default) gets swapped for codex."""
        import cli as _cli_mod
        _clean_config = {
            "model": {
                "default": "",
                "base_url": "",
                "provider": "auto",
            },
            "display": {"compact": False, "tool_progress": "all", "resume_display": "full"},
            "agent": {},
            "terminal": {"env_type": "local"},
        }
        # Don't pass model= so _model_is_default is True
        with (
            patch("cli.get_tool_definitions", return_value=[]),
            patch.dict("os.environ", {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}, clear=False),
            patch.dict(_cli_mod.__dict__, {"CLI_CONFIG": _clean_config}),
        ):
            from cli import HermesCLI
            cli = HermesCLI()

        assert cli._model_is_default is True
        with patch(
            "hermes_cli.codex_models.get_codex_model_ids",
            return_value=["gpt-5.3-codex", "gpt-5.4"],
        ):
            changed = cli._normalize_model_for_provider("openai-codex")
        assert changed is True
        # Uses first from available list
        assert cli.model == "gpt-5.3-codex"

    def test_default_fallback_when_api_fails(self):
        """No model configured falls back to gpt-5.3-codex when API unreachable."""
        import cli as _cli_mod
        _clean_config = {
            "model": {
                "default": "",
                "base_url": "",
                "provider": "auto",
            },
            "display": {"compact": False, "tool_progress": "all", "resume_display": "full"},
            "agent": {},
            "terminal": {"env_type": "local"},
        }
        with (
            patch("cli.get_tool_definitions", return_value=[]),
            patch.dict("os.environ", {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}, clear=False),
            patch.dict(_cli_mod.__dict__, {"CLI_CONFIG": _clean_config}),
        ):
            from cli import HermesCLI
            cli = HermesCLI()

        with patch(
            "hermes_cli.codex_models.get_codex_model_ids",
            side_effect=Exception("offline"),
        ):
            changed = cli._normalize_model_for_provider("openai-codex")
        assert changed is True
        assert cli.model == "gpt-5.3-codex"
