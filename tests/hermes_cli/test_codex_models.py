import json
from unittest.mock import patch

from hermes_cli.codex_models import (
    DEFAULT_CODEX_MODELS,
    _FORWARD_COMPAT_TEMPLATE_MODELS,
    get_codex_model_ids,
)


CHATGPT_REJECTED_CODEX_PRO_SLUGS = {
    "gpt-5.6-sol-pro",
    "gpt-5.6-terra-pro",
    "gpt-5.6-luna-pro",
}


def test_curated_codex_fallback_excludes_chatgpt_rejected_pro_slugs(monkeypatch):
    """OAuth fallback retains real models but never synthesizes rejected ones."""
    retained_models = {"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"}
    template_models = {model for model, _fallbacks in _FORWARD_COMPAT_TEMPLATE_MODELS}

    assert retained_models.issubset(DEFAULT_CODEX_MODELS)
    assert retained_models.issubset(template_models)
    assert CHATGPT_REJECTED_CODEX_PRO_SLUGS.isdisjoint(DEFAULT_CODEX_MODELS)
    assert CHATGPT_REJECTED_CODEX_PRO_SLUGS.isdisjoint(template_models)

    monkeypatch.setattr(
        "hermes_cli.codex_models._fetch_models_from_api",
        lambda access_token: ["gpt-5.5"],
    )
    model_ids = get_codex_model_ids(access_token="codex-access-token")

    assert retained_models.issubset(model_ids)
    assert CHATGPT_REJECTED_CODEX_PRO_SLUGS.isdisjoint(model_ids)


def test_picker_synthesizes_900k_variants_for_verified_slugs():
    """Every live-verified large-context slug gets an explicit ``-900k``
    picker variant directly after its base entry; slugs that genuinely
    enforce 272K (gpt-5.5, gpt-5.4-mini) never get one. Base slugs stay
    in the list as the cheaper 272K default."""
    model_ids = get_codex_model_ids()  # offline curated path

    for base in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.4"):
        assert base in model_ids
        assert f"{base}-900k" in model_ids
        assert model_ids.index(f"{base}-900k") == model_ids.index(base) + 1

    assert "gpt-5.5-900k" not in model_ids
    assert "gpt-5.4-mini-900k" not in model_ids
    assert "gpt-5.3-codex-900k" not in model_ids


def test_picker_never_synthesizes_900k_for_pro_or_unknown_slugs():
    """Eligibility is an exact predicate, not a family-prefix match:
    ``-pro`` slugs are not routable on Codex OAuth (backend 400s them) and
    unknown future descendants were never probed — neither may gain a
    synthetic ``-900k`` entry (#92797 review)."""
    from hermes_cli.codex_models import _finalize_codex_models

    out = _finalize_codex_models(["gpt-5.6-sol-pro", "gpt-5.6-nova"])
    assert "gpt-5.6-sol-pro-900k" not in out
    assert "gpt-5.6-nova-900k" not in out




def test_setup_wizard_codex_import_resolves():
    """Regression test for #712: setup.py must import the correct function name."""
    # This mirrors the exact import used in hermes_cli/setup.py line 873.
    # A prior bug had 'get_codex_models' (wrong) instead of 'get_codex_model_ids'.
    from hermes_cli.codex_models import get_codex_model_ids as setup_import
    assert callable(setup_import)




def test_fetch_from_api_keeps_supported_in_api_false_models(monkeypatch):
    """Regression: gpt-5.3-codex-spark is returned by the live Codex backend
    with ``supported_in_api: false`` because it isn't in the public OpenAI
    API. The Codex CLI / OAuth route still serves it for ChatGPT Pro
    accounts, so we must not drop it on that flag. visibility=hidden is
    the separate signal that *should* still filter entries out.
    """
    import sys
    from hermes_cli import codex_models

    class _FakeResp:
        status_code = 200

        def json(self):
            return {
                "models": [
                    {"slug": "gpt-5.5", "priority": 0, "supported_in_api": True},
                    {"slug": "gpt-5.3-codex-spark", "priority": 7, "supported_in_api": False},
                    {"slug": "gpt-5-internal", "priority": 99, "visibility": "hidden"},
                ]
            }

    class _FakeHttpx:
        @staticmethod
        def get(url, headers=None, timeout=None):
            return _FakeResp()

    monkeypatch.setitem(sys.modules, "httpx", _FakeHttpx)

    models = codex_models._fetch_models_from_api(access_token="tok")

    assert "gpt-5.5" in models
    assert "gpt-5.3-codex-spark" in models
    assert "gpt-5-internal" not in models






def test_model_command_prompts_to_reuse_or_reauthenticate_codex_session(monkeypatch, capsys):
    from hermes_cli.main import _model_flow_openai_codex

    captured = {"login_calls": 0}
    choices = iter(["2"])

    monkeypatch.setattr("builtins.input", lambda prompt="": next(choices))
    monkeypatch.setattr(
        "hermes_cli.auth.get_codex_auth_status",
        lambda: {"logged_in": True, "source": "hermes-auth-store"},
    )
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_codex_runtime_credentials",
        lambda *args, **kwargs: {"api_key": "fresh-codex-token"},
    )

    def _fake_login(*args, force_new_login=False, **kwargs):
        captured["login_calls"] += 1
        captured["force_new_login"] = force_new_login

    monkeypatch.setattr("hermes_cli.auth._login_openai_codex", _fake_login)
    monkeypatch.setattr(
        "hermes_cli.codex_models.get_codex_model_ids",
        lambda access_token=None: ["gpt-5.4", "gpt-5.3-codex"],
    )
    monkeypatch.setattr(
        "hermes_cli.auth._prompt_model_selection",
        lambda model_ids, current_model="", **_kwargs: None,
    )

    _model_flow_openai_codex({}, current_model="gpt-5.4")

    out = capsys.readouterr().out
    assert "Use existing credentials" in out
    assert "Reauthenticate (new OAuth login)" in out
    assert captured["login_calls"] == 1
    assert captured["force_new_login"] is True


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

