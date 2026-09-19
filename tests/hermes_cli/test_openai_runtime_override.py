"""model.openai_runtime: codex_app_server must survive runtime resolution (#115169).

The rewrite was only applied on the credential-pool rung; the OAuth path (the
normal openai-codex route) and explicit-key paths silently dropped it.
"""

from __future__ import annotations

from unittest.mock import patch

from hermes_cli import runtime_provider as rp

FAKE_CODEX_CREDS = {
    "base_url": "https://chatgpt.com/backend-api/codex",
    "api_key": "tok",
    "source": "hermes-auth-store",
    "last_refresh": 1,
}


def _model_cfg(runtime_value):
    cfg = {"provider": "openai-codex", "default": "gpt-5.6-luna"}
    if runtime_value is not None:
        cfg["openai_runtime"] = runtime_value
    return cfg


class TestOAuthPathHonorsOpenaiRuntime:
    def test_codex_app_server_applied(self):
        with patch.object(rp, "resolve_codex_runtime_credentials", return_value=dict(FAKE_CODEX_CREDS)):
            rt = rp._resolve_oauth_runtime("openai-codex", "openai-codex",
                                           _model_cfg("codex_app_server"), None)
        assert rt["api_mode"] == "codex_app_server"

    def test_unset_stays_codex_responses(self):
        with patch.object(rp, "resolve_codex_runtime_credentials", return_value=dict(FAKE_CODEX_CREDS)):
            rt = rp._resolve_oauth_runtime("openai-codex", "openai-codex",
                                           _model_cfg(None), None)
        assert rt["api_mode"] == "codex_responses"

    def test_auto_stays_codex_responses(self):
        with patch.object(rp, "resolve_codex_runtime_credentials", return_value=dict(FAKE_CODEX_CREDS)):
            rt = rp._resolve_oauth_runtime("openai-codex", "openai-codex",
                                           _model_cfg("auto"), None)
        assert rt["api_mode"] == "codex_responses"


class TestFullLadderHonorsOpenaiRuntime:
    def test_codex_app_server_survives_ladder(self):
        # The hermetic conftest points HERMES_HOME at an empty tempdir, so the
        # ambient config cannot carry the flag — inject model_cfg at the same
        # seam the rest of this suite uses (monkeypatch.setattr(rp, "_get_model_config", ...)).
        with patch.object(rp, "resolve_codex_runtime_credentials", return_value=dict(FAKE_CODEX_CREDS)), \
             patch.object(rp, "_get_model_config", return_value=_model_cfg("codex_app_server")):
            rt = rp.resolve_runtime_provider(requested="openai-codex")
        assert rt["provider"] == "openai-codex"
        assert rt["api_mode"] == "codex_app_server"


class TestNonEligibleProvidersUnaffected:
    def test_actual_route_clobber_unchanged(self):
        # provider "actual" is not eligible for the openai_runtime rewrite; the
        # is_actual_route chat_completions normalization must keep working.
        rt = rp._runtime("actual", "codex_app_server", "https://api.actual.inc/v1", "k")
        assert rt["api_mode"] == "chat_completions"
