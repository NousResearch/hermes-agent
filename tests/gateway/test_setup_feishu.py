"""Tests for _setup_feishu() in hermes_cli/gateway.py.

Verifies that the interactive setup writes env vars that correctly drive the
Feishu adapter: credentials, connection mode, DM policy, and group policy.
"""

import os
import urllib.error
from unittest.mock import patch

import pytest


class _FakeHTTPResponse:
    def __init__(self, body: bytes):
        self.body = body
        self.read_calls: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.read_calls.append(size)
        if size is None or size < 0:
            return self.body
        return self.body[:size]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_setup_feishu(
    *,
    qr_result=None,
    prompt_yes_no_responses=None,
    prompt_choice_responses=None,
    prompt_responses=None,
    existing_env=None,
):
    """Run _setup_feishu() with mocked I/O and return the env vars that were saved.

    Returns a dict of {env_var_name: value} for all save_env_value calls.
    """
    existing_env = existing_env or {}
    prompt_yes_no_responses = list(prompt_yes_no_responses or [True])
    # QR path: method(0), dm(0), group(0) — 3 choices (no connection mode)
    # Manual path: method(1), domain(0), connection(0), dm(0), group(0) — 5 choices
    prompt_choice_responses = list(prompt_choice_responses or [0, 0, 0])
    prompt_responses = list(prompt_responses or [""])

    saved_env = {}
    removed_keys = []

    def mock_save(name, value):
        saved_env[name] = value

    def mock_get(name):
        return existing_env.get(name, "")

    def mock_remove(name):
        removed_keys.append(name)
        if name in existing_env:
            del existing_env[name]
            return True
        if name in saved_env:
            del saved_env[name]
            return True
        return False

    with patch("hermes_cli.config.save_env_value", side_effect=mock_save), \
         patch("hermes_cli.config.get_env_value", side_effect=mock_get), \
         patch("hermes_cli.config.remove_env_value", side_effect=mock_remove), \
         patch("hermes_cli.cli_output.prompt_yes_no", side_effect=prompt_yes_no_responses), \
         patch("hermes_cli.setup.prompt_choice", side_effect=prompt_choice_responses), \
         patch("hermes_cli.cli_output.prompt", side_effect=prompt_responses), \
         patch("hermes_cli.cli_output.print_header"), \
         patch("hermes_cli.cli_output.print_info"), \
         patch("hermes_cli.cli_output.print_success"), \
         patch("hermes_cli.cli_output.print_warning"), \
         patch("hermes_cli.cli_output.print_error"), \
         patch("plugins.platforms.feishu.adapter.qr_register", return_value=qr_result):

        from plugins.platforms.feishu.adapter import interactive_setup
        interactive_setup()

    return saved_env, removed_keys


# ---------------------------------------------------------------------------
# QR scan-to-create path
# ---------------------------------------------------------------------------

class TestSetupFeishuQrPath:
    """Tests for the QR scan-to-create happy path."""

    def test_post_registration_bounds_success_response(self, monkeypatch):
        from plugins.platforms.feishu import adapter as feishu

        monkeypatch.setattr(feishu, "_FEISHU_ONBOARD_JSON_BODY_MAX_BYTES", 8)
        response = _FakeHTTPResponse(b"x" * 9)

        with patch("plugins.platforms.feishu.adapter.urlopen", return_value=response):
            with pytest.raises(
                feishu._FeishuOnboardResponseTooLarge,
                match="Feishu registration response body exceeded 8 bytes",
            ):
                feishu._post_registration("https://accounts.example", {"action": "init"})

        assert response.read_calls == [9]

    def test_post_registration_bounds_http_error_response(self, monkeypatch):
        from plugins.platforms.feishu import adapter as feishu

        monkeypatch.setattr(feishu, "_FEISHU_ONBOARD_ERROR_BODY_MAX_BYTES", 8)
        response = _FakeHTTPResponse(b"x" * 9)
        http_error = urllib.error.HTTPError(
            url="https://accounts.example/oauth/v1/app/registration",
            code=400,
            msg="Bad Request",
            hdrs={},
            fp=response,
        )

        with patch("plugins.platforms.feishu.adapter.urlopen", side_effect=http_error):
            with pytest.raises(urllib.error.HTTPError):
                feishu._post_registration("https://accounts.example", {"action": "poll"})

        assert response.read_calls == [9]

    def test_probe_bot_http_bounds_token_response(self, monkeypatch):
        from plugins.platforms.feishu import adapter as feishu

        monkeypatch.setattr(feishu, "_FEISHU_ONBOARD_JSON_BODY_MAX_BYTES", 8)
        response = _FakeHTTPResponse(b"x" * 9)

        with patch("plugins.platforms.feishu.adapter.urlopen", return_value=response):
            assert feishu._probe_bot_http("cli_test", "secret_test", "feishu") is None

        assert response.read_calls == [9]

    def test_probe_bot_http_bounds_bot_info_response(self, monkeypatch):
        import json

        from plugins.platforms.feishu import adapter as feishu

        monkeypatch.setattr(feishu, "_FEISHU_ONBOARD_JSON_BODY_MAX_BYTES", 64)
        token_response = _FakeHTTPResponse(
            json.dumps({"tenant_access_token": "tenant-token"}).encode()
        )
        bot_response = _FakeHTTPResponse(b"x" * 65)

        with patch(
            "plugins.platforms.feishu.adapter.urlopen",
            side_effect=[token_response, bot_response],
        ):
            assert feishu._probe_bot_http("cli_test", "secret_test", "feishu") is None

        assert token_response.read_calls == [65]
        assert bot_response.read_calls == [65]

    def test_qr_success_does_not_persist_bot_identity(self):
        """Bot identity is discovered at runtime by _hydrate_bot_identity — not persisted
        in env, so it stays fresh if the user renames the bot later."""
        env, _ = _run_setup_feishu(
            qr_result={
                "app_id": "cli_test",
                "app_secret": "secret_test",
                "domain": "feishu",
                "open_id": "ou_owner",
                "bot_name": "TestBot",
                "bot_open_id": "ou_bot",
            },
            prompt_yes_no_responses=[True],
            prompt_choice_responses=[0, 0, 0],
            prompt_responses=[""],
        )
        assert "FEISHU_BOT_OPEN_ID" not in env
        assert "FEISHU_BOT_NAME" not in env


# ---------------------------------------------------------------------------
# Connection mode
# ---------------------------------------------------------------------------

class TestSetupFeishuConnectionMode:
    """Connection mode: QR always websocket, manual path lets user choose."""


    @patch("plugins.platforms.feishu.adapter.probe_bot", return_value=None)
    def test_manual_path_websocket(self, _mock_probe):
        env, _ = _run_setup_feishu(
            qr_result=None,
            prompt_choice_responses=[1, 0, 0, 0, 0],  # method=manual, domain=feishu, connection=ws, dm=pairing, group=open
            prompt_responses=["cli_manual", "secret_manual", ""],  # app_id, app_secret, home_channel
        )
        assert env["FEISHU_CONNECTION_MODE"] == "websocket"


# ---------------------------------------------------------------------------
# DM security policy
# ---------------------------------------------------------------------------

class TestSetupFeishuDmPolicy:
    """DM policy must use platform-scoped FEISHU_ALLOW_ALL_USERS, not the global flag."""

    def _run_with_dm_choice(self, dm_choice_idx, prompt_responses=None):
        env, _ = _run_setup_feishu(
            qr_result={
                "app_id": "cli_test", "app_secret": "s", "domain": "feishu",
                "open_id": "ou_owner", "bot_name": None, "bot_open_id": None,
            },
            prompt_yes_no_responses=[True],
            prompt_choice_responses=[0, dm_choice_idx, 0],  # method=QR, dm=<choice>, group=open
            prompt_responses=prompt_responses or [""],
        )
        return env


    def test_allowlist_sets_feishu_allow_all_false_with_list(self):
        env = self._run_with_dm_choice(2, prompt_responses=["ou_user1,ou_user2", ""])
        assert env["FEISHU_ALLOW_ALL_USERS"] == "false"
        assert env["FEISHU_ALLOWED_USERS"] == "ou_user1,ou_user2"
        assert "GATEWAY_ALLOW_ALL_USERS" not in env


# ---------------------------------------------------------------------------
# Group policy
# ---------------------------------------------------------------------------

class TestSetupFeishuGroupPolicy:

    def test_open_with_mention(self):
        env, _ = _run_setup_feishu(
            qr_result={
                "app_id": "cli_test", "app_secret": "s", "domain": "feishu",
                "open_id": None, "bot_name": None, "bot_open_id": None,
            },
            prompt_yes_no_responses=[True],
            prompt_choice_responses=[0, 0, 0],  # method=QR, dm=pairing, group=open
            prompt_responses=[""],
        )
        assert env["FEISHU_GROUP_POLICY"] == "open"


# ---------------------------------------------------------------------------
# Home channel (optional clear — Issue #12423)
# ---------------------------------------------------------------------------

class TestSetupFeishuHomeChannel:
    """Blank home-channel answer must clear FEISHU_HOME_CHANNEL."""

    def test_blank_removes_existing_home_channel(self):
        env, removed = _run_setup_feishu(
            qr_result={
                "app_id": "cli_test", "app_secret": "s", "domain": "feishu",
                "open_id": None, "bot_name": None, "bot_open_id": None,
            },
            prompt_yes_no_responses=[True],
            prompt_choice_responses=[0, 0, 0],
            prompt_responses=[""],
            existing_env={"FEISHU_HOME_CHANNEL": "chat_old"},
        )
        assert "FEISHU_HOME_CHANNEL" in removed
        assert "FEISHU_HOME_CHANNEL" not in env


# ---------------------------------------------------------------------------
# Adapter integration: env vars → FeishuAdapterSettings
# ---------------------------------------------------------------------------

class TestSetupFeishuAdapterIntegration:
    """Verify that env vars written by _setup_feishu() produce a valid adapter config.

    This bridges the gap between 'setup wrote the right env vars' and
    'the adapter will actually initialize correctly from those vars'.
    """

    def _make_env_from_setup(self, dm_idx=0, group_idx=0):
        """Run _setup_feishu via QR path and return the env vars it would write."""
        env, _ = _run_setup_feishu(
            qr_result={
                "app_id": "cli_test_app",
                "app_secret": "test_secret_value",
                "domain": "feishu",
                "open_id": "ou_owner",
                "bot_name": "IntegrationBot",
                "bot_open_id": "ou_bot_integration",
            },
            prompt_yes_no_responses=[True],
            prompt_choice_responses=[0, dm_idx, group_idx],  # method=QR, dm, group
            prompt_responses=[""],
        )
        return env

    @patch.dict(os.environ, {}, clear=True)
    def test_qr_env_produces_valid_adapter_settings(self):
        """QR setup → adapter initializes with websocket mode."""
        env = self._make_env_from_setup()

        with patch.dict(os.environ, env, clear=True):
            from gateway.config import PlatformConfig
            from plugins.platforms.feishu.adapter import FeishuAdapter
            adapter = FeishuAdapter(PlatformConfig())
            assert adapter._app_id == "cli_test_app"
            assert adapter._app_secret == "test_secret_value"
            assert adapter._domain_name == "feishu"
            assert adapter._connection_mode == "websocket"


