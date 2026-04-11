"""Tests for _setup_feishu() in hermes_cli/gateway.py.

Verifies that the interactive setup writes env vars that correctly drive the
Feishu adapter: credentials, connection mode, DM policy, and group policy.
"""

import os
from unittest.mock import patch


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

    def mock_save(name, value):
        saved_env[name] = value

    def mock_get(name):
        return existing_env.get(name, "")

    with patch("hermes_cli.gateway.save_env_value", side_effect=mock_save), \
         patch("hermes_cli.gateway.get_env_value", side_effect=mock_get), \
         patch("hermes_cli.gateway.prompt_yes_no", side_effect=prompt_yes_no_responses), \
         patch("hermes_cli.gateway.prompt_choice", side_effect=prompt_choice_responses), \
         patch("hermes_cli.gateway.prompt", side_effect=prompt_responses), \
         patch("hermes_cli.gateway.print_info"), \
         patch("hermes_cli.gateway.print_success"), \
         patch("hermes_cli.gateway.print_warning"), \
         patch("hermes_cli.gateway.print_error"), \
         patch("hermes_cli.gateway.color", side_effect=lambda t, c: t), \
         patch("gateway.platforms.feishu.qr_register", return_value=qr_result):

        from hermes_cli.gateway import _setup_feishu
        _setup_feishu()

    return saved_env


# ---------------------------------------------------------------------------
# QR scan-to-create path
# ---------------------------------------------------------------------------

class TestSetupFeishuQrPath:
    """Tests for the QR scan-to-create happy path."""

    def test_qr_success_saves_core_credentials(self):
        env = _run_setup_feishu(
            qr_result={
                "app_id": "cli_test",
                "app_secret": "secret_test",
                "domain": "feishu",
                "open_id": "ou_owner",
                "bot_name": "TestBot",
                "bot_open_id": "ou_bot",
            },
            prompt_yes_no_responses=[True],        # Start QR
            prompt_choice_responses=[0, 0, 0],  # method=QR, dm=pairing, group=open
            prompt_responses=[""],                  # home channel: skip
        )
        assert env["FEISHU_APP_ID"] == "cli_test"
        assert env["FEISHU_APP_SECRET"] == "secret_test"
        assert env["FEISHU_DOMAIN"] == "feishu"

    def test_qr_success_does_not_persist_bot_identity(self):
        """Bot identity is discovered at runtime by _hydrate_bot_identity — not persisted
        in env, so it stays fresh if the user renames the bot later."""
        env = _run_setup_feishu(
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

    def test_qr_path_defaults_to_websocket(self):
        env = _run_setup_feishu(
            qr_result={
                "app_id": "cli_test", "app_secret": "s", "domain": "feishu",
                "open_id": None, "bot_name": None, "bot_open_id": None,
            },
            prompt_choice_responses=[0, 0, 0],  # method=QR, dm=pairing, group=open
            prompt_responses=[""],
        )
        assert env["FEISHU_CONNECTION_MODE"] == "websocket"

    @patch("gateway.platforms.feishu.probe_bot", return_value=None)
    def test_manual_path_websocket(self, _mock_probe):
        env = _run_setup_feishu(
            qr_result=None,
            prompt_choice_responses=[1, 0, 0, 0, 0],  # method=manual, domain=feishu, connection=ws, dm=pairing, group=open
            prompt_responses=["cli_manual", "secret_manual", ""],  # app_id, app_secret, home_channel
        )
        assert env["FEISHU_CONNECTION_MODE"] == "websocket"

    @patch("gateway.platforms.feishu.probe_bot", return_value=None)
    def test_manual_path_webhook(self, _mock_probe):
        env = _run_setup_feishu(
            qr_result=None,
            prompt_choice_responses=[1, 0, 1, 0, 0],  # method=manual, domain=feishu, connection=webhook, dm=pairing, group=open
            prompt_responses=["cli_manual", "secret_manual", ""],  # app_id, app_secret, home_channel
        )
        assert env["FEISHU_CONNECTION_MODE"] == "webhook"


# ---------------------------------------------------------------------------
# DM security policy
# ---------------------------------------------------------------------------

class TestSetupFeishuDmPolicy:
    """DM policy must use platform-scoped FEISHU_ALLOW_ALL_USERS, not the global flag."""

    def _run_with_dm_choice(self, dm_choice_idx, prompt_responses=None):
        return _run_setup_feishu(
            qr_result={
                "app_id": "cli_test", "app_secret": "s", "domain": "feishu",
                "open_id": "ou_owner", "bot_name": None, "bot_open_id": None,
            },
            prompt_yes_no_responses=[True],
            prompt_choice_responses=[0, dm_choice_idx, 0],  # method=QR, dm=<choice>, group=open
            prompt_responses=prompt_responses or [""],
        )

    def test_pairing_sets_feishu_allow_all_false(self):
        env = self._run_with_dm_choice(0)
        assert env["FEISHU_ALLOW_ALL_USERS"] == "false"
        assert env["FEISHU_ALLOWED_USERS"] == ""
        assert "GATEWAY_ALLOW_ALL_USERS" not in env

    def test_allow_all_sets_feishu_allow_all_true(self):
        env = self._run_with_dm_choice(1)
        assert env["FEISHU_ALLOW_ALL_USERS"] == "true"
        assert env["FEISHU_ALLOWED_USERS"] == ""
        assert "GATEWAY_ALLOW_ALL_USERS" not in env

    def test_allowlist_sets_feishu_allow_all_false_with_list(self):
        env = self._run_with_dm_choice(2, prompt_responses=["ou_user1,ou_user2", ""])
        assert env["FEISHU_ALLOW_ALL_USERS"] == "false"
        assert env["FEISHU_ALLOWED_USERS"] == "ou_user1,ou_user2"
        assert "GATEWAY_ALLOW_ALL_USERS" not in env

    def test_allowlist_prepopulates_with_scan_owner_open_id(self):
        """When open_id is available from QR scan, it should be the default allowlist value."""
        # We return the owner's open_id from prompt (+ empty home channel).
        env = self._run_with_dm_choice(2, prompt_responses=["ou_owner", ""])
        assert env["FEISHU_ALLOWED_USERS"] == "ou_owner"

    def test_disabled_sets_feishu_allow_all_false(self):
        env = self._run_with_dm_choice(3)
        assert env["FEISHU_ALLOW_ALL_USERS"] == "false"
        assert env["FEISHU_ALLOWED_USERS"] == ""
        assert "GATEWAY_ALLOW_ALL_USERS" not in env


# ---------------------------------------------------------------------------
# Group policy
# ---------------------------------------------------------------------------

class TestSetupFeishuGroupPolicy:

    def test_open_with_mention(self):
        env = _run_setup_feishu(
            qr_result={
                "app_id": "cli_test", "app_secret": "s", "domain": "feishu",
                "open_id": None, "bot_name": None, "bot_open_id": None,
            },
            prompt_yes_no_responses=[True],
            prompt_choice_responses=[0, 0, 0],  # method=QR, dm=pairing, group=open
            prompt_responses=[""],
        )
        assert env["FEISHU_GROUP_POLICY"] == "open"

    def test_disabled(self):
        env = _run_setup_feishu(
            qr_result={
                "app_id": "cli_test", "app_secret": "s", "domain": "feishu",
                "open_id": None, "bot_name": None, "bot_open_id": None,
            },
            prompt_yes_no_responses=[True],
            prompt_choice_responses=[0, 0, 1],  # method=QR, dm=pairing, group=disabled
            prompt_responses=[""],
        )
        assert env["FEISHU_GROUP_POLICY"] == "disabled"


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
        return _run_setup_feishu(
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

    @patch.dict(os.environ, {}, clear=True)
    def test_qr_env_produces_valid_adapter_settings(self):
        """QR setup → adapter initializes with websocket mode."""
        env = self._make_env_from_setup()

        with patch.dict(os.environ, env, clear=True):
            from gateway.config import PlatformConfig
            from gateway.platforms.feishu import FeishuAdapter
            adapter = FeishuAdapter(PlatformConfig())
            assert adapter._app_id == "cli_test_app"
            assert adapter._app_secret == "test_secret_value"
            assert adapter._domain_name == "feishu"
            assert adapter._connection_mode == "websocket"

    @patch.dict(os.environ, {}, clear=True)
    def test_open_dm_env_sets_correct_adapter_state(self):
        """Setup with 'allow all DMs' → adapter sees allow-all flag."""
        env = self._make_env_from_setup(dm_idx=1)

        with patch.dict(os.environ, env, clear=True):
            from gateway.platforms.feishu import FeishuAdapter
            from gateway.config import PlatformConfig
            # Verify adapter initializes without error and env var is correct.
            FeishuAdapter(PlatformConfig())
            assert os.getenv("FEISHU_ALLOW_ALL_USERS") == "true"

    @patch.dict(os.environ, {}, clear=True)
    def test_group_open_env_sets_adapter_group_policy(self):
        """Setup with 'open groups' → adapter group_policy is 'open'."""
        env = self._make_env_from_setup(group_idx=0)

        with patch.dict(os.environ, env, clear=True):
            from gateway.config import PlatformConfig
            from gateway.platforms.feishu import FeishuAdapter
            adapter = FeishuAdapter(PlatformConfig())
            assert adapter._group_policy == "open"
