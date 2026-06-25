"""Tests for set_config_value — verifying secrets route to .env and config to config.yaml."""

import argparse
import os
from unittest.mock import patch

import pytest

from hermes_cli.config import set_config_value, config_command


@pytest.fixture(autouse=True)
def _isolated_hermes_home(tmp_path):
    """Point HERMES_HOME at a temp dir so tests never touch real config."""
    env_file = tmp_path / ".env"
    env_file.touch()
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        yield tmp_path


def _read_env(tmp_path):
    return (tmp_path / ".env").read_text()


def _read_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    return config_path.read_text() if config_path.exists() else ""


# ---------------------------------------------------------------------------
# Explicit allowlist keys → .env
# ---------------------------------------------------------------------------

class TestEnvRouting:
    """Known env vars should always go to .env."""

    @pytest.mark.parametrize("key", [
        'AGENT_BROWSER_ARGS',
        'ALIBABA_CODING_PLAN_API_KEY',
        'ALIBABA_CODING_PLAN_BASE_URL',
        'ANTHROPIC_API_KEY',
        'ANTHROPIC_BASE_URL',
        'ANTHROPIC_TOKEN',
        'API_SERVER_CORS_ORIGINS',
        'API_SERVER_ENABLED',
        'API_SERVER_HOST',
        'API_SERVER_KEY',
        'API_SERVER_MODEL_NAME',
        'API_SERVER_PORT',
        'ARCEEAI_API_KEY',
        'ARCEE_BASE_URL',
        'AUXILIARY_VISION_API_KEY',
        'AUXILIARY_VISION_BASE_URL',
        'AUXILIARY_VISION_MODEL',
        'AUXILIARY_VISION_PROVIDER',
        'AUXILIARY_WEB_EXTRACT_API_KEY',
        'AUXILIARY_WEB_EXTRACT_BASE_URL',
        'AUXILIARY_WEB_EXTRACT_MODEL',
        'AUXILIARY_WEB_EXTRACT_PROVIDER',
        'AWS_PROFILE',
        'AWS_REGION',
        'AZURE_ANTHROPIC_KEY',
        'AZURE_AUTHORITY_HOST',
        'AZURE_CLIENT_CERTIFICATE_PATH',
        'AZURE_CLIENT_ID',
        'AZURE_CLIENT_SECRET',
        'AZURE_FEDERATED_TOKEN_FILE',
        'AZURE_FOUNDRY_API_KEY',
        'AZURE_FOUNDRY_BASE_URL',
        'AZURE_TENANT_ID',
        'BEDROCK_BASE_URL',
        'BLUEBUBBLES_ALLOWED_USERS',
        'BLUEBUBBLES_ALLOW_ALL_USERS',
        'BLUEBUBBLES_HOME_CHANNEL',
        'BLUEBUBBLES_PASSWORD',
        'BLUEBUBBLES_SERVER_URL',
        'BLUEBUBBLES_WEBHOOK_HOST',
        'BLUEBUBBLES_WEBHOOK_PORT',
        'BROWSERBASE_API_KEY',
        'BROWSERBASE_PROJECT_ID',
        'BROWSER_CDP_URL',
        'BROWSER_INACTIVITY_TIMEOUT',
        'BROWSER_USE_API_KEY',
        'CAMOFOX_ADOPT_EXISTING_TAB',
        'CAMOFOX_SESSION_KEY',
        'CAMOFOX_URL',
        'CAMOFOX_USER_ID',
        'CLAUDE_CODE_OAUTH_TOKEN',
        'CODEX_HOME',
        'COPILOT_ACP_BASE_URL',
        'COPILOT_API_BASE_URL',
        'COPILOT_CLI_PATH',
        'COPILOT_GITHUB_TOKEN',
        'DASHSCOPE_API_KEY',
        'DASHSCOPE_BASE_URL',
        'DAYTONA_API_KEY',
        'DEEPSEEK_API_KEY',
        'DEEPSEEK_BASE_URL',
        'DINGTALK_ALLOWED_USERS',
        'DINGTALK_CLIENT_ID',
        'DINGTALK_CLIENT_SECRET',
        'DISCORD_ALLOWED_CHANNELS',
        'DISCORD_ALLOWED_ROLES',
        'DISCORD_ALLOWED_USERS',
        'DISCORD_ALLOW_ANY_ATTACHMENT',
        'DISCORD_ALLOW_MENTION_EVERYONE',
        'DISCORD_ALLOW_MENTION_REPLIED_USER',
        'DISCORD_ALLOW_MENTION_ROLES',
        'DISCORD_ALLOW_MENTION_USERS',
        'DISCORD_AUTO_THREAD',
        'DISCORD_BOT_TOKEN',
        'DISCORD_COMMAND_SYNC_POLICY',
        'DISCORD_FREE_RESPONSE_CHANNELS',
        'DISCORD_HOME_CHANNEL',
        'DISCORD_HOME_CHANNEL_NAME',
        'DISCORD_IGNORED_CHANNELS',
        'DISCORD_MAX_ATTACHMENT_BYTES',
        'DISCORD_NO_THREAD_CHANNELS',
        'DISCORD_PROXY',
        'DISCORD_REACTIONS',
        'DISCORD_REPLY_TO_MODE',
        'DISCORD_REQUIRE_MENTION',
        'ELEVENLABS_API_KEY',
        'EMAIL_ADDRESS',
        'EMAIL_ALLOWED_USERS',
        'EMAIL_ALLOW_ALL_USERS',
        'EMAIL_HOME_ADDRESS',
        'EMAIL_HOME_ADDRESS_NAME',
        'EMAIL_IMAP_HOST',
        'EMAIL_IMAP_PORT',
        'EMAIL_PASSWORD',
        'EMAIL_POLL_INTERVAL',
        'EMAIL_SMTP_HOST',
        'EMAIL_SMTP_PORT',
        'EXA_API_KEY',
        'FAL_KEY',
        'FEISHU_ALLOWED_USERS',
        'FEISHU_ALLOW_BOTS',
        'FEISHU_APP_ID',
        'FEISHU_APP_SECRET',
        'FEISHU_CONNECTION_MODE',
        'FEISHU_DOMAIN',
        'FEISHU_ENCRYPT_KEY',
        'FEISHU_HOME_CHANNEL',
        'FEISHU_REQUIRE_MENTION',
        'FEISHU_VERIFICATION_TOKEN',
        'FIRECRAWL_API_KEY',
        'FIRECRAWL_API_URL',
        'FIRECRAWL_BROWSER_TTL',
        'FIRECRAWL_GATEWAY_URL',
        'GATEWAY_ALLOWED_USERS',
        'GATEWAY_ALLOW_ALL_USERS',
        'GATEWAY_PROXY_KEY',
        'GATEWAY_PROXY_URL',
        'GATEWAY_RELAY_BOT_ID',
        'GATEWAY_RELAY_DELIVERY_KEY',
        'GATEWAY_RELAY_ENDPOINT',
        'GATEWAY_RELAY_ENROLL_TOKEN',
        'GATEWAY_RELAY_ID',
        'GATEWAY_RELAY_PLATFORM',
        'GATEWAY_RELAY_ROUTE_KEYS',
        'GATEWAY_RELAY_SECRET',
        'GATEWAY_RELAY_URL',
        'GEMINI_API_KEY',
        'GEMINI_BASE_URL',
        'GH_TOKEN',
        'GITHUB_TOKEN',
        'GLM_API_KEY',
        'GLM_BASE_URL',
        'GMI_API_KEY',
        'GMI_BASE_URL',
        'GOOGLE_API_KEY',
        'GOOGLE_APPLICATION_CREDENTIALS',
        'GOOGLE_CHAT_ALLOWED_USERS',
        'GOOGLE_CHAT_ALLOW_ALL_USERS',
        'GOOGLE_CHAT_BOOTSTRAP_SPACES',
        'GOOGLE_CHAT_DEBUG_RAW',
        'GOOGLE_CHAT_HOME_CHANNEL',
        'GOOGLE_CHAT_HOME_CHANNEL_NAME',
        'GOOGLE_CHAT_MAX_BYTES',
        'GOOGLE_CHAT_MAX_MESSAGES',
        'GOOGLE_CHAT_PROJECT_ID',
        'GOOGLE_CHAT_SERVICE_ACCOUNT_JSON',
        'GOOGLE_CHAT_SUBSCRIPTION',
        'GOOGLE_CHAT_SUBSCRIPTION_NAME',
        'GOOGLE_CLOUD_PROJECT',
        'GROQ_API_KEY',
        'GROQ_BASE_URL',
        'HASS_TOKEN',
        'HASS_URL',
        'HERMES_API_CALL_STALE_TIMEOUT',
        'HERMES_API_TIMEOUT',
        'HERMES_COPILOT_ACP_ARGS',
        'HERMES_COPILOT_ACP_COMMAND',
        'HERMES_CRON_MAX_PARALLEL',
        'HERMES_CRON_SCRIPT_TIMEOUT',
        'HERMES_CRON_TIMEOUT',
        'HERMES_DASHBOARD_BASIC_AUTH_PASSWORD',
        'HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH',
        'HERMES_DASHBOARD_BASIC_AUTH_SECRET',
        'HERMES_DASHBOARD_BASIC_AUTH_TTL_SECONDS',
        'HERMES_DASHBOARD_BASIC_AUTH_USERNAME',
        'HERMES_DASHBOARD_OAUTH_CLIENT_ID',
        'HERMES_DASHBOARD_OIDC_CLIENT_ID',
        'HERMES_DASHBOARD_OIDC_ISSUER',
        'HERMES_DASHBOARD_OIDC_SCOPES',
        'HERMES_DASHBOARD_PUBLIC_URL',
        'HERMES_DUMP_REQUESTS',
        'HERMES_DUMP_REQUEST_STDOUT',
        'HERMES_INFERENCE_MODEL',
        'HERMES_KANBAN_BOARD',
        'HERMES_KANBAN_DB',
        'HERMES_KANBAN_DISPATCH_IN_GATEWAY',
        'HERMES_KANBAN_HOME',
        'HERMES_KANBAN_TASK',
        'HERMES_KANBAN_WORKSPACES_ROOT',
        'HERMES_LANGFUSE_BASE_URL',
        'HERMES_LANGFUSE_DEBUG',
        'HERMES_LANGFUSE_ENV',
        'HERMES_LANGFUSE_MAX_CHARS',
        'HERMES_LANGFUSE_PUBLIC_KEY',
        'HERMES_LANGFUSE_RELEASE',
        'HERMES_LANGFUSE_SAMPLE_RATE',
        'HERMES_LANGFUSE_SECRET_KEY',
        'HERMES_LOCAL_STT_COMMAND',
        'HERMES_LOCAL_STT_LANGUAGE',
        'HERMES_MAX_ITERATIONS',
        'HERMES_MODEL',
        'HERMES_NOUS_MIN_KEY_TTL_SECONDS',
        'HERMES_NOUS_TIMEOUT_SECONDS',
        'HERMES_PORTAL_BASE_URL',
        'HERMES_PREFILL_MESSAGES_FILE',
        'HERMES_REDACT_SECRETS',
        'HERMES_STREAM_READ_TIMEOUT',
        'HERMES_STREAM_RETRIES',
        'HERMES_STREAM_STALE_TIMEOUT',
        'HERMES_TIMEZONE',
        'HERMES_TOOL_PROGRESS',
        'HERMES_TOOL_PROGRESS_MODE',
        'HERMES_VISION_DOWNLOAD_TIMEOUT',
        'HERMES_YOLO_MODE',
        'HF_BASE_URL',
        'HF_TOKEN',
        'HONCHO_API_KEY',
        'HONCHO_BASE_URL',
        'IDENTITY_ENDPOINT',
        'KILOCODE_API_KEY',
        'KILOCODE_BASE_URL',
        'KIMI_API_KEY',
        'KIMI_BASE_URL',
        'KIMI_CN_API_KEY',
        'KIMI_CODING_API_KEY',
        'LINE_ALLOWED_GROUPS',
        'LINE_ALLOWED_ROOMS',
        'LINE_ALLOWED_USERS',
        'LINE_ALLOW_ALL_USERS',
        'LINE_BUTTON_LABEL',
        'LINE_CHANNEL_ACCESS_TOKEN',
        'LINE_CHANNEL_SECRET',
        'LINE_DELIVERED_TEXT',
        'LINE_HOME_CHANNEL',
        'LINE_HOST',
        'LINE_INTERRUPTED_TEXT',
        'LINE_PENDING_TEXT',
        'LINE_PORT',
        'LINE_PUBLIC_URL',
        'LINE_SLOW_RESPONSE_THRESHOLD',
        'LM_API_KEY',
        'LM_BASE_URL',
        'MATRIX_ACCESS_TOKEN',
        'MATRIX_ALLOWED_ROOMS',
        'MATRIX_ALLOWED_USERS',
        'MATRIX_ALLOW_PUBLIC_ROOMS',
        'MATRIX_ALLOW_ROOM_MENTIONS',
        'MATRIX_APPROVAL_REQUIRE_SENDER',
        'MATRIX_APPROVAL_TIMEOUT_SECONDS',
        'MATRIX_AUTO_THREAD',
        'MATRIX_DEVICE_ID',
        'MATRIX_DM_MENTION_THREADS',
        'MATRIX_E2EE_MODE',
        'MATRIX_ENCRYPTION',
        'MATRIX_FREE_RESPONSE_ROOMS',
        'MATRIX_HOMESERVER',
        'MATRIX_HOME_ROOM',
        'MATRIX_IGNORE_USER_PATTERNS',
        'MATRIX_MAX_MEDIA_BYTES',
        'MATRIX_PASSWORD',
        'MATRIX_PROCESS_NOTICES',
        'MATRIX_REACTIONS',
        'MATRIX_RECOVERY_KEY',
        'MATRIX_RECOVERY_KEY_OUTPUT_FILE',
        'MATRIX_REQUIRE_MENTION',
        'MATRIX_SESSION_SCOPE',
        'MATRIX_TOOLS_ALLOW_CROSS_ROOM',
        'MATRIX_TOOLS_ALLOW_CROSS_ROOM_DESTRUCTIVE',
        'MATRIX_TOOLS_ALLOW_INVITES',
        'MATRIX_TOOLS_ALLOW_REDACTION',
        'MATRIX_TOOLS_ALLOW_ROOM_CREATE',
        'MATRIX_USER_ID',
        'MATTERMOST_ALLOWED_USERS',
        'MATTERMOST_FREE_RESPONSE_CHANNELS',
        'MATTERMOST_HOME_CHANNEL',
        'MATTERMOST_REPLY_MODE',
        'MATTERMOST_REQUIRE_MENTION',
        'MATTERMOST_TOKEN',
        'MATTERMOST_URL',
        'MINIMAX_API_KEY',
        'MINIMAX_BASE_URL',
        'MINIMAX_CN_API_KEY',
        'MINIMAX_CN_BASE_URL',
        'MISTRAL_API_KEY',
        'MSGRAPH_AUTHORITY_URL',
        'MSGRAPH_CLIENT_ID',
        'MSGRAPH_CLIENT_SECRET',
        'MSGRAPH_SCOPE',
        'MSGRAPH_TENANT_ID',
        'MSGRAPH_WEBHOOK_ACCEPTED_RESOURCES',
        'MSGRAPH_WEBHOOK_ALLOWED_SOURCE_CIDRS',
        'MSGRAPH_WEBHOOK_CLIENT_STATE',
        'MSGRAPH_WEBHOOK_ENABLED',
        'MSGRAPH_WEBHOOK_PORT',
        'MSI_ENDPOINT',
        'NOUS_BASE_URL',
        'NOUS_INFERENCE_BASE_URL',
        'NOVITA_API_KEY',
        'NOVITA_BASE_URL',
        'NVIDIA_API_KEY',
        'NVIDIA_BASE_URL',
        'OLLAMA_API_KEY',
        'OLLAMA_BASE_URL',
        'OPENAI_API_KEY',
        'OPENAI_BASE_URL',
        'OPENCODE_GO_API_KEY',
        'OPENCODE_GO_BASE_URL',
        'OPENCODE_ZEN_API_KEY',
        'OPENCODE_ZEN_BASE_URL',
        'OPENROUTER_API_KEY',
        'OPENROUTER_BASE_URL',
        'PARALLEL_API_KEY',
        'QQ_ALLOWED_USERS',
        'QQ_ALLOW_ALL_USERS',
        'QQ_APP_ID',
        'QQ_CLIENT_SECRET',
        'QQ_GROUP_ALLOWED_USERS',
        'QQ_HOME_CHANNEL',
        'QQ_HOME_CHANNEL_NAME',
        'QQ_PORTAL_HOST',
        'QQ_STT_API_KEY',
        'QQ_STT_BASE_URL',
        'QQ_STT_MODEL',
        'SEARXNG_URL',
        'SESSION_IDLE_MINUTES',
        'SESSION_RESET_HOUR',
        'SIGNAL_ACCOUNT',
        'SIGNAL_ALLOWED_USERS',
        'SIGNAL_ALLOW_ALL_USERS',
        'SIGNAL_GROUP_ALLOWED_USERS',
        'SIGNAL_HOME_CHANNEL_NAME',
        'SIGNAL_HTTP_URL',
        'SIGNAL_IGNORE_STORIES',
        'SLACK_ALLOWED_USERS',
        'SLACK_APP_TOKEN',
        'SLACK_BOT_TOKEN',
        'SLACK_HOME_CHANNEL',
        'SLACK_HOME_CHANNEL_NAME',
        'SMS_ALLOWED_USERS',
        'SMS_ALLOW_ALL_USERS',
        'SMS_HOME_CHANNEL',
        'SMS_HOME_CHANNEL_NAME',
        'SMS_INSECURE_NO_SIGNATURE',
        'SMS_WEBHOOK_HOST',
        'SMS_WEBHOOK_PORT',
        'SMS_WEBHOOK_URL',
        'STEPFUN_API_KEY',
        'STEPFUN_BASE_URL',
        'STT_GROQ_MODEL',
        'STT_OPENAI_BASE_URL',
        'STT_OPENAI_MODEL',
        'SUDO_PASSWORD',
        'SUPERMEMORY_API_KEY',
        'TAVILY_API_KEY',
        'TAVILY_BASE_URL',
        'TEAMS_CHANNEL_ID',
        'TEAMS_CHAT_ID',
        'TEAMS_DELIVERY_MODE',
        'TEAMS_GRAPH_ACCESS_TOKEN',
        'TEAMS_INCOMING_WEBHOOK_URL',
        'TEAMS_TEAM_ID',
        'TELEGRAM_ALLOWED_USERS',
        'TELEGRAM_BOT_TOKEN',
        'TELEGRAM_CRON_THREAD_ID',
        'TELEGRAM_EXCLUSIVE_BOT_MENTIONS',
        'TELEGRAM_GROUP_ALLOWED_CHATS',
        'TELEGRAM_GROUP_ALLOWED_USERS',
        'TELEGRAM_HOME_CHANNEL',
        'TELEGRAM_HOME_CHANNEL_NAME',
        'TELEGRAM_HOME_CHANNEL_THREAD_ID',
        'TELEGRAM_IGNORED_THREADS',
        'TELEGRAM_MENTION_PATTERNS',
        'TELEGRAM_PROXY',
        'TELEGRAM_REACTIONS',
        'TELEGRAM_REPLY_TO_MODE',
        'TELEGRAM_REQUIRE_MENTION',
        'TELEGRAM_WEBHOOK_PORT',
        'TELEGRAM_WEBHOOK_SECRET',
        'TELEGRAM_WEBHOOK_URL',
        'TERMINAL_CONTAINER_CPU',
        'TERMINAL_CONTAINER_DISK',
        'TERMINAL_CONTAINER_MEMORY',
        'TERMINAL_CONTAINER_PERSISTENT',
        'TERMINAL_CWD',
        'TERMINAL_DAYTONA_IMAGE',
        'TERMINAL_DOCKER_FORWARD_ENV',
        'TERMINAL_DOCKER_IMAGE',
        'TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE',
        'TERMINAL_DOCKER_VOLUMES',
        'TERMINAL_ENV',
        'TERMINAL_LIFETIME_SECONDS',
        'TERMINAL_LOCAL_PERSISTENT',
        'TERMINAL_MODAL_IMAGE',
        'TERMINAL_PERSISTENT_SHELL',
        'TERMINAL_SANDBOX_DIR',
        'TERMINAL_SINGULARITY_IMAGE',
        'TERMINAL_SSH_HOST',
        'TERMINAL_SSH_KEY',
        'TERMINAL_SSH_PERSISTENT',
        'TERMINAL_SSH_PORT',
        'TERMINAL_SSH_USER',
        'TERMINAL_TIMEOUT',
        'TOKENHUB_API_KEY',
        'TOKENHUB_BASE_URL',
        'TOOL_GATEWAY_DOMAIN',
        'TOOL_GATEWAY_SCHEME',
        'TOOL_GATEWAY_USER_TOKEN',
        'TWILIO_ACCOUNT_SID',
        'TWILIO_AUTH_TOKEN',
        'TWILIO_PHONE_NUMBER',
        'VOICE_TOOLS_OPENAI_KEY',
        'WEBHOOK_ENABLED',
        'WEBHOOK_PORT',
        'WEBHOOK_SECRET',
        'WECOM_ALLOWED_USERS',
        'WECOM_BOT_ID',
        'WECOM_CALLBACK_AGENT_ID',
        'WECOM_CALLBACK_ALLOWED_USERS',
        'WECOM_CALLBACK_ALLOW_ALL_USERS',
        'WECOM_CALLBACK_CORP_ID',
        'WECOM_CALLBACK_CORP_SECRET',
        'WECOM_CALLBACK_ENCODING_AES_KEY',
        'WECOM_CALLBACK_HOST',
        'WECOM_CALLBACK_PORT',
        'WECOM_CALLBACK_TOKEN',
        'WECOM_HOME_CHANNEL',
        'WECOM_SECRET',
        'WECOM_WEBSOCKET_URL',
        'WEIXIN_ACCOUNT_ID',
        'WEIXIN_ALLOWED_USERS',
        'WEIXIN_ALLOW_ALL_USERS',
        'WEIXIN_BASE_URL',
        'WEIXIN_CDN_BASE_URL',
        'WEIXIN_DM_POLICY',
        'WEIXIN_GROUP_ALLOWED_USERS',
        'WEIXIN_GROUP_POLICY',
        'WEIXIN_HOME_CHANNEL',
        'WEIXIN_HOME_CHANNEL_NAME',
        'WEIXIN_TOKEN',
        'WHATSAPP_ALLOWED_USERS',
        'WHATSAPP_ALLOW_ALL_USERS',
        'WHATSAPP_CLOUD_ACCESS_TOKEN',
        'WHATSAPP_CLOUD_ALLOWED_USERS',
        'WHATSAPP_CLOUD_ALLOW_ALL_USERS',
        'WHATSAPP_CLOUD_ALLOW_FROM',
        'WHATSAPP_CLOUD_API_VERSION',
        'WHATSAPP_CLOUD_APP_ID',
        'WHATSAPP_CLOUD_APP_SECRET',
        'WHATSAPP_CLOUD_DM_POLICY',
        'WHATSAPP_CLOUD_GROUP_ALLOW_FROM',
        'WHATSAPP_CLOUD_GROUP_POLICY',
        'WHATSAPP_CLOUD_HOME_CHANNEL',
        'WHATSAPP_CLOUD_PHONE_NUMBER_ID',
        'WHATSAPP_CLOUD_VERIFY_TOKEN',
        'WHATSAPP_CLOUD_WABA_ID',
        'WHATSAPP_CLOUD_WEBHOOK_HOST',
        'WHATSAPP_CLOUD_WEBHOOK_PATH',
        'WHATSAPP_CLOUD_WEBHOOK_PORT',
        'WHATSAPP_DEBUG',
        'WHATSAPP_DM_POLICY',
        'WHATSAPP_ENABLED',
        'WHATSAPP_GROUP_POLICY',
        'WHATSAPP_MODE',
        'XAI_API_KEY',
        'XAI_BASE_URL',
        'XIAOMI_API_KEY',
        'XIAOMI_BASE_URL',
        'ZAI_API_KEY',
        'Z_AI_API_KEY',
    ])
    def test_explicit_key_routes_to_env(self, key, _isolated_hermes_home):
        set_config_value(key, "test-value-123")
        env_content = _read_env(_isolated_hermes_home)
        assert f"{key}=test-value-123" in env_content
        # Must NOT appear in config.yaml
        assert key not in _read_config(_isolated_hermes_home)

# ---------------------------------------------------------------------------
# Non-secret keys → config.yaml
# ---------------------------------------------------------------------------

class TestConfigYamlRouting:
    """Regular config keys should go to config.yaml, NOT .env."""

    def test_simple_key(self, _isolated_hermes_home):
        set_config_value("model", "gpt-4o")
        config = _read_config(_isolated_hermes_home)
        assert "gpt-4o" in config
        assert "model" not in _read_env(_isolated_hermes_home)

    def test_nested_key(self, _isolated_hermes_home):
        set_config_value("terminal.backend", "docker")
        config = _read_config(_isolated_hermes_home)
        assert "docker" in config
        assert "terminal" not in _read_env(_isolated_hermes_home)

    def test_terminal_image_goes_to_config(self, _isolated_hermes_home):
        """TERMINAL_DOCKER_IMAGE doesn't match _API_KEY or _TOKEN, so config.yaml."""
        set_config_value("terminal.docker_image", "python:3.12")
        config = _read_config(_isolated_hermes_home)
        assert "python:3.12" in config

    def test_terminal_docker_cwd_mount_flag_goes_to_config_and_env(self, _isolated_hermes_home):
        set_config_value("terminal.docker_mount_cwd_to_workspace", "true")
        config = _read_config(_isolated_hermes_home)
        env_content = _read_env(_isolated_hermes_home)
        assert "docker_mount_cwd_to_workspace: 'true'" in config or "docker_mount_cwd_to_workspace: true" in config
        assert (
            "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE=true" in env_content
            or "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE=True" in env_content
        )


# ---------------------------------------------------------------------------
# Empty / falsy values — regression tests for #4277
# ---------------------------------------------------------------------------

class TestFalsyValues:
    """config set should accept empty strings and falsy values like '0'."""

    def test_empty_string_routes_to_env(self, _isolated_hermes_home):
        """Blanking an API key should write an empty value to .env."""
        set_config_value("OPENROUTER_API_KEY", "")
        env_content = _read_env(_isolated_hermes_home)
        assert "OPENROUTER_API_KEY=" in env_content

    def test_empty_string_routes_to_config(self, _isolated_hermes_home):
        """Blanking a config key should write an empty string to config.yaml."""
        set_config_value("model", "")
        config = _read_config(_isolated_hermes_home)
        assert "model: ''" in config or "model: \"\"" in config

    def test_zero_routes_to_config(self, _isolated_hermes_home):
        """Setting a config key to '0' should write 0 to config.yaml."""
        set_config_value("verbose", "0")
        config = _read_config(_isolated_hermes_home)
        assert "verbose: 0" in config

    def test_config_command_rejects_missing_value(self):
        """config set with no value arg (None) should still exit."""
        args = argparse.Namespace(config_command="set", key="model", value=None)
        with pytest.raises(SystemExit):
            config_command(args)

    def test_config_command_accepts_empty_string(self, _isolated_hermes_home):
        """config set KEY '' should not exit — it should set the value."""
        args = argparse.Namespace(config_command="set", key="model", value="")
        config_command(args)
        config = _read_config(_isolated_hermes_home)
        assert "model" in config


# ---------------------------------------------------------------------------
# List navigation — regression tests for #17876
# ---------------------------------------------------------------------------

class TestListNavigation:
    """hermes config set must preserve YAML list fields when using numeric
    indices.  Before #17876, _set_nested would silently replace the entire
    list with a dict, destroying every sibling entry.
    """

    def _write_config(self, tmp_path, body):
        (tmp_path / "config.yaml").write_text(body)

    def test_indexed_set_preserves_sibling_list_entries(self, _isolated_hermes_home):
        """Setting custom_providers.0.api_key must not destroy entry 1."""
        self._write_config(_isolated_hermes_home, (
            "custom_providers:\n"
            "- name: provider-a\n"
            "  api_key: old-a\n"
            "  base_url: https://a.example.com\n"
            "- name: provider-b\n"
            "  api_key: old-b\n"
            "  base_url: https://b.example.com\n"
        ))

        set_config_value("custom_providers.0.api_key", "new-a")

        import yaml
        reloaded = yaml.safe_load(_read_config(_isolated_hermes_home))
        # The list must still be a list
        assert isinstance(reloaded["custom_providers"], list)
        assert len(reloaded["custom_providers"]) == 2
        # Entry 0 was updated
        assert reloaded["custom_providers"][0]["api_key"] == "new-a"
        assert reloaded["custom_providers"][0]["name"] == "provider-a"
        assert reloaded["custom_providers"][0]["base_url"] == "https://a.example.com"
        # Entry 1 is untouched
        assert reloaded["custom_providers"][1]["name"] == "provider-b"
        assert reloaded["custom_providers"][1]["api_key"] == "old-b"
        assert reloaded["custom_providers"][1]["base_url"] == "https://b.example.com"

    def test_indexed_set_preserves_non_targeted_fields(self, _isolated_hermes_home):
        """Setting one field in a list entry must not drop other fields."""
        self._write_config(_isolated_hermes_home, (
            "custom_providers:\n"
            "- name: provider-a\n"
            "  api_key: old\n"
            "  base_url: https://a.example.com\n"
            "  models:\n"
            "    foo: {}\n"
            "    bar: {}\n"
        ))

        set_config_value("custom_providers.0.api_key", "rotated")

        import yaml
        reloaded = yaml.safe_load(_read_config(_isolated_hermes_home))
        entry = reloaded["custom_providers"][0]
        assert entry["api_key"] == "rotated"
        assert entry["name"] == "provider-a"
        assert entry["base_url"] == "https://a.example.com"
        assert set(entry["models"].keys()) == {"foo", "bar"}

    def test_deeper_nesting_through_list(self, _isolated_hermes_home):
        """Navigation path mixing dict → list → dict → scalar."""
        self._write_config(_isolated_hermes_home, (
            "platforms:\n"
            "  telegram:\n"
            "    allowlist:\n"
            "    - name: alice\n"
            "      role: admin\n"
            "    - name: bob\n"
            "      role: user\n"
        ))

        set_config_value("platforms.telegram.allowlist.1.role", "admin")

        import yaml
        reloaded = yaml.safe_load(_read_config(_isolated_hermes_home))
        allowlist = reloaded["platforms"]["telegram"]["allowlist"]
        assert isinstance(allowlist, list)
        assert allowlist[0] == {"name": "alice", "role": "admin"}
        assert allowlist[1] == {"name": "bob", "role": "admin"}


# ---------------------------------------------------------------------------
# Secret redaction in display output (issue #50245)
# ---------------------------------------------------------------------------

class TestSecretRedactionInDisplay:
    """`config set`/`config show` must not echo credential values in plaintext."""

    def test_redact_config_value_masks_nested_api_key(self):
        from hermes_cli.config import redact_config_value
        secret = "cfut_SUPERSECRETTOKEN1234567890abcdef"
        model = {"default": "@cf/foo", "provider": "custom", "api_key": secret}

        out = redact_config_value(model)

        assert out["api_key"] != secret
        assert secret not in str(out)
        # Non-secret fields pass through unchanged.
        assert out["default"] == "@cf/foo"
        assert out["provider"] == "custom"

    def test_redact_config_value_walks_lists(self):
        from hermes_cli.config import redact_config_value
        secret = "sk-deadbeefdeadbeefdeadbeef"
        cfg = {"custom_providers": [{"name": "p", "api_key": secret}]}

        out = redact_config_value(cfg)

        assert secret not in str(out)
        assert out["custom_providers"][0]["name"] == "p"

    def test_redact_config_value_ignores_benign_keys(self):
        from hermes_cli.config import redact_config_value
        cfg = {"token_count": 1234, "secret_santa": "alice", "max_turns": 90}

        out = redact_config_value(cfg)

        # Exact-match only — substrings like token_count must NOT be masked.
        assert out == cfg

    def test_set_echo_masks_secret_value(self, _isolated_hermes_home, capsys):
        secret = "cfut_ANOTHERSECRET0987654321zyxwvu"
        set_config_value("model.api_key", secret)

        captured = capsys.readouterr()
        assert secret not in captured.out
        assert "Set model.api_key" in captured.out

    def test_set_echo_keeps_nonsecret_value(self, _isolated_hermes_home, capsys):
        set_config_value("model.reasoning_effort", "high")

        captured = capsys.readouterr()
        assert "Set model.reasoning_effort = high" in captured.out
