"""Env-var names the hermetic ``_hermetic_environment`` fixture blanks before every test.

Pure data, split out of ``tests/conftest.py`` to keep it under the file-size
gate; the fixture itself stays in conftest next to the sandbox it maintains.
"""


# ── Credential env-var filter ──────────────────────────────────────────────
#
# Any env var in the current process matching ONE of these patterns is
# unset for every test. Developers' local keys cannot leak into assertions
# about "auto-detect provider when key present".

_CREDENTIAL_SUFFIXES = (
    "_API_KEY",
    "_TOKEN",
    "_SECRET",
    "_PASSWORD",
    "_CREDENTIALS",
    "_ACCESS_KEY",
    "_SECRET_ACCESS_KEY",
    "_PRIVATE_KEY",
    "_OAUTH_TOKEN",
    "_WEBHOOK_SECRET",
    "_ENCRYPT_KEY",
    "_APP_SECRET",
    "_CLIENT_SECRET",
    "_CORP_SECRET",
    "_AES_KEY",
)

# Explicit names (for ones that don't fit the suffix pattern)
_CREDENTIAL_NAMES = frozenset({
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "ANTHROPIC_TOKEN",
    "FAL_KEY",
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "NOUS_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "XAI_API_KEY",
    "MISTRAL_API_KEY",
    "DEEPSEEK_API_KEY",
    "KIMI_API_KEY",
    "MOONSHOT_API_KEY",
    "GLM_API_KEY",
    "ZAI_API_KEY",
    "MINIMAX_API_KEY",
    "OLLAMA_API_KEY",
    "OPENVIKING_API_KEY",
    "COPILOT_API_KEY",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "BROWSERBASE_API_KEY",
    "FIRECRAWL_API_KEY",
    "PARALLEL_API_KEY",
    "EXA_API_KEY",
    "TAVILY_API_KEY",
    "PERPLEXITY_API_KEY",
    "WANDB_API_KEY",
    "ELEVENLABS_API_KEY",
    "HONCHO_API_KEY",
    "MEM0_API_KEY",
    "SUPERMEMORY_API_KEY",
    "RETAINDB_API_KEY",
    "HINDSIGHT_API_KEY",
    "HINDSIGHT_LLM_API_KEY",
    "DAYTONA_API_KEY",
    "TWILIO_AUTH_TOKEN",
    "TELEGRAM_BOT_TOKEN",
    "DISCORD_BOT_TOKEN",
    "SLACK_BOT_TOKEN",
    "SLACK_APP_TOKEN",
    "MATTERMOST_TOKEN",
    "MATRIX_ACCESS_TOKEN",
    "MATRIX_PASSWORD",
    "MATRIX_RECOVERY_KEY",
    "HASS_TOKEN",
    "EMAIL_PASSWORD",
    "BLUEBUBBLES_PASSWORD",
    "FEISHU_APP_SECRET",
    "FEISHU_ENCRYPT_KEY",
    "FEISHU_VERIFICATION_TOKEN",
    "DINGTALK_CLIENT_SECRET",
    "QQ_CLIENT_SECRET",
    "QQ_STT_API_KEY",
    "WECOM_SECRET",
    "WECOM_CALLBACK_CORP_SECRET",
    "WECOM_CALLBACK_TOKEN",
    "WECOM_CALLBACK_ENCODING_AES_KEY",
    "WEIXIN_TOKEN",
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
    "TERMINAL_SSH_KEY",
    "SUDO_PASSWORD",
    "GATEWAY_PROXY_KEY",
    "API_SERVER_KEY",
    "TOOL_GATEWAY_USER_TOKEN",
    "TELEGRAM_WEBHOOK_SECRET",
    "WEBHOOK_SECRET",
    "AI_GATEWAY_API_KEY",
    "VOICE_TOOLS_OPENAI_KEY",
    "BROWSER_USE_API_KEY",
    "CUSTOM_API_KEY",
    "GATEWAY_PROXY_URL",
    "GEMINI_BASE_URL",
    "OPENAI_BASE_URL",
    "OPENROUTER_BASE_URL",
    "OLLAMA_BASE_URL",
    "GROQ_BASE_URL",
    "XAI_BASE_URL",
    "AI_GATEWAY_BASE_URL",
    "ANTHROPIC_BASE_URL",
})


def _looks_like_credential(name: str) -> bool:
    """True if env var name matches a credential-shaped pattern."""
    if name in _CREDENTIAL_NAMES:
        return True
    return any(name.endswith(suf) for suf in _CREDENTIAL_SUFFIXES)


# HERMES_* vars that change test behavior by being set. Unset all of these
# unconditionally — individual tests that need them set do so explicitly.
_HERMES_BEHAVIORAL_VARS = frozenset({
    # Voice/TTS runtime flags. ``tui_gateway/server.py`` reads these straight
    # off ``os.environ`` at call time (``_voice_mode_enabled`` /
    # ``_voice_tts_enabled``) and, on every completed turn, hands the turn's
    # final response text to ``hermes_cli.voice.speak_text`` — real synthesis,
    # real playback, out of the developer's speakers. Blank them per-test so a
    # leak (from the shell, or from an earlier test that drove the
    # ``voice.toggle`` RPC, which writes ``os.environ`` directly) cannot carry
    # into the next test. See ``_audio_playback_guard`` for the second layer.
    "HERMES_VOICE",
    "HERMES_VOICE_TTS",
    "HERMES_YOLO_MODE",
    # Injected into subprocess envs by the terminal tool (_make_run_env), so
    # any test run launched FROM a Hermes agent session inherits them and
    # hermes_constants home-resolution helpers prefer them over monkeypatched
    # HOME (test_subprocess_home_isolation red locally, green on CI).
    "HERMES_REAL_HOME",
    "TERMINAL_HOME_MODE",
    "HERMES_INTERACTIVE",
    "HERMES_QUIET",
    "HERMES_TOOL_PROGRESS",
    "HERMES_TOOL_PROGRESS_MODE",
    "HERMES_MAX_ITERATIONS",
    "HERMES_SESSION_PLATFORM",
    "HERMES_SESSION_CHAT_ID",
    "HERMES_SESSION_CHAT_NAME",
    "HERMES_SESSION_CHAT_TYPE",
    "HERMES_SESSION_THREAD_ID",
    "HERMES_SESSION_SOURCE",
    "HERMES_SESSION_KEY",
    "HERMES_GATEWAY_SESSION",
    "HERMES_CRON_SESSION",
    "_HERMES_GATEWAY",
    "HERMES_PLATFORM",
    "HERMES_MODEL",
    "HERMES_INFERENCE_MODEL",
    "HERMES_INFERENCE_PROVIDER",
    "HERMES_TUI_PROVIDER",
    "HERMES_MANAGED",
    "HERMES_MANAGED_DIR",
    # A Nix-wrapped `hermes` on the developer's host exports the store's read-only plugins
    # tree; tests must discover the checkout's plugins/ (get_bundled_plugins_dir), not a
    # different release's.
    "HERMES_BUNDLED_PLUGINS",
    "HERMES_DEV",
    "HERMES_CONTAINER",
    "HERMES_EPHEMERAL_SYSTEM_PROMPT",
    "HERMES_TIMEZONE",
    "HERMES_REDACT_SECRETS",
    "HERMES_BACKGROUND_NOTIFICATIONS",
    "HERMES_EXEC_ASK",
    "HERMES_HOME_MODE",
    "HERMES_AGENT_USE_LEGACY_SESSION_KEYS",
    # Kanban path/board pins must never leak from a developer shell or
    # dispatched worker into tests; otherwise tests can write fake tasks to
    # the real ~/.hermes/kanban.db instead of the per-test HERMES_HOME.
    "HERMES_KANBAN_DB",
    "HERMES_KANBAN_BOARD",
    "HERMES_KANBAN_HOME",
    "HERMES_KANBAN_WORKSPACES_ROOT",
    "HERMES_KANBAN_LOGS_ROOT",
    "HERMES_KANBAN_TASK",
    "HERMES_KANBAN_WORKSPACE",
    "HERMES_KANBAN_RUN_ID",
    "HERMES_KANBAN_CLAIM_LOCK",
    "HERMES_KANBAN_DISPATCH_IN_GATEWAY",
    # Pytest is routinely launched from a delegated worker.  The worker
    # lineage marker must not make parent-state tests run as delegated
    # children; tests that exercise child behavior set it explicitly.
    "HERMES_DELEGATED_CHILD_CONTEXT",
    "HERMES_TENANT",
    # Honcho host selection changes which nested config block wins. A local
    # shell override leaked "myhost" into the full suite and flipped 20
    # otherwise-unrelated config tests away from the default "hermes" host.
    "HERMES_HONCHO_HOST",
    # Dashboard OAuth auth gate (PR #30156). When set, the bundled
    # dashboard-auth `nous` plugin auto-registers itself on plugin discovery,
    # which is triggered by any `/api/status` call. That leaks a provider
    # into the dashboard_auth registry across tests in the same worker and
    # makes assertions like `auth_providers == []` flaky. CI never sets
    # these, so production tests must not see them either.
    "HERMES_DASHBOARD_OAUTH_CLIENT_ID",
    "HERMES_DASHBOARD_PORTAL_URL",
    "TERMINAL_CWD",
    "TERMINAL_ENV",
    "TERMINAL_VERCEL_RUNTIME",
    "TERMINAL_CONTAINER_CPU",
    "TERMINAL_CONTAINER_DISK",
    "TERMINAL_CONTAINER_MEMORY",
    "TERMINAL_CONTAINER_PERSISTENT",
    "TERMINAL_DOCKER_PERSIST_ACROSS_PROCESSES",
    "TERMINAL_DOCKER_ORPHAN_REAPER",
    "TERMINAL_DOCKER_RUN_AS_HOST_USER",
    "BROWSER_CDP_URL",
    "CAMOFOX_URL",
    # Platform allowlists — not credentials, but if set from any source
    # (user shell, earlier leaky test, CI env), they change gateway auth
    # behavior and flake button-authorization tests.
    "TELEGRAM_ALLOWED_USERS",
    "TELEGRAM_GROUP_ALLOWED_USERS",
    "TELEGRAM_GROUP_ALLOWED_CHATS",
    "QQ_ALLOWED_USERS",
    "QQ_GROUP_ALLOWED_USERS",
    "DISCORD_ALLOWED_USERS",
    "WHATSAPP_ALLOWED_USERS",
    "SLACK_ALLOWED_USERS",
    "SIGNAL_ALLOWED_USERS",
    "SIGNAL_GROUP_ALLOWED_USERS",
    "EMAIL_ALLOWED_USERS",
    "SMS_ALLOWED_USERS",
    "MATTERMOST_ALLOWED_USERS",
    "MATRIX_ALLOWED_USERS",
    "DINGTALK_ALLOWED_USERS",
    "FEISHU_ALLOWED_USERS",
    "WECOM_ALLOWED_USERS",
    "PHOTON_ALLOWED_USERS",
    "GATEWAY_ALLOWED_USERS",
    "GATEWAY_ALLOW_ALL_USERS",
    "TELEGRAM_ALLOW_ALL_USERS",
    "DISCORD_ALLOW_ALL_USERS",
    "WHATSAPP_ALLOW_ALL_USERS",
    "SLACK_ALLOW_ALL_USERS",
    "SIGNAL_ALLOW_ALL_USERS",
    "EMAIL_ALLOW_ALL_USERS",
    "SMS_ALLOW_ALL_USERS",
    "PHOTON_ALLOW_ALL_USERS",
    # Gateway home channels are set by /sethome in real profiles. Tests that
    # exercise dashboard notification toggles must opt in explicitly or they
    # can accidentally subscribe against a developer's real home channel.
    "TELEGRAM_HOME_CHANNEL",
    "TELEGRAM_HOME_CHANNEL_THREAD_ID",
    "TELEGRAM_HOME_CHANNEL_NAME",
    "TELEGRAM_CRON_THREAD_ID",
    "DISCORD_HOME_CHANNEL",
    "DISCORD_HOME_CHANNEL_THREAD_ID",
    "DISCORD_HOME_CHANNEL_NAME",
    "SLACK_HOME_CHANNEL",
    "SLACK_HOME_CHANNEL_THREAD_ID",
    "SLACK_HOME_CHANNEL_NAME",
    "WHATSAPP_HOME_CHANNEL",
    "WHATSAPP_HOME_CHANNEL_THREAD_ID",
    "WHATSAPP_HOME_CHANNEL_NAME",
    "SIGNAL_HOME_CHANNEL",
    "SIGNAL_HOME_CHANNEL_THREAD_ID",
    "SIGNAL_HOME_CHANNEL_NAME",
    "EMAIL_HOME_CHANNEL",
    "EMAIL_HOME_CHANNEL_THREAD_ID",
    "EMAIL_HOME_CHANNEL_NAME",
    "SMS_HOME_CHANNEL",
    "SMS_HOME_CHANNEL_THREAD_ID",
    "SMS_HOME_CHANNEL_NAME",
    "MATTERMOST_HOME_CHANNEL",
    "MATTERMOST_HOME_CHANNEL_THREAD_ID",
    "MATTERMOST_HOME_CHANNEL_NAME",
    "MATRIX_HOME_CHANNEL",
    "MATRIX_HOME_CHANNEL_THREAD_ID",
    "MATRIX_HOME_CHANNEL_NAME",
    "DINGTALK_HOME_CHANNEL",
    "DINGTALK_HOME_CHANNEL_THREAD_ID",
    "DINGTALK_HOME_CHANNEL_NAME",
    "FEISHU_HOME_CHANNEL",
    "FEISHU_HOME_CHANNEL_THREAD_ID",
    "FEISHU_HOME_CHANNEL_NAME",
    "WECOM_HOME_CHANNEL",
    "WECOM_HOME_CHANNEL_THREAD_ID",
    "WECOM_HOME_CHANNEL_NAME",
    "PHOTON_HOME_CHANNEL",
    "PHOTON_HOME_CHANNEL_THREAD_ID",
    "PHOTON_HOME_CHANNEL_NAME",
    # API server bind/auth settings are common in local gateway profiles and
    # change adapter defaults plus load_gateway_config() enablement. Tests that
    # need them set opt in explicitly with monkeypatch.
    "API_SERVER_ENABLED",
    "API_SERVER_HOST",
    "API_SERVER_PORT",
    "API_SERVER_KEY",
    "API_SERVER_CORS_ORIGINS",
    "API_SERVER_MODEL_NAME",
    # Platform gating — set by load_gateway_config() as a side effect when
    # a config.yaml is present, so individual test bodies that call the
    # loader leak these values into later tests in the same process.
    # Force-clear on every test setup so the leak can't happen.
    "SLACK_REQUIRE_MENTION",
    "SLACK_STRICT_MENTION",
    "SLACK_THREAD_REQUIRE_MENTION",
    "SLACK_IGNORE_OTHER_USER_MENTIONS",
    "SLACK_REQUIRE_MENTION_CHANNELS",
    "SLACK_FREE_RESPONSE_CHANNELS",
    "SLACK_ALLOWED_CHANNELS",
    "SLACK_IGNORED_CHANNELS",
    "SLACK_DISABLE_DMS",
    "SLACK_ALLOW_BOTS",
    "SLACK_REACTIONS",
    "DISCORD_REQUIRE_MENTION",
    "DISCORD_FREE_RESPONSE_CHANNELS",
    "TELEGRAM_REQUIRE_MENTION",
    "WHATSAPP_REQUIRE_MENTION",
    "DINGTALK_REQUIRE_MENTION",
    "MATRIX_REQUIRE_MENTION",
})
