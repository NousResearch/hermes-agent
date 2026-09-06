"""Messaging-platform catalog and onboarding helpers: platform overrides/env discovery, WhatsApp
and Telegram onboarding state.
"""

import logging
import os
import subprocess
import threading
from dataclasses import dataclass
from fastapi import HTTPException
from pathlib import Path
from typing import Any, Optional
from hermes_cli import __version__
from hermes_cli.config import OPTIONAL_ENV_VARS, write_platform_config_field
from hermes_cli.setup_hidden_env import is_setup_hidden_env as _is_setup_hidden_env

# Same logger the code used before extraction (record parity).
_log = logging.getLogger("hermes_cli.web_server")


# Entries omit fields they don't need to override; the catalog builder fills
# in env_vars from OPTIONAL_ENV_VARS via prefix matching when not specified,
# and pulls required_env from a plugin's PlatformEntry when available.
_PLATFORM_OVERRIDES: dict[str, dict[str, Any]] = {
    "telegram": {
        "name": "Telegram", "description": "Run Hermes from Telegram DMs, groups, and topics.",
        "docs_url": "https://core.telegram.org/bots/features#botfather",
        "env_vars": ("TELEGRAM_BOT_TOKEN", "TELEGRAM_ALLOWED_USERS", "TELEGRAM_PROXY"),
        "required_env": ("TELEGRAM_BOT_TOKEN",),
    },
    "discord": {
        "name": "Discord", "description": "Connect Hermes to Discord DMs, channels, and threads.",
        "docs_url": "https://discord.com/developers/applications",
        "env_vars": ("DISCORD_BOT_TOKEN", "DISCORD_ALLOWED_USERS"),
        "required_env": ("DISCORD_BOT_TOKEN",),
    },
    "slack": {
        "name": "Slack",
        "description": "Use Hermes from Slack via Socket Mode. Add allowed Slack member IDs so connected bots can respond.",
        "docs_url": "https://api.slack.com/apps",
        "env_vars": ("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "SLACK_ALLOWED_USERS"),
        "required_env": ("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN"),
    },
    "mattermost": {
        "name": "Mattermost",
        "description": "Connect Hermes to Mattermost channels and direct messages.",
        "docs_url": "https://mattermost.com/deploy/",
        "env_vars": ("MATTERMOST_URL", "MATTERMOST_TOKEN", "MATTERMOST_ALLOWED_USERS"),
        "required_env": ("MATTERMOST_URL", "MATTERMOST_TOKEN"),
    },
    "matrix": {
        "name": "Matrix", "description": "Use Hermes in Matrix rooms and direct messages.",
        "docs_url": "https://matrix.org/ecosystem/servers/",
        "env_vars": (
            "MATRIX_HOMESERVER", "MATRIX_ACCESS_TOKEN", "MATRIX_USER_ID", "MATRIX_ALLOWED_USERS",
        ),
        "required_env": ("MATRIX_HOMESERVER", "MATRIX_ACCESS_TOKEN", "MATRIX_USER_ID"),
    },
    "signal": {
        "name": "Signal", "description": "Connect through a signal-cli REST bridge.",
        "docs_url": "https://github.com/bbernhard/signal-cli-rest-api",
        "env_vars": ("SIGNAL_HTTP_URL", "SIGNAL_ACCOUNT", "SIGNAL_ALLOWED_USERS"),
        "required_env": ("SIGNAL_HTTP_URL", "SIGNAL_ACCOUNT"),
    },
    "whatsapp": {
        "name": "WhatsApp",
        "description": "Use Hermes through the bundled WhatsApp bridge with QR-based auth.",
        "docs_url": "https://github.com/tulir/whatsmeow",
        "env_vars": (
            "WHATSAPP_ENABLED", "WHATSAPP_MODE", "WHATSAPP_DM_POLICY", "WHATSAPP_ALLOWED_USERS",
        ),
        "required_env": (),
    },
    "homeassistant": {
        "name": "Home Assistant",
        "description": "Control your smart home from Hermes via Home Assistant.",
        "docs_url": "https://www.home-assistant.io/docs/authentication/",
        "env_vars": ("HASS_URL", "HASS_TOKEN"), "required_env": ("HASS_URL", "HASS_TOKEN"),
    },
    "email": {
        "name": "Email", "description": "Talk to Hermes through an IMAP/SMTP mailbox.",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/",
        "env_vars": ("EMAIL_ADDRESS", "EMAIL_PASSWORD", "EMAIL_IMAP_HOST", "EMAIL_SMTP_HOST"),
        "required_env": ("EMAIL_ADDRESS", "EMAIL_PASSWORD", "EMAIL_IMAP_HOST", "EMAIL_SMTP_HOST"),
    },
    "sms": {
        "name": "SMS (Twilio)", "description": "Send and receive text messages via Twilio.",
        "docs_url": "https://www.twilio.com/console",
        "env_vars": ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN"),
        "required_env": ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN"),
    },
    "dingtalk": {
        "name": "DingTalk", "description": "Connect Hermes to DingTalk groups (钉钉).",
        "docs_url": "https://open.dingtalk.com/document/orgapp/the-robot-development-process",
        "env_vars": ("DINGTALK_CLIENT_ID", "DINGTALK_CLIENT_SECRET"),
        "required_env": ("DINGTALK_CLIENT_ID", "DINGTALK_CLIENT_SECRET"),
    },
    "feishu": {
        "name": "Feishu / Lark", "description": "Use Hermes inside Feishu / Lark.",
        "docs_url": "https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/im-v1/intro",
        "env_vars": (
            "FEISHU_APP_ID", "FEISHU_APP_SECRET", "FEISHU_ENCRYPT_KEY", "FEISHU_VERIFICATION_TOKEN",
        ),
        "required_env": ("FEISHU_APP_ID", "FEISHU_APP_SECRET"),
    },
    "google_chat": {
        "name": "Google Chat", "description": "Connect Hermes to Google Chat via Cloud Pub/Sub.",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/google_chat",
    },
    "wecom": {
        "name": "WeCom (group bot)", "description": "Send-only WeCom group bot via webhook.",
        "docs_url": "https://developer.work.weixin.qq.com/document/path/91770",
        "env_vars": ("WECOM_BOT_ID", "WECOM_SECRET"), "required_env": ("WECOM_BOT_ID",),
    },
    "wecom_callback": {
        "name": "WeCom (app)", "description": "Two-way WeCom integration via callback app.",
        "docs_url": "https://developer.work.weixin.qq.com/document/path/90930",
        "env_vars": (
            "WECOM_CALLBACK_CORP_ID", "WECOM_CALLBACK_CORP_SECRET", "WECOM_CALLBACK_AGENT_ID",
            "WECOM_CALLBACK_TOKEN", "WECOM_CALLBACK_ENCODING_AES_KEY",
        ),
        "required_env": (
            "WECOM_CALLBACK_CORP_ID", "WECOM_CALLBACK_CORP_SECRET", "WECOM_CALLBACK_AGENT_ID",
        ),
    },
    "weixin": {
        "name": "Weixin / WeChat (Personal)",
        "description": "Connect a personal WeChat account through Tencent's iLink Bot API.",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/weixin/",
        "env_vars": ("WEIXIN_ACCOUNT_ID", "WEIXIN_TOKEN", "WEIXIN_BASE_URL"),
        "required_env": ("WEIXIN_ACCOUNT_ID", "WEIXIN_TOKEN"),
    },
    "bluebubbles": {
        "name": "BlueBubbles (iMessage)",
        "description": "Use Hermes through iMessage via a BlueBubbles server.",
        "docs_url": "https://bluebubbles.app/",
        "env_vars": (
            "BLUEBUBBLES_SERVER_URL", "BLUEBUBBLES_PASSWORD", "BLUEBUBBLES_ALLOWED_USERS",
        ),
        "required_env": ("BLUEBUBBLES_SERVER_URL", "BLUEBUBBLES_PASSWORD"),
    },
    "qqbot": {
        "name": "QQ Bot", "description": "Connect Hermes to a QQ Bot from the QQ Open Platform.",
        "docs_url": "https://q.qq.com",
        "env_vars": ("QQ_APP_ID", "QQ_CLIENT_SECRET", "QQ_ALLOWED_USERS"),
        "required_env": ("QQ_APP_ID", "QQ_CLIENT_SECRET"),
    },
    # Teams ships as a platform plugin, so its name/env vars come from the
    # plugin registry. Only the docs link needs an override here so the
    # Channels page can point at the Microsoft Teams setup guide.
    "teams": {
        "description": "Connect Hermes to Microsoft Teams chats via the Bot Framework.",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/teams",
    },
    # Bundled platform plugins: name comes from the plugin registry label;
    # give each a human description (the registry's install_hint is a
    # dependency note, not a description) and a docs link.
    "irc": {
        "description": "Relay messages between an IRC channel (or DMs) and Hermes.",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/irc",
    },
    "line": {
        "description": "Use Hermes from LINE via the LINE Messaging API webhook.",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/line",
    },
    "ntfy": {
        "description": "Chat with Hermes over ntfy push topics (ntfy.sh or self-hosted).",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/ntfy",
    },
    "photon": {
        "description": "Use Hermes through iMessage via Photon's managed Spectrum platform.",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/photon",
    },
    "raft": {
        "description": "Join a Raft workspace as an external agent.",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/raft",
    },
    "simplex": {
        "description": "Talk to Hermes over SimpleX Chat via a local simplex-chat daemon.",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/simplex",
    },
    "yuanbao": {
        "name": "Yuanbao (元宝)", "description": "Connect Hermes to Tencent Yuanbao.", "docs_url": "",
        "required_env": (),
    },
    "api_server": {
        "name": "API server",
        "description": "Expose Hermes as an OpenAI-compatible HTTP API for tools like Open WebUI.",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/",
        "env_vars": (
            "API_SERVER_ENABLED", "API_SERVER_KEY", "API_SERVER_PORT", "API_SERVER_HOST",
            "API_SERVER_MODEL_NAME",
        ),
        "required_env": (),
    },
    "webhook": {
        "name": "Webhooks",
        "description": "Receive events from GitHub, GitLab, and other webhook sources.",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/webhooks/",
        "env_vars": ("WEBHOOK_ENABLED", "WEBHOOK_PORT", "WEBHOOK_SECRET"), "required_env": (),
    },
    "msgraph_webhook": {
        "name": "Microsoft Graph Webhook",
        "description": "Receive Microsoft Graph change notifications (Teams meetings, Outlook, …).",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/msgraph-webhook",
        "required_env": (),
    },
    "whatsapp_cloud": {
        "name": "WhatsApp Cloud API",
        "description": "Use Hermes via Meta's hosted WhatsApp Cloud API (no local bridge).",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/whatsapp-cloud",
    },
    "relay": {
        "name": "Relay (experimental)",
        "description": "Generic relay adapter fronted by the Hermes Relay connector.",
        "docs_url": "", "required_env": (),
    },
}

# Display order: well-known platforms surface first; unknown plugins fall to
# the end alphabetically.
_PLATFORM_ORDER: tuple[str, ...] = (
    "telegram", "discord", "slack", "mattermost", "matrix", "whatsapp", "signal", "bluebubbles",
    "homeassistant", "email", "sms", "dingtalk", "feishu", "google_chat", "wecom", "wecom_callback",
    "weixin", "qqbot", "yuanbao", "api_server", "webhook",
)


def _messaging_platform_catalog() -> tuple[dict[str, Any], ...]:
    """Build the messaging catalog from the gateway's Platform enum + plugin registry.

    Built-ins come from ``gateway.config.Platform`` (LOCAL excluded); plugin platforms from
    ``platform_registry.plugin_entries()`` so new adapters appear without a code change here.
    UI metadata lives in :data:`_PLATFORM_OVERRIDES`; the rest is derived from id/required_env.
    """
    from gateway.config import Platform

    # Resolve plugin entries FIRST: plugin platforms leak into ``Platform.__members__`` as
    # pseudo-members once anything calls ``Platform("<plugin id>")``, and iterating the enum
    # first would claim them with no plugin metadata (nameless "Irc"/"Ntfy" cards).
    plugin_map: dict[str, Any] = {}
    try:
        # Plugin discovery normally runs as a side effect of importing model_tools, which this
        # server process doesn't do — trigger it explicitly (idempotent).
        from hermes_cli.plugins import discover_plugins
        discover_plugins()
        from gateway.platform_registry import platform_registry
        for plugin_entry in platform_registry.plugin_entries():
            plugin_map[plugin_entry.name] = plugin_entry
    except Exception:
        _log.debug("plugin platform registry unavailable", exc_info=True)

    seen: set[str] = set()
    entries: list[dict[str, Any]] = []
    builtin = [m.value for m in Platform.__members__.values() if m.value != "local"]
    for pid in builtin + list(plugin_map):
        if pid in seen:
            continue
        seen.add(pid)
        entries.append(_build_catalog_entry(pid, plugin_map.get(pid)))

    order = {pid: idx for idx, pid in enumerate(_PLATFORM_ORDER)}
    entries.sort(key=lambda e: (order.get(e["id"], len(_PLATFORM_ORDER)), e["name"].lower()))
    return tuple(entries)


def _channel_managed_env_keys() -> frozenset[str]:
    """Env-var keys owned by a Channels page platform card; the Keys/Env page hides them so the
    same fields aren't duplicated. Best-effort: if the catalog can't be built, nothing is hidden."""
    try:
        return frozenset(k for entry in _messaging_platform_catalog() for k in entry.get("env_vars", ()))
    except Exception:
        _log.debug("could not build channel-managed env key set", exc_info=True)
        return frozenset()


# Cross-cutting gateway / relay knobs stay on the Keys → Settings tab even though
# they use the ``messaging`` category in OPTIONAL_ENV_VARS. Platform-scoped vars
# (``DISCORD_*``, ``MATRIX_*``, …) are owned by the Messaging UI instead.
_MESSAGING_KEYS_PAGE_KEYS = frozenset({
    "GATEWAY_ALLOW_ALL_USERS", "GATEWAY_PROXY_KEY", "GATEWAY_PROXY_URL"})


_PLATFORM_ENV_PREFIX_ALIASES: dict[str, tuple[str, ...]] = {
    "email": ("EMAIL_",),
    "homeassistant": ("HASS_",),
    "qqbot": ("QQ_", "QQBOT_"),
    "sms": ("TWILIO_",),
    "wecom": ("WECOM_BOT_", "WECOM_SECRET"),
    "wecom_callback": ("WECOM_CALLBACK_",)}


def _platform_env_prefixes(platform_id: str) -> tuple[str, ...]:
    """Env-var prefixes owned by a messaging platform card."""
    return _PLATFORM_ENV_PREFIX_ALIASES.get(platform_id, (platform_id.upper().replace("-", "_") + "_",))


def _discover_platform_env_vars(platform_id: str) -> tuple[str, ...]:
    """All messaging-category env vars for a platform (override + plugin + prefix)."""
    prefixes = _platform_env_prefixes(platform_id)
    return tuple(sorted({
        name for name, info in OPTIONAL_ENV_VARS.items()
        if info.get("category") == "messaging"
        and name not in _MESSAGING_KEYS_PAGE_KEYS
        and not _is_setup_hidden_env(name)
        and any(name.startswith(prefix) for prefix in prefixes)}))


def _merge_platform_env_vars(platform_id: str, override: dict[str, Any], plugin_entry: Any | None) -> tuple[str, ...]:
    """Canonical env-var list for a platform card. Required credentials always survive: hiding a
    required field would make the platform unconfigurable."""
    discovered = _discover_platform_env_vars(platform_id)
    if "env_vars" in override:
        explicit = tuple(key for key in override["env_vars"] if not _is_setup_hidden_env(key))
        return tuple(dict.fromkeys((*explicit, *discovered)))
    if plugin_entry is not None and plugin_entry.required_env:
        return tuple(dict.fromkeys((*tuple(plugin_entry.required_env), *discovered)))
    return discovered


def _build_catalog_entry(platform_id: str, plugin_entry: Any | None = None) -> dict[str, Any]:
    override = _PLATFORM_OVERRIDES.get(platform_id, {})
    if "required_env" in override:
        required_env = tuple(override["required_env"])
    else:
        required_env = tuple(plugin_entry.required_env or ()) if plugin_entry is not None else ()
    plugin_label = plugin_entry.label if plugin_entry is not None else None
    plugin_hint = (plugin_entry.install_hint or "") if plugin_entry is not None else None
    return {
        "id": platform_id,
        "name": override.get("name") or plugin_label or platform_id.replace("_", " ").title(),
        "description": override.get("description") or plugin_hint or "",
        "docs_url": override.get("docs_url", ""),
        "env_vars": _merge_platform_env_vars(platform_id, override, plugin_entry),
        "required_env": required_env,
    }


def _write_platform_enabled(platform_id: str, enabled: bool) -> None:
    write_platform_config_field(platform_id, "enabled", enabled)


@dataclass
class _WhatsAppOnboardingSession:
    proc: subprocess.Popen | None
    mode: str
    allowed_users: str
    session_path: str
    expires_at: str
    expires_at_ts: float
    profile: str | None = None
    status: str = "starting"
    qr_payload: str | None = None
    account_id: str | None = None
    account_name: str | None = None
    account_phone: str | None = None
    error: str | None = None


_whatsapp_onboarding_sessions: dict[str, _WhatsAppOnboardingSession] = {}


def _whatsapp_session_path() -> Path:
    from hermes_constants import get_hermes_dir
    return get_hermes_dir("platforms/whatsapp/session", "whatsapp/session")


_WHATSAPP_PAYLOAD_FIELDS = (
    "status", "qr_payload", "expires_at", "mode", "allowed_users", "account_id", "account_name",
    "account_phone", "error",
)


def _whatsapp_onboarding_payload(pairing_id: str, record: _WhatsAppOnboardingSession) -> dict[str, Any]:
    return {"pairing_id": pairing_id, **{f: getattr(record, f) for f in _WHATSAPP_PAYLOAD_FIELDS}}


def _restart_gateway_after_whatsapp_onboarding(profile: Optional[str] = None) -> dict[str, Any]:
    from hermes_cli.web_server_gateway import _restart_gateway_after
    return _restart_gateway_after(profile, what="WhatsApp onboarding", label="WhatsApp onboarding")


_TELEGRAM_ONBOARDING_DEFAULT_URL = "https://setup.hermes-agent.nousresearch.com"
_TELEGRAM_ONBOARDING_USER_AGENT = f"HermesDashboard/{__version__}"


@dataclass
class _TelegramOnboardingPairing:
    poll_token: str
    expires_at: str
    expires_at_ts: float
    bot_token: str | None = None
    bot_username: str | None = None
    owner_user_id: str | None = None


_telegram_onboarding_pairings: dict[str, _TelegramOnboardingPairing] = {}
_telegram_onboarding_lock = threading.RLock()


def _telegram_onboarding_base_url() -> str:
    return os.getenv("TELEGRAM_ONBOARDING_URL", _TELEGRAM_ONBOARDING_DEFAULT_URL).strip().rstrip("/")


def _telegram_onboarding_error_message(error: str, fallback: str) -> str:
    return {
        "not_found": "Telegram pairing was not found. Start a new setup.",
        "expired": "Telegram setup expired. Start a new setup.",
        "claimed": "Telegram setup was already claimed. Start a new setup.",
        "unauthorized": "Telegram setup service rejected this request.",
        "telegram_manager_bot_token_not_configured": "Telegram setup service is not configured.",
        "telegram_token_fetch_failed": "Telegram could not finish bot setup. Try again.",
    }.get(error, fallback)


_TELEGRAM_UNAVAILABLE = "Telegram setup service is unavailable. Try again shortly."
_TELEGRAM_INVALID = "Telegram setup service returned an invalid response."


def _telegram_onboarding_request_sync(
    method: str, path: str, *, body: dict[str, Any] | None = None, bearer_token: str | None = None
) -> dict[str, Any]:
    import httpx
    headers = {"Accept": "application/json", "User-Agent": _TELEGRAM_ONBOARDING_USER_AGENT}
    request_kwargs: dict[str, Any] = {}
    if body is not None:
        headers["Content-Type"] = "application/json"
        request_kwargs["json"] = body
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    try:
        with httpx.Client(timeout=httpx.Timeout(10.0)) as client:
            response = client.request(method, f"{_telegram_onboarding_base_url()}{path}", headers=headers, **request_kwargs)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        try:
            parsed = exc.response.json()
        except Exception:
            parsed = {}
        error = str(parsed.get("error") or parsed.get("status") or "")
        detail = _telegram_onboarding_error_message(error, "Telegram setup service returned an error.")
        if error in {"expired", "claimed"}:
            status_code = 410
        else:
            status_code = 404 if exc.response.status_code == 404 else 502
        raise HTTPException(status_code=status_code, detail=detail) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_TELEGRAM_UNAVAILABLE) from exc

    try:
        parsed = response.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_TELEGRAM_INVALID) from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=502, detail=_TELEGRAM_INVALID)
    return parsed
