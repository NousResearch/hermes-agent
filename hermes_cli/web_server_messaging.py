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
from hermes_cli.config import OPTIONAL_ENV_VARS, write_platform_config_field
from hermes_cli.setup_hidden_env import is_setup_hidden_env as _is_setup_hidden_env
from hermes_cli.version_info import get_version_info

# Same logger the code used before extraction (record parity).
_log = logging.getLogger("hermes_cli.web_server")


# Card metadata for core-only platforms (no plugin, so no PlatformEntry to declare it). Plugin
# platforms declare theirs on their PlatformEntry (``dashboard_*`` / ``docs_url``, see
# :func:`_plugin_card_fields`). Entries omit fields they don't need; the catalog builder fills in
# env_vars from OPTIONAL_ENV_VARS via prefix matching when not specified.
_PLATFORM_OVERRIDES: dict[str, dict[str, Any]] = {
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
    "email", "sms", "dingtalk", "feishu", "google_chat", "wecom", "wecom_callback",
    "weixin", "qqbot", "yuanbao", "api_server", "webhook",
)


def _messaging_platform_catalog() -> tuple[dict[str, Any], ...]:
    """Build the messaging catalog from the gateway's Platform enum + plugin registry.

    Built-ins come from ``gateway.config.Platform`` (LOCAL excluded); plugin platforms from
    ``platform_registry.plugin_entries()`` so new adapters appear without a code change here.
    UI metadata comes from each plugin's PlatformEntry, or :data:`_PLATFORM_OVERRIDES` for core-only
    platforms; the rest is derived from id/required_env.
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


def _platform_env_prefixes(platform_id: str) -> tuple[str, ...]:
    """Env-var prefixes owned by a messaging platform card (shared with the profile-clone
    channel stripper so a card and a clone agree on which keys belong to a platform)."""
    from hermes_cli.profile_channels import platform_env_prefixes
    return platform_env_prefixes(platform_id)


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


def _plugin_card_fields(plugin_entry: Any) -> dict[str, Any]:
    """A plugin's declared card metadata in override shape; undeclared fields are left out so
    the builder falls back exactly as for a platform with no metadata."""
    fields: dict[str, Any] = {
        "name": plugin_entry.dashboard_label, "description": plugin_entry.dashboard_description,
        "docs_url": plugin_entry.docs_url,
    }
    if plugin_entry.dashboard_env_vars:
        fields["env_vars"] = plugin_entry.dashboard_env_vars
    if plugin_entry.dashboard_required_env is not None:
        fields["required_env"] = plugin_entry.dashboard_required_env
    return fields


def _build_catalog_entry(platform_id: str, plugin_entry: Any | None = None) -> dict[str, Any]:
    override = _PLATFORM_OVERRIDES.get(platform_id) or (
        _plugin_card_fields(plugin_entry) if plugin_entry is not None else {})
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
    starting_process: bool = False
    session_reserved: bool = False
    expiry_timer: threading.Timer | None = None


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
_TELEGRAM_ONBOARDING_USER_AGENT = f"HermesDashboard/{get_version_info().base_version}"


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
