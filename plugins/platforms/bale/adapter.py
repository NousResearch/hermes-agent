"""Bale messenger platform adapter.

Bale exposes a Telegram-compatible Bot API.  This plugin reuses Hermes'
Telegram transport, but deliberately keeps Bale credentials, authorization,
session identity, and standalone delivery separate from Telegram.

The adapter does not log inbound updates or message bodies.  This is important
for a messaging transport because diagnostic logs are commonly retained much
longer than the messages they describe.
"""

from __future__ import annotations

from typing import Optional

from agent.secret_scope import get_secret
from gateway.config import Platform, PlatformConfig
from gateway.session import SessionSource
from plugins.platforms.telegram.adapter import (
    TelegramAdapter,
    check_telegram_requirements,
    telegram_deps_present,
)

_TOKEN_ENV = "BALE_BOT_TOKEN"
_ALLOWED_ENV = "BALE_ALLOWED_USERS"
_ALLOW_ALL_ENV = "BALE_ALLOW_ALL_USERS"
_HOME_ENV = "BALE_HOME_CHANNEL"
_DEFAULT_API_BASE = "https://tapi.bale.ai"


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _api_base() -> str:
    """Return the PTB base URL, which must end in ``/bot``."""
    base = get_secret("BALE_API_BASE_URL", _DEFAULT_API_BASE).strip().rstrip("/")
    return base if base.endswith("/bot") else f"{base}/bot"


def _allowed_users() -> set[str]:
    raw = get_secret(_ALLOWED_ENV, "")
    allowed = {item.strip() for item in raw.split(",") if item.strip()}
    if _truthy(get_secret(_ALLOW_ALL_ENV, "")):
        allowed.add("*")
    return allowed


class BaleAdapter(TelegramAdapter):
    """Hermes' Telegram transport pointed at Bale's Bot API."""

    def __init__(self, config: PlatformConfig):
        token = get_secret(_TOKEN_ENV, "").strip()
        if token:
            config.token = token

        extra = dict(getattr(config, "extra", {}) or {})
        base = _api_base()
        extra["base_url"] = base
        extra["base_file_url"] = base
        extra["proxy_env_var"] = "BALE_PROXY"

        allowed = _allowed_users()
        if allowed:
            # TelegramAdapter consults adapter-local allowlists before its
            # Telegram-specific environment fallback.
            extra["allow_from"] = sorted(allowed)
            extra["group_allow_from"] = sorted(allowed)
        config.extra = extra

        super().__init__(config)
        self.platform = Platform("bale")

    def _source_from_message_for_auth(self, message):
        source = super()._source_from_message_for_auth(message)
        source.platform = self.platform
        return source

    def _telegram_auth_env_configured(self) -> bool:
        # The inherited intake gate must not borrow TELEGRAM_* authorization.
        return bool(_allowed_users())

    def _is_callback_user_authorized(
        self,
        user_id: str,
        *,
        chat_id: Optional[str] = None,
        chat_type: Optional[str] = None,
        thread_id: Optional[str] = None,
        user_name: Optional[str] = None,
    ) -> bool:
        normalized = str(user_id or "").strip()
        if not normalized:
            return False

        runner = getattr(getattr(self, "_message_handler", None), "__self__", None)
        auth_fn = getattr(runner, "_is_user_authorized", None)
        if callable(auth_fn):
            normalized_chat_type = str(chat_type or "dm").strip().lower() or "dm"
            if normalized_chat_type == "private":
                normalized_chat_type = "dm"
            elif normalized_chat_type == "supergroup":
                normalized_chat_type = "forum" if thread_id is not None else "group"
            source = SessionSource(
                platform=self.platform,
                chat_id=str(chat_id or normalized),
                chat_type=normalized_chat_type,
                user_id=normalized,
                user_name=str(user_name).strip() if user_name else None,
                thread_id=str(thread_id) if thread_id is not None else None,
            )
            try:
                return bool(auth_fn(source))
            except Exception:
                # Fall through to the fail-closed Bale allowlist below.
                pass

        allowed = _allowed_users()
        return "*" in allowed or normalized in allowed

    def _should_pass_unauthorized_dm_for_pairing(self, source) -> bool:
        if source.chat_type != "dm":
            return False
        runner = getattr(getattr(self, "_message_handler", None), "__self__", None)
        behavior_fn = getattr(runner, "_get_unauthorized_dm_behavior", None)
        if callable(behavior_fn):
            try:
                return (
                    behavior_fn(
                        self.platform,
                        profile=getattr(source, "profile", None),
                    )
                    == "pair"
                )
            except Exception:
                pass
        extra = getattr(getattr(self, "config", None), "extra", None) or {}
        return str(extra.get("unauthorized_dm_behavior", "")).strip().lower() == "pair"


def _build_adapter(config: PlatformConfig) -> BaleAdapter:
    adapter = BaleAdapter(config)
    adapter._notifications_mode = "important"
    return adapter


def _is_connected(config: PlatformConfig) -> bool:
    token = get_secret(_TOKEN_ENV, "").strip()
    return bool(token or str(getattr(config, "token", "") or "").strip())


def _apply_yaml_config(_yaml_cfg: dict, _bale_cfg: dict) -> dict:
    base = _api_base()
    return {
        "base_url": base,
        "base_file_url": base,
        "proxy_env_var": "BALE_PROXY",
    }


async def _standalone_send(
    pconfig,
    chat_id,
    message,
    *,
    thread_id=None,
    media_files=None,
    force_document=False,
):
    """Deliver cron/tool messages without a co-resident gateway process."""
    token = get_secret(_TOKEN_ENV, "").strip()
    if not token:
        token = str(getattr(pconfig, "token", "") or "").strip()
    from tools.send_message_senders import _send_telegram

    return await _send_telegram(
        token,
        chat_id,
        message,
        media_files=media_files,
        thread_id=thread_id,
        disable_link_previews=bool(
            getattr(pconfig, "extra", {})
            and pconfig.extra.get("disable_link_previews")
        ),
        force_document=force_document,
        base_url=_api_base(),
        base_file_url=_api_base(),
        proxy_env_var="BALE_PROXY",
    )


def register(ctx) -> None:
    ctx.register_platform(
        name="bale",
        label="Bale",
        adapter_factory=_build_adapter,
        check_fn=telegram_deps_present,
        ensure_deps_fn=check_telegram_requirements,
        is_connected=_is_connected,
        required_env=[_TOKEN_ENV],
        install_hint="Set BALE_BOT_TOKEN in ~/.hermes/.env.",
        apply_yaml_config_fn=_apply_yaml_config,
        allowed_users_env=_ALLOWED_ENV,
        allow_all_env=_ALLOW_ALL_ENV,
        cron_deliver_env_var=_HOME_ENV,
        standalone_sender_fn=_standalone_send,
        max_message_length=4096,
        emoji="💬",
        allow_update_command=True,
    )
