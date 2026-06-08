"""Multi-account QQBot adapter.

Runs multiple official QQ Bot connections under one Hermes ``qqbot`` platform.
Each child connection owns one app_id/client_secret pair; the parent keeps the
normal gateway/session semantics and routes outbound messages back through the
child that received the inbound chat.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Iterable
from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult

from .adapter import QQAdapter

logger = logging.getLogger(__name__)

Credential = Tuple[str, str, str]


def _strip(value: Any) -> str:
    return str(value or "").strip()


def _indexed_qq_credential_indexes(env: Optional[dict[str, str]] = None) -> list[int]:
    """Return sorted numeric suffixes for QQ_APP_ID_N / QQ_CLIENT_SECRET_N."""
    environ = env if env is not None else os.environ
    indexes: set[int] = set()
    prefixes = ("QQ_APP_ID_", "QQ_CLIENT_SECRET_")
    for key in environ:
        for prefix in prefixes:
            if key.startswith(prefix):
                suffix = key[len(prefix):]
                if suffix.isdigit():
                    indexes.add(int(suffix))
    return sorted(indexes)


def collect_qq_credentials(config: PlatformConfig) -> list[Credential]:
    """Collect QQ credentials from config, legacy env, and indexed env pairs.

    Returns ``[(label, app_id, client_secret), ...]``. Secrets are never logged.
    Duplicate app IDs are skipped; the first occurrence wins.
    """
    extra = config.extra or {}
    credentials: list[Credential] = []
    seen_app_ids: set[str] = set()

    def add(label: str, app_id: Any, client_secret: Any) -> None:
        app = _strip(app_id)
        secret = _strip(client_secret)
        if not app and not secret:
            return
        if not app or not secret:
            missing = "app_id" if not app else "client_secret"
            logger.warning("QQBot %s credential ignored: missing %s", label, missing)
            return
        if app in seen_app_ids:
            logger.info("QQBot %s credential ignored: duplicate app_id %s", label, app)
            return
        seen_app_ids.add(app)
        credentials.append((label, app, secret))

    # Config / legacy single-bot env.  Keep first so existing deployments keep
    # their default outbound route when QQ_APP_ID and QQ_APP_ID_0 both exist.
    add(
        "primary",
        extra.get("app_id") or os.getenv("QQ_APP_ID"),
        extra.get("client_secret") or os.getenv("QQ_CLIENT_SECRET"),
    )

    # Optional config-native account lists for future use:
    # platforms.qqbot.extra.accounts: [{app_id, client_secret, name?}, ...]
    accounts = extra.get("accounts") or extra.get("bots") or []
    if isinstance(accounts, Iterable) and not isinstance(accounts, (str, bytes, dict)):
        for idx, account in enumerate(accounts):
            if not isinstance(account, dict):
                continue
            add(
                _strip(account.get("name")) or f"config:{idx}",
                account.get("app_id") or account.get("appId"),
                account.get("client_secret") or account.get("clientSecret"),
            )

    # Indexed env pairs: QQ_APP_ID_0 / QQ_CLIENT_SECRET_0, etc.
    for idx in _indexed_qq_credential_indexes():
        add(
            f"env:{idx}",
            os.getenv(f"QQ_APP_ID_{idx}"),
            os.getenv(f"QQ_CLIENT_SECRET_{idx}"),
        )

    return credentials


def has_any_qq_credentials(config: PlatformConfig) -> bool:
    """True when at least one complete QQ credential pair is available."""
    return bool(collect_qq_credentials(config))


def _child_config(base: PlatformConfig, app_id: str, client_secret: str) -> PlatformConfig:
    extra = dict(base.extra or {})
    extra["app_id"] = app_id
    extra["client_secret"] = client_secret
    return replace(base, extra=extra)


class _QQChildAdapter(QQAdapter):
    """QQAdapter child that forwards normalized inbound events to the parent."""

    def __init__(self, config: PlatformConfig, parent: "QQMultiAdapter", label: str):
        self._multi_parent = parent
        self._multi_label = label
        super().__init__(config)

    @property
    def name(self) -> str:
        return f"QQBot[{self._multi_label}]"

    async def handle_message(self, event: MessageEvent) -> None:
        await self._multi_parent.handle_child_message(self, event)

    def _write_runtime_status_safe(self, context: str, **kwargs) -> None:
        # Parent owns the aggregate qqbot runtime status.  Child status writes
        # would race each other on the same platform key and make a partial
        # failure look like the whole qqbot platform died.
        del context, kwargs


class QQMultiAdapter(BasePlatformAdapter):
    """One Hermes qqbot adapter backed by multiple QQ bot credentials."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.QQBOT)
        self._credentials = collect_qq_credentials(config)
        self._children: list[_QQChildAdapter] = [
            _QQChildAdapter(_child_config(config, app_id, secret), self, label)
            for label, app_id, secret in self._credentials
        ]
        self._connected_children: list[_QQChildAdapter] = []
        self._chat_routes: Dict[str, _QQChildAdapter] = {}
        self._app_routes: Dict[str, _QQChildAdapter] = {
            _strip(getattr(child, "_app_id", "")): child for child in self._children
        }

    @property
    def name(self) -> str:
        return "QQBotMulti"

    @property
    def is_connected(self) -> bool:
        return any(child.is_connected for child in self._children)

    def _default_child(self) -> Optional[_QQChildAdapter]:
        for child in self._connected_children:
            if child.is_connected:
                return child
        for child in self._children:
            if child.is_connected:
                return child
        return self._children[0] if self._children else None

    def _resolve_child(
        self,
        chat_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[_QQChildAdapter]:
        meta = metadata or {}
        app_id = _strip(meta.get("qq_app_id") or meta.get("app_id"))
        if app_id and app_id in self._app_routes:
            return self._app_routes[app_id]
        child = self._chat_routes.get(str(chat_id))
        if child is not None:
            return child
        return self._default_child()

    def _remember_route(self, child: _QQChildAdapter, event: MessageEvent) -> None:
        source = event.source
        if not source:
            return
        if source.chat_id:
            self._chat_routes[str(source.chat_id)] = child
        child._chat_type_map[str(source.chat_id)] = self._qq_chat_type_for_source(event)
        if event.message_id:
            child._last_msg_id[str(source.chat_id)] = str(event.message_id)

    @staticmethod
    def _qq_chat_type_for_source(event: MessageEvent) -> str:
        source = event.source
        if not source:
            return "c2c"
        raw = event.raw_message if isinstance(event.raw_message, dict) else {}
        if source.chat_type == "group":
            return "guild" if raw.get("channel_id") else "group"
        if source.chat_type == "dm" and raw.get("guild_id"):
            return "dm"
        return "c2c"

    async def handle_child_message(self, child: _QQChildAdapter, event: MessageEvent) -> None:
        self._remember_route(child, event)
        await self.handle_message(event)

    async def connect(self) -> bool:
        if not self._children:
            message = (
                "QQ startup failed: QQ_APP_ID/QQ_CLIENT_SECRET or "
                "QQ_APP_ID_N/QQ_CLIENT_SECRET_N are required"
            )
            self._set_fatal_error("qq_missing_credentials", message, retryable=True)
            logger.warning("[%s] %s", self.name, message)
            return False

        self._connected_children.clear()
        results = await asyncio.gather(
            *(child.connect() for child in self._children),
            return_exceptions=True,
        )
        failures: list[str] = []
        for child, result in zip(self._children, results):
            if result is True and child.is_connected:
                self._connected_children.append(child)
                continue
            if isinstance(result, Exception):
                failures.append(f"{child.name}: {result}")
                logger.error("[%s] connect raised: %s", child.name, result, exc_info=True)
            else:
                failures.append(
                    f"{child.name}: {child.fatal_error_message or 'failed to connect'}"
                )

        if self._connected_children:
            self._mark_connected()
            if failures:
                logger.warning(
                    "[%s] connected %d/%d QQ bots; failures: %s",
                    self.name,
                    len(self._connected_children),
                    len(self._children),
                    "; ".join(failures),
                )
            else:
                logger.info("[%s] connected %d QQ bots", self.name, len(self._connected_children))
            return True

        message = "; ".join(failures) or "no QQ bot connected"
        retryable = any(child.fatal_error_retryable for child in self._children)
        self._set_fatal_error("qq_connect_error", message, retryable=retryable)
        logger.error("[%s] %s", self.name, message)
        return False

    async def disconnect(self) -> None:
        self._running = False
        await self.cancel_background_tasks()
        await asyncio.gather(
            *(child.disconnect() for child in self._children),
            return_exceptions=True,
        )
        self._connected_children.clear()
        self._chat_routes.clear()
        self._mark_disconnected()

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        child = self._resolve_child(chat_id, metadata)
        if child is None:
            return SendResult(success=False, error="No QQ bot credentials configured", retryable=True)
        return await child.send(chat_id, content, reply_to=reply_to, metadata=metadata)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        child = self._resolve_child(chat_id, metadata)
        if child is not None and hasattr(child, "send_typing"):
            await child.send_typing(chat_id, metadata=metadata)

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        child = self._resolve_child(chat_id, metadata)
        if child is None:
            return SendResult(success=False, error="No QQ bot credentials configured", retryable=True)
        return await child.send_image(chat_id, image_url, caption=caption, reply_to=reply_to, metadata=metadata)

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        child = self._resolve_child(chat_id, kwargs.get("metadata"))
        if child is None:
            return SendResult(success=False, error="No QQ bot credentials configured", retryable=True)
        return await child.send_image_file(chat_id, image_path, caption=caption, reply_to=reply_to, **kwargs)

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        child = self._resolve_child(chat_id, kwargs.get("metadata"))
        if child is None:
            return SendResult(success=False, error="No QQ bot credentials configured", retryable=True)
        return await child.send_voice(chat_id, audio_path, caption=caption, reply_to=reply_to, **kwargs)

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        child = self._resolve_child(chat_id, kwargs.get("metadata"))
        if child is None:
            return SendResult(success=False, error="No QQ bot credentials configured", retryable=True)
        return await child.send_video(chat_id, video_path, caption=caption, reply_to=reply_to, **kwargs)

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        child = self._resolve_child(chat_id, kwargs.get("metadata"))
        if child is None:
            return SendResult(success=False, error="No QQ bot credentials configured", retryable=True)
        return await child.send_document(
            chat_id,
            file_path,
            caption=caption,
            file_name=file_name,
            reply_to=reply_to,
            **kwargs,
        )

    async def send_exec_approval(
        self,
        chat_id: str,
        command: str,
        session_key: str,
        description: str = "dangerous command",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        child = self._resolve_child(chat_id, metadata)
        if child is None:
            return SendResult(success=False, error="No QQ bot credentials configured", retryable=True)
        return await child.send_exec_approval(chat_id, command, session_key, description, metadata=metadata)

    async def send_update_prompt(
        self,
        chat_id: str,
        prompt: str,
        default: str = "",
        session_key: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        child = self._resolve_child(chat_id, metadata)
        if child is None:
            return SendResult(success=False, error="No QQ bot credentials configured", retryable=True)
        return await child.send_update_prompt(chat_id, prompt, default, session_key, metadata=metadata)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        child = self._resolve_child(chat_id)
        if child is None:
            return {"name": chat_id, "type": "dm"}
        return await child.get_chat_info(chat_id)

    def format_message(self, content: str) -> str:
        child = self._default_child()
        if child is None:
            return content
        return child.format_message(content)
