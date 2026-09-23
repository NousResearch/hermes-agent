"""Small, proof-gated MTProto controller for private bot forum topics."""

from __future__ import annotations

import asyncio
import os
import re
from datetime import UTC, datetime
from typing import Any, Callable


_TELEGRAM_ID_RE = re.compile(r"^-?\d+$")
_API_HASH_RE = re.compile(r"^[0-9a-fA-F]{32}$")
_START_TIMEOUT_SECONDS = 30.0
_REQUEST_TIMEOUT_SECONDS = 20.0
_STOP_TIMEOUT_SECONDS = 10.0
_SAFE_ERROR_CODES = frozenset({
    "topic_already_closed",
    "topic_already_open",
    "topic_control_forbidden",
    "topic_control_unavailable",
    "topic_control_unsupported",
    "topic_not_found",
})


class MtprotoTopicControlError(RuntimeError):
    """Closed-set failure from the MTProto topic controller."""

    def __init__(self, code: str) -> None:
        self.code = code if code in _SAFE_ERROR_CODES else "topic_control_unavailable"
        super().__init__(self.code)


def _classify_error(error: BaseException, *, closed: bool) -> str:
    text = str(error).lower()
    error_name = type(error).__name__.lower()
    if (
        any(
            marker in text
            for marker in ("message not modified", "message was not modified")
        )
        or "notmodified" in error_name
    ):
        return "topic_already_closed" if closed else "topic_already_open"
    if any(
        marker in text
        for marker in (
            "topic_not_modified",
            "topic not modified",
            "topic is already closed",
            "already closed",
            "topic_closed",
        )
    ):
        return "topic_already_closed" if closed else "topic_control_unavailable"
    if any(
        marker in text
        for marker in (
            "channel_invalid",
            "chat_not_found",
            "peer_id_invalid",
            "topic_id_invalid",
            "message_thread_not_found",
            "topic not found",
        )
    ) or any(
        marker in error_name
        for marker in (
            "channelinvalid",
            "chatidinvalid",
            "messageidinvalid",
            "msgidinvalid",
            "peeridinvalid",
            "topicdeleted",
        )
    ):
        return "topic_not_found"
    if any(
        marker in text
        for marker in (
            "chat admin required",
            "chat write forbidden",
            "user is banned",
            "forbidden",
        )
    ) or any(
        marker in error_name
        for marker in ("chatadminrequired", "chatwriteforbidden", "banned")
    ):
        return "topic_control_forbidden"
    if any(
        marker in text
        for marker in (
            "method not found",
            "not a forum",
            "forum topics are disabled",
            "forums_disabled",
        )
    ) or any(
        marker in error_name for marker in ("channelforummissing", "forumdisabled")
    ):
        return "topic_control_unsupported"
    return "topic_control_unavailable"


def _default_client_factory(session: Any, api_id: int, api_hash: str) -> Any:
    try:
        from telethon import TelegramClient
    except ImportError as error:  # pragma: no cover - optional runtime dependency
        raise MtprotoTopicControlError("topic_control_unavailable") from error
    return TelegramClient(session, api_id, api_hash)


def _default_request_factory(**kwargs: Any) -> Any:
    try:
        from telethon.tl.functions.messages import EditForumTopicRequest
    except ImportError as error:  # pragma: no cover - optional runtime dependency
        raise MtprotoTopicControlError("topic_control_unavailable") from error
    return EditForumTopicRequest(**kwargs)


class MTProtoPrivateTopicController:
    """Close one configured private bot topic through ``messages.editForumTopic``.

    The controller deliberately keeps an in-memory Telethon session.  A supplied
    ``HERMES_BECKY_LOOPS_MTPROTO_SESSION`` value may restore a session, but the
    bot token remains the source of authentication when no session is present.
    """

    method = "mtproto_private_topic"

    def __init__(
        self,
        *,
        api_id: int,
        api_hash: str,
        bot_token: str,
        chat_id: str,
        session_string: str | None = None,
        client_factory: Callable[[Any, int, str], Any] | None = None,
        request_factory: Callable[..., Any] | None = None,
    ) -> None:
        if api_id <= 0 or not _API_HASH_RE.fullmatch(api_hash) or not bot_token:
            raise ValueError("MTProto credentials are invalid")
        if not _TELEGRAM_ID_RE.fullmatch(str(chat_id).strip()):
            raise ValueError("MTProto chat id is invalid")
        self._api_id = api_id
        self._api_hash = api_hash
        self._bot_token = bot_token
        self._chat_id = str(chat_id).strip()
        self._session_string = session_string.strip() if session_string else None
        self._client_factory = client_factory or _default_client_factory
        self._request_factory = request_factory or _default_request_factory
        self._client: Any | None = None
        self._start_lock = asyncio.Lock()

    @classmethod
    def from_environment(cls, *, chat_id: str) -> MTProtoPrivateTopicController | None:
        raw_api_id = os.getenv("TELEGRAM_API_ID", "").strip()
        api_hash = os.getenv("TELEGRAM_API_HASH", "").strip()
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        if not raw_api_id or not api_hash or not bot_token:
            return None
        try:
            api_id = int(raw_api_id)
        except ValueError:
            return None
        try:
            return cls(
                api_id=api_id,
                api_hash=api_hash,
                bot_token=bot_token,
                chat_id=chat_id,
                session_string=os.getenv("HERMES_BECKY_LOOPS_MTPROTO_SESSION", ""),
            )
        except ValueError:
            return None

    @property
    def is_connected(self) -> bool:
        client = self._client
        if client is None:
            return False
        state = getattr(client, "is_connected", False)
        try:
            return bool(state() if callable(state) else state)
        except Exception:
            return False

    @property
    def supports_close(self) -> bool:
        return self.is_connected

    async def start(self) -> None:
        async with self._start_lock:
            if self.is_connected:
                return
            session = self._session_string
            if session:
                try:
                    from telethon.sessions import StringSession
                except ImportError as error:  # pragma: no cover
                    raise MtprotoTopicControlError(
                        "topic_control_unavailable"
                    ) from error
                try:
                    session_value: Any = StringSession(session)
                except Exception:
                    raise MtprotoTopicControlError(
                        "topic_control_unavailable"
                    ) from None
            else:
                session_value = None
            client = self._client_factory(session_value, self._api_id, self._api_hash)
            try:
                await asyncio.wait_for(
                    client.start(bot_token=self._bot_token),
                    timeout=_START_TIMEOUT_SECONDS,
                )
            except asyncio.CancelledError:
                try:
                    await asyncio.shield(
                        asyncio.wait_for(
                            client.disconnect(), timeout=_STOP_TIMEOUT_SECONDS
                        )
                    )
                except BaseException:
                    pass
                raise
            except Exception as error:
                try:
                    await asyncio.wait_for(
                        client.disconnect(), timeout=_STOP_TIMEOUT_SECONDS
                    )
                except Exception:
                    pass
                raise MtprotoTopicControlError(
                    _classify_error(error, closed=True)
                ) from None
            if self._session_string:
                try:
                    get_me = getattr(client, "get_me")
                    identity = await asyncio.wait_for(
                        get_me(), timeout=_REQUEST_TIMEOUT_SECONDS
                    )
                except asyncio.CancelledError:
                    try:
                        await asyncio.shield(
                            asyncio.wait_for(
                                client.disconnect(), timeout=_STOP_TIMEOUT_SECONDS
                            )
                        )
                    except BaseException:
                        pass
                    raise
                except Exception:
                    try:
                        await asyncio.wait_for(
                            client.disconnect(), timeout=_STOP_TIMEOUT_SECONDS
                        )
                    except Exception:
                        pass
                    raise MtprotoTopicControlError(
                        "topic_control_unavailable"
                    ) from None
                bot_id, separator, _ = self._bot_token.partition(":")
                if (
                    not separator
                    or not bot_id.isdigit()
                    or not bool(getattr(identity, "bot", False))
                    or str(getattr(identity, "id", "")) != bot_id
                ):
                    try:
                        await asyncio.wait_for(
                            client.disconnect(), timeout=_STOP_TIMEOUT_SECONDS
                        )
                    except Exception:
                        pass
                    raise MtprotoTopicControlError(
                        "topic_control_unavailable"
                    ) from None
            self._client = client
            if not self.is_connected:
                self._client = None
                try:
                    await asyncio.wait_for(
                        client.disconnect(), timeout=_STOP_TIMEOUT_SECONDS
                    )
                except Exception:
                    pass
                raise MtprotoTopicControlError("topic_control_unavailable")

    async def stop(self) -> None:
        client, self._client = self._client, None
        if client is None:
            return
        try:
            await asyncio.wait_for(client.disconnect(), timeout=_STOP_TIMEOUT_SECONDS)
        except asyncio.CancelledError:
            try:
                await asyncio.shield(
                    asyncio.wait_for(client.disconnect(), timeout=_STOP_TIMEOUT_SECONDS)
                )
            except BaseException:
                pass
            raise
        except Exception:
            return

    async def close_topic(self, *, chat_id: str, thread_id: str) -> datetime:
        return await self.set_topic_closed(
            chat_id=chat_id,
            thread_id=thread_id,
            closed=True,
        )

    async def set_topic_closed(
        self, *, chat_id: str, thread_id: str, closed: bool
    ) -> datetime:
        if str(chat_id).strip() != self._chat_id:
            raise MtprotoTopicControlError("topic_control_unavailable")
        if not _TELEGRAM_ID_RE.fullmatch(str(thread_id).strip()) or str(
            thread_id
        ).strip() in {"0", "1"}:
            raise MtprotoTopicControlError("topic_not_found")
        if not isinstance(closed, bool):
            raise MtprotoTopicControlError("topic_control_unavailable")
        if not self.is_connected:
            raise MtprotoTopicControlError("topic_control_unavailable")
        client = self._client
        assert client is not None
        try:
            request = self._request_factory(
                peer=int(self._chat_id),
                topic_id=int(thread_id),
                closed=closed,
            )
            await asyncio.wait_for(client(request), timeout=_REQUEST_TIMEOUT_SECONDS)
        except asyncio.CancelledError:
            raise
        except MtprotoTopicControlError:
            raise
        except Exception as error:
            raise MtprotoTopicControlError(
                _classify_error(error, closed=closed)
            ) from None
        return datetime.now(UTC)
