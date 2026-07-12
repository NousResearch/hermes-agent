"""Concrete, token-owning Discord REST v10 edge adapter.

This module is the narrow external-I/O implementation for
``gateway.discord_edge_runtime``.  Its public surface contains only the three
fixed mutations and their matching proof/readback operations.  It deliberately
does not expose a raw URL, HTTP method, or Discord request dispatcher.

The adapter loads the bot token once from an explicitly named credential file.
The file is opened without following symlinks and its owner, mode, link count,
path, and inode are checked before and after the bounded read.  The token is
never included in a return value or exception.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import http.client
import io
import json
import math
import os
import re
import socket
import stat
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol

from gateway.discord_edge_protocol import (
    DiscordEdgeOperation,
    DiscordEdgeThreadReadback,
    DiscordPublicTarget,
    DiscordPublicTargetType,
)
from gateway.discord_edge_runtime import (
    DiscordLivePublicTargetProof,
    DiscordMutationAccepted,
    DiscordMutationReadback,
)

_DISCORD_API_V10 = "https://discord.com/api/v10"
_USER_AGENT = "Muncho-Privileged-Discord-Edge/1"
_MAX_CREDENTIAL_BYTES = 512
_MAX_REQUEST_BODY_BYTES = 16 * 1024
_MAX_RESPONSE_BODY_BYTES = 256 * 1024
_MAX_RATE_LIMIT_RETRY_SECONDS = 2.0
_MAX_HTTP_ATTEMPTS = 3
_MAX_ROLES = 1_000
_MAX_OVERWRITES = 2_000
_MAX_JSON_DEPTH = 64
_MAX_JSON_NODES = 50_000
_MAX_JSON_NUMBER_CHARS = 64
_SNOWFLAKE_RE = re.compile(r"^[0-9]{1,25}$")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9._-]{16,511}$")

# Discord permission bits.  These constants are mechanical API contract data;
# they do not classify user text or choose an operation.
_ADMINISTRATOR = 1 << 3
_VIEW_CHANNEL = 1 << 10
_SEND_MESSAGES = 1 << 11
_READ_MESSAGE_HISTORY = 1 << 16
_CREATE_PUBLIC_THREADS = 1 << 35
_SEND_MESSAGES_IN_THREADS = 1 << 38
_ALL_PERMISSION_BITS = (1 << 64) - 1

_GUILD_TEXT = 0
_DM = 1
_GUILD_VOICE = 2
_GROUP_DM = 3
_GUILD_ANNOUNCEMENT = 5
_ANNOUNCEMENT_THREAD = 10
_PUBLIC_THREAD = 11
_PRIVATE_THREAD = 12
_GUILD_STAGE_VOICE = 13
_GUILD_FORUM = 15

_PUBLIC_MESSAGE_CHANNEL_TYPES = frozenset(
    {
        _GUILD_TEXT,
        _GUILD_VOICE,
        _GUILD_ANNOUNCEMENT,
        _GUILD_STAGE_VOICE,
    }
)
_PUBLIC_THREAD_TYPES = frozenset({_ANNOUNCEMENT_THREAD, _PUBLIC_THREAD})
_PUBLIC_THREAD_PARENT_TYPES = frozenset(
    {_GUILD_TEXT, _GUILD_ANNOUNCEMENT, _GUILD_FORUM}
)
_FORBIDDEN_CHANNEL_TYPES = frozenset({_DM, _GROUP_DM, _PRIVATE_THREAD})
_SAFE_ALLOWED_MENTIONS: Mapping[str, object] = {
    "parse": [],
    "roles": [],
    "users": [],
    "replied_user": False,
}
_IGNORE_REPLY_BINDING = object()


class DiscordRestEdgeErrorCode(StrEnum):
    """Stable, secret-free error categories for the concrete edge."""

    CREDENTIAL_INVALID = "credential_invalid"
    API_UNAVAILABLE = "discord_api_unavailable"
    API_REJECTED = "discord_api_rejected"
    API_RATE_LIMITED = "discord_api_rate_limited"
    RESPONSE_INVALID = "discord_response_invalid"
    RESPONSE_TOO_LARGE = "discord_response_too_large"
    TARGET_MISMATCH = "discord_target_mismatch"
    TARGET_NOT_PUBLIC = "discord_target_not_public"
    BOT_PERMISSION_REVOKED = "discord_bot_permission_revoked"
    REQUEST_DEADLINE_EXPIRED = "request_deadline_expired"
    BOT_IDENTITY_MISMATCH = "discord_bot_identity_mismatch"
    MUTATION_BINDING_MISMATCH = "discord_mutation_binding_mismatch"


class DiscordRestEdgeError(RuntimeError):
    """One stable failure that never reflects Discord bodies or credentials."""

    def __init__(self, code: DiscordRestEdgeErrorCode, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}: {detail}")


def _fail(code: DiscordRestEdgeErrorCode, detail: str) -> None:
    raise DiscordRestEdgeError(code, detail)


def _normalized_absolute_path(value: str | os.PathLike[str], label: str) -> Path:
    raw = Path(value)
    if not raw.is_absolute():
        _fail(DiscordRestEdgeErrorCode.CREDENTIAL_INVALID, f"{label} must be absolute")
    normalized = Path(os.path.normpath(os.fspath(raw)))
    if normalized != raw:
        _fail(
            DiscordRestEdgeErrorCode.CREDENTIAL_INVALID,
            f"{label} must be normalized",
        )
    return normalized


class _DiscordBotToken:
    """Opaque in-process token value with an intentionally redacted repr."""

    __slots__ = ("_authorization",)

    def __init__(self, token: str) -> None:
        self._authorization = f"Bot {token}"

    def authorization_header(self) -> str:
        return self._authorization

    def __repr__(self) -> str:
        return "<Discord bot credential: redacted>"


def _assert_credential_stat(
    file_stat: os.stat_result,
    *,
    expected_owner_uid: int,
) -> None:
    if not stat.S_ISREG(file_stat.st_mode):
        _fail(
            DiscordRestEdgeErrorCode.CREDENTIAL_INVALID,
            "Discord credential must be a regular file",
        )
    if file_stat.st_nlink != 1:
        _fail(
            DiscordRestEdgeErrorCode.CREDENTIAL_INVALID,
            "Discord credential must have exactly one link",
        )
    if file_stat.st_uid != expected_owner_uid:
        _fail(
            DiscordRestEdgeErrorCode.CREDENTIAL_INVALID,
            "Discord credential owner does not match the service contract",
        )
    if stat.S_IMODE(file_stat.st_mode) != 0o400:
        _fail(
            DiscordRestEdgeErrorCode.CREDENTIAL_INVALID,
            "Discord credential must have exact mode 0400",
        )
    if file_stat.st_size <= 0 or file_stat.st_size > _MAX_CREDENTIAL_BYTES:
        _fail(
            DiscordRestEdgeErrorCode.CREDENTIAL_INVALID,
            "Discord credential size is outside the fixed bound",
        )


def _credential_identity(file_stat: os.stat_result) -> tuple[int, ...]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _load_bot_token(
    credential_path: str | os.PathLike[str],
    *,
    credentials_directory: str | os.PathLike[str],
    expected_owner_uid: int,
) -> _DiscordBotToken:
    """Load one explicit systemd/root credential through a TOCTOU-safe read."""

    if (
        isinstance(expected_owner_uid, bool)
        or not isinstance(expected_owner_uid, int)
        or expected_owner_uid < 0
    ):
        _fail(
            DiscordRestEdgeErrorCode.CREDENTIAL_INVALID,
            "expected credential owner is invalid",
        )
    path = _normalized_absolute_path(credential_path, "credential path")
    directory = _normalized_absolute_path(
        credentials_directory,
        "credentials directory",
    )
    if path.parent != directory:
        _fail(
            DiscordRestEdgeErrorCode.CREDENTIAL_INVALID,
            "Discord credential must be directly inside its explicit directory",
        )
    try:
        resolved_directory = directory.resolve(strict=True)
        directory_stat = os.stat(directory, follow_symlinks=False)
    except OSError:
        raise DiscordRestEdgeError(
            DiscordRestEdgeErrorCode.CREDENTIAL_INVALID,
            "Discord credentials directory is unavailable",
        ) from None
    if resolved_directory != directory or not stat.S_ISDIR(directory_stat.st_mode):
        _fail(
            DiscordRestEdgeErrorCode.CREDENTIAL_INVALID,
            "Discord credentials directory must be canonical and symlink-free",
        )
    if directory_stat.st_uid not in {0, expected_owner_uid}:
        _fail(
            DiscordRestEdgeErrorCode.CREDENTIAL_INVALID,
            "Discord credentials directory owner is invalid",
        )
    if directory_stat.st_mode & 0o022:
        _fail(
            DiscordRestEdgeErrorCode.CREDENTIAL_INVALID,
            "Discord credentials directory is group/world writable",
        )

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError:
        raise DiscordRestEdgeError(
            DiscordRestEdgeErrorCode.CREDENTIAL_INVALID,
            "Discord credential cannot be opened safely",
        ) from None
    try:
        opened_stat = os.fstat(fd)
        _assert_credential_stat(opened_stat, expected_owner_uid=expected_owner_uid)
        before_stat = os.lstat(path)
        _assert_credential_stat(before_stat, expected_owner_uid=expected_owner_uid)
        expected_identity = _credential_identity(opened_stat)
        if _credential_identity(before_stat) != expected_identity:
            _fail(
                DiscordRestEdgeErrorCode.CREDENTIAL_INVALID,
                "Discord credential identity changed before read",
            )
        body = os.read(fd, _MAX_CREDENTIAL_BYTES + 1)
        if len(body) > _MAX_CREDENTIAL_BYTES or os.read(fd, 1):
            _fail(
                DiscordRestEdgeErrorCode.CREDENTIAL_INVALID,
                "Discord credential exceeds the fixed bound",
            )
        after_stat = os.fstat(fd)
        path_after_stat = os.lstat(path)
        _assert_credential_stat(after_stat, expected_owner_uid=expected_owner_uid)
        _assert_credential_stat(path_after_stat, expected_owner_uid=expected_owner_uid)
        if (
            _credential_identity(after_stat) != expected_identity
            or _credential_identity(path_after_stat) != expected_identity
        ):
            _fail(
                DiscordRestEdgeErrorCode.CREDENTIAL_INVALID,
                "Discord credential identity changed during read",
            )
    except OSError:
        raise DiscordRestEdgeError(
            DiscordRestEdgeErrorCode.CREDENTIAL_INVALID,
            "Discord credential could not be read safely",
        ) from None
    finally:
        os.close(fd)

    if body.endswith(b"\n"):
        body = body[:-1]
    invalid_encoding = False
    try:
        token = body.decode("ascii")
    except UnicodeDecodeError:
        invalid_encoding = True
        token = ""
    if invalid_encoding:
        _fail(
            DiscordRestEdgeErrorCode.CREDENTIAL_INVALID,
            "Discord credential is not valid ASCII",
        )
    if not _TOKEN_RE.fullmatch(token):
        _fail(
            DiscordRestEdgeErrorCode.CREDENTIAL_INVALID,
            "Discord credential has an invalid bounded format",
        )
    return _DiscordBotToken(token)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ValueError("non-JSON numeric constant")


def _parse_json_int(value: str) -> int:
    if len(value) > _MAX_JSON_NUMBER_CHARS:
        raise ValueError("JSON integer exceeds the fixed digit bound")
    return int(value)


def _parse_json_float(value: str) -> float:
    if len(value) > _MAX_JSON_NUMBER_CHARS:
        raise ValueError("JSON float exceeds the fixed character bound")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("JSON float is not finite")
    return result


def _validate_json_tree(value: object) -> None:
    stack: list[tuple[object, int]] = [(value, 0)]
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            _fail(
                DiscordRestEdgeErrorCode.RESPONSE_INVALID,
                "Discord JSON structure exceeds the fixed complexity bound",
            )
        if isinstance(current, dict):
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)


def _strict_json(body: bytes) -> object:
    invalid = False
    try:
        value = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
            parse_float=_parse_json_float,
            parse_int=_parse_json_int,
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        OverflowError,
        RecursionError,
    ):
        invalid = True
        value = None
    if invalid:
        _fail(
            DiscordRestEdgeErrorCode.RESPONSE_INVALID,
            "Discord returned malformed strict JSON",
        )
    _validate_json_tree(value)
    return value


def _json_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        _fail(
            DiscordRestEdgeErrorCode.RESPONSE_INVALID,
            f"Discord {label} response is not an object",
        )
    return value


def _snowflake(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not _SNOWFLAKE_RE.fullmatch(value)
        or int(value) == 0
    ):
        _fail(
            DiscordRestEdgeErrorCode.RESPONSE_INVALID,
            f"Discord {label} is not a snowflake",
        )
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(
            DiscordRestEdgeErrorCode.RESPONSE_INVALID,
            f"Discord {label} is not an integer",
        )
    return value


def _permission_bits(value: object, label: str) -> int:
    if (
        not isinstance(value, str)
        or len(value) > 20
        or not value.isascii()
        or not value.isdecimal()
    ):
        _fail(
            DiscordRestEdgeErrorCode.RESPONSE_INVALID,
            f"Discord {label} permission value is invalid",
        )
    bits = int(value)
    if bits < 0 or bits > _ALL_PERMISSION_BITS:
        _fail(
            DiscordRestEdgeErrorCode.RESPONSE_INVALID,
            f"Discord {label} permission value is out of range",
        )
    return bits


class _ResponseLike(Protocol):
    status: int
    headers: Mapping[str, str]

    def read(self, amount: int = -1) -> bytes: ...

    def __enter__(self) -> _ResponseLike: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...


class _OpenerLike(Protocol):
    def open(
        self,
        request: urllib.request.Request,
        timeout: float,
    ) -> _ResponseLike: ...


class _BufferedDiscordResponse:
    def __init__(
        self,
        *,
        status: int,
        headers: Mapping[str, str],
        body: bytes,
    ) -> None:
        self.status = status
        self.headers = headers
        self._body = io.BytesIO(body)

    def read(self, amount: int = -1) -> bytes:
        return self._body.read(amount)

    def __enter__(self) -> _BufferedDiscordResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        self._body.close()


class _AiohttpTotalDeadlineOpener:
    """Production HTTPS exchange with a hard total monotonic deadline."""

    uses_environment_proxies = False

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run_loop,
            name="discord-edge-http-loop",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=1.0):
            self._closed = True
            raise RuntimeError("Discord HTTPS event loop did not start")

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()
        pending = asyncio.all_tasks(self._loop)
        for task in pending:
            task.cancel()
        if pending:
            self._loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
        self._loop.close()

    @staticmethod
    async def _exchange(
        request: urllib.request.Request,
        *,
        timeout: float,
    ) -> _BufferedDiscordResponse:
        try:
            import aiohttp
        except ImportError:
            raise urllib.error.URLError("Discord HTTPS transport unavailable") from None

        client_timeout = aiohttp.ClientTimeout(total=timeout)
        headers = dict(request.header_items())
        headers["Accept-Encoding"] = "identity"
        headers["Connection"] = "close"
        try:
            async with asyncio.timeout(timeout):
                async with aiohttp.ClientSession(
                    auto_decompress=False,
                    timeout=client_timeout,
                    trust_env=False,
                ) as session:
                    async with session.request(
                        request.method,
                        request.full_url,
                        allow_redirects=False,
                        data=request.data,
                        headers=headers,
                    ) as response:
                        response_headers = dict(response.headers)
                        chunks: list[bytes] = []
                        size = 0
                        async for chunk in response.content.iter_chunked(16 * 1024):
                            size += len(chunk)
                            if size > _MAX_RESPONSE_BODY_BYTES:
                                chunks = [b"x" * (_MAX_RESPONSE_BODY_BYTES + 1)]
                                break
                            chunks.append(bytes(chunk))
                        body = b"".join(chunks)
                        buffered = _BufferedDiscordResponse(
                            status=response.status,
                            headers=response_headers,
                            body=body,
                        )
        except (TimeoutError, aiohttp.ClientError, OSError):
            raise urllib.error.URLError("Discord HTTPS transport failed") from None

        if buffered.status >= 400:
            raise urllib.error.HTTPError(
                request.full_url,
                buffered.status,
                "Discord HTTP status",
                dict(buffered.headers),
                io.BytesIO(body),
            )
        return buffered

    def open(
        self,
        request: urllib.request.Request,
        timeout: float,
    ) -> _BufferedDiscordResponse:
        if self._closed:
            raise urllib.error.URLError("Discord HTTPS transport is closed")
        coroutine = self._exchange(request, timeout=timeout)
        try:
            exchange = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        except RuntimeError:
            coroutine.close()
            raise urllib.error.URLError(
                "Discord HTTPS event loop unavailable"
            ) from None
        try:
            return exchange.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            exchange.cancel()
            raise urllib.error.URLError("Discord HTTPS total deadline expired") from None
        except concurrent.futures.CancelledError:
            raise urllib.error.URLError("Discord HTTPS exchange was cancelled") from None
        except RuntimeError:
            exchange.cancel()
            raise urllib.error.URLError("Discord HTTPS event loop unavailable") from None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=1.0)


class _DiscordApiV10:
    """Private fixed-endpoint client; no raw request surface escapes this class."""

    def __init__(
        self,
        token: _DiscordBotToken,
        *,
        timeout_seconds: float,
        opener: _OpenerLike | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(token, _DiscordBotToken):
            raise TypeError("token must be loaded by the credential boundary")
        if not 0.1 <= timeout_seconds <= 15.0:
            raise ValueError("Discord timeout must be between 0.1 and 15 seconds")
        self._token = token
        self._timeout_seconds = timeout_seconds
        self._opener = opener or _AiohttpTotalDeadlineOpener()
        self._sleeper = sleeper

    @staticmethod
    def _encode_body(payload: Mapping[str, object] | None) -> bytes | None:
        if payload is None:
            return None
        try:
            body = json.dumps(
                dict(payload),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        except (
            TypeError,
            ValueError,
            OverflowError,
            RecursionError,
            UnicodeEncodeError,
        ):
            raise DiscordRestEdgeError(
                DiscordRestEdgeErrorCode.RESPONSE_INVALID,
                "Discord request payload is not canonical JSON",
            ) from None
        if not body or len(body) > _MAX_REQUEST_BODY_BYTES:
            _fail(
                DiscordRestEdgeErrorCode.RESPONSE_INVALID,
                "Discord request payload exceeds the fixed bound",
            )
        return body

    @staticmethod
    def _read_bounded(response: _ResponseLike) -> bytes:
        raw_length = response.headers.get("Content-Length")
        if raw_length is not None:
            invalid_length = False
            try:
                content_length = int(raw_length)
            except ValueError:
                invalid_length = True
                content_length = 0
            if invalid_length:
                _fail(
                    DiscordRestEdgeErrorCode.RESPONSE_INVALID,
                    "Discord Content-Length is invalid",
                )
            if content_length < 0:
                _fail(
                    DiscordRestEdgeErrorCode.RESPONSE_INVALID,
                    "Discord Content-Length is negative",
                )
            if content_length > _MAX_RESPONSE_BODY_BYTES:
                _fail(
                    DiscordRestEdgeErrorCode.RESPONSE_TOO_LARGE,
                    "Discord response exceeds the fixed bound",
                )
        body = response.read(_MAX_RESPONSE_BODY_BYTES + 1)
        if not isinstance(body, bytes):
            _fail(
                DiscordRestEdgeErrorCode.RESPONSE_INVALID,
                "Discord response body is not bytes",
            )
        if len(body) > _MAX_RESPONSE_BODY_BYTES:
            _fail(
                DiscordRestEdgeErrorCode.RESPONSE_TOO_LARGE,
                "Discord response exceeds the fixed bound",
            )
        return body

    @staticmethod
    def _rate_limit_delay(body: bytes) -> float | None:
        try:
            value = _strict_json(body)
        except DiscordRestEdgeError:
            return None
        if not isinstance(value, dict):
            return None
        delay = value.get("retry_after")
        if isinstance(delay, bool) or not isinstance(delay, (int, float)):
            return None
        try:
            numeric = float(delay)
        except (OverflowError, ValueError):
            return None
        if (
            not math.isfinite(numeric)
            or not 0 <= numeric <= _MAX_RATE_LIMIT_RETRY_SECONDS
        ):
            return None
        return numeric

    def _request(
        self,
        *,
        method: str,
        path: str,
        accepted_statuses: frozenset[int],
        payload: Mapping[str, object] | None = None,
        timeout_seconds: float | None = None,
    ) -> object:
        body = self._encode_body(payload)
        url = f"{_DISCORD_API_V10}{path}"
        total_timeout = (
            self._timeout_seconds
            if timeout_seconds is None
            else min(self._timeout_seconds, timeout_seconds)
        )
        if not 0 < total_timeout <= self._timeout_seconds:
            _fail(
                DiscordRestEdgeErrorCode.API_UNAVAILABLE,
                "Discord API deadline is exhausted",
            )
        monotonic_deadline = time.monotonic() + total_timeout
        for attempt in range(_MAX_HTTP_ATTEMPTS):
            failure: tuple[DiscordRestEdgeErrorCode, str] | None = None
            remaining_timeout = monotonic_deadline - time.monotonic()
            if remaining_timeout <= 0:
                _fail(
                    DiscordRestEdgeErrorCode.API_UNAVAILABLE,
                    "Discord API deadline is exhausted",
                )
            headers = {
                "Accept": "application/json",
                "Authorization": self._token.authorization_header(),
                "User-Agent": _USER_AGENT,
            }
            if body is not None:
                headers["Content-Type"] = "application/json"
            request = urllib.request.Request(
                url,
                data=body,
                headers=headers,
                method=method,
            )
            try:
                with self._opener.open(
                    request,
                    timeout=remaining_timeout,
                ) as response:
                    response_body = self._read_bounded(response)
                    if response.status not in accepted_statuses:
                        _fail(
                            DiscordRestEdgeErrorCode.API_REJECTED,
                            "Discord rejected a fixed API operation",
                        )
                    if not response_body:
                        _fail(
                            DiscordRestEdgeErrorCode.RESPONSE_INVALID,
                            "Discord returned an empty response",
                        )
                    return _strict_json(response_body)
            except urllib.error.HTTPError as exc:
                try:
                    response_body = self._read_bounded(exc)
                except DiscordRestEdgeError:
                    response_body = b""
                if exc.code == 429:
                    delay = self._rate_limit_delay(response_body)
                    if (
                        delay is not None
                        and delay < monotonic_deadline - time.monotonic()
                        and attempt + 1 < _MAX_HTTP_ATTEMPTS
                    ):
                        self._sleeper(delay)
                        continue
                    failure = (
                        DiscordRestEdgeErrorCode.API_RATE_LIMITED,
                        "Discord rate limit exceeded the bounded retry policy",
                    )
                else:
                    failure = (
                        DiscordRestEdgeErrorCode.API_REJECTED,
                        "Discord rejected a fixed API operation",
                    )
            except (
                urllib.error.URLError,
                http.client.HTTPException,
                TimeoutError,
                socket.timeout,
                OSError,
            ):
                failure = (
                    DiscordRestEdgeErrorCode.API_UNAVAILABLE,
                    "Discord API is unavailable",
                )
            if failure is not None:
                _fail(*failure)
        _fail(
            DiscordRestEdgeErrorCode.API_RATE_LIMITED,
            "Discord rate limit exceeded the bounded retry policy",
        )

    def current_user(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        return _json_object(
            self._request(
                method="GET",
                path="/users/@me",
                accepted_statuses=frozenset({200}),
                timeout_seconds=timeout_seconds,
            ),
            "current user",
        )

    def guild(
        self,
        guild_id: str,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        return _json_object(
            self._request(
                method="GET",
                path=f"/guilds/{guild_id}",
                accepted_statuses=frozenset({200}),
                timeout_seconds=timeout_seconds,
            ),
            "guild",
        )

    def bot_member(
        self,
        guild_id: str,
        bot_user_id: str,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        return _json_object(
            self._request(
                method="GET",
                path=f"/guilds/{guild_id}/members/{bot_user_id}",
                accepted_statuses=frozenset({200}),
                timeout_seconds=timeout_seconds,
            ),
            "bot member",
        )

    def channel(
        self,
        channel_id: str,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        return _json_object(
            self._request(
                method="GET",
                path=f"/channels/{channel_id}",
                accepted_statuses=frozenset({200}),
                timeout_seconds=timeout_seconds,
            ),
            "channel",
        )

    def create_message(
        self,
        channel_id: str,
        payload: Mapping[str, object],
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        return _json_object(
            self._request(
                method="POST",
                path=f"/channels/{channel_id}/messages",
                accepted_statuses=frozenset({200, 201}),
                payload=payload,
                timeout_seconds=timeout_seconds,
            ),
            "created message",
        )

    def edit_message(
        self,
        channel_id: str,
        message_id: str,
        payload: Mapping[str, object],
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        return _json_object(
            self._request(
                method="PATCH",
                path=f"/channels/{channel_id}/messages/{message_id}",
                accepted_statuses=frozenset({200}),
                payload=payload,
                timeout_seconds=timeout_seconds,
            ),
            "edited message",
        )

    def message(
        self,
        channel_id: str,
        message_id: str,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        return _json_object(
            self._request(
                method="GET",
                path=f"/channels/{channel_id}/messages/{message_id}",
                accepted_statuses=frozenset({200}),
                timeout_seconds=timeout_seconds,
            ),
            "message",
        )

    def create_thread(
        self,
        channel_id: str,
        payload: Mapping[str, object],
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        return _json_object(
            self._request(
                method="POST",
                path=f"/channels/{channel_id}/threads",
                accepted_statuses=frozenset({200, 201}),
                payload=payload,
                timeout_seconds=timeout_seconds,
            ),
            "created thread",
        )

    def create_forum_thread(
        self,
        channel_id: str,
        payload: Mapping[str, object],
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        return _json_object(
            self._request(
                method="POST",
                path=f"/channels/{channel_id}/threads?use_nested_fields=1",
                accepted_statuses=frozenset({200, 201}),
                payload=payload,
                timeout_seconds=timeout_seconds,
            ),
            "created forum thread",
        )


@dataclass(frozen=True)
class _TargetContext:
    target_channel: Mapping[str, Any]
    permission_channel: Mapping[str, Any]
    channel_type: int
    bot_user_id: str
    everyone_permissions: int
    bot_permissions: int


class DiscordRestEdgeAdapter:
    """Discord REST adapter implementing the runtime's two fixed Protocols."""

    def __init__(
        self,
        api: _DiscordApiV10,
        *,
        clock_ms: Callable[[], int] = lambda: int(time.time() * 1000),
    ) -> None:
        if not isinstance(api, _DiscordApiV10):
            raise TypeError("api must be the private Discord API v10 client")
        if not callable(clock_ms):
            raise TypeError("clock_ms must be callable")
        self._api = api
        self._clock_ms = clock_ms

    def close(self) -> None:
        close = getattr(self._api._opener, "close", None)
        if callable(close):
            close()

    @classmethod
    def from_credential_file(
        cls,
        credential_path: str | os.PathLike[str],
        *,
        credentials_directory: str | os.PathLike[str],
        expected_owner_uid: int,
        timeout_seconds: float = 5.0,
        _opener: _OpenerLike | None = None,
        _sleeper: Callable[[float], None] = time.sleep,
        _clock_ms: Callable[[], int] = lambda: int(time.time() * 1000),
    ) -> DiscordRestEdgeAdapter:
        """Construct from one explicit, strictly checked credential file."""

        token = _load_bot_token(
            credential_path,
            credentials_directory=credentials_directory,
            expected_owner_uid=expected_owner_uid,
        )
        return cls(
            _DiscordApiV10(
                token,
                timeout_seconds=timeout_seconds,
                opener=_opener,
                sleeper=_sleeper,
            ),
            clock_ms=_clock_ms,
        )

    @staticmethod
    def _validate_channel_binding(
        target: DiscordPublicTarget,
        channel: Mapping[str, Any],
    ) -> int:
        channel_id = _snowflake(channel.get("id"), "channel.id")
        if channel_id != target.channel_id:
            _fail(
                DiscordRestEdgeErrorCode.TARGET_MISMATCH,
                "Discord channel does not match the authorized channel",
            )
        guild_id = _snowflake(channel.get("guild_id"), "channel.guild_id")
        if guild_id != target.guild_id:
            _fail(
                DiscordRestEdgeErrorCode.TARGET_MISMATCH,
                "Discord channel does not match the authorized guild",
            )
        channel_type = _integer(channel.get("type"), "channel.type")
        if channel_type in _FORBIDDEN_CHANNEL_TYPES:
            _fail(
                DiscordRestEdgeErrorCode.TARGET_NOT_PUBLIC,
                "Discord DMs and private threads are forbidden",
            )
        if target.target_type is DiscordPublicTargetType.PUBLIC_GUILD_CHANNEL:
            allowed = _PUBLIC_MESSAGE_CHANNEL_TYPES
        elif target.target_type is DiscordPublicTargetType.PUBLIC_GUILD_FORUM:
            allowed = frozenset({_GUILD_FORUM})
        else:
            allowed = _PUBLIC_THREAD_TYPES
        if channel_type not in allowed:
            _fail(
                DiscordRestEdgeErrorCode.TARGET_MISMATCH,
                "Discord channel type does not match the authorized public target",
            )
        raw_parent = channel.get("parent_id")
        if target.target_type is DiscordPublicTargetType.PUBLIC_GUILD_THREAD:
            parent_id = _snowflake(raw_parent, "channel.parent_id")
            if parent_id != target.parent_channel_id:
                _fail(
                    DiscordRestEdgeErrorCode.TARGET_MISMATCH,
                    "Discord thread parent does not match the authorized parent",
                )
        elif raw_parent is not None and not isinstance(raw_parent, str):
            _fail(
                DiscordRestEdgeErrorCode.RESPONSE_INVALID,
                "Discord channel.parent_id is invalid",
            )
        return channel_type

    @staticmethod
    def _overwrites(channel: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        raw = channel.get("permission_overwrites", [])
        if not isinstance(raw, list) or len(raw) > _MAX_OVERWRITES:
            _fail(
                DiscordRestEdgeErrorCode.RESPONSE_INVALID,
                "Discord permission overwrites are invalid",
            )
        result: list[Mapping[str, Any]] = []
        seen: set[tuple[int, str]] = set()
        for item in raw:
            if not isinstance(item, dict):
                _fail(
                    DiscordRestEdgeErrorCode.RESPONSE_INVALID,
                    "Discord permission overwrite is not an object",
                )
            overwrite_type = _integer(item.get("type"), "overwrite.type")
            if overwrite_type not in {0, 1}:
                _fail(
                    DiscordRestEdgeErrorCode.RESPONSE_INVALID,
                    "Discord permission overwrite type is invalid",
                )
            overwrite_id = _snowflake(item.get("id"), "overwrite.id")
            identity = (overwrite_type, overwrite_id)
            if identity in seen:
                _fail(
                    DiscordRestEdgeErrorCode.RESPONSE_INVALID,
                    "Discord permission overwrite identity is duplicated",
                )
            seen.add(identity)
            _permission_bits(item.get("allow"), "overwrite.allow")
            _permission_bits(item.get("deny"), "overwrite.deny")
            result.append(item)
        return result

    @classmethod
    def _apply_everyone_overwrite(
        cls,
        permissions: int,
        *,
        guild_id: str,
        channel: Mapping[str, Any],
    ) -> int:
        for overwrite in cls._overwrites(channel):
            if overwrite["type"] == 0 and overwrite["id"] == guild_id:
                deny = _permission_bits(overwrite["deny"], "overwrite.deny")
                allow = _permission_bits(overwrite["allow"], "overwrite.allow")
                return (permissions & ~deny) | allow
        return permissions

    @classmethod
    def _apply_bot_overwrites(
        cls,
        permissions: int,
        *,
        guild_id: str,
        bot_user_id: str,
        role_ids: frozenset[str],
        channel: Mapping[str, Any],
    ) -> int:
        overwrites = cls._overwrites(channel)
        permissions = cls._apply_everyone_overwrite(
            permissions,
            guild_id=guild_id,
            channel=channel,
        )
        role_deny = 0
        role_allow = 0
        member_overwrite: Mapping[str, Any] | None = None
        for overwrite in overwrites:
            overwrite_id = str(overwrite["id"])
            overwrite_type = int(overwrite["type"])
            if overwrite_type == 0 and overwrite_id in role_ids:
                role_deny |= _permission_bits(overwrite["deny"], "overwrite.deny")
                role_allow |= _permission_bits(overwrite["allow"], "overwrite.allow")
            elif overwrite_type == 1 and overwrite_id == bot_user_id:
                member_overwrite = overwrite
        permissions = (permissions & ~role_deny) | role_allow
        if member_overwrite is not None:
            permissions = (
                permissions
                & ~_permission_bits(member_overwrite["deny"], "overwrite.deny")
            ) | _permission_bits(member_overwrite["allow"], "overwrite.allow")
        return permissions

    def _target_context(
        self,
        target: DiscordPublicTarget,
        *,
        deadline_unix_ms: int | None = None,
    ) -> _TargetContext:
        if not isinstance(target, DiscordPublicTarget):
            raise TypeError("target must be DiscordPublicTarget")

        def remaining_timeout() -> float | None:
            if deadline_unix_ms is None:
                return None
            return self._remaining_deadline_timeout(deadline_unix_ms)

        current_user = self._api.current_user(
            timeout_seconds=remaining_timeout(),
        )
        bot_user_id = _snowflake(current_user.get("id"), "current_user.id")
        if current_user.get("bot") is not True:
            _fail(
                DiscordRestEdgeErrorCode.BOT_IDENTITY_MISMATCH,
                "Discord credential is not bound to a bot user",
            )
        guild = self._api.guild(
            target.guild_id,
            timeout_seconds=remaining_timeout(),
        )
        if _snowflake(guild.get("id"), "guild.id") != target.guild_id:
            _fail(
                DiscordRestEdgeErrorCode.TARGET_MISMATCH,
                "Discord guild does not match the authorized guild",
            )
        owner_id = _snowflake(guild.get("owner_id"), "guild.owner_id")
        roles = guild.get("roles")
        if not isinstance(roles, list) or not 1 <= len(roles) <= _MAX_ROLES:
            _fail(
                DiscordRestEdgeErrorCode.RESPONSE_INVALID,
                "Discord guild roles are invalid",
            )
        role_permissions: dict[str, int] = {}
        for role in roles:
            if not isinstance(role, dict):
                _fail(
                    DiscordRestEdgeErrorCode.RESPONSE_INVALID,
                    "Discord guild role is not an object",
                )
            role_id = _snowflake(role.get("id"), "role.id")
            if role_id in role_permissions:
                _fail(
                    DiscordRestEdgeErrorCode.RESPONSE_INVALID,
                    "Discord guild role identity is duplicated",
                )
            role_permissions[role_id] = _permission_bits(
                role.get("permissions"),
                "role.permissions",
            )
        if target.guild_id not in role_permissions:
            _fail(
                DiscordRestEdgeErrorCode.RESPONSE_INVALID,
                "Discord @everyone role is missing",
            )

        member = self._api.bot_member(
            target.guild_id,
            bot_user_id,
            timeout_seconds=remaining_timeout(),
        )
        member_user = _json_object(member.get("user"), "member.user")
        if _snowflake(member_user.get("id"), "member.user.id") != bot_user_id:
            _fail(
                DiscordRestEdgeErrorCode.BOT_IDENTITY_MISMATCH,
                "Discord guild member does not match the credential bot",
            )
        raw_member_roles = member.get("roles")
        if not isinstance(raw_member_roles, list) or len(raw_member_roles) > _MAX_ROLES:
            _fail(
                DiscordRestEdgeErrorCode.RESPONSE_INVALID,
                "Discord bot role membership is invalid",
            )
        member_role_ids: set[str] = set()
        for role_id_value in raw_member_roles:
            role_id = _snowflake(role_id_value, "member.role_id")
            if role_id in member_role_ids or role_id == target.guild_id:
                _fail(
                    DiscordRestEdgeErrorCode.RESPONSE_INVALID,
                    "Discord bot role membership is duplicated or invalid",
                )
            if role_id not in role_permissions:
                _fail(
                    DiscordRestEdgeErrorCode.RESPONSE_INVALID,
                    "Discord bot references an unknown guild role",
                )
            member_role_ids.add(role_id)

        # Fetch the permission-bearing channel state last.  The proof timestamp
        # is taken only after these reads so a slow identity/role lookup cannot
        # make an old channel snapshot appear freshly observed.
        target_channel = self._api.channel(
            target.channel_id,
            timeout_seconds=remaining_timeout(),
        )
        channel_type = self._validate_channel_binding(target, target_channel)
        permission_channel = target_channel
        if target.target_type is DiscordPublicTargetType.PUBLIC_GUILD_THREAD:
            assert target.parent_channel_id is not None
            permission_channel = self._api.channel(
                target.parent_channel_id,
                timeout_seconds=remaining_timeout(),
            )
            parent_id = _snowflake(permission_channel.get("id"), "parent.id")
            parent_guild = _snowflake(
                permission_channel.get("guild_id"),
                "parent.guild_id",
            )
            parent_type = _integer(permission_channel.get("type"), "parent.type")
            if (
                parent_id != target.parent_channel_id
                or parent_guild != target.guild_id
                or parent_type not in _PUBLIC_THREAD_PARENT_TYPES
            ):
                _fail(
                    DiscordRestEdgeErrorCode.TARGET_MISMATCH,
                    "Discord thread parent binding is invalid",
                )

        everyone_base = role_permissions[target.guild_id]
        if everyone_base & _ADMINISTRATOR:
            everyone_permissions = _ALL_PERMISSION_BITS
        else:
            everyone_permissions = self._apply_everyone_overwrite(
                everyone_base,
                guild_id=target.guild_id,
                channel=permission_channel,
            )
        bot_base = everyone_base
        for role_id in member_role_ids:
            bot_base |= role_permissions[role_id]
        if bot_user_id == owner_id or bot_base & _ADMINISTRATOR:
            bot_permissions = _ALL_PERMISSION_BITS
        else:
            bot_permissions = self._apply_bot_overwrites(
                bot_base,
                guild_id=target.guild_id,
                bot_user_id=bot_user_id,
                role_ids=frozenset(member_role_ids),
                channel=permission_channel,
            )
        return _TargetContext(
            target_channel=target_channel,
            permission_channel=permission_channel,
            channel_type=channel_type,
            bot_user_id=bot_user_id,
            everyone_permissions=everyone_permissions,
            bot_permissions=bot_permissions,
        )

    @staticmethod
    def _required_permissions(
        operation: DiscordEdgeOperation,
        target: DiscordPublicTarget,
        channel_type: int,
        *,
        has_initial_message: bool,
    ) -> int:
        base = _VIEW_CHANNEL | _READ_MESSAGE_HISTORY
        if operation is DiscordEdgeOperation.PUBLIC_MESSAGE_EDIT:
            return base
        if operation is DiscordEdgeOperation.PUBLIC_MESSAGE_SEND:
            if target.target_type is DiscordPublicTargetType.PUBLIC_GUILD_THREAD:
                return base | _SEND_MESSAGES_IN_THREADS
            return base | _SEND_MESSAGES
        if channel_type == _GUILD_FORUM:
            # Discord names SEND_MESSAGES "Create Posts" for forum channels;
            # the forum-thread endpoint requires that bit for the initial post.
            return _VIEW_CHANNEL | _SEND_MESSAGES | _READ_MESSAGE_HISTORY
        required = _VIEW_CHANNEL | _CREATE_PUBLIC_THREADS | _SEND_MESSAGES
        if has_initial_message:
            required |= _READ_MESSAGE_HISTORY
        return required

    def _prove(
        self,
        operation: DiscordEdgeOperation,
        target: DiscordPublicTarget,
        *,
        now_unix_ms: int,
        deadline_unix_ms: int,
        has_initial_message: bool = False,
    ) -> DiscordLivePublicTargetProof:
        if isinstance(now_unix_ms, bool) or not isinstance(now_unix_ms, int):
            raise TypeError("now_unix_ms must be an integer")
        context = self._target_context(
            target,
            deadline_unix_ms=deadline_unix_ms,
        )
        required = self._required_permissions(
            operation,
            target,
            context.channel_type,
            has_initial_message=has_initial_message,
        )
        bot_can_view = bool(context.bot_permissions & _VIEW_CHANNEL)
        has_required = context.bot_permissions & required == required
        if (
            operation is DiscordEdgeOperation.PUBLIC_THREAD_CREATE
            and context.channel_type not in {_GUILD_TEXT, _GUILD_FORUM}
        ):
            # The fixed operation starts a thread without a source message.
            # Discord announcement threads require the separate start-from-
            # message endpoint, which is intentionally outside this surface.
            has_required = False
        if (
            operation is DiscordEdgeOperation.PUBLIC_MESSAGE_SEND
            and target.target_type is DiscordPublicTargetType.PUBLIC_GUILD_THREAD
        ):
            metadata = context.target_channel.get("thread_metadata")
            if not isinstance(metadata, dict):
                _fail(
                    DiscordRestEdgeErrorCode.RESPONSE_INVALID,
                    "Discord public thread metadata is invalid",
                )
            archived = metadata.get("archived")
            if not isinstance(archived, bool):
                _fail(
                    DiscordRestEdgeErrorCode.RESPONSE_INVALID,
                    "Discord public thread archived state is invalid",
                )
            locked = metadata.get("locked")
            if locked is not None and not isinstance(locked, bool):
                _fail(
                    DiscordRestEdgeErrorCode.RESPONSE_INVALID,
                    "Discord public thread locked state is invalid",
                )
            # Discord automatically reopens an unlocked archived public thread
            # when an authorized member sends.  A locked thread needs a
            # separate management mutation, intentionally outside this edge.
            if locked is True:
                has_required = False
        return DiscordLivePublicTargetProof(
            operation=operation,
            target=target,
            bot_user_id=context.bot_user_id,
            observed_at_unix_ms=self._clock_ms(),
            publicly_viewable=bool(context.everyone_permissions & _VIEW_CHANNEL),
            bot_can_view=bot_can_view,
            bot_has_required_permission=has_required,
        )

    def prove_public_message_send(
        self,
        target: DiscordPublicTarget,
        *,
        deadline_unix_ms: int,
        now_unix_ms: int,
    ) -> DiscordLivePublicTargetProof:
        return self._prove(
            DiscordEdgeOperation.PUBLIC_MESSAGE_SEND,
            target,
            now_unix_ms=now_unix_ms,
            deadline_unix_ms=deadline_unix_ms,
        )

    def prove_public_message_edit(
        self,
        target: DiscordPublicTarget,
        *,
        deadline_unix_ms: int,
        now_unix_ms: int,
    ) -> DiscordLivePublicTargetProof:
        return self._prove(
            DiscordEdgeOperation.PUBLIC_MESSAGE_EDIT,
            target,
            now_unix_ms=now_unix_ms,
            deadline_unix_ms=deadline_unix_ms,
        )

    def prove_public_thread_create(
        self,
        target: DiscordPublicTarget,
        *,
        has_initial_message: bool,
        deadline_unix_ms: int,
        now_unix_ms: int,
    ) -> DiscordLivePublicTargetProof:
        if not isinstance(has_initial_message, bool):
            raise TypeError("has_initial_message must be a boolean")
        return self._prove(
            DiscordEdgeOperation.PUBLIC_THREAD_CREATE,
            target,
            now_unix_ms=now_unix_ms,
            deadline_unix_ms=deadline_unix_ms,
            has_initial_message=has_initial_message,
        )

    def prove_public_readback(
        self,
        operation: DiscordEdgeOperation,
        target: DiscordPublicTarget,
        *,
        require_message_history: bool,
        deadline_unix_ms: int,
        now_unix_ms: int,
    ) -> DiscordLivePublicTargetProof:
        if not isinstance(operation, DiscordEdgeOperation):
            raise TypeError("operation must be a DiscordEdgeOperation")
        if not isinstance(require_message_history, bool):
            raise TypeError("require_message_history must be a boolean")
        if isinstance(now_unix_ms, bool) or not isinstance(now_unix_ms, int):
            raise TypeError("now_unix_ms must be an integer")
        context = self._target_context(
            target,
            deadline_unix_ms=deadline_unix_ms,
        )
        required = _VIEW_CHANNEL
        if require_message_history:
            required |= _READ_MESSAGE_HISTORY
        return DiscordLivePublicTargetProof(
            operation=operation,
            target=target,
            bot_user_id=context.bot_user_id,
            observed_at_unix_ms=self._clock_ms(),
            publicly_viewable=bool(context.everyone_permissions & _VIEW_CHANNEL),
            bot_can_view=bool(context.bot_permissions & _VIEW_CHANNEL),
            bot_has_required_permission=(
                context.bot_permissions & required == required
            ),
        )

    def _authorize_live_mutation(
        self,
        operation: DiscordEdgeOperation,
        target: DiscordPublicTarget,
        *,
        has_initial_message: bool = False,
        deadline_unix_ms: int,
    ) -> DiscordLivePublicTargetProof:
        """Re-prove the exact public target immediately at the mutation edge."""

        self._remaining_deadline_timeout(deadline_unix_ms)
        proof = self._prove(
            operation,
            target,
            now_unix_ms=self._clock_ms(),
            deadline_unix_ms=deadline_unix_ms,
            has_initial_message=has_initial_message,
        )
        if not proof.publicly_viewable:
            _fail(
                DiscordRestEdgeErrorCode.TARGET_NOT_PUBLIC,
                "Discord target is not currently visible to @everyone",
            )
        if not proof.bot_can_view or not proof.bot_has_required_permission:
            _fail(
                DiscordRestEdgeErrorCode.BOT_PERMISSION_REVOKED,
                "Discord bot lacks the current operation-specific permissions",
            )
        return proof

    def _remaining_deadline_timeout(self, deadline_unix_ms: int) -> float:
        if isinstance(deadline_unix_ms, bool) or not isinstance(
            deadline_unix_ms,
            int,
        ):
            raise TypeError("deadline_unix_ms must be an integer")
        remaining = (deadline_unix_ms - self._clock_ms()) / 1_000
        if remaining <= 0:
            _fail(
                DiscordRestEdgeErrorCode.REQUEST_DEADLINE_EXPIRED,
                "Discord mutation deadline expired before the API operation",
            )
        return min(self._api._timeout_seconds, remaining)

    def _authorize_live_readback(
        self,
        target: DiscordPublicTarget,
        *,
        require_message_history: bool = True,
    ) -> str:
        """Require current public visibility and only the permissions needed to read."""

        context = self._target_context(target)
        if not context.everyone_permissions & _VIEW_CHANNEL:
            _fail(
                DiscordRestEdgeErrorCode.TARGET_NOT_PUBLIC,
                "Discord target is not currently visible to @everyone",
            )
        required = _VIEW_CHANNEL
        if require_message_history:
            required |= _READ_MESSAGE_HISTORY
        if context.bot_permissions & required != required:
            _fail(
                DiscordRestEdgeErrorCode.BOT_PERMISSION_REVOKED,
                "Discord bot lacks the current readback permissions",
            )
        return context.bot_user_id

    @staticmethod
    def _validate_message(
        message: Mapping[str, Any],
        *,
        target: DiscordPublicTarget,
        message_id: str | None,
        bot_user_id: str,
        expected_content: str | None,
        expected_reply_to_message_id: str | None | object = _IGNORE_REPLY_BINDING,
    ) -> tuple[str, str, str, str | None]:
        observed_id = _snowflake(message.get("id"), "message.id")
        if message_id is not None and observed_id != message_id:
            _fail(
                DiscordRestEdgeErrorCode.MUTATION_BINDING_MISMATCH,
                "Discord message identity does not match the fixed operation",
            )
        if _snowflake(message.get("channel_id"), "message.channel_id") != target.channel_id:
            _fail(
                DiscordRestEdgeErrorCode.MUTATION_BINDING_MISMATCH,
                "Discord message channel does not match the fixed operation",
            )
        if _snowflake(message.get("guild_id"), "message.guild_id") != target.guild_id:
            _fail(
                DiscordRestEdgeErrorCode.MUTATION_BINDING_MISMATCH,
                "Discord message guild does not match the fixed operation",
            )
        author = _json_object(message.get("author"), "message.author")
        author_id = _snowflake(author.get("id"), "message.author.id")
        if author_id != bot_user_id or author.get("bot") is not True:
            _fail(
                DiscordRestEdgeErrorCode.BOT_IDENTITY_MISMATCH,
                "Discord message is not authored by the credential bot",
            )
        content = message.get("content")
        if not isinstance(content, str):
            _fail(
                DiscordRestEdgeErrorCode.RESPONSE_INVALID,
                "Discord message content is invalid",
            )
        if expected_content is not None and content != expected_content:
            _fail(
                DiscordRestEdgeErrorCode.MUTATION_BINDING_MISMATCH,
                "Discord message content does not match the fixed operation",
            )
        raw_reference = message.get("message_reference")
        reply_to_message_id: str | None = None
        if raw_reference is not None:
            reference = _json_object(raw_reference, "message.message_reference")
            reference_type = reference.get("type", 0)
            if _integer(reference_type, "message_reference.type") != 0:
                _fail(
                    DiscordRestEdgeErrorCode.MUTATION_BINDING_MISMATCH,
                    "Discord message reference is not a reply",
                )
            reply_to_message_id = _snowflake(
                reference.get("message_id"),
                "message_reference.message_id",
            )
            if _snowflake(
                reference.get("channel_id"),
                "message_reference.channel_id",
            ) != target.channel_id:
                _fail(
                    DiscordRestEdgeErrorCode.MUTATION_BINDING_MISMATCH,
                    "Discord reply reference channel does not match",
                )
            reference_guild_id = reference.get("guild_id")
            if reference_guild_id is not None and _snowflake(
                reference_guild_id,
                "message_reference.guild_id",
            ) != target.guild_id:
                _fail(
                    DiscordRestEdgeErrorCode.MUTATION_BINDING_MISMATCH,
                    "Discord reply reference guild does not match",
                )
        if (
            expected_reply_to_message_id is not _IGNORE_REPLY_BINDING
            and reply_to_message_id != expected_reply_to_message_id
        ):
            _fail(
                DiscordRestEdgeErrorCode.MUTATION_BINDING_MISMATCH,
                "Discord reply reference does not match the fixed operation",
            )
        return observed_id, author_id, content, reply_to_message_id

    def send_public_message(
        self,
        target: DiscordPublicTarget,
        *,
        content: str,
        reply_to_message_id: str | None,
        deadline_unix_ms: int,
    ) -> DiscordMutationAccepted:
        proof = self._authorize_live_mutation(
            DiscordEdgeOperation.PUBLIC_MESSAGE_SEND,
            target,
            deadline_unix_ms=deadline_unix_ms,
        )
        bot_user_id = proof.bot_user_id
        if reply_to_message_id is not None:
            reply_to_message_id = _snowflake(
                reply_to_message_id,
                "reply_to_message_id",
            )
        payload: dict[str, object] = {
            "allowed_mentions": dict(_SAFE_ALLOWED_MENTIONS),
            "content": content,
        }
        if reply_to_message_id is not None:
            payload["message_reference"] = {
                "channel_id": target.channel_id,
                "fail_if_not_exists": True,
                "guild_id": target.guild_id,
                "message_id": reply_to_message_id,
            }
        message = self._api.create_message(
            target.channel_id,
            payload,
            timeout_seconds=self._remaining_deadline_timeout(deadline_unix_ms),
        )
        message_id, _, _, _ = self._validate_message(
            message,
            target=target,
            message_id=None,
            bot_user_id=bot_user_id,
            expected_content=content,
            expected_reply_to_message_id=reply_to_message_id,
        )
        return DiscordMutationAccepted(
            operation=DiscordEdgeOperation.PUBLIC_MESSAGE_SEND,
            target=target,
            discord_object_id=message_id,
            bot_user_id=bot_user_id,
        )

    def edit_public_message(
        self,
        target: DiscordPublicTarget,
        *,
        message_id: str,
        content: str,
        deadline_unix_ms: int,
    ) -> DiscordMutationAccepted:
        proof = self._authorize_live_mutation(
            DiscordEdgeOperation.PUBLIC_MESSAGE_EDIT,
            target,
            deadline_unix_ms=deadline_unix_ms,
        )
        bot_user_id = proof.bot_user_id
        message_id = _snowflake(message_id, "message_id")
        existing = self._api.message(
            target.channel_id,
            message_id,
            timeout_seconds=self._remaining_deadline_timeout(deadline_unix_ms),
        )
        self._validate_message(
            existing,
            target=target,
            message_id=message_id,
            bot_user_id=bot_user_id,
            expected_content=None,
        )
        message = self._api.edit_message(
            target.channel_id,
            message_id,
            {
                "allowed_mentions": dict(_SAFE_ALLOWED_MENTIONS),
                "content": content,
            },
            timeout_seconds=self._remaining_deadline_timeout(deadline_unix_ms),
        )
        self._validate_message(
            message,
            target=target,
            message_id=message_id,
            bot_user_id=bot_user_id,
            expected_content=content,
        )
        return DiscordMutationAccepted(
            operation=DiscordEdgeOperation.PUBLIC_MESSAGE_EDIT,
            target=target,
            discord_object_id=message_id,
            bot_user_id=bot_user_id,
        )

    @staticmethod
    def _validate_created_thread(
        thread: Mapping[str, Any],
        *,
        target: DiscordPublicTarget,
        name: str,
        bot_user_id: str,
        requested_archive: int | None,
    ) -> tuple[str, int, str | None]:
        thread_id = _snowflake(thread.get("id"), "thread.id")
        if _snowflake(thread.get("guild_id"), "thread.guild_id") != target.guild_id:
            _fail(
                DiscordRestEdgeErrorCode.MUTATION_BINDING_MISMATCH,
                "Discord created thread guild does not match",
            )
        if _snowflake(thread.get("parent_id"), "thread.parent_id") != target.channel_id:
            _fail(
                DiscordRestEdgeErrorCode.MUTATION_BINDING_MISMATCH,
                "Discord created thread parent does not match",
            )
        if _integer(thread.get("type"), "thread.type") != _PUBLIC_THREAD:
            _fail(
                DiscordRestEdgeErrorCode.TARGET_NOT_PUBLIC,
                "Discord did not create an explicit public guild thread",
            )
        if thread.get("name") != name:
            _fail(
                DiscordRestEdgeErrorCode.MUTATION_BINDING_MISMATCH,
                "Discord created thread name does not match",
            )
        owner_id = _snowflake(thread.get("owner_id"), "thread.owner_id")
        if owner_id != bot_user_id:
            _fail(
                DiscordRestEdgeErrorCode.BOT_IDENTITY_MISMATCH,
                "Discord created thread is not owned by the credential bot",
            )
        metadata = _json_object(thread.get("thread_metadata"), "thread metadata")
        if metadata.get("archived") is not False:
            _fail(
                DiscordRestEdgeErrorCode.MUTATION_BINDING_MISMATCH,
                "Discord created thread is unexpectedly archived",
            )
        locked = metadata.get("locked")
        if locked is not None and not isinstance(locked, bool):
            _fail(
                DiscordRestEdgeErrorCode.RESPONSE_INVALID,
                "Discord created thread locked state is invalid",
            )
        if locked is True:
            _fail(
                DiscordRestEdgeErrorCode.MUTATION_BINDING_MISMATCH,
                "Discord created thread is unexpectedly locked",
            )
        archive = _integer(
            metadata.get("auto_archive_duration"),
            "thread_metadata.auto_archive_duration",
        )
        if archive not in {60, 1_440, 4_320, 10_080}:
            _fail(
                DiscordRestEdgeErrorCode.RESPONSE_INVALID,
                "Discord thread auto-archive value is invalid",
            )
        if requested_archive is not None and archive != requested_archive:
            _fail(
                DiscordRestEdgeErrorCode.MUTATION_BINDING_MISMATCH,
                "Discord thread auto-archive value does not match",
            )
        embedded = thread.get("message")
        message_id: str | None = None
        if embedded is not None:
            message = _json_object(embedded, "thread.message")
            message_id = _snowflake(message.get("id"), "thread.message.id")
        return thread_id, archive, message_id

    def create_public_thread(
        self,
        target: DiscordPublicTarget,
        *,
        name: str,
        initial_message: str | None,
        auto_archive_minutes: int | None,
        deadline_unix_ms: int,
    ) -> DiscordMutationAccepted:
        if (
            target.target_type
            is DiscordPublicTargetType.PUBLIC_GUILD_CHANNEL
            and initial_message
        ):
            _fail(
                DiscordRestEdgeErrorCode.MUTATION_BINDING_MISMATCH,
                "Discord text thread initial content requires a separate receipted send",
            )
        proof = self._authorize_live_mutation(
            DiscordEdgeOperation.PUBLIC_THREAD_CREATE,
            target,
            has_initial_message=bool(initial_message),
            deadline_unix_ms=deadline_unix_ms,
        )
        bot_user_id = proof.bot_user_id
        channel_type = (
            _GUILD_FORUM
            if target.target_type is DiscordPublicTargetType.PUBLIC_GUILD_FORUM
            else _GUILD_TEXT
        )
        payload: dict[str, object] = {"name": name}
        if auto_archive_minutes is not None:
            payload["auto_archive_duration"] = auto_archive_minutes
        if channel_type == _GUILD_FORUM:
            if not initial_message:
                _fail(
                    DiscordRestEdgeErrorCode.MUTATION_BINDING_MISMATCH,
                    "Discord forum thread requires a non-empty initial message",
                )
            payload["message"] = {
                "allowed_mentions": dict(_SAFE_ALLOWED_MENTIONS),
                "content": initial_message,
            }
            payload["type"] = _PUBLIC_THREAD
            thread = self._api.create_forum_thread(
                target.channel_id,
                payload,
                timeout_seconds=self._remaining_deadline_timeout(
                    deadline_unix_ms
                ),
            )
        else:
            payload["type"] = _PUBLIC_THREAD
            thread = self._api.create_thread(
                target.channel_id,
                payload,
                timeout_seconds=self._remaining_deadline_timeout(
                    deadline_unix_ms
                ),
            )
        thread_id, _, embedded_message_id = self._validate_created_thread(
            thread,
            target=target,
            name=name,
            bot_user_id=bot_user_id,
            requested_archive=auto_archive_minutes,
        )
        if initial_message and channel_type == _GUILD_FORUM:
            if embedded_message_id != thread_id:
                _fail(
                    DiscordRestEdgeErrorCode.MUTATION_BINDING_MISMATCH,
                    "Discord forum starter message is not bound to the thread id",
                )
            thread_target = DiscordPublicTarget(
                DiscordPublicTargetType.PUBLIC_GUILD_THREAD,
                target.guild_id,
                thread_id,
                target.channel_id,
            )
            embedded = _json_object(thread.get("message"), "thread.message")
            self._validate_message(
                embedded,
                target=thread_target,
                message_id=thread_id,
                bot_user_id=bot_user_id,
                expected_content=initial_message,
                expected_reply_to_message_id=None,
            )
        return DiscordMutationAccepted(
            operation=DiscordEdgeOperation.PUBLIC_THREAD_CREATE,
            target=target,
            discord_object_id=thread_id,
            bot_user_id=bot_user_id,
        )

    def read_public_message(
        self,
        target: DiscordPublicTarget,
        *,
        operation: DiscordEdgeOperation,
        message_id: str,
        expected_reply_to_message_id: str | None,
    ) -> DiscordMutationReadback:
        if not isinstance(operation, DiscordEdgeOperation) or operation not in {
            DiscordEdgeOperation.PUBLIC_MESSAGE_SEND,
            DiscordEdgeOperation.PUBLIC_MESSAGE_EDIT,
        }:
            raise TypeError("message readback requires a typed message operation")
        bot_user_id = self._authorize_live_readback(target)
        message_id = _snowflake(message_id, "message_id")
        message = self._api.message(target.channel_id, message_id)
        observed_id, author_id, content, reply_to_message_id = self._validate_message(
            message,
            target=target,
            message_id=message_id,
            bot_user_id=bot_user_id,
            expected_content=None,
            expected_reply_to_message_id=(
                expected_reply_to_message_id
                if operation is DiscordEdgeOperation.PUBLIC_MESSAGE_SEND
                else _IGNORE_REPLY_BINDING
            ),
        )
        return DiscordMutationReadback(
            operation=operation,
            target=target,
            discord_object_id=observed_id,
            author_user_id=author_id,
            content=content,
            reply_to_message_id=reply_to_message_id,
        )

    def read_created_public_thread(
        self,
        target: DiscordPublicTarget,
        *,
        thread_id: str,
        expected_content: str,
    ) -> DiscordMutationReadback:
        if not isinstance(expected_content, str):
            raise TypeError("expected_content must be a string")
        bot_user_id = self._authorize_live_readback(
            target,
            require_message_history=bool(expected_content),
        )
        thread_id = _snowflake(thread_id, "thread_id")
        thread = self._api.channel(thread_id)
        thread_target = DiscordPublicTarget(
            DiscordPublicTargetType.PUBLIC_GUILD_THREAD,
            target.guild_id,
            thread_id,
            target.channel_id,
        )
        self._validate_channel_binding(thread_target, thread)
        name = thread.get("name")
        if not isinstance(name, str) or not name:
            _fail(
                DiscordRestEdgeErrorCode.RESPONSE_INVALID,
                "Discord thread name is invalid",
            )
        owner_id = _snowflake(thread.get("owner_id"), "thread.owner_id")
        if owner_id != bot_user_id:
            _fail(
                DiscordRestEdgeErrorCode.BOT_IDENTITY_MISMATCH,
                "Discord thread is not owned by the credential bot",
            )
        metadata = _json_object(thread.get("thread_metadata"), "thread metadata")
        archived = metadata.get("archived")
        if not isinstance(archived, bool):
            _fail(
                DiscordRestEdgeErrorCode.RESPONSE_INVALID,
                "Discord thread archived state is invalid",
            )
        locked = metadata.get("locked")
        if locked is not None and not isinstance(locked, bool):
            _fail(
                DiscordRestEdgeErrorCode.RESPONSE_INVALID,
                "Discord thread locked state is invalid",
            )
        archive = _integer(
            metadata.get("auto_archive_duration"),
            "thread_metadata.auto_archive_duration",
        )
        if archive not in {60, 1_440, 4_320, 10_080}:
            _fail(
                DiscordRestEdgeErrorCode.RESPONSE_INVALID,
                "Discord thread auto-archive value is invalid",
            )

        if not expected_content:
            author_id = owner_id
            content = ""
        else:
            if target.target_type is DiscordPublicTargetType.PUBLIC_GUILD_FORUM:
                message_target = thread_target
                message_channel_id = thread_id
            else:
                message_target = target
                message_channel_id = target.channel_id
            starter = self._api.message(message_channel_id, thread_id)
            _, author_id, content, _ = self._validate_message(
                starter,
                target=message_target,
                message_id=thread_id,
                bot_user_id=bot_user_id,
                expected_content=expected_content,
                expected_reply_to_message_id=None,
            )
        return DiscordMutationReadback(
            operation=DiscordEdgeOperation.PUBLIC_THREAD_CREATE,
            target=target,
            discord_object_id=thread_id,
            author_user_id=author_id,
            content=content,
            thread=DiscordEdgeThreadReadback(
                target=thread_target,
                name=name,
                auto_archive_minutes=archive,
            ),
        )


__all__ = [
    "DiscordRestEdgeAdapter",
    "DiscordRestEdgeError",
    "DiscordRestEdgeErrorCode",
]
