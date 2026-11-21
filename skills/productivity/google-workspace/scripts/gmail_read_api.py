#!/usr/bin/env python3
"""Strict, account-scoped, read-only Gmail adapter for Hermes skills.

Named accounts resolve only below ``HERMES_HOME/google-accounts``.  Every
operation verifies the Gmail profile against an independent owner-only
deployment mapping before returning mailbox data.  Token metadata is never an
identity authority, and the CLI has no mutation commands.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import re
import stat
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Any, Callable, Iterable

# Ensure sibling modules are importable when the script runs standalone.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from _hermes_home import get_hermes_home


SCHEMA_VERSION = 1
GMAIL_READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
DRIVE_READONLY_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
APPROVED_READONLY_SCOPES = frozenset({GMAIL_READONLY_SCOPE, DRIVE_READONLY_SCOPE})
ALIAS_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,512}$")
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
RETRYABLE_HTTP = frozenset({429, 500, 502, 503, 504})
MAX_TOKEN_FILE_BYTES = 64 * 1024
MAX_QUERY_CHARS = 4096
MAX_PAGE_TOKEN_CHARS = 4096
HARD_MAX_RESULTS = 100
HARD_MAX_BODY_BYTES = 25 * 1024 * 1024
HARD_MAX_MESSAGE_BYTES = 100 * 1024 * 1024
HARD_MAX_ATTACHMENT_BYTES = 100 * 1024 * 1024
HARD_MAX_API_CALLS = 512
HARD_MAX_DEADLINE_S = 600.0
HARD_MAX_ATTEMPTS = 4
HARD_MAX_MIME_PARTS = 10_000
HARD_MAX_ATTACHMENTS = 1_000


class AdapterError(RuntimeError):
    """A typed failure whose code is safe to expose in a body-free receipt."""

    def __init__(
        self,
        code: str,
        *,
        exit_code: int = 4,
        api_calls: int = 0,
        identity_verified: bool = False,
        request_shape: dict[str, Any] | None = None,
        started_at: str | None = None,
    ):
        self.code = code
        self.exit_code = exit_code
        self.api_calls = api_calls
        self.identity_verified = identity_verified
        self.request_shape = request_shape
        self.started_at = started_at
        super().__init__(code)

    def bind_runtime(
        self,
        *,
        api_calls: int,
        identity_verified: bool,
        request_shape: dict[str, Any],
        started_at: str,
    ) -> None:
        self.api_calls = api_calls
        self.identity_verified = identity_verified
        self.request_shape = request_shape
        self.started_at = started_at


@dataclass(frozen=True)
class AccountContext:
    alias: str
    expected_email: str
    token_path: Path
    token_payload: dict[str, Any]
    scopes: tuple[str, ...]

    @property
    def source_id(self) -> str:
        return f"gmail:{self.alias}"


@dataclass(frozen=True)
class Limits:
    deadline_s: float = 60.0
    max_api_calls: int = 128
    attempts: int = 3

    def __post_init__(self) -> None:
        _bounded_number("deadline", self.deadline_s, 0.1, HARD_MAX_DEADLINE_S)
        _bounded_number("max_api_calls", self.max_api_calls, 1, HARD_MAX_API_CALLS)
        _bounded_number("attempts", self.attempts, 1, HARD_MAX_ATTEMPTS)


def _utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _bounded_number(name: str, value: float, low: float, high: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not low <= value <= high
    ):
        raise AdapterError(f"INVALID_{name.upper()}", exit_code=2)


def _bounded_int(low: int, high: int) -> Callable[[str], int]:
    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be an integer") from exc
        if not low <= parsed <= high:
            raise argparse.ArgumentTypeError(f"must be between {low} and {high}")
        return parsed

    return parse


def _bounded_float(low: float, high: float) -> Callable[[str], float]:
    def parse(value: str) -> float:
        try:
            parsed = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be a number") from exc
        if not low <= parsed <= high:
            raise argparse.ArgumentTypeError(f"must be between {low} and {high}")
        return parsed

    return parse


def _validate_alias(value: str) -> str:
    alias = str(value or "")
    if not ALIAS_RE.fullmatch(alias):
        raise AdapterError("INVALID_ACCOUNT_ALIAS", exit_code=2)
    return alias


def _validate_id(value: str, kind: str) -> str:
    if not isinstance(value, str) or not ID_RE.fullmatch(value):
        raise AdapterError(f"INVALID_{kind.upper()}_ID", exit_code=2)
    return value


def _validate_query(value: str) -> str:
    query = str(value or "")
    if not query or len(query) > MAX_QUERY_CHARS or "\x00" in query:
        raise AdapterError("INVALID_QUERY", exit_code=2)
    return query


def _validate_page_token(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    if (
        not isinstance(value, str)
        or len(value) > MAX_PAGE_TOKEN_CHARS
        or any(ord(char) < 0x20 for char in value)
    ):
        raise AdapterError("INVALID_PAGE_TOKEN", exit_code=2)
    return value


def _is_private_file(path: Path) -> bool:
    info = path.lstat()
    getuid = getattr(os, "getuid", None)
    owner_matches = getuid is None or info.st_uid == getuid()
    return (
        stat.S_ISREG(info.st_mode)
        and not path.is_symlink()
        and stat.S_IMODE(info.st_mode) == 0o600
        and owner_matches
    )


def _read_private_json(path: Path, *, error_code: str) -> dict[str, Any]:
    if not _is_private_file(path):
        raise AdapterError(error_code, exit_code=3)
    try:
        if path.stat().st_size > MAX_TOKEN_FILE_BYTES:
            raise AdapterError(error_code, exit_code=3)
        payload = json.loads(path.read_text(encoding="utf-8"))
    except AdapterError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise AdapterError(error_code, exit_code=3) from None
    if not isinstance(payload, dict):
        raise AdapterError(error_code, exit_code=3)
    return payload


def _normalize_account_mapping(payload: dict[str, Any]) -> dict[str, str]:
    """Normalize the deployed flat map and the versioned nested contract."""

    keys = set(payload)
    accounts = payload.get("accounts")
    if not isinstance(accounts, dict) or not accounts:
        raise AdapterError("INVALID_ACCOUNT_MAPPING", exit_code=3)
    if keys == {"accounts"}:
        if not all(isinstance(value, str) for value in accounts.values()):
            raise AdapterError("INVALID_ACCOUNT_MAPPING", exit_code=3)
        raw_accounts = accounts
    elif keys == {"schema_version", "accounts"} and payload.get("schema_version") == 1:
        if not all(
            isinstance(value, dict) and set(value) == {"email"}
            for value in accounts.values()
        ):
            raise AdapterError("INVALID_ACCOUNT_MAPPING", exit_code=3)
        raw_accounts = {alias: value["email"] for alias, value in accounts.items()}
    else:
        raise AdapterError("INVALID_ACCOUNT_MAPPING", exit_code=3)
    normalized: dict[str, str] = {}
    seen_mailboxes: set[str] = set()
    for alias, email in raw_accounts.items():
        if (
            not isinstance(alias, str)
            or not ALIAS_RE.fullmatch(alias)
            or not isinstance(email, str)
            or email != email.strip()
            or not EMAIL_RE.fullmatch(email)
        ):
            raise AdapterError("INVALID_ACCOUNT_MAPPING", exit_code=3)
        normalized_email = email.lower()
        if normalized_email in seen_mailboxes:
            raise AdapterError("INVALID_ACCOUNT_MAPPING", exit_code=3)
        seen_mailboxes.add(normalized_email)
        normalized[alias] = normalized_email
    return normalized


def load_account_context(alias: str, hermes_home: Path | None = None) -> AccountContext:
    """Resolve one named account without accepting an arbitrary credential path."""

    alias = _validate_alias(alias)
    home = Path(hermes_home) if hermes_home is not None else get_hermes_home()
    accounts_root = home / "google-accounts"
    account_dir = accounts_root / alias
    account_map_path = accounts_root / "accounts.json"
    token_path = account_dir / "google_token.json"
    try:
        resolved_root = accounts_root.resolve(strict=True)
        resolved_dir = account_dir.resolve(strict=True)
        resolved_map = account_map_path.resolve(strict=True)
        resolved_token = token_path.resolve(strict=True)
        resolved_dir.relative_to(resolved_root)
        resolved_map.relative_to(resolved_root)
        resolved_token.relative_to(resolved_dir)
    except (FileNotFoundError, OSError, ValueError):
        raise AdapterError("ACCOUNT_CREDENTIALS_MISSING", exit_code=3) from None
    if resolved_map != account_map_path or resolved_token != token_path:
        raise AdapterError("UNSAFE_ACCOUNT_CREDENTIALS", exit_code=3)
    account_map = _read_private_json(
        account_map_path, error_code="UNSAFE_ACCOUNT_MAPPING"
    )
    normalized_accounts = _normalize_account_mapping(account_map)
    expected_email = normalized_accounts.get(alias)
    if expected_email is None:
        raise AdapterError("ACCOUNT_IDENTITY_MISSING", exit_code=3)
    payload = _read_private_json(token_path, error_code="UNSAFE_ACCOUNT_CREDENTIALS")
    raw_scopes = payload.get("scopes")
    if (
        not isinstance(raw_scopes, list)
        or not raw_scopes
        or not all(isinstance(v, str) for v in raw_scopes)
    ):
        raise AdapterError("GMAIL_SCOPE_MISMATCH", exit_code=3)
    scopes = tuple(dict.fromkeys(raw_scopes))
    scope_set = set(scopes)
    if (
        GMAIL_READONLY_SCOPE not in scope_set
        or not scope_set <= APPROVED_READONLY_SCOPES
    ):
        raise AdapterError("GMAIL_SCOPE_MISMATCH", exit_code=3)
    return AccountContext(alias, expected_email, token_path, payload, scopes)


def build_service(context: AccountContext, *, timeout_s: float) -> Any:
    """Build an authorized Gmail service with a per-request transport timeout."""

    try:
        import google_auth_httplib2
        import httplib2
        from google.auth.exceptions import GoogleAuthError, RefreshError
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        from googleapiclient.errors import Error as GoogleApiError
    except ImportError:
        raise AdapterError("GOOGLE_DEPENDENCIES_MISSING", exit_code=3) from None
    try:
        credentials = Credentials.from_authorized_user_info(
            context.token_payload, list(context.scopes)
        )
        transport = google_auth_httplib2.AuthorizedHttp(
            credentials,
            http=httplib2.Http(timeout=float(timeout_s)),
        )
        return build("gmail", "v1", http=transport, cache_discovery=False)
    except RefreshError:
        raise AdapterError("REAUTH_REQUIRED", exit_code=3) from None
    except (GoogleAuthError, GoogleApiError, OSError, TypeError, ValueError):
        raise AdapterError("ACCOUNT_AUTH_UNAVAILABLE", exit_code=3) from None


def _http_status(exc: BaseException) -> int | None:
    status = getattr(getattr(exc, "resp", None), "status", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _set_transport_timeout(service: Any, timeout_s: float) -> None:
    authorized_http = getattr(service, "_http", None)
    transport = getattr(authorized_http, "http", None)
    if transport is not None and hasattr(transport, "timeout"):
        transport.timeout = timeout_s


def _is_refresh_error(exc: BaseException) -> bool:
    try:
        from google.auth.exceptions import RefreshError
    except ImportError:
        error_type = type(exc)
        return (
            error_type.__module__ == "google.auth.exceptions"
            and error_type.__name__ == "RefreshError"
        )
    return isinstance(exc, RefreshError)


def _decode_urlsafe(value: str, *, error_code: str) -> bytes:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]*={0,2}", value):
        raise AdapterError(error_code)
    try:
        padded = value + "=" * (-len(value) % 4)
        decoded = base64.b64decode(
            padded.encode("ascii"), altchars=b"-_", validate=True
        )
    except (binascii.Error, ValueError, UnicodeEncodeError):
        raise AdapterError(error_code) from None
    # Reject noncanonical padding bits as well as invalid alphabet characters.
    if base64.urlsafe_b64encode(decoded).decode("ascii").rstrip("=") != value.rstrip(
        "="
    ):
        raise AdapterError(error_code)
    if "=" in value and len(value) % 4:
        raise AdapterError(error_code)
    return decoded


def _decode_body_prefix(value: str, max_bytes: int) -> tuple[bytes, bool]:
    if not isinstance(value, str):
        raise AdapterError("INVALID_MESSAGE_BODY")
    max_encoded = ((max_bytes + 2) // 3) * 4
    prefix_text = value[:max_encoded]
    try:
        prefix = prefix_text.encode("ascii")
    except UnicodeEncodeError:
        raise AdapterError("INVALID_MESSAGE_BODY") from None
    padded = prefix + b"=" * (-len(prefix) % 4)
    try:
        decoded = base64.b64decode(padded, altchars=b"-_", validate=True)
    except (binascii.Error, ValueError):
        raise AdapterError("INVALID_MESSAGE_BODY") from None
    truncated = len(value) > len(prefix_text) or len(decoded) > max_bytes
    return decoded[:max_bytes], truncated


def _headers(payload: dict[str, Any]) -> dict[str, str]:
    output: dict[str, str] = {}
    for header in payload.get("headers") or []:
        if isinstance(header, dict) and isinstance(header.get("name"), str):
            output[header["name"].lower()] = str(header.get("value") or "")
    return output


def _walk_parts(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    stack = [payload]
    visited = 0
    while stack:
        part = stack.pop()
        if not isinstance(part, dict):
            continue
        visited += 1
        if visited > HARD_MAX_MIME_PARTS:
            raise AdapterError("MESSAGE_STRUCTURE_TOO_LARGE")
        yield part
        children = part.get("parts")
        if isinstance(children, list):
            stack.extend(reversed(children))


def _body_and_attachments(
    payload: dict[str, Any], max_body_bytes: int
) -> tuple[str, bool, list[dict[str, Any]]]:
    candidates: dict[int, tuple[str, bool]] = {}
    attachments: list[dict[str, Any]] = []
    seen_attachments: set[str] = set()
    for part in _walk_parts(payload):
        body = part.get("body") if isinstance(part.get("body"), dict) else {}
        attachment_id = body.get("attachmentId")
        if attachment_id is not None:
            _validate_id(attachment_id, "attachment")
        if attachment_id is not None and attachment_id not in seen_attachments:
            if len(attachments) >= HARD_MAX_ATTACHMENTS:
                raise AdapterError("MESSAGE_STRUCTURE_TOO_LARGE")
            seen_attachments.add(attachment_id)
            size = body.get("size", 0)
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                size = 0
            attachments.append(
                {
                    "attachment_id": attachment_id,
                    "filename": str(part.get("filename") or ""),
                    "mime_type": str(
                        part.get("mimeType") or "application/octet-stream"
                    ),
                    "size": size,
                }
            )
        data = body.get("data")
        mime = str(part.get("mimeType") or "").lower()
        if isinstance(data, str) and mime in {"text/plain", "text/html"}:
            priority = 0 if mime == "text/plain" else 1
            if priority not in candidates:
                decoded, truncated = _decode_body_prefix(data, max_body_bytes)
                candidates[priority] = (
                    decoded.decode("utf-8", errors="ignore"),
                    truncated,
                )
    text, truncated = candidates[min(candidates)] if candidates else ("", False)
    return text, truncated, attachments


def _secure_dir(path: Path, *, boundary: Path) -> None:
    """Create a private tree without following an in-tree symlink."""

    boundary = boundary.absolute()
    path = path.absolute()
    try:
        relative = path.relative_to(boundary)
    except ValueError:
        raise AdapterError("UNSAFE_PRIVATE_STORAGE") from None
    if boundary.is_symlink() or not boundary.is_dir():
        raise AdapterError("UNSAFE_PRIVATE_STORAGE")
    current = boundary
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise AdapterError("UNSAFE_PRIVATE_STORAGE")
        current.mkdir(mode=0o700, exist_ok=True)
        if not current.is_dir():
            raise AdapterError("UNSAFE_PRIVATE_STORAGE")
        current.chmod(0o700)


def _atomic_private_json(
    path: Path, payload: dict[str, Any], *, boundary: Path
) -> None:
    _secure_dir(path.parent, boundary=boundary)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _private_regular_file(info: os.stat_result) -> bool:
    getuid = getattr(os, "getuid", None)
    owner_matches = getuid is None or info.st_uid == getuid()
    return (
        stat.S_ISREG(info.st_mode)
        and stat.S_IMODE(info.st_mode) == 0o600
        and owner_matches
    )


def _valid_publication_links(path: Path, descriptor: int) -> bool:
    info = os.fstat(descriptor)
    if info.st_nlink == 1:
        return True
    if info.st_nlink != 2:
        return False
    # A writer can be interrupted between link and unlink. Only its private,
    # same-directory publication name is allowed; arbitrary hardlinks fail.
    for temporary in path.parent.glob(f".{path.name}.*.tmp"):
        nonce = temporary.name[len(path.name) + 2 : -4]
        if re.fullmatch(r"[0-9a-f]{32}", nonce) is None:
            continue
        try:
            linked = temporary.lstat()
        except FileNotFoundError:
            continue
        if _private_regular_file(linked) and (linked.st_dev, linked.st_ino) == (
            info.st_dev,
            info.st_ino,
        ):
            return True
    # The publishing writer may have removed its temporary during the scan.
    return os.fstat(descriptor).st_nlink == 1


def _stored_attachment(path: Path, digest: str) -> tuple[Path, str, bool]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise AdapterError("UNSAFE_PRIVATE_STORAGE") from None
    with os.fdopen(descriptor, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not _private_regular_file(info):
            raise AdapterError("UNSAFE_PRIVATE_STORAGE")
        file_hash = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            file_hash.update(chunk)
        if file_hash.hexdigest() != digest:
            raise AdapterError("ATTACHMENT_STORAGE_MISMATCH")
        named = path.lstat()
        if (
            not _private_regular_file(named)
            or (named.st_dev, named.st_ino) != (info.st_dev, info.st_ino)
            or not _valid_publication_links(path, handle.fileno())
        ):
            raise AdapterError("UNSAFE_PRIVATE_STORAGE")
    return path, digest, True


def _sync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _store_attachment(
    root: Path, message_id: str, data: bytes, *, boundary: Path
) -> tuple[Path, str, bool]:
    digest = hashlib.sha256(data).hexdigest()
    directory = root / "attachments" / message_id
    _secure_dir(directory, boundary=boundary)
    destination = directory / digest
    if destination.exists():
        stored = _stored_attachment(destination, digest)
        _sync_directory(directory)
        return stored
    temporary = directory / f".{digest}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    reused = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        # link publishes complete bytes atomically and cannot replace a winner.
        try:
            os.link(temporary, destination)
        except FileExistsError:
            reused = True
        _stored_attachment(destination, digest)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    _sync_directory(directory)
    return destination, digest, reused


class GmailReadClient:
    def __init__(
        self,
        service: Any,
        context: AccountContext,
        limits: Limits,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        request_timeout_s: float | None = None,
        started_monotonic: float | None = None,
    ) -> None:
        self.service = service
        self.context = context
        self.limits = limits
        self.monotonic = monotonic
        self.sleep = sleep
        self.started = monotonic() if started_monotonic is None else started_monotonic
        self.request_timeout_s = min(
            request_timeout_s if request_timeout_s is not None else limits.deadline_s,
            limits.deadline_s,
        )
        self.last_request_timeout_s = self.request_timeout_s
        self.api_calls = 0
        self.identity_verified = False

    def _execute(self, factory: Callable[[], Any]) -> dict[str, Any]:
        for attempt in range(self.limits.attempts):
            remaining = self.limits.deadline_s - (self.monotonic() - self.started)
            if remaining <= 0:
                raise AdapterError("DEADLINE_EXCEEDED")
            if self.api_calls >= self.limits.max_api_calls:
                raise AdapterError("API_CALL_BUDGET_EXCEEDED")
            self.api_calls += 1
            self.last_request_timeout_s = min(self.request_timeout_s, remaining)
            _set_transport_timeout(self.service, self.last_request_timeout_s)
            try:
                result = factory().execute()
                if not isinstance(result, dict):
                    raise AdapterError("INVALID_GMAIL_RESPONSE")
                return result
            except AdapterError:
                raise
            except Exception as exc:  # health: allow BLE001 -- redact the third-party transport boundary
                if _is_refresh_error(exc):
                    raise AdapterError("REAUTH_REQUIRED", exit_code=3) from None
                status = _http_status(exc)
                if status == 401:
                    raise AdapterError("REAUTH_REQUIRED", exit_code=3) from None
                if status == 403:
                    raise AdapterError("GMAIL_READ_FORBIDDEN", exit_code=3) from None
                if status not in RETRYABLE_HTTP or attempt + 1 >= self.limits.attempts:
                    raise AdapterError("GMAIL_API_UNAVAILABLE") from None
                delay = min(2**attempt, 4)
                if self.monotonic() - self.started + delay >= self.limits.deadline_s:
                    raise AdapterError("DEADLINE_EXCEEDED") from None
                self.sleep(delay)
        raise AdapterError("GMAIL_API_UNAVAILABLE")

    def verify_identity(self) -> dict[str, Any]:
        profile = self._execute(lambda: self.service.users().getProfile(userId="me"))
        actual = profile.get("emailAddress")
        if (
            not isinstance(actual, str)
            or actual.strip().lower() != self.context.expected_email
        ):
            raise AdapterError("ACCOUNT_IDENTITY_MISMATCH", exit_code=3)
        self.identity_verified = True
        return profile

    def profile(self) -> dict[str, Any]:
        profile = self.verify_identity()
        return {
            "history_id_present": bool(profile.get("historyId")),
            "messages_total": _nonnegative_int(profile.get("messagesTotal")),
            "threads_total": _nonnegative_int(profile.get("threadsTotal")),
        }

    def search(
        self, query: str, max_results: int, page_token: str | None
    ) -> dict[str, Any]:
        self.verify_identity()
        page = self._execute(
            lambda: (
                self.service.users()
                .messages()
                .list(
                    userId="me",
                    q=query,
                    maxResults=max_results,
                    pageToken=page_token,
                )
            )
        )
        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        raw_messages = page.get("messages") or []
        if not isinstance(raw_messages, list):
            raise AdapterError("INVALID_GMAIL_RESPONSE")
        if len(raw_messages) > max_results:
            raise AdapterError("RESPONSE_ITEM_BUDGET_EXCEEDED")
        for item in raw_messages:
            if not isinstance(item, dict):
                continue
            message_id = _validate_id(item.get("id"), "message")
            if message_id in seen:
                continue
            seen.add(message_id)
            message = self._execute(
                lambda message_id=message_id: (
                    self.service.users()
                    .messages()
                    .get(
                        userId="me",
                        id=message_id,
                        format="metadata",
                        metadataHeaders=[
                            "From",
                            "To",
                            "Cc",
                            "Subject",
                            "Date",
                            "Message-ID",
                        ],
                    )
                )
            )
            if _validate_id(message.get("id"), "message") != message_id:
                raise AdapterError("MESSAGE_ID_MISMATCH")
            headers = _headers(message.get("payload") or {})
            items.append(_message_metadata(self.context, message, headers))
        next_page_token = _validate_page_token(page.get("nextPageToken"))
        return {
            "items": items,
            "next_page_token": next_page_token,
            "deduplicated": len(seen),
        }

    def get_message(
        self, message_id: str, max_body_bytes: int, max_message_bytes: int
    ) -> dict[str, Any]:
        self.verify_identity()
        metadata = self._execute(
            lambda: (
                self.service.users()
                .messages()
                .get(
                    userId="me",
                    id=message_id,
                    format="metadata",
                    fields="id,sizeEstimate",
                )
            )
        )
        if _validate_id(metadata.get("id"), "message") != message_id:
            raise AdapterError("MESSAGE_ID_MISMATCH")
        size_estimate = _required_nonnegative_int(
            metadata.get("sizeEstimate"), "INVALID_MESSAGE_SIZE"
        )
        if size_estimate > max_message_bytes:
            raise AdapterError("MESSAGE_TOO_LARGE")
        message = self._execute(
            lambda: (
                self.service.users()
                .messages()
                .get(userId="me", id=message_id, format="full")
            )
        )
        if _validate_id(message.get("id"), "message") != message_id:
            raise AdapterError("MESSAGE_ID_MISMATCH")
        payload = (
            message.get("payload") if isinstance(message.get("payload"), dict) else {}
        )
        headers = _headers(payload)
        body, truncated, attachments = _body_and_attachments(payload, max_body_bytes)
        result = _message_metadata(self.context, message, headers)
        result.update(
            {
                "size_estimate": size_estimate,
                "body": body,
                "body_truncated": truncated,
                "attachments": attachments,
            }
        )
        return result

    def get_attachment(
        self,
        message_id: str,
        attachment_id: str,
        max_bytes: int,
        root: Path,
        boundary: Path,
    ) -> dict[str, Any]:
        self.verify_identity()
        response = self._execute(
            lambda: (
                self.service.users()
                .messages()
                .attachments()
                .get(userId="me", messageId=message_id, id=attachment_id)
            )
        )
        encoded = response.get("data")
        if (
            not isinstance(encoded, str)
            or len(encoded) > ((max_bytes + 2) // 3) * 4 + 8
        ):
            raise AdapterError("ATTACHMENT_TOO_LARGE")
        data = _decode_urlsafe(encoded, error_code="INVALID_ATTACHMENT_DATA")
        if len(data) > max_bytes:
            raise AdapterError("ATTACHMENT_TOO_LARGE")
        try:
            path, digest, reused = _store_attachment(
                root, message_id, data, boundary=boundary
            )
        except OSError:
            raise AdapterError("ATTACHMENT_STORAGE_UNAVAILABLE") from None
        return {
            "source_ref": f"{self.context.source_id}:{message_id}:attachment:{attachment_id}",
            "bytes": len(data),
            "sha256": digest,
            "path": str(path),
            "reused": reused,
        }


def _nonnegative_int(value: Any) -> int:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else 0
    )


def _required_nonnegative_int(value: Any, error_code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AdapterError(error_code)
    return value


def _message_metadata(
    context: AccountContext, message: dict[str, Any], headers: dict[str, str]
) -> dict[str, Any]:
    message_id = _validate_id(message.get("id"), "message")
    thread_id = _validate_id(message.get("threadId"), "thread")
    labels = message.get("labelIds")
    return {
        "source_ref": f"{context.source_id}:{message_id}",
        "id": message_id,
        "thread_id": thread_id,
        "from": headers.get("from", ""),
        "to": headers.get("to", ""),
        "cc": headers.get("cc", ""),
        "subject": headers.get("subject", ""),
        "date": headers.get("date", ""),
        "message_id_header": headers.get("message-id", ""),
        "snippet": str(message.get("snippet") or ""),
        "labels": [str(value) for value in labels] if isinstance(labels, list) else [],
    }


def _operation_key(alias: str, operation: str, request_shape: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"account": alias, "operation": operation, "request": request_shape},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _write_receipt(
    home: Path, alias: str, receipt: dict[str, Any], operation_key: str
) -> Path:
    path = (
        home
        / "source-cache"
        / "gmail-read"
        / alias
        / "receipts"
        / f"{operation_key}.json"
    )
    _atomic_private_json(path, receipt, boundary=home)
    return path


def _run_profile(
    args: argparse.Namespace,
    client: GmailReadClient,
    home: Path,
    context: AccountContext,
) -> tuple[dict[str, Any], dict[str, Any]]:
    del args, home, context
    return client.profile(), {}


def _run_search(
    args: argparse.Namespace,
    client: GmailReadClient,
    home: Path,
    context: AccountContext,
) -> tuple[dict[str, Any], dict[str, Any]]:
    del home, context
    query = _validate_query(args.query)
    token = _validate_page_token(args.page_token)
    request_shape = {
        "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
        "max_results": args.max_results,
        "page_token_present": token is not None,
        "page_token_sha256": (
            hashlib.sha256(token.encode()).hexdigest() if token is not None else None
        ),
    }
    return client.search(query, args.max_results, token), request_shape


def _run_get(
    args: argparse.Namespace,
    client: GmailReadClient,
    home: Path,
    context: AccountContext,
) -> tuple[dict[str, Any], dict[str, Any]]:
    del home, context
    message_id = _validate_id(args.message_id, "message")
    request_shape = {
        "message_id_sha256": hashlib.sha256(message_id.encode()).hexdigest(),
        "max_body_bytes": args.max_body_bytes,
        "max_message_bytes": args.max_message_bytes,
    }
    data = client.get_message(message_id, args.max_body_bytes, args.max_message_bytes)
    return data, request_shape


def _run_attachment_get(
    args: argparse.Namespace,
    client: GmailReadClient,
    home: Path,
    context: AccountContext,
) -> tuple[dict[str, Any], dict[str, Any]]:
    message_id = _validate_id(args.message_id, "message")
    attachment_id = _validate_id(args.attachment_id, "attachment")
    request_shape = {
        "message_id_sha256": hashlib.sha256(message_id.encode()).hexdigest(),
        "attachment_id_sha256": hashlib.sha256(attachment_id.encode()).hexdigest(),
        "max_bytes": args.max_bytes,
    }
    root = home / "source-cache" / "gmail-read" / context.alias
    data = client.get_attachment(message_id, attachment_id, args.max_bytes, root, home)
    return data, request_shape


_OPERATION_HANDLERS: dict[
    str,
    Callable[
        [argparse.Namespace, GmailReadClient, Path, AccountContext],
        tuple[dict[str, Any], dict[str, Any]],
    ],
] = {
    "profile": _run_profile,
    "search": _run_search,
    "get": _run_get,
    "attachment-get": _run_attachment_get,
}


def _profile_shape(args: argparse.Namespace) -> dict[str, Any]:
    del args
    return {}


def _search_shape(args: argparse.Namespace) -> dict[str, Any]:
    query = _validate_query(args.query)
    token = _validate_page_token(args.page_token)
    return {
        "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
        "max_results": args.max_results,
        "page_token_present": token is not None,
        "page_token_sha256": (
            hashlib.sha256(token.encode()).hexdigest() if token is not None else None
        ),
    }


def _get_shape(args: argparse.Namespace) -> dict[str, Any]:
    message_id = _validate_id(args.message_id, "message")
    return {
        "message_id_sha256": hashlib.sha256(message_id.encode()).hexdigest(),
        "max_body_bytes": args.max_body_bytes,
        "max_message_bytes": args.max_message_bytes,
    }


def _attachment_shape(args: argparse.Namespace) -> dict[str, Any]:
    message_id = _validate_id(args.message_id, "message")
    attachment_id = _validate_id(args.attachment_id, "attachment")
    return {
        "message_id_sha256": hashlib.sha256(message_id.encode()).hexdigest(),
        "attachment_id_sha256": hashlib.sha256(attachment_id.encode()).hexdigest(),
        "max_bytes": args.max_bytes,
    }


_REQUEST_SHAPE_BUILDERS: dict[str, Callable[[argparse.Namespace], dict[str, Any]]] = {
    "profile": _profile_shape,
    "search": _search_shape,
    "get": _get_shape,
    "attachment-get": _attachment_shape,
}


def execute(
    args: argparse.Namespace,
    *,
    service: Any | None = None,
    hermes_home: Path | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    home = Path(hermes_home) if hermes_home is not None else get_hermes_home()
    started_at = _utcnow()
    started_monotonic = monotonic()
    handler = _OPERATION_HANDLERS.get(args.operation)
    shape_builder = _REQUEST_SHAPE_BUILDERS.get(args.operation)
    if handler is None or shape_builder is None:
        raise AdapterError("UNSUPPORTED_OPERATION", exit_code=2)
    request_shape = shape_builder(args)
    client: GmailReadClient | None = None
    try:
        context = load_account_context(args.account, home)
        limits = Limits(args.deadline, args.max_api_calls, args.attempts)
        remaining_for_build = args.deadline - (monotonic() - started_monotonic)
        if remaining_for_build <= 0:
            raise AdapterError("DEADLINE_EXCEEDED")
        gmail = (
            service
            if service is not None
            else build_service(
                context, timeout_s=min(args.timeout, remaining_for_build)
            )
        )
        client = GmailReadClient(
            gmail,
            context,
            limits,
            monotonic=monotonic,
            request_timeout_s=min(args.timeout, args.deadline),
            started_monotonic=started_monotonic,
        )
        data, handler_shape = handler(args, client, home, context)
        if handler_shape != request_shape:
            raise AdapterError("REQUEST_SHAPE_MISMATCH")
    except AdapterError as exc:
        exc.bind_runtime(
            api_calls=client.api_calls if client is not None else 0,
            identity_verified=(
                client.identity_verified if client is not None else False
            ),
            request_shape=request_shape,
            started_at=started_at,
        )
        raise
    finished_at = _utcnow()
    operation_key = _operation_key(context.alias, args.operation, request_shape)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "source_id": context.source_id,
        "account": context.alias,
        "operation": args.operation,
        "status": "complete",
        "started_at": started_at,
        "finished_at": finished_at,
        "identity_verified": client.identity_verified,
        "api_calls": client.api_calls,
        "operation_key": operation_key,
    }
    receipt_path = _write_receipt(home, context.alias, receipt, operation_key)
    return {
        "schema_version": SCHEMA_VERSION,
        "source_id": context.source_id,
        "account": context.alias,
        "operation": args.operation,
        "status": "complete",
        "identity_verified": client.identity_verified,
        "data": data,
        "receipt": {**receipt, "path": str(receipt_path)},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strict account-scoped read-only Gmail adapter"
    )
    parser.add_argument(
        "--account", required=True, help="Named account alias under google-accounts/"
    )
    parser.add_argument("--timeout", type=_bounded_float(1.0, 120.0), default=30.0)
    parser.add_argument(
        "--deadline", type=_bounded_float(1.0, HARD_MAX_DEADLINE_S), default=60.0
    )
    parser.add_argument(
        "--max-api-calls", type=_bounded_int(1, HARD_MAX_API_CALLS), default=128
    )
    parser.add_argument(
        "--attempts", type=_bounded_int(1, HARD_MAX_ATTEMPTS), default=3
    )
    commands = parser.add_subparsers(dest="operation", required=True)
    commands.add_parser("profile")
    search = commands.add_parser("search")
    search.add_argument("query")
    search.add_argument(
        "--max-results", type=_bounded_int(1, HARD_MAX_RESULTS), default=10
    )
    search.add_argument("--page-token", default="")
    message = commands.add_parser("get")
    message.add_argument("message_id")
    message.add_argument(
        "--max-body-bytes",
        type=_bounded_int(1, HARD_MAX_BODY_BYTES),
        default=1024 * 1024,
    )
    message.add_argument(
        "--max-message-bytes",
        type=_bounded_int(1, HARD_MAX_MESSAGE_BYTES),
        default=25 * 1024 * 1024,
    )
    attachment = commands.add_parser("attachment-get")
    attachment.add_argument("message_id")
    attachment.add_argument("attachment_id")
    attachment.add_argument(
        "--max-bytes",
        type=_bounded_int(1, HARD_MAX_ATTACHMENT_BYTES),
        default=25 * 1024 * 1024,
    )
    return parser


def _persist_error_receipt(
    args: argparse.Namespace, error: AdapterError, home: Path | None = None
) -> Path:
    home = Path(home) if home is not None else get_hermes_home()
    alias = _validate_alias(args.account)
    operation = str(args.operation)
    request_shape = error.request_shape
    if request_shape is None:
        shape_builder = _REQUEST_SHAPE_BUILDERS.get(operation)
        request_shape = shape_builder(args) if shape_builder is not None else {}
    operation_key = _operation_key(
        alias,
        operation,
        {**request_shape, "error_code": error.code},
    )
    now = _utcnow()
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "source_id": f"gmail:{alias}",
        "account": alias,
        "operation": operation,
        "status": "blocked" if error.exit_code == 3 else "error",
        "started_at": error.started_at or now,
        "finished_at": now,
        "identity_verified": error.identity_verified,
        "api_calls": error.api_calls,
        "operation_key": operation_key,
        "error_code": error.code,
    }
    return _write_receipt(home, alias, receipt, operation_key)


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        output = execute(args)
    except AdapterError as exc:
        try:
            _persist_error_receipt(args, exc)
        except (AdapterError, OSError):
            # Keep the original typed error authoritative if private receipt
            # storage is unavailable.
            pass
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "blocked" if exc.exit_code == 3 else "error",
                    "error": {"code": exc.code},
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return exc.exit_code
    print(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
