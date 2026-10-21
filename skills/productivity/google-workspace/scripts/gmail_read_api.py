#!/usr/bin/env python3
"""Strict, account-scoped, read-only Gmail adapter for Hermes skills.

Named accounts resolve only below ``HERMES_HOME/google-accounts``.  Every
operation verifies the Gmail profile against the identity recorded in that
account's token before returning mailbox data.  The CLI has no mutation
commands.
"""

from __future__ import annotations

import argparse
import base64
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

    def __init__(self, code: str, *, exit_code: int = 4):
        self.code = code
        self.exit_code = exit_code
        super().__init__(code)


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
    candidate = str(value or "")
    if not ID_RE.fullmatch(candidate):
        raise AdapterError(f"INVALID_{kind.upper()}_ID", exit_code=2)
    return candidate


def _validate_query(value: str) -> str:
    query = str(value or "")
    if not query or len(query) > MAX_QUERY_CHARS or "\x00" in query:
        raise AdapterError("INVALID_QUERY", exit_code=2)
    return query


def _validate_page_token(value: str | None) -> str | None:
    if value in (None, ""):
        return None
    token = str(value)
    if len(token) > MAX_PAGE_TOKEN_CHARS or any(ord(char) < 0x20 for char in token):
        raise AdapterError("INVALID_PAGE_TOKEN", exit_code=2)
    return token


def _is_private_file(path: Path) -> bool:
    info = path.lstat()
    return (
        stat.S_ISREG(info.st_mode)
        and not path.is_symlink()
        and not (stat.S_IMODE(info.st_mode) & 0o077)
    )


def load_account_context(alias: str, hermes_home: Path | None = None) -> AccountContext:
    """Resolve one named account without accepting an arbitrary credential path."""

    alias = _validate_alias(alias)
    home = Path(hermes_home) if hermes_home is not None else get_hermes_home()
    accounts_root = home / "google-accounts"
    account_dir = accounts_root / alias
    token_path = account_dir / "google_token.json"
    try:
        resolved_root = accounts_root.resolve(strict=True)
        resolved_dir = account_dir.resolve(strict=True)
        resolved_token = token_path.resolve(strict=True)
        resolved_dir.relative_to(resolved_root)
        resolved_token.relative_to(resolved_dir)
    except (FileNotFoundError, OSError, ValueError):
        raise AdapterError("ACCOUNT_CREDENTIALS_MISSING", exit_code=3) from None
    if resolved_token != token_path or not _is_private_file(token_path):
        raise AdapterError("UNSAFE_ACCOUNT_CREDENTIALS", exit_code=3)
    try:
        if token_path.stat().st_size > MAX_TOKEN_FILE_BYTES:
            raise AdapterError("INVALID_ACCOUNT_CREDENTIALS", exit_code=3)
        payload = json.loads(token_path.read_text(encoding="utf-8"))
    except AdapterError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise AdapterError("INVALID_ACCOUNT_CREDENTIALS", exit_code=3) from None
    if not isinstance(payload, dict):
        raise AdapterError("INVALID_ACCOUNT_CREDENTIALS", exit_code=3)
    expected_email = payload.get("account")
    if not isinstance(expected_email, str) or not EMAIL_RE.fullmatch(expected_email):
        raise AdapterError("ACCOUNT_IDENTITY_MISSING", exit_code=3)
    raw_scopes = payload.get("scopes")
    if (
        not isinstance(raw_scopes, list)
        or not raw_scopes
        or not all(isinstance(v, str) for v in raw_scopes)
    ):
        raise AdapterError("GMAIL_SCOPE_MISMATCH", exit_code=3)
    scopes = tuple(dict.fromkeys(raw_scopes))
    gmail_scopes = {
        scope
        for scope in scopes
        if scope == "https://mail.google.com/"
        or scope.startswith("https://www.googleapis.com/auth/gmail.")
    }
    if gmail_scopes != {GMAIL_READONLY_SCOPE}:
        raise AdapterError("GMAIL_SCOPE_MISMATCH", exit_code=3)
    return AccountContext(alias, expected_email.lower(), token_path, payload, scopes)


def build_service(context: AccountContext, *, timeout_s: float) -> Any:
    """Build an authorized Gmail service with a per-request transport timeout."""

    try:
        import google_auth_httplib2
        import httplib2
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
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
    except (OSError, TypeError, ValueError):
        raise AdapterError("ACCOUNT_AUTH_UNAVAILABLE", exit_code=3) from None


def _http_status(exc: BaseException) -> int | None:
    status = getattr(getattr(exc, "resp", None), "status", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _decode_urlsafe(value: str, *, error_code: str) -> bytes:
    if not isinstance(value, str):
        raise AdapterError(error_code)
    try:
        padded = value + "=" * (-len(value) % 4)
        return base64.urlsafe_b64decode(padded.encode("ascii"))
    except (ValueError, UnicodeEncodeError):
        raise AdapterError(error_code) from None


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
    candidates: list[tuple[int, str]] = []
    attachments: list[dict[str, Any]] = []
    seen_attachments: set[str] = set()
    for part in _walk_parts(payload):
        body = part.get("body") if isinstance(part.get("body"), dict) else {}
        attachment_id = body.get("attachmentId")
        if (
            isinstance(attachment_id, str)
            and attachment_id
            and attachment_id not in seen_attachments
        ):
            if len(attachments) >= HARD_MAX_ATTACHMENTS:
                raise AdapterError("MESSAGE_STRUCTURE_TOO_LARGE")
            _validate_id(attachment_id, "attachment")
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
            decoded = _decode_urlsafe(data, error_code="INVALID_MESSAGE_BODY")
            candidates.append(
                (
                    0 if mime == "text/plain" else 1,
                    decoded.decode("utf-8", errors="replace"),
                )
            )
    text = min(candidates, key=lambda item: item[0])[1] if candidates else ""
    encoded = text.encode("utf-8")
    truncated = len(encoded) > max_body_bytes
    if truncated:
        encoded = encoded[:max_body_bytes]
        text = encoded.decode("utf-8", errors="ignore")
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


def _stored_attachment(path: Path, digest: str) -> tuple[Path, str, bool]:
    if path.is_symlink() or not path.is_file():
        raise AdapterError("UNSAFE_PRIVATE_STORAGE")
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise AdapterError("ATTACHMENT_STORAGE_MISMATCH")
    return path, digest, True


def _store_attachment(
    root: Path, message_id: str, data: bytes, *, boundary: Path
) -> tuple[Path, str, bool]:
    digest = hashlib.sha256(data).hexdigest()
    directory = root / "attachments" / message_id
    _secure_dir(directory, boundary=boundary)
    destination = directory / digest
    if destination.exists():
        return _stored_attachment(destination, digest)
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return _stored_attachment(destination, digest)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            destination.unlink()
        except FileNotFoundError:
            pass
        raise
    return destination, digest, False


class GmailReadClient:
    def __init__(
        self,
        service: Any,
        context: AccountContext,
        limits: Limits,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.service = service
        self.context = context
        self.limits = limits
        self.monotonic = monotonic
        self.sleep = sleep
        self.started = monotonic()
        self.api_calls = 0
        self.identity_verified = False

    def _execute(self, factory: Callable[[], Any]) -> dict[str, Any]:
        for attempt in range(self.limits.attempts):
            if self.monotonic() - self.started >= self.limits.deadline_s:
                raise AdapterError("DEADLINE_EXCEEDED")
            if self.api_calls >= self.limits.max_api_calls:
                raise AdapterError("API_CALL_BUDGET_EXCEEDED")
            self.api_calls += 1
            try:
                result = factory().execute()
                if not isinstance(result, dict):
                    raise AdapterError("INVALID_GMAIL_RESPONSE")
                return result
            except AdapterError:
                raise
            except Exception as exc:  # health: allow BLE001 -- redact the third-party transport boundary
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
        for item in page.get("messages") or []:
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
        size_estimate = _nonnegative_int(metadata.get("sizeEstimate"))
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
        path, digest, reused = _store_attachment(
            root, message_id, data, boundary=boundary
        )
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
        "message_id_sha256": hashlib.sha256(message_id.encode()).hexdigest()
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


def execute(
    args: argparse.Namespace,
    *,
    service: Any | None = None,
    hermes_home: Path | None = None,
) -> dict[str, Any]:
    home = Path(hermes_home) if hermes_home is not None else get_hermes_home()
    context = load_account_context(args.account, home)
    limits = Limits(args.deadline, args.max_api_calls, args.attempts)
    gmail = (
        service
        if service is not None
        else build_service(context, timeout_s=args.timeout)
    )
    client = GmailReadClient(gmail, context, limits)
    started_at = _utcnow()
    handler = _OPERATION_HANDLERS.get(args.operation)
    if handler is None:  # argparse prevents this, but keep dispatch fail-closed.
        raise AdapterError("UNSUPPORTED_OPERATION", exit_code=2)
    data, request_shape = handler(args, client, home, context)
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
    operation_key = _operation_key(alias, operation, {"error_code": error.code})
    now = _utcnow()
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "source_id": f"gmail:{alias}",
        "account": alias,
        "operation": operation,
        "status": "blocked" if error.exit_code == 3 else "error",
        "started_at": now,
        "finished_at": now,
        "identity_verified": False,
        "api_calls": 0,
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
