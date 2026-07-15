"""Synthetic, non-private Telegram gateway delivery canary.

The canary reuses the production authorization and Telegram send paths. It
injects one pre-send network failure (which cannot reach Telegram), delivers a
payload that must be split, and submits the same idempotency key twice so the
second attempt is suppressed locally. Raw chat/user identifiers and bot tokens
are never written to the receipt.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from gateway.authz_mixin import GatewayAuthorizationMixin
from gateway.config import Platform
from gateway.platforms.base import SendResult, utf16_len
from gateway.session import SessionSource


CANARY_SCHEMA = "hermes.telegram-canary/v1"
_MAX_STATE_RUNS = 100


class SyntheticRetryProbe:
    """Wrap a Telegram Bot and inject one known pre-send connection failure."""

    def __init__(self, bot: Any, *, injected_failures: int = 1):
        self._bot = bot
        self._remaining_failures = injected_failures
        self.injected_failures = 0
        self.attempts = 0

    async def send_message(self, **kwargs):
        self.attempts += 1
        if self._remaining_failures:
            self._remaining_failures -= 1
            self.injected_failures += 1
            from telegram.error import NetworkError

            raise NetworkError("synthetic pre-send connection failure")
        return await self._bot.send_message(**kwargs)

    def __getattr__(self, name: str):
        return getattr(self._bot, name)


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _parse_csv_env(name: str) -> set[str]:
    return {
        item.strip()
        for item in os.getenv(name, "").split(",")
        if item.strip()
    }


def _auth_checks() -> dict[str, bool]:
    """Exercise the production gateway DM authorization method fail-closed."""
    platform_ids = _parse_csv_env("TELEGRAM_ALLOWED_USERS")
    global_ids = _parse_csv_env("GATEWAY_ALLOWED_USERS")
    allowed_ids = platform_ids | global_ids
    allow_all = any(
        os.getenv(name, "").strip().lower() in {"true", "1", "yes"}
        for name in ("TELEGRAM_ALLOW_ALL_USERS", "GATEWAY_ALLOW_ALL_USERS")
    )
    strict = not allow_all and "*" not in allowed_ids and len(allowed_ids) == 1
    owner_id = next(iter(allowed_ids), "")

    checker = GatewayAuthorizationMixin()
    checker.adapters = {}
    checker.pairing_stores = {}
    checker.pairing_store = None

    def source(user_id: str) -> SessionSource:
        return SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="synthetic-canary-auth",
            chat_type="dm",
            user_id=user_id,
            user_name=None,
        )

    owner_allowed = bool(owner_id) and checker._is_user_authorized(source(owner_id))
    unknown_denied = not checker._is_user_authorized(
        source("synthetic-canary-unknown-user")
    )
    return {
        "gateway_path_exercised": True,
        "strict_single_owner_allowlist": strict,
        "owner_allowed": owner_allowed,
        "unknown_denied": unknown_denied,
    }


def _secure_parent(path: Path) -> None:
    parent = path.parent
    if parent.exists() and parent.is_symlink():
        raise ValueError(f"refusing symlinked receipt directory: {parent}")
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent.chmod(0o700)


def _reject_symlink(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"refusing symlinked canary file: {path}")


def _load_state(path: Path) -> dict[str, Any]:
    _secure_parent(path)
    _reject_symlink(path)
    if not path.exists():
        return {"schema": CANARY_SCHEMA, "runs": {}}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("canary state is unreadable; refusing delivery") from exc
    if loaded.get("schema") != CANARY_SCHEMA or not isinstance(loaded.get("runs"), dict):
        raise ValueError("canary state has an unsupported schema; refusing delivery")
    return loaded


def _write_state(path: Path, state: dict[str, Any]) -> None:
    _secure_parent(path)
    _reject_symlink(path)
    runs = state.get("runs", {})
    if len(runs) > _MAX_STATE_RUNS:
        ordered = sorted(
            runs.items(), key=lambda item: str(item[1].get("updated_at", ""))
        )
        state["runs"] = dict(ordered[-_MAX_STATE_RUNS:])
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(temp, flags, 0o600)
    try:
        payload = json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n"
        os.write(fd, payload.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    temp.replace(path)
    path.chmod(0o600)


def append_private_receipt(path: Path, receipt: dict[str, Any]) -> str:
    """Append one receipt record and return the SHA-256 of the full file."""
    _secure_parent(path)
    _reject_symlink(path)
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        line = json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
        os.write(fd, line.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    path.chmod(0o600)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def redact_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    """Return a shareable receipt without operational Telegram message IDs."""
    redacted = deepcopy(receipt)
    delivery = redacted.get("checks", {}).get("delivery", {})
    if delivery.get("message_ids"):
        delivery["message_ids"] = ["<redacted>"] * len(delivery["message_ids"])
    return redacted


async def _deliver_once(
    *,
    adapter: Any,
    chat_id: str,
    content: str,
    idempotency_key: str,
    state_path: Path,
    updated_at: str,
) -> tuple[SendResult | None, bool]:
    state = _load_state(state_path)
    runs = state["runs"]
    existing = runs.get(idempotency_key)
    if isinstance(existing, dict):
        return None, True

    # Claim before sending. If the process dies after Telegram accepts the
    # message but before the final state write, the retained pending claim
    # suppresses an unsafe resend of an ambiguously delivered payload.
    runs[idempotency_key] = {"status": "pending", "updated_at": updated_at}
    _write_state(state_path, state)

    result = await adapter.send(chat_id, content, metadata={"notify": False})
    state = _load_state(state_path)
    state["runs"][idempotency_key] = {
        "status": "delivered" if result.success else "delivery_failed",
        "updated_at": updated_at,
    }
    _write_state(state_path, state)
    return result, False


async def run_canary(
    *,
    adapter: Any,
    probe: SyntheticRetryProbe,
    chat_id: str,
    destination_alias: str,
    receipt_path: Path,
    state_path: Path,
    runtime_sha: str,
    run_id: str,
    created_at: str,
) -> tuple[dict[str, Any], str]:
    """Run the canary and return ``(private_receipt, receipt_file_sha256)``."""
    auth = _auth_checks()
    content = (
        f"Hermes synthetic gateway canary {run_id}. Non-private test payload.\n"
        + ("A" * 5000)
    )
    content_sha256 = _sha256_text(content)
    idempotency_key = _sha256_text(
        f"{CANARY_SCHEMA}|{run_id}|{destination_alias}|{content_sha256}"
    )

    result: SendResult | None = None
    first_duplicate = False
    second_duplicate = False
    if all(auth.values()):
        result, first_duplicate = await _deliver_once(
            adapter=adapter,
            chat_id=chat_id,
            content=content,
            idempotency_key=idempotency_key,
            state_path=state_path,
            updated_at=created_at,
        )
        _, second_duplicate = await _deliver_once(
            adapter=adapter,
            chat_id=chat_id,
            content=content,
            idempotency_key=idempotency_key,
            state_path=state_path,
            updated_at=created_at,
        )

    raw = result.raw_response if result and isinstance(result.raw_response, dict) else {}
    message_ids = [str(item) for item in raw.get("message_ids", [])]
    attempt_counts = [int(item) for item in raw.get("attempt_counts", [])]
    chunk_units = [int(item) for item in raw.get("chunk_utf16_units", [])]
    chunk_hashes = [str(item) for item in raw.get("chunk_sha256", [])]
    chunk_count = int(raw.get("chunk_count", 0) or 0)
    delivered = bool(result and result.success)
    retry_ok = bool(attempt_counts and attempt_counts[0] >= 2)
    length_ok = bool(
        chunk_count >= 2
        and len(message_ids) == chunk_count
        and len(chunk_units) == chunk_count
        and len(chunk_hashes) == chunk_count
        and all(units <= 4096 for units in chunk_units)
    )
    idempotency_ok = not first_duplicate and second_duplicate and delivered
    passed = all(auth.values()) and retry_ok and length_ok and idempotency_ok

    receipt = {
        "schema": CANARY_SCHEMA,
        "run_id": run_id,
        "created_at": created_at,
        "runtime_sha": runtime_sha,
        "synthetic": True,
        "private_data": False,
        "qualifies_for_health_p6": False,
        "destination_alias": destination_alias,
        "result": "pass" if passed else "fail",
        "checks": {
            "authentication": auth,
            "idempotency": {
                "idempotency_key_sha256": idempotency_key,
                "delivery_attempts": 2 if all(auth.values()) else 0,
                "actual_deliveries": 1 if delivered else 0,
                "duplicates_suppressed": 1 if second_duplicate else 0,
            },
            "retry": {
                "injected_failures": probe.injected_failures,
                "retried": retry_ok,
                "attempt_counts": attempt_counts,
            },
            "length": {
                "input_utf16_units": utf16_len(content),
                "chunk_count": chunk_count,
                "chunk_utf16_units": chunk_units,
                "chunk_sha256": chunk_hashes,
                "max_chunk_utf16_units": 4096,
                "no_truncation": length_ok,
            },
            "delivery": {
                "acknowledged": delivered,
                "message_ids": message_ids,
                "content_sha256": content_sha256,
            },
        },
    }
    receipt_sha256 = append_private_receipt(receipt_path, receipt)
    return receipt, receipt_sha256
