"""Converge product-managed Hindsight bank configuration through the public API."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from http.client import HTTPException as HttpClientError
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_MAX_ATTEMPTS = 3


class ReconcileError(RuntimeError):
    """Raised when managed policy or Hindsight's response violates the contract."""


@dataclass(frozen=True)
class ManagedBankPolicy:
    manifest: dict[str, Any]
    bank: dict[str, Any]


@dataclass(frozen=True)
class ReconcileSummary:
    discovered: int
    updated: int
    unchanged: int
    failed_bank_ids: tuple[str, ...]
    pending_bank_ids: tuple[str, ...]
    deadline_exceeded: bool

    @property
    def complete(self) -> bool:
        return not self.pending_bank_ids


RequestJson = Callable[[str, str, dict[str, Any] | None], dict[str, Any]]


def _required_env(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise ReconcileError(f"{name} must be configured")
    return value


def _positive_number_env(env: Mapping[str, str], name: str, *, default: float) -> float:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ReconcileError(f"{name} must be a positive number") from exc
    if value <= 0:
        raise ReconcileError(f"{name} must be a positive number")
    return value


def _json_object(raw: str, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReconcileError(f"{name} must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ReconcileError(f"{name} must be a JSON object")
    return value


def load_policy(env: Mapping[str, str]) -> ManagedBankPolicy:
    manifest = _json_object(
        _required_env(env, "HINDSIGHT_MANAGED_BANK_TEMPLATE"),
        name="HINDSIGHT_MANAGED_BANK_TEMPLATE",
    )
    try:
        allowed_value = json.loads(_required_env(env, "HINDSIGHT_MANAGED_BANK_FIELDS"))
    except json.JSONDecodeError as exc:
        raise ReconcileError("HINDSIGHT_MANAGED_BANK_FIELDS must be valid JSON") from exc
    if (
        not isinstance(allowed_value, list)
        or not allowed_value
        or any(not isinstance(field, str) or not field for field in allowed_value)
    ):
        raise ReconcileError("HINDSIGHT_MANAGED_BANK_FIELDS must be a non-empty JSON string list")
    allowed_fields = set(allowed_value)
    if len(allowed_fields) != len(allowed_value):
        raise ReconcileError("HINDSIGHT_MANAGED_BANK_FIELDS must not contain duplicates")
    if set(manifest) != {"version", "bank"} or manifest.get("version") != "1":
        raise ReconcileError(
            "HINDSIGHT_MANAGED_BANK_TEMPLATE must be a version 1 bank-only manifest"
        )
    bank = manifest.get("bank")
    if not isinstance(bank, dict) or set(bank) != allowed_fields:
        raise ReconcileError(
            "managed bank template fields must exactly match the reviewed allowlist"
        )
    return ManagedBankPolicy(manifest=manifest, bank=bank)


def _bank_ids(body: dict[str, Any]) -> list[str]:
    banks = body.get("banks")
    if not isinstance(banks, list):
        raise ReconcileError("Hindsight bank list response has no banks array")
    result: list[str] = []
    seen: set[str] = set()
    for item in banks:
        bank_id = item.get("bank_id") if isinstance(item, dict) else None
        if not isinstance(bank_id, str) or not bank_id:
            raise ReconcileError("Hindsight bank list contains an invalid bank_id")
        if bank_id not in seen:
            seen.add(bank_id)
            result.append(bank_id)
    return result


def _matches_policy(exported: dict[str, Any], policy: ManagedBankPolicy) -> bool:
    current = exported.get("bank")
    if not isinstance(current, dict):
        return False
    return all(current.get(field) == value for field, value in policy.bank.items())


def reconcile_once(
    policy: ManagedBankPolicy,
    request_json: RequestJson,
    *,
    deadline_seconds: float,
    bank_ids: list[str] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> ReconcileSummary:
    started = monotonic()
    if bank_ids is None:
        bank_ids = _bank_ids(request_json("GET", "/v1/default/banks", None))
    updated = 0
    unchanged = 0
    failures: list[str] = []
    unprocessed: list[str] = []
    deadline_exceeded = False
    for index, bank_id in enumerate(bank_ids):
        if monotonic() - started >= deadline_seconds:
            deadline_exceeded = True
            unprocessed.extend(bank_ids[index:])
            break
        encoded = quote(bank_id, safe="")
        try:
            exported = request_json("GET", f"/v1/default/banks/{encoded}/export", None)
            if _matches_policy(exported, policy):
                unchanged += 1
                continue
            request_json("POST", f"/v1/default/banks/{encoded}/import", policy.manifest)
            updated += 1
        except ReconcileError:
            failures.append(bank_id)
    return ReconcileSummary(
        discovered=len(bank_ids),
        updated=updated,
        unchanged=unchanged,
        failed_bank_ids=tuple(failures),
        # Advance past a slow/failing prefix before retrying its members.
        pending_bank_ids=tuple(unprocessed + failures),
        deadline_exceeded=deadline_exceeded,
    )


def _http_requester(*, base_url: str, api_key: str, timeout_seconds: float) -> RequestJson:
    endpoint = base_url.rstrip("/")

    def request_json(method: str, path: str, body: dict[str, Any] | None) -> dict[str, Any]:
        data = None if body is None else json.dumps(body, separators=(",", ":")).encode()
        request = Request(
            endpoint + path,
            data=data,
            method=method,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
                **({"Content-Type": "application/json"} if data is not None else {}),
            },
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                payload = response.read(_MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            raise ReconcileError(f"Hindsight {method} {path} returned HTTP {exc.code}") from exc
        except (URLError, OSError, HttpClientError) as exc:
            raise ReconcileError(f"Hindsight {method} {path} failed") from exc
        if len(payload) > _MAX_RESPONSE_BYTES:
            raise ReconcileError(f"Hindsight {method} {path} response is too large")
        try:
            result = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReconcileError(f"Hindsight {method} {path} returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise ReconcileError(f"Hindsight {method} {path} returned a non-object response")
        return result

    return request_json


def _log(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, **fields}, separators=(",", ":")), flush=True)


def main() -> int:
    try:
        policy = load_policy(os.environ)
        timeout_seconds = _positive_number_env(
            os.environ,
            "HINDSIGHT_BANK_RECONCILE_REQUEST_TIMEOUT_SECONDS",
            default=10,
        )
        deadline_seconds = _positive_number_env(
            os.environ,
            "HINDSIGHT_BANK_RECONCILE_DEADLINE_SECONDS",
            default=600,
        )
        retry_seconds = _positive_number_env(
            os.environ,
            "HINDSIGHT_BANK_RECONCILE_RETRY_SECONDS",
            default=60,
        )
        request_json = _http_requester(
            base_url=_required_env(os.environ, "HINDSIGHT_BANK_RECONCILE_URL"),
            api_key=_required_env(os.environ, "HINDSIGHT_BANK_RECONCILE_API_KEY"),
            timeout_seconds=timeout_seconds,
        )
    except ReconcileError as exc:
        _log("managed_bank_reconcile_invalid_configuration", error=str(exc))
        return 2

    pending_bank_ids: list[str] | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            summary = reconcile_once(
                policy,
                request_json,
                deadline_seconds=deadline_seconds,
                bank_ids=pending_bank_ids,
            )
        except ReconcileError as exc:
            _log(
                "managed_bank_reconcile_attempt_failed",
                attempt=attempt,
                error_type=type(exc).__name__,
            )
            if attempt < _MAX_ATTEMPTS:
                time.sleep(retry_seconds)
            continue
        _log(
            "managed_bank_reconcile_attempt",
            attempt=attempt,
            discovered=summary.discovered,
            updated=summary.updated,
            unchanged=summary.unchanged,
            failed=len(summary.failed_bank_ids),
            failed_bank_ids=list(summary.failed_bank_ids),
            pending=len(summary.pending_bank_ids),
            deadline_exceeded=summary.deadline_exceeded,
        )
        if summary.complete:
            return 0
        pending_bank_ids = list(summary.pending_bank_ids)
        if attempt < _MAX_ATTEMPTS:
            time.sleep(retry_seconds)
    _log(
        "managed_bank_reconcile_exhausted",
        attempts=_MAX_ATTEMPTS,
        pending=len(pending_bank_ids or []),
        pending_bank_ids=pending_bank_ids or [],
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
