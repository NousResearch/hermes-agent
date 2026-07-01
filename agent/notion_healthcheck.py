"""Notion healthcheck probe and narrow fail-closed guard."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Mapping

from agent.runtime_evidence import current_runtime_evidence, update_runtime_evidence
from hermes_cli.config import get_env_value

_NOTION_API_BASE_URL = "https://api.notion.com/v1"
_NOTION_API_VERSION = os.getenv("NOTION_API_VERSION", "2022-06-28")
_NOTION_HEALTHCHECK_TTL_SEC = 300.0


class NotionHealthcheckError(RuntimeError):
    """Base error for explicit Notion route failures."""


class NotionRouteUnavailableError(NotionHealthcheckError):
    """Raised when the Notion route/tool is unavailable."""


class NotionAuthError(NotionHealthcheckError):
    """Raised when Notion auth/config is missing or rejected."""


class NotionInvalidHealthResultError(NotionHealthcheckError):
    """Raised when the health response is empty or malformed."""


FetchResult = Mapping[str, Any]
Fetcher = Callable[[str, str], FetchResult]


def probe_notion_healthcheck(
    *,
    fetcher: Fetcher | None = None,
    notion_api_key: str | None = None,
    notion_api_base_url: str | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Probe the live Notion route and persist the result in runtime evidence."""
    checked_at = float(now if now is not None else time.time())
    latest_request = str((current_runtime_evidence() or {}).get("latest_user_request") or "").strip() or None
    token = _resolve_notion_api_key(notion_api_key)
    base_url = _normalize_base_url(notion_api_base_url)

    if not token:
        return _record_notion_healthcheck_result(
            {
                "ok": False,
                "reason": "missing_config",
                "message": (
                    "Blocked Notion healthcheck: NOTION_API_KEY is missing or empty. "
                    "Set the integration token first, then retry the healthcheck."
                ),
                "checked_at": checked_at,
                "checked_request": latest_request,
                "token_present": False,
                "route_url": f"{base_url}/users/me",
            }
        )

    fetch = fetcher or _default_notion_users_me_fetcher
    try:
        fetched = fetch(f"{base_url}/users/me", token)
    except NotionHealthcheckError as exc:
        return _record_notion_healthcheck_result(
            _failure_result(
                checked_at=checked_at,
                latest_request=latest_request,
                token_present=True,
                route_url=f"{base_url}/users/me",
                reason=_reason_for_exception(exc),
                message=str(exc),
            )
        )
    except urllib.error.HTTPError as exc:
        return _record_notion_healthcheck_result(
            _failure_result(
                checked_at=checked_at,
                latest_request=latest_request,
                token_present=True,
                route_url=f"{base_url}/users/me",
                reason=_reason_for_http_status(exc.code),
                message=_message_for_http_status(exc.code),
                status_code=exc.code,
            )
        )
    except urllib.error.URLError as exc:
        return _record_notion_healthcheck_result(
            _failure_result(
                checked_at=checked_at,
                latest_request=latest_request,
                token_present=True,
                route_url=f"{base_url}/users/me",
                reason="missing_tool",
                message=(
                    "Blocked Notion healthcheck: the Notion route/tool is unavailable "
                    f"({exc.reason}). Verify the configured route and retry."
                ),
            )
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        return _record_notion_healthcheck_result(
            _failure_result(
                checked_at=checked_at,
                latest_request=latest_request,
                token_present=True,
                route_url=f"{base_url}/users/me",
                reason="missing_tool",
                message=(
                    "Blocked Notion healthcheck: the Notion route/tool is unavailable. "
                    f"Detail: {exc}"
                ),
            )
        )

    parsed = _parse_health_result(fetched)
    if parsed is None:
        return _record_notion_healthcheck_result(
            _failure_result(
                checked_at=checked_at,
                latest_request=latest_request,
                token_present=True,
                route_url=f"{base_url}/users/me",
                reason="invalid_result",
                message=(
                    "Blocked Notion healthcheck: the Notion /v1/users/me response was "
                    "empty or invalid. Refresh the route/tool and retry."
                ),
            )
        )

    state = {
        "ok": True,
        "reason": "ok",
        "message": "Notion healthcheck passed.",
        "checked_at": checked_at,
        "checked_request": latest_request,
        "token_present": True,
        "route_url": f"{base_url}/users/me",
        "status_code": parsed.get("status_code", 200),
        "workspace": parsed.get("workspace"),
        "bot_name": parsed.get("bot_name"),
        "bot_id": parsed.get("bot_id"),
        "user_type": parsed.get("user_type"),
        "raw_object_type": parsed.get("raw_object_type"),
    }
    return _record_notion_healthcheck_result(state)


def require_fresh_notion_healthcheck(
    *,
    action_label: str,
    user_task: str | None = None,
    evidence: Mapping[str, Any] | None = None,
    now: float | None = None,
) -> str | None:
    """Block Notion-dependent actions unless a fresh, successful check exists."""
    if not _looks_notion_dependent(action_label, user_task, evidence):
        return None

    evidence_map = dict(evidence or current_runtime_evidence() or {})
    state = dict(evidence_map.get("notion_healthcheck_state") or {})
    if _is_explicit_healthcheck_request(action_label, user_task) and not state:
        probe_notion_healthcheck(now=now)
        evidence_map = dict(current_runtime_evidence() or {})
        state = dict(evidence_map.get("notion_healthcheck_state") or {})
    if not state:
        return _missing_healthcheck_block(action_label=action_label, user_task=user_task)

    if not state.get("ok"):
        return state.get("message") or _missing_healthcheck_block(
            action_label=action_label,
            user_task=user_task,
        )

    checked_at = _coerce_float(state.get("checked_at"))
    if checked_at is None:
        return _stale_healthcheck_block(
            action_label=action_label,
            user_task=user_task,
            reason="missing timestamp",
        )

    now_value = float(now if now is not None else time.time())
    age = now_value - checked_at
    if age > _NOTION_HEALTHCHECK_TTL_SEC:
        return _stale_healthcheck_block(
            action_label=action_label,
            user_task=user_task,
            reason=f"age {age:.0f}s exceeds {_NOTION_HEALTHCHECK_TTL_SEC:.0f}s",
        )

    current_request = normalize_notion_text(
        evidence_map.get("latest_user_request") or user_task or ""
    )
    checked_request = normalize_notion_text(state.get("checked_request") or "")
    if checked_request and current_request and checked_request != current_request:
        return _stale_healthcheck_block(
            action_label=action_label,
            user_task=user_task,
            reason="healthcheck ran for a different user request",
        )

    return None


def normalize_notion_text(text: Any) -> str:
    return " ".join(str(text or "").split()).strip().lower()


def _looks_notion_dependent(
    action_label: str,
    user_task: str | None,
    evidence: Mapping[str, Any] | None,
) -> bool:
    text_blobs = [
        action_label,
        user_task or "",
        str((evidence or {}).get("latest_user_request") or ""),
        str((evidence or {}).get("active_user_task") or ""),
    ]
    normalized = " ".join(normalize_notion_text(blob) for blob in text_blobs if blob)
    if "notion" not in normalized:
        return False
    if action_label in {"memory", "session_search", "todo", "skill_view", "skill_manage"}:
        return True
    if action_label in {"terminal", "execute_code"} and any(
        keyword in normalized
        for keyword in ("health", "healthcheck", "health check", "check connection", "users/me", "route", "tool", "auth", "token")
    ):
        return True
    return False


def _is_explicit_healthcheck_request(action_label: str, user_task: str | None) -> bool:
    normalized = " ".join(
        normalize_notion_text(blob)
        for blob in (action_label, user_task or "")
        if blob
    )
    return "notion" in normalized and any(keyword in normalized for keyword in ("healthcheck", "health check", "check connection", "connection"))


def _missing_healthcheck_block(*, action_label: str, user_task: str | None) -> str:
    request_text = str(user_task or "").strip() or "the current task"
    return (
        f"Blocked {action_label}: Notion healthcheck is required before relying on Notion-backed work.\n"
        f"Latest real user request: {request_text}\n"
        "Run the Notion healthcheck route first, then retry once it has passed."
    )


def _stale_healthcheck_block(*, action_label: str, user_task: str | None, reason: str) -> str:
    request_text = str(user_task or "").strip() or "the current task"
    evidence = current_runtime_evidence() or {}
    state = dict(evidence.get("notion_healthcheck_state") or {})
    return (
        f"Blocked {action_label}: Notion healthcheck evidence is stale or insufficient ({reason}).\n"
        f"Latest real user request: {request_text}\n"
        f"Checked request: {state.get('checked_request') or 'unknown'}\n"
        f"Checked at: {state.get('checked_at') or 'unknown'}\n"
        "Refresh the Notion healthcheck route first, then retry with fresh evidence."
    )


def _reason_for_exception(exc: NotionHealthcheckError) -> str:
    if isinstance(exc, NotionRouteUnavailableError):
        return "missing_tool"
    if isinstance(exc, NotionAuthError):
        return "auth_config"
    if isinstance(exc, NotionInvalidHealthResultError):
        return "invalid_result"
    return "missing_tool"


def _reason_for_http_status(status_code: int | None) -> str:
    if status_code in {401, 403}:
        return "auth_config"
    if status_code in {404, 405, 501}:
        return "missing_tool"
    return "missing_tool"


def _message_for_http_status(status_code: int | None) -> str:
    if status_code in {401, 403}:
        return (
            "Blocked Notion healthcheck: Notion authentication/configuration is not "
            "valid. Re-check the integration token and permissions."
        )
    if status_code in {404, 405, 501}:
        return (
            "Blocked Notion healthcheck: the Notion route/tool is unavailable. "
            "Verify the configured route and retry."
        )
    return "Blocked Notion healthcheck: Notion route returned an unexpected status."


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _failure_result(
    *,
    checked_at: float,
    latest_request: str | None,
    token_present: bool,
    route_url: str,
    reason: str,
    message: str,
    status_code: int | None = None,
) -> dict[str, Any]:
    result = {
        "ok": False,
        "reason": reason,
        "message": message,
        "checked_at": checked_at,
        "checked_request": latest_request,
        "token_present": token_present,
        "route_url": route_url,
    }
    if status_code is not None:
        result["status_code"] = status_code
    return result


def _record_notion_healthcheck_result(result: Mapping[str, Any]) -> dict[str, Any]:
    state = dict(result)
    update_runtime_evidence(
        notion_healthcheck_state=state,
        notion_healthcheck_checked_at=state.get("checked_at"),
        notion_healthcheck_reason=state.get("reason"),
        notion_healthcheck_ok=state.get("ok"),
        notion_healthcheck_route_url=state.get("route_url"),
        notion_healthcheck_checked_request=state.get("checked_request"),
    )
    return state


def _resolve_notion_api_key(notion_api_key: str | None = None) -> str:
    if notion_api_key is not None:
        return str(notion_api_key).strip()
    return str(get_env_value("NOTION_API_KEY") or os.getenv("NOTION_API_KEY") or "").strip()


def _normalize_base_url(notion_api_base_url: str | None = None) -> str:
    base_url = str(
        notion_api_base_url
        or get_env_value("NOTION_API_BASE_URL")
        or os.getenv("NOTION_API_BASE_URL")
        or _NOTION_API_BASE_URL
    ).strip()
    return base_url.rstrip("/")


def _default_notion_users_me_fetcher(url: str, token: str) -> FetchResult:
    request = urllib.request.Request(
        url,
        method="GET",
        headers={
            "Authorization": f"Bearer {token}",
            "Notion-Version": _NOTION_API_VERSION,
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=8) as response:
        raw = response.read().decode("utf-8", errors="replace").strip()
        if not raw:
            raise NotionInvalidHealthResultError("Notion /v1/users/me returned an empty body")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise NotionInvalidHealthResultError("Notion /v1/users/me returned invalid JSON") from exc
        return {
            "status_code": getattr(response, "status", 200),
            "body": body,
        }


def _parse_health_result(result: FetchResult) -> dict[str, Any] | None:
    if not isinstance(result, Mapping):
        return None
    status_code = result.get("status_code")
    body = result.get("body")
    if not isinstance(body, Mapping):
        return None
    if not body:
        return None
    raw_object_type = body.get("object")
    user_id = body.get("id")
    user_type = body.get("type")
    if not raw_object_type or not user_id or not user_type:
        return None
    bot = body.get("bot") if isinstance(body.get("bot"), Mapping) else {}
    workspace = body.get("workspace_name") or body.get("workspace")
    if workspace is None and isinstance(bot, Mapping):
        workspace = bot.get("workspace_name") or bot.get("workspace_name_override")
    bot_name = body.get("name") or (bot.get("name") if isinstance(bot, Mapping) else None)
    bot_id = bot.get("owner_id") if isinstance(bot, Mapping) else None
    return {
        "status_code": status_code if isinstance(status_code, int) else 200,
        "raw_object_type": str(raw_object_type),
        "workspace": workspace,
        "bot_name": bot_name,
        "bot_id": bot_id,
        "user_type": str(user_type),
    }
