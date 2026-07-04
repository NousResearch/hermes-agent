"""Chatwoot labels tool — list, bootstrap, and assign conversation labels.

Self-contained: resolves the current conversation from gateway session context
(``HERMES_SESSION_PLATFORM`` / ``HERMES_SESSION_CHAT_ID``) and calls Chatwoot's
Application API directly. Graceful JSON responses — never raises into the agent
loop.

Auth prefers ``CHATWOOT_AGENT_TOKEN`` (label CRUD often requires a user/agent
token); falls back to ``CHATWOOT_TOKEN``.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from plugins.platforms.chatwoot.labels import PREDEFINED_LABELS, PREDEFINED_LABEL_TITLES

logger = logging.getLogger(__name__)

_TIMEOUT_S = 8


# --- Availability ---


def _agent_token() -> str:
    return (os.getenv("CHATWOOT_AGENT_TOKEN", "") or os.getenv("CHATWOOT_TOKEN", "")).strip()


def _base_url() -> str:
    return os.getenv("CHATWOOT_BASE_URL", "").strip().rstrip("/")


def check_chatwoot_labels_requirements() -> bool:
    """Available when Chatwoot base URL and a usable token are configured."""
    return bool(_base_url() and _agent_token())


# --- Conversation resolution ---


def _resolve_conversation(
    account_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(account_id, conversation_id)`` for the current or override session."""
    acct = (account_id or "").strip()
    conv = (conversation_id or "").strip()
    if acct and conv:
        return acct, conv

    try:
        from gateway.session_context import get_session_env
    except Exception:  # pragma: no cover
        return None, None

    platform = (get_session_env("HERMES_SESSION_PLATFORM", "") or "").strip().lower()
    if platform and platform != "chatwoot":
        return None, None

    chat_id = (get_session_env("HERMES_SESSION_CHAT_ID", "") or "").strip()
    if not chat_id:
        return None, None

    default_account = os.getenv("CHATWOOT_ACCOUNT_ID", "").strip()
    if ":" in chat_id:
        parsed_acct, _, parsed_conv = chat_id.partition(":")
        parsed_acct = parsed_acct.strip() or default_account
        parsed_conv = parsed_conv.strip()
    else:
        parsed_acct, parsed_conv = default_account, chat_id
    if not parsed_acct or not parsed_conv:
        return None, None
    return parsed_acct, parsed_conv


# --- HTTP helpers ---


def _api_request(
    method: str,
    path: str,
    body: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Any, str]:
    """Call Chatwoot Application API. Returns ``(ok, parsed_json_or_none, error)``."""
    base = _base_url()
    token = _agent_token()
    if not base or not token:
        return False, None, "Chatwoot not configured"

    url = f"{base}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Content-Type": "application/json",
            "api_access_token": token,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            raw = resp.read().decode("utf-8")
            parsed = json.loads(raw) if raw.strip() else None
            if 200 <= resp.status < 300:
                return True, parsed, ""
            return False, parsed, f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        err_body = ""
        try:
            err_body = exc.read().decode("utf-8")
        except Exception:
            pass
        parsed = None
        if err_body.strip():
            try:
                parsed = json.loads(err_body)
            except json.JSONDecodeError:
                parsed = {"message": err_body}
        return False, parsed, f"HTTP {exc.code}"
    except Exception as exc:
        return False, None, str(exc)


def _normalize_label(title: str) -> str:
    return str(title or "").strip().lower()


def _extract_account_label_titles(payload: Any) -> List[str]:
    """Parse GET /accounts/{id}/labels response into lowercase title strings."""
    if not isinstance(payload, dict):
        return []
    items = payload.get("payload")
    if not isinstance(items, list):
        return []
    titles: List[str] = []
    for item in items:
        if isinstance(item, dict):
            title = _normalize_label(item.get("title", ""))
        elif isinstance(item, str):
            title = _normalize_label(item)
        else:
            continue
        if title:
            titles.append(title)
    return titles


def _extract_conversation_labels(payload: Any) -> List[str]:
    """Parse GET conversation labels response."""
    if not isinstance(payload, dict):
        return []
    items = payload.get("payload")
    if not isinstance(items, list):
        return []
    return [_normalize_label(x) for x in items if _normalize_label(str(x))]


def _merge_labels(existing: List[str], new: List[str]) -> List[str]:
    """Union preserving order: existing first, then new."""
    seen: set[str] = set()
    merged: List[str] = []
    for title in existing + new:
        norm = _normalize_label(title)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        merged.append(norm)
    return merged


# --- Actions ---


def _get_all_labels(account_id: str) -> Dict[str, Any]:
    ok, data, err = _api_request("GET", f"/api/v1/accounts/{account_id}/labels")
    if not ok:
        return {"success": False, "labels": [], "error": err, "detail": data}
    titles = _extract_account_label_titles(data)
    return {"success": True, "labels": titles, "payload": data, "error": None}


def _create_labels_if_not_exists(account_id: str) -> Dict[str, Any]:
    ok, data, err = _api_request("GET", f"/api/v1/accounts/{account_id}/labels")
    if not ok:
        return {"success": False, "created": [], "existing": [], "error": err, "detail": data}

    existing = set(_extract_account_label_titles(data))
    created: List[str] = []
    errors: List[str] = []

    for entry in PREDEFINED_LABELS:
        title = _normalize_label(entry.get("title", ""))
        if not title or title in existing:
            continue
        body = {
            "title": title,
            "description": str(entry.get("description", "")),
            "color": str(entry.get("color", "#1f93ff")),
            "show_on_sidebar": True,
        }
        ok_post, post_data, post_err = _api_request(
            "POST",
            f"/api/v1/accounts/{account_id}/labels",
            body,
        )
        if ok_post:
            created.append(title)
            existing.add(title)
        else:
            errors.append(f"{title}: {post_err}")
            logger.warning("[chatwoot_labels] create %s failed: %s", title, post_err)

    return {
        "success": not errors,
        "created": created,
        "existing": sorted(existing),
        "errors": errors or None,
        "error": errors[0] if errors else None,
    }


def _assign_labels(
    account_id: str,
    conversation_id: str,
    labels: List[str],
    replace: bool,
) -> Dict[str, Any]:
    normalized = [_normalize_label(x) for x in labels if _normalize_label(x)]
    if not normalized:
        return {
            "success": False,
            "labels": [],
            "error": "labels must be a non-empty array of label title strings",
        }

    bootstrap = _create_labels_if_not_exists(account_id)
    if not bootstrap.get("success") and bootstrap.get("created") == []:
        # Allow proceed if labels already existed; fail only on hard GET failure.
        if bootstrap.get("error") and not bootstrap.get("existing"):
            return {
                "success": False,
                "labels": [],
                "error": bootstrap.get("error"),
                "detail": bootstrap,
            }

    if replace:
        final_labels = normalized
    else:
        ok_get, conv_data, err_get = _api_request(
            "GET",
            f"/api/v1/accounts/{account_id}/conversations/{conversation_id}/labels",
        )
        if not ok_get:
            return {
                "success": False,
                "labels": [],
                "error": err_get,
                "detail": conv_data,
            }
        current = _extract_conversation_labels(conv_data)
        final_labels = _merge_labels(current, normalized)

    ok_post, post_data, err_post = _api_request(
        "POST",
        f"/api/v1/accounts/{account_id}/conversations/{conversation_id}/labels",
        {"labels": final_labels},
    )
    if not ok_post:
        return {
            "success": False,
            "labels": final_labels,
            "error": err_post,
            "detail": post_data,
        }

    applied = _extract_conversation_labels(post_data) if post_data else final_labels
    return {
        "success": True,
        "labels": applied or final_labels,
        "replaced": replace,
        "error": None,
    }


# --- Handler ---


def chatwoot_labels_tool(args: Dict[str, Any], **_kw: Any) -> str:
    action = str(args.get("action", "") or "").strip().lower()
    result: Dict[str, Any] = {"_type": "chatwoot_labels", "action": action}

    if not check_chatwoot_labels_requirements():
        result.update(
            {
                "success": False,
                "reason": "Chatwoot not configured; skip label operations.",
                "error": None,
            }
        )
        return json.dumps(result, ensure_ascii=False)

    account_id, conversation_id = _resolve_conversation(
        args.get("account_id"),
        args.get("conversation_id"),
    )

    if action == "get_all_labels":
        if not account_id:
            account_id = os.getenv("CHATWOOT_ACCOUNT_ID", "").strip()
        if not account_id:
            result.update(
                {
                    "success": False,
                    "reason": "No account_id; set CHATWOOT_ACCOUNT_ID or use a Chatwoot session.",
                    "error": None,
                }
            )
            return json.dumps(result, ensure_ascii=False)
        out = _get_all_labels(account_id)
        result.update(out)
        return json.dumps(result, ensure_ascii=False)

    if action == "create_labels_if_not_exists":
        if not account_id:
            account_id = os.getenv("CHATWOOT_ACCOUNT_ID", "").strip()
        if not account_id:
            result.update(
                {
                    "success": False,
                    "reason": "No account_id; set CHATWOOT_ACCOUNT_ID or use a Chatwoot session.",
                    "error": None,
                }
            )
            return json.dumps(result, ensure_ascii=False)
        out = _create_labels_if_not_exists(account_id)
        result.update(out)
        return json.dumps(result, ensure_ascii=False)

    if action == "assign_labels":
        if not account_id or not conversation_id:
            result.update(
                {
                    "success": False,
                    "reason": "No current Chatwoot conversation; skip label assignment.",
                    "error": None,
                }
            )
            return json.dumps(result, ensure_ascii=False)

        raw_labels = args.get("labels")
        if not isinstance(raw_labels, list):
            result.update(
                {
                    "success": False,
                    "error": "labels must be an array of label title strings",
                }
            )
            return json.dumps(result, ensure_ascii=False)

        replace = bool(args.get("replace", False))
        out = _assign_labels(account_id, conversation_id, raw_labels, replace)
        result.update(out)
        return json.dumps(result, ensure_ascii=False)

    result.update(
        {
            "success": False,
            "error": (
                "action must be one of: get_all_labels, "
                "create_labels_if_not_exists, assign_labels"
            ),
        }
    )
    return json.dumps(result, ensure_ascii=False)


# --- Schema (for plugin registration) ---

CHATWOOT_LABELS_SCHEMA = {
    "name": "chatwoot_labels",
    "description": (
        "Manage Chatwoot conversation labels on the current support conversation. "
        "Actions: get_all_labels (list account labels), create_labels_if_not_exists "
        "(bootstrap predefined labels into the account), assign_labels (apply one "
        "or more labels to the current conversation; merges by default). Safe to "
        "call outside Chatwoot — it no-ops gracefully."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "get_all_labels",
                    "create_labels_if_not_exists",
                    "assign_labels",
                ],
                "description": "Which label operation to perform.",
            },
            "labels": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Label title strings to assign (required for assign_labels). "
                    f"Predefined titles include: {', '.join(sorted(PREDEFINED_LABEL_TITLES))}."
                ),
            },
            "replace": {
                "type": "boolean",
                "description": (
                    "For assign_labels only. false (default) merges with existing "
                    "conversation labels; true replaces the full set."
                ),
            },
            "account_id": {
                "type": "string",
                "description": "Optional account id override (defaults to session or CHATWOOT_ACCOUNT_ID).",
            },
            "conversation_id": {
                "type": "string",
                "description": "Optional conversation id override (defaults to current session).",
            },
        },
        "required": ["action"],
    },
}


def register_labels_tool(ctx) -> None:
    """Register chatwoot_labels with the plugin context."""
    ctx.register_tool(
        name="chatwoot_labels",
        toolset="chatwoot",
        schema=CHATWOOT_LABELS_SCHEMA,
        handler=chatwoot_labels_tool,
        check_fn=check_chatwoot_labels_requirements,
        requires_env=["CHATWOOT_BASE_URL"],
        emoji="🏷️",
    )
