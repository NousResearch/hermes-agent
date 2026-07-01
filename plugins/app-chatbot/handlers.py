"""Hermes tool handlers for CRWD support tools."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from . import queries
from ._utils import default_user_id

logger = logging.getLogger(__name__)


def _resolve_user_id(args: Dict[str, Any]) -> str:
    return str(args.get("user_id") or default_user_id() or "").strip()


def _handle_result(result: Dict[str, Any]) -> str:
    return json.dumps(result, default=str)


def _handle_error(exc: Exception) -> str:
    logger.warning("app-chatbot tool error: %s", exc)
    return json.dumps({"success": False, "error": str(exc)})


def get_active_gigs(args: Dict[str, Any], **_: Any) -> str:
    try:
        user_id = _resolve_user_id(args)
        page = args.get("page", 1)
        limit = args.get("limit", 10)
        return _handle_result(queries.get_active_gigs(user_id, page=page, limit=limit))
    except Exception as exc:
        return _handle_error(exc)


def get_user_profile_by_id(args: Dict[str, Any], **_: Any) -> str:
    try:
        user_id = str(args.get("user_id") or "").strip()
        if not user_id:
            return json.dumps({"success": False, "error": "user_id is required"})
        return _handle_result(queries.get_user_profile_by_id(user_id))
    except Exception as exc:
        return _handle_error(exc)


def get_gig_details(args: Dict[str, Any], **_: Any) -> str:
    try:
        gig_id = args.get("gig_id")
        name = args.get("name")
        if not gig_id and not name:
            return json.dumps({"success": False, "error": "Provide gig_id or name"})
        return _handle_result(queries.get_gig_details(gig_id=gig_id, name=name))
    except Exception as exc:
        return _handle_error(exc)


def get_user_gig_history(args: Dict[str, Any], **_: Any) -> str:
    try:
        user_id = _resolve_user_id(args)
        limit = args.get("limit", 50)
        return _handle_result(queries.get_user_gig_history(user_id, limit=limit))
    except Exception as exc:
        return _handle_error(exc)


def get_user_joined_gigs(args: Dict[str, Any], **_: Any) -> str:
    try:
        user_id = _resolve_user_id(args)
        limit = args.get("limit", 50)
        return _handle_result(queries.get_user_joined_gigs(user_id, limit=limit))
    except Exception as exc:
        return _handle_error(exc)
