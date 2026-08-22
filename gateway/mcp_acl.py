"""ACL cho MCP native — port 1:1 từ facade/mcp/acl.js.

Resolve user từ headers MCP request qua Mongo stack (FORK-1a):
  - X-Hermes-User-Id (uuid OWUI) → users.owuiId — uuid lạ = unknown
    (KHÔNG fallback username — chặn leo quyền, như facade).
  - Fallback X-Hermes-User (username) cho client cũ (curl, mcp-bridge).
  - Không header → unknown (read-only).

Native KHÔNG có multi-token (API_SERVER_KEY chỉ xác thực kết nối — xem
PLAN-MCP-NATIVE HỆ QUẢ KIẾN TRÚC): role hiệu lực = role từ header + Mongo.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_mongo_client: Any = None
_mongo_db: Any = None
_mongo_failed: bool = False

DENY = "ERROR: admin permission required"

# Danh sách headers identity (starlette Headers — lowercase keys).
_HEADER_USER_ID = "x-hermes-user-id"
_HEADER_USER = "x-hermes-user"


def _get_mongo_uri() -> Optional[str]:
    """Secret MCP_MONGO_URI từ ~/.hermes/.env (KHÔNG in ra log)."""
    try:
        from hermes_cli.env_loader import load_hermes_dotenv

        load_hermes_dotenv()
    except Exception:
        pass
    try:
        from agent.secret_scope import get_secret

        return get_secret("MCP_MONGO_URI")
    except Exception:
        import os

        return os.getenv("MCP_MONGO_URI")


def get_db() -> Any:
    """Lazy Mongo client (sync pymongo — gọi qua asyncio.to_thread ở call site).

    Trả None khi Mongo không khả dụng (fail-closed: resolveUser → unknown,
    admin tools → error).
    """
    global _mongo_client, _mongo_db, _mongo_failed
    if _mongo_db is not None:
        return _mongo_db
    if _mongo_failed:
        return None
    try:
        import pymongo

        uri = _get_mongo_uri()
        if not uri:
            _mongo_failed = True
            logger.error("mcp_acl: MCP_MONGO_URI chưa cấu hình")
            return None
        _mongo_client = pymongo.MongoClient(
            uri, serverSelectionTimeoutMS=2000, connect=False
        )
        _mongo_db = _mongo_client["hermes"]
        # Ping sớm để fail nhanh lần đầu.
        _mongo_db.command("ping")
        return _mongo_db
    except Exception:
        _mongo_failed = True
        logger.error("mcp_acl: Mongo không khả dụng", exc_info=True)
        return None


def reset_mongo() -> None:
    """Reset client (dùng khi test / Mongo restart)."""
    global _mongo_client, _mongo_db, _mongo_failed
    _mongo_client = None
    _mongo_db = None
    _mongo_failed = False


def _unknown_user() -> dict:
    return {"_id": None, "username": "unknown", "role": "unknown"}


def resolve_user(headers: Any) -> dict:
    """Port acl.js resolveUser: headers → user doc từ Mongo.users.

    Sync (pymongo) — gọi qua asyncio.to_thread từ tools.
    """
    if headers is None:
        return _unknown_user()
    owui_id = headers.get(_HEADER_USER_ID)
    if owui_id:
        db = get_db()
        if db is not None:
            user = db["users"].find_one({"owuiId": owui_id})
            if user:
                return user
        return _unknown_user()
    username = headers.get(_HEADER_USER)
    if username:
        db = get_db()
        if db is not None:
            user = db["users"].find_one({"username": username})
            if user:
                return user
    return _unknown_user()


def is_admin(user: Any) -> bool:
    return bool(user and user.get("role") == "admin")


def most_restrictive_role(a: str, b: str) -> str:
    """Port acl.js mostRestrictiveRole ('admin' > 'user' > 'unknown')."""
    rank = {"admin": 3, "user": 2, "unknown": 1}
    ra = rank.get(a, 1)
    rb = rank.get(b, 1)
    return a if ra <= rb else b


def get_headers_from_ctx(ctx: Any) -> Any:
    """Headers của MCP request từ FastMCP Context (None nếu không có)."""
    try:
        req = getattr(getattr(ctx, "_request_context", None), "request", None)
        return getattr(req, "headers", None)
    except Exception:
        return None
