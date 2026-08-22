"""8 admin tools port từ facade/mcp/adminTools.js — chạy NATIVE qua Mongo stack.

- skill_list / skill_read / skill_create / skill_update / skill_delete
- audit_recent
- mcp_list / mcp_remove

Giữ NGUYÊN schema Mongo (collections users/skills/events — KHÔNG đổi cấu trúc).
Mọi tool mutate gate admin; audit ghi vào events (format audit.js).
skill_list gộp Mongo skills + filesystem shared/skills (skillLoader parity:
đọc SKILL.md frontmatter name/enabled, source='filesystem').

Paths host (REMAP từ /shared/* trong container):
  ~/dev/hermes-openwebui-stack/shared/skills + shared/mcp
(override qua config.yaml gateway.mcp.shared_skills_dir / shared_mcp_dir).
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

from gateway.mcp_acl import DENY, get_db, get_headers_from_ctx, is_admin, resolve_user

logger = logging.getLogger(__name__)

try:
    from mcp.server.fastmcp import Context
except Exception:  # pragma: no cover
    Context = Any  # type: ignore[assignment,misc]

_DEFAULT_SHARED = Path.home() / "dev" / "hermes-openwebui-stack" / "shared"


def _shared_dirs() -> tuple[Path, Path]:
    """(skills_dir, mcp_dir) — default stack shared/, override qua config."""
    skills = _DEFAULT_SHARED / "skills"
    mcp = _DEFAULT_SHARED / "mcp"
    try:
        from gateway.mcp_tools_exec import _mcp_config

        block = _mcp_config()
        s = block.get("shared_skills_dir")
        m = block.get("shared_mcp_dir")
        if isinstance(s, str) and s:
            skills = Path(s).expanduser()
        if isinstance(m, str) and m:
            mcp = Path(m).expanduser()
    except Exception:
        pass
    return skills, mcp


# --------------------------------------------------------------------------
# skillLoader parity (filesystem skills)
# --------------------------------------------------------------------------

def _parse_skill_dir(skill_dir: Path) -> Optional[dict]:
    """Đọc SKILL.md + frontmatter (port skillLoader.parseSkillFile)."""
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.is_file():
        return None
    raw = skill_file.read_text(encoding="utf-8", errors="replace")
    name = skill_dir.name
    description = ""
    lines = raw.split("\n")
    if lines and lines[0].strip() == "---":
        for line in lines[1:]:
            if line.strip() == "---":
                break
            if line.startswith("description:"):
                description = line.split(":", 1)[1].strip()
                break
    return {
        "name": name,
        "description": description,
        "version": "fs",
        "author": "filesystem",
        "content": raw,
        "source": "filesystem",
        "path": str(skill_file),
        "triggers": [],
        "enabled": True,
    }


def _list_fs_skills(skills_dir: Path) -> list[dict]:
    if not skills_dir.is_dir():
        return []
    out = []
    for entry in sorted(skills_dir.iterdir(), key=lambda p: p.name.lower()):
        if entry.is_dir():
            skill = _parse_skill_dir(entry)
            if skill:
                out.append(skill)
    return out


def _get_fs_skill(skills_dir: Path, name: str) -> Optional[dict]:
    skill = _parse_skill_dir(skills_dir / name)
    return skill


# --------------------------------------------------------------------------
# Event/audit (format Event.js)
# --------------------------------------------------------------------------

def _log_event(db: Any, event: dict) -> None:
    """Ghi 1 event vào events (fire-and-forget, lỗi không làm chết tool)."""
    try:
        db["events"].insert_one(
            {
                "type": event.get("type"),
                "action": event.get("action"),
                "userId": event.get("userId"),
                "resourceId": event.get("resourceId"),
                "details": event.get("details") or {},
                "timestamp": event.get("timestamp"),
                "ipAddress": None,
            }
        )
    except Exception:
        logger.warning("log_event thất bại", exc_info=True)


def _now():
    import datetime

    return datetime.datetime.now(datetime.timezone.utc)


def _resolve_username_db_id(db: Any, identity: str) -> Any:
    """Resolve identity (username HOẶC owuiId) → users._id."""
    try:
        doc = db["users"].find_one(
            {"$or": [{"username": identity}, {"owuiId": identity}]}
        )
        return doc.get("_id") if doc else None
    except Exception:
        return None


def _sync_audit_writer(event: dict) -> None:
    """Audit writer cho 9 execution tools (P2) — ghi events Mongo.

    event: {type, action, tool, identity, ...} → doc format Event.js.
    identity (username string) được resolve thành userId ObjectId nếu có.
    """
    db = get_db()
    if db is None:
        return
    identity = event.get("identity")
    user_id = None
    if identity and identity != "unknown":
        user_id = _resolve_username_db_id(db, identity)
    details = {
        k: v
        for k, v in event.items()
        if k not in ("type", "action", "identity", "timestamp")
    }
    _log_event(
        db,
        {
            "type": event.get("type", "tool"),
            "action": event.get("action", "execute"),
            "userId": user_id,
            "resourceId": None,
            "details": details,
            "timestamp": event.get("timestamp") or _now(),
        },
    )


def install_audit_writer() -> None:
    """Gắn writer Mongo vào mcp_tools_exec (audit 9 execution tools)."""
    from gateway.mcp_tools_exec import set_audit_writer

    set_audit_writer(_sync_audit_writer)


# --------------------------------------------------------------------------
# 8 admin tools
# --------------------------------------------------------------------------

def register_admin_tools(mcp: Any) -> None:
    """Đăng ký 8 admin tools vào FastMCP (semantics y hệt adminTools.js)."""
    install_audit_writer()

    @mcp.tool()
    async def skill_list(ctx: Context = None) -> Any:
        """List all skills (filesystem + db)."""
        skills_dir, _ = _shared_dirs()
        db = get_db()
        db_skills = []
        if db is not None:
            try:
                db_skills = list(db["skills"].find({}).sort("name", 1))
            except Exception:
                logger.warning("skill_list: đọc Mongo skills lỗi", exc_info=True)
        fs_skills = await asyncio.to_thread(_list_fs_skills, skills_dir)
        merged: dict[str, dict] = {}
        for s in fs_skills:
            merged[s["name"]] = s
        for s in db_skills:
            merged[s["name"]] = {**s, "source": "mongodb"}
        # Giữ semantics facade: 1 text block JSON duy nhất (facade trả
        # JSON.stringify của array) — FastMCP flatten list thành multi-block.
        return json.dumps(
            [
                {"name": s["name"], "enabled": s.get("enabled", True), "source": s.get("source")}
                for s in merged.values()
            ]
        )

    @mcp.tool()
    async def skill_read(name: str, ctx: Context = None) -> Any:
        """Read a skill by name."""
        db = get_db()
        if db is not None:
            try:
                doc = db["skills"].find_one({"name": name})
                if doc:
                    return doc.get("content") or "(no content)"
            except Exception:
                pass
        skills_dir, _ = _shared_dirs()
        fs_skill = await asyncio.to_thread(_get_fs_skill, skills_dir, name)
        if fs_skill:
            return fs_skill.get("content") or "(no content)"
        return f"skill not found: {name}"

    @mcp.tool()
    async def skill_create(
        name: str, content: str, description: str = "", ctx: Context = None
    ) -> Any:
        """Create a new skill (admin)."""
        db = get_db()
        if db is None:
            return {"error": "audit database unavailable"}
        headers = get_headers_from_ctx(ctx)
        user = await asyncio.to_thread(resolve_user, headers)
        if not is_admin(user):
            return DENY
        try:
            existing = db["skills"].find_one({"name": name})
            if existing:
                return f"skill already exists: {name}"
            now = _now()
            result = db["skills"].insert_one(
                {
                    "name": name,
                    "description": description or "",
                    "version": "1.0.0",
                    "author": user.get("username") or "admin",
                    "content": content,
                    "triggers": [],
                    "enabled": True,
                    "createdAt": now,
                    "updatedAt": now,
                }
            )
            _log_event(
                db,
                {
                    "type": "skill",
                    "action": "create",
                    "userId": user.get("_id"),
                    "resourceId": str(result.inserted_id),
                    "details": {"name": name},
                    "timestamp": now,
                },
            )
            return f"created skill: {name}"
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    async def skill_update(
        id: str,
        content: Optional[str] = None,
        description: Optional[str] = None,
        enabled: Optional[bool] = None,
        ctx: Context = None,
    ) -> Any:
        """Update a skill by id (admin)."""
        db = get_db()
        if db is None:
            return {"error": "audit database unavailable"}
        headers = get_headers_from_ctx(ctx)
        user = await asyncio.to_thread(resolve_user, headers)
        if not is_admin(user):
            return DENY
        try:
            from bson import ObjectId

            oid = ObjectId(id)
        except Exception:
            return f"skill not found: {id}"
        fields = {}
        if content is not None:
            fields["content"] = content
        if description is not None:
            fields["description"] = description
        if enabled is not None:
            fields["enabled"] = enabled
        result = db["skills"].update_one(
            {"_id": oid}, {"$set": {**fields, "updatedAt": _now()}}
        )
        if result.modified_count == 0:
            return f"skill not found: {id}"
        _log_event(
            db,
            {
                "type": "skill",
                "action": "update",
                "userId": user.get("_id"),
                "resourceId": id,
                "details": fields,
                "timestamp": _now(),
            },
        )
        return f"updated skill: {id}"

    @mcp.tool()
    async def skill_delete(id: str, ctx: Context = None) -> Any:
        """Delete a skill by id (admin)."""
        db = get_db()
        if db is None:
            return {"error": "audit database unavailable"}
        headers = get_headers_from_ctx(ctx)
        user = await asyncio.to_thread(resolve_user, headers)
        if not is_admin(user):
            return DENY
        try:
            from bson import ObjectId

            oid = ObjectId(id)
        except Exception:
            return f"skill not found: {id}"
        result = db["skills"].delete_one({"_id": oid})
        if result.deleted_count == 0:
            return f"skill not found: {id}"
        _log_event(
            db,
            {
                "type": "skill",
                "action": "delete",
                "userId": user.get("_id"),
                "resourceId": id,
                "details": {},
                "timestamp": _now(),
            },
        )
        return f"deleted skill: {id}"

    @mcp.tool()
    async def audit_recent(
        limit: Optional[float] = None,
        sinceHours: Optional[float] = None,
        type: Optional[str] = None,
        ctx: Context = None,
    ) -> Any:
        """Recent audit events (admin). Params: limit (default 20, max 100),
        sinceHours (default 24), type (optional filter)."""
        db = get_db()
        if db is None:
            return {"error": "audit database unavailable"}
        headers = get_headers_from_ctx(ctx)
        user = await asyncio.to_thread(resolve_user, headers)
        if not is_admin(user):
            return DENY
        try:
            n = min(max(int(limit) if limit else 20, 1), 100)
        except (TypeError, ValueError):
            n = 20
        try:
            hours = max(int(sinceHours) if sinceHours else 24, 1)
        except (TypeError, ValueError):
            hours = 24
        import datetime

        since = _now() - datetime.timedelta(hours=hours)
        query: dict[str, Any] = {"timestamp": {"$gte": since}}
        if type:
            query["type"] = type
        rows = list(
            db["events"].find(query).sort("timestamp", -1).limit(n)
        )
        _log_event(
            db,
            {
                "type": "audit",
                "action": "recent",
                "userId": user.get("_id"),
                "resourceId": None,
                "details": {
                    "limit": n,
                    "sinceHours": hours,
                    "type": type or None,
                    "returned": len(rows),
                },
                "timestamp": _now(),
            },
        )
        def _iso_utc(ts: Any) -> Optional[str]:
            """Facade toISOString() -> 'YYYY-MM-DDTHH:MM:SS.mmmZ' (UTC + Z)."""
            if not ts:
                return None
            import datetime as _dt

            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=_dt.timezone.utc)
            return ts.astimezone(_dt.timezone.utc).isoformat().replace("+00:00", "Z")

        return json.dumps(
            [
                {
                    "id": str(e["_id"]),
                    "timestamp": _iso_utc(e.get("timestamp")),
                    "type": e.get("type"),
                    "action": e.get("action"),
                    "userId": str(e["userId"]) if e.get("userId") else None,
                    "details": e.get("details") or {},
                }
                for e in rows
            ]
        )

    @mcp.tool()
    async def mcp_list(ctx: Context = None) -> Any:
        """List MCP server config files in shared/mcp."""
        _, mcp_dir = _shared_dirs()
        try:
            if not mcp_dir.is_dir():
                return json.dumps([])
            return json.dumps(
                sorted(f.name for f in mcp_dir.iterdir() if f.suffix == ".json")
            )
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    async def mcp_remove(name: str, ctx: Context = None) -> Any:
        """Delete an MCP server config file (admin)."""
        db = get_db()
        headers = get_headers_from_ctx(ctx)
        user = await asyncio.to_thread(resolve_user, headers)
        if not is_admin(user):
            return DENY
        _, mcp_dir = _shared_dirs()
        if "/" in name or ".." in name:
            return f"config not found: {name}"
        file = mcp_dir / f"{name}.json"
        if not file.is_file():
            return f"config not found: {name}"
        try:
            file.unlink()
        except Exception as e:
            return {"error": str(e)}
        if db is not None:
            _log_event(
                db,
                {
                    "type": "mcp",
                    "action": "delete",
                    "userId": user.get("_id"),
                    "resourceId": None,
                    "details": {"name": name},
                    "timestamp": _now(),
                },
            )
        return f"removed mcp config: {name}"
