"""Identity resolution and persistence for Hermes' RetainDB integration."""

from __future__ import annotations

import getpass
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from retaindb_integration.client import RetainDBClientConfig


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_fallback(value: str | None) -> str:
    text = (value or "").strip()
    if not text:
        return "hermes-user"
    cleaned = []
    for char in text:
        if char.isalnum() or char in {"-", "_", ".", ":"}:
            cleaned.append(char)
        elif char.isspace():
            cleaned.append("-")
    collapsed = "".join(cleaned).strip("-")
    while "--" in collapsed:
        collapsed = collapsed.replace("--", "-")
    return collapsed or "hermes-user"


@dataclass
class ResolvedRetainDBIdentity:
    user_id: str
    session_id: str
    agent_id: str
    project: str
    source: str
    peer_name: str | None = None
    platform: str | None = None
    chat_id: str | None = None


class RetainDBIdentityResolver:
    """Resolve and persist a stable user identity for RetainDB lookups."""

    def __init__(self, config: RetainDBClientConfig):
        self.config = config
        self.cache_path = config.identity_cache_path

    def _read_cache(self) -> dict[str, Any]:
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_cache(self, payload: dict[str, Any]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def resolve(
        self,
        *,
        session_id: str,
        runtime_identity: dict[str, Any] | None = None,
    ) -> ResolvedRetainDBIdentity:
        runtime = dict(runtime_identity or {})
        cached = self._read_cache()

        explicit_override = (
            str(runtime.get("user_id_override") or "").strip()
            or str(self.config.user_id_override or "").strip()
        )
        platform_user_id = str(
            runtime.get("platform_user_id")
            or runtime.get("user_id")
            or ""
        ).strip()
        peer_name = str(
            runtime.get("peer_name")
            or runtime.get("user_name")
            or ""
        ).strip() or None
        os_username = str(runtime.get("os_username") or getpass.getuser() or "").strip()

        if explicit_override:
            resolved_user_id = explicit_override
            source = "config_override"
        elif platform_user_id:
            resolved_user_id = platform_user_id
            source = "platform"
        elif peer_name:
            if cached.get("source") in {"peer", "os"} and cached.get("user_id"):
                resolved_user_id = str(cached["user_id"])
            else:
                resolved_user_id = _sanitize_fallback(peer_name)
            source = "peer"
        else:
            if cached.get("source") in {"peer", "os"} and cached.get("user_id"):
                resolved_user_id = str(cached["user_id"])
            else:
                resolved_user_id = _sanitize_fallback(os_username)
            source = "os"

        result = ResolvedRetainDBIdentity(
            user_id=resolved_user_id,
            session_id=session_id,
            agent_id=str(runtime.get("agent_id") or self.config.agent_id or "hermes"),
            project=str(runtime.get("project") or self.config.project or "").strip(),
            source=source,
            peer_name=peer_name,
            platform=str(runtime.get("platform") or "").strip() or None,
            chat_id=str(runtime.get("chat_id") or "").strip() or None,
        )

        self._write_cache(
            {
                "user_id": result.user_id,
                "session_id": result.session_id,
                "agent_id": result.agent_id,
                "project": result.project,
                "source": result.source,
                "peer_name": result.peer_name,
                "platform": result.platform,
                "chat_id": result.chat_id,
                "updated_at": _now_iso(),
            }
        )
        return result
