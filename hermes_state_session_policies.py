"""Session policy override mixin for SessionDB."""

import json
import sqlite3
from typing import Any, Optional


class SessionPolicyMixin:
    def set_session_policy_override(
        self, session_id: str, key: str, override: bool | None
    ) -> None:
        """Atomically set or clear one host-validated session policy key."""
        if override is not None and not isinstance(override, bool):
            raise ValueError(
                "session policy override must be a literal bool or None"
            )

        def _do(conn):
            row = conn.execute(
                "SELECT model_config FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"Session row not found: {session_id}")

            raw = row["model_config"] if isinstance(row, sqlite3.Row) else row[0]
            if raw is None:
                config: dict[str, Any] = {}
            elif isinstance(raw, dict):
                config = dict(raw)
            elif isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"Session {session_id} model_config is malformed JSON"
                    ) from exc
                if not isinstance(parsed, dict):
                    raise ValueError(
                        f"Session {session_id} model_config must be a JSON object"
                    )
                config = parsed
            else:
                raise ValueError(
                    f"Session {session_id} model_config must be a JSON object"
                )

            if override is None:
                config.pop(key, None)
            else:
                config[key] = override
            conn.execute(
                "UPDATE sessions SET model_config = ? WHERE id = ?",
                (json.dumps(config), session_id),
            )

        self._execute_write(_do)

    @staticmethod
    def session_policy_override(
        session_meta: Optional[dict[str, Any]], key: str,
    ) -> bool | None:
        """Read a literal bool policy from an existing row; malformed values inherit."""
        if not isinstance(session_meta, dict):
            return None
        raw = session_meta.get("model_config")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (TypeError, ValueError, json.JSONDecodeError):
                return None
        if not isinstance(raw, dict):
            return None
        value = raw.get(key)
        return value if isinstance(value, bool) else None
