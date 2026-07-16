"""Durable, user-scoped profile for the Chief.

This store intentionally accepts only stable profile fields. Temporary task
state, logs, credentials, and technical specialist memory are rejected.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from hermes_constants import get_hermes_home


_ALLOWED_FACT_TYPES = frozenset({"preference", "goal", "constraint", "decision", "communication"})
_FORBIDDEN_KEYS = frozenset({"password", "token", "secret", "api_key", "private_key"})


class ChiefFact(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str
    value: str = Field(min_length=1, max_length=4000)
    source: str = "user-confirmed"
    confidence: float = Field(default=1.0, ge=0, le=1)

    def validate_policy(self) -> None:
        if self.type not in _ALLOWED_FACT_TYPES:
            raise ValueError(f"unsupported Chief fact type: {self.type}")
        lowered = self.value.lower()
        if any(key in lowered for key in _FORBIDDEN_KEYS):
            raise ValueError("secrets and credentials are not allowed in the Chief profile")


class ChiefProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: str
    display_name: str = "Chief"
    facts: tuple[ChiefFact, ...] = ()
    revision: int = 1

    def prompt_block(self) -> str:
        if not self.facts:
            return ""
        lines = ["# Chief profile", f"Address the user as {self.display_name}."]
        lines.extend(f"- {fact.type}: {fact.value}" for fact in self.facts)
        lines.append("These facts guide communication and planning; they do not grant technical permissions.")
        return "\n".join(lines)


class ChiefProfileStore:
    def __init__(self, root: Path | None = None):
        configured = os.getenv("BETA_PROFILE_DIR")
        self.root = root or (Path(configured) if configured else get_hermes_home() / "beta" / "profiles")
        self._lock = threading.RLock()

    @staticmethod
    def _safe_user_id(user_id: str | None) -> str:
        value = (user_id or "local").strip()
        safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
        return safe[:128] or "local"

    def _path(self, user_id: str | None) -> Path:
        return self.root / f"{self._safe_user_id(user_id)}.json"

    def load(self, user_id: str | None, *, display_name: str | None = None) -> ChiefProfile:
        path = self._path(user_id)
        with self._lock:
            try:
                data: Any = json.loads(path.read_text(encoding="utf-8"))
                return ChiefProfile.model_validate(data)
            except FileNotFoundError:
                return ChiefProfile(user_id=self._safe_user_id(user_id), display_name=display_name or "Chief")
            except (OSError, ValueError, json.JSONDecodeError):
                return ChiefProfile(user_id=self._safe_user_id(user_id), display_name=display_name or "Chief")

    def save(self, profile: ChiefProfile) -> ChiefProfile:
        self.root.mkdir(parents=True, exist_ok=True)
        for fact in profile.facts:
            fact.validate_policy()
        path = self._path(profile.user_id)
        temporary = path.with_suffix(".tmp")
        payload = json.dumps(profile.model_dump(mode="json"), ensure_ascii=False, indent=2)
        with self._lock:
            temporary.write_text(payload, encoding="utf-8")
            temporary.replace(path)
        return profile

    def add_fact(self, user_id: str | None, fact: ChiefFact, *, display_name: str | None = None) -> ChiefProfile:
        fact.validate_policy()
        current = self.load(user_id, display_name=display_name)
        deduped = tuple(item for item in current.facts if not (item.type == fact.type and item.value == fact.value))
        updated = current.model_copy(update={"facts": deduped + (fact,), "revision": current.revision + 1})
        return self.save(updated)
