"""Plugin-owned canonical role catalogue."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import yaml


@dataclass(frozen=True)
class RoleDefinition:
    title: str
    slug: str
    aliases: Tuple[str, ...]
    allowed_execution_modes: Tuple[str, ...]
    prompt: str


class RoleCatalog:
    def __init__(self, roles: Iterable[RoleDefinition]):
        self._roles = tuple(roles)
        self._lookup: Dict[str, RoleDefinition] = {}
        for role in self._roles:
            for value in (role.title, role.slug, *role.aliases):
                key = self._key(value)
                if key in self._lookup and self._lookup[key] != role:
                    raise ValueError(f"duplicate role alias: {value}")
                self._lookup[key] = role

    @staticmethod
    def _key(value: str) -> str:
        return "-".join(str(value).strip().lower().replace("/", " ").split()).replace("_", "-")

    @classmethod
    def from_path(cls, path: Path) -> "RoleCatalog":
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        raw_roles = payload.get("roles")
        if not isinstance(raw_roles, list) or not raw_roles:
            raise ValueError("role catalogue must contain a non-empty roles list")
        roles = []
        for item in raw_roles:
            if not isinstance(item, dict):
                raise ValueError("each role catalogue entry must be a mapping")
            roles.append(
                RoleDefinition(
                    title=str(item["title"]).strip(),
                    slug=str(item["slug"]).strip(),
                    aliases=tuple(str(v).strip() for v in item.get("aliases", [])),
                    allowed_execution_modes=tuple(
                        str(v).strip() for v in item.get("allowed_execution_modes", [])
                    ),
                    prompt=str(item["prompt"]).strip(),
                )
            )
        return cls(roles)

    @classmethod
    def default(cls) -> "RoleCatalog":
        return cls.from_path(Path(__file__).with_name("roles.yaml"))

    def resolve(self, value: str) -> RoleDefinition:
        role = self._lookup.get(self._key(value))
        if role is None:
            choices = ", ".join(role.title for role in self._roles)
            raise ValueError(f"unknown role {value!r}; available roles: {choices}")
        return role

    @property
    def roles(self) -> Tuple[RoleDefinition, ...]:
        return self._roles
