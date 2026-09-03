"""Fail-closed resolution for cron ``skill`` / ``skills`` references."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Iterable


@dataclass(frozen=True)
class ResolvedSkillReference:
    name: str
    kind: str
    content: str
    payload: dict | None = None


@dataclass(frozen=True)
class UnresolvedSkillReference:
    name: str
    reason: str


class CronSkillReferenceError(ValueError):
    """A scheduled job declared a skill or bundle that cannot fully resolve."""


def normalize_skill_references(values: Iterable[object] | None) -> list[str]:
    normalized: list[str] = []
    for value in values or []:
        name = str(value or "").strip()
        if name and name not in normalized:
            normalized.append(name)
    return normalized


def resolve_skill_references(
    values: Iterable[object] | None,
    *,
    task_id: str | None = None,
) -> tuple[list[ResolvedSkillReference], list[UnresolvedSkillReference]]:
    """Resolve with the same skill/bundle machinery used by interactive loads."""
    names = normalize_skill_references(values)
    if not names:
        return [], []

    from agent.skill_bundles import (
        build_bundle_invocation_message,
        resolve_bundle_command_key,
    )
    from agent.skill_utils import normalize_skill_lookup_name
    from tools.skills_tool import skill_view

    resolved: list[ResolvedSkillReference] = []
    unresolved: list[UnresolvedSkillReference] = []
    for name in names:
        bundle_key = resolve_bundle_command_key(name.lstrip("/"))
        if bundle_key:
            bundle = build_bundle_invocation_message(
                bundle_key,
                user_instruction="",
                task_id=task_id,
            )
            if not bundle:
                unresolved.append(
                    UnresolvedSkillReference(name, "bundle could not load any skills")
                )
                continue
            message, _loaded_members, missing_members = bundle
            if missing_members:
                unresolved.append(
                    UnresolvedSkillReference(
                        name,
                        "bundle has unresolved member(s): "
                        + ", ".join(sorted(str(item) for item in missing_members)),
                    )
                )
                continue
            resolved.append(
                ResolvedSkillReference(name=name, kind="bundle", content=message)
            )
            continue

        try:
            payload = json.loads(skill_view(normalize_skill_lookup_name(name)))
        except (json.JSONDecodeError, TypeError) as exc:
            unresolved.append(
                UnresolvedSkillReference(name, f"skill returned invalid JSON: {exc}")
            )
            continue
        if not isinstance(payload, dict) or not payload.get("success"):
            reason = (
                payload.get("error")
                if isinstance(payload, dict)
                else "skill returned a non-object payload"
            )
            unresolved.append(
                UnresolvedSkillReference(name, str(reason or "skill not found"))
            )
            continue
        resolved.append(
            ResolvedSkillReference(
                name=name,
                kind="skill",
                content=str(payload.get("content") or "").strip(),
                payload=payload,
            )
        )
    return resolved, unresolved


def require_skill_references(
    values: Iterable[object] | None,
    *,
    task_id: str | None = None,
) -> list[ResolvedSkillReference]:
    resolved, unresolved = resolve_skill_references(values, task_id=task_id)
    if unresolved:
        details = "; ".join(f"{item.name}: {item.reason}" for item in unresolved)
        raise CronSkillReferenceError(
            f"Unresolved cron skill reference(s): {details}"
        )
    return resolved
