"""Build a validated wake manifest from installed skills.

This is the Step-1 bridge between human-maintained skill frontmatter and
runtime/audit consumers. The manifest stays intentionally small:
- route behavior still lives in gateway/background_wakeups.py
- skill-to-route bindings come from metadata.hermes.wake.route
- invalid wake metadata is reported via manifest errors instead of crashing
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from agent.skill_utils import (
    extract_skill_wake_metadata,
    get_all_skills_dirs,
    get_disabled_skill_names,
    iter_skill_index_files,
    parse_frontmatter,
    skill_matches_platform,
)
from hermes_cli.config import read_raw_config

logger = logging.getLogger(__name__)

_ALLOWED_WAKE_RISKS = {"read_only", "internal_write", "external_write", "low"}


def _normalize_key(value: str) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def _normalize_optional_string(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _dedupe_strings(values: Iterable[Any]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in values:
        item = str(raw or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)


def _clean_wake_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        field: value
        for field, value in (payload or {}).items()
        if value not in (None, (), [], "")
    }


class WakeMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    route: str | None = None
    aliases: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    route_examples: tuple[str, ...] = ()
    risk: str | None = None
    delivery: str | None = None

    @field_validator("route", "risk", "delivery", mode="before")
    @classmethod
    def _normalize_scalar(cls, value: Any) -> str | None:
        return _normalize_optional_string(value)

    @field_validator("risk")
    @classmethod
    def _validate_risk(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value not in _ALLOWED_WAKE_RISKS:
            raise ValueError(
                f"risk must be one of {', '.join(sorted(_ALLOWED_WAKE_RISKS))}"
            )
        return value

    @field_validator("delivery")
    @classmethod
    def _validate_delivery(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.replace("_", "").isalnum() or value.lower() != value:
            raise ValueError("delivery must be lowercase snake_case")
        return value

    @field_validator("aliases", "keywords", "route_examples", mode="before")
    @classmethod
    def _normalize_sequence(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple, set)):
            raise TypeError("expected a list of strings")
        return _dedupe_strings(value)


class WakeSkillRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    name: str
    path: str
    wake: dict[str, Any] = Field(default_factory=dict)

    @field_validator("key", "name", "path", mode="before")
    @classmethod
    def _normalize_required_string(cls, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("must be a non-empty string")
        return text

    @field_validator("wake", mode="before")
    @classmethod
    def _normalize_wake(cls, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError("wake must be a dict")
        return {str(key).strip(): item for key, item in value.items() if str(key).strip()}


class WakeRouteRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str
    display_command: str
    skills: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    route_examples: tuple[str, ...] = ()
    append_toolsets: tuple[str, ...] = ()
    prepend_toolsets: tuple[str, ...] = ()
    drop_toolsets: tuple[str, ...] = ()
    conditional_skills: dict[str, tuple[str, ...]] = Field(default_factory=dict)

    @field_validator("command", "display_command", mode="before")
    @classmethod
    def _normalize_required_string(cls, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("must be a non-empty string")
        return text

    @field_validator(
        "skills",
        "aliases",
        "keywords",
        "route_examples",
        "append_toolsets",
        "prepend_toolsets",
        "drop_toolsets",
        mode="before",
    )
    @classmethod
    def _normalize_skills(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple, set)):
            raise TypeError("expected a list of skill names")
        return _dedupe_strings(value)

    @field_validator("conditional_skills", mode="before")
    @classmethod
    def _normalize_conditional_skills(cls, value: Any) -> dict[str, tuple[str, ...]]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError("expected conditional_skills to be a dict")
        result: dict[str, tuple[str, ...]] = {}
        for key, raw in value.items():
            skill_name = str(key or "").strip()
            if not skill_name:
                continue
            if isinstance(raw, str):
                raw = [raw]
            if not isinstance(raw, (list, tuple, set)):
                raise TypeError("expected conditional skill keywords to be a list of strings")
            result[skill_name] = _dedupe_strings(raw)
        return result


class WakeManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    platform: str
    routes: dict[str, WakeRouteRecord]
    skills: list[WakeSkillRecord]
    errors: tuple[str, ...] = ()

    @field_validator("platform", mode="before")
    @classmethod
    def _normalize_platform(cls, value: Any) -> str:
        text = str(value or "").strip().lower()
        if not text:
            raise ValueError("platform is required")
        return text

    @field_validator("errors", mode="before")
    @classmethod
    def _normalize_errors(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple, set)):
            raise TypeError("expected a list of error strings")
        return _dedupe_strings(value)


def _build_route_catalog(base_route_catalog: dict[str, dict[str, Any]], errors: list[str]) -> dict[str, dict[str, Any]]:
    routes: dict[str, dict[str, Any]] = {}
    for route_name, raw_spec in base_route_catalog.items():
        key = str(route_name or "").strip()
        if not key:
            continue
        try:
            spec = WakeRouteRecord.model_validate(raw_spec)
        except ValidationError as exc:
            errors.append(f"route:{key}: {exc.errors()[0]['msg']}")
            continue
        routes[key] = spec.model_dump(mode="python")
    return routes


def _scan_config_wake_overrides(
    allowed_routes: set[str],
    errors: list[str],
) -> dict[str, dict[str, Any]]:
    raw_config = read_raw_config()
    skills_cfg = raw_config.get("skills") if isinstance(raw_config, dict) else {}
    if not isinstance(skills_cfg, dict):
        return {}

    raw_overrides = skills_cfg.get("wake_overrides")
    if raw_overrides in (None, {}):
        return {}
    if not isinstance(raw_overrides, dict):
        errors.append("skills.wake_overrides: expected a dict of skill overrides")
        return {}

    overrides: dict[str, dict[str, Any]] = {}
    for raw_skill_name, raw_override in raw_overrides.items():
        skill_name = str(raw_skill_name or "").strip()
        key = _normalize_key(skill_name)
        if not key:
            continue
        if not isinstance(raw_override, dict):
            errors.append(f"wake_override:{skill_name}: expected a dict")
            continue

        explicit_fields = {
            str(field).strip()
            for field in raw_override.keys()
            if str(field).strip() in {"route", "aliases", "keywords", "route_examples", "risk", "delivery"}
        }
        if not explicit_fields:
            continue

        try:
            validated = WakeMetadata.model_validate(raw_override).model_dump(
                mode="python",
                exclude_none=False,
            )
        except ValidationError as exc:
            errors.append(f"wake_override:{skill_name}: invalid wake metadata ({exc.errors()[0]['msg']})")
            continue

        route = str(validated.get("route") or "").strip()
        if "route" in explicit_fields and route and route not in allowed_routes:
            errors.append(f"wake_override:{skill_name}: unknown wake route '{route}'")
            explicit_fields.remove("route")

        overrides[key] = {
            "fields": tuple(sorted(explicit_fields)),
            "wake": validated,
        }

    return overrides


def _merge_wake_override_payload(
    wake_payload: dict[str, Any],
    override_spec: dict[str, Any] | None,
) -> dict[str, Any]:
    if not override_spec:
        return dict(wake_payload or {})

    merged = dict(wake_payload or {})
    explicit_fields = set(override_spec.get("fields") or ())
    override_wake = dict(override_spec.get("wake") or {})

    for field in ("route", "risk", "delivery"):
        if field not in explicit_fields:
            continue
        value = _normalize_optional_string(override_wake.get(field))
        if value is None:
            merged.pop(field, None)
        else:
            merged[field] = value

    for field in ("aliases", "keywords", "route_examples"):
        if field not in explicit_fields:
            continue
        values = tuple(override_wake.get(field) or ())
        if values:
            merged[field] = values
        else:
            merged.pop(field, None)

    return _clean_wake_payload(merged)


def _scan_skill_records(
    platform: str,
    allowed_routes: set[str],
    errors: list[str],
    *,
    wake_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    disabled = {_normalize_key(name) for name in get_disabled_skill_names(platform=platform)}
    records: dict[str, dict[str, Any]] = {}

    for skills_dir in get_all_skills_dirs():
        if not skills_dir.is_dir():
            continue
        for skill_file in iter_skill_index_files(skills_dir, "SKILL.md"):
            try:
                raw = skill_file.read_text(encoding="utf-8")
                frontmatter, _ = parse_frontmatter(raw)
            except Exception as exc:  # pragma: no cover - defensive only
                logger.debug("Could not parse skill file %s: %s", skill_file, exc)
                continue

            if not skill_matches_platform(frontmatter):
                continue

            skill_name = str(frontmatter.get("name") or skill_file.parent.name).strip()
            key = _normalize_key(skill_name)
            if not key or key in disabled:
                continue

            wake_payload: dict[str, Any] = {}
            raw_wake = extract_skill_wake_metadata(frontmatter)
            if raw_wake:
                try:
                    wake_payload = WakeMetadata.model_validate(raw_wake).model_dump(
                        mode="python",
                        exclude_none=True,
                    )
                except ValidationError as exc:
                    errors.append(f"skill:{skill_name}: invalid wake metadata ({exc.errors()[0]['msg']})")
                    wake_payload = {}

            route = str(wake_payload.get("route") or "").strip()
            if route and route not in allowed_routes:
                errors.append(
                    f"skill:{skill_name}: unknown wake route '{route}'"
                )
                wake_payload.pop("route", None)

            wake_payload = _merge_wake_override_payload(wake_payload, (wake_overrides or {}).get(key))

            try:
                record = WakeSkillRecord.model_validate(
                    {
                        "key": key,
                        "name": skill_name,
                        "path": str(skill_file.parent),
                        "wake": wake_payload or None,
                    }
                )
            except ValidationError as exc:
                errors.append(f"skill:{skill_name}: {exc.errors()[0]['msg']}")
                continue

            payload = record.model_dump(mode="python", exclude_none=True)
            payload["wake"] = _clean_wake_payload(payload.get("wake") or {})
            records[key] = payload

    return records


def build_wake_manifest(
    base_route_catalog: dict[str, dict[str, Any]],
    *,
    platform: str = "feishu",
) -> dict[str, Any]:
    """Build a validated wake manifest for one platform.

    The manifest always returns successfully. Any invalid wake metadata is surfaced
    in ``errors`` and excluded from route bindings instead of breaking runtime.
    """

    normalized_platform = str(platform or "feishu").strip().lower() or "feishu"
    errors: list[str] = []

    routes = _build_route_catalog(base_route_catalog, errors)
    allowed_routes = set(routes)
    wake_overrides = _scan_config_wake_overrides(allowed_routes, errors)
    skills_by_key = _scan_skill_records(
        normalized_platform,
        allowed_routes,
        errors,
        wake_overrides=wake_overrides,
    )

    route_bindings: dict[str, list[str]] = {
        route_name: list(route_spec.get("skills", ()) or ())
        for route_name, route_spec in routes.items()
    }
    route_aliases: dict[str, list[str]] = {
        route_name: list(route_spec.get("aliases", ()) or ())
        for route_name, route_spec in routes.items()
    }
    route_keywords: dict[str, list[str]] = {
        route_name: list(route_spec.get("keywords", ()) or ())
        for route_name, route_spec in routes.items()
    }
    route_examples: dict[str, list[str]] = {
        route_name: list(route_spec.get("route_examples", ()) or ())
        for route_name, route_spec in routes.items()
    }

    ordered_skill_records = sorted(
        skills_by_key.values(),
        key=lambda record: (str(record.get("name") or "").lower(), str(record.get("path") or "")),
    )
    for record in ordered_skill_records:
        wake = record.get("wake") or {}
        route = str(wake.get("route") or "").strip()
        if route and route in route_bindings:
            route_bindings[route].append(str(record["name"]))
            route_aliases[route].extend(str(item) for item in (wake.get("aliases") or ()))
            route_keywords[route].extend(str(item) for item in (wake.get("keywords") or ()))
            route_examples[route].extend(str(item) for item in (wake.get("route_examples") or ()))

    manifest_routes: dict[str, dict[str, Any]] = {}
    for route_name, route_spec in routes.items():
        merged_spec = dict(route_spec)
        merged_spec["skills"] = _dedupe_strings(route_bindings.get(route_name, ()))
        merged_spec["aliases"] = _dedupe_strings(route_aliases.get(route_name, ()))
        merged_spec["keywords"] = _dedupe_strings(route_keywords.get(route_name, ()))
        merged_spec["route_examples"] = _dedupe_strings(route_examples.get(route_name, ()))
        manifest_routes[route_name] = merged_spec

    manifest_payload = {
        "version": 1,
        "platform": normalized_platform,
        "routes": manifest_routes,
        "skills": ordered_skill_records,
        "errors": tuple(errors),
    }
    return WakeManifest.model_validate(manifest_payload).model_dump(mode="python")
