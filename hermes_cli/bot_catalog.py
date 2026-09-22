"""Reviewed Bot Marketplace catalog with a closed, non-executable schema."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

logger = logging.getLogger(__name__)

LIVE_BOT_CATALOG_URL = "https://hermes-agent.nousresearch.com/docs/api/bot-catalog.json"
LIVE_BOT_CATALOG_TTL_SECONDS = 6 * 60 * 60
LIVE_BOT_CATALOG_NEGATIVE_TTL_SECONDS = 5 * 60
_REQUEST_TIMEOUT_SECONDS = 5.0
_MAX_LIVE_BYTES = 2 * 1024 * 1024
_MAX_ENTRY_BYTES = 128 * 1024
_MAX_ENTRIES = 500
_MAX_NEGATIVE_CACHE_KEYS = 64

_LIVE_FETCH_LOCK = threading.Lock()
_LIVE_FETCH_NEGATIVE_UNTIL: dict[str, float] = {}

_NAME_RE = re.compile(r"^[a-z0-9_-]{1,64}$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,31}$")
_SKILL_RE = re.compile(r"^official/[a-z0-9_-]+(?:/[a-z0-9_-]+)+$")
_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
_SECRET_VALUE_RE = re.compile(
    r"(?:-----BEGIN [A-Z ]+PRIVATE KEY-----|\b(?:sk|ghp|github_pat|xox[baprs])_[A-Za-z0-9_-]{16,}|"
    r"\b(?:api[_-]?key|token|password|secret)\s*[:=]\s*[^\s]{8,})",
    re.IGNORECASE,
)


class BotCatalogError(ValueError):
    """The reviewed catalog could not be validated as a whole."""


class BotRemovedError(BotCatalogError):
    """The requested canonical name is on the catalog kill list."""


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BotProfile(_ClosedModel):
    suggested_name: str
    description: str = Field(min_length=1, max_length=2000)
    soul: str = Field(min_length=1, max_length=64 * 1024)
    starter_prompt: str = Field(min_length=1, max_length=4000)

    @field_validator("suggested_name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not _NAME_RE.fullmatch(value):
            raise ValueError("must match [a-z0-9_-]{1,64}")
        return value


class BotCapabilities(_ClosedModel):
    skills: list[str] = Field(default_factory=list, max_length=32)
    toolsets: list[str] = Field(default_factory=list, max_length=32)

    @field_validator("skills")
    @classmethod
    def _reviewed_skills(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("duplicate skill identifier")
        for value in values:
            if not _SKILL_RE.fullmatch(value):
                raise ValueError(f"unreviewed skill identifier {value!r}")
        return values

    @field_validator("toolsets")
    @classmethod
    def _known_toolsets(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("duplicate toolset")
        if any(not _NAME_RE.fullmatch(value) for value in values):
            raise ValueError("toolsets must be lowercase identifiers")
        return values


class BotSetupRequirement(_ClosedModel):
    kind: Literal["toolset", "command", "connector", "plugin"]
    id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/+-]*$")
    required: bool = True
    purpose: str = Field(min_length=1, max_length=500)


class BotSetup(_ClosedModel):
    requirements: list[BotSetupRequirement] = Field(default_factory=list, max_length=64)

    @field_validator("requirements")
    @classmethod
    def _unique_requirements(cls, values: list[BotSetupRequirement]) -> list[BotSetupRequirement]:
        keys = [(value.kind, value.id) for value in values]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate setup requirement")
        return values


class BotRoutine(_ClosedModel):
    id: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    name: str = Field(min_length=1, max_length=120)
    prompt: str = Field(min_length=1, max_length=16 * 1024)
    schedule: str = Field(min_length=1, max_length=200)


class BotPresentation(_ClosedModel):
    emoji: str = Field(min_length=1, max_length=16)
    color: str

    @field_validator("color")
    @classmethod
    def _valid_color(cls, value: str) -> str:
        if not _COLOR_RE.fullmatch(value):
            raise ValueError("must be a six-digit hex color")
        return value.upper()


class BotCatalogEntry(_ClosedModel):
    name: str
    version: str
    maintainer: str = Field(min_length=1, max_length=200)
    tier: Literal["official", "community"]
    category: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_-]+$")
    tags: list[str] = Field(default_factory=list, max_length=16)
    title: str = Field(min_length=1, max_length=120)
    summary: str = Field(min_length=1, max_length=500)
    profile: BotProfile
    capabilities: BotCapabilities
    setup: BotSetup = Field(default_factory=BotSetup)
    routines: list[BotRoutine] = Field(default_factory=list, max_length=32)
    presentation: BotPresentation

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not _NAME_RE.fullmatch(value):
            raise ValueError("must match [a-z0-9_-]{1,64}")
        return value

    @field_validator("version")
    @classmethod
    def _valid_version(cls, value: str) -> str:
        if not _VERSION_RE.fullmatch(value):
            raise ValueError("invalid version label")
        return value

    @field_validator("tags")
    @classmethod
    def _valid_tags(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)) or any(not re.fullmatch(r"[a-z0-9_-]{1,32}", value) for value in values):
            raise ValueError("tags must be unique lowercase identifiers")
        return values

    @field_validator("routines")
    @classmethod
    def _unique_routines(cls, values: list[BotRoutine]) -> list[BotRoutine]:
        ids = [value.id for value in values]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate routine id")
        return values

    @model_validator(mode="after")
    def _no_secret_values(self):
        for value in _walk_strings(self.model_dump(mode="python")):
            if _SECRET_VALUE_RE.search(value):
                raise ValueError("secret-like value is forbidden in bot blueprints")
        return self


class RemovedBot(_ClosedModel):
    name: str
    reason: str = Field(min_length=1, max_length=500)

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not _NAME_RE.fullmatch(value):
            raise ValueError("must match [a-z0-9_-]{1,64}")
        return value


class RemovedBots(_ClosedModel):
    removed: list[RemovedBot] = Field(default_factory=list, max_length=_MAX_ENTRIES)


def get_bot_catalog_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "bot-catalog"


def resolve_reviewed_skill_path(identifier: str) -> Path | None:
    """Map an ``official/...`` identifier to a shipped bundled/optional skill directory."""
    if not _SKILL_RE.fullmatch(identifier):
        return None
    relative = Path(*identifier.split("/")[1:])
    root = Path(__file__).resolve().parent.parent
    for base in (root / "skills", root / "optional-skills"):
        candidate = base / relative
        if candidate.is_dir() and (candidate / "SKILL.md").is_file():
            return candidate
    return None


def _walk_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _walk_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_strings(item)


def validate_bot_catalog_dependencies(entry: BotCatalogEntry, *, label: str) -> None:
    """Validate install-time dependencies without coupling immutable snapshots to the checkout."""
    missing_skills = [
        identifier for identifier in entry.capabilities.skills
        if resolve_reviewed_skill_path(identifier) is None
    ]
    if missing_skills:
        raise BotCatalogError(
            f"{label}: reviewed skill is not available in this checkout: {missing_skills[0]!r}"
        )

    from toolsets import get_toolset

    reviewed_toolsets = list(dict.fromkeys([
        *entry.capabilities.toolsets,
        *(requirement.id for requirement in entry.setup.requirements if requirement.kind == "toolset"),
    ]))
    unknown_toolsets = [
        name for name in reviewed_toolsets
        if get_toolset(name, include_registry=False) is None
    ]
    if unknown_toolsets:
        raise BotCatalogError(f"{label}: unknown toolset(s): {', '.join(unknown_toolsets)}")


def _parse_entry(raw: Any, label: str) -> BotCatalogEntry:
    encoded = json.dumps(raw, ensure_ascii=False, default=str).encode("utf-8")
    if len(encoded) > _MAX_ENTRY_BYTES:
        raise BotCatalogError(f"{label}: record exceeds {_MAX_ENTRY_BYTES} bytes")
    try:
        entry = BotCatalogEntry.model_validate(raw)
    except ValidationError as exc:
        raise BotCatalogError(f"{label}: {exc}") from exc
    validate_bot_catalog_dependencies(entry, label=label)
    return entry


def _parse_removed(raw: Any, label: str) -> list[RemovedBot]:
    try:
        parsed = RemovedBots.model_validate(raw)
    except ValidationError as exc:
        raise BotCatalogError(f"{label}: {exc}") from exc
    names = [item.name for item in parsed.removed]
    if len(names) != len(set(names)):
        raise BotCatalogError(f"{label}: duplicate removed bot name")
    return list(parsed.removed)


def _read_yaml(path: Path) -> Any:
    try:
        if path.stat().st_size > _MAX_ENTRY_BYTES:
            raise BotCatalogError(f"{path}: record exceeds {_MAX_ENTRY_BYTES} bytes")
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except BotCatalogError:
        raise
    except Exception as exc:
        raise BotCatalogError(f"failed to read {path}: {exc}") from exc


def _tree_catalog(catalog_dir: Path) -> tuple[list[BotCatalogEntry], list[RemovedBot]]:
    if not catalog_dir.is_dir():
        raise BotCatalogError(f"bot catalog directory is missing: {catalog_dir}")
    paths = [path for path in sorted(catalog_dir.glob("*.yaml")) if path.name != "removed.yaml"]
    if len(paths) > _MAX_ENTRIES:
        raise BotCatalogError(f"catalog has more than {_MAX_ENTRIES} entries")
    entries = [_parse_entry(_read_yaml(path), str(path)) for path in paths]
    removed_path = catalog_dir / "removed.yaml"
    removed = _parse_removed(_read_yaml(removed_path), str(removed_path)) if removed_path.is_file() else []
    _validate_unique(entries, "in-tree catalog")
    return entries, removed


def _validate_unique(entries: list[BotCatalogEntry], label: str) -> None:
    names = [entry.name for entry in entries]
    if len(names) != len(set(names)):
        raise BotCatalogError(f"{label}: duplicate catalog name")


def _live_cache_path() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "cache" / "bot-catalog.json"


def _negative_fetch_cached(cache: Path, now: float) -> bool:
    key = str(cache.resolve(strict=False))
    with _LIVE_FETCH_LOCK:
        expired = [item for item, until in _LIVE_FETCH_NEGATIVE_UNTIL.items() if until <= now]
        for item in expired:
            _LIVE_FETCH_NEGATIVE_UNTIL.pop(item, None)
        return _LIVE_FETCH_NEGATIVE_UNTIL.get(key, 0.0) > now


def _record_negative_fetch(cache: Path, now: float) -> None:
    key = str(cache.resolve(strict=False))
    with _LIVE_FETCH_LOCK:
        if len(_LIVE_FETCH_NEGATIVE_UNTIL) >= _MAX_NEGATIVE_CACHE_KEYS and key not in _LIVE_FETCH_NEGATIVE_UNTIL:
            oldest = min(_LIVE_FETCH_NEGATIVE_UNTIL, key=lambda item: _LIVE_FETCH_NEGATIVE_UNTIL[item])
            _LIVE_FETCH_NEGATIVE_UNTIL.pop(oldest, None)
        _LIVE_FETCH_NEGATIVE_UNTIL[key] = now + LIVE_BOT_CATALOG_NEGATIVE_TTL_SECONDS


def fetch_live_bot_catalog(*, force: bool = False) -> dict[str, Any] | None:
    """Fetch the published feed; failures are briefly cached before falling back to the tree."""
    cache = _live_cache_path()
    now = time.time()
    try:
        if not force and cache.is_file() and now - cache.stat().st_mtime < LIVE_BOT_CATALOG_TTL_SECONDS:
            raw = cache.read_bytes()
            if len(raw) <= _MAX_LIVE_BYTES:
                data = json.loads(raw)
                return data if isinstance(data, dict) else None
        if not force and _negative_fetch_cached(cache, now):
            return None
        import httpx
        from hermes_constants import mkdir_under_hermes_home

        response = httpx.get(LIVE_BOT_CATALOG_URL, timeout=_REQUEST_TIMEOUT_SECONDS, follow_redirects=True)
        response.raise_for_status()
        if len(response.content) > _MAX_LIVE_BYTES:
            raise BotCatalogError("live bot catalog payload is too large")
        data = response.json()
        if not isinstance(data, dict):
            raise BotCatalogError("live bot catalog payload must be an object")
        mkdir_under_hermes_home(cache.parent)
        cache.write_bytes(response.content)
        with _LIVE_FETCH_LOCK:
            _LIVE_FETCH_NEGATIVE_UNTIL.pop(str(cache.resolve(strict=False)), None)
        return data
    except Exception as exc:
        _record_negative_fetch(cache, now)
        logger.debug("Bot catalog live fetch failed: %s", exc)
        return None


def _live_entries(data: dict[str, Any]) -> list[BotCatalogEntry]:
    raw_entries = data.get("entries")
    if not isinstance(raw_entries, list) or len(raw_entries) > _MAX_ENTRIES:
        raise BotCatalogError("live catalog entries must be a bounded list")
    entries = [_parse_entry(raw, f"{LIVE_BOT_CATALOG_URL}#{index}") for index, raw in enumerate(raw_entries)]
    _validate_unique(entries, "live catalog")
    return entries


def _live_removed(data: dict[str, Any]) -> list[RemovedBot]:
    return _parse_removed({"removed": data.get("removed", [])}, LIVE_BOT_CATALOG_URL)


def _resolved_catalog(catalog_dir: Path | None, include_live: bool) -> tuple[list[BotCatalogEntry], list[RemovedBot]]:
    root = catalog_dir or get_bot_catalog_dir()
    tree_entries, tree_removed = _tree_catalog(root)
    if include_live and catalog_dir is None:
        data = fetch_live_bot_catalog()
        if data is not None:
            live_removed: list[RemovedBot] = []
            try:
                live_removed = _live_removed(data)
            except BotCatalogError as exc:
                logger.warning("Bot catalog: invalid live removal list ignored: %s", exc)
            try:
                entries = _live_entries(data)
            except BotCatalogError as exc:
                logger.warning("Bot catalog: invalid live entries, using reviewed in-tree catalog: %s", exc)
                entries = tree_entries
            # A docs build that emitted its empty fallback is an outage, not an authoritative
            # request to erase the bundled catalog. Valid revocations remain authoritative even
            # when a newer entry schema cannot be loaded by this client.
            return (entries or tree_entries), [*tree_removed, *live_removed]
    return tree_entries, tree_removed


def load_bot_catalog(catalog_dir: Path | None = None, *, include_live: bool = True) -> list[BotCatalogEntry]:
    entries, removed = _resolved_catalog(catalog_dir, include_live)
    blocked = {item.name for item in removed}
    return [entry for entry in entries if entry.name not in blocked]


def removed_bots(catalog_dir: Path | None = None, *, include_live: bool = True) -> list[RemovedBot]:
    return _resolved_catalog(catalog_dir, include_live)[1]


def is_removed_bot(name: str, catalog_dir: Path | None = None, *, include_live: bool = True) -> bool:
    return any(item.name == name for item in removed_bots(catalog_dir, include_live=include_live))


def resolve_bot_catalog_entry(
    name: str, catalog_dir: Path | None = None, *, include_live: bool = True
) -> BotCatalogEntry:
    if not _NAME_RE.fullmatch(name):
        raise BotCatalogError("invalid canonical bot catalog name")
    entries, removed = _resolved_catalog(catalog_dir, include_live)
    killed = next((item for item in removed if item.name == name), None)
    if killed is not None:
        raise BotRemovedError(f"bot {name!r} was removed: {killed.reason}")
    entry = next((item for item in entries if item.name == name), None)
    if entry is None:
        raise BotCatalogError(f"unknown bot catalog name: {name}")
    return entry
