"""Discoverable domain policy modules for StudyOS.

Each public module in this package exports one ``PACK``.  That keeps domain
growth at the plugin edge: adding a pack does not require editing the learning
runtime, intervention orchestration, prompt loader, or project tools.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Any

from plugins.study_os.activities import ActivityAdapter


ScheduleTemplate = Callable[[dict[str, Any]], dict[str, Any]]
_PACK_ID_RE = re.compile(r"^[a-z][a-z0-9_-]*\.v[1-9][0-9]*$")
_FALLBACK_PACK_ID = "general.v1"


@dataclass(frozen=True)
class DomainPack:
    """All domain-specific policy consumed by the shared StudyOS runtime."""

    id: str
    activity_adapter: ActivityAdapter
    prompt_skill: str | None
    intervention_duration: int
    project_defaults: Mapping[str, Any]
    schedule_template: ScheduleTemplate

    def __post_init__(self) -> None:
        if not _PACK_ID_RE.fullmatch(self.id):
            raise ValueError(
                "DomainPack.id must match <domain>.v<positive-version>"
            )
        if not isinstance(self.activity_adapter, ActivityAdapter):
            raise TypeError("DomainPack.activity_adapter must be an ActivityAdapter")
        if self.prompt_skill is not None and not self.prompt_skill.strip():
            raise ValueError("DomainPack.prompt_skill must be non-empty when provided")
        if (
            not isinstance(self.intervention_duration, int)
            or isinstance(self.intervention_duration, bool)
            or not 1 <= self.intervention_duration <= 720
        ):
            raise ValueError(
                "DomainPack.intervention_duration must be an integer from 1 to 720"
            )
        defaults = deepcopy(dict(self.project_defaults))
        if defaults.get("domain_pack") != self.id:
            raise ValueError(
                "DomainPack.project_defaults.domain_pack must match DomainPack.id"
            )
        if not callable(self.schedule_template):
            raise TypeError("DomainPack.schedule_template must be callable")
        object.__setattr__(self, "project_defaults", MappingProxyType(defaults))


@lru_cache(maxsize=1)
def domain_pack_registry() -> Mapping[str, DomainPack]:
    """Load one ``PACK`` from every public module in this package."""

    discovered: dict[str, DomainPack] = {}
    for module_info in pkgutil.iter_modules(__path__):
        if module_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{module_info.name}")
        pack = getattr(module, "PACK", None)
        if not isinstance(pack, DomainPack):
            raise RuntimeError(
                f"Domain pack module {module.__name__} must export PACK: DomainPack"
            )
        if pack.id in discovered:
            raise RuntimeError(f"Duplicate StudyOS DomainPack id: {pack.id}")
        discovered[pack.id] = pack
    if _FALLBACK_PACK_ID not in discovered:
        raise RuntimeError(
            f"StudyOS requires fallback DomainPack {_FALLBACK_PACK_ID}"
        )
    return MappingProxyType(discovered)


def _family_match(value: str, registry: Mapping[str, DomainPack]) -> DomainPack | None:
    family = value.split(".", 1)[0]
    matches = [
        pack
        for pack_id, pack in registry.items()
        if pack_id.split(".", 1)[0] == family
    ]
    return matches[0] if len(matches) == 1 else None


def domain_pack_for(selector: Mapping[str, Any] | str | None) -> DomainPack:
    """Resolve a pack, with ``domain_pack`` authoritative over ``domain``.

    Unknown ids retain the general behavior used by older manifests.  A bare
    domain name resolves when exactly one version of that domain is installed.
    """

    registry = domain_pack_registry()
    if isinstance(selector, str) or selector is None:
        requested = str(selector or "").strip().casefold()
        domain = ""
    else:
        requested = str(selector.get("domain_pack") or "").strip().casefold()
        domain = str(selector.get("domain") or "").strip().casefold()

    if requested:
        return (
            registry.get(requested)
            or _family_match(requested, registry)
            or registry[_FALLBACK_PACK_ID]
        )
    if domain:
        return (
            registry.get(domain)
            or _family_match(domain, registry)
            or registry[_FALLBACK_PACK_ID]
        )
    return registry[_FALLBACK_PACK_ID]


__all__ = ["DomainPack", "domain_pack_for", "domain_pack_registry"]
