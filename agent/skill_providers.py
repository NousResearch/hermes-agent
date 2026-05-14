"""Runtime skill provider registry.

Skill providers expose virtual skills without requiring a SKILL.md file on disk.
Plugins can register providers directly or via decorators; skill loading paths then
consult this registry before falling back to filesystem-backed skills.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import logging
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

from agent.skill_utils import _NAMESPACE_RE

logger = logging.getLogger(__name__)


@dataclass
class SkillMetadata:
    """Minimal metadata for progressive skill disclosure."""

    name: str
    description: str = ""
    namespace: Optional[str] = None
    category: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class SkillPayload:
    """Resolved virtual skill content returned by a provider."""

    name: str
    content: str
    description: str = ""
    linked_files: Optional[Dict[str, List[str]]] = None
    readiness_status: str = "available"
    tags: List[str] = field(default_factory=list)
    related_skills: List[str] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
    setup_needed: bool = False
    setup_skipped: bool = False


@runtime_checkable
class SkillProvider(Protocol):
    """Protocol implemented by virtual skill providers."""

    namespace: str

    def list_skills(self) -> List[SkillMetadata | Dict[str, Any]]:
        """Return lightweight metadata for skills this provider exposes."""

    def resolve_skill(self, name: str) -> SkillPayload | Dict[str, Any] | None:
        """Return the full skill payload for a bare skill name, or None."""

    def read_supporting_file(self, name: str, file_path: str) -> str | Dict[str, Any] | None:
        """Return virtual supporting-file content, or None if unavailable."""


_skill_providers: Dict[str, SkillProvider] = {}


def _validate_namespace(namespace: str) -> str:
    namespace = str(namespace or "").strip()
    if not namespace or not _NAMESPACE_RE.match(namespace):
        raise ValueError(
            f"Invalid skill provider namespace '{namespace}'. Must match [a-zA-Z0-9_-]+."
        )
    return namespace


def register_skill_provider(provider: SkillProvider, namespace: str | None = None) -> SkillProvider:
    """Register a virtual skill provider under a namespace."""

    ns = _validate_namespace(namespace or getattr(provider, "namespace", ""))
    setattr(provider, "namespace", ns)
    _skill_providers[ns] = provider
    logger.debug("Registered skill provider namespace: %s", ns)
    return provider


def skill_provider(namespace: str, *factory_args: Any, **factory_kwargs: Any) -> Callable[[type], type]:
    """Class decorator that instantiates and registers a skill provider.

    Example:
        @skill_provider("fmem", endpoint="http://127.0.0.1:18765")
        class FmemSkillProvider:
            ...
    """

    ns = _validate_namespace(namespace)

    def decorator(cls: type) -> type:
        instance = cls(*factory_args, **factory_kwargs)
        register_skill_provider(instance, ns)
        return cls

    return decorator


def unregister_skill_provider(namespace: str) -> None:
    _skill_providers.pop(namespace, None)


def clear_skill_providers() -> None:
    """Clear all providers. Intended for tests and plugin rediscovery."""

    _skill_providers.clear()


def get_skill_provider(namespace: str) -> SkillProvider | None:
    return _skill_providers.get(namespace)


def list_skill_provider_names() -> List[str]:
    return sorted(_skill_providers)


def list_skill_providers() -> Dict[str, SkillProvider]:
    return dict(_skill_providers)


def _coerce_metadata(raw: SkillMetadata | Dict[str, Any], namespace: str) -> Dict[str, Any]:
    data = asdict(raw) if is_dataclass(raw) else dict(raw or {})
    bare_name = str(data.get("name") or "").strip()
    if not bare_name:
        raise ValueError("Provider returned skill metadata without a name")
    if ":" in bare_name:
        qualified_name = bare_name
    else:
        qualified_name = f"{namespace}:{bare_name}"
    data["name"] = qualified_name
    data.setdefault("description", "")
    data["namespace"] = namespace
    data.setdefault("category", namespace)
    data.setdefault("provider", namespace)
    return data


def list_provider_skill_metadata(*, skip_errors: bool = True) -> List[Dict[str, Any]]:
    """Return normalized metadata for all registered provider skills."""

    skills: List[Dict[str, Any]] = []
    for namespace, provider in list_skill_providers().items():
        try:
            for raw in provider.list_skills() or []:
                skills.append(_coerce_metadata(raw, namespace))
        except Exception:
            if not skip_errors:
                raise
            logger.debug("Skill provider '%s' failed to list skills", namespace, exc_info=True)
    return skills


def resolve_provider_skill(namespace: str, name: str) -> SkillPayload | Dict[str, Any] | None:
    provider = get_skill_provider(namespace)
    if not provider:
        return None
    return provider.resolve_skill(name)


def read_provider_supporting_file(namespace: str, name: str, file_path: str) -> str | Dict[str, Any] | None:
    provider = get_skill_provider(namespace)
    if not provider or not hasattr(provider, "read_supporting_file"):
        return None
    return provider.read_supporting_file(name, file_path)
