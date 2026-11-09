"""The skills index port: the skills an agent offers, as one plugin kind.

Skills are an agent's records: the index the prompt builder and the skills tools read (every
loadable skill's name, description and category), the bodies a load reads (``SKILL.md`` and its
supporting files), and the writes a turn or an administration makes (create, replace, delete).
:class:`SkillIndex` is that surface, at the level today's call sites use (``tools/skills_tool``,
``tools/skill_manager_tool``, ``agent/prompt_builder``, the curator and the ``hermes skills``
commands), never a filesystem layout: a backend may keep the bodies anywhere.

Skill bodies are behind :meth:`SkillIndex.read_body` alone, so a later store can move them (a
per-agent blob bucket, one object per file) without touching the port; until then the local
backend serves them from today's files at today's paths.

Ownership follows the scope, never a path or an argument: every operation reads its pin
(tenant, agent, owner, actor) from ``hermes_scope.current_scope()`` when it runs, and a backend
refuses a pin it cannot serve with :class:`~hermes_state_port.StoragePinRefused`. Reads resolve
through :func:`resolution_order`: the agent's own skills, then the owner's owner-wide ones, then
the tenant's — the one documented policy function, the same precedence for a personal and a
group owner. Writes name their level explicitly (:data:`SKILL_LEVELS`); a backend declares the
levels it stores (``SkillsIndexProvider.levels``) and refuses a write at any other with
:class:`SkillsLevelRefused`, a refusal that says nothing about whether the skill exists.

This is the interface and its local backend only: no production caller goes through it yet, and
nothing in config.yaml selects a backend. Today's callers still read the profile ``skills/``
directory directly; moving them behind this port, call site by call site, is a follow-up.
:func:`resolve_skills_index` takes the backend's name explicitly: ``"local"`` (or empty) is
``hermes_skills_port_local`` (today's per-profile ``skills/`` directory); any other name is a
provider from ``plugins/skills_index/<name>/`` or ``$HERMES_HOME/plugins/<name>/``. A named
provider that is missing or unavailable is an error, never a fallback to the local index.

Providers load in-process only. Under ``plugins.isolation: host`` a user provider is refused
with the reason: a host process would hand back opaque objects for the port, and its calls would
not carry the caller's scope.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional

from hermes_scope import Owner
from hermes_state_port import LEVELS, StoragePinRefused

#: The ownership levels a skills index can hold (see :data:`~hermes_state_port.LEVELS`): an
#: agent's own skills, an owner's skills for every agent it owns, a tenant's for all of them.
#: There is no participant level: a skill is not part of any one conversation.
SKILL_LEVELS = frozenset({"agent", "owner", "tenant"})

#: The levels today's call sites write: every skill write today lands on the agent.
AGENT_LEVEL = "agent"


class SkillsLevelRefused(PermissionError):
    """The backend does not store skills at this ownership level (its provider declares the
    levels it does). Says nothing about whether the named skill exists at that level."""


def resolution_order(scope) -> tuple[tuple[str, Optional[Owner]], ...]:
    """The levels a read of the bound agent's skills resolves through, in order: the agent's own
    skills first, then the ones its owner set once for every agent it owns, then the tenant's.

    The one documented policy for every skills backend and every owner kind (a principal or a
    group): which agents an owner-wide skill reaches is decided by the owner the scope names,
    never by the reader, so a member's own owner-wide skills never reach a group's agent. Changing
    the policy is a change to this one function and its tests, not to any schema or backend."""
    return (("agent", None), ("owner", scope.owner), ("tenant", None))


class SkillIndex(ABC):
    """The skills the bound agent offers, every operation pinned to the bound scope."""

    @abstractmethod
    def list(self) -> list[dict[str, Any]]:
        """Every skill resolved for the pinned agent, in the backend's stable order. Rows carry
        ``name``, ``description``, ``category``, ``status`` (``unique`` or ``ambiguous``),
        ``load_name`` (the name :meth:`get` and :meth:`read_body` resolve it by) and ``level``
        (the step of :func:`resolution_order` the row resolved at). A name two skills share in
        one level resolves only by the exact ``load_name`` the index reports. The offer-time
        gates (platform, environment, app availability, the disabled list) stay with the callers
        that apply them today; this is what is stored."""

    @abstractmethod
    def get(self, name: str) -> Optional[dict[str, Any]]:
        """One resolved row (see :meth:`list`). ``None`` for a skill the pinned agent does not
        offer: missing, another tenant's and another agent's look the same. A name two skills
        share is refused with today's ambiguity error, never guessed; it resolves by the exact
        ``load_name`` its :meth:`list` row reports."""

    @abstractmethod
    def read_body(self, name: str, file_path: Optional[str] = None) -> Optional[str]:
        """The text of the skill's ``SKILL.md``, or of one supporting file under its directory.
        ``None`` for a skill the pinned agent does not offer, and for a file it does not hold.
        A name two skills share is refused as in :meth:`get`, never guessed.

        Skill bodies live behind this one operation so a later store can move them without
        touching the port."""

    @abstractmethod
    def put(self, name: str, content: str, *, category: Optional[str] = None,
            level: str = AGENT_LEVEL) -> dict[str, Any]:
        """Create or replace one skill: its ``SKILL.md`` becomes *content*, its metadata what the
        index parses from it. *level* says where the write lands, explicitly. An existing skill
        of that name is replaced where it lives; a new one is created under *category*. A name
        two skills share is refused, never guessed. Returns the row :meth:`get` resolves for it."""

    @abstractmethod
    def put_file(self, name: str, file_path: str, content: str, *, level: str = AGENT_LEVEL) -> None:
        """Create or replace one supporting file of an existing skill. Refused for a path that
        is not a relative path under the skill's support directories."""

    @abstractmethod
    def delete(self, name: str, *, level: str = AGENT_LEVEL) -> bool:
        """Remove one skill, its directory and every file in it. ``False`` when the pinned agent
        offers no such skill at *level* — a foreign skill looks the same as a missing one. A
        name two skills share is refused as in :meth:`get`, never guessed."""

    @abstractmethod
    def delete_file(self, name: str, file_path: str, *, level: str = AGENT_LEVEL) -> bool:
        """Remove one supporting file. ``False`` when it is not there, a foreign skill's file
        included."""


@dataclass(frozen=True)
class SkillsIndexing:
    """One opened skills index. ``close`` gives back whatever ``open`` took."""

    index: SkillIndex
    close: Callable[[], None] = lambda: None


class SkillsIndexProvider(ABC):
    """The plugin kind: supplies one :class:`SkillIndex`. ``name``, ``open`` and
    ``assert_isolating`` are required; every other method has a safe default, and growth stays
    additive."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. ``"local"``."""

    #: The ownership levels this backend stores (see :data:`SKILL_LEVELS`, a subset of
    #: :data:`~hermes_state_port.LEVELS`); writes at any other level are refused with
    #: :class:`SkillsLevelRefused`.
    levels: frozenset[str] = frozenset({AGENT_LEVEL})

    def is_available(self) -> bool:
        """Whether this backend can run here. No network calls."""
        return True

    @abstractmethod
    def open(self, skills_dir: Any = None) -> SkillsIndexing:
        """The index for the bound scope. *skills_dir* is the local backend's per-profile handle
        (the profile's ``skills/`` directory, at today's path); a backend with no local files
        ignores it."""

    @abstractmethod
    def assert_isolating(self) -> None:
        """Raise unless this backend isolates tenants and agents from each other in storage
        itself (a backend whose role could read another tenant's skills fails). Required, with
        no default: a backend that has not checked must not pass. Nothing calls it yet: the
        first caller that serves a scope from a resolved provider must call it, and refuse the
        provider when it raises, before :meth:`open`."""


def resolve_skills_index(name: str) -> SkillsIndexProvider:
    """The provider named *name* (``"local"`` or empty: the local backend). Raises ``LookupError``
    with the reason for a named provider that is missing, refused or unavailable: a skills
    backend never falls back to another store."""
    name = name.strip()
    if name in ("", "local"):
        from hermes_skills_port_local import LocalSkillsIndexProvider
        return LocalSkillsIndexProvider()
    from plugins.skills_index import SkillsIndexRefused, load_skills_index
    try:
        provider = load_skills_index(name)
    except SkillsIndexRefused as refused:
        raise LookupError(f"skills index {name!r}: {refused}") from refused
    if provider is None:
        raise LookupError(f"skills index {name!r}: no such skills index provider")
    if not provider.is_available():
        raise LookupError(f"skills index {name!r} is not available here")
    return provider


__all__ = [
    "AGENT_LEVEL", "LEVELS", "SKILL_LEVELS", "SkillIndex", "SkillsIndexProvider",
    "SkillsIndexing", "SkillsLevelRefused", "StoragePinRefused",
    "resolution_order", "resolve_skills_index",
]
