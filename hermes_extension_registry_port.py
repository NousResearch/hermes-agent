"""The user extension registry port: the user plugins and MCP servers a tenant's people install.

Every extension a user adds — a plugin, or an MCP server definition — is one registry entry
carrying the ownership key at one of two levels: **owner-wide** (``(tenant, owner, agent
NULL)``: the principal's, or the group's, for every agent it owns) or **tenant-wide**
(``(tenant, NULL, NULL)``: every principal of the tenant). An entry records its ``name``,
``kind``, ``version``, ``content_hash``, ``manifest``, ``scan_result`` and ``enabled``; its
files are never stored in the row — they live in the blob store, addressed by the content hash.
Switching an entry on for one agent is a separate row at the agent level (``(tenant, NULL,
X)``), default off, so enabling an extension for one agent never enables it for another.

Every operation reads its pin (tenant, agent, owner, actor) from
``hermes_scope.current_scope()`` when it runs, never from a path, ``HERMES_HOME`` or an
argument, and the port exposes no way to pin anything else. A backend refuses a pin it cannot
serve with :class:`RegistryPinRefused`. Writes are authorised in two disjoint classes (the v1
rule): the agent's administration right — ``can_administer`` — reaches the agent's switches and
the owner-wide entries of the bound agent's owner, and nothing wider; a tenant-wide entry is
written only by the tenant's provisioning actor, a scope whose actor is
``System("provisioning")`` (:func:`require_provisioning` refuses every other actor, before the
target is looked up, so a refusal never reveals whether the entry exists; no
tenant-administration claim exists yet). A scope without ``can_administer`` is refused with
:class:`RegistryAdminRefused` for every write, likewise before the target is looked up. Reads
resolve through the one documented policy function
(:func:`entry_visible_to`): a tenant-wide entry is visible to every principal of the tenant, an
owner-wide one only on agents the same principal or group owns — a member's own entries never
reach a group's agent, because that agent's owner is the group, not the member.

A :class:`ExtensionRegistryProvider` supplies the registry. This is the interface and its local
backend only: no production caller goes through it yet, and nothing in config.yaml selects a
backend, so a provider plugin stores nothing that Hermes reads. Local installs keep today's
in-process plugin loading; the local backend's rows (``hermes_extension_registry_port_local``)
are not read by that path — and that backend keeps entries owner-wide, as the local owner's,
refusing the tenant level outright (one owner; nothing to share), which is the registry's
documented local exception. :func:`resolve_extension_registry` takes the backend's name
explicitly: ``"local"`` (or empty) is the local backend; any other name is a provider from
``plugins/extension_registry/<name>/`` or ``$HERMES_HOME/plugins/<name>/``, and a named provider
that is missing or unavailable is an error, never a fallback.

This port stores rows only. Validating a plugin, running its ``register`` in a sandbox, the
static scan, materialising files from the blob store and dispatching calls are separate,
later surfaces; ``install`` here records an entry the caller has already taken in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from hermes_scope import Group, Owner, Principal, System

#: What a registry entry can be: a user plugin, or a user MCP server definition.
KINDS = frozenset({"plugin", "mcp_server"})

#: The levels an entry may sit at. The per-agent switch is not one of them: it is a separate
#: row at the agent level, and an entry never sits there by design.
ENTRY_LEVELS = frozenset({"owner", "tenant"})

#: The one actor that may write a tenant-wide entry (the v1 rule): the tenant's provisioning.
PROVISIONING_ACTOR = System("provisioning")


class RegistryPinRefused(PermissionError):
    """The bound scope is not one this registry serves: another tenant, another agent, another
    owner, or (for a backend that requires one) no scope at all. Says nothing about whether the
    target exists."""


class RegistryAdminRefused(PermissionError):
    """The bound scope may not administer this agent, so it may not install, remove, disable or
    switch on an extension. Raised before the target is looked up, so the refusal is the same
    whether the entry exists, is foreign, or is missing."""


def require_provisioning(scope) -> None:
    """The v1 rule for writing a tenant-wide entry: only the tenant's provisioning actor — a
    scope whose actor is :data:`PROVISIONING_ACTOR` — may make one, and every other actor (any
    principal, administering or not) is refused with :class:`RegistryAdminRefused`. Backends
    call this before the target is looked up, so the refusal never says whether the name is
    taken; no tenant-administration claim exists yet, and when one does this is the one place
    it changes."""
    if scope is None or scope.actor != PROVISIONING_ACTOR:
        raise RegistryAdminRefused(
            "a tenant-wide user extension is written by the tenant's provisioning, "
            "not by an agent's administrator")


def entry_visible_to(owner: Owner, entry_owner: Optional[Owner]) -> bool:
    """The one read-resolution rule for registry entries, the same for every owner kind.

    A tenant-wide entry (``entry_owner`` ``None``) is visible to every principal of the tenant.
    An owner-wide entry is visible only on agents the same principal or group owns: the scope
    names the agent's owner, so a group's entries resolve on the group's agents for any member's
    turn, and a member's own owner-wide entries never resolve there — that agent's owner is the
    group, not the member. Changing this policy later is a change to this one function and its
    tests, not to any schema or backend.
    """
    return entry_owner is None or entry_owner == owner


def owner_key(owner: Optional[Owner]) -> Optional[tuple[str, str]]:
    """The hashable identity of an owner, for backends that key rows by it."""
    if owner is None:
        return None
    return ("group", owner.id) if isinstance(owner, Group) else ("principal", owner.id)


class ExtensionRegistry(ABC):
    """The user extension registry for the pinned agent: the entries visible to its owner, and
    this agent's per-agent switches.

    Rows are dicts with ``name``, ``kind``, ``version``, ``content_hash``, ``manifest``,
    ``scan_result``, ``enabled``, ``level`` (``"owner"`` or ``"tenant"``) and ``owner`` (the
    owning :class:`~hermes_scope.Principal` or :class:`~hermes_scope.Group`, ``None`` for a
    tenant-wide entry). One name may exist at both levels; the owner-wide entry is the one its
    owner's scopes resolve, and everyone else resolves the tenant-wide one.

    Who may write (the v1 rule): a scope with ``can_administer`` may write this agent's switches
    and the owner-wide entries of the bound agent's owner — its administration right reaches
    the agent and its owner, never the tenant. A tenant-wide entry is written only by the
    tenant's provisioning actor, a scope whose actor is
    :data:`~hermes_extension_registry_port.PROVISIONING_ACTOR`; any principal actor attempting
    one is refused with :class:`RegistryAdminRefused`, checked before the target is looked up,
    and for a named write once the resolved entry is known to be tenant-wide, before anything
    is changed. No tenant-administration claim exists yet; :func:`require_provisioning` is the
    one place that changes when one does.
    """

    def _check_install_args(self, *, name: str, kind: str, version: str, content_hash: str,
                            manifest: dict[str, Any], scan_result: Optional[dict[str, Any]],
                            level: str) -> None:
        """Reject a malformed entry before anything is written. Shared by every backend, so the
        registry never stores a row one backend accepts and another cannot serve."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("an extension entry needs a non-empty name")
        if kind not in KINDS:
            raise ValueError(f"an extension entry's kind must be one of {sorted(KINDS)}, not {kind!r}")
        if not isinstance(version, str) or not version:
            raise ValueError("an extension entry needs a version")
        if not isinstance(content_hash, str) or not content_hash:
            raise ValueError("an extension entry needs the content hash of its files")
        if not isinstance(manifest, dict):
            raise ValueError("an extension entry needs its manifest as a mapping")
        if scan_result is not None and not isinstance(scan_result, dict):
            raise ValueError("an extension entry's scan result must be a mapping or None")
        if level not in ENTRY_LEVELS:
            raise ValueError(f"an extension entry's level must be one of {sorted(ENTRY_LEVELS)}, not {level!r}")

    @abstractmethod
    def install(self, *, name: str, kind: str, version: str, content_hash: str,
                manifest: dict[str, Any], scan_result: Optional[dict[str, Any]] = None,
                level: str = "owner") -> dict[str, Any]:
        """Record an entry at *level* (``"owner"``: the bound agent's owner; ``"tenant"``: the
        whole tenant), replacing any entry already at that level and name. The entry starts
        ``enabled`` and switched off for every agent. An owner-wide install needs the agent's
        administration right; a tenant-wide one the provisioning actor (the v1 rule), refused
        before anything is read. This stores the row; validating the plugin, scanning it and
        discovering its tools are the caller's, and their refusals happen before this is
        called."""

    @abstractmethod
    def get(self, name: str) -> Optional[dict[str, Any]]:
        """The entry *name* resolves to for the pinned agent's owner, or ``None`` (missing and
        foreign look the same)."""

    @abstractmethod
    def list(self, *, kind: Optional[str] = None) -> list[dict[str, Any]]:
        """Every entry visible to the pinned agent's owner, sorted by name; *kind* narrows it."""

    @abstractmethod
    def set_enabled(self, name: str, enabled: bool) -> bool:
        """Offer or withdraw the resolved entry for its whole level; ``False`` when no visible
        entry has that name (a foreign entry answers the same as a missing one). The agent's
        administration right reaches the resolved entry when it is owner-wide; a tenant-wide
        one needs the provisioning actor (the v1 rule), refused once the entry is known to be
        tenant-wide, before anything is changed."""

    @abstractmethod
    def remove(self, name: str) -> bool:
        """Delete the resolved entry and every per-agent switch for it; ``False`` when no
        visible entry has that name. The agent's administration right reaches the resolved
        entry when it is owner-wide; a tenant-wide one needs the provisioning actor (the v1
        rule), refused once the entry is known to be tenant-wide, before anything is changed."""

    @abstractmethod
    def set_agent_enabled(self, name: str, enabled: bool) -> bool:
        """Turn this agent's per-agent switch for the resolved entry on or off; ``False`` when no
        visible entry has that name. The switch is this agent's own row, keyed by the entry's
        identity — its level, owner and name — whatever level the entry sits at: other agents
        keep their own, another entry the name resolves to keeps its own, and the agent's
        administration right is the one that writes it."""

    @abstractmethod
    def agent_enabled(self, name: str) -> bool:
        """Whether this agent's switch for the resolved entry is on; ``False`` for a missing or
        foreign entry, or one whose switch is off. Any bound scope may read this."""

    @abstractmethod
    def enabled_extensions(self, *, kind: Optional[str] = None) -> list[dict[str, Any]]:
        """The entries this agent offers next session: one entry per name — the one
        :meth:`get` resolves to, by the same precedence, never both levels of one name —
        narrowed by *kind*, offered when it is entry-``enabled`` and switched on, sorted by
        name. The caller still re-checks ``content_hash`` and ``scan_result`` before running
        anything. Any bound scope may read this."""


class ExtensionRegistryProvider(ABC):
    """The plugin kind: supplies the registry. ``name``, ``open`` and ``assert_isolating`` are
    required; growth stays additive."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. ``"local"``."""

    def is_available(self) -> bool:
        """Whether this backend can run here. No network calls."""
        return True

    @abstractmethod
    def open(self, home: Any = None) -> ExtensionRegistry:
        """The registry for the bound scope. *home* is the local backend's profile home (the
        ambient one when omitted); a backend with no local file ignores it. Every operation is
        pinned when it runs; opening pins nothing."""

    @abstractmethod
    def assert_isolating(self) -> None:
        """Raise unless this backend isolates tenants, agents and owners from each other in
        storage itself (a Postgres backend whose role is a superuser or has ``BYPASSRLS`` fails).
        Required, with no default: a backend that has not checked must not pass. Nothing calls it
        yet: the first caller that serves a scope from a resolved provider must call it, and
        refuse the provider when it raises, before :meth:`open`."""


def resolve_extension_registry(name: str) -> ExtensionRegistryProvider:
    """The provider named *name* (``"local"`` or empty: the local backend). Raises ``LookupError``
    with the reason for a named provider that is missing, refused or unavailable: the registry
    never falls back to another backend."""
    name = name.strip()
    if name in ("", "local"):
        from hermes_extension_registry_port_local import LocalExtensionRegistryProvider
        return LocalExtensionRegistryProvider()
    from plugins.extension_registry import ExtensionRegistryRefused, load_extension_registry
    try:
        provider = load_extension_registry(name)
    except ExtensionRegistryRefused as refused:
        raise LookupError(f"extension registry {name!r}: {refused}") from refused
    if provider is None:
        raise LookupError(f"extension registry {name!r}: no such extension registry provider")
    if not provider.is_available():
        raise LookupError(f"extension registry {name!r} is not available here")
    return provider
