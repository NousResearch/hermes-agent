"""Broken extension registry backends the conformance suite must reject (``test_mutants.py``).

Each passed an earlier version of the suite. The local backend's pin forgets the agent
(``local_pin_forgets_agent``: another profile's scope, and the ambient pin of a scopeless
caller, are served), the tenant (``local_pin_forgets_tenant``) or the owner
(``local_pin_forgets_owner``), ``local_admin_not_required`` lets a scope that may not
administer install and switch, and ``local_accepts_tenant_entries`` stores tenant-wide
entries despite the registry's local exception. ``local_resets_malformed_rows`` reads a
file whose row containers are of the wrong type as if it held no rows, so a write resets
it around the defaults. The
reference backend accepts a principal administrator's
tenant-wide writes (``reference_tenant_writes_without_provisioning``: the tenant gate is gone),
its named writes refuse a missing name as if it were taken
(``reference_refuses_names_the_reads_do_not_return``: a refusal for a name the scope's reads do
not return), its resolution ignores the owner
(``reference_owner_blind``: every entry of the tenant reaches every principal), the per-agent
switch forgets its agent (``reference_switch_forgets_agent``: one switch shared by every
agent), owner-wide entries resolve by the actor instead of the agent's owner
(``reference_resolves_by_actor``: a member's own entries reach a group agent), and its view pin
forgets the agent, the tenant or the owner. The offered list of either backend offers every
visible entry of a name instead of the one ``get`` resolves to
(``reference_offers_every_visible_entry``, ``local_offers_every_visible_entry``), and the
switch rows forget the entry's identity (``reference_switch_forgets_entry_identity``,
``local_switch_forgets_entry_identity``: keyed by the name alone). Each mutant is applied by
the fixtures when ``HERMES_EXTENSION_REGISTRY_MUTANT`` names it.
"""

from hermes_extension_registry_port import RegistryPinRefused, owner_key
from hermes_scope import LOCAL_OWNER, LOCAL_TENANT, current_scope


def _local_pin_forgets_agent(monkeypatch):
    """The local pin keeps the tenant and owner checks but never compares the agent, and serves
    a scopeless caller instead of falling back to the ambient profile."""
    from hermes_extension_registry_port_local import LocalExtensionRegistry

    def _pin(self):
        scope = current_scope()
        if scope is not None and scope.tenant != LOCAL_TENANT:
            raise RegistryPinRefused(f"a local extension registry serves tenant {LOCAL_TENANT!r} only")
        owner = LOCAL_OWNER if scope is None or scope.owner is None else scope.owner
        if owner != LOCAL_OWNER:
            raise RegistryPinRefused("a local extension registry serves the local owner only")
        return scope
    monkeypatch.setattr(LocalExtensionRegistry, "_pin", _pin)


def _local_pin_forgets_tenant(monkeypatch):
    """The local pin compares the agent and owner but not the tenant."""
    from hermes_constants import get_hermes_home, hermes_home_key
    from hermes_extension_registry_port_local import LocalExtensionRegistry

    def _pin(self):
        scope = current_scope()
        agent = scope.agent if scope is not None else get_hermes_home()
        if hermes_home_key(agent) != self._home_key:
            raise RegistryPinRefused("the bound agent is not this extension registry's profile")
        owner = LOCAL_OWNER if scope is None or scope.owner is None else scope.owner
        if owner != LOCAL_OWNER:
            raise RegistryPinRefused("a local extension registry serves the local owner only")
        return scope
    monkeypatch.setattr(LocalExtensionRegistry, "_pin", _pin)


def _local_pin_forgets_owner(monkeypatch):
    """The local pin compares the tenant and agent but not the owner."""
    from hermes_constants import get_hermes_home, hermes_home_key
    from hermes_extension_registry_port_local import LocalExtensionRegistry

    def _pin(self):
        scope = current_scope()
        if scope is not None and scope.tenant != LOCAL_TENANT:
            raise RegistryPinRefused(f"a local extension registry serves tenant {LOCAL_TENANT!r} only")
        agent = scope.agent if scope is not None else get_hermes_home()
        if hermes_home_key(agent) != self._home_key:
            raise RegistryPinRefused("the bound agent is not this extension registry's profile")
        return scope
    monkeypatch.setattr(LocalExtensionRegistry, "_pin", _pin)


def _local_admin_not_required(monkeypatch):
    """A scope that may not administer installs and switches anyway."""
    from hermes_extension_registry_port_local import LocalExtensionRegistry
    monkeypatch.setattr(LocalExtensionRegistry, "_administer", lambda self, scope: None)


def _local_accepts_tenant_entries(monkeypatch):
    """The local backend stores tenant-wide entries anyway: the local exception's level refusal
    is gone."""
    from hermes_extension_registry_port_local import LocalExtensionRegistry
    monkeypatch.setattr(LocalExtensionRegistry, "_check_levels", lambda self, level: None)


def _local_resets_malformed_rows(monkeypatch):
    """The local backend reads a file whose row containers are of the wrong type as if it
    held no rows: a write resets the file around the defaults, silently dropping the
    rows it could not read."""
    from hermes_extension_registry_port_local import LocalExtensionRegistry

    def _load(self):
        from utils import read_json_or_empty
        data = read_json_or_empty(self.path)
        if not isinstance(data.get("entries"), list):
            data["entries"] = []
        if not isinstance(data.get("switches"), dict):
            data["switches"] = {}
        return data
    monkeypatch.setattr(LocalExtensionRegistry, "_load", _load)


def _reference_tenant_writes_without_provisioning(monkeypatch):
    """A principal administering its own agent installs, disables and removes tenant-wide
    entries anyway: the tenant gate is gone."""
    from hermes_conformance.extensions import reference
    monkeypatch.setattr(reference, "require_provisioning", lambda scope: None)


def _reference_refuses_names_the_reads_do_not_return(monkeypatch):
    """The named writes' tenant gate refuses a missing name as if it were taken, before
    anything is resolved: a refusal for a name the same scope's ``get`` does not return,
    saying the name exists when the reads say it does not."""
    from hermes_conformance.extensions import reference
    from hermes_extension_registry_port import require_provisioning

    def _set_enabled(self, name, enabled):
        scope = self._pin()
        self._administer(scope)
        with self.store.lock:
            row = self._resolve(scope, name)
            if row is None or row["level"] == "tenant":
                require_provisioning(scope)
            assert row is not None  # the refusal above raised for the missing name
            row["enabled"] = bool(enabled)
            return True

    def _remove(self, name):
        scope = self._pin()
        self._administer(scope)
        with self.store.lock:
            row = self._resolve(scope, name)
            if row is None or row["level"] == "tenant":
                require_provisioning(scope)
            assert row is not None  # the refusal above raised for the missing name
            del self.store.entries[(owner_key(row["owner"]), name)]
            for agent in {key[0] for key in self.store.switches}:
                self.store.switches.pop(self._switch_key(agent, row), None)
            return True

    monkeypatch.setattr(reference.ReferenceExtensionRegistry, "set_enabled", _set_enabled)
    monkeypatch.setattr(reference.ReferenceExtensionRegistry, "remove", _remove)


def _reference_owner_blind(monkeypatch):
    """Every entry of the tenant is visible to every principal."""
    from hermes_conformance.extensions.reference import ReferenceExtensionRegistry
    monkeypatch.setattr(ReferenceExtensionRegistry, "_visible", lambda self, scope, key: True)


def _reference_switch_forgets_agent(monkeypatch):
    """One switch row per entry, shared by every agent of the tenant."""
    from hermes_conformance.extensions.reference import ReferenceExtensionRegistry
    monkeypatch.setattr(ReferenceExtensionRegistry, "_switch_key",
                       lambda self, agent, entry: ("everyone", entry["level"],
                                                   owner_key(entry["owner"]), entry["name"]))


def _reference_switch_forgets_entry_identity(monkeypatch):
    """The switch row is keyed by the name alone, not the entry's identity: removing one
    owner's entry drops another owner's switch, and a new entry of the name inherits one."""
    from hermes_conformance.extensions.reference import ReferenceExtensionRegistry
    monkeypatch.setattr(ReferenceExtensionRegistry, "_switch_key",
                       lambda self, agent, entry: (agent, entry["name"]))


def _local_switch_forgets_entry_identity(monkeypatch):
    """The local switch row is keyed by the name alone, not the entry's identity."""
    from hermes_extension_registry_port_local import LocalExtensionRegistry
    monkeypatch.setattr(LocalExtensionRegistry, "_switch_key", lambda self, entry: entry["name"])


def _reference_offers_every_visible_entry(monkeypatch):
    """The offered list offers every visible entry of a name — both levels — reading the
    switch by the name alone, instead of the one entry ``get`` resolves to."""
    from hermes_conformance.extensions.reference import ReferenceExtensionRegistry

    def _enabled_extensions(self, kind=None):
        scope = self._pin()
        with self.store.lock:
            return [self._row(r) for r in self._visible_rows(scope, kind=kind)
                    if r["enabled"] and self.store.switches.get((self.agent, r["name"]))]
    monkeypatch.setattr(ReferenceExtensionRegistry, "enabled_extensions", _enabled_extensions)


def _local_offers_every_visible_entry(monkeypatch):
    """The local offered list offers every visible entry of a name — both levels — reading
    the switch by the name alone, instead of the one entry ``get`` resolves to."""
    from hermes_extension_registry_port_local import LocalExtensionRegistry

    def _enabled_extensions(self, kind=None):
        scope = self._pin()
        owner = LOCAL_OWNER if scope is None or scope.owner is None else scope.owner
        data = self._load()
        return [r for r in self._visible(data, owner, kind=kind)
                if r["enabled"] and data["switches"].get(r["name"])]
    monkeypatch.setattr(LocalExtensionRegistry, "enabled_extensions", _enabled_extensions)


def _reference_resolves_by_actor(monkeypatch):
    """Owner-wide entries resolve for the actor that installed them, not for the agent's owner:
    a member's own entries reach a group agent."""
    from hermes_conformance.extensions.reference import ReferenceExtensionRegistry

    def _visible(self, scope, entry_owner_key):
        return entry_owner_key is None or entry_owner_key == owner_key(scope.actor)
    monkeypatch.setattr(ReferenceExtensionRegistry, "_visible", _visible)


def _reference_pin_forgets_agent(monkeypatch):
    """The view pin checks the tenant and owner but not the agent."""
    from hermes_conformance.extensions.reference import ReferenceExtensionRegistry

    def _pin(self):
        scope = current_scope()
        if scope is None:
            raise RegistryPinRefused("an extension registry operation needs a scope bound")
        if scope.tenant != self.store.tenant:
            raise RegistryPinRefused(f"this extension registry serves tenant {self.store.tenant!r} only")
        if scope.owner != self.owner:
            raise RegistryPinRefused("the bound scope's owner is not this agent's owner")
        return scope
    monkeypatch.setattr(ReferenceExtensionRegistry, "_pin", _pin)


def _reference_pin_forgets_tenant(monkeypatch):
    from hermes_conformance.extensions.reference import ReferenceExtensionRegistry

    def _pin(self):
        scope = current_scope()
        if scope is None:
            raise RegistryPinRefused("an extension registry operation needs a scope bound")
        if scope.agent != self.agent:
            raise RegistryPinRefused("the bound agent is not this extension registry's agent")
        if scope.owner != self.owner:
            raise RegistryPinRefused("the bound scope's owner is not this agent's owner")
        return scope
    monkeypatch.setattr(ReferenceExtensionRegistry, "_pin", _pin)


def _reference_pin_forgets_owner(monkeypatch):
    from hermes_conformance.extensions.reference import ReferenceExtensionRegistry

    def _pin(self):
        scope = current_scope()
        if scope is None:
            raise RegistryPinRefused("an extension registry operation needs a scope bound")
        if scope.tenant != self.store.tenant:
            raise RegistryPinRefused(f"this extension registry serves tenant {self.store.tenant!r} only")
        if scope.agent != self.agent:
            raise RegistryPinRefused("the bound agent is not this extension registry's agent")
        return scope
    monkeypatch.setattr(ReferenceExtensionRegistry, "_pin", _pin)


MUTANTS = {
    "local_pin_forgets_agent": _local_pin_forgets_agent,
    "local_pin_forgets_tenant": _local_pin_forgets_tenant,
    "local_pin_forgets_owner": _local_pin_forgets_owner,
    "local_admin_not_required": _local_admin_not_required,
    "local_accepts_tenant_entries": _local_accepts_tenant_entries,
    "local_resets_malformed_rows": _local_resets_malformed_rows,
    "local_switch_forgets_entry_identity": _local_switch_forgets_entry_identity,
    "local_offers_every_visible_entry": _local_offers_every_visible_entry,
    "reference_tenant_writes_without_provisioning": _reference_tenant_writes_without_provisioning,
    "reference_refuses_names_the_reads_do_not_return": _reference_refuses_names_the_reads_do_not_return,
    "reference_owner_blind": _reference_owner_blind,
    "reference_switch_forgets_agent": _reference_switch_forgets_agent,
    "reference_switch_forgets_entry_identity": _reference_switch_forgets_entry_identity,
    "reference_offers_every_visible_entry": _reference_offers_every_visible_entry,
    "reference_resolves_by_actor": _reference_resolves_by_actor,
    "reference_pin_forgets_agent": _reference_pin_forgets_agent,
    "reference_pin_forgets_tenant": _reference_pin_forgets_tenant,
    "reference_pin_forgets_owner": _reference_pin_forgets_owner,
}


def apply_mutant(monkeypatch):
    import os
    name = os.environ.get("HERMES_EXTENSION_REGISTRY_MUTANT")
    if name:
        MUTANTS[name](monkeypatch)
