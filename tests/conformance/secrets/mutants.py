"""Broken secret store backends the conformance suite must reject (``test_mutants.py``).

Each passed an earlier version of the suite. ``local_reads_ignore_tenant`` and
``local_reads_ignore_the_bound_agent`` leak reads through a tenant- or agent-blind pin;
``local_writes_skip_the_administration_check`` lets a non-administering scope change credentials;
``local_accepts_undeclared_levels`` writes an owner-wide or tenant-wide secret onto the agent
level instead of refusing it; ``local_refusal_leaks_the_value`` and
``local_logs_the_value_on_a_refusal`` publish the value on the write refusal path. Against the
leveled reference backend: ``reference_reads_ignore_tenant`` drops the tenant half of the pin,
``reference_writes_skip_the_administration_check`` drops the administration check,
``reference_owner_wide_resolves_everywhere`` resolves another owner's entries,
``reference_owner_wide_keys_on_the_actor`` keys an owner-wide write on the acting principal, and
``reference_drops_level_precedence`` reads the tenant-wide value over the agent's own. Each
mutant is applied by the ``backend`` fixtures when ``HERMES_SECRETS_MUTANT`` names it.
"""

import logging


def local_reads_ignore_tenant(monkeypatch):
    """The local read pin checks the bound agent but not the tenant: another tenant naming the
    home reads its secrets."""
    from hermes_constants import get_hermes_home, hermes_home_key
    from hermes_scope import current_scope
    from hermes_secret_port import SecretPinRefused
    from hermes_secret_port_local import LocalSecretStore

    def get(self, name):
        scope = current_scope()
        agent = scope.agent if scope is not None else get_hermes_home()
        if hermes_home_key(agent) != self._home_key:
            raise SecretPinRefused("the bound agent is not this secret store's profile")
        return self._resolved().get(name)

    def names(self):
        return sorted(self._resolved())

    def get_auth(self, provider):
        from hermes_cli import auth
        return auth._load_provider_state_with_source(auth._load_auth_store(), provider)[0]

    monkeypatch.setattr(LocalSecretStore, "get", get)
    monkeypatch.setattr(LocalSecretStore, "names", names)
    monkeypatch.setattr(LocalSecretStore, "get_auth", get_auth)


def local_reads_ignore_the_bound_agent(monkeypatch):
    """The local reads skip the pin check entirely: any bound profile or the ambient one reads."""
    from hermes_secret_port_local import LocalSecretStore

    monkeypatch.setattr(LocalSecretStore, "get", lambda self, name: self._resolved().get(name))
    monkeypatch.setattr(LocalSecretStore, "names", lambda self: sorted(self._resolved()))

    def get_auth(self, provider):
        from hermes_cli import auth
        return auth._load_provider_state_with_source(auth._load_auth_store(), provider)[0]
    monkeypatch.setattr(LocalSecretStore, "get_auth", get_auth)


def local_writes_skip_the_administration_check(monkeypatch):
    """The local pin keeps its tenant and agent halves but never asks the administration
    right: a non-administering scope on the same profile changes credentials."""
    from hermes_constants import get_hermes_home, hermes_home_key
    from hermes_scope import LOCAL_TENANT, current_scope
    from hermes_secret_port import SecretPinRefused
    from hermes_secret_port_local import LocalSecretStore

    def _pin(self, *, administering=False):
        scope = current_scope()
        if scope is not None and scope.tenant != LOCAL_TENANT:
            raise SecretPinRefused(f"a local secret store serves tenant {LOCAL_TENANT!r} only")
        agent = scope.agent if scope is not None else get_hermes_home()
        if hermes_home_key(agent) != self._home_key:
            raise SecretPinRefused("the bound agent is not this secret store's profile")
    monkeypatch.setattr(LocalSecretStore, "_pin", _pin)


def local_accepts_undeclared_levels(monkeypatch):
    """A write naming the owner-wide or tenant-wide level lands on the agent level instead of
    being refused: the backend's declared levels mean nothing."""
    from hermes_secret_port import LEVELS
    from hermes_secret_port_local import LocalSecretStore

    def _require_agent_level(self, level):
        if level not in LEVELS:
            raise ValueError(f"not a secret level: {level!r}")
    monkeypatch.setattr(LocalSecretStore, "_require_agent_level", _require_agent_level)


def local_refusal_leaks_the_value(monkeypatch):
    """The administration refusal names the value it refused, in the exception message."""
    from hermes_secret_port import SecretAdminRefused
    from hermes_secret_port_local import LocalSecretStore
    original = LocalSecretStore.set

    def set(self, name, value, *, level):
        try:
            return original(self, name, value, level=level)
        except SecretAdminRefused:
            raise SecretAdminRefused(f"refused to store {name}={value!r}: no administration right")
    monkeypatch.setattr(LocalSecretStore, "set", set)


def local_logs_the_value_on_a_refusal(monkeypatch):
    """The administration refusal logs the value it refused."""
    from hermes_secret_port import SecretAdminRefused
    from hermes_secret_port_local import LocalSecretStore
    original = LocalSecretStore.set

    def set(self, name, value, *, level):
        try:
            return original(self, name, value, level=level)
        except SecretAdminRefused:
            logging.getLogger(__name__).warning("refused to set %s to %r", name, value)
            raise
    monkeypatch.setattr(LocalSecretStore, "set", set)


def reference_reads_ignore_tenant(monkeypatch):
    """The reference pin keeps the agent half but drops the tenant: another tenant naming the
    agent reads its secrets."""
    from hermes_scope import current_scope
    from hermes_secret_port import SecretAdminRefused, SecretPinRefused
    from tests.conformance.secrets.reference import MemorySecretStore

    def _pin(self, *, administering=False):
        scope = current_scope()
        if scope is None:
            raise SecretPinRefused("no scope is bound; this store serves none by default")
        if scope.agent != self.agent:
            raise SecretPinRefused("the bound agent is not this store's")
        if administering and not scope.can_administer:
            raise SecretAdminRefused("this scope may not administer this agent's credentials")
        return scope
    monkeypatch.setattr(MemorySecretStore, "_pin", _pin)


def reference_writes_skip_the_administration_check(monkeypatch):
    """The reference pin keeps its tenant and agent halves but never asks the administration
    right."""
    from hermes_scope import current_scope
    from hermes_secret_port import SecretPinRefused
    from tests.conformance.secrets.reference import MemorySecretStore

    def _pin(self, *, administering=False):
        scope = current_scope()
        if scope is None:
            raise SecretPinRefused("no scope is bound; this store serves none by default")
        if scope.tenant != "t":
            raise SecretPinRefused("this store serves tenant 't' only")
        if scope.agent != self.agent:
            raise SecretPinRefused("the bound agent is not this store's")
        return scope
    monkeypatch.setattr(MemorySecretStore, "_pin", _pin)


def reference_owner_wide_resolves_everywhere(monkeypatch):
    """The owner-wide level is read as one bucket for the whole tenant: another owner's entries
    resolve on this agent too."""
    from tests.conformance.secrets.reference import MemorySecretStore

    def get(self, name):
        scope = self._pin()
        if name in self.state["agent"].setdefault(scope.agent, {}):
            return self.state["agent"][scope.agent][name]
        for bucket in self.state["owner"].values():
            if name in bucket:
                return bucket[name]
        return self.state["tenant"].get(None, {}).get(name)

    def names(self):
        scope = self._pin()
        found = set(self.state["agent"].get(scope.agent, {}))
        for bucket in self.state["owner"].values():
            found.update(bucket)
        found.update(self.state["tenant"].get(None, {}))
        return sorted(found)
    monkeypatch.setattr(MemorySecretStore, "get", get)
    monkeypatch.setattr(MemorySecretStore, "names", names)


def reference_owner_wide_keys_on_the_actor(monkeypatch):
    """An owner-wide write keys on the acting principal, not the owner the scope names, so a
    group's members' writes never reach the group's other agents."""
    from hermes_secret_port import LEVELS
    from tests.conformance.secrets.reference import MemorySecretStore
    original = MemorySecretStore._place

    def _place(self, level):
        if level not in LEVELS:
            raise ValueError(f"not a secret level: {level!r}")
        if level == "owner":
            scope = self._pin(administering=True)
            return "owner", ("Principal", scope.actor.id)
        return original(self, level)
    monkeypatch.setattr(MemorySecretStore, "_place", _place)


def reference_drops_level_precedence(monkeypatch):
    """Reads scan the levels from the widest to the narrowest, so a tenant-wide value hides the
    agent's own."""
    from tests.conformance.secrets.reference import MemorySecretStore

    def get(self, name):
        scope = self._pin()
        for level, key in reversed(self._levels(scope)):
            if name in self.state[level].setdefault(key, {}):
                return self.state[level][key][name]
        return None
    monkeypatch.setattr(MemorySecretStore, "get", get)


MUTANTS = {
    "local_reads_ignore_tenant": local_reads_ignore_tenant,
    "local_reads_ignore_the_bound_agent": local_reads_ignore_the_bound_agent,
    "local_writes_skip_the_administration_check": local_writes_skip_the_administration_check,
    "local_accepts_undeclared_levels": local_accepts_undeclared_levels,
    "local_refusal_leaks_the_value": local_refusal_leaks_the_value,
    "local_logs_the_value_on_a_refusal": local_logs_the_value_on_a_refusal,
    "reference_reads_ignore_tenant": reference_reads_ignore_tenant,
    "reference_writes_skip_the_administration_check": reference_writes_skip_the_administration_check,
    "reference_owner_wide_resolves_everywhere": reference_owner_wide_resolves_everywhere,
    "reference_owner_wide_keys_on_the_actor": reference_owner_wide_keys_on_the_actor,
    "reference_drops_level_precedence": reference_drops_level_precedence,
}


def apply_mutant(monkeypatch):
    import os
    name = os.environ.get("HERMES_SECRETS_MUTANT")
    if name:
        MUTANTS[name](monkeypatch)
