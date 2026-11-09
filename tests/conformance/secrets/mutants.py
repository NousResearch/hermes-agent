"""Broken secret store backends the conformance suite must reject (``test_mutants.py``).

Each passed an earlier version of the suite. ``local_reads_ignore_tenant`` and
``local_reads_ignore_the_bound_agent`` leak reads through a tenant- or agent-blind pin;
``local_writes_skip_the_administration_check`` lets a non-administering scope change credentials;
``local_accepts_undeclared_levels`` writes an owner-wide or tenant-wide secret onto the agent
level instead of refusing it; ``local_refusal_leaks_the_value`` and
``local_logs_the_value_on_a_refusal`` publish the value on the write refusal path, and
``local_refusal_keeps_the_value_in_frame_locals`` / ``reference_refusal_keeps_the_value_in_frame_locals``
leave it bound in a frame the refusal propagates through, where a traceback-locals renderer
reads it. Against the
leveled reference backend: ``reference_reads_ignore_tenant`` drops the tenant half of the pin,
``reference_writes_skip_the_administration_check`` drops the administration check,
``reference_owner_wide_resolves_everywhere`` resolves another owner's entries,
``reference_owner_wide_keys_on_the_actor`` keys an owner-wide write on the acting principal, and
``reference_drops_level_precedence`` reads the tenant-wide value over the agent's own. Each
mutant declares, next to how it is applied, the suite test that must kill it; the harness
(``test_mutants.py``) asserts that test is among the failures. Each mutant is applied by the
``backend`` fixtures when ``HERMES_SECRETS_MUTANT`` names it.
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


def local_refusal_keeps_the_value_in_frame_locals(monkeypatch):
    """A credential write reaches the store through a frame that never clears the caller's
    value, so a refusal propagates with the plaintext still bound in that frame's locals —
    readable by anything that renders traceback locals."""
    from hermes_secret_port_local import LocalSecretStore
    original_set, original_set_auth = LocalSecretStore.set, LocalSecretStore.set_auth

    def set(self, name, value, *, level):
        return original_set(self, name, value, level=level)

    def set_auth(self, provider, record):
        return original_set_auth(self, provider, record)

    monkeypatch.setattr(LocalSecretStore, "set", set)
    monkeypatch.setattr(LocalSecretStore, "set_auth", set_auth)


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


def reference_refusal_keeps_the_value_in_frame_locals(monkeypatch):
    """The reference write leaves the caller's value (or record) bound in a frame the refusal
    propagates through — the store's own clearing never reaches it."""
    from tests.conformance.secrets.reference import MemorySecretStore
    original_set, original_set_auth = MemorySecretStore.set, MemorySecretStore.set_auth

    def set(self, name, value, *, level):
        return original_set(self, name, value, level=level)

    def set_auth(self, provider, record):
        return original_set_auth(self, provider, record)

    monkeypatch.setattr(MemorySecretStore, "set", set)
    monkeypatch.setattr(MemorySecretStore, "set_auth", set_auth)


def _local(node: str) -> str:
    """The node id of a local-suite test (``TestLocalSecrets``)."""
    return f"tests/conformance/secrets/test_local.py::TestLocalSecrets::{node}"


def _reference(node: str) -> str:
    """The node id of a reference-suite test (``TestReferencePinning``/``TestReferenceLeveled``)."""
    return f"tests/conformance/secrets/test_reference.py::{node}"


#: Each entry: how to apply the broken backend, and the suite test that must kill it (its node
#: id). The harness asserts that declared test is among the failures with the mutant applied —
#: a mutant the suite rejects for some other reason alone has not been shown killed for the
#: reason it exists.
MUTANTS = {
    "local_reads_ignore_tenant": (
        local_reads_ignore_tenant,
        _local("test_every_read_is_refused_for_any_pin_but_its_agents[get]")),
    "local_reads_ignore_the_bound_agent": (
        local_reads_ignore_the_bound_agent,
        _local("test_every_read_is_refused_for_any_pin_but_its_agents[get]")),
    "local_writes_skip_the_administration_check": (
        local_writes_skip_the_administration_check,
        _local("test_every_write_is_refused_without_administration_and_changes_nothing[set]")),
    "local_accepts_undeclared_levels": (
        local_accepts_undeclared_levels,
        _local("test_a_write_at_a_level_the_backend_does_not_hold_is_refused")),
    "local_refusal_leaks_the_value": (
        local_refusal_leaks_the_value,
        _local("test_no_secret_value_appears_in_log_lines_or_exception_messages")),
    "local_logs_the_value_on_a_refusal": (
        local_logs_the_value_on_a_refusal,
        _local("test_no_secret_value_appears_in_log_lines_or_exception_messages")),
    "local_refusal_keeps_the_value_in_frame_locals": (
        local_refusal_keeps_the_value_in_frame_locals,
        _local("test_no_refusal_traceback_frame_holds_the_value_or_record")),
    "reference_reads_ignore_tenant": (
        reference_reads_ignore_tenant,
        _reference("TestReferencePinning::test_every_read_is_refused_for_any_pin_but_its_agents[get]")),
    "reference_writes_skip_the_administration_check": (
        reference_writes_skip_the_administration_check,
        _reference("TestReferencePinning::test_every_write_is_refused_without_administration_and_changes_nothing[set]")),
    "reference_owner_wide_resolves_everywhere": (
        reference_owner_wide_resolves_everywhere,
        _reference("TestReferenceLeveled::test_an_owner_wide_secret_reaches_no_agent_the_owner_does_not_own")),
    "reference_owner_wide_keys_on_the_actor": (
        reference_owner_wide_keys_on_the_actor,
        _reference("TestReferenceLeveled::test_an_owner_wide_write_on_a_group_agent_keys_on_the_owners_scope_not_the_actors")),
    "reference_drops_level_precedence": (
        reference_drops_level_precedence,
        _reference("TestReferenceLeveled::test_the_agent_level_shadows_the_owner_and_tenant_levels")),
    "reference_refusal_keeps_the_value_in_frame_locals": (
        reference_refusal_keeps_the_value_in_frame_locals,
        _reference("TestReferencePinning::test_no_refusal_traceback_frame_holds_the_value_or_record")),
}


def apply_mutant(monkeypatch):
    import os
    name = os.environ.get("HERMES_SECRETS_MUTANT")
    if name:
        MUTANTS[name][0](monkeypatch)
