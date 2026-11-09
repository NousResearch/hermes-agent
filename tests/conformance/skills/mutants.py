"""Broken local and reference backends the skills conformance suite must reject
(``test_mutants.py``).

Each passed an earlier version of the suite. ``unpinned_reads``, ``reads_ignore_tenant``,
``put_ignores_pin``, ``put_file_ignores_pin`` and ``delete_ignores_pin`` are the pin holes;
``put_commits_before_refusal`` writes before raising the refusal; ``accepts_other_levels``
stores a foreign-level write as the agent's own. The two reference mutants break resolution:
``tenant_first_resolution`` consults the tenant's shelf first, ``foreign_owner_resolves`` lets
any owner's owner-wide skill resolve. Each mutant is applied by the ``backend`` and ``resolver``
fixtures when ``HERMES_SKILLS_MUTANT`` names it."""

import contextvars


def _agent_pin(self):
    """The pin's agent half alone: another tenant naming the store's home passes."""
    from hermes_constants import get_hermes_home, hermes_home_key
    from hermes_scope import current_scope
    from hermes_state_port import StoragePinRefused
    scope = current_scope()
    agent = scope.agent if scope is not None else get_hermes_home()
    if hermes_home_key(agent) != self._home_key:
        raise StoragePinRefused("the bound agent is not this skills index's profile")


def _with_pin(monkeypatch, method, pin):
    """One :class:`~hermes_skills_port_local.LocalSkillIndex` operation runs with its pin check
    replaced by *pin* (a callable of the index, or ``None`` for no check at all)."""
    from hermes_skills_port_local import LocalSkillIndex
    original = getattr(LocalSkillIndex, method)

    def call(self, *args, **kwargs):
        saved = self._pin
        self._pin = (lambda: None) if pin is None else (lambda: pin(self))
        try:
            return original(self, *args, **kwargs)
        finally:
            self._pin = saved
    monkeypatch.setattr(LocalSkillIndex, method, call)


def _unpinned_reads(monkeypatch):
    """list, get and read_body skip the pin check entirely."""
    for method in ("list", "get", "read_body"):
        _with_pin(monkeypatch, method, None)


def _reads_ignore_tenant(monkeypatch):
    """Every read checks the bound agent but not the tenant: another tenant naming the home
    reads the agent's skills."""
    for method in ("list", "get", "read_body"):
        _with_pin(monkeypatch, method, _agent_pin)


def _put_ignores_pin(monkeypatch):
    """put skips the pin check: a foreign pin writes the agent's skills."""
    _with_pin(monkeypatch, "put", None)


def _put_file_ignores_pin(monkeypatch):
    """put_file skips the pin check."""
    _with_pin(monkeypatch, "put_file", None)


def _delete_ignores_pin(monkeypatch):
    """delete skips the pin check: a foreign pin removes the agent's skills."""
    _with_pin(monkeypatch, "delete", None)


def _put_commits_before_refusal(monkeypatch):
    """put performs today's write under its own profile even for a foreign pin, then raises the
    refusal: the caller sees the refusal, the skill changed anyway."""
    from agent.runtime_scope import bind_profile_scope
    from hermes_skills_port_local import LocalSkillIndex
    from hermes_state_port import StoragePinRefused
    original = LocalSkillIndex.put

    def put(self, name, content, *, category=None, level="agent"):
        try:
            self._pin()
        except StoragePinRefused:
            def write():
                with bind_profile_scope(self.home):
                    return original(self, name, content, category=category, level=level)
            contextvars.Context().run(write)
            raise
        return original(self, name, content, category=category, level=level)
    monkeypatch.setattr(LocalSkillIndex, "put", put)


def _accepts_other_levels(monkeypatch):
    """A write naming an owner-wide or tenant-wide level lands on the agent's own shelf instead
    of being refused."""
    from hermes_skills_port import SKILL_LEVELS, SkillsLevelRefused
    from hermes_skills_port_local import LocalSkillIndex

    def store_level(self, level):
        if level not in SKILL_LEVELS:
            raise SkillsLevelRefused(f"a skills index stores levels {sorted(SKILL_LEVELS)} only")
    monkeypatch.setattr(LocalSkillIndex, "_store_level", store_level)


def _tenant_first_resolution(monkeypatch):
    """The reference backend consults the tenant's shelf before the agent's own."""
    from hermes_conformance.skills.reference import ReferenceSkillsIndex
    original = ReferenceSkillsIndex._read_keys

    def read_keys(self, scope):
        order = {"tenant": 0, "owner": 1, "agent": 2}
        return sorted(original(self, scope), key=lambda key: order[key[0]])
    monkeypatch.setattr(ReferenceSkillsIndex, "_read_keys", read_keys)


def _foreign_owner_resolves(monkeypatch):
    """The reference backend resolves any owner's owner-wide skill, not just the agent's own
    owner's."""
    from hermes_conformance.skills.reference import ReferenceSkillsIndex
    original = ReferenceSkillsIndex._resolved

    def resolved(self, scope):
        view = original(self, scope)
        tenant = self.store.setdefault(scope.tenant, {})
        for key, shelf in tenant.items():
            if key[0] == "owner":
                for name, stored in shelf.items():
                    view.setdefault(name, ("owner", stored))
        return view
    monkeypatch.setattr(ReferenceSkillsIndex, "_resolved", resolved)


MUTANTS = {
    "unpinned_reads": _unpinned_reads,
    "reads_ignore_tenant": _reads_ignore_tenant,
    "put_ignores_pin": _put_ignores_pin,
    "put_file_ignores_pin": _put_file_ignores_pin,
    "delete_ignores_pin": _delete_ignores_pin,
    "put_commits_before_refusal": _put_commits_before_refusal,
    "accepts_other_levels": _accepts_other_levels,
    "tenant_first_resolution": _tenant_first_resolution,
    "foreign_owner_resolves": _foreign_owner_resolves,
}


def apply_mutant(monkeypatch):
    import os
    name = os.environ.get("HERMES_SKILLS_MUTANT")
    if name:
        MUTANTS[name](monkeypatch)
