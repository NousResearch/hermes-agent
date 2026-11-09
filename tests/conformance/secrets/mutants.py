"""Broken secret store backends the conformance suite must reject (``test_mutants.py``).

Each passed an earlier version of the suite. ``local_reads_ignore_tenant`` and
``local_reads_ignore_the_bound_agent`` leak reads through a tenant- or agent-blind pin;
``local_writes_skip_the_administration_check`` lets a non-administering scope change credentials;
``local_accepts_undeclared_levels`` writes an owner-wide or tenant-wide secret onto the agent
level instead of refusing it; ``local_refusal_leaks_the_value`` and
``local_logs_the_value_on_a_refusal`` publish the value on the write refusal path, and
``local_refusal_keeps_the_value_in_frame_locals`` / ``reference_refusal_keeps_the_value_in_frame_locals``
leave it bound in a frame the refusal propagates through, where a traceback-locals renderer
reads it. The same leak, one frame of the whole exception chain deeper:
``local_refusal_keeps_the_value_in_a_chained_frame`` re-raises the refusal clean but leaves the
value in the frames of a failure chained to it as context, ``local_managed_key_refusal_keeps_the_value``
keeps it on the write lock's plain refusal of an administrator-pinned key, and
``local_write_failure_keeps_the_value_in_the_writer_frames`` keeps it in the writer's own frames
on a mid-write I/O error. Three more failures of the write path:
``local_failure_clears_the_callers_own_binding`` reaches out of the store's own frames and
nulls the caller's live value binding (a caller retrying from an error handler finds its
variable gone), ``local_failure_keeps_the_escaped_secret`` clears the raw value but keeps the
serialized, escaped form a writer quoted it into, and
``local_failure_keeps_the_original_failure_as_context`` raises a fresh error but chains the
original failure to it, frames and all. Against the fresh error's own text:
``local_keeps_strerror_in_the_fresh_error`` copies the failing writer's own message, strerror
and filename into the error the caller reads, and
``local_logs_exc_info_while_holding_the_value`` logs the failure with the live exception —
exc_info and all — from a frame that still binds the value, so a log handler that keeps the
record renders the plaintext from it later; ``local_loads_the_auth_store_after_binding_the_record``
binds the record in a helper frame that then loads and parses the store, putting the record in
every frame a load failure — logged with its live exception — unwinds. One more, against the
suite's own coverage: ``local_logs_exc_info_only_on_an_encode_failure`` logs the failed write
with the live exception only when the serialization itself fails, so a suite whose failure
hook can break the write at one point alone never reaches it. Against the
leveled reference backend: ``reference_reads_ignore_tenant`` drops the tenant half of the pin,
``reference_writes_skip_the_administration_check`` drops the administration check,
``reference_owner_wide_resolves_everywhere`` resolves another owner's entries,
``reference_owner_wide_keys_on_the_actor`` keys an owner-wide write on the acting principal, and
``reference_drops_level_precedence`` reads the tenant-wide value over the agent's own. Each
mutant declares, next to how it is applied, the suite test that must kill it; the harness
(``test_mutants.py``) asserts that test is among the failures. Each mutant is applied by the
``backend`` fixtures when ``HERMES_SECRETS_MUTANT`` names it.
"""

import json
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
            del level
            raise ValueError("not a secret level; the secret levels are 'agent', 'owner' and 'tenant'")
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


def local_refusal_keeps_the_value_in_a_chained_frame(monkeypatch):
    """The write refusal is re-raised clean from its own frames, but an inner failure chained
    to it as its context still holds the caller's value in its frames — only a check that
    walks the whole exception chain, not just the refusal's own traceback, catches it."""
    from hermes_secret_port_local import LocalSecretStore
    original = LocalSecretStore.set

    def _fail_holding_the_value(value):
        raise ValueError("an inner failure that still holds the refused value")

    def set(self, name, value, *, level):
        try:
            return original(self, name, value, level=level)
        except PermissionError as refusal:
            try:
                _fail_holding_the_value(value)
            except ValueError:
                del value
                raise refusal
    monkeypatch.setattr(LocalSecretStore, "set", set)


def local_managed_key_refusal_keeps_the_value(monkeypatch):
    """The write path clears the caller's value only from the store's own refusals: the
    write lock's plain refusal of an administrator-pinned key propagates with the value
    still bound in the backend's own frame."""
    from hermes_cli.config import require_env_writable, save_env_value
    from hermes_secret_port_local import LocalSecretStore

    def set(self, name, value, *, level):
        try:
            self._pin(administering=True)
            self._require_agent_level(level)
        except PermissionError:
            del value
            raise
        require_env_writable(name, "set")
        save_env_value(name, value)

    monkeypatch.setattr(LocalSecretStore, "set", set)


def local_write_failure_keeps_the_value_in_the_writer_frames(monkeypatch):
    """A mid-write I/O failure clears the caller's value from the backend's own frame but not
    from the frames of the writers it called — the traceback still renders the plaintext from
    the writer's locals."""
    from hermes_cli.config import require_env_writable, save_env_value
    from hermes_secret_port_local import LocalSecretStore

    def set(self, name, value, *, level):
        try:
            self._pin(administering=True)
            self._require_agent_level(level)
            require_env_writable(name, "set")
            save_env_value(name, value)
        except BaseException:  # health: allow BLE001 -- re-raised below
            del value
            raise

    monkeypatch.setattr(LocalSecretStore, "set", set)


def local_failure_clears_the_callers_own_binding(monkeypatch):
    """The failed write reaches out of its own frames and clears the caller's live value
    binding — the frames of the caller and of every exception chained to the failure are
    scrubbed too, so a caller retrying the write from inside an error handler finds its
    variable nulled. The store's own frames are cleared as well, so only the check on the
    caller's untouched bindings catches it."""
    from hermes_secret_port_local import LocalSecretStore
    original = LocalSecretStore.set

    def clear_the_value(exc, value):
        # The scrubber runs in a frame the error's traceback never passes through, so it
        # survives scrubbing the wrapper's own frames and reaches the caller's.
        needle = str(value)
        pending = [exc]
        while pending:
            linked = pending.pop()
            tb = linked.__traceback__
            while tb is not None:
                for key in list(tb.tb_frame.f_locals):
                    try:
                        if needle in repr(tb.tb_frame.f_locals[key]):
                            tb.tb_frame.f_locals[key] = None
                    except Exception:  # health: allow BLE001 -- a binding that cannot render holds nothing
                        continue
                tb = tb.tb_next
            for chained in (linked.__context__, linked.__cause__):
                if chained is not None:
                    pending.append(chained)

    def set(self, name, value, *, level):
        try:
            return original(self, name, value, level=level)
        except BaseException as exc:  # health: allow BLE001 -- scrubbed and re-raised below
            clear_the_value(exc, value)
            raise

    monkeypatch.setattr(LocalSecretStore, "set", set)


def local_failure_keeps_the_escaped_secret(monkeypatch):
    """The failed write clears the raw value from every surviving frame but keeps the
    serialized, escaped form — the multiline secret the way a writer quotes it into a line —
    bound in a frame of a failure chained to the error: only a check that renders frame
    locals under their escaped forms too catches it."""
    from hermes_secret_port_local import LocalSecretStore
    original = LocalSecretStore.set

    def set(self, name, value, *, level):
        try:
            return original(self, name, value, level=level)
        except OSError:
            def the_writers_serialized_line():
                line = json.dumps(value)  # the value, escaped the way a writer serialized it
                raise OSError(f"the write failed part-way ({len(line)} characters in)")

            try:
                the_writers_serialized_line()
            except OSError as chained:
                fresh = OSError("the credential write failed")
                fresh.__context__ = chained
                del value
                raise fresh

    monkeypatch.setattr(LocalSecretStore, "set", set)


def local_failure_keeps_the_original_failure_as_context(monkeypatch):
    """The failed write raises its own fresh error with clean frames of its own, but leaves
    the original failure chained to it as context — the original's traceback still holds the
    value in the writer's frames, so only a check that walks the whole chained graph catches
    it."""
    from hermes_secret_port_local import LocalSecretStore
    original = LocalSecretStore.set

    def set(self, name, value, *, level):
        try:
            return original(self, name, value, level=level)
        except OSError:
            def the_writers_last_frame():
                held = value  # the writer's own frame, still holding the caller's value
                raise OSError(f"the write failed part-way ({len(held)} characters in)")

            try:
                the_writers_last_frame()
            except OSError as chained:
                fresh = OSError("the credential write failed")
                fresh.__context__ = chained  # the original failure rides along, frames and all
                del value
                raise fresh

    monkeypatch.setattr(LocalSecretStore, "set", set)


def local_keeps_strerror_in_the_fresh_error(monkeypatch):
    """The fresh error copies the failing writer's own text — an OSError's strerror and
    filename, and any other failure's own message — so a writer that names the value it saw
    puts it in the error the caller reads."""
    from hermes_secret_port import SecretWriteFailed
    from hermes_secret_port_local import LocalSecretStore

    def fresh_write_error(exc):
        if isinstance(exc, OSError):
            detail = type(exc).__name__
            if exc.errno is not None:
                detail += f", errno {exc.errno}"
            if exc.strerror:
                detail += f" ({exc.strerror})"
            if exc.filename:
                detail += f", file {exc.filename}"
            return SecretWriteFailed(f"the credential write failed: {detail}")
        try:
            return type(exc)(str(exc))
        except Exception:  # health: allow BLE001 -- a signature this mutant cannot call falls back
            return SecretWriteFailed(f"the credential write failed: {type(exc).__name__}")

    monkeypatch.setattr("hermes_secret_port_local.fresh_write_error", fresh_write_error)


def local_logs_exc_info_while_holding_the_value(monkeypatch):
    """The failed write logs the failure — exc_info and all — from a frame that still binds the
    value, so a log handler that keeps the record renders the plaintext from that frame's
    locals later."""
    from hermes_secret_port_local import LocalSecretStore
    original = LocalSecretStore.set

    def set(self, name, value, *, level):
        try:
            return original(self, name, value, level=level)
        except BaseException:  # health: allow BLE001 -- re-raised below
            logging.getLogger(__name__).error("the credential write failed", exc_info=True)
            raise

    monkeypatch.setattr(LocalSecretStore, "set", set)


def local_loads_the_auth_store_after_binding_the_record(monkeypatch):
    """The record is bound in a helper frame that then loads and parses the store: a load
    failure is logged by today's loader with the live exception, whose traceback unwinds that
    record-holding frame — a log handler that keeps the record renders the plaintext from it."""
    from hermes_secret_port import fresh_write_error
    from hermes_secret_port_local import LocalSecretStore

    def _write(self, provider, record):
        from hermes_cli import auth
        with auth._auth_store_lock():
            store = auth._load_auth_store()  # the record is bound in this very frame
            auth._store_provider_state(store, provider, dict(record), set_active=False)
            auth._save_auth_store(store)

    def set_auth(self, provider, record):
        failure = None
        try:
            _write(self, provider, record)
        except BaseException as exc:  # health: allow BLE001 -- the fresh error below is all that survives
            failure = fresh_write_error(exc)
        del record
        if failure is not None:
            failure.__context__ = None
            failure.__cause__ = None
            raise failure

    monkeypatch.setattr(LocalSecretStore, "set_auth", set_auth)


def local_logs_exc_info_only_on_an_encode_failure(monkeypatch):
    """The failed write logs the failure — exc_info and all — from a frame that still binds
    the record, but only when the failure is the serialization's own ``TypeError``: a suite
    whose failure hook breaks the write at one point alone never reaches that path, and a
    backend that logs there passes it untouched."""
    from hermes_secret_port_local import LocalSecretStore
    original = LocalSecretStore._write_auth_record

    def _write_auth_record(self, store, provider, record):
        try:
            return original(self, store, provider, record)
        except TypeError:
            logging.getLogger(__name__).error("the credential write failed", exc_info=True)
            raise

    monkeypatch.setattr(LocalSecretStore, "_write_auth_record", _write_auth_record)


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
    "local_refusal_keeps_the_value_in_a_chained_frame": (
        local_refusal_keeps_the_value_in_a_chained_frame,
        _local("test_no_refusal_traceback_frame_holds_the_value_or_record")),
    "local_managed_key_refusal_keeps_the_value": (
        local_managed_key_refusal_keeps_the_value,
        _local("test_no_failure_traceback_frame_holds_the_value_or_record[write]")),
    "local_write_failure_keeps_the_value_in_the_writer_frames": (
        local_write_failure_keeps_the_value_in_the_writer_frames,
        _local("test_no_failure_traceback_frame_holds_the_value_or_record[write]")),
    "local_failure_clears_the_callers_own_binding": (
        local_failure_clears_the_callers_own_binding,
        _local("test_a_failed_write_from_an_error_handler_leaves_the_callers_bindings_untouched")),
    "local_failure_keeps_the_escaped_secret": (
        local_failure_keeps_the_escaped_secret,
        _local("test_no_failure_traceback_frame_holds_the_value_or_record[write]")),
    "local_failure_keeps_the_original_failure_as_context": (
        local_failure_keeps_the_original_failure_as_context,
        _local("test_no_failure_traceback_frame_holds_the_value_or_record[write]")),
    "local_keeps_strerror_in_the_fresh_error": (
        local_keeps_strerror_in_the_fresh_error,
        "tests/conformance/secrets/test_local.py::test_a_failed_writes_fresh_error_names_none_of_the_writers_text[strerror]"),
    "local_logs_exc_info_while_holding_the_value": (
        local_logs_exc_info_while_holding_the_value,
        "tests/conformance/secrets/test_local.py::"
        "test_a_failed_write_leaves_no_plaintext_in_a_log_handlers_kept_exception[set-write]"),
    "local_loads_the_auth_store_after_binding_the_record": (
        local_loads_the_auth_store_after_binding_the_record,
        "tests/conformance/secrets/test_local.py::"
        "test_a_failed_write_leaves_no_plaintext_in_a_log_handlers_kept_exception[set_auth-write]"),
    "local_logs_exc_info_only_on_an_encode_failure": (
        local_logs_exc_info_only_on_an_encode_failure,
        "tests/conformance/secrets/test_local.py::"
        "test_a_failed_write_leaves_no_plaintext_in_a_log_handlers_kept_exception[set_auth-encode]"),
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
