"""The secret store conformance suite: isolation invariants every backend meets.

Parametrized over backends through the ``backend`` fixture (:class:`Backend`). Each test talks to
the port only (``hermes_secret_port``); the fixture says how to bind each agent's scope and open
its store. Agents are named by short keys: ``"a"`` and ``"b"`` for the pinning suite (``"a"`` is
the agent under test), and ``"a"``/``"b"`` (one owner) plus ``"c"``/``"d"`` (another owner) for
the leveled suite.

Covered:

- pinning: every read and every write of the port is refused for another agent, another tenant
  and no pin, and a refusal changes nothing the agent observes through the port
  (:func:`observe`: the secret, its names, the auth record); every read works for the agent's
  own pin, and every write succeeds under it and is observed;
- administration: every write is refused (:class:`SecretAdminRefused`) for a scope on the same
  agent without the administration right, changing nothing;
- levels: a write naming a level the backend does not hold is refused
  (:class:`SecretLevelRefused`), and a name that is not a secret level at all is a programming
  error (``ValueError``);
- indistinguishability: a secret, a name and an auth record held only by another agent look the
  same as ones defined nowhere;
- no leakage: through every refusal path, and a missing name under the agent's own pin, a
  secret's value never appears in a log line or in an exception message the store raises;
  through every refusal of a credential write, and every failure that is not one — a level that
  is not a secret level at all, a key an administrator has pinned, a mid-write I/O error — the
  value (or record) is bound in no frame the error's traceback passes through below this suite,
  and none is bound in the frames of any exception chained to it, as context or cause,
  recursively, or of an exception group's members (error reporters render traceback frame
  locals, under str and repr and as JSON and bytes, and the escaped form a writer serialized is
  as much a leak as the raw text); a failed write called from inside a caller's ``except``
  handler leaves the caller's own bindings untouched — the store owns its frames alone and
  never touches a frame it does not own;
- resolution (backends that hold the owner-wide and tenant-wide levels,
  :class:`LeveledSecretConformance`): an owner-wide secret resolves on every agent its owner owns
  and on no other; a tenant-wide secret resolves on every agent of the tenant; the agent level
  shadows the owner and tenant levels; an owner-wide write on a group's agent keys on the
  agent's owner from the scope, never on the acting principal.

NOT covered. A backend that passes this suite may still violate these, and must not claim them:

- no plaintext at rest (a hosted backend must store values encrypted and check its own storage;
  the local backend is today's plaintext profile files by design, so the suite cannot demand it
  — a backend's ``assert_isolating`` is the seam that must);
- the hosted boot refusal (a profile ``.env`` or ``auth.json`` on disk, a tenant secret in
  ``os.environ``);
- auth records resolving from owner-wide or tenant-wide levels (the port's ``set_auth`` writes
  the agent level only, so no owner-wide record can be created through the port to resolve);
- the external secret sources (per-process startup fetchers, local-only by design);
- concurrency (two turns writing the same name through one store).
"""

from __future__ import annotations

import json
import logging
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable, cast

import pytest

from hermes_secret_port import (
    LEVELS, SecretAdminRefused, SecretLevelRefused, SecretPinRefused, SecretStore,
)


@dataclass
class Backend:
    """How the suite drives one backend.

    - ``open(agent)``: the agent's store.
    - ``bind(agent)``: bind the agent's scope as its owner (or a group's administering member),
      with the administration right.
    - ``bind_member(agent)``: bind a scope on the same agent without the administration right.
    - ``foreign_tenant(agent)``: bind a scope of ANOTHER tenant that names the same agent id.
    - ``unbound()``: run with no scope pinned. A local backend's pin is then the ambient
      profile, so its ``unbound`` names a third profile; a hosted one binds nothing.
    - ``levels``: the ownership levels the backend holds (:data:`LEVELS` names them).
    - ``managed_key()``: REQUIRED. While the context is active, the key it yields is
      refused by the backend's write path with a plain error — not the store's refusal —
      raised where the port's structure puts every check that can fail: before the value is
      bound in any frame below the write method (a backend whose write lock refuses only
      part-way through the write fails there instead, its plain error re-made fresh like any
      other writer failure). A backend with a managed-key write lock pins one.
    - ``fail_writes()``: REQUIRED. While the context is active, the seeded agent ``"a"``'s
      writes fail part-way through with an ``OSError``, with the caller's value (or record)
      already bound in the writer's frames — a plain I/O failure, not the store's refusal.
      A backend whose write path the fixture cannot break must supply a hook that fails
      the write some other plain way, mid-path, or this failure path goes untested.
    """

    open: Callable[[str], SecretStore]
    bind: Callable[[str], AbstractContextManager]
    bind_member: Callable[[str], AbstractContextManager]
    foreign_tenant: Callable[[str], AbstractContextManager]
    unbound: Callable[[], AbstractContextManager]
    levels: frozenset[str]
    managed_key: Callable[[], AbstractContextManager]
    fail_writes: Callable[[], AbstractContextManager]


def _refused(call: Callable[[], Any]) -> str:
    """The refusal *call* raises (its message), asserting it is the store's pin refusal."""
    with pytest.raises(SecretPinRefused) as refused:
        call()
    return str(refused.value)


SEED_NAME = "API_KEY"
SEED_PROVIDER = "probe"
NEW_NAME = "ROTATED_KEY"


def _seed(backend, agent="a", *, name=SEED_NAME, value="agent-level-value", provider=SEED_PROVIDER,
          record=None) -> SecretStore:
    """Plant one agent-level secret and auth record on *agent*, as its administering owner."""
    store = backend.open(agent)
    with backend.bind(agent):
        store.set(name, value, level="agent")
        store.set_auth(provider, dict(record or {"token": "agent-level-record"}))
    return store


def observe(backend, store, agent="a") -> dict[str, Any]:
    """Everything about agent ``a`` the port exposes: the seeded secret, the whole resolved name
    set, and the seeded auth record. A refused operation must leave all of it unchanged."""
    with backend.bind(agent):
        return {"get": store.get(SEED_NAME), "names": store.names(), "auth": store.get_auth(SEED_PROVIDER)}


#: Every read of the port, naming what a seeded agent holds.
ALL_READS: dict[str, Callable[[SecretStore], Any]] = {
    "get": lambda st: st.get(SEED_NAME),
    "get_auth": lambda st: st.get_auth(SEED_PROVIDER),
    "names": lambda st: st.names(),
}

#: Every write of the port, with arguments that change what :func:`observe` reads.
ALL_WRITES: dict[str, Callable[[SecretStore], Any]] = {
    "set": lambda st: st.set(NEW_NAME, "written-value", level="agent"),
    "delete": lambda st: st.delete(SEED_NAME, level="agent"),
    "set_auth": lambda st: st.set_auth(SEED_PROVIDER, {"token": "rotated-record"}),
}


def _foreign_pins(backend, agent="a"):
    """Every pin but *agent*'s own: another agent, another tenant naming it, no pin."""
    return [lambda: backend.bind("b"), lambda: backend.foreign_tenant(agent), backend.unbound]


def _rendered_texts(value: Any) -> list[str]:
    """Every way an error reporter might render a frame's local binding: its ``str`` and
    ``repr``, its JSON form, its bytes decoded — and, because a container's rendering
    escapes the strings nested inside it, each nested string read out of its container
    raw. A writer that serialized the value leaves the escaped form where the raw text
    never appears, and a multiline secret never appears in a ``repr`` at all, so every
    rendering of every binding is checked."""
    texts: list[str] = []

    def add(node: Any) -> None:
        if isinstance(node, (bytes, bytearray)):
            try:
                texts.append(node.decode("utf-8", "backslashreplace"))
            except Exception:  # health: allow BLE001 -- a binding that cannot decode holds nothing to render
                return
        try:
            texts.append(str(node))
            texts.append(repr(node))
        except Exception:  # health: allow BLE001 -- a binding that cannot render holds nothing
            return
        try:
            texts.append(json.dumps(node, default=repr))
        except Exception:  # health: allow BLE001 -- a binding JSON cannot serialize holds nothing to render
            return

    def walk(node: Any, seen: set[int]) -> None:
        if id(node) in seen:  # a record that contains itself renders nothing twice
            return
        seen.add(id(node))
        add(node)
        if isinstance(node, dict):
            for held in node.values():
                walk(held, seen)
        elif isinstance(node, (list, tuple, set, frozenset)):
            for held in node:
                walk(held, seen)

    walk(value, set())
    return texts


def _plaintext_forms(marker: str) -> tuple[str, ...]:
    """The *marker* and its serialized forms: the renderings a writer or reporter that
    handled the value holds where the raw text never appears — the escaped forms that
    quote it (JSON, ``repr``), and the form a writer that refuses multiline values
    leaves."""
    forms = {marker}
    for escaped in (json.dumps(marker)[1:-1], repr(marker)[1:-1]):
        if escaped != marker:
            forms.add(escaped)
    stripped = marker.replace("\n", "").replace("\r", "")
    if len(stripped) >= 16 and stripped != marker:
        # A shorter stripped form is a fragment short enough to appear in unrelated text (a
        # one-letter marker of "a\n" would flag every frame holding "cat"), so it is hunted
        # only for markers whose stripped form is substantial.
        forms.add(stripped)
    return tuple(forms)


def assert_no_frame_holds(exc: BaseException, marker: str) -> None:
    """Assert the *marker* — the plaintext, or its escaped forms — appears in the locals of
    no frame the exception *exc* passes through, and none of the frames of any exception
    chained to it — as context or cause, recursively (a chain that loops visits each
    exception once), or as an ``ExceptionGroup`` member — below this suite's own frames.
    Error reporters render the frame locals of the whole chain, so the check walks all of
    it, under every rendering a reporter might use (:func:`_rendered_texts`); the suite's
    frames are skipped because it planted the marker."""
    forms = _plaintext_forms(marker)
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending:
        linked = pending.pop()
        if id(linked) in seen:
            continue
        seen.add(id(linked))
        tb = linked.__traceback__
        while tb is not None:
            if tb.tb_frame.f_code.co_filename != __file__:
                for binding, local in tb.tb_frame.f_locals.items():
                    for text in _rendered_texts(local):
                        leaked = next((form for form in forms if form in text), None)
                        if leaked is not None:
                            raise AssertionError(
                                f"an exception kept the plaintext in local {binding!r} of "
                                f"{tb.tb_frame.f_code.co_filename}:{tb.tb_frame.f_lineno} "
                                f"{tb.tb_frame.f_code.co_name!r}: {text!r}")
            tb = tb.tb_next
        for chained in (linked.__context__, linked.__cause__):
            if chained is not None:
                pending.append(chained)
        if isinstance(linked, BaseExceptionGroup):
            pending.extend(linked.exceptions)


class SecretPinningConformance:
    """Every operation is pinned to its agent, writes are administration, and nothing leaks.
    Inherit and supply a ``backend`` fixture."""

    @pytest.mark.parametrize("operation", sorted(ALL_READS))
    def test_every_read_is_refused_for_any_pin_but_its_agents(self, backend, operation):
        store = _seed(backend)
        for bind in _foreign_pins(backend):
            with bind():
                _refused(lambda: ALL_READS[operation](store))
        with backend.bind("a"):
            assert ALL_READS[operation](store)

    @pytest.mark.parametrize("operation", sorted(ALL_WRITES))
    @pytest.mark.parametrize("pin", ["other_agent", "other_tenant", "unbound"])
    def test_every_write_is_refused_for_any_pin_but_its_agents_and_changes_nothing(self, backend, operation, pin):
        store = _seed(backend)
        before = observe(backend, store)
        binds = {"other_agent": lambda: backend.bind("b"), "other_tenant": lambda: backend.foreign_tenant("a"),
                 "unbound": backend.unbound}
        with binds[pin]():
            _refused(lambda: ALL_WRITES[operation](store))
        assert observe(backend, store) == before

    @pytest.mark.parametrize("operation", sorted(ALL_WRITES))
    def test_every_write_succeeds_for_its_own_agent_and_is_observed(self, backend, operation):
        """The positive control for the refusal tests: each write is one the agent's administering
        owner may make, and it changes what the agent observes."""
        store = _seed(backend)
        before = observe(backend, store)
        with backend.bind("a"):
            ALL_WRITES[operation](store)
        assert observe(backend, store) != before

    @pytest.mark.parametrize("operation", sorted(ALL_WRITES))
    def test_every_write_is_refused_without_administration_and_changes_nothing(self, backend, operation):
        store = _seed(backend)
        before = observe(backend, store)
        with backend.bind_member("a"):
            with pytest.raises(SecretAdminRefused):
                ALL_WRITES[operation](store)
        assert observe(backend, store) == before

    def test_an_unknown_level_is_a_programming_error(self, backend):
        store = _seed(backend)
        with backend.bind("a"):
            for level in ("participant", "nonsense", ""):
                with pytest.raises(ValueError):
                    store.set("X", "v", level=level)

    def test_an_invalid_level_never_echoes_the_callers_text(self, backend):
        """The level is caller-supplied text, so no port-authored message carries it: a caller
        that passes its own value as the level — the aliasing that turns write metadata into a
        leak — reads a programming error that names the valid levels and nothing of its own,
        and the level binding survives in no frame the error's traceback passes through below
        this suite, nor in any frame of an exception chained to it."""
        marker = 'zAliased\nLevel "Echo" 6f2b'
        store = _seed(backend)

        def refused_without_the_callers_text(call):
            with pytest.raises(ValueError) as refused:
                call()
            message = str(refused.value)
            for form in _plaintext_forms(marker):
                assert form not in message + repr(refused.value) + repr(refused.value.args), (
                    f"the refusal echoes the caller's own text: {refused.value!r}")
            assert all(level in message for level in LEVELS), (
                f"the refusal names no valid level: {refused.value!r}")
            assert_no_frame_holds(refused.value, marker)

        with backend.bind("a"):
            refused_without_the_callers_text(lambda: store.set(NEW_NAME, marker, level=marker))
            refused_without_the_callers_text(lambda: store.delete(NEW_NAME, level=marker))

    def test_a_write_at_a_level_the_backend_does_not_hold_is_refused(self, backend):
        """A backend holding every level has nothing unheld to refuse here; its refusals are
        exercised by the rest of the suite. A backend that holds less must refuse the rest."""
        store = _seed(backend)
        before = observe(backend, store)
        with backend.bind("a"):
            for level in sorted(LEVELS - backend.levels):
                with pytest.raises(SecretLevelRefused):
                    store.set(SEED_NAME, "v", level=level)
                with pytest.raises(SecretLevelRefused):
                    store.delete(SEED_NAME, level=level)
        assert observe(backend, store) == before

    def test_a_secret_held_only_by_another_agent_is_indistinguishable_from_a_missing_one(self, backend):
        _seed(backend, name="ONLY_A", value="private-a")
        theirs = backend.open("b")
        with backend.bind("b"):
            assert theirs.get("ONLY_A") is None and theirs.get("absent") is None
            assert "ONLY_A" not in theirs.names() and "absent" not in theirs.names()
            assert theirs.get_auth("only-a") is None and theirs.get_auth("absent") is None

    def test_no_secret_value_appears_in_log_lines_or_exception_messages(self, backend, caplog):
        """Through every refusal path — foreign pins on reads and writes, a write without the
        administration right, a write at an unheld level, and a read of a missing name under the
        agent's own pin — the value the store was seeded with never reaches a log line or an
        exception message."""
        marker = "zMarker7f31c9e2"
        store = _seed(backend, value=marker, record={"token": marker, "refresh": marker})
        raised: list[BaseException] = []
        with caplog.at_level(logging.DEBUG):
            for bind in _foreign_pins(backend):
                with bind():
                    for call in (lambda: store.get("ONLY_A"), lambda: store.get_auth(SEED_PROVIDER),
                                 lambda: store.names(),
                                 lambda: store.set("ONLY_A", "rotated", level="agent"),
                                 lambda: store.delete("ONLY_A", level="agent"),
                                 lambda: store.set_auth(SEED_PROVIDER, {"token": "rotated"})):
                        try:
                            call()
                        except BaseException as exc:  # health: allow BLE001 -- collected, asserted below
                            raised.append(exc)
            with backend.bind_member("a"):
                for write in ALL_WRITES.values():
                    try:
                        write(store)
                    except BaseException as exc:  # health: allow BLE001 -- collected, asserted below
                        raised.append(exc)
                # Refusals of the seeded value itself: a refusal that echoes or logs what it was
                # asked to store is a leak the marker below must catch.
                for leaky in (lambda st: st.set(SEED_NAME, marker, level="agent"),
                              lambda st: st.set_auth(SEED_PROVIDER, {"token": marker})):
                    try:
                        leaky(store)
                    except BaseException as exc:  # health: allow BLE001 -- collected, asserted below
                        raised.append(exc)
            with backend.bind("a"):
                for level in sorted(LEVELS - backend.levels):
                    try:
                        store.set("ONLY_A", "v", level=level)
                    except BaseException as exc:  # health: allow BLE001 -- collected, asserted below
                        raised.append(exc)
                store.get("absent")
                store.get_auth("absent")
        assert marker not in caplog.text
        assert raised, "the store refused nothing, so the leak assertion would be vacuous"
        for exc in raised:
            text = str(exc) + "".join(str(note) for note in getattr(exc, "__notes__", []))
            assert marker not in text, f"a refusal leaked a secret value: {exc!r}"

    def test_no_refusal_traceback_frame_holds_the_value_or_record(self, backend):
        """Through every refusal of a credential write — a foreign pin, a scope without the
        administration right, a level the backend does not hold — the value (or record) the
        caller asked to store is bound in no frame the refusal's traceback passes through
        below this suite, and none is bound in the frames of any exception chained to the
        refusal, as context or cause, recursively, or of an exception group's members: error
        reporters render the frame locals of the whole chain, under str and repr and as JSON
        and bytes, so a refused write must leave the plaintext in no frame it unwinds, in no
        rendering — not raw, not escaped. This suite's own frames are skipped: it planted the
        value."""
        marker = 'zRefused\nFrame "Locals" 3b8c'
        store = _seed(backend)
        record = {"token": marker, "refresh": marker}

        def refused_without_the_value(call):
            with pytest.raises(PermissionError) as refused:
                call()
            assert_no_frame_holds(refused.value, marker)

        for bind in _foreign_pins(backend):
            with bind():
                refused_without_the_value(lambda: store.set(NEW_NAME, marker, level="agent"))
                refused_without_the_value(lambda: store.set_auth(SEED_PROVIDER, record))
        with backend.bind_member("a"):
            refused_without_the_value(lambda: store.set(NEW_NAME, marker, level="agent"))
            refused_without_the_value(lambda: store.set_auth(SEED_PROVIDER, record))
        with backend.bind("a"):
            for level in sorted(LEVELS - backend.levels):
                refused_without_the_value(lambda level=level: store.set(NEW_NAME, marker, level=level))

    def test_no_failure_traceback_frame_holds_the_value_or_record(self, backend):
        """Through every failure of a credential write that is not the store's refusal — a
        level that is not a secret level at all (a programming error), a key an administrator
        has pinned (the backend's own write lock), and a mid-write I/O error the fixture
        injects — the value (or record) is bound in no frame the failure's traceback passes
        through below this suite, nor in any frame of an exception chained to it, as context
        or cause, recursively, or of an exception group's members. A backend clears its
        frames on a refusal; it must hold no value on every failure too, whatever broke the
        write: error reporters render the frame locals of the whole chain, under str and
        repr and as JSON and bytes, so the serialized, escaped form is as much a leak as the
        raw text. The managed-key and mid-write failure hooks are required fields of the
        fixture, so both paths are exercised on every backend."""
        marker = 'zFailed\nWrite "Locals" 9a4e'
        store = _seed(backend)
        record = {"token": marker, "refresh": marker}

        def failed_without_the_value(expected, call):
            with pytest.raises(expected) as failed:
                call()
            assert not isinstance(failed.value, PermissionError), (
                "the failure path surfaced as the store's own refusal")
            for form in _plaintext_forms(marker):
                assert form not in str(failed.value)
            assert_no_frame_holds(failed.value, marker)

        with backend.bind("a"):
            # Each names no secret level; some are not strings at all — the port's signature
            # says ``str``, and the failure path is exactly a caller that breaks it.
            for level in cast(tuple[str, ...], ("participant", "", 123, [])):
                failed_without_the_value((ValueError, TypeError),
                                         lambda level=level: store.set(NEW_NAME, marker, level=level))
            with backend.managed_key() as key:
                failed_without_the_value(ValueError, lambda: store.set(key, marker, level="agent"))
            with backend.fail_writes():
                failed_without_the_value(OSError, lambda: store.set(NEW_NAME, marker, level="agent"))
                failed_without_the_value(OSError, lambda: store.set_auth(SEED_PROVIDER, record))

    def test_a_failed_write_from_an_error_handler_leaves_the_callers_bindings_untouched(self, backend):
        """The caller-retry shape: a caller holds the value in its own frame and calls the
        write from inside an ``except`` handler, to retry it. The store owns only its own
        frames: a failed write leaves the caller's bindings, its live exception and its
        traceback exactly as they were, so the retry finds the value it held — the store
        never clears, rebinds or otherwise touches a frame it does not own."""
        marker = "zCallerBindings4d61"
        store = _seed(backend)

        def retry_from_the_handler():
            value = marker
            try:
                raise RuntimeError("the caller's own earlier failure")
            except RuntimeError as own_failure:
                try:
                    store.set(NEW_NAME, value, level="agent")
                except OSError:
                    pass  # the write failed; the caller keeps its value to retry
                return value, own_failure

        with backend.bind("a"), backend.fail_writes():
            value_after, own_failure = retry_from_the_handler()
            assert value_after == marker, "the store cleared a binding in a frame it does not own"
            assert str(own_failure) == "the caller's own earlier failure"
            assert own_failure.__traceback__ is not None
        with backend.bind("a"):  # the failure is over: the caller's own value still writes
            store.set(NEW_NAME, value_after, level="agent")
            assert store.get(NEW_NAME) == marker


class LeveledSecretConformance:
    """Resolution through the ownership levels, for backends that hold the owner-wide and
    tenant-wide levels. Agents ``a`` and ``b`` share one owner; ``c`` and ``d`` share another
    (a group's). Inherit and supply a ``backend`` fixture whose levels include ``owner`` and
    ``tenant``."""

    def test_an_owner_wide_secret_resolves_on_every_agent_the_owner_owns(self, backend):
        with backend.bind("a"):
            backend.open("a").set("SHARED", "owner-wide-value", level="owner")
        for agent in ("a", "b"):
            with backend.bind(agent):
                store = backend.open(agent)
                assert store.get("SHARED") == "owner-wide-value"
                assert "SHARED" in store.names()

    def test_a_tenant_wide_secret_resolves_on_every_agent_of_the_tenant(self, backend):
        with backend.bind("a"):
            backend.open("a").set("ORG", "tenant-wide-value", level="tenant")
        for agent in ("a", "b", "c"):
            with backend.bind(agent):
                assert backend.open(agent).get("ORG") == "tenant-wide-value"

    def test_an_owner_wide_secret_reaches_no_agent_the_owner_does_not_own(self, backend):
        with backend.bind("a"):  # a's owner also owns b, never c or d
            backend.open("a").set("SHARED", "owner-wide-value", level="owner")
        for agent in ("c", "d"):
            with backend.bind(agent):
                store = backend.open(agent)
                assert store.get("SHARED") is None
                assert "SHARED" not in store.names()

    def test_the_agent_level_shadows_the_owner_and_tenant_levels(self, backend):
        with backend.bind("a"):
            store = backend.open("a")
            store.set("OVERLAP", "tenant-value", level="tenant")
            store.set("OVERLAP", "owner-value", level="owner")
            store.set("OVERLAP", "agent-value", level="agent")
            assert store.get("OVERLAP") == "agent-value"
            assert store.delete("OVERLAP", level="agent")
            assert store.get("OVERLAP") == "owner-value"
            assert store.delete("OVERLAP", level="owner")
            assert store.get("OVERLAP") == "tenant-value"
            assert store.names().count("OVERLAP") == 1

    def test_an_owner_wide_write_on_a_group_agent_keys_on_the_owners_scope_not_the_actors(self, backend):
        """``c`` and ``d`` are owned by one group, so a turn on ``c`` acts for a principal that
        is not the owner: its owner-wide write keys on the owner the scope names, resolving on
        the group's every agent and on no other owner's."""
        with backend.bind("c"):
            backend.open("c").set("GROUP_SHARED", "group-owner-value", level="owner")
        with backend.bind("d"):
            assert backend.open("d").get("GROUP_SHARED") == "group-owner-value"
        with backend.bind("a"):
            assert backend.open("a").get("GROUP_SHARED") is None


class SecretsConformance(SecretPinningConformance, LeveledSecretConformance):
    """The whole suite. Inherit and supply a ``backend`` fixture (or inherit the parts separately,
    e.g. one test module each, to spread a slow backend over several runners)."""
