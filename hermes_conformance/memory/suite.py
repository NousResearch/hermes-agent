"""The memory conformance suite: pinning, isolation and cap invariants every backend meets.

Parametrized over backends through the ``backend`` fixture (:class:`Backend`). Each test talks to
the port only (``hermes_memory_port``); the fixture says how to bind each agent's scope and open
its storage. Agents are named by short keys (``"a"``, ``"b"``); both belong to one tenant.

Covered:

- pinning: every operation of the port (:data:`WRITES`, :data:`READS`) is refused for another
  agent, another tenant and no pin, and a refusal changes nothing the agent can observe through
  the port (:func:`observe`: both targets' entries, their frozen system-prompt blocks, their
  caps and enabled flags, and the staged-entry resolutions);
- isolation: another agent never sees the agent's entries, and a foreign entry is
  indistinguishable from a missing one (a read under a foreign pin is the same refusal whatever
  it names);
- the home the files resolve under: a scope that names this store's profile without moving the
  ambient home (a bind with no home override) writes nothing, instead of redirecting the write
  into the profile the ambient home names;
- levels: a write names its level; one the backend does not hold, or a level the agent's own
  notes never take, is refused with a clear error and changes nothing;
- caps: a write whose result would exceed the target's char cap is refused before anything is
  written — a single add over the cap, a batch whose final result is over the cap (all-or-nothing:
  none of its operations lands), while a batch that frees room and adds in one call goes through,
  because the cap is checked on the batch's final result only;
- replicas: two handles on the same agent's storage interleave writes without loss.

NOT covered. A backend that passes this suite may still violate these, and must not claim them:

- participant visibility (a participant-level user profile visible to that participant only) and
  the resolution policy (agent level, then owner-wide, then tenant-wide): the fixture cannot vary
  the participant or owner (the local backend holds the agent level only);
- the hosted isolation assertion (``MemoryStorageProvider.assert_isolating`` failing for a role
  that bypasses tenant isolation); only that the method is required is tested, elsewhere;
- the local file format's own guards (threat scanning, drift refusals, atomic writes): today's
  store keeps them, and the port does not re-test its backend's internals;
- turn fences: memory is the agent's own learning, not session-owned data, so a turn's fence does
  not gate these writes.
"""

from __future__ import annotations

import threading
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable

import pytest

from hermes_memory_port import MemoryLevelRefused, MemoryStorage
from hermes_state_port import StoragePinRefused


@dataclass
class Backend:
    """How the suite drives one backend.

    - ``open(agent)``: the agent's storage.
    - ``replica(agent)``: an independent second handle on the same agent's storage (another
      replica's connection).
    - ``bind(agent)``: bind the agent's scope (a non-turn activity of its owner).
    - ``home(agent)``: the agent id the fixture's scope names (a local backend's profile path).
    - ``foreign_tenant(agent)``: bind a scope of ANOTHER tenant that names the same agent id.
    - ``unbound()``: run with no scope pinned. A local backend's pin is then the ambient profile,
      so its ``unbound`` names a third profile; a hosted one binds nothing.
    """

    open: Callable[[str], MemoryStorage]
    replica: Callable[[str], MemoryStorage]
    bind: Callable[[str], AbstractContextManager]
    home: Callable[[str], Any]
    foreign_tenant: Callable[[str], AbstractContextManager]
    unbound: Callable[[], AbstractContextManager]


def _refused(call: Callable[[], Any]) -> str:
    """The refusal *call* raises (its message), asserting it is the store's pin refusal."""
    with pytest.raises(StoragePinRefused) as refused:
        call()
    return str(refused.value)


def _replica_thread(target: Callable[..., Any], *args: Any) -> threading.Thread:
    """An unstarted thread running *target* in a fresh, empty context: another replica's worker,
    which must not inherit this test's scope."""
    # health: allow HX012 -- a replica's worker deliberately starts with no scope; each target binds its own
    import contextvars
    return threading.Thread(target=contextvars.Context().run, args=(target, *args))


# --- the world every per-operation test runs against ---------------------------------------------

@dataclass
class World:
    """Agent ``a``: two agent-note entries (``alpha note``, ``beta note``), one user-profile entry
    (``gamma fact``), and the frozen prompt blocks still empty (loaded before the entries were
    written)."""

    storage: MemoryStorage
    limit: int


def make_world(backend) -> World:
    storage = backend.open("a")
    store = storage.store
    with backend.bind("a"):
        store.add("memory", "alpha note")
        store.add("memory", "beta note")
        store.add("user", "gamma fact")
        limit = store.char_limit("memory")
    return World(storage, limit)


def observe(backend, storage) -> dict[str, Any]:
    """Everything about agent ``a`` the port exposes: both targets' entries, their frozen
    system-prompt blocks, their caps and enabled flags, and the staged-entry resolutions. A
    refused operation must leave all of it unchanged."""
    store = storage.store
    with backend.bind("a"):
        state: dict[str, Any] = {}
        for target in ("memory", "user"):
            state[target] = {"entries": store.entries(target), "block": store.system_prompt_block(target),
                             "enabled": store.target_enabled(target), "limit": store.char_limit(target)}
        state["resolved"] = store.resolve_entry("memory", "alpha", "replace")
        state["resolved_batch"] = store.resolve_batch_entries(
            "memory", [{"action": "remove", "old_text": "beta"}])
    return state


#: Every mutating operation, with arguments that succeed in a :class:`World`.
WRITES: dict[str, Callable[[MemoryStorage, World], Any]] = {
    "add": lambda st, w: st.store.add("memory", "delta note"),
    "add_user": lambda st, w: st.store.add("user", "epsilon fact"),
    "replace": lambda st, w: st.store.replace("memory", "beta", "beta revised"),
    "patch": lambda st, w: st.store.patch("memory", "alpha", "ALPHA"),
    "remove": lambda st, w: st.store.remove("user", "gamma"),
    "apply_batch": lambda st, w: st.store.apply_batch(
        "memory", [{"action": "remove", "old_text": "alpha"}, {"action": "add", "content": "zeta"}]),
    "reload": lambda st, w: st.store.reload(),
}

#: Every read, naming entries agent ``a`` holds in a :class:`World`.
READS: dict[str, Callable[[MemoryStorage, World], Any]] = {
    "entries": lambda st, w: st.store.entries("memory"),
    "entries_user": lambda st, w: st.store.entries("user"),
    "system_prompt_block": lambda st, w: st.store.system_prompt_block("memory"),
    "target_enabled": lambda st, w: st.store.target_enabled("memory"),
    "char_limit": lambda st, w: st.store.char_limit("memory"),
    "resolve_entry": lambda st, w: st.store.resolve_entry("memory", "alpha", "replace"),
    "resolve_batch_entries": lambda st, w: st.store.resolve_batch_entries(
        "memory", [{"action": "remove", "old_text": "beta"}]),
}

#: Operations whose success changes nothing :func:`observe` reads. ``reload`` is NOT one: the
#: frozen blocks were loaded before the world's entries were written, so a reload that picks
#: them up changes them.
_NO_VISIBLE_CHANGE: frozenset[str] = frozenset()

_TARGETS = ("memory", "user")


def _foreign_pins(backend):
    """Every pin but agent ``a``'s own: another agent, another tenant naming ``a``, no pin."""
    return {"other_agent": lambda: backend.bind("b"), "other_tenant": lambda: backend.foreign_tenant("a"),
            "unbound": backend.unbound}


class MemoryConformance:
    """Every operation is pinned to its agent, isolation holds, and caps refuse before the
    write. Inherit and supply a ``backend`` fixture."""

    # --- pinning

    @pytest.mark.parametrize("operation", sorted(READS))
    def test_every_read_is_refused_for_any_pin_but_its_agents(self, backend, operation):
        world = make_world(backend)
        for bind in _foreign_pins(backend).values():
            with bind():
                _refused(lambda: READS[operation](world.storage, world))
        with backend.bind("a"):
            READS[operation](world.storage, world)

    @pytest.mark.parametrize("operation", sorted(WRITES))
    @pytest.mark.parametrize("pin", ["other_agent", "other_tenant", "unbound"])
    def test_every_write_is_refused_for_any_pin_but_its_agents_and_changes_nothing(self, backend, operation, pin):
        world = make_world(backend)
        before = observe(backend, world.storage)
        with _foreign_pins(backend)[pin]():
            _refused(lambda: WRITES[operation](world.storage, world))
        assert observe(backend, world.storage) == before

    @pytest.mark.parametrize("operation", sorted(WRITES))
    def test_every_write_succeeds_for_its_own_agent(self, backend, operation):
        """The positive control for the refusal tests: each call is one the agent may make, and it
        changes what the agent observes."""
        world = make_world(backend)
        before = observe(backend, world.storage)
        with backend.bind("a"):
            WRITES[operation](world.storage, world)
        assert observe(backend, world.storage) != before

    def test_with_no_scope_pinned_reads_see_nothing(self, backend):
        world = make_world(backend)
        with backend.unbound():
            for read in READS.values():
                _refused(lambda read=read: read(world.storage, world))

    # --- isolation

    def test_another_agent_never_sees_the_agents_entries(self, backend):
        world = make_world(backend)
        theirs = backend.open("b")
        with backend.bind("b"):
            assert theirs.store.entries("memory") == []
            assert theirs.store.entries("user") == []
            assert theirs.store.system_prompt_block("memory") is None
            _refused(lambda: world.storage.store.entries("memory"))
            _refused(lambda: world.storage.store.add("user", "stolen"))
            _refused(lambda: world.storage.store.reload())
        with backend.bind("a"):  # and the refusals wrote nothing into either store
            assert world.storage.store.entries("user") == ["gamma fact"]
        with backend.bind("b"):
            assert theirs.store.entries("user") == []

    def test_a_foreign_entry_is_indistinguishable_from_a_missing_one(self, backend):
        world = make_world(backend)
        with backend.bind("b"):
            assert _refused(lambda: world.storage.store.resolve_entry("memory", "alpha", "replace")) \
                == _refused(lambda: world.storage.store.resolve_entry("memory", "absent", "replace"))
            assert _refused(lambda: world.storage.store.remove("memory", "alpha")) \
                == _refused(lambda: world.storage.store.remove("memory", "absent"))

    def test_a_bind_that_names_the_profile_without_moving_the_home_writes_nothing(self, backend):
        """The store's files resolve through the ambient profile, so a scope that names this
        store's profile while the ambient home is another profile's must refuse: the pin would
        otherwise redirect the write into the ambient profile's files."""
        from hermes_scope import LOCAL_TENANT, LOCAL_OWNER, Scope, push_state
        world = make_world(backend)
        with backend.unbound():  # the ambient home names a profile that owns no store here
            scope = Scope(LOCAL_TENANT, str(backend.home("a")), LOCAL_OWNER)
            with push_state(scope=scope):  # names agent a, moves no home
                _refused(lambda: world.storage.store.entries("memory"))
                _refused(lambda: world.storage.store.add("memory", "redirected"))
        with backend.bind("a"):
            assert world.storage.store.entries("memory") == ["alpha note", "beta note"]

    # --- levels

    @pytest.mark.parametrize("level", ["participant", "owner", "tenant"])
    def test_a_write_at_a_level_the_store_does_not_hold_is_refused(self, backend, level):
        """A write says its level; one the backend does not hold is refused with a clear error and
        changes nothing — the local backend holds the agent level only."""
        world = make_world(backend)
        before = observe(backend, world.storage)
        with backend.bind("a"):
            for target in _TARGETS:
                with pytest.raises(MemoryLevelRefused):
                    world.storage.store.add(target, "new fact", level=level)
                with pytest.raises(MemoryLevelRefused):
                    world.storage.store.apply_batch(
                        target, [{"action": "add", "content": "new fact"}], level=level)
        assert observe(backend, world.storage) == before

    def test_the_agents_own_notes_are_never_a_participant_record(self, backend):
        """The agent's own notes are agent-level records only: a participant-level write on them
        is refused even for a backend that holds the participant level for the user profile."""
        world = make_world(backend)
        before = observe(backend, world.storage)
        with backend.bind("a"):
            with pytest.raises(MemoryLevelRefused):
                world.storage.store.add("memory", "new fact", level="participant")
            with pytest.raises(MemoryLevelRefused):
                world.storage.store.remove("memory", "alpha", level="participant")
        assert observe(backend, world.storage) == before

    def test_the_stores_levels_are_declared_not_discovered_from_writes(self, backend):
        world = make_world(backend)
        with backend.bind("a"):
            assert world.storage.store.levels <= {"participant", "agent", "owner", "tenant"}

    # --- caps refuse before the write

    def test_a_write_over_the_cap_is_refused_before_it_writes(self, backend):
        world = make_world(backend)
        store = world.storage.store
        limit = world.limit
        with backend.bind("a"):
            big = "x" * (limit - 100)  # one entry nearly filling the whole store
            assert store.add("memory", big)["success"] is True
            before = observe(backend, world.storage)
            refused = store.add("memory", "y" * 100)  # would exceed the cap
            assert refused["success"] is False and "chars" in refused["error"]
            assert observe(backend, world.storage) == before  # nothing landed, not even partially
            assert store.add("memory", "z" * 50)["success"] is True  # the room left is still writable

    def test_a_batch_over_the_cap_writes_nothing_not_even_its_first_operation(self, backend):
        world = make_world(backend)
        store = world.storage.store
        with backend.bind("a"):
            before = observe(backend, world.storage)
            refused = store.apply_batch("memory", [
                {"action": "remove", "old_text": "alpha"},  # frees a little room...
                {"action": "add", "content": "q" * (world.limit - 10)},  # ...not this much
            ])
            assert refused["success"] is False
            assert observe(backend, world.storage) == before  # the remove never landed either

    def test_a_batch_checks_the_cap_on_its_final_result_only(self, backend):
        """One call can free space and add entries: a batch whose final result fits goes through
        even when one of its adds alone would not."""
        world = make_world(backend)
        store = world.storage.store
        with backend.bind("a"):
            filler = "filler" + "f" * (world.limit - 106)
            assert store.add("memory", filler)["success"] is True
            result = store.apply_batch("memory", [
                {"action": "remove", "old_text": "filler"},  # frees the room the add needs
                {"action": "add", "content": "y" * (world.limit - 100)},
            ])
            assert result["success"] is True
            entries = store.entries("memory")
            assert "y" * (world.limit - 100) in entries and filler not in entries

    # --- replicas

    def test_two_handles_interleave_writes_without_loss(self, backend):
        first = make_world(backend).storage
        second = backend.replica("a")
        errors = []

        def add(storage, n):
            try:
                with backend.bind("a"):
                    for i in range(5):
                        assert storage.store.add("memory", f"entry {n}-{i}")["success"] is True
            except Exception as exc:  # health: allow BLE001 -- the thread's failure is asserted below
                errors.append(exc)

        threads = [_replica_thread(add, handle, n) for n, handle in enumerate((first, second))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)
        assert errors == []
        with backend.bind("a"):
            entries = first.store.entries("memory")
        assert sorted(entries[-10:]) == sorted(f"entry {n}-{i}" for n in range(2) for i in range(5))
