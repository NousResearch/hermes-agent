"""The blob conformance suite: isolation invariants every blob backend meets.

Parametrized over backends through the ``backend`` fixture (:class:`Backend`). Each test talks
to the port only (``hermes_blob_port``); the fixture says how to bind each agent's scope and
open its store. Agents are named by short keys (``"a"``, ``"b"``); both belong to one tenant.

Covered:

- pinning: every read (``get``, ``exists``, both ``list`` shapes) and every write (``put``,
  ``delete``) is refused for another agent, another tenant and no pin, and a refusal changes
  nothing the agent can observe through the port (:func:`observe`: every visible key and its
  bytes); a positive control shows each operation succeeds for its own agent;
- a foreign blob and a missing one are indistinguishable: the same refusal for a key the
  agent holds and one it does not, under every pin but its own, and another agent's store
  sees nothing at the key — not its bytes, not its presence, not in a listing — while a write
  under that key lands in the writer's own namespace and never overwrites the other agent's
  blob;
- containment: a path-like key (``..``, an absolute path, a separator, a drive colon, a NUL)
  cannot escape the agent's namespace — every operation on it is refused as malformed, never
  resolved elsewhere and never reported as merely missing — and neither can a symlink planted
  inside the tree pointing outside it;
- levels: a backend that does not store the participant level refuses a write naming it with
  the level in the error (:class:`BlobLevelRefusalConformance`); one that does stores
  conversation files per participant (:class:`ParticipantConformance`): a member of a group
  agent cannot read, list or delete another member's conversation file, a participant's record
  resolves before an agent record of the same key, two members' files of one key never
  collide, a member still reads the agent's own files, and an actor that is no principal
  cannot write a participant record;
- the turn fence on conversation-file writes (:class:`ParticipantWriteFenceConformance`): a
  participant-level write with no bound turn is refused, one under a fence its session's lease
  no longer holds is refused with nothing written, one under the current lease lands, work
  designed to outlive the turn keeps the fence's owner-epoch half (it writes after the session's
  next claim and is refused after a takeover), and an agent-level write is never turn-fenced.
  Today's local backend stores no participant level, so nothing changes there: its participant
  writes are refused by the level conformance before any fence is asked for, and this class
  runs only against backends that store the participant level.

NOT covered. A backend that passes this suite may still violate these, and must not claim
them:

- who may read a conversation (participation is the session store's business: the blob port
  scopes a conversation file to the principal whose turn wrote it, and a participant-level
  blob is visible to that principal alone);
- the owner-wide and tenant-wide levels (blobs have none);
- retention and cleanup (a deleted or swept blob leaves nothing readable through the port,
  but when a sweep runs and what it may remove is not asserted here);
- the hosted isolation assertion (``BlobStoreProvider.assert_isolating`` failing for a role
  that bypasses tenant isolation); only that the method is required is tested, elsewhere.

The fixture cannot vary tenant names, agent ids or principals beyond the members it binds, so
cross-tenant invariants are exercised through the one foreign tenant it supplies.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import pytest

from hermes_blob_port import (
    BlobFenceRequired, BlobFenceSuperseded, BlobKeyRefused, BlobLevelRefused, BlobNotFound,
    BlobPinRefused, BlobStore,
)


@dataclass
class Backend:
    """How the suite drives one backend.

    - ``open(agent)``: the agent's store.
    - ``bind(agent)``: bind the agent's owner scope (a non-turn activity of its owner).
    - ``foreign_tenant(agent)``: bind a scope of ANOTHER tenant that names the same agent id.
    - ``unbound()``: run with no scope pinned. A local backend's pin is then the ambient
      profile, so its ``unbound`` names a third profile; a hosted one binds nothing.
    - ``bind_member(agent, member)``: bind a member principal's scope on the agent (a
      participant). Required by :class:`ParticipantConformance`.
    - ``bind_system(agent)``: bind a non-principal actor's scope on the agent. Required by
      :class:`ParticipantConformance`'s non-principal test.
    - ``claim(agent, session)``: the lease claim of *session*'s next input, on the backend's
      admission side, returning the fence it issued (``hermes_scope.Fence``). Required by
      :class:`ParticipantConformance`'s writes and :class:`ParticipantWriteFenceConformance`.
    - ``turn(agent, member, session, fence, outlives=False)``: bind *member*'s claimed turn of
      *session* under *fence* — the scope a conversation-file write runs under (*outlives*:
      work designed to outlive the turn, keeping the fence's owner-epoch half). Required by
      the same classes.
    - ``supersede(agent, session, how)``: move the lease out from under the fence a claim
      issued — ``"owner_epoch"`` (another owner took the agent over) or ``"next_claim"`` (the
      session's next input was claimed elsewhere). Required by
      :class:`ParticipantWriteFenceConformance`.
    - ``key_path(agent, key)``: where a key names its file on disk. Required by the
      planted-symlink test; without it that test fails on the missing hook, not the backend.
    """

    open: Callable[[str], BlobStore]
    bind: Callable[[str], AbstractContextManager]
    foreign_tenant: Callable[[str], AbstractContextManager]
    unbound: Callable[[], AbstractContextManager]
    bind_member: Optional[Callable[[str, str], AbstractContextManager]] = None
    bind_system: Optional[Callable[[str], AbstractContextManager]] = None
    claim: Optional[Callable[[str, str], Any]] = None
    turn: Optional[Callable[..., AbstractContextManager]] = None
    supersede: Optional[Callable[[str, str, str], None]] = None
    key_path: Optional[Callable[[str, str], Path]] = None


#: The blobs agent ``a`` holds when a per-operation test runs, one per family.
WORLD_BLOBS: dict[str, bytes] = {
    "media/image/photo.png": b"photo-bytes",
    "media/audio/voice.ogg": b"voice-bytes",
    "attachments/room.bin": b"attachment-bytes",
    "pets/ghost/sheet.webp": b"sheet-bytes",
    "skills/greet/SKILL.md": b"skill-bytes",
}


def make_world(backend) -> BlobStore:
    """Agent ``a``'s store holding one blob per family, written at the agent level."""
    store = backend.open("a")
    with backend.bind("a"):
        for key, data in WORLD_BLOBS.items():
            store.put(key, data, level="agent")
    return store


def observe(backend, store) -> dict[str, bytes]:
    """Everything the port exposes about agent ``a``: every visible key and its bytes. A
    refused operation must leave all of it unchanged."""
    with backend.bind("a"):
        return {key: store.get(key) for key in store.list("")}


def _foreign_pins(backend):
    """Every pin but agent ``a``'s own: another agent, another tenant naming ``a``, no pin."""
    return {"other_agent": lambda: backend.bind("b"), "other_tenant": lambda: backend.foreign_tenant("a"),
            "unbound": backend.unbound}


def _refused(call: Callable[[], Any]) -> str:
    """The refusal *call* raises (its message), asserting it is the store's pin refusal."""
    with pytest.raises(BlobPinRefused) as refused:
        call()
    return str(refused.value)


def member_turn(backend, agent: str, member: str) -> AbstractContextManager:
    """*member*'s claimed turn on the agent: the scope a conversation-file write runs under.
    Its fence is the lease the claim issued, which the backend checks inside the write."""
    assert backend.claim is not None and backend.turn is not None, (
        "the backend fixture must supply claim and turn: a conversation-file write runs under "
        "its turn's fence")
    return backend.turn(agent, member, "s1", backend.claim(agent, "s1"))


#: Every read, naming blobs agent ``a`` holds in the world.
ALL_READS: dict[str, Callable[[BlobStore], Any]] = {
    "get": lambda store: store.get("pets/ghost/sheet.webp"),
    "exists": lambda store: store.exists("pets/ghost/sheet.webp"),
    "list_all": lambda store: store.list(""),
    "list_prefix": lambda store: store.list("pets/"),
}

#: Every write, with arguments that succeed in the world.
ALL_WRITES: dict[str, Callable[[BlobStore], Any]] = {
    "put_new": lambda store: store.put("media/image/fresh.png", b"fresh", level="agent"),
    "put_overwrite": lambda store: store.put("pets/ghost/sheet.webp", b"replaced", level="agent"),
    "delete": lambda store: store.delete("media/audio/voice.ogg", level="agent"),
}

#: Keys that cannot name a blob of the agent's namespace: parent components, an absolute
#: path, separators and drives inside components, empty and current-directory components.
ESCAPING_KEYS = (
    "../escape.txt", "media/image/../../escape.txt", "pets/../escape.txt", "/etc/passwd",
    r"media\image\x.png", "pets/ghost/..", "skills/./SKILL.md", "media/clip:01.png",
    "pets/ghost/\x00name", "media/image//photo.png",
)


class PinningConformance:
    """Every operation is pinned to its agent, and a key cannot leave the namespace. Inherit
    and supply a ``backend`` fixture."""

    @pytest.mark.parametrize("operation", sorted(ALL_READS))
    def test_every_read_is_refused_for_any_pin_but_its_agents(self, backend, operation):
        store = make_world(backend)
        for bind in _foreign_pins(backend).values():
            with bind():
                _refused(lambda: ALL_READS[operation](store))
        with backend.bind("a"):  # the positive control: its own pin reads
            assert ALL_READS[operation](store)

    @pytest.mark.parametrize("operation", sorted(ALL_WRITES))
    @pytest.mark.parametrize("pin", ["other_agent", "other_tenant", "unbound"])
    def test_every_write_is_refused_for_any_pin_but_its_agents_and_changes_nothing(self, backend, operation, pin):
        store = make_world(backend)
        before = observe(backend, store)
        with _foreign_pins(backend)[pin]():
            _refused(lambda: ALL_WRITES[operation](store))
        assert observe(backend, store) == before

    @pytest.mark.parametrize("operation", sorted(ALL_WRITES))
    def test_every_write_succeeds_for_its_own_agent(self, backend, operation):
        """The positive control for the refusal tests: each write is one the agent may make,
        and it changes what the agent observes."""
        store = make_world(backend)
        before = observe(backend, store)
        with backend.bind("a"):
            ALL_WRITES[operation](store)
        assert observe(backend, store) != before

    def test_a_foreign_blob_is_indistinguishable_from_a_missing_one(self, backend):
        """Under every pin but its own, the refusal for a key the agent holds is the same
        refusal as for one it does not: the pin says nothing about what exists."""
        store = make_world(backend)
        for bind in _foreign_pins(backend).values():
            with bind():
                assert _refused(lambda: store.get("pets/ghost/sheet.webp")) == \
                    _refused(lambda: store.get("pets/ghost/absent.webp"))
                assert _refused(lambda: store.exists("pets/ghost/sheet.webp")) == \
                    _refused(lambda: store.exists("pets/ghost/absent.webp"))

    def test_another_agent_cannot_read_list_or_overwrite_a_blob(self, backend):
        """Through the other agent's own store the key names nothing — no bytes, no presence,
        no listing — and a write under it lands in the writer's namespace alone: the first
        agent's blob is still the first agent's."""
        store = make_world(backend)
        theirs = backend.open("b")
        with backend.bind("b"):
            with pytest.raises(BlobNotFound):
                theirs.get("pets/ghost/sheet.webp")
            assert not theirs.exists("pets/ghost/sheet.webp")
            assert "pets/ghost/sheet.webp" not in theirs.list("pets/")
            theirs.put("pets/ghost/sheet.webp", b"theirs", level="agent")
            assert theirs.get("pets/ghost/sheet.webp") == b"theirs"
        with backend.bind("a"):
            assert store.get("pets/ghost/sheet.webp") == b"sheet-bytes"

    @pytest.mark.parametrize("key", ESCAPING_KEYS)
    def test_path_like_keys_cannot_escape_the_namespace(self, backend, key):
        """A key that would leave the agent's namespace is refused as malformed by every
        operation — never resolved elsewhere, never reported as merely missing."""
        store = make_world(backend)
        before = observe(backend, store)
        with backend.bind("a"):
            for operation in (lambda: store.get(key), lambda: store.exists(key),
                              lambda: store.put(key, b"escape"),
                              lambda: store.delete(key, level="agent"),
                              lambda: store.list(key)):
                with pytest.raises(BlobKeyRefused):
                    operation()
        assert observe(backend, store) == before

    def test_an_escaping_key_creates_nothing_outside_the_namespace(self, backend):
        """The refused writes landed nothing: the file the escaping key tried to name, one
        level above the namespace's own tree, was never created. ``key_path`` says where a
        key names its file, so the test can name the directory the key aimed to leave
        through from the same depth in both backends' trees."""
        assert backend.key_path is not None, "the backend fixture must supply key_path"
        store = backend.open("a")
        make_world(backend)  # the agent's tree exists, so the escape has a target directory
        escape = backend.key_path("a", "pets/ghost/sheet.webp").parents[3] / "escape.txt"
        assert escape.parent.exists()
        with backend.bind("a"):
            with pytest.raises(BlobKeyRefused):
                store.put("pets/../../escape.txt", b"gone")
        assert not escape.exists()

    def test_a_symlink_planted_in_the_namespace_cannot_escape(self, backend, tmp_path):
        """A link planted inside the tree pointing outside it names no blob of this store:
        reading it is a refusal, writing through it never touches its target, and it appears
        in no listing. ``key_path`` says where a key names its file."""
        assert backend.key_path is not None, "the backend fixture must supply key_path"
        store = make_world(backend)
        outside = tmp_path / "outside-secret.bin"
        outside.write_bytes(b"outside")
        planted = backend.key_path("a", "pets/ghost/sheet.webp")
        planted.parent.parent.joinpath("planted").symlink_to(outside)
        with backend.bind("a"):
            with pytest.raises(BlobKeyRefused):
                store.get("pets/planted")
            with pytest.raises(BlobKeyRefused):
                store.put("pets/planted", b"hijack")
            assert "pets/planted" not in store.list("pets/")
        assert outside.read_bytes() == b"outside"


class BlobLevelRefusalConformance:
    """A backend that does not store the participant level refuses a write naming it, with the
    level in the error, and stores nothing anyway. Inherit and supply a ``backend`` fixture
    whose backend declares the agent level only."""

    @pytest.mark.parametrize("level", ["participant", "bogus"])
    def test_a_write_at_a_level_the_backend_does_not_store_is_refused(self, backend, level):
        store = backend.open("a")
        before = observe(backend, store)
        with backend.bind("a"):
            for write in (lambda: store.put("media/image/inbound.png", b"x", level=level),
                          lambda: store.delete("media/image/inbound.png", level=level)):
                with pytest.raises(BlobLevelRefused) as refused:
                    write()
                assert level in str(refused.value)
            assert not store.exists("media/image/inbound.png")
        assert observe(backend, store) == before


class ParticipantConformance:
    """Conversation files are per participant: a member of a group agent sees another member's
    nowhere, and the participant level resolves before the agent level. Inherit and supply a
    ``backend`` fixture whose backend stores the participant level (``bind_member``,
    ``bind_system``, ``claim`` and ``turn`` required: a conversation-file write runs under its
    writer's claimed turn)."""

    def test_a_member_cannot_read_list_or_delete_another_members_conversation_file(self, backend):
        assert backend.bind_member is not None, "the backend fixture must supply bind_member"
        store = make_world(backend)
        with member_turn(backend, "a", "m1"):
            store.put("media/image/inbound.png", b"m1-photo", level="participant")
        with backend.bind_member("a", "m2"):
            with pytest.raises(BlobNotFound):
                store.get("media/image/inbound.png")
            assert not store.exists("media/image/inbound.png")
            assert "media/image/inbound.png" not in store.list("media/")
        with member_turn(backend, "a", "m2"):
            assert store.delete("media/image/inbound.png", level="participant") is False
        with backend.bind_member("a", "m1"):
            assert store.get("media/image/inbound.png") == b"m1-photo"

    def test_a_participant_record_resolves_before_an_agent_record_of_the_same_key(self, backend):
        assert backend.bind_member is not None, "the backend fixture must supply bind_member"
        store = make_world(backend)
        with backend.bind("a"):
            store.put("pets/ghost/shared.bin", b"agent-level", level="agent")
        with member_turn(backend, "a", "m1"):
            store.put("pets/ghost/shared.bin", b"m1-private", level="participant")
        with backend.bind_member("a", "m1"):
            assert store.get("pets/ghost/shared.bin") == b"m1-private"
        with backend.bind_member("a", "m2"):
            assert store.get("pets/ghost/shared.bin") == b"agent-level"
        with backend.bind("a"):
            assert store.get("pets/ghost/shared.bin") == b"agent-level"

    def test_two_members_conversation_files_of_one_key_never_collide(self, backend):
        assert backend.bind_member is not None, "the backend fixture must supply bind_member"
        store = backend.open("a")
        with member_turn(backend, "a", "m1"):
            store.put("media/image/inbound.png", b"m1-photo", level="participant")
        with member_turn(backend, "a", "m2"):
            store.put("media/image/inbound.png", b"m2-photo", level="participant")
        with backend.bind_member("a", "m2"):
            assert store.get("media/image/inbound.png") == b"m2-photo"
        with backend.bind_member("a", "m1"):
            assert store.get("media/image/inbound.png") == b"m1-photo"

    def test_a_member_still_reads_the_agents_own_files(self, backend):
        assert backend.bind_member is not None, "the backend fixture must supply bind_member"
        store = make_world(backend)
        with backend.bind_member("a", "m2"):
            assert store.get("pets/ghost/sheet.webp") == b"sheet-bytes"
            assert store.exists("pets/ghost/sheet.webp")
            assert "pets/ghost/sheet.webp" in store.list("pets/")

    def test_a_participant_write_by_an_actor_that_is_no_principal_is_refused(self, backend):
        assert backend.bind_system is not None, "the backend fixture must supply bind_system"
        store = backend.open("a")
        before = observe(backend, store)
        with backend.bind_system("a"):
            with pytest.raises(BlobLevelRefused):
                store.put("media/image/inbound.png", b"x", level="participant")
        assert observe(backend, store) == before


class ParticipantWriteFenceConformance:
    """A conversation-file write is session-owned data, so it runs under its turn's claim: the
    backend checks the bound turn's fence against the session's lease inside the write, before
    anything of it lands. A participant-level write with no turn bound is refused, one under a
    fence the lease no longer holds is refused with nothing written, one under the current lease
    lands, work designed to outlive the turn keeps the fence's owner-epoch half, and an
    agent-level write is never turn-fenced. Inherit and supply a ``backend`` fixture whose
    backend stores the participant level (``claim``, ``turn`` and ``supersede`` required)."""

    def _hooks(self, backend):
        assert backend.claim is not None and backend.turn is not None \
            and backend.supersede is not None, (
            "the backend fixture must supply claim, turn and supersede: a conversation-file "
            "write is fenced by its turn's claim")

    def test_a_conversation_file_write_with_no_turn_bound_is_refused(self, backend):
        self._hooks(backend)
        store = make_world(backend)
        before = observe(backend, store)
        with backend.bind_member("a", "m1"):
            for write in (lambda: store.put("media/image/inbound.png", b"unfenced", level="participant"),
                          lambda: store.delete("media/image/inbound.png", level="participant")):
                with pytest.raises(BlobFenceRequired):
                    write()
            assert not store.exists("media/image/inbound.png")
            assert "media/image/inbound.png" not in store.list("media/")
        assert observe(backend, store) == before

    def test_a_conversation_file_write_under_its_sessions_current_lease_lands(self, backend):
        self._hooks(backend)
        store = make_world(backend)
        fence = backend.claim("a", "s1")
        with backend.turn("a", "m1", "s1", fence):
            store.put("media/image/inbound.png", b"fenced", level="participant")
        with backend.bind_member("a", "m1"):
            assert store.get("media/image/inbound.png") == b"fenced"

    @pytest.mark.parametrize("how", ["owner_epoch", "next_claim"])
    def test_a_conversation_file_write_under_a_superseded_fence_is_refused_and_writes_nothing(
            self, backend, how):
        self._hooks(backend)
        store = make_world(backend)
        fence = backend.claim("a", "s1")
        with backend.turn("a", "m1", "s1", fence):
            store.put("media/image/inbound.png", b"first", level="participant")
        backend.supersede("a", "s1", how)
        with backend.turn("a", "m1", "s1", fence):
            with pytest.raises(BlobFenceSuperseded):
                store.put("media/image/inbound.png", b"stale", level="participant")
            with pytest.raises(BlobFenceSuperseded):
                store.delete("media/image/inbound.png", level="participant")
        with backend.bind_member("a", "m1"):
            assert store.get("media/image/inbound.png") == b"first"

    def test_work_that_outlives_the_turn_writes_after_the_sessions_next_claim(self, backend):
        self._hooks(backend)
        store = make_world(backend)
        fence = backend.claim("a", "s1")
        backend.supersede("a", "s1", "next_claim")
        with backend.turn("a", "m1", "s1", fence, outlives=True):
            store.put("media/image/inbound.png", b"outlives", level="participant")
        with backend.bind_member("a", "m1"):
            assert store.get("media/image/inbound.png") == b"outlives"

    def test_work_that_outlives_the_turn_is_refused_after_a_takeover(self, backend):
        self._hooks(backend)
        store = make_world(backend)
        fence = backend.claim("a", "s1")
        backend.supersede("a", "s1", "owner_epoch")
        with backend.turn("a", "m1", "s1", fence, outlives=True):
            with pytest.raises(BlobFenceSuperseded):
                store.put("media/image/inbound.png", b"outlives", level="participant")
        with backend.bind_member("a", "m1"):
            assert not store.exists("media/image/inbound.png")

    def test_an_agent_level_write_is_never_turn_fenced(self, backend):
        self._hooks(backend)
        store = make_world(backend)
        fence = backend.claim("a", "s1")
        backend.supersede("a", "s1", "owner_epoch")
        with backend.turn("a", "m1", "s1", fence):
            store.put("pets/ghost/sheet.webp", b"replaced", level="agent")
            assert store.delete("media/audio/voice.ogg", level="agent") is True
        with backend.bind("a"):
            assert store.get("pets/ghost/sheet.webp") == b"replaced"
            assert not store.exists("media/audio/voice.ogg")


class BlobConformance(PinningConformance, ParticipantConformance, ParticipantWriteFenceConformance):
    """The whole suite for a backend that stores both levels. Inherit and supply a ``backend``
    fixture (or inherit the parts separately, e.g. one test module each, to spread a slow
    backend over several runners)."""
