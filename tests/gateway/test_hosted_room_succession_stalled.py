"""A step promised to a computer that never took it, in ask mode, across simulated gateways on a virtual
clock (``tests/gateway/fixtures/succession_world.py``).

H hosts; C and D are the successors the owner allowed. While H is down the owner continues on C, whose
attempt fences C for epoch 2 and then stops. When H comes back it pauses for that promise. It continues
past the step by itself only when every computer that could have continued the group answers that nothing
happened and promises the fresh step; otherwise it stays paused for the owner to decide.
"""

from contextlib import closing

import pytest

from gateway import hosted_room_custody as custody
from gateway import hosted_room_succession as succession
from gateway import hosted_room_succession_automatic as automatic
from gateway import hosted_room_succession_move as move
from gateway import hosted_rooms as rooms
from tests.gateway.fixtures.succession import ROOM, events
from tests.gateway.fixtures.succession_world import World


@pytest.fixture
def world(tmp_path, monkeypatch):
    value = World(tmp_path, monkeypatch, ("h", "c", "d"), voters=("h", "c", "d"), automatic_on=False)
    yield value
    value.close()


def owner_continues(world, name):
    gateway = world.gateways[name]
    with gateway.acting():
        ctx = world.context(gateway)
        preview = move.preview(ctx, ROOM, gateway.install_id)
        return move.continue_here(ctx, ROOM, gateway.install_id, preview_id=preview["preview_id"], confirm=True)


def stall_on_c(world, monkeypatch, *, split=False):
    """H goes down and the owner continues on C, whose attempt stops right after the fences."""
    world.advance(10, observe=False)
    assert world.status("h")["automatic"]["mode"] == "ask"
    world.network.stopped.add("h")
    if split:
        world.network.split({"c"}, {"d"})
    adopt = move._adopt

    def gives_up(ctx, room_id, record):
        monkeypatch.setattr(move, "_adopt", adopt)
        raise succession.SuccessionError("catching up made no progress", reason="target_not_ready")

    monkeypatch.setattr(move, "_adopt", gives_up)
    with pytest.raises(succession.SuccessionError):
        owner_continues(world, "c")
    assert not world.head("c")["authoritative"]


def back_and_paused(world):
    """H comes back and pauses at its first check: C holds a promise of the next step."""
    world.network.stopped.discard("h")
    world.until(lambda: not world.head("h")["serving"], limit=120, observe=False)
    assert world.status("h")["state"] == "moving"


def last_transition(world, name):
    found = [event["payload"] for event in events(world.gateways[name]) if event["kind"] == "authority.transition"]
    return found[-1]


def test_a_host_continues_past_a_step_nobody_took_once_every_successor_promises(world, monkeypatch):
    stall_on_c(world, monkeypatch)
    back_and_paused(world)
    h = world.gateways["h"]
    world.until(lambda: world.head("h")["serving"] and world.head("h")["authority_epoch"] == 3,
                limit=move.STALLED_PROMISE_SECONDS + 180, observe=True)
    moved = last_transition(world, "h")
    assert moved["reason"] == "automatic" and moved["proof"]["statement"] == succession.RECOVER_TEXT
    assert moved["proof"]["stalled"] == {"install_id": world.gateways["c"].install_id, "epoch": 2}
    assert sorted(world.by_id[r["custodian_install_id"]] for r in moved["proof"]["receipts"]) == ["c", "d", "h"]
    for name in ("c", "d"):
        world.until(lambda name=name: (world.head(name)["authority_gateway_id"], world.head(name)["authority_epoch"])
                    == (h.install_id, 3), limit=60, observe=True)
    world.advance(60, observe=True)
    assert world.admitting() == {("h", 3)}
    # Every computer checks the rule: the same proof without D's promise is refused.
    proof = dict(moved["proof"])
    proof["receipts"] = [r for r in proof["receipts"] if r["custodian_install_id"] != world.gateways["d"].install_id]
    with h.acting():
        proof["signature"] = succession.sign(succession.ATTESTATION, succession._unsigned(proof))
    c = world.gateways["c"]
    with c.acting(), closing(rooms._read_connection(c.db)) as conn:
        configuration = next({"configuration_seq": item["seq"], **{k: v for k, v in item.items() if k != "seq"}}
                             for item in custody.configurations_locked(conn, ROOM)
                             if item["seq"] == proof["configuration_seq"])
        verify = dict(proof_kind="attested", from_epoch=1, to_epoch=3, successor=h.install_id,
                      configuration=configuration)
        succession.verify_proof_locked(conn, ROOM, proof=moved["proof"], **verify)
        with pytest.raises(succession.ProofInvalid, match="every computer that could have continued"):
            succession.verify_proof_locked(conn, ROOM, proof=proof, **verify)


def test_a_host_stays_paused_while_a_successor_it_cant_reach_might_have_continued(world, monkeypatch):
    stall_on_c(world, monkeypatch)
    world.network.split({"h"}, {"d"})  # D is up, hosts nothing, but H can't hear that from it
    back_and_paused(world)
    world.until(lambda: world.status("h")["state"] == "paused", limit=move.STALLED_PROMISE_SECONDS + 120,
                observe=True)
    current = world.status("h")
    assert current["paused"]["reason"] == "step_not_taken"
    assert [item["install_id"] for item in current["paused"]["waiting_for"]] == [world.gateways["d"].install_id]
    assert {"action": "continue_anyway"} in current["actions"]
    world.advance(600, observe=True)
    assert world.admitting() == set() and world.head("h")["authority_epoch"] == 1
    # The owner decides: continuing anyway passes the step C never took.
    h = world.gateways["h"]
    with h.acting():
        assert automatic.continue_anyway(world.context(h), ROOM)["state"] == "ok"
    moved = last_transition(world, "h")
    assert moved["proof"]["statement"] == succession.ANYWAY_TEXT and moved["to_epoch"] == 3
    world.advance(30, observe=True)
    assert world.admitting() == {("h", 3)}


def test_a_host_never_continues_beside_a_host_it_cant_reach(world, monkeypatch):
    """C out of reach, the owner continued on D instead. H comes back reaching only C: it never continues
    by itself beside D, and once the link heals it follows D."""
    stall_on_c(world, monkeypatch, split=True)
    owner_continues(world, "d")
    assert world.head("d")["authoritative"] and world.head("d")["authority_epoch"] == 2
    world.network.split({"h"}, {"d"})
    back_and_paused(world)
    world.advance(move.STALLED_PROMISE_SECONDS + 300, observe=True)
    assert world.admitting() == {("d", 2)}
    current = world.status("h")
    assert current["state"] == "paused" and current["paused"]["reason"] == "step_not_taken"
    world.network.heal()
    world.until(lambda: world.status("h")["state"] == "moved_away", limit=180, observe=True)
    assert world.admitting() == {("d", 2)}
