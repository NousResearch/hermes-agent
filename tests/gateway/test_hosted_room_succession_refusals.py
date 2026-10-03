"""Refusals in a fence round: only a refusal signed by the computer that gave it can stop a continuation
or move its epoch, and never by more than a bounded step."""

import pytest

from gateway import hosted_room_succession as succession
from gateway import hosted_room_succession_backup as backup
from gateway import hosted_room_succession_move as move
from tests.gateway.fixtures.succession import ROOM, context, copy_to, head, home_room, make_gateways


@pytest.fixture
def gateways(tmp_path):
    values = make_gateways(tmp_path, "h", "s", "e")
    home_room(values, "h", messages=3, successors=("s",))
    copy_to(values["h"], values["s"])
    copy_to(values["h"], values["e"])
    yield values
    for gateway in values.values():
        gateway.close()


def lying_refusal(e, monkeypatch, *, signed, epoch=2 ** 62):
    real, lie = backup.answer_fence, {"on": True}

    def answer_fence(ctx, request):
        if not lie["on"] or succession.local_install_id() != e.install_id:
            return real(ctx, request)
        detail = {"room_id": ROOM, "custodian_install_id": e.install_id, "fenced_epoch": epoch - 1,
                  "promise": {"epoch": epoch, "candidate_install_id": "install:" + "f" * 32}}
        if signed:
            detail["signature"] = succession.sign(succession.ANSWER, detail)
        raise succession.SuccessionError("promised", reason="room_authority_promised", detail=detail)
    monkeypatch.setattr(backup, "answer_fence", answer_fence)
    return lie


def test_an_unsigned_refusal_counts_as_no_answer(gateways, monkeypatch):
    s, e = gateways["s"], gateways["e"]
    lying_refusal(e, monkeypatch, signed=False)
    with s.acting():
        ctx = context(s, gateways, down=("h",))
        preview = move.preview(ctx, ROOM, s.install_id)
        move.continue_here(ctx, ROOM, s.install_id, preview_id=preview["preview_id"], confirm=True)
    assert head(s)["authoritative"] and head(s)["authority_epoch"] == 2


def test_a_signed_claim_stops_the_move_but_moves_the_epoch_only_a_bounded_step(gateways, monkeypatch):
    s, e = gateways["s"], gateways["e"]
    lie = lying_refusal(e, monkeypatch, signed=True)
    with s.acting():
        ctx = context(s, gateways, down=("h",))
        preview = move.preview(ctx, ROOM, s.install_id)
        with pytest.raises(succession.SuccessionError) as refused:
            move.continue_here(ctx, ROOM, s.install_id, preview_id=preview["preview_id"], confirm=True)
        assert refused.value.reason == "room_authority_promised"
        lie["on"] = False
        preview = move.preview(ctx, ROOM, s.install_id)
        move.continue_here(ctx, ROOM, s.install_id, preview_id=preview["preview_id"], confirm=True)
    assert head(s)["authority_epoch"] == 1 + move.MAX_EPOCH_STEP + 1
