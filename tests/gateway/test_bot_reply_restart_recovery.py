"""A settled Bot Chat result can survive the interruption of its sender's relay."""

import json

import pytest

from gateway.session_results import finish_result
from gateway.session_unattended_unknown import settle_recovered_unattended
from hermes_state_runtime import (
    admit_session_input,
    begin_runtime_epoch,
    claim_session_input,
    get_session_admission,
    recover_session_inputs,
)
from tools.bot_live_delivery import DELIVERY_DIR_NAME


@pytest.mark.parametrize("shutdown", ["stop", "crash"])
@pytest.mark.parametrize("saved_reply", ["The fictional room is booked.", None])
def test_interrupted_relay_recovers_only_a_settled_reply(owner, tmp_path, shutdown, saved_reply):
    db = owner.db
    db.create_session("sender", source="test")
    recipient = tmp_path / "recipient"
    delivery_id = "a" * 32
    if saved_reply:
        mailbox = recipient / "runtime" / DELIVERY_DIR_NAME
        mailbox.mkdir(parents=True)
        (mailbox / f"{delivery_id}.json").write_text(json.dumps({
            "status": "settled", "reply": saved_reply, "delivery_id": delivery_id,
        }))
    payload = {
        "text": "Synthetic local completion",
        "local_automation_v1": {
            "identity": '["completion","proc_fixture",1]',
            "bot_reply_receipt": {"home": str(recipient), "delivery_id": delivery_id},
        },
    }
    adm = admit_session_input(db, epoch=owner.epoch, principal_id="automation:test",
                              session_id="sender", request_id="completion", payload=payload)
    claimed = claim_session_input(db, epoch=owner.epoch, session_id="sender")
    db.append_message("sender", "user", payload["text"], display_kind="process_complete")
    db.append_message("sender", "assistant", "The fictional room")
    if shutdown == "stop":
        finish_result(db, epoch=owner.epoch, row=claimed, response="", outcome="interrupted",
                      result={"result": {"interrupted": True}, "usage": {}}, stopping=True)

    for index in range(2):
        epoch = begin_runtime_epoch(db, instance_id=f"restart-{index}")
        recover_session_inputs(db, epoch=epoch)
        settle_recovered_unattended(db, epoch)
    assert get_session_admission(db, admission_id=adm["admission_id"])["outcome"] == "interrupted"
    rows = db.get_messages("sender")
    if saved_reply:
        assert [row["content"] for row in rows if row["role"] == "assistant"
                and row["display_kind"] != "hidden"] == [saved_reply]
    else:
        assert not any(row["content"] == "The fictional room is booked." for row in rows)
        if shutdown == "crash":
            assert rows[-1]["display_kind"] == "failed_turn"
