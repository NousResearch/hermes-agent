"""Canonical turns survive custody suppression; discard uses the live writer."""
import asyncio
import json
import time

import pytest

from tests.gateway.test_classic_current_export import (  # noqa: F401 -- shared pytest fixture
    classic_runtime, rpc, _classic_request, _publish_classic_file,
)


@pytest.mark.asyncio
async def test_group_discard_before_queued_cancel_does_not_strand_admission(classic_runtime):
    from gateway.session_contract import SessionRef
    f = classic_runtime
    installation, request = await _classic_request(f, "discard-before-cancel")
    held, release = asyncio.Event(), asyncio.Event()
    executed = []

    async def handle(event):
        executed.append(event.text)
        held.set()
        await release.wait()
        return "done"

    f.runner._handle_message = handle
    await rpc(f, "prompt.submit", session_id=f.session_id, input_id="head", text="head")
    queued = await rpc(f, "prompt.submit", session_id=f.session_id,
                       text="never execute", classic_export=request)
    drain = asyncio.create_task(f.authority._drain(SessionRef(str(f.home), f.session_id)))
    try:
        await asyncio.wait_for(held.wait(), timeout=5)
        retired = await rpc(f, "session.export.discard", session_id=f.session_id,
                            installation=installation, group_id=request["group_id"])
        assert retired.get("result") == {"retired": True}
        cancelled = await rpc(f, "prompt.cancel", session_id=f.session_id,
                              admission_id=queued["result"]["admission_id"])
        assert cancelled["result"]["status"] == "terminal"
        assert cancelled["result"]["outcome"] == "cancelled"
        assert (await rpc(f, "prompt.receipt", session_id=f.session_id,
                          admission_id=queued["result"]["admission_id"]))["result"]["classic_export"]["state"] == "retired"
    finally:
        release.set()
        await asyncio.wait_for(drain, timeout=5)
    assert executed == ["head"]


@pytest.mark.asyncio
@pytest.mark.parametrize("suppression", ["exact", "group", "expiry"])
async def test_preterminal_suppression_keeps_canonical_completion(classic_runtime, suppression):
    from gateway.session_contract import SessionRef
    from gateway.session_results import execution_result
    from hermes_state_runtime import get_session_admission
    from tools import hosted_room_artifact  # noqa: F401
    from tools.registry import registry
    f = classic_runtime
    installation, request = await _classic_request(f, "suppressed-finish")
    path = f.home / "private.txt"
    path.write_bytes(b"private exact bytes")
    submitted = {}
    blob_paths = []

    async def handle(_event):
        shared = json.loads(await asyncio.to_thread(
            registry.dispatch, "share_group_file", {"path": str(path)}, task_id="default"))
        assert shared["ok"] is True
        with f.db._read_ctx() as conn:
            names = [row[0] for row in conn.execute("SELECT blob_name FROM hosted_room_output_artifacts")]
        blob_paths.extend(f.home / "hosted-room-artifact-outbox" / "blobs" / n for n in names)
        if suppression == "expiry":
            f.db._execute_write(lambda conn: conn.execute(
                "UPDATE classic_output_exports SET expires=? WHERE export_id=?",
                (time.time() - 1, submitted["classic_export"]["export_id"])))
        else:
            selectors = {"export_id": submitted["classic_export"]["export_id"]} if suppression == "exact" else {}
            reply = await rpc(f, "session.export.discard", session_id=f.session_id,
                              installation=installation, group_id=request["group_id"], **selectors)
            assert reply.get("result") == {"retired": True}
        execution_result.get().update(result={"final_response": "done", "messages": [], "completed": True}, usage={})
        return "done"

    f.runner._handle_message = handle
    submitted.update((await rpc(f, "prompt.submit", session_id=f.session_id,
                                text="share", classic_export=request))["result"])
    await f.authority._drain(SessionRef(str(f.home), f.session_id))
    row = get_session_admission(f.db, admission_id=submitted["admission_id"])
    assert (row["status"], row["outcome"]) == ("terminal", "completed")
    with f.db._read_ctx() as conn:
        assert conn.execute("SELECT state FROM classic_output_exports WHERE export_id=?",
                            (submitted["classic_export"]["export_id"],)).fetchone()[0] == "retired"
        assert conn.execute("SELECT COUNT(*) FROM hosted_room_output_artifacts").fetchone()[0] == 0
    assert blob_paths and not any(p.exists() for p in blob_paths)


@pytest.mark.asyncio
async def test_unknown_resolution_after_exact_discard_keeps_terminal_transition(classic_runtime):
    from hermes_state_runtime import begin_runtime_epoch, claim_session_input, recover_session_inputs
    f = classic_runtime
    installation, request = await _classic_request(f, "discard-before-unknown")
    submitted = (await rpc(f, "prompt.submit", session_id=f.session_id,
                           text="uncertain turn", classic_export=request))["result"]
    claimed = claim_session_input(f.db, epoch=f.authority.epoch, session_id=f.session_id)
    assert claimed is not None
    assert (await rpc(f, "session.export.discard", session_id=f.session_id,
                      installation=installation, group_id=request["group_id"],
                      export_id=submitted["classic_export"]["export_id"]))["result"] == {"retired": True}
    epoch = begin_runtime_epoch(f.db, instance_id="recovered-owner")
    assert recover_session_inputs(f.db, epoch=epoch) == 1
    f.authority.epoch = epoch
    resolved = await rpc(f, "prompt.resolve_unknown", session_id=f.session_id,
                         admission_id=claimed["admission_id"], execution_generation=claimed["generation"])
    assert resolved["result"]["status"] == "terminal"
    assert resolved["result"]["outcome"] == "interrupted"


def _custody_snapshot(f):
    tables = ["classic_output_exports", "classic_retired_groups", "hosted_room_output_artifacts",
              "hosted_room_output_generation_fences"]
    with f.db._read_ctx() as conn:
        rows = {t: [tuple(r) for r in conn.execute(f"SELECT * FROM {t} ORDER BY rowid")] for t in tables}
    root = f.home / "hosted-room-artifact-outbox" / "blobs"
    return rows, {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}


@pytest.mark.asyncio
@pytest.mark.parametrize("selector", ["group", "exact"])
async def test_stale_epoch_discard_has_no_custody_mutation(classic_runtime, selector):
    from hermes_state_runtime import begin_runtime_epoch
    f = classic_runtime
    installation, request, _submitted, terminal, _item = await _publish_classic_file(f, "stale-discard")
    begin_runtime_epoch(f.db, instance_id="replacement-owner")
    before = _custody_snapshot(f)
    selectors = {"export_id": terminal["classic_export"]["export_id"]} if selector == "exact" else {}
    reply = await rpc(f, "session.export.discard", session_id=f.session_id,
                      installation=installation, group_id=request["group_id"], **selectors)
    assert "error" in reply
    assert _custody_snapshot(f) == before


@pytest.mark.asyncio
async def test_unrelated_group_discard_cannot_run_expiry_housekeeping(classic_runtime):
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.session_authority import LiveSession
    f = classic_runtime
    installation, request, _s, _terminal, _item = await _publish_classic_file(f, "group-a", group_id="group-a")
    _i, other_request = await _classic_request(f, "unrelated-expired", group_id="other-group")
    expired = (await rpc(f, "prompt.submit", session_id=f.session_id,
                         text="queued unrelated turn", classic_export=other_request))["result"]
    f.db._execute_write(lambda conn: conn.execute(
        "UPDATE classic_output_exports SET expires=? WHERE export_id=?",
        (time.time() - 1, expired["classic_export"]["export_id"])))
    other = "unrelated-session"
    f.db.create_session(other, source="gui")
    source = SessionSource(platform=Platform.LOCAL, chat_id=other,
                           user_id=f.connection.actor.subject, chat_type="dm")
    f.authority.sessions[other] = LiveSession(source, "other-route")
    f.connection.subscriptions[other] = "other-subscription"
    before = _custody_snapshot(f)
    reply = await rpc(f, "session.export.discard", session_id=other,
                      installation=installation, group_id=request["group_id"])
    assert reply["error"]["message"] == "classic_export_unavailable"
    assert _custody_snapshot(f) == before
