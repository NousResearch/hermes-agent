"""Current canonical admission -> classic share -> exact authorized read."""

import asyncio
import base64
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def classic_runtime(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionEntry, SessionSource, SessionStore
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import LocalSessionAdapter
    from gateway.session_policy import build_policy
    import hermes_state
    from hermes_state_local import POLICY_PREFIX
    from hermes_state_local_migration import LEGACY_PREFIX
    from hermes_state_runtime import begin_runtime_epoch

    home = tmp_path.resolve()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")
    (home / "install_id").write_text("c" * 32, encoding="ascii")
    store = SessionStore(home / "sessions", GatewayConfig())
    db = store._db
    runner = SimpleNamespace(
        adapters={},
        _draining=False,
        config=GatewayConfig(),
        session_store=store,
        _session_db=db,
        _cached_agent_for=lambda _route: None,
    )
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = SessionAuthority(
        runner,
        profile_id=str(home),
        instance_id="classic-current-test",
        db=db,
        epoch=begin_runtime_epoch(db, instance_id="classic-current-test"),
    )
    runner.session_authority = authority
    transport = object()
    connection = AuthorityConnection(
        authority,
        transport,
        {
            "user_id": "owner",
            "profile_id": str(home),
            "instance_id": "classic-current-test",
            "capabilities": ["session:read", "session:submit", "session:control"],
        },
    )
    session_id = "classic-producer"
    source = SessionSource(
        platform=Platform.LOCAL,
        chat_id=session_id,
        user_id=connection.actor.subject,
        chat_type="dm",
    )
    route = store._generate_session_key(source)
    entry = SessionEntry(
        route,
        session_id,
        datetime.fromtimestamp(1),
        datetime.fromtimestamp(1),
        origin=source,
        platform=Platform.LOCAL,
    )
    policy = build_policy(
        {"source": "gui", "cwd": str(home), "model": "fixture", "toolsets": []},
        {},
    )
    receipt = {
        "profile_id": str(home),
        "principal_id": source.user_id,
        "session_id": session_id,
        "legacy_session_id": session_id,
        "request_id": "existing-" + session_id,
        "route": route,
        "entry": entry.to_dict(),
        "policy": asdict(policy),
    }

    def seed(conn):
        conn.execute(
            "INSERT INTO sessions(id,source,user_id,session_key,chat_id,started_at) VALUES(?,?,?,?,?,?)",
            (session_id, "gui", source.user_id, route, session_id, 1.0),
        )
        conn.execute(
            "INSERT INTO state_meta(key,value) VALUES(?,?)",
            (POLICY_PREFIX + session_id, json.dumps(receipt)),
        )
        conn.execute(
            "INSERT INTO state_meta(key,value) VALUES(?,?)",
            (
                LEGACY_PREFIX + session_id,
                json.dumps(
                    {
                        key: receipt[key]
                        for key in (
                            "session_id",
                            "profile_id",
                            "principal_id",
                            "legacy_session_id",
                        )
                    }
                ),
            ),
        )

    db._execute_write(seed)
    authority.sessions[session_id] = LiveSession(source, route)
    adapter = LocalSessionAdapter(authority)
    adapter.policies[session_id] = policy
    adapter.register_source(source)
    runner.adapters[Platform.LOCAL] = adapter
    with store._lock:
        store._ensure_loaded_locked()
        store._entries[route] = entry
    # Model the already-authenticated owner attachment. The behavior under test
    # starts at the registered prompt RPC; no legacy restore/build path is used.
    connection.subscriptions[session_id] = "fixture-subscription"
    monkeypatch.setattr(authority, "_schedule", lambda _ref: None)
    fixture = SimpleNamespace(
        home=home,
        db=db,
        runner=runner,
        authority=authority,
        connection=connection,
        session_id=session_id,
    )
    try:
        yield fixture
    finally:
        await connection.close()
        db.close()


async def rpc(fixture, method, **params):
    return await fixture.connection.dispatch(
        {"id": 20, "method": method, "params": params}
    )


@pytest.mark.asyncio
async def test_registered_classic_admission_shares_and_reads_exact_bytes(
    classic_runtime, monkeypatch
):
    from gateway.session_contract import SessionRef
    from gateway.session_results import execution_result
    from model_tools import get_tool_definitions
    from tools import hosted_room_artifact  # noqa: F401 -- register the real tool
    from tools.registry import registry

    fixture = classic_runtime
    from gateway.session_classic_output import classic_turn_toolsets
    assert classic_turn_toolsets([]) == []
    assert classic_turn_toolsets(None) is None
    assert not any(
        tool["function"]["name"] == "share_group_file"
        for tool in get_tool_definitions(enabled_toolsets=[], quiet_mode=True)
    )
    capability = await rpc(fixture, "gateway.capabilities")
    assert capability["result"]["classic_output_export_v1"] is True
    installation = capability["result"]["installation"]

    output = fixture.home / "report.bin"
    expected = b"classic current exact bytes\n\x00\xff"
    output.write_bytes(expected)
    request = {
        "request_id": "classic-request-1",
        "group_id": "classic-room",
        "thread_id": "classic-thread",
        "recipients": [{"installation": installation, "profile": "default"}],
        "issued_at": time.time(),
    }
    executions = []

    async def handle(event):
        from gateway.session_classic_output import (
            classic_turn_toolsets,
            current_classic_output_binding,
        )

        binding = current_classic_output_binding()
        assert binding is not None
        assert binding.ref.session_id == fixture.session_id
        assert binding.group_id == request["group_id"]
        assert any(
            tool["function"]["name"] == "share_group_file"
            for tool in get_tool_definitions(
                enabled_toolsets=classic_turn_toolsets([]), quiet_mode=True
            )
        )
        first = json.loads(
            await asyncio.to_thread(
                registry.dispatch,
                "share_group_file",
                {"path": str(output)},
                task_id="default",
            )
        )
        second = json.loads(
            await asyncio.to_thread(
                registry.dispatch,
                "share_group_file",
                {"path": str(output)},
                task_id="default",
            )
        )
        assert first["ok"] is True
        assert second["artifact_id"] == first["artifact_id"]
        executions.append((event.message_id, first))
        execution_result.get().update(
            result={
                "final_response": "Shared report.bin",
                "messages": [],
                "completed": True,
            },
            usage={},
        )
        return "Shared report.bin"

    fixture.runner._handle_message = handle
    submitted = await rpc(
        fixture,
        "prompt.submit",
        session_id=fixture.session_id,
        text="Create and share the report",
        classic_export=request,
    )
    assert "result" in submitted, submitted
    admitted = submitted["result"]
    assert admitted["classic_export"]["state"] == "running"
    assert admitted["classic_export"]["group_id"] == request["group_id"]

    ref = SessionRef(str(fixture.home), fixture.session_id)
    await fixture.authority._drain(ref)
    terminal = await rpc(
        fixture,
        "prompt.receipt",
        session_id=fixture.session_id,
        admission_id=admitted["admission_id"],
    )
    status = terminal["result"]["classic_export"]
    assert terminal["result"]["status"] == "terminal"
    assert status["state"] == "published"
    assert status["text"] == "Shared report.bin"
    assert status["recipients"] == request["recipients"]
    item, = status["items"]

    polled = await rpc(
        fixture,
        "session.export.read",
        session_id=fixture.session_id,
        installation=installation,
        group_id=request["group_id"],
        request_id=request["request_id"],
    )
    assert polled["result"] == status

    downloaded = await rpc(
        fixture,
        "session.export.read",
        session_id=fixture.session_id,
        installation=installation,
        group_id=request["group_id"],
        export_id=status["export_id"],
        artifact_id=item["artifact_id"],
        generation=status["generation"],
    )
    assert "result" in downloaded, downloaded
    assert downloaded["result"]["item"] == item
    assert base64.b64decode(downloaded["result"]["content_base64"], validate=True) == expected
    missing_generation = await rpc(
        fixture,
        "session.export.read",
        session_id=fixture.session_id,
        installation=installation,
        group_id=request["group_id"],
        export_id=status["export_id"],
        artifact_id=item["artifact_id"],
    )
    assert missing_generation["error"]["message"] == "invalid_params"

    original_actor = fixture.connection.actor
    fixture.connection.actor = replace(original_actor, subject="foreign-owner")
    try:
        denied = await rpc(
            fixture,
            "session.export.read",
            session_id=fixture.session_id,
            installation=installation,
            group_id=request["group_id"],
            export_id=status["export_id"],
            artifact_id=item["artifact_id"],
            generation=status["generation"],
        )
    finally:
        fixture.connection.actor = original_actor
    assert denied["error"]["message"] == "permission_denied"

    replay = await rpc(
        fixture,
        "prompt.submit",
        session_id=fixture.session_id,
        text="Create and share the report",
        classic_export=request,
    )
    assert replay["result"]["admission_id"] == admitted["admission_id"]
    assert replay["result"]["classic_export"] == status
    await fixture.authority._drain(ref)
    assert len(executions) == 1
    from gateway.hosted_room_artifacts_classic import ClassicExportScope
    scope = ClassicExportScope(status["export_id"], status["generation"])
    with fixture.db._read_ctx() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM hosted_room_output_artifacts WHERE scope_key=?",
            (scope.key,),
        ).fetchone()[0] == 1

    assert json.loads(
        registry.dispatch("share_group_file", {"path": str(output)})
    )["ok"] is False
    retired = await rpc(
        fixture,
        "session.export.discard",
        session_id=fixture.session_id,
        installation=installation,
        group_id=request["group_id"],
        export_id=status["export_id"],
    )
    assert retired["result"] == {"retired": True}
    unavailable = await rpc(
        fixture,
        "session.export.read",
        session_id=fixture.session_id,
        installation=installation,
        group_id=request["group_id"],
        export_id=status["export_id"],
        artifact_id=item["artifact_id"],
        generation=status["generation"],
    )
    assert unavailable["error"]["message"] == "classic_export_unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["failure", "stop"])
async def test_failure_and_stop_retire_shared_bytes(classic_runtime, ending):
    from gateway.session_contract import SessionRef
    from gateway.session_results import execution_result
    from tools import hosted_room_artifact  # noqa: F401 -- register the real tool
    from tools.registry import registry

    fixture = classic_runtime
    capability = (await rpc(fixture, "gateway.capabilities"))["result"]
    installation = capability["installation"]
    output = fixture.home / f"{ending}.txt"
    output.write_text(f"private {ending} bytes", encoding="utf-8")
    request = {
        "request_id": f"classic-{ending}",
        "group_id": "classic-room",
        "thread_id": "classic-thread",
        "recipients": [{"installation": installation, "profile": "default"}],
        "issued_at": time.time(),
    }
    shared = asyncio.Event()
    release = asyncio.Event()
    tool_results = []

    async def handle(_event):
        result = json.loads(
            await asyncio.to_thread(
                registry.dispatch,
                "share_group_file",
                {"path": str(output)},
                task_id="default",
            )
        )
        assert result["ok"] is True
        tool_results.append(result)
        shared.set()
        if ending == "failure":
            raise RuntimeError("inert provider failure")
        await release.wait()
        execution_result.get().update(
            result={
                "final_response": "",
                "messages": [],
                "interrupted": True,
                "completed": False,
            },
            usage={},
        )
        return ""

    fixture.runner._handle_message = handle
    if ending == "stop":
        fixture.runner._cached_agent_for = lambda _route: SimpleNamespace(
            interrupt=release.set
        )
    submitted = await rpc(
        fixture,
        "prompt.submit",
        session_id=fixture.session_id,
        text=f"Share then {ending}",
        classic_export=request,
    )
    assert "result" in submitted, submitted
    ref = SessionRef(str(fixture.home), fixture.session_id)
    drain = asyncio.create_task(fixture.authority._drain(ref))
    await asyncio.wait_for(shared.wait(), timeout=5)
    if ending == "stop":
        generation = fixture.authority._handle(ref).execution_generation
        stopped = await rpc(
            fixture,
            "session.interrupt",
            session_id=fixture.session_id,
            execution_generation=generation,
        )
        assert stopped["result"]["execution_state"] == "running"
    await asyncio.wait_for(drain, timeout=5)

    terminal = await rpc(
        fixture,
        "prompt.receipt",
        session_id=fixture.session_id,
        admission_id=submitted["result"]["admission_id"],
    )
    classic = terminal["result"]["classic_export"]
    assert classic["state"] == "retired"
    assert classic["items"] == []
    assert terminal["result"]["outcome"] == (
        "interrupted" if ending == "stop" else "failed"
    )
    item = tool_results[0]
    denied = await rpc(
        fixture,
        "session.export.read",
        session_id=fixture.session_id,
        installation=installation,
        group_id=request["group_id"],
        export_id=classic["export_id"],
        artifact_id=item["artifact_id"],
        generation=classic["generation"],
    )
    assert denied["error"]["message"] == "classic_export_unavailable"
    with fixture.db._read_ctx() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM hosted_room_output_artifacts WHERE artifact_id=?",
            (item["artifact_id"],),
        ).fetchone()[0] == 0


@pytest.mark.asyncio
async def test_forged_scope_and_named_owner_never_gain_classic_sharing(classic_runtime):
    from gateway.classic_output_exports import ClassicExports
    from gateway.session_contract import SessionRef, Submission
    from gateway.session_results import execution_result
    from tools import hosted_room_artifact  # noqa: F401 -- register the real tool
    from tools.registry import registry

    fixture = classic_runtime
    ClassicExports(fixture.home)
    output = fixture.home / "forged.txt"
    output.write_text("must remain private", encoding="utf-8")
    observed = []

    async def handle(_event):
        observed.append(
            json.loads(
                await asyncio.to_thread(
                    registry.dispatch,
                    "share_group_file",
                    {"path": str(output)},
                    task_id="default",
                )
            )
        )
        execution_result.get().update(
            result={"final_response": "No share", "messages": [], "completed": True},
            usage={},
        )
        return "No share"

    fixture.runner._handle_message = handle
    ref = SessionRef(str(fixture.home), fixture.session_id)
    receipt = await fixture.authority.submit(
        fixture.connection.actor,
        Submission(
            "forged-classic",
            ref,
            {
                "text": "Try forged sharing",
                "classic_export_v1": {
                    "export_id": "ce_" + "0" * 64,
                    "generation": 1,
                    "group_id": "classic-room",
                    "principal_id": fixture.connection.actor.subject,
                    "binding_version": "canonical_v1",
                },
            },
            "queue",
        ),
    )
    await fixture.authority._drain(ref)
    assert observed == [
        {
            "ok": False,
            "error": "File sharing is available only during a Group Chat turn.",
        }
    ]
    assert (await fixture.authority.receipt(
        fixture.connection.actor, ref, receipt.admission_id
    )).outcome == "completed"
    with fixture.db._read_ctx() as conn:
        assert conn.execute("SELECT COUNT(*) FROM classic_output_exports").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM hosted_room_output_artifacts").fetchone()[0] == 0

    original_profile = fixture.authority.profile_id
    fixture.authority.profile_id = str(fixture.home / "profiles" / "named")
    try:
        unsupported = await rpc(fixture, "gateway.capabilities")
    finally:
        fixture.authority.profile_id = original_profile
    assert unsupported["result"] == {"classic_output_export_v1": False}


@pytest.mark.asyncio
async def test_retired_classic_cleanup_replays_after_physical_failure(
    classic_runtime, monkeypatch
):
    from gateway.classic_output_exports import ClassicExports
    from gateway.hosted_room_artifacts import RoomArtifactError

    fixture = classic_runtime
    installation = (await rpc(fixture, "gateway.capabilities"))["result"][
        "installation"
    ]
    store = ClassicExports(fixture.home)
    request = {
        "request_id": "cleanup-replay",
        "group_id": "classic-room",
        "thread_id": "classic-thread",
        "recipients": [{"installation": installation, "profile": "default"}],
        "issued_at": time.time(),
    }
    row, fresh = store.admit(fixture.session_id, request, "share")
    assert fresh
    source = fixture.home / "cleanup.txt"
    source.write_text("cleanup replay bytes", encoding="utf-8")
    item = store.outbox.put_path(scope=store.scope(row), path=source)
    with store.outbox._connect() as conn:
        blob_name = conn.execute(
            "SELECT blob_name FROM hosted_room_output_artifacts WHERE artifact_id=?",
            (item["artifact_id"],),
        ).fetchone()[0]
    blob = store.outbox.blob_root / blob_name

    def fail_cleanup(_scope):
        raise OSError("injected physical cleanup failure")

    monkeypatch.setattr(store.outbox, "discard", fail_cleanup)
    with pytest.raises(OSError, match="physical cleanup"):
        store.retire(row["export_id"])
    assert blob.exists()
    assert store.lookup(row["export_id"])["state"] == "retired"
    with pytest.raises(RoomArtifactError, match="not published"):
        store.read(row["export_id"], item["artifact_id"])

    reopened = ClassicExports(fixture.home)
    assert not blob.exists()
    with reopened.outbox._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM hosted_room_output_artifacts WHERE artifact_id=?",
            (item["artifact_id"],),
        ).fetchone()[0] == 0


async def _classic_request(fixture, request_id, *, group_id="classic-room"):
    installation = (await rpc(fixture, "gateway.capabilities"))["result"]["installation"]
    return installation, {
        "request_id": request_id,
        "group_id": group_id,
        "thread_id": "classic-thread",
        "recipients": [{"installation": installation, "profile": "default"}],
        "issued_at": time.time(),
    }


async def _publish_classic_file(fixture, request_id, *, group_id="classic-room"):
    from gateway.session_contract import SessionRef
    from gateway.session_results import execution_result
    from tools import hosted_room_artifact  # noqa: F401 -- register the real tool
    from tools.registry import registry

    installation, request = await _classic_request(
        fixture, request_id, group_id=group_id
    )
    output = fixture.home / f"{request_id}.txt"
    output.write_text(f"private bytes for {request_id}", encoding="utf-8")
    shared = []

    async def handle(_event):
        item = json.loads(
            await asyncio.to_thread(
                registry.dispatch,
                "share_group_file",
                {"path": str(output)},
                task_id="default",
            )
        )
        assert item["ok"] is True
        shared.append(item)
        execution_result.get().update(
            result={
                "final_response": f"Shared {output.name}",
                "messages": [],
                "completed": True,
            },
            usage={},
        )
        return f"Shared {output.name}"

    fixture.runner._handle_message = handle
    submitted = await rpc(
        fixture,
        "prompt.submit",
        session_id=fixture.session_id,
        text=f"Share {output.name}",
        classic_export=request,
    )
    assert "result" in submitted, submitted
    await fixture.authority._drain(
        SessionRef(str(fixture.home), fixture.session_id)
    )
    terminal = await rpc(
        fixture,
        "prompt.receipt",
        session_id=fixture.session_id,
        admission_id=submitted["result"]["admission_id"],
    )
    assert terminal["result"]["status"] == "terminal"
    assert terminal["result"]["classic_export"]["state"] == "published"
    return installation, request, submitted["result"], terminal["result"], shared[0]


@pytest.mark.asyncio
async def test_generation_fence_loss_never_publishes_classic_bytes(
    classic_runtime,
):
    from gateway.session_contract import SessionRef
    from gateway.session_results import execution_result
    from tools import hosted_room_artifact  # noqa: F401 -- register the real tool
    from tools.registry import registry

    fixture = classic_runtime
    installation, request = await _classic_request(fixture, "generation-fence-loss")
    output = fixture.home / "generation-fence-loss.txt"
    expected = b"must remain unreadable after settlement fence loss"
    output.write_bytes(expected)
    shared = []

    async def handle(_event):
        item = json.loads(
            await asyncio.to_thread(
                registry.dispatch,
                "share_group_file",
                {"path": str(output)},
                task_id="default",
            )
        )
        assert item["ok"] is True
        shared.append(item)
        execution_result.get().update(
            result={
                "final_response": "unsettled response",
                "messages": [],
                "completed": True,
            },
            usage={},
        )

        def advance_generation(conn):
            conn.execute(
                "UPDATE sessions SET runtime_generation=runtime_generation+1 WHERE id=?",
                (fixture.session_id,),
            )

        fixture.db._execute_write(advance_generation)
        return "unsettled response"

    fixture.runner._handle_message = handle
    submitted = await rpc(
        fixture,
        "prompt.submit",
        session_id=fixture.session_id,
        text="Share, then lose the settlement fence",
        classic_export=request,
    )
    await fixture.authority._drain(
        SessionRef(str(fixture.home), fixture.session_id)
    )
    receipt = await rpc(
        fixture,
        "prompt.receipt",
        session_id=fixture.session_id,
        admission_id=submitted["result"]["admission_id"],
    )
    assert receipt["result"]["status"] == "started"
    assert receipt["result"]["classic_export"]["state"] == "running"
    item = shared[0]
    exact = await rpc(
        fixture,
        "session.export.read",
        session_id=fixture.session_id,
        installation=installation,
        group_id=request["group_id"],
        export_id=receipt["result"]["classic_export"]["export_id"],
        artifact_id=item["artifact_id"],
        generation=receipt["result"]["classic_export"]["generation"],
    )
    assert exact["error"]["message"] == "classic_export_unavailable"


@pytest.mark.asyncio
async def test_queued_cancel_retires_classic_export_and_replays_exact_status(
    classic_runtime,
):
    from gateway.session_contract import SessionRef

    fixture = classic_runtime
    _installation, request = await _classic_request(fixture, "queued-cancel")
    held = asyncio.Event()
    release = asyncio.Event()
    executed = []

    async def handle(event):
        executed.append(event.text)
        if event.text == "held head":
            held.set()
            await release.wait()
        return "done"

    fixture.runner._handle_message = handle
    head = await rpc(
        fixture,
        "prompt.submit",
        session_id=fixture.session_id,
        input_id="held-head",
        text="held head",
    )
    queued = await rpc(
        fixture,
        "prompt.submit",
        session_id=fixture.session_id,
        text="must never execute",
        classic_export=request,
    )
    ref = SessionRef(str(fixture.home), fixture.session_id)
    drain = asyncio.create_task(fixture.authority._drain(ref))
    await asyncio.wait_for(held.wait(), timeout=5)
    cancelled = await rpc(
        fixture,
        "prompt.cancel",
        session_id=fixture.session_id,
        admission_id=queued["result"]["admission_id"],
    )
    assert cancelled["result"]["status"] == "terminal"
    assert cancelled["result"]["outcome"] == "cancelled"
    replay = await rpc(
        fixture,
        "prompt.cancel",
        session_id=fixture.session_id,
        admission_id=queued["result"]["admission_id"],
    )
    assert replay["result"] == cancelled["result"]
    receipt = await rpc(
        fixture,
        "prompt.receipt",
        session_id=fixture.session_id,
        admission_id=queued["result"]["admission_id"],
    )
    assert receipt["result"]["status"] == "terminal"
    assert receipt["result"]["outcome"] == "cancelled"
    assert receipt["result"]["classic_export"]["state"] == "retired"
    release.set()
    await asyncio.wait_for(drain, timeout=5)
    assert executed == ["held head"]
    assert head["result"]["status"] == "queued"


@pytest.mark.asyncio
async def test_unknown_resolution_retires_and_replays_failed_physical_cleanup(
    classic_runtime, monkeypatch
):
    from gateway.classic_output_exports import ClassicExports
    from gateway.hosted_room_artifacts import RoomArtifactOutbox
    from gateway.session_classic_output import classic_output_scope
    from gateway.session_contract import SessionRef
    from gateway.session_finite import finite_turn_scope
    from hermes_state_runtime import (
        begin_runtime_epoch,
        claim_session_input,
        get_session_admission,
        recover_session_inputs,
    )
    from tools import hosted_room_artifact  # noqa: F401 -- register the real tool
    from tools.registry import registry

    fixture = classic_runtime
    installation, request = await _classic_request(fixture, "unknown-cleanup")
    source = fixture.home / "unknown-cleanup.txt"
    source.write_text("unknown private bytes", encoding="utf-8")
    submitted = await rpc(
        fixture,
        "prompt.submit",
        session_id=fixture.session_id,
        text="share before owner loss",
        classic_export=request,
    )
    ref = SessionRef(str(fixture.home), fixture.session_id)
    claimed = claim_session_input(
        fixture.db, epoch=fixture.authority.epoch, session_id=fixture.session_id
    )
    assert claimed["admission_id"] == submitted["result"]["admission_id"]
    with finite_turn_scope(False), classic_output_scope(
        fixture.authority, ref, claimed
    ) as binding:
        assert binding is not None
        item = json.loads(
            await asyncio.to_thread(
                registry.dispatch,
                "share_group_file",
                {"path": str(source)},
                task_id="default",
            )
        )
    assert item["ok"] is True
    with fixture.db._read_ctx() as conn:
        blob_name = conn.execute(
            "SELECT blob_name FROM hosted_room_output_artifacts WHERE artifact_id=?",
            (item["artifact_id"],),
        ).fetchone()[0]
    blob = fixture.home / "hosted-room-artifact-outbox" / "blobs" / blob_name
    assert blob.exists()

    new_epoch = begin_runtime_epoch(fixture.db, instance_id="classic-restarted-owner")
    assert recover_session_inputs(fixture.db, epoch=new_epoch) == 1
    fixture.authority.epoch = new_epoch
    unknown = get_session_admission(
        fixture.db, admission_id=claimed["admission_id"]
    )
    assert unknown["status"] == "unknown"
    before = await rpc(
        fixture,
        "session.export.read",
        session_id=fixture.session_id,
        installation=installation,
        group_id=request["group_id"],
        export_id=submitted["result"]["classic_export"]["export_id"],
        artifact_id=item["artifact_id"],
        generation=submitted["result"]["classic_export"]["generation"],
    )
    assert before["error"]["message"] == "classic_export_unavailable"

    original_discard = RoomArtifactOutbox.discard
    from gateway import session_classic_output
    original_unlink = session_classic_output.unlink_blob_names

    def fail_cleanup(_self, _scope):
        raise OSError("injected physical cleanup failure")

    monkeypatch.setattr(RoomArtifactOutbox, "discard", fail_cleanup)
    monkeypatch.setattr(session_classic_output, "unlink_blob_names", fail_cleanup)
    resolved = await rpc(
        fixture,
        "prompt.resolve_unknown",
        session_id=fixture.session_id,
        admission_id=claimed["admission_id"],
        execution_generation=claimed["generation"],
    )
    assert resolved["result"]["status"] == "terminal"
    assert resolved["result"]["outcome"] == "interrupted"
    status = await rpc(
        fixture,
        "prompt.receipt",
        session_id=fixture.session_id,
        admission_id=claimed["admission_id"],
    )
    assert status["result"]["classic_export"]["state"] == "retired"
    assert blob.exists()
    with fixture.db._read_ctx() as conn:
        cleanup_required = conn.execute(
            "SELECT cleanup_required_at FROM hosted_room_output_artifacts WHERE artifact_id=?",
            (item["artifact_id"],),
        ).fetchone()[0]
    assert cleanup_required is not None

    monkeypatch.setattr(RoomArtifactOutbox, "discard", original_discard)
    monkeypatch.setattr(session_classic_output, "unlink_blob_names", original_unlink)
    ClassicExports(fixture.home)
    assert not blob.exists()
    with fixture.db._read_ctx() as conn:
        assert conn.execute(
            "SELECT 1 FROM hosted_room_output_artifacts WHERE artifact_id=?",
            (item["artifact_id"],),
        ).fetchone() is None


@pytest.mark.asyncio
async def test_group_discard_requires_live_bound_producer_session(
    classic_runtime,
):
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.session_authority import LiveSession
    from gateway.session_controls import AuthorityConnection

    fixture = classic_runtime
    installation, request, _submitted, terminal, item = await _publish_classic_file(
        fixture, "group-owner", group_id="owned-group"
    )
    classic = terminal["classic_export"]
    other_id = "unrelated-session"
    fixture.db.create_session(other_id, source="gui")
    other_source = SessionSource(
        platform=Platform.LOCAL,
        chat_id=other_id,
        user_id=fixture.connection.actor.subject,
        chat_type="dm",
    )
    fixture.authority.sessions[other_id] = LiveSession(other_source, "other-route")
    fixture.connection.subscriptions[other_id] = "other-subscription"
    params = {
        "installation": installation,
        "group_id": request["group_id"],
    }

    refused = await rpc(
        fixture,
        "session.export.discard",
        session_id=other_id,
        **params,
    )
    assert refused["error"]["message"] == "classic_export_unavailable"

    closed = AuthorityConnection(
        fixture.authority,
        object(),
        {
            "user_id": "owner",
            "profile_id": str(fixture.home),
            "instance_id": "classic-current-test",
            "capabilities": ["session:read", "session:control"],
        },
    )
    await closed.close()
    closed_reply = await closed.dispatch(
        {
            "id": 21,
            "method": "session.export.discard",
            "params": {"session_id": fixture.session_id, **params},
        }
    )
    assert closed_reply["error"]["message"] == "classic_export_unavailable"

    with fixture.db._read_ctx() as conn:
        assert conn.execute(
            "SELECT state FROM classic_output_exports WHERE export_id=?",
            (classic["export_id"],),
        ).fetchone()[0] == "published"
        assert conn.execute(
            "SELECT 1 FROM classic_retired_groups WHERE group_id=?",
            (request["group_id"],),
        ).fetchone() is None
        assert conn.execute(
            "SELECT 1 FROM hosted_room_output_artifacts WHERE artifact_id=?",
            (item["artifact_id"],),
        ).fetchone() is not None

    retired = await rpc(
        fixture,
        "session.export.discard",
        session_id=fixture.session_id,
        **params,
    )
    assert retired["result"] == {"retired": True}
    after = await rpc(
        fixture,
        "prompt.receipt",
        session_id=fixture.session_id,
        admission_id=terminal["admission_id"],
    )
    assert after["result"]["classic_export"]["state"] == "retired"


@pytest.mark.asyncio
async def test_precommit_admission_failure_removes_preparation_for_exact_retry(
    classic_runtime,
):
    fixture = classic_runtime
    _installation, request = await _classic_request(fixture, "precommit-retry")

    def install_failure(conn):
        conn.execute(
            """CREATE TRIGGER fail_classic_admission BEFORE INSERT ON session_admissions
               BEGIN SELECT RAISE(ABORT, 'injected classic precommit failure'); END"""
        )

    fixture.db._execute_write(install_failure)
    failed = await rpc(
        fixture,
        "prompt.submit",
        session_id=fixture.session_id,
        text="retry this exact classic input",
        classic_export=request,
    )
    assert failed["error"]["message"] == "storage_unavailable"

    def remove_failure(conn):
        conn.execute("DROP TRIGGER fail_classic_admission")

    fixture.db._execute_write(remove_failure)
    with fixture.db._read_ctx() as conn:
        assert conn.execute("SELECT COUNT(*) FROM session_admissions").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM classic_output_exports").fetchone()[0] == 0

    retried = await rpc(
        fixture,
        "prompt.submit",
        session_id=fixture.session_id,
        text="retry this exact classic input",
        classic_export=request,
    )
    assert retried["result"]["status"] == "queued"
    assert retried["result"]["classic_export"]["state"] == "running"
    with fixture.db._read_ctx() as conn:
        assert conn.execute("SELECT COUNT(*) FROM session_admissions").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM classic_output_exports").fetchone()[0] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_point", ["_publish_pending", "_schedule"])
async def test_postcommit_projection_failure_preserves_exact_classic_replay(
    classic_runtime, monkeypatch, failure_point
):
    fixture = classic_runtime
    _installation, request = await _classic_request(
        fixture, f"postcommit-{failure_point.removeprefix('_')}"
    )
    original = getattr(fixture.authority, failure_point)
    failed_once = False

    def fail_once(ref):
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise RuntimeError(f"injected {failure_point} failure")
        return original(ref)

    monkeypatch.setattr(fixture.authority, failure_point, fail_once)
    with pytest.raises(RuntimeError, match="injected"):
        await rpc(
            fixture,
            "prompt.submit",
            session_id=fixture.session_id,
            text="recover committed classic admission",
            classic_export=request,
        )
    with fixture.db._read_ctx() as conn:
        admission = dict(conn.execute("SELECT * FROM session_admissions").fetchone())
        export = dict(conn.execute("SELECT * FROM classic_output_exports").fetchone())
    assert admission["status"] == "queued"
    assert export["state"] == "running"

    retried = await rpc(
        fixture,
        "prompt.submit",
        session_id=fixture.session_id,
        text="recover committed classic admission",
        classic_export=request,
    )
    assert retried["result"]["admission_id"] == admission["admission_id"]
    assert retried["result"]["classic_export"]["export_id"] == export["export_id"]
    assert retried["result"]["classic_export"]["state"] == "running"
    with fixture.db._read_ctx() as conn:
        assert conn.execute("SELECT COUNT(*) FROM session_admissions").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM classic_output_exports").fetchone()[0] == 1


@pytest.mark.asyncio
async def test_operator_principal_cannot_acquire_existing_classic_request_scope(
    classic_runtime,
):
    from gateway.session_controls import AuthorityConnection

    fixture = classic_runtime
    _installation, request = await _classic_request(fixture, "principal-bound")
    owner = await rpc(
        fixture,
        "prompt.submit",
        session_id=fixture.session_id,
        text="principal-owned classic input",
        classic_export=request,
    )
    operator = AuthorityConnection(
        fixture.authority,
        object(),
        {
            "user_id": "verified-operator",
            "profile_id": str(fixture.home),
            "instance_id": "classic-current-test",
            "capabilities": ["session:read", "session:submit", "session:control"],
        },
        operator=True,
    )
    operator.subscriptions[fixture.session_id] = "operator-subscription"
    try:
        denied = await operator.dispatch(
            {
                "id": 22,
                "method": "prompt.submit",
                "params": {
                    "session_id": fixture.session_id,
                    "text": "principal-owned classic input",
                    "classic_export": request,
                },
            }
        )
        assert denied["error"]["message"] == "classic_export_unavailable"
        with fixture.db._read_ctx() as conn:
            admissions = conn.execute(
                "SELECT principal_id FROM session_admissions ORDER BY seq"
            ).fetchall()
            binding = json.loads(
                conn.execute("SELECT binding FROM classic_output_exports").fetchone()[0]
            )
        assert [row[0] for row in admissions] == [fixture.connection.actor.subject]
        assert binding["principal_id"] == fixture.connection.actor.subject
        assert binding["binding_version"] == "canonical_v1"
        assert owner["result"]["classic_export"]["state"] == "running"
    finally:
        await operator.close()
