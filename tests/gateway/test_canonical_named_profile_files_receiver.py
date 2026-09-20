"""Served named-profile RoomLink Files through real owner/API/custody assembly."""
from __future__ import annotations

import asyncio
import io
import json
import threading
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlsplit

import pytest
from aiohttp.test_utils import make_mocked_request
from unittest.mock import AsyncMock

from tests.gateway.test_canonical_peer_files_target import source_files
from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner


ROOT_KEY = "root-profile-api-key-1234567890"
PROFILE_KEYS = {
    "alpha": "alpha-profile-api-key-1234567890",
    "beta": "beta-profile-api-key-1234567890",
}


@pytest.fixture
def named_files_runtime(tmp_path, monkeypatch):
    """Real multiplex owner registry and startup custody; no listener or model."""
    from agent import secret_scope
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.run import _profile_runtime_scope
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from hermes_cli import runtime_provider as runtime_provider
    import run_agent
    import socket
    import subprocess

    root, entries = _reserve_homes(tmp_path, monkeypatch)
    homes = dict(entries)
    config = "agent:\n  max_turns: 7\napprovals:\n  mode: manual\nmodel:\n  provider: anthropic\n  default: fixture-model\n"
    (root / "config.yaml").write_text(config)
    (root / ".env").write_text(f"API_SERVER_KEY={ROOT_KEY}\n")
    for name in ("alpha", "beta"):
        (homes[name] / "config.yaml").write_text(config)
        (homes[name] / ".env").write_text(f"API_SERVER_KEY={PROFILE_KEYS[name]}\n")

    secret_scope.set_multiplex_active(True)
    process_ownership.reserve(homes.values())
    runner = _runner(root, entries)
    with _profile_runtime_scope(root, hydrate_secrets=False):
        asyncio.run(initialize_gateway_runtime(runner))
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": ROOT_KEY}))
    adapter.gateway_runner = runner
    runner.adapters[Platform.API_SERVER] = adapter

    monkeypatch.setattr(runtime_provider, "resolve_provider", lambda requested, **kwargs: requested)

    def pool(provider):
        entry = SimpleNamespace(
            runtime_api_key="fixture-only-key",
            base_url="https://api.anthropic.com",
            source="fixture",
        )
        return SimpleNamespace(
            provider=provider,
            has_credentials=lambda: True,
            select=lambda **kwargs: entry,
            current=lambda: entry,
        )

    monkeypatch.setattr(runtime_provider, "load_pool", pool)

    def forbidden(*args, **kwargs):
        pytest.fail("named Files receiver crossed the inert execution boundary")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(run_agent.AIAgent, "__init__", forbidden)
    launched = []

    async def inert(adapter, launch, **kwargs):
        launched.append(launch.admission)

    from gateway.platforms import api_server_runs

    original_execute_run = api_server_runs._execute_run
    monkeypatch.setattr(api_server_runs, "_execute_run", inert)
    case = SimpleNamespace(
        root=root,
        homes=homes,
        runner=runner,
        adapter=adapter,
        authorities={name: runner.session_authorities.require(home) for name, home in entries},
        launched=launched,
        original_execute_run=original_execute_run,
    )
    try:
        yield case
    finally:
        from gateway.platforms import api_server_runs

        api_server_runs._close_run_state(adapter)
        for authority in {id(value): value for value in case.authorities.values()}.values():
            authority.db.close()
        for home in homes.values():
            process_ownership.release(home)
        secret_scope.set_multiplex_active(False)


def _route_table(adapter):
    native = adapter._http_route_table()
    return native + [
        (verb, "/p/{profile}" + path, handler)
        for verb, path, handler in native
    ]


def install_inprocess_profile_http(case, monkeypatch, *, responses=None):
    """Replace only the socket while running the real profile middleware."""
    from tui_gateway import hosted_room_peer_http

    loop = asyncio.get_running_loop()
    middleware = case.adapter._make_profile_prefix_middleware()
    table = _route_table(case.adapter)
    calls = []

    async def send(wire):
        path, method = urlsplit(wire.full_url).path, wire.get_method()
        calls.append((method, path))
        match, handler = {}, None
        for verb, template, candidate in table:
            if verb != method:
                continue
            actual, wanted = path.split("/"), template.split("/")
            if len(actual) != len(wanted):
                continue
            if all(left == right or right.startswith("{") for left, right in zip(actual, wanted)):
                match = {
                    right[1:-1]: unquote(left)
                    for left, right in zip(actual, wanted)
                    if right.startswith("{")
                }
                handler = candidate
                break
        assert handler is not None, (method, path)
        data = wire.data or b""
        if not isinstance(data, bytes):
            data = b"".join(data)

        class Content:
            async def iter_chunked(self, size):
                for offset in range(0, len(data), size):
                    yield data[offset : offset + size]

        request = make_mocked_request(
            method,
            path,
            headers=dict(wire.header_items()),
            match_info=match,
            payload=Content(),
        )
        request.json = AsyncMock(
            return_value=json.loads(data) if method == "POST" and data else {}
        )

        async def dispatch(scoped_request):
            return await handler(scoped_request)

        response = await middleware(request, dispatch)
        body = json.loads(response.body)
        if responses is not None:
            responses.append((method, path, response.status, body))
        if response.status >= 400:
            raise urllib.error.HTTPError(
                wire.full_url,
                response.status,
                response.reason,
                response.headers,
                io.BytesIO(response.body),
            )
        return io.BytesIO(response.body)

    def open_wire(wire, **kwargs):
        return asyncio.run_coroutine_threadsafe(send(wire), loop).result(10)

    monkeypatch.setattr(hosted_room_peer_http, "_open_roomlink_url", open_wire)
    return calls


def _client(profile, key):
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient

    return PeerRunsHTTPClient(
        base_url="http://127.0.0.1:8642",
        api_key=key,
        target_profile=profile,
    )


def _invitation(client, suffix="one"):
    return client.issue_invitation(
        room_id="room-one",
        home_install_id="home-install",
        authority_gateway_id="home-gateway",
        authority_epoch=1,
        member_id="member-one",
        grant_id=f"grant-{suffix}",
        ttl_seconds=600,
        status_ttl_seconds=1200,
    )


def _files_dispatch(invitation, attachments, *, target_profile="beta"):
    import hashlib
    from gateway.hosted_room_peer import (
        PROTOCOL_VERSION,
        HostedMemberDispatch,
        attachment_manifest_digest,
    )

    manifest = [{key: value for key, value in item.items() if key != "data"} for item in attachments]
    prompt = "original named prompt"
    catalog = invitation["catalog"]
    return HostedMemberDispatch.from_mapping(
        {
            "protocol_version": PROTOCOL_VERSION,
            "room_id": "room-one",
            "home_install_id": "home-install",
            "authority_gateway_id": "home-gateway",
            "authority_epoch": 1,
            "member_id": "member-one",
            "target_install_id": catalog["installation_id"],
            "target_profile": target_profile,
            "task_id": "task-one",
            "execution_generation": 1,
            "source_event_seq": 1,
            "cancellation_scope_id": "cancel-one",
            "prompt": prompt,
            "prompt_digest": hashlib.sha256(prompt.encode()).hexdigest(),
            "capability_digest": catalog["catalog_digest"],
            "execution_policy_digest": catalog["execution_policy"]["policy_digest"],
            "attachment_manifest_digest": attachment_manifest_digest(manifest),
            "trace_id": "trace-one",
        }
    )


def _source_batch(case):
    from tui_gateway.hosted_room_peer_attachments import bound_attachment_payloads

    store, _event, attachments, raw = source_files(case.root)
    payloads = bound_attachment_payloads(store, "room-one", "member-one", attachments)
    assert [item["data"] for item in payloads] == raw
    return payloads, raw


@pytest.mark.asyncio
async def test_named_profile_stage_upload_new_admission_and_replay_are_owner_scoped(
    named_files_runtime, monkeypatch
):
    case = named_files_runtime
    calls = install_inprocess_profile_http(case, monkeypatch)
    client = _client("beta", PROFILE_KEYS["beta"])
    invitation = await asyncio.to_thread(_invitation, client)
    assert invitation["target_profile"] == "beta"
    assert invitation["catalog"]["attachments"] is True

    attachments, raw = _source_batch(case)
    dispatch = _files_dispatch(invitation, attachments)
    staged = await asyncio.to_thread(
        client.stage_attachments,
        dispatch=dispatch.as_mapping(),
        attachments=attachments,
        grant=invitation["grant"],
    )
    assert staged["complete"] is True
    accepted = await asyncio.to_thread(
        client.dispatch, dispatch=dispatch.as_mapping(), grant=invitation["grant"]
    )
    for _ in range(20):
        if case.launched:
            break
        await asyncio.sleep(0)
    discarded = await asyncio.to_thread(
        client.discard_attachments,
        task_id=dispatch.task_id,
        execution_generation=dispatch.execution_generation,
        grant=invitation["grant"],
    )
    assert discarded["removed"] == 1
    from gateway.platforms.api_server_room_attachments import _request_spool

    with _request_spool()._transaction() as connection:
        assert connection.execute(
            "SELECT count(*) FROM roomlink_attachment_batches"
        ).fetchone()[0] == 0
    replay = await asyncio.to_thread(
        client.dispatch, dispatch=dispatch.as_mapping(), grant=invitation["grant"]
    )
    assert accepted["replayed"] is False
    assert replay == dict(accepted, replayed=True)
    assert len(case.launched) == 1

    authority, ref, row = case.launched[0]
    assert authority is case.authorities["beta"]
    refs = row["payload"]["api_turn_v1"]["settings"]["room_input_media"]["media"]
    assert [Path(reference["path"]).read_bytes() for reference in refs] == raw
    assert all(Path(reference["path"]).is_relative_to(case.homes["beta"]) for reference in refs)
    assert authority.db._conn.execute("SELECT count(*) FROM session_admissions").fetchone()[0] == 1
    assert authority.db._conn.execute("SELECT count(*) FROM input_custody_refs").fetchone()[0] == 2
    assert case.authorities["default"].db._conn.execute(
        "SELECT count(*) FROM session_admissions"
    ).fetchone()[0] == 0
    assert case.authorities["alpha"].db._conn.execute(
        "SELECT count(*) FROM session_admissions"
    ).fetchone()[0] == 0

    beta_receipts = case.homes["beta"] / "runs_idempotency.db"
    assert beta_receipts.is_file()
    import sqlite3

    with sqlite3.connect(beta_receipts) as conn:
        assert conn.execute("SELECT count(*) FROM run_idempotency").fetchone()[0] == 1
    assert case.adapter._run_idempotency_store._conn.execute(
        "SELECT count(*) FROM run_idempotency"
    ).fetchone()[0] == 0
    assert calls.count(("POST", "/p/beta/v1/room-members/attachments")) == 1
    assert len([call for call in calls if call[0] == "PUT" and "/p/beta/" in call[1]]) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("denial", ["home", "cross-profile", "api-key", "owner-replacement"])
async def test_named_profile_receiver_refuses_cross_scope_and_changed_owner(
    named_files_runtime, monkeypatch, denial
):
    from hermes_state_runtime import RuntimeStoreError
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError

    case = named_files_runtime
    install_inprocess_profile_http(case, monkeypatch)
    beta = case.authorities["beta"]
    before = beta.db._conn.execute("SELECT count(*) FROM session_admissions").fetchone()[0]

    if denial == "home":
        from gateway.session_peer_target import root_target

        with pytest.raises(RuntimeStoreError, match="canonical_room_peer_unsupported"):
            root_target(case.adapter, "beta")
    elif denial == "api-key":
        client = _client("beta", ROOT_KEY)
        with pytest.raises(PeerRunsHTTPError) as error:
            await asyncio.to_thread(_invitation, client, "wrong-key")
        assert error.value.status_code == 401
    else:
        client = _client("beta", PROFILE_KEYS["beta"])
        invitation = await asyncio.to_thread(_invitation, client, denial)
        attachments, _raw = _source_batch(case)
        dispatch = _files_dispatch(invitation, attachments)
        if denial == "cross-profile":
            foreign = _client("alpha", PROFILE_KEYS["alpha"])
            with pytest.raises(PeerRunsHTTPError) as error:
                await asyncio.to_thread(
                    foreign.stage_attachments,
                    dispatch=dispatch.as_mapping(),
                    attachments=attachments,
                    grant=invitation["grant"],
                )
            assert error.value.status_code in {401, 403}
        else:
            registry = case.runner.session_authorities
            key = next(key for key, value in registry._by_key.items() if value is beta)
            registry._by_key[key] = case.authorities["alpha"]
            try:
                with pytest.raises(PeerRunsHTTPError) as error:
                    await asyncio.to_thread(
                        client.stage_attachments,
                        dispatch=dispatch.as_mapping(),
                        attachments=attachments,
                        grant=invitation["grant"],
                    )
                assert error.value.status_code in {401, 403}
            finally:
                registry._by_key[key] = beta

    assert beta.db._conn.execute("SELECT count(*) FROM session_admissions").fetchone()[0] == before
    assert case.authorities["default"].db._conn.execute(
        "SELECT count(*) FROM session_admissions"
    ).fetchone()[0] == 0
    assert case.authorities["alpha"].db._conn.execute(
        "SELECT count(*) FROM session_admissions"
    ).fetchone()[0] == 0


@pytest.mark.asyncio
async def test_root_control_and_named_profile_key_rotation_remain_isolated(
    named_files_runtime, monkeypatch
):
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError

    case = named_files_runtime
    install_inprocess_profile_http(case, monkeypatch)
    root_client = _client("default", ROOT_KEY)
    root_invitation = await asyncio.to_thread(_invitation, root_client, "root-control")
    assert root_invitation["target_profile"] == "default"
    assert root_invitation["catalog"]["attachments"] is True

    beta_env = case.homes["beta"] / ".env"
    replacement = "beta-replacement-api-key-1234567890"
    beta_env.write_text(f"API_SERVER_KEY={replacement}\n")
    stale = _client("beta", PROFILE_KEYS["beta"])
    with pytest.raises(PeerRunsHTTPError) as error:
        await asyncio.to_thread(_invitation, stale, "stale-secret")
    assert error.value.status_code == 401
    fresh = _client("beta", replacement)
    named_invitation = await asyncio.to_thread(_invitation, fresh, "fresh-secret")
    assert named_invitation["target_profile"] == "beta"
    assert named_invitation["catalog"]["attachments"] is True
    assert case.authorities["default"].db._conn.execute(
        "SELECT 1 FROM state_meta WHERE key LIKE 'gateway.peer.invite.%'"
    ).fetchone() is None


@pytest.mark.asyncio
async def test_named_receipt_handle_retires_with_its_exact_unserved_owner(
    named_files_runtime, monkeypatch
):
    import sqlite3
    from gateway.run import _profile_runtime_scope
    from gateway.run_runtime import unserve_profile_runtime
    from gateway.platforms.api_server_store import selected_run_idempotency_store

    case = named_files_runtime
    install_inprocess_profile_http(case, monkeypatch)
    client = _client("beta", PROFILE_KEYS["beta"])
    await asyncio.to_thread(_invitation, client, "retire-owner")
    with _profile_runtime_scope(case.homes["beta"], hydrate_secrets=False):
        store = selected_run_idempotency_store(case.adapter, case.homes["beta"])
    assert store is not None

    await unserve_profile_runtime(case.runner, case.homes["beta"])
    assert case.runner.session_authorities.for_home(case.homes["beta"]) is None
    with pytest.raises(sqlite3.ProgrammingError):
        store._conn.execute("SELECT 1")
    with _profile_runtime_scope(case.homes["beta"], hydrate_secrets=False):
        assert selected_run_idempotency_store(case.adapter, case.homes["beta"]) is None


@pytest.mark.asyncio
async def test_lazy_receipt_publication_revalidates_after_exact_owner_unserve(
    named_files_runtime, monkeypatch
):
    import sqlite3
    from gateway.run import _profile_runtime_scope
    from gateway.run_runtime import serve_profile_runtime, unserve_profile_runtime
    from gateway.platforms import api_server_run_idempotency
    from gateway.platforms.api_server_store import selected_run_idempotency_store
    from hermes_constants import hermes_home_key

    case = named_files_runtime
    old_owner = case.authorities["beta"]
    entered = threading.Event()
    proceed = threading.Event()

    class PublicationGate:
        """Pause the stale selector before it owns the real publication lock."""

        def __init__(self):
            self._lock = threading.Lock()
            self._state = threading.Lock()
            self._gated = False

        def __enter__(self):
            with self._state:
                gate = not self._gated
                self._gated = True
            if gate:
                entered.set()
                assert proceed.wait(5)
            self._lock.acquire()
            return self

        def __exit__(self, *_exc):
            self._lock.release()

    case.adapter._run_idempotency_store_lock = PublicationGate()
    real_store = api_server_run_idempotency.RunIdempotencyStore
    opened = []

    def recording_store(path):
        store = real_store(path)
        opened.append(store)
        return store

    monkeypatch.setattr(api_server_run_idempotency, "RunIdempotencyStore", recording_store)

    async def select():
        with _profile_runtime_scope(case.homes["beta"], hydrate_secrets=False):
            return await asyncio.to_thread(
                selected_run_idempotency_store, case.adapter, case.homes["beta"]
            )

    selection = asyncio.create_task(select())
    assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 5), timeout=6)
    await unserve_profile_runtime(case.runner, case.homes["beta"])
    proceed.set()
    stale = await asyncio.wait_for(selection, timeout=5)

    assert stale is None
    assert hermes_home_key(case.homes["beta"]) not in case.adapter._profile_run_idempotency_stores
    assert opened == []

    first_replacement = await serve_profile_runtime(case.runner, "beta", case.homes["beta"])
    case.authorities["beta"] = first_replacement
    assert first_replacement is not old_owner
    assert first_replacement.epoch > old_owner.epoch

    # A second interleaving withdraws the real owner while SQLite construction
    # holds the publication lock.  Unserve waits on that lock; a helper thread
    # releases the inert constructor only after registry removal is observable.
    case.adapter._run_idempotency_store_lock = threading.Lock()
    construction_entered = threading.Event()
    construction_proceed = threading.Event()
    owner_removed = threading.Event()

    def slow_recording_store(path):
        store = real_store(path)
        opened.append(store)
        construction_entered.set()
        assert construction_proceed.wait(5)
        return store

    registry = case.runner.session_authorities
    real_remove = registry.remove

    def recording_remove(home):
        owner = real_remove(home)
        owner_removed.set()
        return owner

    monkeypatch.setattr(api_server_run_idempotency, "RunIdempotencyStore", slow_recording_store)
    monkeypatch.setattr(registry, "remove", recording_remove)
    releaser = threading.Thread(
        target=lambda: (owner_removed.wait(5), construction_proceed.set()), daemon=True
    )
    releaser.start()
    selection = asyncio.create_task(select())
    assert await asyncio.wait_for(asyncio.to_thread(construction_entered.wait, 5), timeout=6)
    await unserve_profile_runtime(case.runner, case.homes["beta"])
    rejected = await asyncio.wait_for(selection, timeout=5)
    releaser.join(5)

    assert owner_removed.is_set()
    assert rejected is None
    assert hermes_home_key(case.homes["beta"]) not in case.adapter._profile_run_idempotency_stores
    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError):
        opened[0]._conn.execute("SELECT 1")

    replacement = await serve_profile_runtime(case.runner, "beta", case.homes["beta"])
    case.authorities["beta"] = replacement
    assert replacement is not first_replacement
    assert replacement.epoch > first_replacement.epoch
    assert replacement.instance_id == case.runner.session_runtime_descriptor["instance_id"]
    with _profile_runtime_scope(case.homes["beta"], hydrate_secrets=False):
        fresh = selected_run_idempotency_store(case.adapter, case.homes["beta"])
    assert fresh is not None and fresh is not opened[0]
    owner, epoch, instance_id, retained = case.adapter._profile_run_idempotency_stores[
        hermes_home_key(case.homes["beta"])
    ]
    assert (owner, epoch, instance_id, retained) == (
        replacement,
        replacement.epoch,
        replacement.instance_id,
        fresh,
    )


@pytest.mark.asyncio
async def test_unserve_retires_exact_named_run_before_store_close_and_replays(
    named_files_runtime, monkeypatch
):
    import sqlite3
    from gateway.platforms import api_server_runs
    from gateway.run_runtime import serve_profile_runtime, unserve_profile_runtime
    from gateway import session_finite

    case = named_files_runtime
    old_owner = case.authorities["beta"]
    monkeypatch.setattr(api_server_runs, "_execute_run", case.original_execute_run)
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    calls = 0

    async def blocked_execution(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(session_finite, "execute_finite_admission", blocked_execution)
    install_inprocess_profile_http(case, monkeypatch)
    client = _client("beta", PROFILE_KEYS["beta"])
    invitation = await asyncio.to_thread(_invitation, client, "owner-lifecycle")
    attachments, _raw = _source_batch(case)
    dispatch = _files_dispatch(invitation, attachments)
    await asyncio.to_thread(
        client.stage_attachments,
        dispatch=dispatch.as_mapping(),
        attachments=attachments,
        grant=invitation["grant"],
    )
    accepted = await asyncio.to_thread(
        client.dispatch, dispatch=dispatch.as_mapping(), grant=invitation["grant"]
    )
    run_id = accepted["run_id"]
    await asyncio.wait_for(entered.wait(), timeout=5)

    old_store = case.adapter._run_receipt_stores[run_id]
    receipt_path = case.homes["beta"] / "runs_idempotency.db"
    assert case.adapter._active_run_tasks[run_id].done() is False
    assert case.adapter.active_agent_work_count() == 1
    assert old_owner.api_observers
    assert old_owner.waiters
    assert case.adapter._run_idempotency_store._conn.execute(
        "SELECT count(*) FROM run_idempotency"
    ).fetchone()[0] == 0

    await unserve_profile_runtime(case.runner, case.homes["beta"])
    await asyncio.wait_for(cancelled.wait(), timeout=5)

    assert run_id not in case.adapter._active_run_tasks
    assert run_id not in case.adapter._run_authorities
    assert case.adapter.active_agent_work_count() == 0
    assert not old_owner.api_observers
    assert not old_owner.waiters
    assert run_id not in case.adapter._run_receipt_stores
    case.adapter._set_run_status(run_id, "failed", error="late retired-owner callback")
    assert case.adapter._run_idempotency_store._conn.execute(
        "SELECT count(*) FROM run_idempotency"
    ).fetchone()[0] == 0
    with pytest.raises(sqlite3.ProgrammingError):
        old_store._conn.execute("SELECT 1")
    with sqlite3.connect(receipt_path) as connection:
        status = json.loads(connection.execute(
            "SELECT status_json FROM run_idempotency WHERE run_id=?", (run_id,)
        ).fetchone()[0])
    assert status["status"] == "cancelled"

    replacement = await serve_profile_runtime(case.runner, "beta", case.homes["beta"])
    case.authorities["beta"] = replacement
    assert replacement is not old_owner
    assert replacement.epoch > old_owner.epoch
    replay = await asyncio.to_thread(
        client.dispatch, dispatch=dispatch.as_mapping(), grant=invitation["grant"]
    )
    assert replay == dict(accepted, replayed=True)
    assert calls == 1
    assert replacement.db._conn.execute(
        "SELECT count(*) FROM session_admissions WHERE principal_id='api'"
    ).fetchone()[0] == 1
    assert case.authorities["default"].db._conn.execute(
        "SELECT count(*) FROM session_admissions"
    ).fetchone()[0] == 0
    assert case.adapter._run_idempotency_store._conn.execute(
        "SELECT count(*) FROM run_idempotency"
    ).fetchone()[0] == 0
