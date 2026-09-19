"""Phase 8 Task 29/30: one-shot artifact transport + broker gates.

Exercises the transport-neutral store core
(:mod:`gateway.browser_control_artifacts`), the API-server routes
(``/v1/artifacts/upload`` + ``/v1/artifacts/download/{artifact_id}`` with
auth and per-principal rate limits), and the broker's Developer Mode gates
for ``browser_evaluate`` / raw CDP plus artifact referencing
("approved artifact id only").
"""

import hashlib
import os

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.browser_control_artifacts import (
    ArtifactChecksumMismatch,
    ArtifactError,
    ArtifactExpired,
    ArtifactMimeRejected,
    ArtifactNotFound,
    ArtifactRateLimiter,
    ArtifactScopeMismatch,
    ArtifactStore,
    ArtifactTooLarge,
    ArtifactTraversal,
    artifact_scope_key,
)
from gateway.browser_control_broker import (
    BROWSER_CONTROL_ARTIFACT_CAPABILITIES,
    BROWSER_CONTROL_CAPABILITIES,
    BROWSER_CONTROL_DEVELOPER_CAPABILITIES,
    BrowserControlBroker,
    ControllerRejected,
    ControllerScope,
    ControllerUnavailable,
    browser_control_developer_mode,
    filter_browser_control_capabilities,
)
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


API_KEY = "-".join(("fixture", "neutral", "api", "key", "123"))
PNG_BYTES = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489")
TEXT_BYTES = b"fixture artifact payload\n"


class _Scope:
    """Minimal attribute scope accepted by artifact_scope_key."""

    def __init__(self, principal="principal-fixture", session="session-fixture", family="local-api"):
        self.principal_id = principal
        self.session_id = session
        self.transport_family = family


def _adapter(*, key=API_KEY):
    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": key} if key else {})
    )
    return adapter


def _app(adapter):
    app = web.Application()
    app.router.add_post("/v1/artifacts/upload", adapter._handle_artifact_upload)
    app.router.add_get(
        "/v1/artifacts/download/{artifact_id}", adapter._handle_artifact_download
    )
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    return app


def _auth():
    return {"Authorization": f"Bearer {API_KEY}"}


# ----------------------------------------------------------------------
# Store core: size/MIME caps, SHA-256, scope binding, one-shot, TTL
# ----------------------------------------------------------------------


def test_store_rejects_oversize_and_disallowed_mime_before_writing(tmp_path):
    store = ArtifactStore(tmp_path / "root", max_bytes=8, allowed_mime_types=frozenset({"image/png"}))
    with pytest.raises(ArtifactTooLarge):
        store.store(b"123456789", filename="big.png", content_type="image/png", scope=_Scope())
    with pytest.raises(ArtifactMimeRejected):
        store.store(b"123", filename="doc.txt", content_type="text/plain", scope=_Scope())
    assert store.count() == 0
    assert not list((tmp_path / "root").iterdir())


def test_store_round_trip_validates_sha256_and_is_one_shot(tmp_path):
    store = ArtifactStore(tmp_path / "root")
    receipt = store.store(
        PNG_BYTES,
        filename="shot.png",
        content_type="image/png",
        scope=_Scope(),
    )
    assert receipt.size_bytes == len(PNG_BYTES)
    assert receipt.sha256 == hashlib.sha256(PNG_BYTES).hexdigest()
    assert receipt.filename == "shot.png"
    assert receipt.expires_at > receipt.created_at

    data, loaded = store.load(receipt.artifact_id, scope=_Scope())
    assert data == PNG_BYTES
    assert loaded.artifact_id == receipt.artifact_id
    # One-shot: a second load must fail.
    with pytest.raises(ArtifactNotFound):
        store.load(receipt.artifact_id, scope=_Scope())


def test_store_rejects_tampered_file_via_checksum(tmp_path):
    store = ArtifactStore(tmp_path / "root")
    receipt = store.store(
        TEXT_BYTES, filename="note.txt", content_type="text/plain", scope=_Scope()
    )
    target = tmp_path / "root" / receipt.artifact_id
    target.write_bytes(b"tampered bytes")
    with pytest.raises(ArtifactChecksumMismatch):
        store.load(receipt.artifact_id, scope=_Scope())
    # The tampered artifact is not consumed; a later store to a fresh id works.
    assert store.count() == 1


def test_store_is_scope_bound_and_principal_required(tmp_path):
    store = ArtifactStore(tmp_path / "root")
    receipt = store.store(
        TEXT_BYTES, filename="note.txt", content_type="text/plain", scope=_Scope()
    )
    with pytest.raises(ArtifactScopeMismatch):
        store.load(receipt.artifact_id, scope=_Scope(principal="other-principal"))
    with pytest.raises(ArtifactScopeMismatch):
        store.load(receipt.artifact_id, scope=_Scope(family="remote-api"))
    with pytest.raises(ArtifactError, match="principal"):
        store.store(
            TEXT_BYTES,
            filename="note.txt",
            content_type="text/plain",
            scope=_Scope(principal=""),
        )


def test_store_rejects_traversal_ids_and_mints_server_ids(tmp_path):
    store = ArtifactStore(tmp_path / "root")
    with pytest.raises(ArtifactTraversal):
        store.validate("../escape", scope=_Scope())
    with pytest.raises(ArtifactTraversal):
        store.validate("not-hex!", scope=_Scope())
    with pytest.raises(ArtifactTraversal):
        store.validate("", scope=_Scope())
    receipt = store.store(
        TEXT_BYTES, filename="note.txt", content_type="text/plain", scope=_Scope()
    )
    # Ids are server-minted 32-hex; filenames never become paths.
    assert len(receipt.artifact_id) == 32
    assert all(character in "0123456789abcdef" for character in receipt.artifact_id)
    assert store.validate(receipt.artifact_id, scope=_Scope()).artifact_id == receipt.artifact_id


def test_store_ttl_prunes_expired_and_drops_orphan_temps(tmp_path):
    now = [1000.0]
    store = ArtifactStore(tmp_path / "root", ttl_seconds=10.0, clock=lambda: now[0])
    receipt = store.store(
        TEXT_BYTES, filename="note.txt", content_type="text/plain", scope=_Scope()
    )
    assert store.count() == 1
    # Load also fails once TTL elapses, and the entry is pruned.
    now[0] = 1011.0
    with pytest.raises(ArtifactExpired):
        store.load(receipt.artifact_id, scope=_Scope())
    assert store.count() == 0
    # Explicit sweep is idempotent and removes stale temp files.
    orphan = tmp_path / "root" / ("deadbeef" * 4 + ".tmp")
    orphan.write_bytes(b"x")
    os.utime(orphan, (900.0, 900.0))
    assert store.prune_expired(now[0]) == 0
    assert not orphan.exists()


def test_scope_key_is_stable_across_reconnect_and_distinct_per_principal(tmp_path):
    key = artifact_scope_key(_Scope())
    assert key == artifact_scope_key(_Scope())
    assert key != artifact_scope_key(_Scope(principal="other-principal"))
    assert key != artifact_scope_key(_Scope(family="remote-api"))


# ----------------------------------------------------------------------
# Rate limiter
# ----------------------------------------------------------------------


def test_rate_limiter_sliding_window_per_key():
    now = [100.0]
    limiter = ArtifactRateLimiter(window_seconds=60.0, max_requests=3, clock=lambda: now[0])
    assert limiter.allow("principal-a")
    assert limiter.allow("principal-a")
    assert limiter.allow("principal-a")
    assert limiter.allow("principal-a") is False
    # A different key has its own budget.
    assert limiter.allow("principal-b")
    # Older hits slide out of the window.
    now[0] = 170.0
    assert limiter.allow("principal-a")
    limiter.reset("principal-a")
    assert limiter.allow("principal-a")


# ----------------------------------------------------------------------
# Broker: Developer Mode gates + artifact referencing
# ----------------------------------------------------------------------


def _broker_scope(**overrides):
    values = {
        "principal_id": "principal-fixture",
        "profile_id": "default",
        "session_id": "session-fixture",
        "controller_id": "controller-fixture",
        "browser_profile_id": "browser-profile-fixture",
        "transport_family": "local-api",
        "capabilities": frozenset({"controller.noop"}),
    }
    values.update(overrides)
    return ControllerScope(**values)


def test_developer_capabilities_are_never_in_the_base_allowlist():
    assert BROWSER_CONTROL_CAPABILITIES.isdisjoint(BROWSER_CONTROL_DEVELOPER_CAPABILITIES)
    assert BROWSER_CONTROL_ARTIFACT_CAPABILITIES.isdisjoint(BROWSER_CONTROL_DEVELOPER_CAPABILITIES)


def test_filter_rejects_developer_capabilities_without_developer_mode():
    requested = [
        "controller.noop",
        "browser_navigate",
        "browser_evaluate",
        "browser_cdp",
        "browser_artifact_upload",
        "browser_artifact_download",
        "arbitrary.capability",
    ]
    assert filter_browser_control_capabilities(requested, developer_mode=False) == frozenset(
        {"controller.noop", "browser_navigate", "browser_artifact_upload", "browser_artifact_download"}
    )
    assert filter_browser_control_capabilities(requested, developer_mode=True) == frozenset(
        {
            "controller.noop",
            "browser_navigate",
            "browser_artifact_upload",
            "browser_artifact_download",
            "browser_evaluate",
            "browser_cdp",
        }
    )
    assert filter_browser_control_capabilities("not-a-list", developer_mode=True) == frozenset()


def test_browser_control_developer_mode_reads_config_flag():
    assert browser_control_developer_mode({"browser": {"extension_control": {"developer_mode": True}}}) is True
    assert browser_control_developer_mode({"browser": {"extension_control": {}}}) is False
    assert browser_control_developer_mode({"browser": {}}) is False
    assert browser_control_developer_mode(None) is False


def test_broker_developer_gate_blocks_evaluate_and_cdp_dispatch(tmp_path):
    broker = BrowserControlBroker(developer_mode=False)
    store = ArtifactStore(tmp_path / "root")
    broker.attach_artifact_store(store)
    scope = _broker_scope(
        capabilities=frozenset({"browser_evaluate", "browser_cdp", "controller.noop"})
    )
    broker.attach(scope, lambda _frame: None)

    # Even though the controller claims the capability, Developer Mode off
    # fails closed at selection time.
    with pytest.raises(ControllerUnavailable):
        broker.dispatch(scope, action="browser_evaluate", arguments={"expression": "1"})
    with pytest.raises(ControllerUnavailable):
        broker.dispatch(scope, action="browser_cdp", arguments={"method": "Page.navigate"})
    assert broker.select(scope, "browser_evaluate") is None


def test_broker_developer_mode_allows_negotiated_privileged_dispatch(tmp_path):
    broker = BrowserControlBroker(developer_mode=True)
    store = ArtifactStore(tmp_path / "root")
    broker.attach_artifact_store(store)
    scope = _broker_scope(
        capabilities=frozenset({"browser_evaluate", "browser_cdp", "controller.noop"})
    )

    def send(frame):
        broker.complete(
            frame["params"]["command_id"],
            ok=True,
            result={"expression": frame["params"]["arguments"]["expression"]},
        )

    broker.attach(scope, send)
    result = broker.dispatch(
        scope, action="browser_evaluate", arguments={"expression": "document.title"}
    )
    assert result == {"expression": "document.title"}


def test_broker_artifact_action_requires_attached_store(tmp_path):
    broker = BrowserControlBroker()
    scope = _broker_scope(
        capabilities=frozenset({"browser_artifact_download", "controller.noop"})
    )
    broker.attach(scope, lambda _frame: None)
    with pytest.raises(ControllerRejected, match="artifact store"):
        broker.dispatch(
            scope,
            action="browser_artifact_download",
            arguments={"artifact_id": "a" * 32},
        )


def test_broker_artifact_action_requires_approved_id_only(tmp_path):
    broker = BrowserControlBroker()
    store = ArtifactStore(tmp_path / "root")
    broker.attach_artifact_store(store)
    scope = _broker_scope(
        capabilities=frozenset({"browser_artifact_download", "controller.noop"})
    )
    frames = []

    def send(frame):
        frames.append(frame)
        broker.complete(frame["params"]["command_id"], ok=True, result={"ok": True})

    broker.attach(scope, send)

    # Unknown id fails closed with the artifact error surfaced.
    with pytest.raises(ControllerRejected, match="unknown artifact"):
        broker.dispatch(
            scope,
            action="browser_artifact_download",
            arguments={"artifact_id": "b" * 32},
        )
    assert frames == []

    # Missing id is refused.
    with pytest.raises(ControllerRejected, match="non-empty artifact_id"):
        broker.dispatch(scope, action="browser_artifact_download", arguments={})

    # An approved (stored, scope-bound) id dispatches; the frame carries the
    # id, never the bytes.
    receipt = store.store(
        PNG_BYTES,
        filename="shot.png",
        content_type="image/png",
        scope=_Scope(),
    )
    result = broker.dispatch(
        scope,
        action="browser_artifact_download",
        arguments={"artifact_id": receipt.artifact_id},
    )
    assert result == {"ok": True}
    assert frames[0]["params"]["arguments"]["artifact_id"] == receipt.artifact_id
    # The frame carries only the id — never the payload bytes.
    assert "data" not in frames[0]["params"]["arguments"]
    assert all(
        not isinstance(value, bytes) for value in frames[0]["params"]["arguments"].values()
    )


def test_broker_artifact_action_rejects_other_scope_artifact(tmp_path):
    broker = BrowserControlBroker()
    store = ArtifactStore(tmp_path / "root")
    broker.attach_artifact_store(store)
    scope = _broker_scope(
        capabilities=frozenset({"browser_artifact_upload", "controller.noop"})
    )
    broker.attach(scope, lambda _frame: None)
    receipt = store.store(
        TEXT_BYTES,
        filename="note.txt",
        content_type="text/plain",
        scope=_Scope(principal="someone-else"),
    )
    with pytest.raises(ControllerRejected, match="different scope"):
        broker.dispatch(
            scope,
            action="browser_artifact_upload",
            arguments={"artifact_id": receipt.artifact_id},
        )


# ----------------------------------------------------------------------
# API-server routes: auth, feature gate, size/MIME, one-shot, rate limit
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_artifact_upload_requires_auth_and_feature_flag(monkeypatch):
    adapter = _adapter()
    monkeypatch.setattr(adapter, "_browser_control_enabled", lambda: True)
    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.post(
            "/v1/artifacts/upload",
            data=TEXT_BYTES,
            headers={"Content-Type": "text/plain", "X-Artifact-Filename": "note.txt"},
        )
        assert response.status == 401

    adapter = _adapter()
    monkeypatch.setattr(adapter, "_browser_control_enabled", lambda: False)
    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.post(
            "/v1/artifacts/upload",
            data=TEXT_BYTES,
            headers={
                "Content-Type": "text/plain",
                "X-Artifact-Filename": "note.txt",
                **_auth(),
            },
        )
        assert response.status == 404


@pytest.mark.asyncio
async def test_artifact_upload_download_round_trip_one_shot(monkeypatch, tmp_path):
    adapter = _adapter()
    monkeypatch.setattr(adapter, "_browser_control_enabled", lambda: True)
    store = ArtifactStore(tmp_path / "root")
    adapter._inject_browser_control_artifacts(store)

    async with TestClient(TestServer(_app(adapter))) as client:
        upload = await client.post(
            "/v1/artifacts/upload",
            data=TEXT_BYTES,
            headers={
                "Content-Type": "text/plain",
                "X-Artifact-Filename": "note.txt",
                **_auth(),
            },
        )
        assert upload.status == 201
        receipt = await upload.json()
        assert receipt["size_bytes"] == len(TEXT_BYTES)
        assert receipt["sha256"] == hashlib.sha256(TEXT_BYTES).hexdigest()
        assert receipt["one_shot"] is True
        assert receipt["download_path"] == f"/v1/artifacts/download/{receipt['artifact_id']}"

        download = await client.get(receipt["download_path"], headers=_auth())
        assert download.status == 200
        body = await download.read()
        assert body == TEXT_BYTES
        assert download.headers["X-Artifact-Sha256"] == receipt["sha256"]

        # One-shot: the same id is consumed.
        replay = await client.get(receipt["download_path"], headers=_auth())
        assert replay.status == 404


@pytest.mark.asyncio
async def test_artifact_upload_rejects_mime_and_missing_filename(monkeypatch, tmp_path):
    adapter = _adapter()
    monkeypatch.setattr(adapter, "_browser_control_enabled", lambda: True)
    store = ArtifactStore(tmp_path / "root")
    adapter._inject_browser_control_artifacts(store)
    async with TestClient(TestServer(_app(adapter))) as client:
        bad_mime = await client.post(
            "/v1/artifacts/upload",
            data=b"<html></html>",
            headers={
                "Content-Type": "text/html",
                "X-Artifact-Filename": "page.html",
                **_auth(),
            },
        )
        assert bad_mime.status == 415

        no_name = await client.post(
            "/v1/artifacts/upload",
            data=TEXT_BYTES,
            headers={"Content-Type": "text/plain", **_auth()},
        )
        assert no_name.status == 400


@pytest.mark.asyncio
async def test_artifact_upload_bounded_body_rejects_oversize(monkeypatch, tmp_path):
    adapter = _adapter()
    monkeypatch.setattr(adapter, "_browser_control_enabled", lambda: True)
    store = ArtifactStore(tmp_path / "root", max_bytes=16)
    adapter._inject_browser_control_artifacts(store)
    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.post(
            "/v1/artifacts/upload",
            data=b"x" * 17,
            headers={
                "Content-Type": "text/plain",
                "X-Artifact-Filename": "big.txt",
                **_auth(),
            },
        )
        assert response.status == 413


@pytest.mark.asyncio
async def test_artifact_download_rejects_foreign_scope_and_unknown_id(monkeypatch, tmp_path):
    adapter = _adapter()
    monkeypatch.setattr(adapter, "_browser_control_enabled", lambda: True)
    store = ArtifactStore(tmp_path / "root")
    adapter._inject_browser_control_artifacts(store)
    receipt = store.store(
        TEXT_BYTES,
        filename="note.txt",
        content_type="text/plain",
        scope=_Scope(principal="someone-else"),
    )
    async with TestClient(TestServer(_app(adapter))) as client:
        foreign = await client.get(
            f"/v1/artifacts/download/{receipt.artifact_id}", headers=_auth()
        )
        assert foreign.status == 400

        unknown = await client.get(
            "/v1/artifacts/download/" + "f" * 32, headers=_auth()
        )
        assert unknown.status == 404


@pytest.mark.asyncio
async def test_artifact_routes_rate_limit_per_principal(monkeypatch, tmp_path):
    adapter = _adapter()
    monkeypatch.setattr(adapter, "_browser_control_enabled", lambda: True)
    store = ArtifactStore(tmp_path / "root")
    limiter = ArtifactRateLimiter(window_seconds=60.0, max_requests=2)
    adapter._inject_browser_control_artifacts(store, limiter)
    async with TestClient(TestServer(_app(adapter))) as client:
        statuses = []
        for _index in range(3):
            response = await client.post(
                "/v1/artifacts/upload",
                data=TEXT_BYTES,
                headers={
                    "Content-Type": "text/plain",
                    "X-Artifact-Filename": "note.txt",
                    **_auth(),
                },
            )
            statuses.append(response.status)
        assert statuses == [201, 201, 429]


@pytest.mark.asyncio
async def test_capabilities_advertise_artifact_transport_and_developer_mode(monkeypatch):
    adapter = _adapter()
    monkeypatch.setattr(adapter, "_browser_control_enabled", lambda: True)
    monkeypatch.setattr(adapter, "_browser_control_developer_mode", lambda: True)
    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.get("/v1/capabilities", headers=_auth())
        assert response.status == 200
        data = await response.json()

    control = data["features"]["browser_extension_control"]
    assert control["developer_mode"] is True
    assert "browser_evaluate" in control["developer_capabilities"]
    assert "browser_cdp" in control["developer_capabilities"]
    assert "browser_evaluate" not in control["capabilities"]
    assert control["artifact_transport"]["upload"] == {
        "method": "POST",
        "path": "/v1/artifacts/upload",
    }
    assert control["artifact_transport"]["download"]["path"] == (
        "/v1/artifacts/download/{artifact_id}"
    )
    assert data["endpoints"]["artifact_upload"] == {
        "method": "POST",
        "path": "/v1/artifacts/upload",
    }
    assert data["endpoints"]["artifact_download"] == {
        "method": "GET",
        "path": "/v1/artifacts/download/{artifact_id}",
    }


def test_route_table_advertises_artifact_routes():
    adapter = _adapter()
    routes = {(method, path) for method, path, _handler in adapter._http_route_table()}
    assert ("POST", "/v1/artifacts/upload") in routes
    assert ("GET", "/v1/artifacts/download/{artifact_id}") in routes


def test_http_uploaded_artifact_composes_with_broker_dispatch(tmp_path):
    """The real journey: HTTP upload (no session) -> broker artifact dispatch.

    Regression for the scope-key mismatch review blocker: the HTTP artifact
    routes can never resolve a server session, so artifact ownership is
    principal/transport-family scoped and a session-bearing ControllerScope
    must validate the same artifact.
    """
    store = ArtifactStore(tmp_path / "root")
    # Upload-side scope: what api_server's facade carries (empty session).
    receipt = store.store(
        TEXT_BYTES,
        filename="note.txt",
        content_type="text/plain",
        scope=_Scope(session=""),
    )

    broker = BrowserControlBroker()
    broker.attach_artifact_store(store)
    scope = _broker_scope(
        capabilities=frozenset({"browser_artifact_upload", "controller.noop"})
    )

    def send(frame):
        broker.complete(
            frame["params"]["command_id"], ok=True, result={"ok": True}
        )

    broker.attach(scope, send)
    result = broker.dispatch(
        scope,
        action="browser_artifact_upload",
        arguments={"artifact_id": receipt.artifact_id},
    )
    assert result == {"ok": True}

    # Cross-principal / cross-family access still fails closed.
    with pytest.raises(ArtifactScopeMismatch):
        store.validate(receipt.artifact_id, scope=_Scope(principal="other"))
    with pytest.raises(ArtifactScopeMismatch):
        store.validate(receipt.artifact_id, scope=_Scope(session="", family="remote-api"))


def test_multiplex_profiles_get_distinct_stores_regardless_of_touch_order(tmp_path, monkeypatch):
    """Profile A touching the artifact route first must not pin profile B."""
    import gateway.platforms.api_server as api_server_mod

    adapter = _adapter()
    monkeypatch.setattr(
        "hermes_cli.profiles.get_profile_dir",
        lambda profile: str(tmp_path / f"home-{profile}"),
    )

    store_a = adapter._artifact_store_for("profile-a")
    store_b = adapter._artifact_store_for("profile-b")
    assert store_a is not store_b
    assert str(store_a.root) != str(store_b.root)
    assert "home-profile-a" in str(store_a.root)
    assert "home-profile-b" in str(store_b.root)
    # Repeat lookups return the same cached store per profile.
    assert adapter._artifact_store_for("profile-a") is store_a
    assert adapter._artifact_store_for("profile-b") is store_b

    # The broker resolves each profile's own store from the controller scope.
    broker = adapter._browser_control_broker
    scope_a = _broker_scope(profile_id="profile-a")
    scope_b = _broker_scope(profile_id="profile-b")
    assert broker._artifact_store_for_scope(scope_a) is store_a
    assert broker._artifact_store_for_scope(scope_b) is store_b


def test_developer_mode_flip_revokes_privileged_selection_live(monkeypatch):
    """Turning developer_mode off in config revokes CDP/eval from an
    already-attached controller without a process restart (and on->off
    the reverse: enabling unlocks selection for a new negotiation)."""
    import gateway.browser_control_broker as broker_mod

    flag = {"on": True}
    monkeypatch.setattr(
        broker_mod, "browser_control_developer_mode", lambda config=None: flag["on"]
    )
    broker = BrowserControlBroker()  # developer_mode=None -> live config
    scope = _broker_scope(
        capabilities=frozenset({"browser_evaluate", "browser_cdp", "controller.noop"})
    )
    broker.attach(scope, lambda _frame: None)

    assert broker.select(scope, "browser_evaluate") is not None
    # Revocation: flip the live flag off — the attached controller loses
    # privileged selection immediately.
    flag["on"] = False
    assert broker.select(scope, "browser_evaluate") is None
    assert broker.select(scope, "browser_cdp") is None
    # Base capabilities are unaffected by the developer gate.
    assert broker.select(scope, "controller.noop") is not None
    # And back on: selection resumes without any rebind.
    flag["on"] = True
    assert broker.select(scope, "browser_cdp") is not None
    # Explicit pin still wins over live config (test/multi-tenant contract).
    pinned = BrowserControlBroker(developer_mode=False)
    pinned.attach(scope, lambda _frame: None)
    assert pinned.select(scope, "browser_evaluate") is None


def test_store_construction_sweeps_orphan_files_from_previous_process(tmp_path):
    """Files left by a dead process (unreachable, past advertised TTL) are
    removed when a fresh store opens the same root."""
    root = tmp_path / "root"
    store = ArtifactStore(root)
    receipt = store.store(
        TEXT_BYTES, filename="note.txt", content_type="text/plain", scope=_Scope()
    )
    orphan = root / receipt.artifact_id
    assert orphan.exists()
    stale_tmp = root / "deadbeef.tmp"
    stale_tmp.write_bytes(b"partial")
    unrelated = root / "README"
    unrelated.write_bytes(b"keep me")

    # Simulate restart: a new store over the same root has an empty index.
    fresh = ArtifactStore(root)
    assert not orphan.exists()
    assert not stale_tmp.exists()
    assert unrelated.exists()  # non-artifact-shaped names untouched
    assert fresh.count() == 0
    # New store works normally afterwards.
    fresh.store(TEXT_BYTES, filename="new.txt", content_type="text/plain", scope=_Scope())
    assert fresh.count() == 1
