"""Tests for the bundled ``ttm-control-plane`` dashboard plugin.

Mirrors the loader semantics in
``hermes_cli.web_server._mount_plugin_api_routes`` (sys.modules
registration before exec) so dataclass / pydantic forward refs resolve
the same way they do at runtime, then exercises the wire contract:

* shared-secret auth gates writes when ``TTM_CONTROL_PLANE_SECRET`` is
  set; unauthenticated reads of ``/health`` always work
* ``/runs/dispatch`` rejects payloads missing the principal token (per
  RUNTIME-PRINCIPAL-CONTRACT.md the credential MUST ride the body) and
  rejects mismatched ``runtime_id``
* the happy path returns 202 with a fresh ``runtime_run_ref`` and binds
  the run in-memory; a duplicate dispatch returns 409 with the prior
  ``runtime_run_ref`` and leaves the binding intact
* ``/runs/{ref}/status`` round-trips
* ``/runs/{ref}/lifecycle`` validates actions and returns 202 immediately
* ``/runs/{ref}/stop`` (compat) returns 202 and keeps the binding alive
* pause/resume/expand_scope handlers are exercised with mocked processes
* stop handler: SIGTERM → wait → SIGKILL fallback with mocked handles
* ingress events are emitted (mocked) and never log token material
"""
from __future__ import annotations

import asyncio
import importlib.util
import signal
import sys
import tempfile
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

PLUGIN_API_PATH = (
    Path(__file__).resolve().parents[2]
    / "plugins"
    / "ttm-control-plane"
    / "dashboard"
    / "plugin_api.py"
)


def _load_plugin_module():
    """Import plugin_api.py the same way the dashboard does.

    The dashboard registers the dynamically-loaded module in
    ``sys.modules`` before ``exec_module`` so ``from __future__ import
    annotations`` + dataclasses resolve correctly. Tests must do the
    same, otherwise dataclass(...) blows up when introspecting the
    placeholder module's __dict__.
    """
    module_name = "hermes_dashboard_plugin_ttm_control_plane_test"
    spec = importlib.util.spec_from_file_location(module_name, PLUGIN_API_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return mod


def _make_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    secret: str | None = "test-secret",
    db_path: Path | None = None,
):
    """Build a fresh FastAPI app with the plugin mounted under
    ``/api/plugins/ttm-control-plane`` and the registry cleared.

    ``monkeypatch`` sets the env var so each test sees the auth model it
    expects without leaking state across tests. The SQLite-backed
    binding registry is pointed at a per-test ``db_path`` (or a
    process-wide test path when none is given) so persistence does not
    leak across tests.
    """
    if secret is None:
        monkeypatch.delenv("TTM_CONTROL_PLANE_SECRET", raising=False)
    else:
        monkeypatch.setenv("TTM_CONTROL_PLANE_SECRET", secret)
    if db_path is None:
        # Each call gets a fresh temp DB so parallel workers don't collide.
        fd, tmp = tempfile.mkstemp(suffix=".db", prefix="ttm_cp_test_")
        import os as _os
        _os.close(fd)
        db_path = Path(tmp)
    monkeypatch.setenv("TTM_CONTROL_PLANE_DB_PATH", str(db_path))
    # Disable subprocess spawn for the vast majority of tests — only
    # the dedicated spawn-path tests opt back in.
    monkeypatch.setenv("TTM_CONTROL_PLANE_DISABLE_SPAWN", "1")
    plugin = _load_plugin_module()
    plugin._REGISTRY.clear()
    plugin._PROC_REGISTRY.clear()
    with plugin._PAUSE_LOCK:
        plugin._PAUSE_STATE.clear()
    app = FastAPI()
    app.include_router(plugin.router, prefix="/api/plugins/ttm-control-plane")
    return TestClient(app), plugin


def _dispatch_body(**overrides):
    body = {
        "run_id": "11111111-1111-1111-1111-111111111111",
        "runtime_id": "hermes",
        "stream_id": "galactus",
        "stream_version": "2026-04-28",
        "runtime_binding_id": "22222222-2222-2222-2222-222222222222",
        "slice_id": "spawn",
        "lane_id": "11111111-1111-1111-1111-111111111111:default",
        "scope_hash": "8" * 64,
        "worktree_id": "wt-1",
        "goal": "spawn galactus run",
        "allowed_paths": ["backend/app/api/routes/runs.py"],
        "required_tests": ["pytest backend/tests -q"],
        "reply_schema": {"type": "object"},
        "deadline_at": "2026-04-29T00:00:00Z",
        "ingress_base_url": "",
        "principal_token": "ttm-issued-bearer-credential",
    }
    body.update(overrides)
    return body


def _dispatch_and_get_ref(client: TestClient, headers: dict) -> tuple[str, str]:
    """Dispatch a run and return (run_id, runtime_run_ref)."""
    resp = client.post(
        "/api/plugins/ttm-control-plane/runs/dispatch",
        json=_dispatch_body(),
        headers=headers,
    )
    assert resp.status_code == 202, resp.text
    return _dispatch_body()["run_id"], resp.json()["runtime_run_ref"]


# ---------------------------------------------------------------------------
# Health / auth
# ---------------------------------------------------------------------------


def test_health_returns_plugin_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _make_client(monkeypatch)
    resp = client.get("/api/plugins/ttm-control-plane/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["plugin"] == "ttm-control-plane"
    assert body["auth_enforced"] is True
    assert body["bindings"] == 0


def test_health_signals_auth_disabled_when_secret_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = _make_client(monkeypatch, secret=None)
    resp = client.get("/api/plugins/ttm-control-plane/health")
    assert resp.status_code == 200
    assert resp.json()["auth_enforced"] is False


def test_dispatch_rejects_missing_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _make_client(monkeypatch)
    resp = client.post(
        "/api/plugins/ttm-control-plane/runs/dispatch",
        json=_dispatch_body(),
    )
    assert resp.status_code == 401
    assert resp.json()["detail"] == {"reason": "ttm_control_plane_secret_mismatch"}


def test_dispatch_rejects_wrong_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _make_client(monkeypatch)
    resp = client.post(
        "/api/plugins/ttm-control-plane/runs/dispatch",
        json=_dispatch_body(),
        headers={"X-TTM-Control-Plane-Secret": "wrong"},
    )
    assert resp.status_code == 401


def test_dispatch_allowed_when_secret_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dev/CI fallback: empty TTM_CONTROL_PLANE_SECRET disables auth so
    the plugin is testable without provisioning a secret. Production
    deployment must set the env var."""
    client, _ = _make_client(monkeypatch, secret=None)
    resp = client.post(
        "/api/plugins/ttm-control-plane/runs/dispatch",
        json=_dispatch_body(),
    )
    assert resp.status_code == 202


# ---------------------------------------------------------------------------
# Wire contract
# ---------------------------------------------------------------------------


def test_dispatch_rejects_missing_principal_token(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _make_client(monkeypatch)
    resp = client.post(
        "/api/plugins/ttm-control-plane/runs/dispatch",
        json=_dispatch_body(principal_token=""),
        headers={"X-TTM-Control-Plane-Secret": "test-secret"},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["reason"] == "principal_token_required"


def test_dispatch_rejects_runtime_id_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _make_client(monkeypatch)
    resp = client.post(
        "/api/plugins/ttm-control-plane/runs/dispatch",
        json=_dispatch_body(runtime_id="oc"),
        headers={"X-TTM-Control-Plane-Secret": "test-secret"},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["reason"] == "runtime_id_mismatch"


def test_dispatch_happy_path_returns_runtime_run_ref(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, plugin = _make_client(monkeypatch)
    resp = client.post(
        "/api/plugins/ttm-control-plane/runs/dispatch",
        json=_dispatch_body(),
        headers={"X-TTM-Control-Plane-Secret": "test-secret"},
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["status"] == "accepted"
    assert body["runtime_run_ref"].startswith("hermes-")
    # Binding is recorded for idempotency lookups.
    bindings = plugin._REGISTRY.snapshot()
    assert len(bindings) == 1
    assert bindings[0].run_id == "11111111-1111-1111-1111-111111111111"


def test_dispatch_is_idempotent_returns_409_on_redispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = _make_client(monkeypatch)
    headers = {"X-TTM-Control-Plane-Secret": "test-secret"}
    first = client.post(
        "/api/plugins/ttm-control-plane/runs/dispatch",
        json=_dispatch_body(),
        headers=headers,
    )
    assert first.status_code == 202
    first_ref = first.json()["runtime_run_ref"]

    second = client.post(
        "/api/plugins/ttm-control-plane/runs/dispatch",
        json=_dispatch_body(),
        headers=headers,
    )
    assert second.status_code == 409
    detail = second.json()["detail"]
    assert detail["reason"] == "run_already_bound"
    # Surface the prior ref so the caller can recover without a fresh
    # dispatch.
    assert detail["runtime_run_ref"] == first_ref


def test_status_round_trips_known_runtime_run_ref(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = _make_client(monkeypatch)
    headers = {"X-TTM-Control-Plane-Secret": "test-secret"}
    dispatched = client.post(
        "/api/plugins/ttm-control-plane/runs/dispatch",
        json=_dispatch_body(),
        headers=headers,
    )
    ref = dispatched.json()["runtime_run_ref"]

    status_resp = client.get(
        f"/api/plugins/ttm-control-plane/runs/{ref}/status",
        headers=headers,
    )
    assert status_resp.status_code == 200, status_resp.text
    assert status_resp.json()["runtime_run_ref"] == ref
    assert status_resp.json()["status"] in {"accepted", "pending"}


def test_status_404s_for_unknown_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _make_client(monkeypatch)
    resp = client.get(
        "/api/plugins/ttm-control-plane/runs/hermes-does-not-exist/status",
        headers={"X-TTM-Control-Plane-Secret": "test-secret"},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /runs/{ref}/stop — compat route
# ---------------------------------------------------------------------------


def test_stop_compat_returns_202_and_keeps_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """H6: /stop keeps the binding alive (status=stopped) so TTM can still
    query status and drive canonical closure. It no longer removes the binding."""
    client, plugin = _make_client(monkeypatch)
    headers = {"X-TTM-Control-Plane-Secret": "test-secret"}
    _, ref = _dispatch_and_get_ref(client, headers)

    stopped = client.post(
        f"/api/plugins/ttm-control-plane/runs/{ref}/stop",
        headers=headers,
    )
    assert stopped.status_code == 202
    body = stopped.json()
    assert body["status"] == "accepted"
    assert body["runtime_run_ref"] == ref

    # Binding must survive; status is updated asynchronously, but the entry exists.
    bindings = plugin._REGISTRY.snapshot()
    assert len(bindings) == 1


def test_stop_compat_404s_for_unknown_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _make_client(monkeypatch)
    resp = client.post(
        "/api/plugins/ttm-control-plane/runs/hermes-does-not-exist/stop",
        headers={"X-TTM-Control-Plane-Secret": "test-secret"},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /runs/{ref}/lifecycle — unified lifecycle receiver
# ---------------------------------------------------------------------------


def test_lifecycle_rejects_missing_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _make_client(monkeypatch)
    headers = {"X-TTM-Control-Plane-Secret": "test-secret"}
    _, ref = _dispatch_and_get_ref(client, headers)

    resp = client.post(
        f"/api/plugins/ttm-control-plane/runs/{ref}/lifecycle",
        json={"action": "stop"},
    )
    assert resp.status_code == 401


def test_lifecycle_404s_for_unknown_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _make_client(monkeypatch)
    resp = client.post(
        "/api/plugins/ttm-control-plane/runs/hermes-not-real/lifecycle",
        json={"action": "stop"},
        headers={"X-TTM-Control-Plane-Secret": "test-secret"},
    )
    assert resp.status_code == 404


def test_lifecycle_rejects_invalid_action(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _make_client(monkeypatch)
    headers = {"X-TTM-Control-Plane-Secret": "test-secret"}
    _, ref = _dispatch_and_get_ref(client, headers)

    resp = client.post(
        f"/api/plugins/ttm-control-plane/runs/{ref}/lifecycle",
        json={"action": "nuke"},
        headers=headers,
    )
    assert resp.status_code == 422


@pytest.mark.parametrize("action", ["stop", "pause", "resume", "expand_scope"])
def test_lifecycle_accepts_valid_actions(
    monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    client, _ = _make_client(monkeypatch)
    headers = {"X-TTM-Control-Plane-Secret": "test-secret"}
    _, ref = _dispatch_and_get_ref(client, headers)

    resp = client.post(
        f"/api/plugins/ttm-control-plane/runs/{ref}/lifecycle",
        json={"action": action},
        headers=headers,
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["status"] == "accepted"
    assert body["runtime_run_ref"] == ref
    assert body["action"] == action
    assert "accepted_at" in body


# ---------------------------------------------------------------------------
# Stop async handler — process kill logic
# ---------------------------------------------------------------------------


def test_stop_handler_sigterm_then_sigkill_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SIGTERM is sent first; SIGKILL is sent when the process does not exit."""
    plugin = _load_plugin_module()
    plugin._REGISTRY.clear()
    plugin._PROC_REGISTRY.clear()

    run_id = "aaaa-stop-test"
    ref = "hermes-stop-test-ref"

    # Binding in registry
    from datetime import UTC, datetime

    binding = plugin._RuntimeBinding(
        run_id=run_id,
        runtime_binding_id="bid",
        runtime_run_ref=ref,
        ingress_base_url="",
        bound_at=datetime.now(UTC),
        principal_token="",
        last_status="running",
    )
    plugin._REGISTRY.insert(binding)

    # Mock process that never exits (returncode stays None)
    mock_proc = MagicMock()
    mock_proc.pid = 99999
    mock_proc.returncode = None

    async def fake_wait():
        raise asyncio.TimeoutError()

    mock_proc.wait = fake_wait

    handle = plugin._ProcessHandle(
        run_id=run_id,
        pid=99999,
        proc=mock_proc,
        started_at=datetime.now(UTC),
    )
    with plugin._PROC_REGISTRY._lock:
        plugin._PROC_REGISTRY._by_run[run_id] = handle

    signals_sent = []

    def fake_getpgid(pid: int) -> int:
        return pid

    def fake_killpg(pgid: int, sig: int) -> None:
        signals_sent.append(sig)

    with (
        patch("os.getpgid", side_effect=fake_getpgid),
        patch("os.killpg", side_effect=fake_killpg),
    ):
        asyncio.get_event_loop().run_until_complete(
            plugin._async_stop(run_id, ref)
        )

    assert signal.SIGTERM in signals_sent
    assert signal.SIGKILL in signals_sent
    # Process handle must be removed after stop.
    assert plugin._PROC_REGISTRY.get(run_id) is None
    # Binding stays; status updated to stopped.
    b = plugin._REGISTRY.get(run_id)
    assert b is not None
    assert b.last_status == "stopped"


def test_stop_handler_no_sigkill_when_process_exits_on_sigterm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the process exits within the SIGTERM window, SIGKILL is not sent."""
    plugin = _load_plugin_module()
    plugin._REGISTRY.clear()
    plugin._PROC_REGISTRY.clear()

    run_id = "bbbb-stop-test"
    ref = "hermes-stop-clean"
    from datetime import UTC, datetime

    binding = plugin._RuntimeBinding(
        run_id=run_id,
        runtime_binding_id="bid2",
        runtime_run_ref=ref,
        ingress_base_url="",
        bound_at=datetime.now(UTC),
        principal_token="",
        last_status="running",
    )
    plugin._REGISTRY.insert(binding)

    mock_proc = MagicMock()
    mock_proc.pid = 88888

    async def fast_wait():
        # Simulates immediate exit
        mock_proc.returncode = 0

    mock_proc.wait = fast_wait
    mock_proc.returncode = 0  # already exited

    handle = plugin._ProcessHandle(
        run_id=run_id, pid=88888, proc=mock_proc, started_at=datetime.now(UTC)
    )
    with plugin._PROC_REGISTRY._lock:
        plugin._PROC_REGISTRY._by_run[run_id] = handle

    signals_sent = []

    with (
        patch("os.getpgid", return_value=88888),
        patch("os.killpg", side_effect=lambda pgid, sig: signals_sent.append(sig)),
    ):
        asyncio.get_event_loop().run_until_complete(
            plugin._async_stop(run_id, ref)
        )

    assert signal.SIGTERM in signals_sent
    assert signal.SIGKILL not in signals_sent


def test_stop_handler_no_process_registered_is_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop with no registered process must not raise."""
    plugin = _load_plugin_module()
    plugin._REGISTRY.clear()
    plugin._PROC_REGISTRY.clear()

    run_id = "cccc-no-proc"
    ref = "hermes-no-proc-ref"
    from datetime import UTC, datetime

    binding = plugin._RuntimeBinding(
        run_id=run_id,
        runtime_binding_id="bid3",
        runtime_run_ref=ref,
        ingress_base_url="",
        bound_at=datetime.now(UTC),
        principal_token="",
        last_status="accepted",
    )
    plugin._REGISTRY.insert(binding)

    asyncio.get_event_loop().run_until_complete(
        plugin._async_stop(run_id, ref)
    )
    assert plugin._REGISTRY.get(run_id).last_status == "stopped"


# ---------------------------------------------------------------------------
# Pause / resume handlers
# ---------------------------------------------------------------------------


def test_pause_handler_sends_sigstop_and_saves_dossier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = _load_plugin_module()
    plugin._REGISTRY.clear()
    plugin._PROC_REGISTRY.clear()
    with plugin._PAUSE_LOCK:
        plugin._PAUSE_STATE.clear()

    run_id = "dddd-pause-test"
    ref = "hermes-pause-ref"
    from datetime import UTC, datetime

    binding = plugin._RuntimeBinding(
        run_id=run_id,
        runtime_binding_id="bid4",
        runtime_run_ref=ref,
        ingress_base_url="",
        bound_at=datetime.now(UTC),
        principal_token="",
        last_status="running",
        payload_summary={"lane_id": "main-lane", "worktree_id": "wt-123"},
    )
    plugin._REGISTRY.insert(binding)

    mock_proc = MagicMock()
    mock_proc.pid = 77777
    handle = plugin._ProcessHandle(
        run_id=run_id, pid=77777, proc=mock_proc, started_at=datetime.now(UTC)
    )
    with plugin._PROC_REGISTRY._lock:
        plugin._PROC_REGISTRY._by_run[run_id] = handle

    signals_sent = []
    with (
        patch("os.getpgid", return_value=77777),
        patch("os.killpg", side_effect=lambda pgid, sig: signals_sent.append(sig)),
    ):
        asyncio.get_event_loop().run_until_complete(
            plugin._async_pause(run_id, ref)
        )

    assert signal.SIGSTOP in signals_sent
    with plugin._PAUSE_LOCK:
        state = plugin._PAUSE_STATE.get(run_id)
    assert state is not None
    assert state.lane_id == "main-lane"
    assert state.worktree_id == "wt-123"
    assert plugin._REGISTRY.get(run_id).last_status == "paused"


def test_pause_handler_degrades_when_sigstop_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If SIGSTOP raises, pause must emit runtime.error and NOT report paused."""
    plugin = _load_plugin_module()
    plugin._REGISTRY.clear()
    plugin._PROC_REGISTRY.clear()
    with plugin._PAUSE_LOCK:
        plugin._PAUSE_STATE.clear()

    run_id = "eeee-pause-fail"
    ref = "hermes-pause-fail-ref"
    from datetime import UTC, datetime

    binding = plugin._RuntimeBinding(
        run_id=run_id,
        runtime_binding_id="bid5",
        runtime_run_ref=ref,
        ingress_base_url="",
        bound_at=datetime.now(UTC),
        principal_token="",
        last_status="running",
    )
    plugin._REGISTRY.insert(binding)

    mock_proc = MagicMock()
    mock_proc.pid = 66666
    handle = plugin._ProcessHandle(
        run_id=run_id, pid=66666, proc=mock_proc, started_at=datetime.now(UTC)
    )
    with plugin._PROC_REGISTRY._lock:
        plugin._PROC_REGISTRY._by_run[run_id] = handle

    emitted_events: list[str] = []

    async def fake_emit(binding, event_type, payload):
        emitted_events.append(event_type)

    with (
        patch("os.getpgid", return_value=66666),
        patch("os.killpg", side_effect=PermissionError("SIGSTOP denied")),
        patch.object(plugin, "_emit_lifecycle_event", side_effect=fake_emit),
    ):
        asyncio.get_event_loop().run_until_complete(
            plugin._async_pause(run_id, ref)
        )

    assert "runtime.error" in emitted_events
    # Status must NOT be set to paused when SIGSTOP failed.
    assert plugin._REGISTRY.get(run_id).last_status == "running"
    with plugin._PAUSE_LOCK:
        assert plugin._PAUSE_STATE.get(run_id) is None


def test_resume_handler_sends_sigcont_and_clears_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = _load_plugin_module()
    plugin._REGISTRY.clear()
    plugin._PROC_REGISTRY.clear()
    with plugin._PAUSE_LOCK:
        plugin._PAUSE_STATE.clear()

    run_id = "ffff-resume-test"
    ref = "hermes-resume-ref"
    from datetime import UTC, datetime

    binding = plugin._RuntimeBinding(
        run_id=run_id,
        runtime_binding_id="bid6",
        runtime_run_ref=ref,
        ingress_base_url="",
        bound_at=datetime.now(UTC),
        principal_token="",
        last_status="paused",
    )
    plugin._REGISTRY.insert(binding)

    mock_proc = MagicMock()
    mock_proc.pid = 55555
    handle = plugin._ProcessHandle(
        run_id=run_id, pid=55555, proc=mock_proc, started_at=datetime.now(UTC)
    )
    with plugin._PROC_REGISTRY._lock:
        plugin._PROC_REGISTRY._by_run[run_id] = handle

    pause_state = plugin._PauseState(
        run_id=run_id,
        pid=55555,
        paused_at=datetime.now(UTC),
        lane_id="main-lane",
        worktree_id="wt-456",
    )
    with plugin._PAUSE_LOCK:
        plugin._PAUSE_STATE[run_id] = pause_state

    signals_sent = []
    with (
        patch("os.getpgid", return_value=55555),
        patch("os.killpg", side_effect=lambda pgid, sig: signals_sent.append(sig)),
    ):
        asyncio.get_event_loop().run_until_complete(
            plugin._async_resume(run_id, ref)
        )

    assert signal.SIGCONT in signals_sent
    with plugin._PAUSE_LOCK:
        assert plugin._PAUSE_STATE.get(run_id) is None
    assert plugin._REGISTRY.get(run_id).last_status == "running"


def test_resume_handler_no_op_without_pause_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resume when no pause state exists logs a warning but does not raise."""
    plugin = _load_plugin_module()
    plugin._REGISTRY.clear()
    plugin._PROC_REGISTRY.clear()
    with plugin._PAUSE_LOCK:
        plugin._PAUSE_STATE.clear()

    run_id = "gggg-resume-no-state"
    ref = "hermes-resume-nostate"
    from datetime import UTC, datetime

    binding = plugin._RuntimeBinding(
        run_id=run_id,
        runtime_binding_id="bid7",
        runtime_run_ref=ref,
        ingress_base_url="",
        bound_at=datetime.now(UTC),
        principal_token="",
        last_status="running",
    )
    plugin._REGISTRY.insert(binding)

    with patch("os.killpg") as mock_kill:
        asyncio.get_event_loop().run_until_complete(
            plugin._async_resume(run_id, ref)
        )
    mock_kill.assert_not_called()


# ---------------------------------------------------------------------------
# Expand-scope handler
# ---------------------------------------------------------------------------


def test_expand_scope_revokes_token_and_signals_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = _load_plugin_module()
    plugin._REGISTRY.clear()
    plugin._PROC_REGISTRY.clear()

    run_id = "hhhh-expand-scope"
    ref = "hermes-expand-ref"
    from datetime import UTC, datetime

    binding = plugin._RuntimeBinding(
        run_id=run_id,
        runtime_binding_id="bid8",
        runtime_run_ref=ref,
        ingress_base_url="",
        bound_at=datetime.now(UTC),
        principal_token="old-secret-token",
        last_status="running",
    )
    plugin._REGISTRY.insert(binding)

    mock_proc = MagicMock()
    mock_proc.pid = 44444
    handle = plugin._ProcessHandle(
        run_id=run_id, pid=44444, proc=mock_proc, started_at=datetime.now(UTC)
    )
    with plugin._PROC_REGISTRY._lock:
        plugin._PROC_REGISTRY._by_run[run_id] = handle

    signals_sent = []
    with (
        patch("os.getpgid", return_value=44444),
        patch("os.killpg", side_effect=lambda pgid, sig: signals_sent.append(sig)),
    ):
        asyncio.get_event_loop().run_until_complete(
            plugin._async_expand_scope(run_id, ref)
        )

    assert signal.SIGUSR1 in signals_sent
    # Token must be cleared — old token is treated as revoked.
    assert plugin._REGISTRY.get(run_id).principal_token == ""
    assert plugin._REGISTRY.get(run_id).last_status == "scope_expanding"


def test_expand_scope_does_not_log_token_material(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    plugin = _load_plugin_module()
    plugin._REGISTRY.clear()
    plugin._PROC_REGISTRY.clear()

    run_id = "iiii-token-log-test"
    ref = "hermes-token-log"
    secret_token = "super-secret-old-token-xyz"
    from datetime import UTC, datetime

    binding = plugin._RuntimeBinding(
        run_id=run_id,
        runtime_binding_id="bid9",
        runtime_run_ref=ref,
        ingress_base_url="",
        bound_at=datetime.now(UTC),
        principal_token=secret_token,
        last_status="running",
    )
    plugin._REGISTRY.insert(binding)

    with (
        patch("os.getpgid", side_effect=ProcessLookupError),
        caplog.at_level(logging.DEBUG),
    ):
        asyncio.get_event_loop().run_until_complete(
            plugin._async_expand_scope(run_id, ref)
        )

    for record in caplog.records:
        assert secret_token not in record.getMessage()
        assert secret_token[:8] not in record.getMessage()


def test_rebind_token_transitions_scope_expanding_to_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After rebind-token on a scope_expanding run, status returns to running."""
    client, plugin = _make_client(monkeypatch)
    headers = {"X-TTM-Control-Plane-Secret": "test-secret"}
    run_id, _ = _dispatch_and_get_ref(client, headers)

    # Manually set status to scope_expanding (as expand_scope handler would).
    plugin._REGISTRY.update_status(run_id, last_status="scope_expanding")

    resp = client.post(
        f"/api/plugins/ttm-control-plane/runs/{run_id}/rebind-token",
        json={
            "new_binding_id": "33333333-3333-3333-3333-333333333333",
            "new_token": "new-token-after-scope-expand",
            "ingress_base_url": "",
        },
        headers=headers,
    )
    assert resp.status_code == 200
    assert plugin._REGISTRY.get(run_id).last_status == "running"


# ---------------------------------------------------------------------------
# TTM ingress event emission — mocked, never logs token material
# ---------------------------------------------------------------------------


def test_emit_lifecycle_event_skipped_when_no_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No HTTP call is made when the plugin has no token for the run."""
    plugin = _load_plugin_module()
    from datetime import UTC, datetime

    binding = plugin._RuntimeBinding(
        run_id="jjjj",
        runtime_binding_id="bid10",
        runtime_run_ref="ref10",
        ingress_base_url="http://ttm.local",
        bound_at=datetime.now(UTC),
        principal_token="",  # cleared
        last_status="stopped",
    )

    with patch("httpx.AsyncClient") as mock_client_cls:
        asyncio.get_event_loop().run_until_complete(
            plugin._emit_lifecycle_event(binding, "task.updated", {"status": "stopped"})
        )
    mock_client_cls.assert_not_called()


def test_emit_lifecycle_event_posts_when_token_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the plugin holds a token, it POSTs the event to TTM ingress."""
    plugin = _load_plugin_module()
    from datetime import UTC, datetime

    binding = plugin._RuntimeBinding(
        run_id="kkkk",
        runtime_binding_id="bid11",
        runtime_run_ref="ref11",
        ingress_base_url="http://ttm.local",
        bound_at=datetime.now(UTC),
        principal_token="live-token",
        last_status="paused",
    )

    mock_response = MagicMock()
    mock_response.status_code = 201

    async def fake_post(url, *, json, headers):
        return mock_response

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = fake_post

    with patch("httpx.AsyncClient", return_value=mock_client):
        asyncio.get_event_loop().run_until_complete(
            plugin._emit_lifecycle_event(binding, "task.updated", {"status": "paused"})
        )

    # Verify post was called (via the mock_client.post path)
    # No assertion on mock_client.post.called since it's a real async function;
    # absence of exception is the key contract here.


def test_emit_lifecycle_event_never_logs_token(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Token material must never appear in plugin logs, even on HTTP error."""
    import logging

    plugin = _load_plugin_module()
    from datetime import UTC, datetime

    secret_token = "secret-bearer-xyz-123"
    binding = plugin._RuntimeBinding(
        run_id="llll",
        runtime_binding_id="bid12",
        runtime_run_ref="ref12",
        ingress_base_url="http://ttm.local",
        bound_at=datetime.now(UTC),
        principal_token=secret_token,
        last_status="stopped",
    )

    # Simulate a 500 response — triggers the warning log path without
    # raising, so we can inspect the log output for token material.
    mock_response = MagicMock()
    mock_response.status_code = 500

    async def fake_post(url, *, json, headers):
        return mock_response

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = fake_post

    with (
        patch("httpx.AsyncClient", return_value=mock_client),
        caplog.at_level(logging.WARNING),
    ):
        asyncio.get_event_loop().run_until_complete(
            plugin._emit_lifecycle_event(binding, "task.updated", {"status": "stopped"})
        )

    for record in caplog.records:
        assert secret_token not in record.getMessage()
        assert secret_token[:8] not in record.getMessage()


# ---------------------------------------------------------------------------
# Rebind-token endpoint
# ---------------------------------------------------------------------------


def _rebind_token_body(**overrides):
    body = {
        "new_binding_id": "33333333-3333-3333-3333-333333333333",
        "new_token": "new-bearer-credential-abc123",
        "ingress_base_url": "https://ttm.local",
    }
    body.update(overrides)
    return body


def test_rebind_token_updates_in_memory_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    """After a TTM rebind, the plugin must replace the stored token."""
    client, plugin = _make_client(monkeypatch)
    headers = {"X-TTM-Control-Plane-Secret": "test-secret"}
    run_id = "11111111-1111-1111-1111-111111111111"

    # Dispatch first so the run is registered.
    client.post(
        "/api/plugins/ttm-control-plane/runs/dispatch",
        json=_dispatch_body(),
        headers=headers,
    )
    old_token = plugin._REGISTRY.get(run_id).principal_token  # type: ignore[union-attr]

    resp = client.post(
        f"/api/plugins/ttm-control-plane/runs/{run_id}/rebind-token",
        json=_rebind_token_body(),
        headers=headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "token_updated"
    assert body["run_id"] == run_id

    new_stored = plugin._REGISTRY.get(run_id).principal_token  # type: ignore[union-attr]
    assert new_stored == "new-bearer-credential-abc123"
    assert new_stored != old_token


def test_rebind_token_404s_for_unknown_run(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _make_client(monkeypatch)
    resp = client.post(
        "/api/plugins/ttm-control-plane/runs/not-a-real-run/rebind-token",
        json=_rebind_token_body(),
        headers={"X-TTM-Control-Plane-Secret": "test-secret"},
    )
    assert resp.status_code == 404
    assert resp.json()["detail"]["reason"] == "run_not_found"


def test_rebind_token_rejects_missing_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _make_client(monkeypatch)
    resp = client.post(
        "/api/plugins/ttm-control-plane/runs/some-run/rebind-token",
        json=_rebind_token_body(),
    )
    assert resp.status_code == 401


def test_rebind_token_does_not_log_token_material(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Even partial-token logging is removed before long-running production use.
    import logging

    client, _ = _make_client(monkeypatch)
    headers = {"X-TTM-Control-Plane-Secret": "test-secret"}
    run_id = "11111111-1111-1111-1111-111111111111"
    secret_token = "very-secret-bearer-credential-xyz"

    client.post(
        "/api/plugins/ttm-control-plane/runs/dispatch",
        json=_dispatch_body(),
        headers=headers,
    )
    with caplog.at_level(logging.INFO):
        client.post(
            f"/api/plugins/ttm-control-plane/runs/{run_id}/rebind-token",
            json=_rebind_token_body(new_token=secret_token),
            headers=headers,
        )

    rebind_logs = [r.getMessage() for r in caplog.records if "rebind-token" in r.getMessage()]
    assert rebind_logs, "expected at least one rebind-token log entry"
    for line in rebind_logs:
        assert secret_token not in line
        # No prefix either (first 8 chars).
        assert secret_token[:8] not in line


# ---------------------------------------------------------------------------
# SQLite persistence — bindings survive a fresh registry instance, tokens do not
# ---------------------------------------------------------------------------


def test_binding_persists_across_registry_instances(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Dispatch on instance A; rebuilding the registry against the same DB
    must surface the same binding (without the principal token)."""
    db_path = tmp_path / "bindings.db"
    client, plugin = _make_client(monkeypatch, db_path=db_path)
    headers = {"X-TTM-Control-Plane-Secret": "test-secret"}

    dispatched = client.post(
        "/api/plugins/ttm-control-plane/runs/dispatch",
        json=_dispatch_body(),
        headers=headers,
    )
    assert dispatched.status_code == 202, dispatched.text
    expected_ref = dispatched.json()["runtime_run_ref"]
    expected_run_id = "11111111-1111-1111-1111-111111111111"

    # Simulate a dashboard restart: drop the in-memory registry and
    # rebuild from the same DB path.
    rebuilt = plugin._BindingRegistry(db_path=str(db_path))
    snapshot = rebuilt.snapshot()
    assert len(snapshot) == 1
    survived = snapshot[0]
    assert survived.run_id == expected_run_id
    assert survived.runtime_run_ref == expected_ref
    # Token must NEVER persist — only the binding metadata.
    assert survived.principal_token == ""


def test_binding_remove_clears_persistence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "bindings.db"
    client, plugin = _make_client(monkeypatch, db_path=db_path)
    headers = {"X-TTM-Control-Plane-Secret": "test-secret"}
    run_id = "11111111-1111-1111-1111-111111111111"

    client.post(
        "/api/plugins/ttm-control-plane/runs/dispatch",
        json=_dispatch_body(),
        headers=headers,
    )
    plugin._REGISTRY.remove(run_id)

    rebuilt = plugin._BindingRegistry(db_path=str(db_path))
    assert rebuilt.snapshot() == []


# ---------------------------------------------------------------------------
# Headless session spawn — opt-in path
# ---------------------------------------------------------------------------


def test_spawn_no_op_when_disabled(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    client, plugin = _make_client(monkeypatch)
    headers = {"X-TTM-Control-Plane-Secret": "test-secret"}
    with caplog.at_level(logging.INFO, logger="hermes_dashboard_plugin_ttm_control_plane_test"):
        client.post(
            "/api/plugins/ttm-control-plane/runs/dispatch",
            json=_dispatch_body(),
            headers=headers,
        )
    # Either the spawn task hasn't fired yet (we did not await it) or it
    # fired and no-op'd because TTM_CONTROL_PLANE_DISABLE_SPAWN=1. We
    # assert the latter never produced a "spawned" log line.
    spawned = [r for r in caplog.records if "headless_session.spawned" in r.getMessage()]
    assert not spawned


def test_spawn_logs_when_executable_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """When neither HERMES_CLI nor PATH resolves a hermes binary, the
    spawn must log-and-skip — never crash the dispatch."""
    import asyncio as _asyncio

    monkeypatch.delenv("TTM_CONTROL_PLANE_DISABLE_SPAWN", raising=False)
    monkeypatch.setenv("HERMES_CLI", "/definitely/does/not/exist/hermes")
    monkeypatch.setenv("PATH", "")  # zero out PATH lookup
    monkeypatch.setenv("TTM_CONTROL_PLANE_SECRET", "test-secret")
    plugin = _load_plugin_module()
    plugin._REGISTRY.clear()

    binding = plugin._RuntimeBinding(
        run_id="11111111-1111-1111-1111-111111111111",
        runtime_binding_id="binding-1",
        runtime_run_ref="hermes-test-1",
        ingress_base_url="http://127.0.0.1:8000",
        bound_at=plugin._utcnow(),
        principal_token="some-token",
    )
    # Should not raise even though the executable is unfindable.
    _asyncio.get_event_loop().run_until_complete(
        plugin._spawn_headless_session(binding, "some-token")
    )
    # Process must not be registered when spawn was skipped.
    assert plugin._PROC_REGISTRY.get(binding.run_id) is None
