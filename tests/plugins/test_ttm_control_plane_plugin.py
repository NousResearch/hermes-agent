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
* ``/runs/{ref}/status`` and ``/runs/{ref}/stop`` round-trip
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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


def _make_client(monkeypatch: pytest.MonkeyPatch, *, secret: str | None = "test-secret"):
    """Build a fresh FastAPI app with the plugin mounted under
    ``/api/plugins/ttm-control-plane`` and the registry cleared.

    ``monkeypatch`` sets the env var so each test sees the auth model it
    expects without leaking state across tests.
    """
    if secret is None:
        monkeypatch.delenv("TTM_CONTROL_PLANE_SECRET", raising=False)
    else:
        monkeypatch.setenv("TTM_CONTROL_PLANE_SECRET", secret)
    plugin = _load_plugin_module()
    # Each test gets a clean registry — the singleton persists across
    # imports within a single process, so we wipe it here.
    plugin._REGISTRY._by_run.clear()  # type: ignore[attr-defined]
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


def test_stop_removes_binding_and_allows_rebind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = _make_client(monkeypatch)
    headers = {"X-TTM-Control-Plane-Secret": "test-secret"}
    first = client.post(
        "/api/plugins/ttm-control-plane/runs/dispatch",
        json=_dispatch_body(),
        headers=headers,
    )
    ref = first.json()["runtime_run_ref"]

    stopped = client.post(
        f"/api/plugins/ttm-control-plane/runs/{ref}/stop",
        headers=headers,
    )
    assert stopped.status_code == 202

    # After stop, a re-dispatch should succeed (same run_id, fresh ref).
    rebind = client.post(
        "/api/plugins/ttm-control-plane/runs/dispatch",
        json=_dispatch_body(),
        headers=headers,
    )
    assert rebind.status_code == 202
    assert rebind.json()["runtime_run_ref"] != ref


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
