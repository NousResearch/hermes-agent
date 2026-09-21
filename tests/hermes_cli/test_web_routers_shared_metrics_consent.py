"""PUT /api/telemetry/shared-metrics — the desktop onboarding consent path.

Contract: the route must do exactly what `hermes setup telemetry` does — write both config
keys AND record the send-consent window at the moment of decision. A bare config write
leaves the window to the backend's once-per-process reconcile, so packages collected in
the meantime would never be sent (or, on withdrawal, would still be sent).
"""

from __future__ import annotations

import pytest

from hermes_cli.config import read_raw_config


def _open_consent_windows() -> int:
    from hermes_cli.observability.shared_metrics import SharedMetricsStore

    with SharedMetricsStore()._connection() as connection:
        return connection.execute(
            "SELECT COUNT(*) FROM send_consent_windows WHERE closed_at IS NULL"
        ).fetchone()[0]


@pytest.fixture
def client(_isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return client


def test_opt_in_writes_config_and_opens_consent_window(client):
    assert client.get("/api/telemetry/shared-metrics").json() == {
        "enabled": False, "send": False, "decided": False, "source": "default"}

    resp = client.put("/api/telemetry/shared-metrics", json={"enabled": True, "send": True})

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"ok": True, "enabled": True, "send": True, "decided": True, "source": "profile"}
    shared = read_raw_config()["telemetry"]["shared_metrics"]
    assert (shared["enabled"], shared["send"]) == (True, True)
    assert _open_consent_windows() == 1
    assert client.get("/api/telemetry/shared-metrics").json() == {
        "enabled": True, "send": True, "decided": True, "source": "profile"}


def test_opt_out_closes_consent_window_and_never_leaves_send_without_collection(client):
    client.put("/api/telemetry/shared-metrics", json={"enabled": True, "send": True})

    # send=True with enabled=False is not a valid state: send is forced off.
    resp = client.put("/api/telemetry/shared-metrics", json={"enabled": False, "send": True})

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"ok": True, "enabled": False, "send": False, "decided": True, "source": "profile"}
    shared = read_raw_config()["telemetry"]["shared_metrics"]
    assert (shared["enabled"], shared["send"]) == (False, False)
    assert _open_consent_windows() == 0


def test_non_boolean_body_is_rejected(client):
    resp = client.put("/api/telemetry/shared-metrics", json={"enabled": "yes", "send": 1})

    assert resp.status_code == 422
