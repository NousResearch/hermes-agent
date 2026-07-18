"""HTTP contract for the narrow Hermes kanban claim shim."""

from __future__ import annotations

from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _client():
    from starlette.testclient import TestClient

    from hermes_cli.claim_shim import ClaimShimConfig, create_app

    return TestClient(
        create_app(ClaimShimConfig(bearer_token="test-token", claimer="cc-3.5", ttl_seconds=300))
    )


def _auth() -> dict[str, str]:
    return {"Authorization": "Bearer test-token"}


def test_health_is_the_only_unauthenticated_endpoint():
    from starlette.testclient import TestClient

    from hermes_cli.claim_shim import ClaimShimConfig, create_app

    app = create_app(
        ClaimShimConfig(bearer_token="test-token", claimer="cc-3.5", ttl_seconds=300)
    )
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"status": "ok"}
    assert client.get("/docs").status_code == 404
    assert client.patch("/tasks/task-1").status_code == 404


def test_claim_rejects_missing_or_wrong_bearer_before_processing_body():
    from starlette.testclient import TestClient

    from hermes_cli.claim_shim import ClaimShimConfig, create_app

    client = TestClient(
        create_app(ClaimShimConfig(bearer_token="test-token", claimer="cc-3.5", ttl_seconds=300))
    )

    missing = client.post("/claim", content=b"not json")
    wrong = client.post(
        "/claim",
        content=b"not json",
        headers={"Authorization": "Bearer wrong-token"},
    )

    assert missing.status_code == 401
    assert missing.json() == {"error": "unauthorized"}
    assert wrong.status_code == 401
    assert wrong.json() == {"error": "unauthorized"}


def test_claim_uses_fixed_identity_and_ttl(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="Claim through shim", assignee="cc-3.5")

    response = _client().post("/claim", json={"task_id": task_id}, headers=_auth())

    assert response.status_code == 200
    assert response.json() == {"outcome": "claimed", "task_id": task_id}
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.status == "running"
    assert task.claim_lock == "cc-3.5"
    assert 299 <= task.claim_expires - int(time.time()) <= 300


def test_claim_rejects_invalid_json_unknown_fields_and_oversized_bodies():
    client = _client()

    malformed = client.post(
        "/claim", content=b"{", headers={**_auth(), "Content-Type": "application/json"},
    )
    unknown = client.post(
        "/claim", json={"task_id": "task-1", "claimer": "attacker"}, headers=_auth()
    )
    oversized = client.post(
        "/claim", content=b"x" * 4_097, headers={**_auth(), "Content-Type": "application/json"},
    )
    wrong_media_type = client.post(
        "/claim", content=b'{"task_id":"task-1"}', headers=_auth()
    )

    assert malformed.status_code == 400
    assert malformed.json() == {"error": "invalid_json"}
    assert unknown.status_code == 400
    assert unknown.json() == {"error": "invalid_request"}
    assert oversized.status_code == 413
    assert oversized.json() == {"error": "body_too_large"}
    assert wrong_media_type.status_code == 415
    assert wrong_media_type.json() == {"error": "unsupported_media_type"}


def test_claim_distinguishes_idempotent_conflict_and_not_found(kanban_home):
    with kb.connect() as conn:
        mine = kb.create_task(conn, title="Mine", assignee="cc-3.5")
        other = kb.create_task(conn, title="Other", assignee="cc-3.5")
        assert kb.claim_task(conn, other, claimer="different-worker") is not None

    client = _client()
    assert client.post("/claim", json={"task_id": mine}, headers=_auth()).status_code == 200
    retry = client.post("/claim", json={"task_id": mine}, headers=_auth())
    conflict = client.post("/claim", json={"task_id": other}, headers=_auth())
    missing = client.post("/claim", json={"task_id": "no-such-task"}, headers=_auth())

    assert retry.status_code == 200
    assert retry.json() == {"outcome": "already_owned", "task_id": mine}
    assert conflict.status_code == 409
    assert conflict.json() == {"error": "claim_conflict"}
    assert missing.status_code == 404
    assert missing.json() == {"error": "not_found"}


def test_to_review_transitions_owned_task_and_handles_safe_retries(kanban_home):
    first_pr = "https://github.com/acme/widgets/pull/42"
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="Review me", assignee="cc-3.5")
        assert kb.claim_task(conn, task_id, claimer="cc-3.5") is not None
        foreign_task = kb.create_task(conn, title="Foreign", assignee="cc-3.5")
        assert kb.claim_task(conn, foreign_task, claimer="other-worker") is not None

    client = _client()
    body = {"task_id": task_id, "pr_url": first_pr}
    reviewed = client.post("/to-review", json=body, headers=_auth())
    retry = client.post("/to-review", json=body, headers=_auth())
    conflict = client.post(
        "/to-review",
        json={"task_id": task_id, "pr_url": "https://github.com/acme/widgets/pull/99"},
        headers=_auth(),
    )
    foreign = client.post(
        "/to-review",
        json={"task_id": foreign_task, "pr_url": first_pr},
        headers=_auth(),
    )

    assert reviewed.status_code == 200
    assert reviewed.json() == {"outcome": "review_started", "task_id": task_id}
    assert retry.status_code == 200
    assert retry.json() == {"outcome": "already_in_review", "task_id": task_id}
    assert conflict.status_code == 409
    assert conflict.json() == {"error": "review_conflict"}
    assert foreign.status_code == 409
    assert foreign.json() == {"error": "ownership_conflict"}


def test_to_review_requires_auth_and_rejects_non_github_or_unknown_input():
    client = _client()
    body = {"task_id": "task-1", "pr_url": "https://github.com/acme/widgets/pull/42"}

    unauthenticated = client.post("/to-review", json=body)
    unknown = client.post("/to-review", json={**body, "status": "done"}, headers=_auth())
    non_github = client.post(
        "/to-review", json={**body, "pr_url": "https://example.com/pull/42"}, headers=_auth()
    )

    assert unauthenticated.status_code == 401
    assert unauthenticated.json() == {"error": "unauthorized"}
    assert unknown.status_code == 400
    assert unknown.json() == {"error": "invalid_request"}
    assert non_github.status_code == 400
    assert non_github.json() == {"error": "invalid_request"}


def test_server_config_requires_secret_and_explicit_non_wildcard_bind(monkeypatch):
    from hermes_cli.claim_shim import ClaimShimConfig, load_server_config

    monkeypatch.delenv("HERMES_CLAIM_SHIM_BEARER", raising=False)
    monkeypatch.setenv("HERMES_CLAIM_SHIM_BIND", "100.64.0.10")
    with pytest.raises(ValueError, match="bearer"):
        load_server_config()

    monkeypatch.setenv("HERMES_CLAIM_SHIM_BEARER", "server-secret")
    monkeypatch.setenv("HERMES_CLAIM_SHIM_BIND", "0.0.0.0")
    with pytest.raises(ValueError, match="host"):
        load_server_config()

    monkeypatch.setenv("HERMES_CLAIM_SHIM_BIND", "100.64.0.10")
    monkeypatch.setenv("HERMES_CLAIM_SHIM_CLAIMER", "cc-3.5")
    monkeypatch.setenv("HERMES_CLAIM_SHIM_TTL_SECONDS", "600")
    config, host, port = load_server_config()

    assert config == ClaimShimConfig(bearer_token="server-secret", claimer="cc-3.5", ttl_seconds=600)
    assert (host, port) == ("100.64.0.10", 8787)


def test_main_runs_only_the_configured_private_listener(monkeypatch):
    import hermes_cli.claim_shim as shim

    config = shim.ClaimShimConfig("server-secret", "cc-3.5", 600)
    calls = []
    monkeypatch.setattr(shim, "load_server_config", lambda: (config, "100.64.0.10", 8787))
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=lambda app, **kwargs: calls.append((app, kwargs))))

    shim.main()

    assert len(calls) == 1
    assert calls[0][1] == {"host": "100.64.0.10", "port": 8787}
