"""Native artifact-review completion contract and dependency invariants."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from hermes_cli import config as hc
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect


VERIFIER = r'''import json, sqlite3, sys
request = json.load(sys.stdin)
authority = request["authority"]
candidate = request["candidate"]
if request.get("action") == "verify_non_acceptance_disposition":
    json.dump({
        "schema": "hermes.non-acceptance-disposition/v1",
        "accepted": len(candidate.get("reason", "")) >= 20,
        "bindings": {key: authority[key] for key in (
            "task_id", "run_id", "completion_contract")},
        "actor": "trusted-test-authority",
    }, sys.stdout)
    raise SystemExit(0)
hashes = [item["sha256"] for item in authority["artifacts"]]
if candidate.get("race_db"):
    db = sqlite3.connect(candidate["race_db"])
    db.execute("UPDATE tasks SET current_run_id=current_run_id+1 WHERE id=?", (authority["task_id"],))
    db.commit()
    db.close()
evidence_hashes = hashes[:1] if candidate.get("omit_second_evidence") else hashes
evidence = lambda scope: [{
    "scope": scope,
    "source_revision": candidate["source_revision"],
    "environment": candidate["environment"],
    "artifact_sha256": sha,
    "status": "passed",
} for sha in evidence_hashes]
json.dump({
    "schema": "hermes.artifact-review-acceptance/v1",
    "accepted": candidate.get("approve") is True and candidate.get("reviewed_sha256") == hashes,
    "verdict": "accepted",
    "bindings": {key: authority[key] for key in (
        "task_id", "run_id", "completion_contract", "owner", "reviewer",
        "review_profile", "worker_session_id", "source_revision")},
    "artifact_sha256": hashes,
    "independent_review": {
        "independent": True,
        "reviewer": authority["reviewer"],
        "profile": authority["review_profile"],
        "session_id": authority["worker_session_id"],
        "credential_fingerprint": authority["runtime_observation"]["credential_fingerprint"],
    },
    "evidence": {
        "tests": evidence(candidate.get("test_scope", "targeted-worker")),
        "integration": evidence(candidate.get("integration_scope", "combined-revision")),
    },
    "usage": candidate.get("usage", {"applicable": False, "reason": "no model or paid service used"}),
}, sys.stdout)
'''

SOURCE_REVISION = "workspace-sha256:" + "1" * 64


def _observation(profile: str, session_id: str, principal: str) -> dict:
    return {
        "runtime": "hermes-native:anthropic_messages",
        "provider": "anthropic",
        "model": "claude-test-model",
        "profile": profile,
        "session_id": session_id,
        "auth_class": "api_key",
        "credential_fingerprint": f"credential-sha256:{principal * 64}",
        "billing_mode": "metered",
        "source_revision": SOURCE_REVISION,
        "usage_basis": "hermes-session-model-usage/v1",
        "tokens": {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "reasoning_tokens": 0,
            "api_calls": 1,
        },
    }


def _record_observation(conn, task_id: str, run_id: int, observation: dict) -> bool:
    """Seed an observed receipt directly in the isolated test database."""
    with kb.write_txn(conn):
        row = conn.execute(
            "SELECT metadata FROM task_runs WHERE id=? AND task_id=?",
            (run_id, task_id),
        ).fetchone()
        metadata = json.loads(row["metadata"]) if row and row["metadata"] else {}
        prior = metadata.get("runtime_observation")
        if prior is not None:
            return prior == observation
        metadata["runtime_observation"] = observation
        return conn.execute(
            "UPDATE task_runs SET metadata=? WHERE id=? AND task_id=?",
            (json.dumps(metadata), run_id, task_id),
        ).rowcount == 1


@pytest.fixture
def board(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    managed = tmp_path / "managed"
    managed.mkdir(mode=0o755)
    verifier = managed / "verifier.py"
    python = str(Path(sys.executable).resolve())
    verifier.write_text(f"#!{python}\n" + VERIFIER, encoding="utf-8")
    verifier.chmod(0o555)
    (managed / "config.yaml").write_text(
        "kanban:\n"
        "  artifact_review_verifier_command:\n"
        f"    - {json.dumps(str(verifier))}\n"
        "  artifact_review_evidence_policy:\n"
        "    test_scopes: [targeted-worker]\n"
        "    integration_scopes: [combined-revision]\n",
        encoding="utf-8",
    )
    (managed / "config.yaml").chmod(0o444)
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "caller-controlled-decoy"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for profile in ("builder", "reviewer"):
        (home / "profiles" / profile).mkdir(parents=True)
        (home / "profiles" / profile / "config.yaml").write_text("{}\n")
    hc._LOAD_CONFIG_CACHE.clear()
    hc._RAW_CONFIG_CACHE.clear()
    from hermes_cli import managed_scope
    from hermes_cli import kanban_artifact_acceptance as artifact_acceptance
    monkeypatch.setattr(artifact_acceptance, "_TRUSTED_MANAGED_DIR", managed)
    monkeypatch.setattr(artifact_acceptance, "_TRUSTED_OWNER_UID", managed.stat().st_uid)
    managed_scope.invalidate_managed_cache()
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    yield home


def _reviewed_task(conn, *, reviewer: str = "reviewer") -> tuple[str, int, str]:
    task_id = kb.create_task(
        conn, title="Protected deliverable", assignee="builder",
        completion_contract="artifact-review-v1",
    )
    claimed = kb.claim_task(conn, task_id, claimer="builder:1")
    assert claimed is not None
    assert _record_observation(
        conn, task_id, claimed.current_run_id, _observation("builder", "build-session-1", "a"),
    )
    attachment_id = kb.store_attachment_bytes(
        conn, task_id, "deliverable.bin", b"review this exact artifact", uploaded_by="builder",
    )
    artifact = kb.get_attachment(conn, attachment_id)
    assert artifact is not None
    assert kb.request_review(
        conn, task_id, summary="ready", metadata={"artifacts": [artifact.stored_path]},
        reviewer=reviewer, expected_run_id=claimed.current_run_id,
    )
    review = kb.claim_review_task(conn, task_id, claimer="reviewer:1")
    assert review is not None and review.current_run_id is not None
    assert _record_observation(
        conn, task_id, review.current_run_id, _observation("reviewer", "review-session-1", "b"),
    )
    return task_id, review.current_run_id, artifact.stored_path


def _candidate(artifact_path: str, **overrides):
    reviewed_sha = hashlib.sha256(Path(artifact_path).read_bytes()).hexdigest()
    value = {
        "approve": True,
        "reviewed_sha256": [reviewed_sha],
        "source_revision": SOURCE_REVISION,
        "environment": "python:test-env-lock-1",
        "usage": {
            "applicable": True,
            "runtime": "hermes-native:anthropic_messages",
            "provider": "anthropic",
            "model": "claude-test-model",
            "profile": "reviewer",
            "session_id": "review-session-1",
            "auth_class": "api_key",
            "billing_mode": "metered",
            "policy": "review-budget-v1",
            "budget": {"max_tokens": 50000},
            "basis": "hermes-session-model-usage/v1",
            "source_revision": SOURCE_REVISION,
            "artifact_sha256": [reviewed_sha],
            "tokens": {
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "reasoning_tokens": 0,
                "api_calls": 1,
            },
        },
    }
    value.update(overrides)
    return {"artifact_review_receipt": value}


def test_acceptance_binds_review_artifact_and_atomically_releases_child(board):
    with connect() as conn:
        parent, review_run, original = _reviewed_task(conn)
        child = kb.create_task(conn, title="Release", parents=[parent])
        assert kb.get_task(conn, child).status == "todo"

        assert kb.complete_task(
            conn, parent, summary="accepted", metadata=_candidate(original),
            expected_run_id=review_run,
        )
        assert kb.get_task(conn, parent).status == "done"
        assert kb.get_task(conn, child).status == "ready"
        receipts = [event for event in kb.list_events(conn, parent)
                    if event.kind == "artifact_review_acceptance"]
        assert len(receipts) == 1
        receipt = receipts[0].payload
        assert receipt["ok"] is True
        assert receipt["bindings"]["owner"] == "builder"
        assert receipt["bindings"]["reviewer"] == "reviewer"
        assert receipt["evidence"]["tests"][0]["scope"] == "targeted-worker"
        accepted = Path(receipt["artifacts"][0]["cas_path"])
        assert accepted.name == receipt["artifacts"][0]["sha256"]
        assert accepted.read_bytes() == Path(original).read_bytes()
        assert accepted.stat().st_mode & 0o222 == 0
        closing_run = kb.latest_run(conn, parent)
        assert closing_run is not None
        assert "artifact_review_receipt" not in (closing_run.metadata or {})
        assert closing_run.metadata["runtime_observation"]["session_id"] == "review-session-1"
        assert receipt["usage"]["runtime"] == "hermes-native:anthropic_messages"
        assert receipt["usage"]["model"] == "claude-test-model"
        assert receipt["usage"]["auth_class"] == "api_key"
        assert receipt["usage"]["billing_mode"] == "metered"
        assert receipt["usage"]["tokens"]["api_calls"] == 1
        assert not kb.complete_task(conn, parent, metadata=_candidate(original))
        assert len([event for event in kb.list_events(conn, parent)
                    if event.kind == "artifact_review_acceptance"]) == 1


def test_rejection_and_missing_configuration_fail_closed(board):
    with connect() as conn:
        task_id, review_run, artifact = _reviewed_task(conn)
        assert not kb.complete_task(
            conn, task_id, metadata=_candidate(artifact, approve=False), expected_run_id=review_run,
        )
        assert kb.get_task(conn, task_id).status == "running"
        assert "rejected" in (kb.get_task(conn, task_id).last_failure_error or "").lower()

        monkeypatch = pytest.MonkeyPatch()
        from hermes_cli import kanban_artifact_acceptance as artifact_acceptance
        monkeypatch.setattr(artifact_acceptance, "_TRUSTED_MANAGED_DIR", board / "missing-managed")
        assert not kb.complete_task(
            conn, task_id, metadata=_candidate(artifact), expected_run_id=review_run,
        )
        assert "configure" in (kb.get_task(conn, task_id).last_failure_error or "").lower()
        monkeypatch.undo()


def test_caller_managed_override_and_interpreter_script_cannot_select_authority(board, tmp_path):
    from hermes_cli import kanban_artifact_acceptance as artifact_acceptance

    decoy = tmp_path / "caller-controlled-decoy"
    decoy.mkdir()
    (decoy / "config.yaml").write_text("kanban: {}\n", encoding="utf-8")
    # The inherited override is ignored; the bootstrap-pinned fixture remains active.
    assert artifact_acceptance._configured_command() is not None

    trusted_config = artifact_acceptance._TRUSTED_MANAGED_DIR / "config.yaml"
    trusted_config.chmod(0o644)
    trusted_config.write_text(
        "kanban:\n"
        "  artifact_review_verifier_command:\n"
        f"    - {json.dumps(str(Path(sys.executable).resolve()))}\n"
        f"    - {json.dumps(str(decoy / 'writable.py'))}\n"
        "  artifact_review_evidence_policy:\n"
        "    test_scopes: [targeted-worker]\n"
        "    integration_scopes: [combined-revision]\n",
        encoding="utf-8",
    )
    trusted_config.chmod(0o444)
    assert artifact_acceptance._configured_command() is None


def test_configured_evidence_scope_policy_is_enforced(board):
    with connect() as conn:
        task_id, review_run, artifact = _reviewed_task(conn)
        assert not kb.complete_task(
            conn, task_id,
            metadata=_candidate(artifact, test_scope="some-unrelated-suite"),
            expected_run_id=review_run,
        )
        assert "scope policy" in (kb.get_task(conn, task_id).last_failure_error or "")


def test_model_review_identity_and_usage_receipt_fail_closed(board):
    with connect() as conn:
        task_id, review_run, artifact = _reviewed_task(conn)

        with kb.write_txn(conn):
            conn.execute("UPDATE task_runs SET metadata=NULL WHERE id=?", (review_run,))
        forged = _candidate(artifact)
        forged["runtime_observation"] = _observation(
            "reviewer", "review-session-1", "b",
        )
        assert not kb.complete_task(
            conn, task_id, metadata=forged, expected_run_id=review_run,
        )
        assert "observed runtime" in (kb.get_task(conn, task_id).last_failure_error or "")
        assert _record_observation(
            conn, task_id, review_run, _observation("reviewer", "review-session-1", "b"),
        )

        wrong_session = _candidate(artifact)
        wrong_session["artifact_review_receipt"]["usage"]["session_id"] = "another-session"
        assert not kb.complete_task(
            conn, task_id, metadata=wrong_session, expected_run_id=review_run,
        )
        assert "session" in (kb.get_task(conn, task_id).last_failure_error or "")

        not_applicable = _candidate(artifact)
        not_applicable["artifact_review_receipt"]["usage"] = {
            "applicable": False,
            "reason": "claimed deterministic review",
        }
        assert not kb.complete_task(
            conn, task_id, metadata=not_applicable, expected_run_id=review_run,
        )
        assert "applicable model usage" in (kb.get_task(conn, task_id).last_failure_error or "")

        no_calls = _candidate(artifact)
        no_calls["artifact_review_receipt"]["usage"]["tokens"]["api_calls"] = 0
        assert not kb.complete_task(
            conn, task_id, metadata=no_calls, expected_run_id=review_run,
        )
        assert "at least one API call" in (kb.get_task(conn, task_id).last_failure_error or "")

        conflated_auth = _candidate(artifact)
        conflated_auth["artifact_review_receipt"]["usage"]["auth_class"] = "subscription_cli"
        conflated_auth["artifact_review_receipt"]["usage"]["runtime"] = "hermes-native-anthropic"
        assert not kb.complete_task(
            conn, task_id, metadata=conflated_auth, expected_run_id=review_run,
        )
        assert "explicit CLI runtime" in (kb.get_task(conn, task_id).last_failure_error or "")

        stale_revision = _candidate(artifact)
        stale_revision["artifact_review_receipt"]["usage"]["source_revision"] = "tree:stale"
        assert not kb.complete_task(
            conn, task_id, metadata=stale_revision, expected_run_id=review_run,
        )
        assert "runtime observation" in (kb.get_task(conn, task_id).last_failure_error or "")

        assert kb.complete_task(
            conn, task_id, metadata=_candidate(artifact), expected_run_id=review_run,
        )


def test_runtime_observation_has_no_public_library_minting_path(board):
    assert not hasattr(kb, "record_run_observation")
    assert not hasattr(kb, "_record_run_observation")
    assert not hasattr(kb, "_RUN_OBSERVATION_CAPABILITY")


def test_runtime_observation_is_immutable_and_principal_revision_bound(board):
    with connect() as conn:
        task_id, review_run, artifact = _reviewed_task(conn)
        changed = _observation("reviewer", "review-session-1", "c")
        assert not _record_observation(conn, task_id, review_run, changed)

        run = kb.get_run(conn, review_run)
        metadata = dict(run.metadata)
        observation = dict(metadata["runtime_observation"])
        observation["source_revision"] = "workspace-sha256:" + "2" * 64
        metadata["runtime_observation"] = observation
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_runs SET metadata=? WHERE id=?", (json.dumps(metadata), review_run),
            )
        assert not kb.complete_task(
            conn, task_id, metadata=_candidate(artifact), expected_run_id=review_run,
        )
        assert "workspace revision" in (kb.get_task(conn, task_id).last_failure_error or "")

        observation["source_revision"] = SOURCE_REVISION
        observation["credential_fingerprint"] = "credential-sha256:" + "a" * 64
        metadata["runtime_observation"] = observation
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_runs SET metadata=? WHERE id=?", (json.dumps(metadata), review_run),
            )
        assert not kb.complete_task(
            conn, task_id, metadata=_candidate(artifact), expected_run_id=review_run,
        )
        assert "same observed runtime credential" in (
            kb.get_task(conn, task_id).last_failure_error or ""
        )


def test_each_evidence_category_must_cover_every_artifact(board):
    with connect() as conn:
        task_id = kb.create_task(
            conn, title="Two artifacts", assignee="builder",
            completion_contract="artifact-review-v1",
        )
        claimed = kb.claim_task(conn, task_id, claimer="builder:multi")
        assert claimed and _record_observation(
            conn, task_id, claimed.current_run_id,
            _observation("builder", "build-session-multi", "a"),
        )
        paths = []
        for name, data in (("a.bin", b"a"), ("b.bin", b"b")):
            aid = kb.store_attachment_bytes(conn, task_id, name, data)
            paths.append(kb.get_attachment(conn, aid).stored_path)
        assert claimed and kb.request_review(
            conn, task_id, metadata={"artifacts": paths}, reviewer="reviewer",
            expected_run_id=claimed.current_run_id,
        )
        review = kb.claim_review_task(conn, task_id, claimer="reviewer:multi")
        assert review and _record_observation(
            conn, task_id, review.current_run_id,
            _observation("reviewer", "review-session-1", "b"),
        )
        candidate = _candidate(paths[0], omit_second_evidence=True)
        candidate["artifact_review_receipt"]["reviewed_sha256"] = [
            hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths
        ]
        assert review and not kb.complete_task(
            conn, task_id, metadata=candidate, expected_run_id=review.current_run_id,
        )
        assert "complete artifact set" in (kb.get_task(conn, task_id).last_failure_error or "")


def test_independent_reviewer_and_archive_disposition_are_enforced(board):
    with connect() as conn:
        task_id = kb.create_task(
            conn, title="Self review", assignee="builder",
            completion_contract="artifact-review-v1",
        )
        claimed = kb.claim_task(conn, task_id, claimer="builder:1")
        attachment_id = kb.store_attachment_bytes(conn, task_id, "x", b"x")
        artifact = kb.get_attachment(conn, attachment_id)
        assert claimed and artifact
        assert _record_observation(
            conn, task_id, claimed.current_run_id,
            _observation("builder", "build-session-self", "a"),
        )
        assert kb.request_review(
            conn, task_id, metadata={"artifacts": [artifact.stored_path]},
            reviewer="builder", expected_run_id=claimed.current_run_id,
        )
        review = kb.claim_review_task(conn, task_id, claimer="builder:2")
        assert review
        assert not kb.complete_task(
            conn, task_id, metadata=_candidate(artifact.stored_path), expected_run_id=review.current_run_id,
        )

        parent = kb.create_task(
            conn, title="Abandoned protected task", completion_contract="artifact-review-v1",
        )
        child = kb.create_task(conn, title="Disposition follower", parents=[parent])
        assert not kb.archive_task(conn, parent)
        assert kb.get_task(conn, child).status == "todo"
        assert kb.archive_task(
            conn, parent,
            non_acceptance_disposition="Operator cancelled this deliverable after scope changed.",
        )
        assert kb.get_task(conn, child).status == "ready"
        event = [event for event in kb.list_events(conn, parent)
                 if event.kind == "non_acceptance_disposition"][-1]
        assert event.payload["actor"] == "trusted-test-authority"


def test_direct_terminal_status_without_receipt_does_not_release_dependency(board):
    with connect() as conn:
        parent = kb.create_task(
            conn, title="Raw writer", completion_contract="artifact-review-v1",
        )
        child = kb.create_task(conn, title="Must remain gated", parents=[parent])
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='done' WHERE id=?", (parent,))
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, child).status == "todo"
        assert kb.unsatisfied_parents(conn, child) == [(parent, "done")]


def test_library_force_cli_worker_and_dashboard_routes_share_the_guard(
    board, monkeypatch: pytest.MonkeyPatch,
):
    from hermes_cli import kanban as kc
    from plugins.kanban.dashboard import plugin_api
    from tools import kanban_tools

    with connect() as conn:
        library = kb.create_task(conn, title="library", completion_contract="artifact-review-v1")
        assert not kb.complete_task(conn, library, summary="force", force=True)

        cli = kb.create_task(conn, title="cli", completion_contract="artifact-review-v1")
        output = kc.run_slash(f"complete {cli} --summary protected")
        assert "cannot complete" in output
        assert kb.get_task(conn, cli).status != "done"

        dashboard = kb.create_task(conn, title="dashboard", completion_contract="artifact-review-v1")
        payload = SimpleNamespace(result=None, summary="protected", metadata=None)
        assert not plugin_api._STATUS_HANDLERS["done"](conn, dashboard, payload)
        assert kb.get_task(conn, dashboard).status != "done"

        worker = kb.create_task(
            conn, title="worker", assignee="builder", completion_contract="artifact-review-v1",
        )
        run = kb.claim_task(conn, worker, claimer="builder:route")
        assert run and run.current_run_id
        monkeypatch.setenv("HERMES_KANBAN_TASK", worker)
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run.current_run_id))
        response = json.loads(kanban_tools._handle_complete({"summary": "protected"}))
        assert response.get("error") or response.get("ok") is False
        assert kb.get_task(conn, worker).status != "done"


def test_reviewed_bytes_and_run_snapshot_cannot_change_during_acceptance(board):
    with connect() as conn:
        task_id, review_run, artifact = _reviewed_task(conn)
        candidate = _candidate(artifact)
        original = Path(artifact).read_bytes()
        Path(artifact).write_bytes(bytes([original[0] ^ 1]) + original[1:])
        assert Path(artifact).stat().st_size == len(original)
        assert not kb.complete_task(
            conn, task_id, metadata=candidate, expected_run_id=review_run,
        )
        assert kb.get_task(conn, task_id).status == "running"

        Path(artifact).write_bytes(original)
        raced = _candidate(artifact, race_db=str(kb.kanban_db_path()))
        assert not kb.complete_task(
            conn, task_id, metadata=raced, expected_run_id=review_run,
        )
        assert kb.get_task(conn, task_id).status == "running"
        assert not [event for event in kb.list_events(conn, task_id)
                    if event.kind == "artifact_review_acceptance" and event.payload.get("ok")]


def test_reopened_acceptance_cannot_be_reused_by_archive(board):
    with connect() as conn:
        parent, review_run, artifact = _reviewed_task(conn)
        child = kb.create_task(conn, title="child", parents=[parent])
        assert kb.complete_task(
            conn, parent, metadata=_candidate(artifact), expected_run_id=review_run,
        )
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='todo', completed_at=NULL WHERE id=?", (parent,),
            )
        kb.invalidate_descendants_for_parent_reopen(conn, parent, author="operator")
        assert kb.get_task(conn, child).status == "todo"
        assert not kb.archive_task(conn, parent)
        assert kb.get_task(conn, child).status == "todo"


def test_parked_protected_review_requires_an_active_review_run(board):
    with connect() as conn:
        parent = kb.create_task(
            conn, title="Parked review", assignee="builder",
            completion_contract="artifact-review-v1",
        )
        claimed = kb.claim_task(conn, parent, claimer="builder:parked")
        assert claimed and _record_observation(
            conn, parent, claimed.current_run_id,
            _observation("builder", "build-session-parked", "a"),
        )
        aid = kb.store_attachment_bytes(conn, parent, "parked.bin", b"parked")
        artifact = kb.get_attachment(conn, aid)
        assert claimed and artifact and kb.request_review(
            conn, parent, metadata={"artifacts": [artifact.stored_path]},
            reviewer="reviewer", expected_run_id=claimed.current_run_id,
        )
        child = kb.create_task(conn, title="still gated", parents=[parent])
        assert kb.get_task(conn, parent).status == "review"
        assert kb.get_task(conn, parent).current_run_id is None
        assert not kb.complete_task(
            conn, parent, metadata=_candidate(artifact.stored_path), force=True,
        )
        assert kb.get_task(conn, parent).status == "review"
        assert kb.get_task(conn, child).status == "todo"
        assert not [event for event in kb.list_events(conn, parent)
                    if event.kind == "artifact_review_acceptance"]
