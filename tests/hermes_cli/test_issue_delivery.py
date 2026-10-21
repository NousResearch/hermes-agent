"""BRIDGE-PROD-001: bounded admission, real local worker, and durable result proof.

REST is a fake transport. The board, task claim, subprocess, owner lease, artifacts,
completion, and crash/retry outbox are real. This is not a production E2E claim.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import active_sessions as sessions, issue_ingress as ingress
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_dispatch as dispatch

_native_executor = ingress._issue22_executor


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb.create_board("missions")
    cfg = tmp_path / "source.json"
    cfg.write_text(json.dumps({
        "app_id": 5343540, "installation_id": 172036721,
        "app_private_key": "BEGIN PRIVATE KEY fixture only", "owner": "ISVGitHub", "repo": "autonomous-entity",
        "issue_ingress_repository_id": 1165823465, "issue_ingress_app_id": 5343540,
        "issue_ingress_installation_id": 172036721, "issue_ingress_actor_id": 548464,
        "issue_ingress_issue_number": 27, "issue_ingress_max_body_bytes": 8192,
        "issue_ingress_result_files": ["CANARY_RESULT.json"],
        "issue_ingress_executor_profile": "canary",
    }))
    now = int(time.time())
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
    body = "Compute the sum of 17 and 25 and record the result in CANARY_RESULT.json.\n" + "Full body preserved. " * 80
    comment = {"id": 81, "created_at": stamp, "updated_at": stamp,
               "user": {"id": 548464, "type": "User"}, "body": json.dumps({
                   "kind": "TASK", "nonce": "canary_nonce_123456", "expires_at": now + 300, "task": body})}
    comments = [comment]

    class Transport:
        def __init__(self, cfg):
            self.cfg = cfg

        def fetch_installation_token(self):
            return SimpleNamespace(permissions={"issues": "write"})

        def get_repository(self, *_args):
            return {"id": 1165823465, "full_name": "ISVGitHub/autonomous-entity"}

        def get_issue(self, _owner, _repo, number):
            return {"number": number, "state": "open"}

        def list_issue_comments(self, _owner, _repo, number, **_kw):
            assert number == 27
            return comments.copy()

        def verify_installation(self):
            return {"app_id": 5343540, "app_slug": "hermes-bridge"}

        def post_issue_comment(self, _owner, _repo, number, text):
            assert number == 27
            row = {"id": 900 + len(comments), "body": text,
                   "user": {"login": "hermes-bridge[bot]", "type": "Bot"}}
            comments.append(row)
            return row

    monkeypatch.setattr(ingress, "GitHubAppAuth", Transport)
    monkeypatch.setattr("hermes_cli.issue22_gateway_verb.GitHubAppAuth", Transport)
    monkeypatch.setattr("hermes_cli.profiles.resolve_profile_env", lambda _name: str(home))
    from hermes_state import SessionDB
    db = SessionDB(home / "state.db"); db.close()
    return SimpleNamespace(home=home, cfg=cfg, now=now, comment=comment, comments=comments, body=body)


@pytest.mark.parametrize("mutation,reason", [
    ("actor", "REST_ACTOR_PIN_MISMATCH"), ("edited", "EDITED_COMMENT_REFUSED"),
    ("expired", "EXPIRED_OR_EXCESSIVE_TTL"), ("duplicate", "DUPLICATE_JSON_KEY"),
    ("repo", "REST_REPOSITORY_ID_MISMATCH"), ("app", "APP_PIN_MISMATCH"),
])
def test_authenticated_admission_rejects_changed_identity_or_replay(bridge, mutation, reason):
    cfg = json.loads(bridge.cfg.read_text())
    expired = {**json.loads(bridge.comment["body"]), "expires_at": bridge.now}
    mutations = {
        "actor": (bridge.comment["user"], "id", 999),
        "edited": (bridge.comment, "updated_at", "2026-01-01T00:00:00Z"),
        "expired": (bridge.comment, "body", json.dumps(expired)),
        "duplicate": (bridge.comment, "body", bridge.comment["body"].replace('"kind": "TASK"', '"kind":"TASK","kind":"PING"')),
        "repo": (cfg, "issue_ingress_repository_id", 999),
        "app": (cfg, "issue_ingress_app_id", 999),
    }
    target, key, value = mutations[mutation]
    target[key] = value
    if mutation in {"repo", "app"}:
        bridge.cfg.write_text(json.dumps(cfg))
    with pytest.raises(ingress.AdmissionRefused, match=reason):
        ingress.read_authenticated_intent(bridge.cfg, comment_id=81, now=bridge.now)


def test_nonce_cannot_change_kind(bridge):
    conn = kbc.connect(board="missions")
    intent = ingress.read_authenticated_intent(bridge.cfg, comment_id=81, now=bridge.now)
    ingress.admit_intent(conn, intent=intent, now=bridge.now)
    args = dict(vars(intent)); args.pop("task_text"); args["kind"] = "PING"
    changed = ingress.AuthenticatedIntent(ingress._ISSUE_SEAL, **args, task_text=None)
    with pytest.raises(ingress.AdmissionRefused, match="NONCE_REUSED_ACROSS_KINDS"):
        ingress.admit_intent(conn, intent=changed, now=bridge.now)
    conn.close()


def test_source_task_uses_canonical_owner_session_for_artifacts_and_result(bridge, monkeypatch):
    from hermes_cli.issue_source_handlers import build_source_handlers
    _admit, poll = build_source_handlers(lambda: kbc.connect(board="missions"), bridge.cfg, enabled=True)
    queued = poll({})
    assert queued["verdict"] == "ACK" and queued["state"] == "queued"
    conn = kbc.connect(board="missions")
    ingress.admit_github_comment(conn, config_path=bridge.cfg, comment_id=81, now=bridge.now)
    task_id = queued["task_id"]
    task = kb.get_task(conn, task_id)
    assert task.session_id is None
    assert poll({})["verdict"] == "BUSY"
    assert task.body == bridge.body
    assert bridge.body.strip() in kb.build_worker_context(conn, task_id)
    assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1
    monkeypatch.setattr(dispatch, "_profile_exists_fn", lambda: None)
    dispatch.dispatch_once(
        conn, board="missions",
        spawn_fn=lambda _task, _workspace, board=None: os.getpid(),
    )
    task = kb.get_task(conn, task_id)
    assert task and task.current_run_id
    run_id = task.current_run_id
    session_id = "canonical-owner-session"
    request_id = "kanban:" + json.dumps(["missions", task_id, run_id], separators=(",", ":"))
    with kb.write_txn(conn):
        kb._append_event(conn, task_id, "owner_admitted", {
            "db": str(kb.kanban_db_path(board="missions")), "session_id": session_id,
            "request_id": request_id, "pid": 123, "birth": 456,
        }, run_id=run_id)
    output = Path(task.workspace_path) / "CANARY_RESULT.json"
    output.write_text(json.dumps({"task_id": task_id, "session_id": session_id,
                                  "source_comment_id": 81, "result": 17 + 25}))
    with pytest.raises(ValueError, match="artifact set"):
        kb.complete_task(conn, task_id, result="Diagnostic placeholder", expected_run_id=run_id,
                         metadata={"artifacts": []}, fire_lifecycle_hook=False)
    assert kb.complete_task(conn, task_id, result="Computed 42", expected_run_id=run_id,
                            metadata={"artifacts": [str(output)]}, fire_lifecycle_hook=False)
    assert task.session_id is None
    original = ingress._owner_write_txn
    from contextlib import contextmanager

    @contextmanager
    def fail_commit(db):
        with original(db):
            yield
            raise RuntimeError("simulated commit loss after GitHub POST")

    monkeypatch.setattr(ingress, "_owner_write_txn", fail_commit)
    with pytest.raises(RuntimeError, match="commit loss"):
        ingress.publish_pending_task_result(conn, config_path=bridge.cfg)
    assert len(bridge.comments) == 3
    monkeypatch.setattr(ingress, "_owner_write_txn", original)
    assert ingress.publish_pending_task_result(conn, config_path=bridge.cfg)["state"] == "result_published"
    assert len(bridge.comments) == 3
    assert ingress.publish_pending_task_result(conn, config_path=bridge.cfg) is None
    assert str(task_id) in bridge.comments[-1]["body"]
    assert "CANARY_RESULT.json" in bridge.comments[-1]["body"]
    assert "canary_result" in bridge.comments[-1]["body"] and "42" in bridge.comments[-1]["body"]
    assert "Full body preserved" not in bridge.comments[-1]["body"]
    attached = kb.list_attachments(conn, task_id)
    assert any(json.loads(Path(item.stored_path).read_text()).get("result") == 42 for item in attached if item.filename.endswith(".json"))
    print("BRIDGE_TRACE=" + json.dumps({"scope": "ISOLATED_FAKE_REST_REAL_SUBPROCESS_NO_LLM", "issue_number": 27,
        "source_comment_id": 81, "task_id": task_id, "session_id": session_id,
        "useful_result": 42,
        "events": [dict(row) for row in conn.execute("SELECT id,run_id,kind FROM task_events WHERE task_id=? ORDER BY id", (task_id,))],
        "result_comment_id": bridge.comments[-1]["id"], "post_count": len(bridge.comments) - 1}))
    conn.close()


@pytest.mark.platforms("linux")
def test_lock_is_exclusive_across_processes_and_refuses_replaced_inode(tmp_path):
    path = tmp_path / "owner.lock"
    with sessions._FileLock(path) as owner:
        code = "from hermes_cli.active_sessions import _FileLock; from pathlib import Path; import sys;\nwith _FileLock(Path(sys.argv[1]),timeout=0.1): pass"
        probe = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True, text=True, timeout=5, check=False)
        assert probe.returncode != 0
        assert "lock busy" in probe.stderr
        path.rename(tmp_path / "replaced.lock")
        path.touch()
        with pytest.raises(sessions.ActiveSessionRegistryError, match="inode changed"):
            owner.assert_current()
    with pytest.raises(sessions.ActiveSessionRegistryError, match="not owned"):
        owner.assert_current()


@pytest.mark.platforms("linux")
def test_inherited_lock_does_not_confer_process_ownership(tmp_path):
    with sessions._FileLock(tmp_path / "owner.lock") as owner:
        child = os.fork()
        if child == 0:
            try:
                owner.assert_current()
            except sessions.ActiveSessionRegistryError:
                os._exit(0)
            os._exit(1)
        _, status = os.waitpid(child, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        owner.assert_current()


def test_source_selection_rejects_unconfigured_issue_before_database_access(bridge):
    from hermes_cli.issue_source_handlers import build_source_handlers
    def unexpected_db():
        pytest.fail("unconfigured source must not open the owner database")
    admit, _poll = build_source_handlers(unexpected_db, bridge.cfg, enabled=True)
    assert admit({"issue_number": 22, "comment_id": 81})["reason"] == "SOURCE_NOT_ALLOWED"
    assert admit({"issue_number": True, "comment_id": 81})["reason"] == "SOURCE_NOT_ALLOWED"


def test_admission_owner_fence_detects_replaced_registry_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    lease, refusal = sessions.try_acquire_active_session(session_id="owner-session", surface="cli", config={})
    assert lease is not None and not refusal
    try:
        with pytest.raises(sessions.ActiveSessionRegistryError, match="inode changed"):
            with sessions.session_owner_fence("owner-session") as fence:
                path = sessions._lock_path()
                path.rename(path.with_suffix(".old"))
                path.touch()
                fence.assert_current()
    finally:
        lease.release()


def test_native_executor_resolves_profile_without_creating_a_session(bridge, monkeypatch, tmp_path):
    profile_home = tmp_path / "profile"; profile_home.mkdir()
    cfg = json.loads(bridge.cfg.read_text()); cfg["issue_ingress_executor_profile"] = "canary"
    bridge.cfg.write_text(json.dumps(cfg))
    monkeypatch.setattr("hermes_cli.profiles.resolve_profile_env", lambda _name: str(profile_home))
    assert _native_executor(bridge.cfg) == "canary"
    assert not (profile_home / "state.db").exists()


def test_result_marker_from_wrong_bot_is_not_an_ack(bridge):
    cfg, pins, _repo, _issue, auth = ingress._fetch_issue(bridge.cfg)
    bridge.comments.append({"id": 987, "body": "exact body <!-- stable -->",
                            "user": {"login": "other-bot[bot]", "type": "Bot"}})
    from hermes_cli.issue_result import publish_verified_comment
    with pytest.raises(ingress.AdmissionRefused, match="RESULT_MARKER_UNTRUSTED"):
        publish_verified_comment(auth, cfg, pins, "<!-- stable -->", "exact body <!-- stable -->")


@pytest.mark.platforms("linux")
def test_two_admission_processes_create_only_one_task_and_binding(bridge):
    import multiprocessing
    conn = kbc.connect(board="missions")
    ingress.admit_github_comment(conn, config_path=bridge.cfg, comment_id=81, now=bridge.now)
    conn.close()
    context = multiprocessing.get_context("fork")
    start, results = context.Event(), context.Queue()

    def bind_once():
        start.wait(5)
        db = kbc.connect(board="missions")
        try:
            results.put(ingress.bind_next_pending_task(db, config_path=bridge.cfg, now=bridge.now))
        except ingress.AdmissionRefused as exc:
            results.put({"reason": str(exc)})
        finally:
            db.close()

    children = [context.Process(target=bind_once) for _ in range(2)]
    for child in children:
        child.start()
    start.set()
    for child in children:
        child.join(10)
        assert child.exitcode == 0
    outcomes = [results.get(timeout=5) for _ in children]
    assert any(row and row.get("state") == "queued" for row in outcomes)
    db = kbc.connect(board="missions")
    assert db.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1
    assert db.execute("SELECT COUNT(*) FROM task_events WHERE kind='github_task_binding'").fetchone()[0] == 1
    db.close()


@pytest.mark.parametrize("sources", [{"*": "/tmp/source.json"}, {"28": "/tmp/source.json"}, {"27": "relative.json"}])
def test_source_policy_never_accepts_wildcards_or_relative_paths(bridge, sources):
    from github_bridge.config import load_config
    from github_bridge.errors import ConfigInvalid
    cfg = json.loads(bridge.cfg.read_text()); cfg["issue_ingress_sources"] = sources
    bridge.cfg.write_text(json.dumps(cfg))
    with pytest.raises(ConfigInvalid, match="explicit 22/27 absolute"):
        load_config(bridge.cfg)
