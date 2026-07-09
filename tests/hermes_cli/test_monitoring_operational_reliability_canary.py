from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hermes_cli import kanban_db as kb


POLISH_TEXT = "Za\u017c\u00f3\u0142\u0107 g\u0119\u015bl\u0105 ja\u017a\u0144"
PROVIDER_EVIDENCE_FIXTURES = {
    "set_cookie": "Set-Cookie: CF_Authorization=fixture_cf_access_cookie_value; Path=/; Secure; HttpOnly",
    "cookie": "Cookie: CF_Authorization=fixture_cf_cookie_value; session=fixture_session_cookie_value",
    "authorization": "Authorization: Bearer fixture_authorization_bearer_value",
    "cf_signed_redirect": (
        "https://access.example.invalid/cdn-cgi/access/login/google?"
        "kid=fixture-key&redirect_url=https%3A%2F%2Fcrm.example.invalid%2Fdashboard"
        "&sig=fixture-cloudflare-access-signature"
    ),
    "private_provider_redirect": (
        "https://provider.example.invalid/oauth/callback?"
        "private_redirect_url=https%3A%2F%2Fcrm.internal.invalid%2Fadmin"
        "&provider_token=fixture_provider_redirect_token"
    ),
}


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


class FakeSessionDB:
    def __init__(self, records: dict[str, dict]):
        self.records = records

    def get_session(self, session_id: str):
        return self.records.get(session_id)


class FakeWindowsProcess:
    def __init__(
        self,
        pid: int,
        *,
        cmdline: list[str],
        cwd: str,
        create_time: float,
        alive: bool = True,
        stubborn: bool = False,
    ):
        self.pid = pid
        self._cmdline = cmdline
        self._cwd = cwd
        self._create_time = create_time
        self.alive = alive
        self.stubborn = stubborn
        self._children: list[FakeWindowsProcess] = []
        self._parent: FakeWindowsProcess | None = None

    def add_child(self, child: "FakeWindowsProcess") -> "FakeWindowsProcess":
        child._parent = self
        self._children.append(child)
        return child

    def children(self, recursive: bool = False):
        if not recursive:
            return list(self._children)
        out = []
        stack = list(self._children)
        while stack:
            child = stack.pop(0)
            out.append(child)
            stack[0:0] = child._children
        return out

    def cmdline(self):
        return list(self._cmdline)

    def cwd(self):
        return self._cwd

    def create_time(self):
        return self._create_time

    def is_running(self):
        return self.alive

    def status(self):
        return "running" if self.alive else "terminated"

    def mark_taskkill(self):
        if not self.stubborn:
            self.alive = False


def _agent(*, queued_steer_count=0):
    return SimpleNamespace(
        session_id="monitoring-canary-high-context",
        model="gpt-5.5",
        provider="openai-codex",
        platform="cli",
        _pending_steer_status_count=lambda: queued_steer_count,
    )


def _thresholds():
    from agent.request_watchdog import RequestWatchdogThresholds

    return RequestWatchdogThresholds(
        normal_alert_seconds=90.0,
        high_context_alert_seconds=10.0,
        terminal_recovery_seconds=30.0,
        poll_interval_seconds=999.0,
        high_context_tokens=200_000,
    )


def _profile_with_skills(home: Path, name: str, skills: list[str]) -> None:
    profile_dir = home / "profiles" / name
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "config.yaml").write_text("toolsets:\n  - terminal\n", encoding="utf-8")
    for skill in skills:
        skill_dir = profile_dir / "skills" / skill
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {skill}\n---\n# {skill}\n", encoding="utf-8"
        )


def _write_active_session_registry(home: Path) -> None:
    runtime = home / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "active_sessions.json").write_text(
        json.dumps(
            [
                {
                    "lease_id": "lease-1",
                    "session_id": "session-1",
                    "session_key": "session-1",
                    "surface": "cli",
                    "owner_kind": "cli",
                    "pid": 12345,
                    "process_start_time": 1.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    lock_dir = runtime / "active_sessions.lock.d"
    lock_dir.mkdir()
    (lock_dir / "owner.json").write_text(
        json.dumps(
            {
                "pid": 67890,
                "session_id": "metadata-session",
                "surface": "tui",
                "owner_kind": "metadata_update",
                "cwd": "C:/Users/Admin/private project",
            }
        ),
        encoding="utf-8",
    )


def _make_windows_dev_tree(project_dir: str):
    root = FakeWindowsProcess(
        1000,
        cmdline=["C:/Program Files/Git/bin/bash.exe", "-lic", "set +m; npm run dev"],
        cwd=project_dir,
        create_time=100.0,
    )
    cmd = root.add_child(
        FakeWindowsProcess(
            1001,
            cmdline=["C:/Windows/System32/cmd.exe", "/d", "/s", "/c", "npm run dev"],
            cwd=project_dir,
            create_time=101.0,
        )
    )
    npm = cmd.add_child(
        FakeWindowsProcess(
            1002,
            cmdline=["C:/Program Files/nodejs/node.exe", "npm-cli.js", "run", "dev"],
            cwd=project_dir,
            create_time=102.0,
        )
    )
    next_dev = npm.add_child(
        FakeWindowsProcess(
            1003,
            cmdline=["C:/Program Files/nodejs/node.exe", "next", "dev"],
            cwd=project_dir,
            create_time=103.0,
        )
    )
    server = next_dev.add_child(
        FakeWindowsProcess(
            1004,
            cmdline=["C:/Program Files/nodejs/node.exe", "start-server.js"],
            cwd=project_dir,
            create_time=104.0,
            stubborn=True,
        )
    )
    return root, cmd, npm, next_dev, server


def _assert_access_material_redacted(value: object) -> None:
    haystack = json.dumps(value, ensure_ascii=False, sort_keys=True)
    forbidden_fragments = {
        "fixture_cf_access_cookie_value",
        "fixture_session_cookie_value",
        "fixture_authorization_bearer_value",
        "fixture-cloudflare-access-signature",
        "provider_redirect_token",
        "crm.internal.invalid",
        "Set-Cookie",
        "Cookie",
        "Authorization",
    }
    for fragment in forbidden_fragments:
        assert fragment not in haystack


def test_monitoring_operational_reliability_canary(
    kanban_home, tmp_path, monkeypatch, capsys, all_assignees_spawnable
):
    from agent.request_watchdog import (
        poll_request_watchdog,
        start_request_watchdog,
        write_recoverable_turn_state,
    )
    from hermes_cli import active_sessions, runtime_cli
    from hermes_cli.artifact_contracts import required_artifact_guard_action
    from hermes_cli.subprocess_text import decode_subprocess_bytes
    from tools import approval as approval_module
    from tools import process_registry as process_registry_module
    from tools.approval import check_all_command_guards, reset_current_session_key, set_current_session_key
    from tools.browser_tool import build_front_door_access_boundary_evidence
    from tools.process_registry import ProcessRegistry

    # Busy metadata_update lock: diagnose --no-lock must return degraded evidence,
    # not block on the lock holder or leak local cwd details.
    _write_active_session_registry(kanban_home)
    monkeypatch.setattr(active_sessions, "_pid_alive", lambda *_args: True)
    rc = runtime_cli._cmd_active_sessions_diagnose(SimpleNamespace(json=False, no_lock=True))
    diagnose_out = capsys.readouterr().out
    assert rc == 0
    assert "lock_status=degraded" in diagnose_out
    assert "owner_kind=metadata_update" in diagnose_out
    assert "session_id=metadata-session" in diagnose_out
    assert "private project" not in diagnose_out

    # High-context closeout-only session: produce a compact closeout packet.
    record = start_request_watchdog(
        _agent(),
        request_id="turn-1:api:14",
        api_call_count=14,
        estimated_context_tokens=275_000,
        now=100.0,
        thresholds=_thresholds(),
        start_monitor=False,
    )
    status = poll_request_watchdog(record, now=131.0, thresholds=_thresholds())
    status["closeout_only"] = True
    status["repeated_stale_call_count"] = 2
    status["fixed_model_policy"] = True
    recovery = write_recoverable_turn_state(record, status=status, directory=tmp_path)
    recovery_packet = json.loads(recovery["path"].read_text(encoding="utf-8"))
    assert recovery["recommended_action"] == "compact_finalization_prompt"
    assert recovery_packet["recommended_action"] == "compact_finalization_prompt"
    assert "Do not retry the same huge request unchanged" in recovery_packet["resume_prompt"]
    assert "gpt-5.4" not in recovery_packet["resume_prompt"]

    # Required output artifact: near-budget guard must write before completion.
    required_report = tmp_path / "reports" / "monitoring-closeout.md"
    artifact_action = required_artifact_guard_action(
        prompt=f"Required output artifact path: {required_report}",
        stdout_draft="monitoring canary draft",
        turns_remaining=1,
    )
    assert artifact_action == {"action": "write_required_artifact", "path": str(required_report)}

    # Zero-budget and missing-skill dispatch gates must fire before any worker spawn.
    def profile_config(assignee: str):
        if assignee == "zero-worker":
            return {"goals": {"max_turns": 0}, "agent": {"max_turns": 90}}
        return {"goals": {"max_turns": 12}, "agent": {"max_turns": 12}}

    monkeypatch.setattr(kb, "_load_worker_profile_config", profile_config)
    _profile_with_skills(kanban_home, "skill-worker", ["present-skill"])
    spawned: list[str] = []
    conn = kb.connect()
    try:
        zero_task = kb.create_task(
            conn,
            title="zero budget dispatchable profile",
            assignee="zero-worker",
            goal_mode=True,
        )
        zero_dispatch = kb.dispatch_once(
            conn,
            spawn_fn=lambda task, workspace: spawned.append(task.id) or 1111,
        )
        assert zero_task in zero_dispatch.spawn_blocked_zero_budget
        assert spawned == []
        assert kb.get_task(conn, zero_task).status == "blocked"

        missing_skill_task = kb.create_task(
            conn,
            title="legacy missing worker skill",
            assignee="skill-worker",
            skills=["present-skill"],
        )
        conn.execute(
            "UPDATE tasks SET skills = ? WHERE id = ?",
            (json.dumps(["present-skill", "missing-skill"]), missing_skill_task),
        )
        conn.commit()
        missing_skill_dispatch = kb.dispatch_once(
            conn,
            spawn_fn=lambda task, workspace: spawned.append(task.id) or 2222,
        )
        assert missing_skill_task in missing_skill_dispatch.spawn_blocked_missing_skills
        assert spawned == []
        assert kb.latest_run(conn, missing_skill_task).outcome == "spawn_blocked_missing_skills"

        lineage_task = kb.create_task(
            conn,
            title="lineage guarded card",
            assignee="lineage-worker",
            workspace_kind="dir",
            workspace_path=str(kanban_home / "project-a"),
            session_id="foreign-session",
        )
        session_db = FakeSessionDB(
            {
                "foreign-session": {
                    "id": "foreign-session",
                    "title": "another task",
                    "kanban_task_id": "other-task",
                    "kanban_board": "default",
                    "workspace_path": str(kanban_home / "project-b"),
                }
            }
        )
        lineage_dispatch = kb.dispatch_once(
            conn,
            spawn_fn=lambda task, workspace: spawned.append(task.id) or 3333,
            session_db=session_db,
            board="default",
        )
        assert lineage_task in lineage_dispatch.spawn_blocked_session_lineage
        assert spawned == []
        assert kb.get_task(conn, lineage_task).status == "blocked"
        assert kb.latest_run(conn, lineage_task).outcome == "spawn_blocked_session_lineage"
    finally:
        conn.close()

    # CP1250 subprocess bytes are recovered, while unhelpful bytes are replacement-decoded.
    cp1250 = decode_subprocess_bytes(POLISH_TEXT.encode("cp1250"), allow_fallback=True)
    assert cp1250.text == POLISH_TEXT
    assert cp1250.used_fallback is True
    binary = decode_subprocess_bytes(b"before \xff\xfe after", allow_fallback=True)
    assert binary.text == "before \ufffd\ufffd after"
    assert binary.had_replacement is True

    # Background npm run dev wrapper: tree kill reports verified surviving child details.
    project_dir = str(tmp_path / "app")
    tree = _make_windows_dev_tree(project_dir)
    by_pid = {proc.pid: proc for proc in tree}
    taskkill_pids: list[int] = []

    def fake_process(pid):
        return by_pid[pid]

    def fake_taskkill(args, **kwargs):
        pid = int(args[args.index("/PID") + 1])
        taskkill_pids.append(pid)
        by_pid[pid].mark_taskkill()
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(process_registry_module, "_IS_WINDOWS", True)
    monkeypatch.setattr(
        ProcessRegistry,
        "_host_pid_is_ours",
        classmethod(lambda cls, pid, expected_start: pid == 1000),
    )
    monkeypatch.setattr(process_registry_module.psutil, "Process", fake_process)
    monkeypatch.setattr(process_registry_module.subprocess, "run", fake_taskkill)
    kill_result = ProcessRegistry._terminate_host_pid(1000, expected_start=123456)
    assert kill_result["status"] == "partial_kill_children_remain"
    assert kill_result["child_pids"] == [1004]
    assert "start-server.js" in kill_result["remaining_children"][0]["cmdline_fingerprint"]
    assert set(taskkill_pids) >= {1000, 1001, 1002, 1003, 1004}

    # Protected front-door response: cookie/signed URL material is value-free/redacted.
    front_door = build_front_door_access_boundary_evidence(
        status_code=302,
        expected_auth_provider="cloudflare_access",
        title="Cloudflare Access",
        url="https://crm.example.invalid/",
        final_url=PROVIDER_EVIDENCE_FIXTURES["cf_signed_redirect"],
        headers={
            "Set-Cookie": PROVIDER_EVIDENCE_FIXTURES["set_cookie"],
            "Cookie": PROVIDER_EVIDENCE_FIXTURES["cookie"],
            "Authorization": PROVIDER_EVIDENCE_FIXTURES["authorization"],
            "Location": PROVIDER_EVIDENCE_FIXTURES["private_provider_redirect"],
        },
        body_text="Sign in with Google to continue.",
    )
    assert front_door["boundary_result"] == "access_boundary"
    assert front_door["authenticated_ui_acceptance"] is False
    assert "headers" not in front_door
    assert "url" not in front_door
    _assert_access_material_redacted(front_door)

    # External git push: blocked before execution unless the session has explicit approval.
    monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    approval_module._permanent_approved.clear()
    approval_module.clear_session("monitoring-canary")
    token = set_current_session_key("monitoring-canary")
    try:
        push_gate = check_all_command_guards("git push origin main", "local")
    finally:
        reset_current_session_key(token)
        approval_module.clear_session("monitoring-canary")
        approval_module._permanent_approved.clear()
    assert push_gate["approved"] is False
    assert push_gate["risk_class"] == "external_repo_write"
    assert push_gate["outcome"] == "external_action_approval_required"
