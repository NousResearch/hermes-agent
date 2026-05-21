from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import time

import hermes_cli.github_issue_listener as github_issue_listener
from hermes_cli.github_issue_listener import (
    DEFAULT_WIP_COMMENT_BODY,
    GitHubIssueListener,
    IssueRef,
    IssueState,
    ListenerStore,
    READY_TO_CLOSE_MARKER,
    SubprocessHermesRunner,
    WAITING_MARKER,
    strip_markers,
)


class FakeGitHub:
    def __init__(self):
        self.issues = []
        self.comments = {}
        self.added_comments = []
        self.assignee_updates = []
        self.clear_assignee_updates = []
        self.execution_mode_updates = []
        self.project_status_updates = []

    def list_assigned_issues(self, owner, repo, assignee):
        return list(self.issues)

    def list_project_assigned_issues(self, project_owner, project_number, assignee):
        return list(self.issues)

    def get_issue(self, owner, repo, issue_number):
        for item in self.issues:
            if item["number"] == issue_number:
                return item
        raise AssertionError(f"unexpected issue lookup: {owner}/{repo}#{issue_number}")

    def list_comments(self, owner, repo, issue_number):
        return list(self.comments.get(issue_number, []))

    def add_comment(self, owner, repo, issue_number, body):
        self.added_comments.append((owner, repo, issue_number, body))
        return {"id": 999, "body": body}

    def set_assignees(self, owner, repo, issue_number, assignees):
        self.assignee_updates.append((owner, repo, issue_number, assignees))
        return {"assignees": assignees}

    def clear_assignees(self, owner, repo, issue_number, assignees):
        self.clear_assignee_updates.append((owner, repo, issue_number, assignees))
        for item in self.issues:
            if item["number"] == issue_number:
                item["assignees"] = [a for a in item.get("assignees", []) if a.get("login") not in assignees]
                return item
        return {"assignees": []}

    def set_project_execution_mode(self, project_owner, project_number, owner, repo, issue_number, mode):
        self.execution_mode_updates.append((project_owner, project_number, owner, repo, issue_number, mode))
        return {"updated": True, "mode": mode}

    def set_project_status(self, project_owner, project_number, owner, repo, issue_number, status):
        self.project_status_updates.append((project_owner, project_number, owner, repo, issue_number, status))
        return {"updated": True, "status": status}


class FakeRunner:
    def __init__(self, response="done", session_id="session-1"):
        self.response = response
        self.session_id = session_id
        self.calls = []

    def run_issue_turn(self, prompt, *, session_id):
        self.calls.append((prompt, session_id))
        return session_id or self.session_id, self.response


class ObservingRunner(FakeRunner):
    def __init__(self, github: FakeGitHub):
        super().__init__(response="done", session_id="session-1")
        self.github = github

    def run_issue_turn(self, prompt, *, session_id):
        assert self.github.added_comments[0][3] == DEFAULT_WIP_COMMENT_BODY
        return super().run_issue_turn(prompt, session_id=session_id)


def issue(number=9, owner="ryanleeai", repo="tasks", assignee="wingboot", state="open"):
    return {
        "number": number,
        "title": "Make Hermes listen",
        "body": "Body",
        "html_url": f"https://github.com/{owner}/{repo}/issues/{number}",
        "repository": {"name": repo, "owner": {"login": owner}},
        "assignees": [{"login": assignee}],
        "state": state,
    }


def comment(comment_id, login="seungjaeryanlee", body="reply"):
    return {"id": comment_id, "user": {"login": login}, "body": body}


def test_store_try_claim_skips_fresh_running_claim(tmp_path: Path):
    store = ListenerStore(tmp_path / "listener.db")
    ref = IssueRef("ryanleeai", "tasks", 9)

    assert store.try_claim(ref, run_id="run-1", stale_after_seconds=3600) is True
    assert store.try_claim(ref, run_id="run-2", stale_after_seconds=3600) is False

    state = store.get(ref)
    assert state is not None
    assert state.current_run_id == "run-1"
    assert state.status == "running"


def test_poll_starts_assigned_issue_and_records_session(tmp_path: Path):
    github = FakeGitHub()
    github.issues = [issue(9)]
    github.comments = {9: [comment(100, "seungjaeryanlee", "ready")]}
    runner = FakeRunner(response="Implemented a first step.", session_id="new-session")
    store = ListenerStore(tmp_path / "listener.db")

    result = GitHubIssueListener(
        github=github,
        runner=runner,
        store=store,
        owner="ryanleeai",
        repo="tasks",
        assignee="wingboot",
        human_assignee="seungjaeryanlee",
    ).poll_once()

    assert result["results"][0]["action"] == "ran"
    assert runner.calls[0][1] is None
    assert github.added_comments[0][3] == DEFAULT_WIP_COMMENT_BODY
    assert github.added_comments[1][3] == "Implemented a first step."
    state = store.get(IssueRef("ryanleeai", "tasks", 9))
    assert state is not None
    assert state.session_id == "new-session"
    assert state.last_comment_id_seen == 999
    assert state.status == "idle"


def test_poll_posts_wip_comment_before_running_hermes(tmp_path: Path):
    github = FakeGitHub()
    github.issues = [issue(9)]
    github.comments = {9: [comment(100, "seungjaeryanlee", "ready")]}
    runner = ObservingRunner(github)
    store = ListenerStore(tmp_path / "listener.db")

    GitHubIssueListener(
        github=github,
        runner=runner,
        store=store,
        owner="ryanleeai",
        repo="tasks",
        assignee="wingboot",
    ).poll_once()

    assert runner.calls


def test_project_poll_sets_in_progress_before_running_hermes(tmp_path: Path):
    github = FakeGitHub()
    github.issues = [issue(9, owner="ryanleeai", repo="lyrv")]
    github.comments = {9: [comment(100, "seungjaeryanlee", "ready")]}
    runner = ObservingRunner(github)
    store = ListenerStore(tmp_path / "listener.db")

    GitHubIssueListener(
        github=github,
        runner=runner,
        store=store,
        owner="ryanleeai",
        repo="tasks",
        assignee="wingboot",
        project_owner="ryanleeai",
        project_number=1,
    ).poll_once()

    assert github.project_status_updates == [("ryanleeai", 1, "ryanleeai", "lyrv", 9, "In Progress")]


def test_poll_continues_existing_session_after_human_reply(tmp_path: Path):
    github = FakeGitHub()
    github.issues = [issue(9)]
    github.comments = {9: [comment(100), comment(101, "seungjaeryanlee", "continue")]}
    runner = FakeRunner(response="continued")
    store = ListenerStore(tmp_path / "listener.db")
    state = store.get(IssueRef("ryanleeai", "tasks", 9))
    assert state is None
    store.upsert(
        IssueState(
            owner="ryanleeai",
            repo="tasks",
            issue_number=9,
            session_id="existing-session",
            status="waiting_for_ryan",
            last_comment_id_seen=100,
        )
    )

    result = GitHubIssueListener(
        github=github,
        runner=runner,
        store=store,
        owner="ryanleeai",
        repo="tasks",
        assignee="wingboot",
    ).poll_once()

    assert result["results"][0]["action"] == "ran"
    assert runner.calls[0][1] == "existing-session"
    assert "Continue the existing issue-bound Hermes session" in runner.calls[0][0]


def test_waiting_issue_resumes_on_assignment_even_without_new_comment(tmp_path: Path):
    github = FakeGitHub()
    github.issues = [issue(9)]
    github.comments = {9: [comment(100, "wingboot", "Need input")]}
    runner = FakeRunner(response="resumed")
    store = ListenerStore(tmp_path / "listener.db")
    store.upsert(
        IssueState(
            owner="ryanleeai",
            repo="tasks",
            issue_number=9,
            session_id="existing-session",
            status="waiting_for_ryan",
            last_comment_id_seen=100,
        )
    )

    result = GitHubIssueListener(
        github=github,
        runner=runner,
        store=store,
        owner="ryanleeai",
        repo="tasks",
        assignee="wingboot",
    ).poll_once()

    assert result["results"][0]["action"] == "ran"
    assert runner.calls[0][1] == "existing-session"


def test_idle_issue_with_existing_session_skips_without_new_input(tmp_path: Path):
    github = FakeGitHub()
    github.issues = [issue(9)]
    github.comments = {9: [comment(100, "wingboot", "Done")]}
    runner = FakeRunner(response="should not run")
    store = ListenerStore(tmp_path / "listener.db")
    store.upsert(
        IssueState(
            owner="ryanleeai",
            repo="tasks",
            issue_number=9,
            session_id="existing-session",
            status="idle",
            last_comment_id_seen=100,
        )
    )

    result = GitHubIssueListener(
        github=github,
        runner=runner,
        store=store,
        owner="ryanleeai",
        repo="tasks",
        assignee="wingboot",
    ).poll_once()

    assert result["results"][0]["action"] == "skipped_no_new_input"
    assert runner.calls == []


def test_waiting_marker_assigns_back_to_human(tmp_path: Path):
    github = FakeGitHub()
    github.issues = [issue(9)]
    github.comments = {9: [comment(100)]}
    runner = FakeRunner(response=f"Need a decision. {WAITING_MARKER}")
    store = ListenerStore(tmp_path / "listener.db")

    GitHubIssueListener(
        github=github,
        runner=runner,
        store=store,
        owner="ryanleeai",
        repo="tasks",
        assignee="wingboot",
        human_assignee="seungjaeryanlee",
    ).poll_once()

    assert github.added_comments[0][3] == DEFAULT_WIP_COMMENT_BODY
    assert github.added_comments[1][3] == "Need a decision."
    assert github.assignee_updates == [("ryanleeai", "tasks", 9, ["seungjaeryanlee"])]
    state = store.get(IssueRef("ryanleeai", "tasks", 9))
    assert state is not None
    assert state.status == "waiting_for_ryan"


def test_ready_to_close_marker_assigns_back_to_human(tmp_path: Path):
    github = FakeGitHub()
    github.issues = [issue(9)]
    github.comments = {9: [comment(100)]}
    runner = FakeRunner(response=f"Ready for review. {READY_TO_CLOSE_MARKER}")
    store = ListenerStore(tmp_path / "listener.db")

    GitHubIssueListener(
        github=github,
        runner=runner,
        store=store,
        owner="ryanleeai",
        repo="tasks",
        assignee="wingboot",
        human_assignee="seungjaeryanlee",
    ).poll_once()

    assert github.added_comments[0][3] == DEFAULT_WIP_COMMENT_BODY
    assert github.added_comments[1][3] == "Ready for review."
    assert github.assignee_updates == [("ryanleeai", "tasks", 9, ["seungjaeryanlee"])]
    state = store.get(IssueRef("ryanleeai", "tasks", 9))
    assert state is not None
    assert state.status == "awaiting_close_approval"



def test_closed_issue_clears_assignees_and_sets_automated_execution_mode(tmp_path: Path):
    github = FakeGitHub()
    github.issues = [issue(12, owner="ryanleeai", repo="lyrv", assignee="seungjaeryanlee")]
    github.comments = {12: [comment(100, "seungjaeryanlee", "approved")]}

    class ClosingRunner(FakeRunner):
        def run_issue_turn(self, prompt, *, session_id):
            session_id, response = super().run_issue_turn(prompt, session_id=session_id)
            github.issues[0]["state"] = "closed"
            return session_id, response

    store = ListenerStore(tmp_path / "listener.db")
    assert store.try_claim(IssueRef("ryanleeai", "lyrv", 12), run_id="run-1", stale_after_seconds=3600) is True

    result = GitHubIssueListener(
        github=github,
        runner=ClosingRunner(response="Closed as completed."),
        store=store,
        owner="ryanleeai",
        repo="tasks",
        project_owner="ryanleeai",
        project_number=1,
        assignee="wingboot",
        human_assignee="seungjaeryanlee",
    ).run_claimed_issue(IssueRef("ryanleeai", "lyrv", 12), run_id="run-1")

    assert result["status"] == "closed"
    assert github.clear_assignee_updates == [("ryanleeai", "lyrv", 12, ["seungjaeryanlee"])]
    assert github.execution_mode_updates == [("ryanleeai", 1, "ryanleeai", "lyrv", 12, "automated")]
    assert github.assignee_updates == []
    state = store.get(IssueRef("ryanleeai", "lyrv", 12))
    assert state is not None
    assert state.status == "closed"


def test_subprocess_runner_limits_tools_and_sets_github_token(monkeypatch):
    calls = []

    def fake_run(cmd, text, capture_output, env=None, timeout=None):
        calls.append({"cmd": cmd, "env": env, "timeout": timeout})
        return SimpleNamespace(returncode=0, stdout="final response", stderr="")

    monkeypatch.setattr(github_issue_listener, "_load_github_token", lambda: "wingboot-token")
    monkeypatch.setattr(github_issue_listener.subprocess, "run", fake_run)
    monkeypatch.setattr(github_issue_listener, "_latest_session_id_after", lambda source, before: None)

    session_id, response = SubprocessHermesRunner(hermes_bin="hermes-test").run_issue_turn("Do work", session_id=None)

    assert session_id == ""
    assert response == "final response"
    cmd = calls[0]["cmd"]
    assert cmd[:5] == ["hermes-test", "chat", "--quiet", "--source", "github_issue_listener"]
    assert cmd[cmd.index("--toolsets") + 1] == "terminal,file,skills,web,vision"
    assert calls[0]["env"]["GH_TOKEN"] == "wingboot-token"
    assert calls[0]["env"]["GITHUB_TOKEN"] == "wingboot-token"


def test_poll_project_board_uses_issue_repository_for_refs(tmp_path: Path):
    github = FakeGitHub()
    github.issues = [issue(2, owner="ryanleeai", repo="knowledgebase")]
    github.comments = {2: [comment(100, "seungjaeryanlee", "ready")]}
    runner = FakeRunner(response="board item handled", session_id="board-session")
    store = ListenerStore(tmp_path / "listener.db")

    result = GitHubIssueListener(
        github=github,
        runner=runner,
        store=store,
        owner="ryanleeai",
        repo="tasks",
        project_owner="ryanleeai",
        project_number=1,
        assignee="wingboot",
    ).poll_once()

    assert result["results"][0]["issue"] == "ryanleeai/knowledgebase#2"
    assert github.added_comments[0][:3] == ("ryanleeai", "knowledgebase", 2)
    state = store.get(IssueRef("ryanleeai", "knowledgebase", 2))
    assert state is not None
    assert state.session_id == "board-session"


def test_strip_markers_prefers_waiting_state():
    status, cleaned = strip_markers(f"Question? {WAITING_MARKER}")
    assert status == "waiting_for_ryan"
    assert cleaned == "Question?"


def test_background_poll_dispatches_worker_and_exits_without_running_hermes(tmp_path: Path):
    github = FakeGitHub()
    github.issues = [issue(9)]
    github.comments = {9: [comment(100, "seungjaeryanlee", "ready")]}
    runner = FakeRunner(response="should not run inline")
    store = ListenerStore(tmp_path / "listener.db")
    dispatched = []

    def fake_dispatch(ref, *, run_id):
        dispatched.append((ref, run_id))
        return 12345, str(tmp_path / "run.log")

    result = GitHubIssueListener(
        github=github,
        runner=runner,
        store=store,
        owner="ryanleeai",
        repo="tasks",
        assignee="wingboot",
        dispatch_background=True,
        worker_dispatcher=fake_dispatch,
    ).poll_once()

    assert result["results"][0]["action"] == "dispatched"
    assert runner.calls == []
    assert dispatched[0][0] == IssueRef("ryanleeai", "tasks", 9)
    state = store.get(IssueRef("ryanleeai", "tasks", 9))
    assert state is not None
    assert state.status == "running"
    assert state.worker_pid == 12345
    assert state.log_path == str(tmp_path / "run.log")


def test_worker_process_completes_claim_and_clears_running_state(tmp_path: Path):
    github = FakeGitHub()
    github.issues = [issue(9)]
    github.comments = {9: [comment(100, "seungjaeryanlee", "ready")]}
    runner = FakeRunner(response="worker completed", session_id="worker-session")
    store = ListenerStore(tmp_path / "listener.db")
    ref = IssueRef("ryanleeai", "tasks", 9)
    assert store.try_claim(ref, run_id="run-1", stale_after_seconds=3600)

    result = GitHubIssueListener(
        github=github,
        runner=runner,
        store=store,
        owner="ryanleeai",
        repo="tasks",
        assignee="wingboot",
    ).run_claimed_issue(ref, run_id="run-1")

    assert result["action"] == "ran"
    assert github.added_comments[0][3] == "worker completed"
    state = store.get(ref)
    assert state is not None
    assert state.status == "idle"
    assert state.current_run_id is None
    assert state.worker_pid is None
    assert state.last_comment_id_seen == 999


def test_poll_skips_stale_running_issue_when_worker_pid_is_alive(monkeypatch, tmp_path: Path):
    github = FakeGitHub()
    github.issues = [issue(9)]
    github.comments = {9: [comment(100, "seungjaeryanlee", "ready")]}
    runner = FakeRunner(response="should not run")
    store = ListenerStore(tmp_path / "listener.db")
    ref = IssueRef("ryanleeai", "tasks", 9)
    store.upsert(
        IssueState(
            owner="ryanleeai",
            repo="tasks",
            issue_number=9,
            status="running",
            current_run_id="old-run",
            claimed_at=time.time() - 7200,
            worker_pid=12345,
        )
    )
    monkeypatch.setattr(github_issue_listener, "_pid_is_running", lambda pid: pid == 12345)

    result = GitHubIssueListener(
        github=github,
        runner=runner,
        store=store,
        owner="ryanleeai",
        repo="tasks",
        assignee="wingboot",
        stale_after_seconds=60,
    ).poll_once()

    assert result["results"][0]["action"] == "skipped_worker_alive"
    assert runner.calls == []


def test_fresh_dead_worker_can_be_reclaimed_without_waiting_for_stale_window(tmp_path: Path, monkeypatch):
    github = FakeGitHub()
    github.issues = [issue(9)]
    github.comments = {9: [comment(100, "seungjaeryanlee", "ready")]}
    runner = FakeRunner(response="reclaimed fresh", session_id="session-2")
    store = ListenerStore(tmp_path / "listener.db")
    ref = IssueRef("ryanleeai", "tasks", 9)
    store.upsert(
        IssueState(
            owner="ryanleeai",
            repo="tasks",
            issue_number=9,
            status="running",
            current_run_id="old-run",
            claimed_at=time.time(),
            worker_pid=12345,
        )
    )
    monkeypatch.setattr(github_issue_listener, "_pid_is_running", lambda pid: False)

    result = GitHubIssueListener(
        github=github,
        runner=runner,
        store=store,
        owner="ryanleeai",
        repo="tasks",
        assignee="wingboot",
        stale_after_seconds=3600,
    ).poll_once()

    assert result["results"][0]["action"] == "ran"
    assert github.added_comments[0][3] == DEFAULT_WIP_COMMENT_BODY
    assert github.added_comments[1][3] == "reclaimed fresh"
    state = store.get(ref)
    assert state is not None
    assert state.status == "idle"
    assert state.current_run_id is None


def test_stale_dead_worker_can_be_reclaimed(tmp_path: Path, monkeypatch):
    github = FakeGitHub()
    github.issues = [issue(9)]
    github.comments = {9: [comment(100, "seungjaeryanlee", "ready")]}
    runner = FakeRunner(response="reclaimed", session_id="session-2")
    store = ListenerStore(tmp_path / "listener.db")
    ref = IssueRef("ryanleeai", "tasks", 9)
    store.upsert(
        IssueState(
            owner="ryanleeai",
            repo="tasks",
            issue_number=9,
            status="running",
            current_run_id="old-run",
            claimed_at=time.time() - 7200,
            worker_pid=12345,
        )
    )
    monkeypatch.setattr(github_issue_listener, "_pid_is_running", lambda pid: False)

    result = GitHubIssueListener(
        github=github,
        runner=runner,
        store=store,
        owner="ryanleeai",
        repo="tasks",
        assignee="wingboot",
        stale_after_seconds=60,
    ).poll_once()

    assert result["results"][0]["action"] == "ran"
    assert github.added_comments[0][3] == DEFAULT_WIP_COMMENT_BODY
    assert github.added_comments[1][3] == "reclaimed"
    state = store.get(ref)
    assert state is not None
    assert state.status == "idle"
    assert state.current_run_id is None
