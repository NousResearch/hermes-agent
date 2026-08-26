"""Conformance tests for the versioned read-side ProjectKanbanHost."""

from __future__ import annotations

from pathlib import Path

import pytest

import hermes_constants
from hermes_cli import kanban_db as kb
from hermes_cli import projects_db as pdb
from hermes_cli.project_kanban_host import (
    CONTRACT_VERSION,
    HostError,
    ProjectKanbanHost,
)


READ_CAPABILITIES = {
    "get_board",
    "get_project",
    "get_task",
    "list_comments",
    "list_epics",
    "list_events",
    "list_links",
    "list_projects",
    "list_task_children",
    "list_tasks",
    "list_profiles",
    "validate_project",
}
WRITE_CAPABILITIES = {
    "add_comment",
    "assign_task",
    "attach_task_to_epic",
    "block_task",
    "create_epic",
    "create_task",
    "detach_task_from_epic",
    "get_epic",
    "link_tasks",
    "transition_task",
    "unlink_tasks",
    "unblock_task",
    "update_epic",
    "update_task",
    "provision_project",
}


@pytest.fixture
def canonical_home(tmp_path: Path):
    root = tmp_path / "hermes"
    root.mkdir()
    token = hermes_constants.set_hermes_home_override(root)
    try:
        with pdb.connect_closing() as conn:
            project_id = pdb.create_project(
                conn,
                name="Alpha Project",
                slug="alpha-project",
                board_slug="alpha",
            )
        with kb.scoped_kanban_home(root):
            kb.create_board("alpha", name="Alpha Board", project_id=project_id)
            with kb.connect_closing(board="alpha") as conn:
                epic_id = kb.create_epic(
                    conn,
                    title="Launch Epic",
                    board_slug="alpha",
                )
                parent_id = kb.create_task(
                    conn,
                    title="Parent",
                    board="alpha",
                    project_id=project_id,
                )
                child_id = kb.create_task(
                    conn,
                    title="Child",
                    body="Canonical body",
                    board="alpha",
                    project_id=project_id,
                    task_kind="subtask",
                    parent_task_id=parent_id,
                    epic_id=epic_id,
                )
                second_child_id = kb.create_task(
                    conn,
                    title="Second child",
                    board="alpha",
                    project_id=project_id,
                    task_kind="subtask",
                    parent_task_id=parent_id,
                )
                kb.link_tasks(conn, parent_id, child_id)
                kb.add_comment(conn, child_id, "sahil", "First note")
                kb.add_comment(conn, child_id, "octacon", "Second note")
                kb.assign_task(conn, child_id, "octacon")
        assert (root / "kanban" / "boards" / "alpha" / "kanban.db").is_file()
        yield {
            "root": root,
            "project_id": project_id,
            "epic_id": epic_id,
            "parent_id": parent_id,
            "child_id": child_id,
            "second_child_id": second_child_id,
        }
    finally:
        hermes_constants.reset_hermes_home_override(token)


def test_capabilities_are_versioned_exact_and_fail_closed(canonical_home):
    host = ProjectKanbanHost(
        hermes_home=canonical_home["root"],
        board="alpha",
    )
    assert host.capabilities() == {
        "contract_version": CONTRACT_VERSION,
        "methods": sorted(READ_CAPABILITIES | WRITE_CAPABILITIES),
    }
    host.require_capability("get_task")

    with pytest.raises(HostError) as exc:
        host.require_capability("future_capability")
    assert exc.value.code == "unsupported_capability"
    assert exc.value.to_envelope() == {
        "contract_version": CONTRACT_VERSION,
        "error": {
            "code": "unsupported_capability",
            "message": "capability is not available",
            "fields": {"capability": "future_capability"},
        },
    }


def test_project_board_task_and_epic_reads_are_canonical(canonical_home):
    host = ProjectKanbanHost(
        hermes_home=canonical_home["root"],
        board="alpha",
    )
    projects = host.list_projects()
    assert [project["id"] for project in projects] == [canonical_home["project_id"]]
    assert host.get_project("alpha-project")["board_slug"] == "alpha"

    board = host.get_board()
    assert board["slug"] == "alpha"
    assert board["project_id"] == canonical_home["project_id"]

    tasks = host.list_tasks(limit=10)
    assert tasks["limit"] == 10
    assert tasks["has_more"] is False
    assert canonical_home["child_id"] in {item["id"] for item in tasks["items"]}

    epics = host.list_epics(limit=10)
    assert [item["id"] for item in epics["items"]] == [canonical_home["epic_id"]]


def test_task_detail_is_bounded_ordered_and_complete(canonical_home):
    host = ProjectKanbanHost(
        hermes_home=canonical_home["root"],
        board="alpha",
    )
    detail = host.get_task(canonical_home["child_id"], limit=1)
    assert set(detail) == {
        "task",
        "epic",
        "parent_task",
        "children",
        "links",
        "comments",
        "events",
        "workflow_run",
    }
    assert detail["task"]["id"] == canonical_home["child_id"]
    assert detail["task"]["body"] == "Canonical body"
    assert detail["epic"]["id"] == canonical_home["epic_id"]
    assert detail["parent_task"]["id"] == canonical_home["parent_id"]
    assert detail["workflow_run"] is None

    for key in ("children", "links", "comments", "events"):
        assert set(detail[key]) == {"items", "limit", "has_more"}
        assert detail[key]["limit"] == 1
        assert len(detail[key]["items"]) <= 1

    assert detail["comments"]["has_more"] is True
    assert detail["comments"]["items"][0]["body"] == "First note"
    assert detail["links"]["items"] == [
        {"direction": "parent", "task_id": canonical_home["parent_id"]}
    ]

    children = host.list_task_children(canonical_home["parent_id"], limit=1)
    assert children["has_more"] is True
    assert len(children["items"]) == 1


def test_missing_and_invalid_inputs_return_safe_host_errors(canonical_home, tmp_path):
    host = ProjectKanbanHost(
        hermes_home=canonical_home["root"],
        board="alpha",
    )
    for operation, code in (
        (lambda: host.get_task("missing"), "task_not_found"),
        (lambda: host.get_project("missing"), "project_not_found"),
        (lambda: host.get_board("missing"), "board_not_found"),
        (lambda: host.list_tasks(limit=0), "invalid_limit"),
    ):
        with pytest.raises(HostError) as exc:
            operation()
        assert exc.value.code == code
        rendered = str(exc.value.to_envelope())
        assert "sqlite" not in rendered.lower()
        assert str(canonical_home["root"]) not in rendered

    empty_root = tmp_path / "other-hermes"
    empty_root.mkdir()
    isolated = ProjectKanbanHost(hermes_home=empty_root, board="alpha")
    with pytest.raises(HostError) as exc:
        isolated.get_project(canonical_home["project_id"])
    assert exc.value.code == "project_not_found"


def test_alternate_home_cannot_fall_back_to_process_kanban_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    process_root = tmp_path / "process-root"
    process_root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(process_root))

    kb.create_board("private", name="Process Board")
    with kb.connect_closing(board="private") as conn:
        kb.create_task(conn, title="Process task", board="private")

    alternate_root = tmp_path / "alternate-root"
    alternate_root.mkdir()
    isolated = ProjectKanbanHost(hermes_home=alternate_root, board="private")

    with pytest.raises(HostError) as exc:
        isolated.get_board()
    assert exc.value.code == "board_not_found"
    assert not (alternate_root / "kanban" / "boards" / "private").exists()
    assert kb.kanban_home() == process_root


def test_task_writes_are_canonical_and_idempotent(canonical_home):
    host = ProjectKanbanHost(
        hermes_home=canonical_home["root"],
        board="alpha",
    )
    created = host.create_task(
        title="Host-created child",
        body="Created through the canonical host",
        assignee="quan",
        priority=3,
        task_kind="subtask",
        parent_task_id=canonical_home["parent_id"],
        epic_id=canonical_home["epic_id"],
        project_id=canonical_home["project_id"],
        idempotency_key="dockyard-host-create-1",
    )
    replay = host.create_task(
        title="Ignored replay title",
        task_kind="subtask",
        parent_task_id=canonical_home["parent_id"],
        idempotency_key="dockyard-host-create-1",
    )
    assert replay["id"] == created["id"]
    assert created["body"] == "Created through the canonical host"
    assert created["task_kind"] == "subtask"
    assert created["parent_task_id"] == canonical_home["parent_id"]
    assert created["epic_id"] == canonical_home["epic_id"]

    assigned = host.assign_task(created["id"], "octacon")
    assert assigned["assignee"] == "octacon"
    blocked = host.block_task(created["id"], reason="Waiting for input")
    assert blocked["status"] == "blocked"
    unblocked = host.unblock_task(created["id"])
    assert unblocked["status"] in {"ready", "todo"}

    with kb.scoped_kanban_home(canonical_home["root"]):
        with kb.connect_closing(board="alpha") as conn:
            persisted = kb.get_task(conn, created["id"])
            matches = [
                task
                for task in kb.list_tasks(conn)
                if task.idempotency_key == "dockyard-host-create-1"
            ]
    assert persisted is not None
    assert persisted.assignee == "octacon"
    assert len(matches) == 1


def test_epic_attachment_comment_and_link_writes_persist(canonical_home):
    host = ProjectKanbanHost(
        hermes_home=canonical_home["root"],
        board="alpha",
    )
    epic = host.create_epic(
        title="Host Epic",
        description="Created through the host",
    )
    assert host.get_epic(epic["id"])["title"] == "Host Epic"
    updated = host.update_epic(
        epic["id"],
        title="Updated Host Epic",
        status="done",
        parent_epic_id=canonical_home["epic_id"],
    )
    assert updated["title"] == "Updated Host Epic"
    assert updated["status"] == "done"
    assert updated["parent_epic_id"] == canonical_home["epic_id"]

    attached = host.attach_task_to_epic(canonical_home["second_child_id"], epic["id"])
    assert attached["epic_id"] == epic["id"]
    detached = host.detach_task_from_epic(canonical_home["second_child_id"])
    assert detached["epic_id"] is None

    comment = host.add_comment(
        canonical_home["second_child_id"],
        author="sahil",
        body="Host comment",
    )
    assert comment["body"] == "Host comment"
    link = host.link_tasks(
        canonical_home["child_id"],
        canonical_home["second_child_id"],
    )
    assert link == {
        "parent_task_id": canonical_home["child_id"],
        "child_task_id": canonical_home["second_child_id"],
    }
    assert host.unlink_tasks(
        canonical_home["child_id"],
        canonical_home["second_child_id"],
    ) == link

    with kb.scoped_kanban_home(canonical_home["root"]):
        with kb.connect_closing(board="alpha") as conn:
            assert kb.get_epic(conn, epic["id"]).title == "Updated Host Epic"
            comments = kb.list_comments(conn, canonical_home["second_child_id"])
            assert comments[-1].body == "Host comment"
            assert canonical_home["second_child_id"] not in kb.child_ids(
                conn, canonical_home["child_id"]
            )


def test_write_errors_are_safe_and_writes_stay_isolated(
    canonical_home,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    host = ProjectKanbanHost(
        hermes_home=canonical_home["root"],
        board="alpha",
    )
    for operation, code in (
        (lambda: host.assign_task("missing", "quan"), "task_not_found"),
        (lambda: host.get_epic("missing"), "epic_not_found"),
        (lambda: host.create_task(title=""), "validation_error"),
    ):
        with pytest.raises(HostError) as exc:
            operation()
        assert exc.value.code == code
        rendered = str(exc.value.to_envelope())
        assert "sqlite" not in rendered.lower()
        assert str(canonical_home["root"]) not in rendered

    alternate_root = tmp_path / "write-isolation"
    alternate_root.mkdir()
    isolated = ProjectKanbanHost(hermes_home=alternate_root, board="alpha")
    with pytest.raises(HostError) as exc:
        isolated.create_task(title="Must not leak")
    assert exc.value.code == "board_not_found"
    assert not (alternate_root / "kanban" / "boards" / "alpha").exists()

    before = len(host.list_tasks(limit=100)["items"])
    monkeypatch.setattr(
        kb,
        "_assert_not_delegated_child_mutation",
        lambda: (_ for _ in ()).throw(PermissionError("internal child guard")),
    )
    with pytest.raises(HostError) as exc:
        host.create_task(title="Forbidden child write")
    assert exc.value.code == "write_forbidden"
    assert "internal child guard" not in str(exc.value.to_envelope())
    assert len(host.list_tasks(limit=100)["items"]) == before


def test_task_edit_and_transition_use_canonical_domain(canonical_home):
    host = ProjectKanbanHost(
        hermes_home=canonical_home["root"],
        board="alpha",
    )
    created = host.create_task(
        title="Before",
        body="old",
        priority=1,
        initial_status="backlog",
        tenant="dockyard-test",
        workspace_kind="dir",
        workspace_path="/tmp/dockyard-host-test",
        max_runtime_seconds=900,
        skills=["test-driven-development"],
        goal_mode=True,
        goal_max_turns=3,
        model_override="test-model",
        provider_override="test-provider",
        reasoning_effort="high",
        created_by="dashboard",
    )
    assert created["tenant"] == "dockyard-test"
    assert created["workspace_kind"] == "dir"
    assert created["max_runtime_seconds"] == 900
    assert created["skills"] == ["test-driven-development"]
    assert created["goal_mode"] is True
    assert created["model_override"] == "test-model"
    assert created["provider_override"] == "test-provider"
    assert created["reasoning_effort"] == "high"
    assert created["created_by"] == "dashboard"

    edited = host.update_task(
        created["id"],
        title="After",
        body=None,
        priority=5,
        task_kind="bug",
        model_override="next-model",
        provider_override="next-provider",
        reasoning_effort="low",
    )
    assert edited["title"] == "After"
    assert edited["body"] is None
    assert edited["priority"] == 5
    assert edited["task_kind"] == "bug"
    assert edited["model_override"] == "next-model"
    assert edited["provider_override"] == "next-provider"
    assert edited["reasoning_effort"] == "low"

    assert host.transition_task(created["id"], "triage")["status"] == "triage"
    assert host.transition_task(created["id"], "ready")["status"] == "ready"
    blocked = host.transition_task(
        created["id"],
        "blocked",
        reason="external dependency",
    )
    assert blocked["status"] == "blocked"
    assert host.transition_task(created["id"], "ready")["status"] == "ready"
    completed = host.transition_task(
        created["id"],
        "done",
        result="verified complete",
        summary="host transition",
        metadata={"tests": ["focused"]},
    )
    assert completed["status"] == "done"
    assert completed["result"] == "verified complete"

    with pytest.raises(HostError) as invalid:
        host.transition_task(created["id"], "running")
    assert invalid.value.code == "validation_error"

    with pytest.raises(HostError) as missing:
        host.update_task("t_missing", title="nope")
    assert missing.value.code == "task_not_found"


def test_host_service_does_not_bypass_public_domain_apis():
    source = Path(__file__).parents[2] / "hermes_cli" / "project_kanban_host.py"
    text = source.read_text(encoding="utf-8")
    prohibited = ("sqlite3", ".execute(", "SELECT ", "INSERT ", "UPDATE ", "DELETE ")
    assert [token for token in prohibited if token in text] == []


def _project_host(tmp_path: Path, slug: str = "alpha"):
    root = tmp_path / f"{slug}-hermes"
    root.mkdir()
    profiles = root / "profiles" / "octacon"
    profiles.mkdir(parents=True)
    (profiles / "SOUL.md").write_text("# Octacon\n", encoding="utf-8")
    repos = tmp_path / f"{slug}-repos"
    repo = repos / slug
    repo.mkdir(parents=True)
    host = ProjectKanbanHost(
        hermes_home=root,
        board=slug,
        allowed_repo_root=repos,
    )
    return host, root, repo


def _project_payload(repo: Path, slug: str = "alpha") -> dict:
    return {
        "name": f"{slug.title()} Project",
        "slug": slug,
        "description": f"Deliver the {slug} project safely",
        "repo_path": str(repo),
        "lead_profile": "octacon",
        "board_slug": slug,
    }


def test_project_validation_is_explicit_rooted_and_field_specific(tmp_path):
    host, root, repo = _project_host(tmp_path)

    assert [item["name"] for item in host.list_profiles()] == [
        "default",
        "octacon",
    ]
    valid = host.validate_project(**_project_payload(repo))
    assert valid == {
        **_project_payload(repo),
        "repo_path": str(repo.resolve()),
    }

    with pytest.raises(HostError) as invalid:
        host.validate_project(
            name=" ",
            slug="Bad Slug",
            description="",
            repo_path="relative/path",
            lead_profile="missing",
            board_slug="Bad/Board",
        )
    assert invalid.value.code == "validation_error"
    assert set(invalid.value.fields) == {
        "name",
        "slug",
        "description",
        "repo_path",
        "lead_profile",
        "board_slug",
    }
    rendered = str(invalid.value.to_envelope())
    assert str(root) not in rendered
    assert "traceback" not in rendered.lower()

    outside = tmp_path / "outside"
    outside.mkdir()
    escape = repo.parent / "escape"
    escape.symlink_to(outside, target_is_directory=True)
    with pytest.raises(HostError) as escaped:
        host.validate_project(**{**_project_payload(repo), "repo_path": str(escape)})
    assert set(escaped.value.fields) == {"repo_path"}


def test_project_provisioning_is_native_idempotent_and_persisted(tmp_path):
    host, root, repo = _project_host(tmp_path)
    payload = _project_payload(repo)

    created = host.provision_project(
        **payload,
        idempotency_key="dockyard-onboard-alpha",
    )
    replay = host.provision_project(
        **payload,
        idempotency_key="dockyard-onboard-alpha",
    )
    assert created["status"] == "complete"
    assert replay["replayed"] is True
    assert replay["project"]["id"] == created["project"]["id"]
    assert created["project"]["board_slug"] == "alpha"
    assert created["board"]["project_id"] == created["project"]["id"]
    assert host.get_project("alpha")["id"] == created["project"]["id"]
    assert host.get_board("alpha")["project_id"] == created["project"]["id"]

    with pdb.connect_closing(root / "projects.db") as conn:
        assert len(pdb.list_projects(conn)) == 1
        journal = conn.execute(
            "SELECT status, project_id, project_slug, board_slug "
            "FROM project_provisioning_journal WHERE idempotency_key = ?",
            ("dockyard-onboard-alpha",),
        ).fetchone()
    assert dict(journal) == {
        "status": "complete",
        "project_id": created["project"]["id"],
        "project_slug": "alpha",
        "board_slug": "alpha",
    }

    with pytest.raises(HostError) as conflict:
        host.provision_project(
            **{**payload, "description": "Conflicting payload"},
            idempotency_key="dockyard-onboard-alpha",
        )
    assert conflict.value.code == "idempotency_conflict"
    assert len(host.list_projects()) == 1


def test_project_provisioning_recovers_after_crash_and_hides_pending_state(
    tmp_path,
    monkeypatch,
):
    host, root, repo = _project_host(tmp_path, "crash")
    payload = _project_payload(repo, "crash")
    complete = pdb.complete_project_provisioning

    def crash_before_complete(*args, **kwargs):
        raise SystemExit("simulated process crash")

    monkeypatch.setattr(pdb, "complete_project_provisioning", crash_before_complete)
    with pytest.raises(SystemExit):
        host.provision_project(
            **payload,
            idempotency_key="dockyard-onboard-crash",
        )

    assert host.list_projects() == []
    with pytest.raises(HostError) as hidden_board:
        host.get_board("crash")
    assert hidden_board.value.code == "board_not_found"
    with pdb.connect_closing(root / "projects.db") as conn:
        pending = pdb.find_project_by_provisioning_key(
            conn,
            "dockyard-onboard-crash",
        )
    assert pending is not None
    with kb.scoped_kanban_home(root):
        assert kb.board_exists("crash")

    monkeypatch.setattr(pdb, "complete_project_provisioning", complete)
    recovered = host.provision_project(
        **payload,
        idempotency_key="dockyard-onboard-crash",
    )
    assert recovered["project"]["id"] == pending.id
    assert recovered["status"] == "complete"
    assert host.get_board("crash")["project_id"] == pending.id


def test_project_provisioning_compensates_normal_failure(tmp_path, monkeypatch):
    host, root, repo = _project_host(tmp_path, "compensate")
    payload = _project_payload(repo, "compensate")
    create_project = pdb.create_project

    def fail_project(*args, **kwargs):
        raise RuntimeError("raw internal failure")

    monkeypatch.setattr(pdb, "create_project", fail_project)
    with pytest.raises(HostError) as failed:
        host.provision_project(
            **payload,
            idempotency_key="dockyard-onboard-compensate",
        )
    assert failed.value.code == "write_conflict"
    assert "raw internal failure" not in str(failed.value.to_envelope())
    assert host.list_projects() == []
    with kb.scoped_kanban_home(root):
        assert not kb.board_exists("compensate")

    monkeypatch.setattr(pdb, "create_project", create_project)
    retried = host.provision_project(
        **payload,
        idempotency_key="dockyard-onboard-compensate",
    )
    assert retried["status"] == "complete"
