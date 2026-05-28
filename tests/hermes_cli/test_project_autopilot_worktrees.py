import json
from pathlib import Path

import pytest

from hermes_cli.project_autopilot import (
    InvariantError,
    ProjectAutopilotError,
    generate_cleanup_inventory,
    missing_patch_ids,
    normalize_project_doc,
    parse_patch_id_output,
    write_json,
    validate_workspace_contract,
)


def test_workspace_contract_rejects_wrong_path():
    contract = {
        "task_id": "t_1",
        "workspace_path": "/tmp/demo",
        "branch_name": "kanban/fix-this-shit",
        "base_ref": "origin/main",
    }
    repo = {"worktree_namespace": "/Users/vsletten/src/summation/Code"}

    with pytest.raises(ProjectAutopilotError, match="branch-derived path"):
        validate_workspace_contract(contract, repo=repo)


def test_workspace_contract_accepts_branch_derived_path():
    contract = {
        "task_id": "t_1",
        "workspace_path": "/Users/vsletten/src/summation/Code/kanban/fix-this-shit",
        "branch_name": "kanban/fix-this-shit",
        "base_ref": "origin/main",
    }
    repo = {"worktree_namespace": "/Users/vsletten/src/summation/Code"}

    validate_workspace_contract(contract, repo=repo)


def test_workspace_contract_requires_metadata_fields():
    contract = {
        "task_id": "t_1",
        "workspace_path": "/Users/vsletten/src/summation/Code/kanban/fix-this-shit",
        "branch_name": "kanban/fix-this-shit",
    }
    repo = {"worktree_namespace": "/Users/vsletten/src/summation/Code"}

    with pytest.raises(ProjectAutopilotError, match="workspace contract missing base_ref"):
        validate_workspace_contract(contract, repo=repo)


def test_parse_patch_id_output_returns_first_column_ids():
    output = """
    abc123 commit-one
    def456 commit-two

    abc123 duplicate-commit
    """

    assert parse_patch_id_output(output) == {"abc123", "def456"}


def test_missing_patch_ids_returns_task_ids_absent_from_final_branch():
    task_patch_ids = {"abc123", "def456", "fed999"}
    final_patch_ids = {"def456", "zzz000"}

    assert missing_patch_ids(task_patch_ids, final_patch_ids) == {"abc123", "fed999"}


def _write_project_with_task_graph(
    project_home: Path,
    *,
    worktree_namespace: Path,
    nodes: list[dict],
    final_worktree_path: Path | None = None,
    canonical_checkout: Path | None = None,
) -> dict:
    for dirname in ("status", "refs", "scratch", "artifacts"):
        (project_home / dirname).mkdir(parents=True, exist_ok=True)
    for filename in (
        "PROJECT.md",
        "SESSION-HANDOFF.md",
        "SESSION-LOG.md",
        "PARKING-LOT.md",
        "TASKS.md",
    ):
        (project_home / filename).write_text(f"# {filename}\n", encoding="utf-8")
    (project_home / "STATUS.md").write_text(
        "# Status\n\n## Next action\n\nInspect cleanup inventory.\n",
        encoding="utf-8",
    )
    doc = normalize_project_doc(
        slug="demo",
        title="Demo",
        goal="Demo goal",
        board_slug="demo",
        root_task_id="t_root",
        project_home=project_home,
        repo_org="summation",
        repo_name="Code",
        canonical_checkout=canonical_checkout or worktree_namespace / "main",
        final_branch="feat/demo-pr",
    )
    doc["repo"]["worktree_namespace"] = str(worktree_namespace)
    doc["repo"]["canonical_checkout"] = str(
        canonical_checkout or worktree_namespace / "main"
    )
    doc["final_worktree_path"] = str(
        final_worktree_path or worktree_namespace / "feat" / "demo-pr"
    )
    doc["task_graph"] = {"nodes": nodes, "edges": []}
    write_json(project_home / "project.json", doc)
    return doc


def test_generate_cleanup_inventory_writes_audit_json_and_updates_cleanup_state(
    tmp_path,
):
    project_home = tmp_path / "project"
    namespace = tmp_path / "src" / "summation" / "Code"
    stale_worktree = namespace / "feat" / "slice-one"
    final_worktree = namespace / "feat" / "demo-pr"
    canonical_checkout = namespace / "main"
    sentinel = stale_worktree / "keep.txt"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("do not delete", encoding="utf-8")

    _write_project_with_task_graph(
        project_home,
        worktree_namespace=namespace,
        final_worktree_path=final_worktree,
        canonical_checkout=canonical_checkout,
        nodes=[
            {
                "id": "t_done",
                "title": "slice one",
                "status": "done",
                "workspace_kind": "worktree",
                "workspace_path": str(stale_worktree),
                "branch_name": "feat/slice-one",
            },
            {
                "id": "t_running",
                "title": "running slice",
                "status": "running",
                "workspace_kind": "worktree",
                "workspace_path": str(namespace / "feat" / "slice-two"),
                "branch_name": "feat/slice-two",
            },
            {
                "id": "t_final",
                "title": "final branch",
                "status": "done",
                "workspace_kind": "worktree",
                "workspace_path": str(final_worktree),
                "branch_name": "feat/demo-pr",
            },
            {
                "id": "t_main",
                "title": "canonical checkout",
                "status": "done",
                "workspace_kind": "worktree",
                "workspace_path": str(canonical_checkout),
                "branch_name": "main",
            },
            {
                "id": "t_docs",
                "title": "docs only",
                "status": "done",
                "workspace_kind": "scratch",
                "workspace_path": str(tmp_path / "scratch"),
            },
        ],
    )

    inventory = generate_cleanup_inventory(project_home)

    inventory_path = Path(inventory["inventory_path"])
    assert inventory_path == project_home / "artifacts" / "cleanup" / inventory_path.name
    assert inventory_path.exists()
    saved_inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    assert saved_inventory["project_home"] == str(project_home)
    assert [target["task_id"] for target in saved_inventory["targets"]] == ["t_done"]
    assert saved_inventory["targets"][0]["workspace_path"] == str(stale_worktree)
    assert saved_inventory["exclusions"]["final_worktree_path"] == str(final_worktree)
    assert saved_inventory["exclusions"]["canonical_checkout"] == str(
        canonical_checkout
    )
    assert sentinel.exists()

    saved_doc = json.loads((project_home / "project.json").read_text(encoding="utf-8"))
    assert saved_doc["cleanup"]["state"] == "inventory_ready"
    assert saved_doc["cleanup"]["inventory_path"] == str(inventory_path)
    assert saved_doc["cleanup"]["targets"] == saved_inventory["targets"]


def test_generate_cleanup_inventory_rejects_terminal_worktree_outside_namespace(
    tmp_path,
):
    project_home = tmp_path / "project"
    namespace = tmp_path / "src" / "summation" / "Code"
    outside_worktree = tmp_path / "elsewhere" / "feat" / "slice-one"
    _write_project_with_task_graph(
        project_home,
        worktree_namespace=namespace,
        nodes=[
            {
                "id": "t_done",
                "title": "slice one",
                "status": "done",
                "workspace_kind": "worktree",
                "workspace_path": str(outside_worktree),
                "branch_name": "feat/slice-one",
            }
        ],
    )

    with pytest.raises(InvariantError, match="outside repo.worktree_namespace"):
        generate_cleanup_inventory(project_home)
