from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

PLUGIN_PATH = Path(__file__).resolve().parents[1] / "plugins" / "memory" / "hermes-memory" / "__init__.py"
SPEC = spec_from_file_location("hermes_memory_plugin", PLUGIN_PATH)
MODULE = module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)
_project_scope_from_cwd = MODULE._project_scope_from_cwd


def test_project_scope_prefers_multica_project_id(monkeypatch, tmp_path):
    monkeypatch.setenv("MULTICA_PROJECT_ID", "fc7a7002-d23f-4718-8ae7-4724652533c9")
    monkeypatch.delenv("HERMES_MEMORY_PROJECT_SCOPE", raising=False)
    monkeypatch.delenv("HERMES_MEMORY_PROJECT_ID", raising=False)

    scope = _project_scope_from_cwd(tmp_path)

    assert scope == "project:fc7a7002-d23f-4718-8ae7-4724652533c9"


def test_project_scope_uses_origin_url_not_worktree_path(monkeypatch, tmp_path):
    repo1 = tmp_path / "ws1" / "court-booking-management"
    repo2 = tmp_path / "ws2" / "court-booking-management"
    repo1.mkdir(parents=True)
    repo2.mkdir(parents=True)
    (repo1 / ".git").mkdir()
    (repo2 / ".git").mkdir()

    monkeypatch.delenv("HERMES_MEMORY_PROJECT_SCOPE", raising=False)
    monkeypatch.delenv("HERMES_MEMORY_PROJECT_ID", raising=False)
    monkeypatch.delenv("MULTICA_PROJECT_ID", raising=False)

    def fake_run_git(args, cwd: Path):
        if args == ["remote", "get-url", "origin"]:
            return "https://github.com/dragonguy888/court-booking-management.git"
        if args == ["rev-parse", "--git-common-dir"]:
            return "/stable/common.git"
        return ""

    monkeypatch.setattr(MODULE, "_run_git", fake_run_git)
    monkeypatch.setattr(MODULE, "_run_multica", lambda args, cwd: "")

    scope1 = _project_scope_from_cwd(repo1)
    scope2 = _project_scope_from_cwd(repo2)

    assert scope1 == scope2
    assert scope1 == "project:court-booking-management-498ec503"


def test_project_scope_discovers_single_repo_inside_workdir(monkeypatch, tmp_path):
    workdir = tmp_path / "task-workdir"
    repo = workdir / "court-booking-management"
    repo.mkdir(parents=True)
    (repo / ".git").mkdir()

    monkeypatch.delenv("HERMES_MEMORY_PROJECT_SCOPE", raising=False)
    monkeypatch.delenv("HERMES_MEMORY_PROJECT_ID", raising=False)
    monkeypatch.delenv("MULTICA_PROJECT_ID", raising=False)

    def fake_run_git(args, cwd: Path):
        if args == ["remote", "get-url", "origin"]:
            return "https://github.com/dragonguy888/court-booking-management.git"
        if args == ["rev-parse", "--git-common-dir"]:
            return "/stable/common.git"
        return ""

    monkeypatch.setattr(MODULE, "_run_git", fake_run_git)
    monkeypatch.setattr(MODULE, "_run_multica", lambda args, cwd: "")

    scope = _project_scope_from_cwd(workdir)

    assert scope == "project:court-booking-management-498ec503"


def test_project_scope_uses_multica_issue_context_when_repo_missing(monkeypatch, tmp_path):
    workdir = tmp_path / "task-workdir"
    ctx = workdir / ".agent_context"
    ctx.mkdir(parents=True)
    (ctx / "issue_context.md").write_text(
        "# Task Assignment\n\n**Issue ID:** c1fad34d-b41b-4a29-96ed-5f9a352c275c\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("HERMES_MEMORY_PROJECT_SCOPE", raising=False)
    monkeypatch.delenv("HERMES_MEMORY_PROJECT_ID", raising=False)
    monkeypatch.delenv("MULTICA_PROJECT_ID", raising=False)
    monkeypatch.delenv("MULTICA_ISSUE_ID", raising=False)

    def fake_run_multica(args, cwd: Path):
        assert args == ["issue", "get", "c1fad34d-b41b-4a29-96ed-5f9a352c275c", "--output", "json"]
        return '{"project_id": "fc7a7002-d23f-4718-8ae7-4724652533c9"}'

    monkeypatch.setattr(MODULE, "_run_multica", fake_run_multica)

    scope = _project_scope_from_cwd(workdir)

    assert scope == "project:fc7a7002-d23f-4718-8ae7-4724652533c9"


def test_project_scope_uses_single_multica_workspace_project_for_repo(monkeypatch, tmp_path):
    repo = tmp_path / "court-booking-management"
    repo.mkdir(parents=True)
    (repo / ".git").mkdir()

    monkeypatch.delenv("HERMES_MEMORY_PROJECT_SCOPE", raising=False)
    monkeypatch.delenv("HERMES_MEMORY_PROJECT_ID", raising=False)
    monkeypatch.delenv("MULTICA_PROJECT_ID", raising=False)
    monkeypatch.delenv("MULTICA_ISSUE_ID", raising=False)

    def fake_run_git(args, cwd: Path):
        if args == ["remote", "get-url", "origin"]:
            return "https://github.com/dragonguy888/court-booking-management.git"
        if args == ["rev-parse", "--git-common-dir"]:
            return "/stable/common.git"
        return ""

    def fake_run_multica(args, cwd: Path):
        if args == ["workspace", "get", "--output", "json"]:
            return '{"repos":[{"url":"https://github.com/dragonguy888/court-booking-management.git"}]}'
        if args == ["project", "list", "--output", "json"]:
            return '[{"id":"fc7a7002-d23f-4718-8ae7-4724652533c9"}]'
        return ""

    monkeypatch.setattr(MODULE, "_run_git", fake_run_git)
    monkeypatch.setattr(MODULE, "_run_multica", fake_run_multica)

    scope = _project_scope_from_cwd(repo)

    assert scope == "project:fc7a7002-d23f-4718-8ae7-4724652533c9"
