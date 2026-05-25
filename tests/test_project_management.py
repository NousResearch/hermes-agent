from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1] / "plugins" / "project-management"
sys.path.insert(0, str(PLUGIN_DIR))
import project_management as pm
from hermes_cli import kanban_db as kb  # type: ignore[import-not-found]


@pytest.fixture
def pm_module(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    projects_root = tmp_path / "projects"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_PROJECTS_ROOT", str(projects_root))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return pm


@pytest.fixture
def good_payload():
    return {
        "project_name": "Demo Project",
        "project_type": "web app",
        "goal": "Build a demo project",
        "tech_stack": "Python, React",
        "target_users": "testers",
        "auth": "no",
        "database": "PostgreSQL",
        "deployment": "local first",
        "constraints": "keep it small",
        "success_criteria": "bootstrap completes",
        "avoid": "approval gates",
        "must_haves": "tests",
    }


def test_parse_project_answers_accepts_structured_text():
    parsed = pm.parse_project_answers(
        """
        Project Name: Demo
        Project Type: web app
        Goal: Build it
        Tech Stack: Python, React
        Auth: yes
        Database: PostgreSQL
        """
    )

    assert parsed["project_name"] == "Demo"
    assert parsed["project_type"] == "web app"
    assert parsed["goal"] == "Build it"
    assert parsed["tech_stack"] == "Python, React"
    assert parsed["auth"] == "yes"
    assert parsed["database"] == "PostgreSQL"


def test_parse_project_answers_accepts_flag_form():
    parsed = pm.parse_project_answers('--name ToDoist --type "web app" --goal "Track tasks" --stack "Next.js, Prisma"')
    assert parsed == {
        "project_name": "ToDoist",
        "project_type": "web app",
        "goal": "Track tasks",
        "tech_stack": "Next.js, Prisma",
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Demo Project", "demo_project"),
        ("Demo   Project", "demo_project"),
        ("Demo-project", "demo-project"),
        ("  Demo—Project  ", "demo_project"),
    ],
)
def test_slugify_name_normalizes_spaces_and_punctuation(raw: str, expected: str):
    assert pm._slugify_name(raw) == expected


def test_parse_tech_stack_splits_common_delimiters():
    assert pm._parse_tech_stack("Python, FastAPI / PostgreSQL + React\nRedis") == [
        "Python",
        "FastAPI",
        "PostgreSQL",
        "React",
        "Redis",
    ]


def test_serena_language_ids_maps_frameworks_and_deduplicates():
    assert pm._serena_language_ids(["Python", "React", "Node.js", "Go", "FastAPI", "Vue"]) == [
        "python",
        "typescript",
        "javascript",
        "go",
    ]


def test_apply_defaults_records_assumptions():
    payload, decisions = pm._apply_defaults({"project_name": "Demo", "project_type": "web app", "goal": "Build"})
    assert payload["tech_stack"]
    assert payload["auth"]
    assert any("Assumed tech stack" in d for d in decisions)
    assert any("approval" in payload["avoid"] for _ in [0])


def test_write_serena_project_yml_creates_expected_file(tmp_path):
    project_dir = tmp_path / "demo"
    project_dir.mkdir()

    pm._write_serena_project_yml(project_dir, "Demo Project", ["python", "typescript", "go"])

    project_yml = (project_dir / ".serena" / "project.yml").read_text(encoding="utf-8")
    assert 'project_name: "Demo Project"' in project_yml
    assert "languages:" in project_yml
    assert '  - "python"' in project_yml
    assert '  - "typescript"' in project_yml
    assert '  - "go"' in project_yml
    assert "ignored_paths:" in project_yml


def test_write_serena_project_yml_quotes_special_scalars(tmp_path):
    project_dir = tmp_path / "demo"
    project_dir.mkdir()

    pm._write_serena_project_yml(project_dir, "Demo \"Project\":\n# not yaml", ["python"])

    project_yml = (project_dir / ".serena" / "project.yml").read_text(encoding="utf-8")
    assert 'project_name: "Demo \\"Project\\":\\n# not yaml"' in project_yml


def test_create_project_writes_bootstrap_files_beads_and_kanban(pm_module, good_payload, monkeypatch):
    def fake_docs(tokens, docs_dir):
        docs_dir.mkdir(parents=True, exist_ok=True)
        path = docs_dir / "react.md"
        path.write_text("# React docs", encoding="utf-8")
        return [str(path)], []

    def fake_run(cmd, *, cwd, timeout=120):
        if cmd[:2] == ["git", "init"]:
            (Path(cwd) / ".git").mkdir()
            return True, ""
        if cmd[:2] == ["bd", "init"]:
            (Path(cwd) / ".beads").mkdir()
            return True, ""
        if cmd and cmd[0] == "bd" and cmd[1] == "create":
            title = cmd[2]
            ident = "bd-" + title.lower().replace(" ", "-").replace(":", "").replace("/", "-")[:16]
            return True, json.dumps({"id": ident})
        if cmd[:3] == ["serena", "project", "index"]:
            return True, "indexed"
        return True, ""

    monkeypatch.setattr(pm_module, "_fetch_context7_docs", fake_docs)
    monkeypatch.setattr(pm_module, "_run", fake_run)

    result = pm_module.create_project(payload=good_payload)

    assert result.ok, result.message
    project_dir = Path(result.data["project_path"])
    manifest = json.loads((project_dir / ".hermes-project.json").read_text(encoding="utf-8"))

    assert (project_dir / "README.md").exists()
    assert (project_dir / "project-brief.md").exists()
    assert (project_dir / "tech-stack.md").exists()
    assert (project_dir / "decisions.md").exists()
    assert (project_dir / "AGENTS.md").exists()
    assert (project_dir / "docs" / "design.md").exists()
    assert (project_dir / ".serena" / "project.yml").exists()
    assert (project_dir / ".beads").exists()

    assert manifest["project_name"] == good_payload["project_name"]
    assert manifest["slug"] == "demo_project"
    assert manifest["status"] == "active"
    assert manifest["board_slugs"] == ["demo_project"]
    assert manifest["beads_issues"]
    assert manifest["kanban_tasks"]
    assert manifest["project_memory_page"].endswith("/project-memory/projects/demo_project.md")
    assert manifest["codebase_memory_project"] == "demo_project"
    assert result.data["bd_ok"] is True
    assert result.data["serena_ok"] is True
    assert result.data["docs_fetched"]

    agents_md = (project_dir / "AGENTS.md").read_text(encoding="utf-8")
    assert "No task may wait for user approval" in agents_md
    assert "Beads is canonical" in agents_md
    assert "Serena is the code-intelligence layer" in agents_md
    assert "Designer must create `docs/design.md`" in agents_md

    design_md = (project_dir / "docs" / "design.md").read_text(encoding="utf-8")
    assert "Create a mockup image" in design_md
    assert "Frontend/UI tickets must use" in design_md

    memory_page = Path(result.data["project_memory_page"])
    assert memory_page.exists()
    memory_md = memory_page.read_text(encoding="utf-8")
    assert "Codebase Memory MCP project: `demo_project`" in memory_md
    assert "[[references/design-mockup-workflow]]" in memory_md
    memory_index = (Path(result.data["project_memory_root"]) / "index.md").read_text(encoding="utf-8")
    assert "[[projects/demo_project]]" in memory_index

    with kb.connect(board=result.data["board_slug"]) as conn:
        tasks = kb.list_tasks(conn)
    assert len(tasks) == 10
    assert any(task.title == "craft mockup website design prompt and reference" for task in tasks)
    assert all("approval" not in task.title.lower() for task in tasks)


def test_create_project_defaults_optional_fields(pm_module, monkeypatch):
    monkeypatch.setattr(pm_module, "_fetch_context7_docs", lambda tokens, docs_dir: ([], []))

    def fake_run(cmd, *, cwd, timeout=120):
        if cmd[:2] == ["bd", "init"]:
            (Path(cwd) / ".beads").mkdir()
        if cmd and cmd[0] == "bd" and len(cmd) > 1 and cmd[1] == "create":
            return True, json.dumps({"id": "bd-x" + str(abs(hash(cmd[2])) % 100000)})
        return True, ""

    monkeypatch.setattr(pm_module, "_run", fake_run)
    result = pm_module.create_project(payload={"project_name": "Defaults", "project_type": "API", "goal": "Serve data"})
    assert result.ok, result.message
    assert result.data["decisions"]
    decisions = (Path(result.data["project_path"]) / "decisions.md").read_text(encoding="utf-8")
    assert "Assumed tech stack" in decisions
    assert "FastAPI" in decisions


def test_create_project_rolls_back_board_on_manifest_failure(pm_module, good_payload, monkeypatch):
    monkeypatch.setattr(pm_module, "_fetch_context7_docs", lambda tokens, docs_dir: ([], []))
    monkeypatch.setattr(pm_module, "_run", lambda cmd, *, cwd, timeout=120: (True, json.dumps({"id": "bd-x"}) if cmd and cmd[0] == "bd" and len(cmd) > 1 and cmd[1] == "create" else ""))

    def explode(*args, **kwargs):
        raise RuntimeError("manifest write failed")

    monkeypatch.setattr(pm_module, "_save_manifest", explode)
    result = pm_module.create_project(payload=good_payload)

    assert not result.ok
    slug = pm_module._slugify_name(good_payload["project_name"])
    assert not (pm_module.projects_root() / slug).exists()
    assert not (kb.boards_root() / slug).exists()


def test_archive_then_delete_project_moves_project_to_trash(pm_module, good_payload, monkeypatch):
    monkeypatch.setattr(pm_module, "_fetch_context7_docs", lambda tokens, docs_dir: ([], []))

    def fake_run(cmd, *, cwd, timeout=120):
        if cmd[:2] == ["bd", "init"]:
            (Path(cwd) / ".beads").mkdir()
        if cmd and cmd[0] == "bd" and len(cmd) > 1 and cmd[1] == "create":
            return True, json.dumps({"id": "bd-x" + str(abs(hash(cmd[2])) % 100000)})
        return True, ""

    monkeypatch.setattr(pm_module, "_run", fake_run)
    created = pm_module.create_project(payload=good_payload)
    assert created.ok, created.message
    slug = created.data["slug"]

    archived = pm_module.archive_project(good_payload["project_name"])
    assert archived.ok, archived.message
    archived_path = Path(archived.data["project_path"])
    assert archived_path.exists()
    assert not (pm_module.projects_root() / slug).exists()

    deleted = pm_module.delete_project(good_payload["project_name"])
    assert deleted.ok, deleted.message
    assert not archived_path.exists()
    trash_path = Path(deleted.data["project_path"])
    assert trash_path.exists()
    assert trash_path.parent == pm_module.deleted_projects_root()
    deleted_manifest = json.loads((trash_path / ".hermes-project.json").read_text(encoding="utf-8"))
    assert deleted_manifest["status"] == "deleted"


def test_deleteproject_direct_command_requires_exact_confirmation(pm_module, good_payload, monkeypatch):
    monkeypatch.setattr(pm_module, "_fetch_context7_docs", lambda tokens, docs_dir: ([], []))

    def fake_run(cmd, *, cwd, timeout=120):
        if cmd[:2] == ["bd", "init"]:
            (Path(cwd) / ".beads").mkdir()
        if cmd and cmd[0] == "bd" and len(cmd) > 1 and cmd[1] == "create":
            return True, json.dumps({"id": "bd-x" + str(abs(hash(cmd[2])) % 100000)})
        return True, ""

    monkeypatch.setattr(pm_module, "_run", fake_run)
    created = pm_module.create_project(payload=good_payload)
    assert created.ok, created.message
    slug = created.data["slug"]
    project_path = Path(created.data["project_path"])

    preview = pm_module.handle_direct_command("deleteproject", slug)
    assert "Delete confirmation required" in preview
    assert project_path.exists()

    mismatch = pm_module.handle_direct_command("deleteproject", f"{slug} --confirm wrong")
    assert "confirmation mismatch" in mismatch.lower()
    assert project_path.exists()

    confirmed = pm_module.handle_direct_command("deleteproject", f"{slug} --confirm {slug}")
    assert "Project moved to trash" in confirmed
    assert not project_path.exists()
    trash_path = pm_module.deleted_projects_root() / slug
    assert trash_path.exists()


def test_project_choice_replies_use_explicit_commands(pm_module, good_payload, monkeypatch):
    monkeypatch.setattr(pm_module, "_fetch_context7_docs", lambda tokens, docs_dir: ([], []))
    monkeypatch.setattr(pm_module, "_run", lambda cmd, *, cwd, timeout=120: (True, json.dumps({"id": "bd-x"}) if cmd and cmd[0] == "bd" and len(cmd) > 1 and cmd[1] == "create" else ""))
    created = pm_module.create_project(payload=good_payload)
    assert created.ok, created.message

    archive_reply = pm_module.handle_direct_command("archiveproject", "")
    delete_reply = pm_module.handle_direct_command("deleteproject", "")

    assert "Run `/archiveproject <slug>`" in archive_reply
    assert "Run `/deleteproject <slug>`" in delete_reply
    assert "Reply with the project name or slug" not in archive_reply
    assert "Reply with the project name or slug" not in delete_reply


def test_plugin_init_loads_as_hermes_namespace_package(monkeypatch):
    plugin_dir = Path(__file__).resolve().parents[1] / "plugins" / "project-management"
    package_name = "hermes_plugins.project_management_test"
    monkeypatch.setitem(sys.modules, "hermes_plugins", types.ModuleType("hermes_plugins"))
    spec = importlib.util.spec_from_file_location(
        package_name,
        plugin_dir / "__init__.py",
        submodule_search_locations=[str(plugin_dir)],
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, package_name, module)

    spec.loader.exec_module(module)

    assert callable(module.handle_direct_command)
    assert callable(module.handle_gateway_command)
