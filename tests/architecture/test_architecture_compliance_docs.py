import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "website" / "docs" / "developer-guide"
PR_TEMPLATE = ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md"
SIDEBARS = ROOT / "website" / "sidebars.ts"
TOOLS = ROOT / "tools"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def iter_tool_register_calls():
    for path in sorted(TOOLS.glob("*.py")):
        tree = ast.parse(read(path), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "register":
                continue
            if not isinstance(node.func.value, ast.Name):
                continue
            if node.func.value.id != "registry":
                continue
            yield path, node


def test_architecture_checklist_doc_covers_required_review_topics():
    checklist = DOCS / "architecture-checklist.md"

    assert checklist.exists(), "developer architecture checklist doc is missing"

    text = read(checklist)
    required_phrases = [
        "Subsystem ownership",
        "Prompt stability",
        "Profile isolation",
        "Optional subsystems",
        "Dependency direction",
        "Adding a tool",
        "Adding a provider",
        "Adding a platform",
        "Adding persistent state",
        "Validation checklist",
    ]
    missing = [phrase for phrase in required_phrases if phrase not in text]
    assert missing == []


def test_architecture_page_links_to_adherence_checklist():
    architecture = read(DOCS / "architecture.md")

    assert "Architecture Adherence Checklist" in architecture
    assert "./architecture-checklist" in architecture


def test_sidebar_exposes_architecture_adherence_checklist():
    sidebar = read(SIDEBARS)

    architecture_index = sidebar.index("'developer-guide/architecture'")
    checklist_index = sidebar.index("'developer-guide/architecture-checklist'")
    agent_loop_index = sidebar.index("'developer-guide/agent-loop'")

    assert architecture_index < checklist_index < agent_loop_index


def test_pull_request_template_requires_architecture_review():
    template = read(PR_TEMPLATE)

    required_lines = [
        "### Architecture",
        "Change is located in the owning subsystem",
        "Core agent behavior remains platform-agnostic",
        "Prompt stability/caching assumptions are preserved",
        "Profile-aware paths are used",
        "Optional integrations are gated",
        "New tools self-register and are exposed through toolsets intentionally",
        "Session writes go through the session/state layer",
    ]
    missing = [line for line in required_lines if line not in template]
    assert missing == []


def test_tool_registrations_declare_architectural_metadata():
    missing = []
    for path, call in iter_tool_register_calls():
        keywords = {keyword.arg for keyword in call.keywords if keyword.arg}
        for required in ("name", "toolset", "schema", "handler"):
            if required not in keywords:
                relpath = path.relative_to(ROOT).as_posix()
                missing.append(f"{relpath}:{call.lineno} missing {required}=")

    assert missing == []
