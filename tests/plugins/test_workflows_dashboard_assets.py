from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "plugins" / "workflows" / "dashboard" / "dist" / "index.js"


def test_workflow_dashboard_renders_assistant_error_hints() -> None:
    text = BUNDLE.read_text(encoding="utf-8")
    assert "workflow_assistant_runtime_error" in text
    assert "Check workflow assistant provider/model configuration" in text
    assert "Advanced YAML" in text


def test_workflow_dashboard_contains_privacy_warning() -> None:
    text = BUNDLE.read_text(encoding="utf-8")
    assert "Workflow inputs and outputs are stored locally" in text
    assert "Do not paste secrets" in text


def test_workflow_dashboard_has_accessible_cell_editor_path() -> None:
    text = BUNDLE.read_text(encoding="utf-8")
    assert "Workflow cell list" in text
    assert "Edit cell" in text
    assert "aria-label" in text
    assert "Edit common cell settings here" in text
    assert "Apply cell changes" in text
    assert "This node type does not have a prompt form yet" not in text


def test_workflow_dashboard_hides_default_reactflow_grid_overlays() -> None:
    text = BUNDLE.read_text(encoding="utf-8")
    css = (ROOT / "plugins" / "workflows" / "dashboard" / "dist" / "style.css").read_text(encoding="utf-8")
    assert "Background ? h(Background" not in text
    assert "MiniMap ? h(MiniMap" not in text
    assert ".react-flow__background" in css
    assert "display: none !important" in css


def test_workflow_dashboard_has_responsive_editor_css() -> None:
    css = (ROOT / "plugins" / "workflows" / "dashboard" / "dist" / "style.css").read_text(encoding="utf-8")
    assert "hermes-workflows-editor-layout" in css
    assert "@media" in css


def test_workflow_dashboard_cell_editor_exposes_agent_routing_controls() -> None:
    text = BUNDLE.read_text(encoding="utf-8")
    assert "Assigned profile" in text
    assert "Provider override" in text
    assert "Model override" in text
    assert "Use profile default provider" in text
    assert "Use profile default model" in text
    assert "/agent-routing-options" in text
