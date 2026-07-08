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
    assert "renderInspectorForType" in text
    assert "Apply" in text
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
    assert "@media" in css
    assert "hermes-workflows-app" in css


def test_workflow_dashboard_cell_editor_exposes_agent_routing_controls() -> None:
    text = BUNDLE.read_text(encoding="utf-8")
    assert "Assigned profile" in text
    assert "Provider override" in text
    assert "Model override" in text
    assert "Use profile default provider" in text
    assert "Use profile default model" in text
    assert "/agent-routing-options" in text


def test_workflow_dashboard_summarizes_agent_routing() -> None:
    text = BUNDLE.read_text(encoding="utf-8")
    assert "Provider / model" in text
    assert "providerValue" in text
    assert "modelValue" in text


def test_workflow_dashboard_exposes_ui_only_builder_controls() -> None:
    text = BUNDLE.read_text(encoding="utf-8")
    required_labels = [
        "Start from blank workflow",
        "Add workflow cell",
        "Add trigger",
        "Delete",
        "Add case",
        "Validate draft",
        "Deploy draft",
        "workflow-cell-type-options",
    ]
    for label in required_labels:
        assert label in text
    assert "No JSON/YAML required" in text
    assert "Switch selected cell to: " not in text


def test_workflow_dashboard_uses_palette_instead_of_form_toolbar_clutter() -> None:
    text = BUNDLE.read_text(encoding="utf-8")
    assert "Nodes library" in text
    assert "Drag a node type onto the canvas, or click to add it." in text
    assert "hermes-workflows-node-palette" in text
    assert "Choose a node from the palette, select it on the canvas, then configure it in Properties." in text
    for removed_marker in [
        "placeholder: \"cell id\"",
        "placeholder: \"after\"",
        "placeholder: \"trigger id\"",
        "placeholder: \"from\"",
        "placeholder: \"to\"",
        "Connect cells\")",
    ]:
        assert removed_marker not in text


def test_workflow_dashboard_status_banners_are_dismissible_and_self_clearing() -> None:
    text = BUNDLE.read_text(encoding="utf-8")
    assert "function clearBanners" in text
    assert "\"aria-label\": \"Dismiss alert\"" in text
    assert "hermes-workflows-banner-close" in text
    assert "setTimeout(clearBanners" in text
    assert "formatApiError" in text


def test_workflow_dashboard_shows_ai_draft_routing_context() -> None:
    text = BUNDLE.read_text(encoding="utf-8")
    assert "AI drafts use" in text
    assert "hermes-workflows-ai-routing-note" in text
    assert "agentRoutingOptions.default_provider" in text


def test_workflow_dashboard_blocks_trigger_edges_that_backend_rejects() -> None:
    text = BUNDLE.read_text(encoding="utf-8")
    assert "function isTriggerSource" in text
    assert "Triggers start workflows automatically; connect cells to other cells, not triggers." in text
    assert "if (isTriggerSource(spec, connection.source))" in text


def test_workflow_dashboard_supports_keyboard_delete_and_context_menu() -> None:
    text = BUNDLE.read_text(encoding="utf-8")
    assert "onNodeContextMenu" in text
    assert "contextMenu" in text
    assert "hermes-workflows-context-menu" in text
    assert "Delete cell" in text


def test_workflow_dashboard_prompt_assistant_is_collapsible() -> None:
    text = BUNDLE.read_text(encoding="utf-8")
    assert "promptAssistantAdvanced" in text
    assert "setPromptAssistantAdvanced" in text
    assert "Show advanced fields" in text


def test_workflow_dashboard_inspector_shows_only_relevant_fields() -> None:
    text = BUNDLE.read_text(encoding="utf-8")
    assert "renderInspectorForType" in text
    assert "renderAgentTaskInspector" in text
    assert "renderTriggerInspector" in text
    assert "renderSwitchInspector" in text
    assert "renderWaitInspector" in text
    assert "renderPassFailInspector" in text
    assert "renderMinimalInspector" in text
    assert "\"Cell editor\"" not in text
    assert "hermes-workflows-inspector-header" in text
    assert "hermes-workflows-type-badge" in text


def test_workflow_dashboard_persists_node_positions() -> None:
    text = BUNDLE.read_text(encoding="utf-8")
    assert "function buildFlowNodes" in text
    assert "nodePositions" in text
    assert "setNodePositions" in text
    assert "onNodeDragStop" in text or "onNodesChange" in text
    assert "posMap[id]" in text or "posMap[" in text or "nodePositions[id]" in text or "nodePositions[" in text


def test_workflow_dashboard_supports_drag_from_palette_to_canvas() -> None:
    text = BUNDLE.read_text(encoding="utf-8")
    assert "draggable: true" in text
    assert "onDragStart" in text
    assert "onDragOver" in text
    assert "onDrop" in text
    assert "HERMES_DRAG_NODE_TYPE" in text
    assert "function addWorkflowCellAtPosition" in text
    assert "hermes-workflows-canvas-drop-target" in text


def test_workflow_dashboard_has_no_dead_builder_code() -> None:
    text = BUNDLE.read_text(encoding="utf-8")
    for dead_marker in [
        "function addWorkflowCellFromUi",
        "function addTriggerFromUi",
        "function connectCellsFromUi",
        "function renderBuilderActions",
        'placeholder: "New cell id"',
        'placeholder: "New trigger id"',
        'placeholder: "Connect from source',
        'placeholder: "Connect to target',
    ]:
        assert dead_marker not in text, f"Dead code marker still present: {dead_marker}"


def test_workflow_dashboard_uses_three_zone_builder_layout() -> None:
    text = BUNDLE.read_text(encoding="utf-8")
    css = (ROOT / "plugins" / "workflows" / "dashboard" / "dist" / "style.css").read_text(encoding="utf-8")

    for marker in [
        "function renderTopBar",
        "function renderSidebar",
        "function renderBuilderToolbar",
        "function renderBottomPanel",
        "hermes-workflows-topbar",
        "hermes-workflows-sidebar",
        "hermes-workflows-canvas-area",
        "hermes-workflows-builder-toolbar",
        "hermes-workflows-bottom-panel",
        "sidebarCollapsed.goal === undefined ? !!spec",
        "const stateBottomCollapsed = useState(true)",
        "var persisted = !!(selectedDefinition && workflowIdForDefinition(selectedDefinition)",
    ]:
        assert marker in text

    for marker in [
        ".hermes-workflows .hermes-workflows-app",
        ".hermes-workflows .hermes-workflows-sidebar",
        ".hermes-workflows .hermes-workflows-builder-toolbar",
        ".hermes-workflows .hermes-workflows-flow-surface",
        ".hermes-workflows .hermes-workflows-bottom-panel",
    ]:
        assert marker in css
