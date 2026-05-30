from gateway.dev_control.chat_project_context import build_chat_project_context_overlay


def test_build_chat_project_context_overlay_includes_snapshot_and_structured_context():
    overlay = build_chat_project_context_overlay({
        "project_id": "OrynWorkspace",
        "coordinator_profile": "dev",
        "project_context": {
            "project_id": "OrynWorkspace",
            "project_name": "Oryn Workspace",
            "vision": "Ship project-aware planning.",
            "coordinator_profile": "dev",
            "repositories": [{"label": "Workspace", "path": "/tmp/Oryn"}],
            "work_items": ["Shape backlog [planned · Backlog]"],
        },
        "project_dashboard_snapshot": "## Active workers\n- No active workers",
    })

    assert overlay is not None
    assert "Scoped project_id: OrynWorkspace" in overlay
    assert "Coordinator profile: dev" in overlay
    assert "Ship project-aware planning." in overlay
    assert "/tmp/Oryn" in overlay
    assert "Shape backlog" in overlay
    assert "## Active workers" in overlay


def test_build_chat_project_context_overlay_returns_none_without_payload():
    assert build_chat_project_context_overlay(None) is None
    assert build_chat_project_context_overlay({}) is None
