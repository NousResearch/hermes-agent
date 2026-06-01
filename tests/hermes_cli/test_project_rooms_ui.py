from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
API_TS = ROOT / "web" / "src" / "lib" / "api.ts"
PANEL_TSX = ROOT / "web" / "src" / "components" / "ProjectRoomsPanel.tsx"
MISSION_TSX = ROOT / "web" / "src" / "pages" / "MissionControlPage.tsx"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_project_rooms_api_helpers_use_token_gated_routes():
    source = _read(API_TS)

    for phrase in (
        "listProjectRooms",
        "/api/mission-control/project-rooms",
        "createProjectRoom",
        "listProjectRoomMessages",
        "addProjectRoomMessage",
        "uploadProjectRoomAttachment",
        "getProjectRoomAttachment",
        "downloadProjectRoomAttachment",
        "fetchJSON",
    ):
        assert phrase in source


def test_project_rooms_panel_renders_inert_workspace_without_forbidden_controls():
    source = _read(PANEL_TSX)

    for phrase in (
        "Project Rooms",
        "local inert project context",
        "trusted_for_execution=false",
        "Paste text, logs, and notes",
        "Attach files",
        "Copy text",
        "Mission Packet IDs",
        "No execution path is attached",
    ):
        assert phrase in source

    forbidden = (
        ">Run<",
        ">Execute<",
        ">Send<",
        ">Publish<",
        ">Pay<",
        ">Start Codex<",
        ">Start Worker<",
        ">Start Hermes Run<",
        ">Browser Control<",
        ">Computer Use<",
    )
    for label in forbidden:
        assert label not in source

    assert "api.restartGateway" not in source
    assert "api.executeOpsApprovalAction" not in source
    assert "api.triggerCronJob" not in source


def test_mission_control_page_mounts_project_rooms_panel():
    source = _read(MISSION_TSX)

    assert "ProjectRoomsPanel" in source
    assert "<ProjectRoomsPanel />" in source
