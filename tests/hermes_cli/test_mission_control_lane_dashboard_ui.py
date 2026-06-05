from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PANEL_TSX = ROOT / "web" / "src" / "components" / "MissionControlLaneDashboard.tsx"
MISSION_TSX = ROOT / "web" / "src" / "pages" / "MissionControlPage.tsx"
API_TS = ROOT / "web" / "src" / "lib" / "api.ts"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_lane_dashboard_component_renders_required_read_only_sections():
    source = _read(PANEL_TSX)

    for phrase in (
        "Mission Control Lane Dashboard",
        "Active lane",
        "Task Control Envelope",
        "Approval tier",
        "Allowed actions",
        "Forbidden actions",
        "Start Gate",
        "Guard decision",
        "Next action type",
        "Evidence summaries",
        "Next recommended action",
        "Token/context budget",
        "Quarantine / parent-scan warning",
        "Read-only API/UI only",
        "trusted_for_execution=false",
        "inert_context_only=true",
    ):
        assert phrase in source


def test_lane_dashboard_keeps_evidence_details_on_demand():
    source = _read(PANEL_TSX)

    assert "<details" in source
    assert "Evidence details are collapsed until requested." in source
    assert "details_on_demand" in source


def test_lane_dashboard_omits_transcript_and_execution_controls():
    source = _read(PANEL_TSX)

    for label in (
        ">Run<",
        ">Execute<",
        ">Deploy<",
        ">Restart<",
        ">Merge<",
        ">Commit<",
        ">Push<",
        ">Approve<",
        ">Load transcript<",
        ">Scan parent<",
        ">Open quarantine<",
    ):
        assert label not in source

    for token in (
        "<button",
        "<form",
        "<input",
        "<select",
        "<textarea",
        "onClick",
        "executeOpsApprovalAction",
        "dryRunOpsApprovalAction",
        "getSessionMessages",
        "getSessionLatestDescendant",
        "window.location",
        "nextAction",
    ):
        assert token not in source


def test_lane_dashboard_uses_get_only_client_helper():
    source = _read(PANEL_TSX)
    api_source = _read(API_TS)

    assert "api.getMissionControlLaneDashboard()" in source
    assert "getMissionControlLaneDashboard: () =>" in api_source
    assert 'fetchJSON<MissionControlLaneDashboardResponse>("/api/mission-control/lane-dashboard", { cache: "no-store" })' in api_source

    for token in ("fetch(", "XMLHttpRequest", "method:", "POST", "PUT", "PATCH", "DELETE"):
        assert token not in source


def test_mission_control_page_mounts_lane_dashboard():
    source = _read(MISSION_TSX)

    assert "MissionControlLaneDashboard" in source
    assert "<MissionControlLaneDashboard />" in source
