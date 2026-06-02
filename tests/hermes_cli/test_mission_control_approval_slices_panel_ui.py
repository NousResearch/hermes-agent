from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
API_TS = ROOT / "web" / "src" / "lib" / "api.ts"
PANEL_TSX = ROOT / "web" / "src" / "components" / "ApprovalSlicesPanel.tsx"
MISSION_TSX = ROOT / "web" / "src" / "pages" / "MissionControlPage.tsx"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_approval_slices_api_helper_uses_get_only_existing_g3_route():
    source = _read(API_TS)

    assert "ApprovalSliceStatus" in source
    assert "ApprovalSliceSummary" in source
    assert "ApprovalSlicesResponse" in source
    assert "listApprovalSlices" in source
    assert (
        'fetchJSON<ApprovalSlicesResponse>("/api/mission-control/approval-slices?include_inactive=true", { cache: "no-store" })'
        in source
    )

    helper_block = source[source.index("listApprovalSlices") : source.index("createMissionBrief")]
    for token in ("method:", "POST", "PUT", "PATCH", "DELETE"):
        assert token not in helper_block


def test_approval_slices_panel_renders_existing_g3_records_and_statuses():
    source = _read(PANEL_TSX)

    for phrase in (
        "api.listApprovalSlices",
        "Approval Slices",
        "G3 decision records",
        "active",
        "revoked",
        "expired",
        "completed",
    ):
        assert phrase in source


def test_approval_slices_panel_renders_required_inert_fields():
    source = _read(PANEL_TSX)

    for phrase in (
        "allowed_actions",
        "forbidden_actions",
        "stop_condition",
        "checkpoint",
        "Goal Contract ID",
        "trusted_for_execution=false",
        "inert_context_only=true",
        "Decision record only — does not authorize execution",
    ):
        assert phrase in source


def test_approval_slices_panel_renders_loading_empty_and_error_states():
    source = _read(PANEL_TSX)

    for phrase in (
        "Loading approval slices.",
        "No Approval Slice records found.",
        "Approval Slice records could not be loaded.",
    ):
        assert phrase in source


def test_approval_slices_panel_goal_contract_id_is_plain_text_not_link():
    source = _read(PANEL_TSX)

    assert "linked_goal_contract_id" in source
    assert "Goal Contract ID" in source

    for token in ("<a ", "href=", "Link to goal", "to={`/goal", "to=\"/goal"):
        assert token not in source


def test_approval_slices_panel_omits_forbidden_controls_and_mutations():
    source = _read(PANEL_TSX)

    forbidden_visible_text = (
        ">Approve<",
        ">Authorize<",
        ">Run<",
        ">Execute<",
        ">Start<",
        ">Continue<",
        ">Grant<",
        ">Revoke<",
        ">Expire<",
        ">Complete<",
        ">Cancel Approval<",
        ">Request Approval<",
        ">Create Slice<",
        ">New Approval Slice<",
        ">Edit<",
        ">Delete<",
        ">Enable<",
        ">Disable<",
        ">Launch<",
        ">Deploy<",
        ">Restart<",
        ">Push<",
        ">Publish<",
        ">Sync<",
    )
    for label in forbidden_visible_text:
        assert label not in source

    forbidden_code = (
        "<button",
        "<form",
        "<input",
        "<select",
        "<textarea",
        "onClick",
        "role=\"button\"",
        "createApproval",
        "revokeApproval",
        "expireApproval",
        "completeApproval",
        "fetch(",
        "method:",
        "POST",
        "PUT",
        "PATCH",
        "DELETE",
    )
    for token in forbidden_code:
        assert token not in source


def test_mission_control_page_mounts_approval_slices_panel():
    source = _read(MISSION_TSX)

    assert "ApprovalSlicesPanel" in source
    assert "<ApprovalSlicesPanel />" in source
    assert source.index("<ApprovalSlicesPanel />") > source.index("<ActiveEnvelopePanel />")
