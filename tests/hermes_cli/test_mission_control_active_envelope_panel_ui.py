from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PANEL_TSX = ROOT / "web" / "src" / "components" / "ActiveEnvelopePanel.tsx"
MISSION_TSX = ROOT / "web" / "src" / "pages" / "MissionControlPage.tsx"
API_TS = ROOT / "web" / "src" / "lib" / "api.ts"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_active_envelope_panel_renders_empty_state_heading():
    source = _read(PANEL_TSX)

    assert "No Active Task Envelope" in source


def test_active_envelope_panel_renders_required_empty_state_copy():
    source = _read(PANEL_TSX)

    for phrase in (
        "Display only",
        "No active authorization",
        "Operational actions locked",
        "Discussion/status only",
        "Active lane: Unset",
        "Active mode: Unset",
        "Execution boundary: No active authorization",
        "Allowed actions: None declared",
        "Forbidden actions: Unknown",
        "Checkpoint: None",
        "Repo state: Unknown / not probed",
        "Evidence: No envelope evidence attached",
        "Data source: No persisted envelope",
        "No persisted envelope",
        "exists=false",
    ):
        assert phrase in source


def test_active_envelope_panel_renders_exists_true_metadata():
    source = _read(PANEL_TSX)

    for phrase in (
        "exists=true",
        "Active lane",
        "Active mode",
        "Checkpoint",
        "Data source",
        "Envelope ID",
        "Schema",
        "Status",
        "Title",
        "Mode label",
        "Created at",
        "Updated at",
    ):
        assert phrase in source


def test_active_envelope_panel_renders_selection_metadata():
    source = _read(PANEL_TSX)

    for phrase in (
        "selection_reason",
        "selected_from_count",
        "ambiguous=true",
        "Multiple persisted envelopes matched; newest active envelope is being shown as inert context.",
    ):
        assert phrase in source


def test_active_envelope_panel_renders_required_safety_labels():
    source = _read(PANEL_TSX)

    for phrase in (
        "trusted_for_execution=false",
        "inert_context_only=true",
        "non-authorizing",
    ):
        assert phrase in source


def test_active_envelope_panel_renders_inert_loading_and_error_states():
    source = _read(PANEL_TSX)

    for phrase in (
        "Loading task envelope.",
        "Task envelope could not be loaded.",
    ):
        assert phrase in source

    assert "loading" in source
    assert "error" in source


def test_active_envelope_panel_omits_forbidden_labels_and_controls():
    source = _read(PANEL_TSX)

    for label in (
        ">Approve<",
        ">Authorize<",
        ">Run<",
        ">Execute<",
        ">Deploy<",
        ">Preflight<",
        ">Validate<",
        ">Trust<",
        ">Use for execution<",
        ">Link Evidence Card<",
        ">Resolve ambiguity<",
        ">Select envelope<",
        ">Refresh<",
        ">Start<",
        ">Launch<",
        ">Push<",
        ">Open<",
        ">Edit<",
        ">Create<",
        ">Sync<",
        ">Publish<",
    ):
        assert label not in source

    for token in (
        "<button",
        "<form",
        "<input",
        "<select",
        "<textarea",
        "<a ",
        "href=",
        "onClick",
        "role=\"button\"",
    ):
        assert token not in source


def test_active_envelope_panel_uses_get_only_helper_and_no_mutation_helpers():
    source = _read(PANEL_TSX)
    api_source = _read(API_TS)

    assert "@/lib/api" in source
    assert "api.getMissionControlActiveEnvelope()" in source
    assert "getMissionControlActiveEnvelope: () =>" in api_source
    assert 'fetchJSON<MissionControlActiveEnvelopeResponse>("/api/mission-control/active-envelope", { cache: "no-store" })' in api_source

    for token in ("fetch(", "XMLHttpRequest", "method:", "POST", "PUT", "PATCH", "DELETE"):
        assert token not in source


def test_mission_control_page_mounts_active_envelope_panel():
    source = _read(MISSION_TSX)

    assert "ActiveEnvelopePanel" in source
    assert "<ActiveEnvelopePanel />" in source
