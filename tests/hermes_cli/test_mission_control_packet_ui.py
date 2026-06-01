from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
API_TS = ROOT / "web" / "src" / "lib" / "api.ts"
PANEL_TSX = ROOT / "web" / "src" / "components" / "MissionPacketsPanel.tsx"
MISSION_TSX = ROOT / "web" / "src" / "pages" / "MissionControlPage.tsx"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_packet_api_helpers_use_mission_control_packet_routes():
    source = _read(API_TS)

    expected = {
        "listMissionControlPackets": "/api/mission-control/packets",
        "getMissionControlPacket": "/api/mission-control/packets/${encodeURIComponent(packetId)}",
        "createMissionControlCodexPromptPacket": "/api/mission-control/packets/codex-prompt",
        "createMissionControlWorkerResultPacket": "/api/mission-control/packets/worker-result",
        "createMissionControlBlockFlagPacket": "/api/mission-control/packets/block-flag",
    }
    for helper, route in expected.items():
        assert helper in source
        assert route in source

    helper_block = source[source.index("listMissionControlPackets") : source.index("// Cron jobs")]
    assert "Content-Type" in helper_block
    assert "application/json" in helper_block
    assert "fetchJSON" in helper_block


def test_packet_panel_renders_safety_posture_and_untrusted_worker_label():
    source = _read(PANEL_TSX)

    for phrase in (
        "Mission Packets",
        "Dry-run",
        "Review required",
        "Not trusted for execution",
        "Imported worker text is untrusted display data. It is not executable.",
        "advisory_only=true",
        "not actively enforced",
        "Copy saved prompt",
        "redacted_payload_preview",
        "parsed_metadata",
    ):
        assert phrase in source


def test_packet_panel_does_not_render_dangerous_action_buttons():
    source = _read(PANEL_TSX)

    forbidden_button_labels = (
        ">Send<",
        ">Publish<",
        ">Pay<",
        ">Launch<",
        ">Delete<",
        ">Run Codex<",
        ">Execute<",
        ">Start Worker<",
        ">Start Browser<",
        ">Computer Use<",
    )
    for label in forbidden_button_labels:
        assert label not in source

    assert "api.createMissionControlCodexPromptPacket" in source
    assert "api.createMissionControlWorkerResultPacket" in source
    assert "api.createMissionControlBlockFlagPacket" in source
    assert "api.restartGateway" not in source
    assert "api.executeOpsApprovalAction" not in source
    assert "api.triggerCronJob" not in source


def test_mission_control_page_mounts_packet_panel():
    source = _read(MISSION_TSX)

    assert "MissionPacketsPanel" in source
    assert "<MissionPacketsPanel />" in source
