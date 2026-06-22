"""Docs invariants for AgentCyber Live USB safety gates."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
LIVE_USB_DIR = ROOT / "live-usb"


def _live_usb_section() -> str:
    text = README.read_text(encoding="utf-8")
    start = text.index("## Live USB")
    end = text.index("## Quick Install", start)
    return text[start:end]


def test_readme_live_usb_section_documents_agent_tool_gates() -> None:
    section = _live_usb_section()
    lowered = section.lower()

    for phrase in (
        "disabled by default",
        "status",
        "list_usb",
        "read-only",
        "build",
        "write",
        "provision",
        "root",
        "exact operator approval",
        "hermes_agentcyber_live_usb_approval",
        "operator_approval",
        "removable linux block-device metadata",
        "canonical",
        "/dev/",
    ):
        assert phrase in lowered


def test_readme_live_usb_examples_do_not_imply_root_alone_is_enough() -> None:
    section = _live_usb_section()
    lowered = section.lower()

    assert "root/sudo alone is not sufficient" in lowered
    assert "unattended cron lanes must not" in lowered
    assert "build`, `write`, and `provision` require" in lowered
    assert "operator_approval=\"<matching one-time token>\"" in section

    obsolete_root_only_claims = (
        "build` and `write` require the agent session to run as root",
        "need no root. `build` and `write` require",
        "sudo live-usb/write_usb.sh --list",
    )
    for obsolete_claim in obsolete_root_only_claims:
        assert obsolete_claim not in lowered


def test_direct_live_usb_scripts_fail_closed_on_unverified_media() -> None:
    write_script = (LIVE_USB_DIR / "write_usb.sh").read_text(encoding="utf-8")
    provision_script = (LIVE_USB_DIR / "provision.sh").read_text(encoding="utf-8")

    for script in (write_script, provision_script):
        lowered = script.lower()
        assert "readlink -f --" in script
        assert "target must resolve to a canonical /dev/... block device" in lowered
        assert "target must be a whole removable disk" in lowered
        assert "root/operator approval is not enough" in lowered
        assert "removable" in script
        assert '"$removable" != "1"' in script
        assert "refusing to" in lowered

    root_guidance = (
        "Root/sudo alone is not sufficient; the target must canonicalize "
        "to a whole removable /dev disk with removable=1."
    )
    assert root_guidance in write_script
    assert root_guidance in provision_script

    assert "WARNING: /sys/block" not in write_script
    assert 'PROVISION_PART="$(_partition_path "$DEVICE" 3)"' in provision_script


def test_direct_build_script_rejects_dev_or_block_output_targets() -> None:
    build_script = (LIVE_USB_DIR / "build_iso.sh").read_text(encoding="utf-8")
    lowered = build_script.lower()

    assert "reject_unsafe_output_target" in build_script
    assert build_script.count('reject_unsafe_output_target "$OUTPUT" || exit 1') >= 3
    assert '[[ -b "$output" ]]' in build_script
    assert "canonicalize under /dev" in lowered
    assert "write_usb.sh only during an approved removable-media operation" in lowered


def test_direct_build_completion_guidance_does_not_imply_sudo_is_enough() -> None:
    build_script = (LIVE_USB_DIR / "build_iso.sh").read_text(encoding="utf-8")
    lowered = build_script.lower()

    assert "operator-approved removable media only" in lowered
    assert "root/sudo alone is not sufficient" in lowered
    assert "canonical whole removable /dev disk" in lowered
    assert "write to usb:  sudo ./write_usb.sh" not in lowered
