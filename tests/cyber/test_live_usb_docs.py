"""Docs invariants for AgentCyber Live USB safety gates."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
LIVE_USB_DIR = ROOT / "live-usb"
FIRSTBOOT = LIVE_USB_DIR / "rootfs-overlay" / "usr" / "local" / "bin" / "hermes-firstboot"
GRUB_CFG = LIVE_USB_DIR / "grub" / "grub.cfg"
GATEWAY_SERVICE = LIVE_USB_DIR / "rootfs-overlay" / "etc" / "systemd" / "system" / "hermes-gateway.service"


def _run_script(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    base_env = os.environ.copy()
    base_env.pop("HERMES_AGENTCYBER_LIVE_USB_APPROVAL", None)
    if env:
        base_env.update(env)
    return subprocess.run(
        list(args),
        cwd=ROOT,
        env=base_env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )


def _combined_output(result: subprocess.CompletedProcess[str]) -> str:
    return result.stdout + result.stderr


def _list_usb_stub_env(tmp_path: Path) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    lsblk = bin_dir / "lsblk"
    lsblk.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' 'NAME SIZE TRAN MODEL RM' 'sdb 15G usb FixtureUSB 1'\n",
        encoding="utf-8",
    )
    column = bin_dir / "column"
    column.write_text("#!/usr/bin/env bash\ncat\n", encoding="utf-8")
    lsblk.chmod(0o755)
    column.chmod(0o755)
    return {"PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"}


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
        "whole removable `/dev` disk metadata",
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

    root_guidance = "Root/sudo alone is not sufficient"
    assert root_guidance in write_script
    assert root_guidance in provision_script

    assert "WARNING: /sys/block" not in write_script
    assert 'PROVISION_PART="$(_partition_path "$DEVICE" 3)' in provision_script


def test_direct_live_usb_mutation_scripts_require_exact_operator_approval() -> None:
    build_script = (LIVE_USB_DIR / "build_iso.sh").read_text(encoding="utf-8")
    write_script = (LIVE_USB_DIR / "write_usb.sh").read_text(encoding="utf-8")
    provision_script = (LIVE_USB_DIR / "provision.sh").read_text(encoding="utf-8")

    for script in (build_script, write_script, provision_script):
        assert "HERMES_AGENTCYBER_LIVE_USB_APPROVAL" in script
        assert "--operator-approval" in script
        assert "OPERATOR_APPROVAL_PROVIDED" in script
        assert 'if [[ "$OPERATOR_APPROVAL" != "$HERMES_AGENTCYBER_LIVE_USB_APPROVAL" ]]' in script
        assert "No trimming, case normalization, or aliases are accepted." in script
        assert "Root/sudo alone is not sufficient" in script
        assert "unset HERMES_AGENTCYBER_LIVE_USB_APPROVAL" in script
        assert 'OPERATOR_APPROVAL=""' in script

    assert build_script.index('require_operator_approval "build" || exit 1') < build_script.index(
        "if [[ $EUID -ne 0 ]]"
    )

    assert write_script.index('if [[ "$LIST_ONLY" == "true" ]]') < write_script.index(
        'require_operator_approval "write" || exit 1'
    )
    assert write_script.index('require_operator_approval "write" || exit 1') < write_script.index(
        "if [[ $EUID -ne 0 ]]"
    )
    assert write_script.index('require_operator_approval "write" || exit 1') < write_script.index(
        'DEVICE="$(_canonical_removable_device "$DEVICE")"'
    )

    assert provision_script.index('require_operator_approval "provision" || exit 1') < provision_script.index(
        "if [[ $EUID -ne 0 ]]"
    )
    assert provision_script.index('require_operator_approval "provision" || exit 1') < provision_script.index(
        'DEVICE="$(_canonical_removable_device "$DEVICE")"'
    )


def test_direct_live_usb_approval_gate_behavior_is_fail_closed(tmp_path: Path) -> None:
    missing_env = _run_script("live-usb/build_iso.sh", "--operator-approval", "provided-token")
    assert missing_env.returncode == 1
    missing_env_output = _combined_output(missing_env)
    assert "requires exact operator approval" in missing_env_output
    assert "provided-token" not in missing_env_output
    assert "must run as root" not in missing_env_output

    write_missing_env = _run_script(
        "live-usb/write_usb.sh",
        "--device",
        "/dev/notreal",
        "--operator-approval",
        "provided-token",
    )
    assert write_missing_env.returncode == 1
    write_missing_env_output = _combined_output(write_missing_env)
    assert "requires exact operator approval" in write_missing_env_output
    assert "Not a block device" not in write_missing_env_output
    assert "provided-token" not in write_missing_env_output

    exact_env = {"HERMES_AGENTCYBER_LIVE_USB_APPROVAL": "ExactToken"}
    wrong_case = _run_script(
        "live-usb/build_iso.sh",
        "--operator-approval",
        "exacttoken",
        env=exact_env,
    )
    assert wrong_case.returncode == 1
    wrong_case_output = _combined_output(wrong_case)
    assert "did not match exactly" in wrong_case_output
    assert "ExactToken" not in wrong_case_output
    assert "exacttoken" not in wrong_case_output

    missing_value = _run_script("live-usb/provision.sh", "--usb", "/dev/notreal", "--operator-approval")
    assert missing_value.returncode == 1
    assert "--operator-approval requires a value" in _combined_output(missing_value)

    list_only = _run_script("live-usb/write_usb.sh", "--list", env=_list_usb_stub_env(tmp_path))
    assert list_only.returncode == 0
    list_only_output = _combined_output(list_only)
    assert "Removable / USB block devices" in list_only_output
    assert "sdb" in list_only_output


def test_direct_live_usb_exact_approval_reaches_root_gate_before_device_probe() -> None:
    exact_env = {"HERMES_AGENTCYBER_LIVE_USB_APPROVAL": "ExactToken"}
    approved_write = _run_script(
        "live-usb/write_usb.sh",
        "--device",
        "/dev/notreal",
        "--operator-approval",
        "ExactToken",
        env=exact_env,
    )
    assert approved_write.returncode == 1
    approved_write_output = _combined_output(approved_write)
    assert "Writing to a block device requires root" in approved_write_output
    assert "Not a block device" not in approved_write_output
    assert "ExactToken" not in approved_write_output

    approved_provision = _run_script(
        "live-usb/provision.sh",
        "--usb",
        "/dev/notreal",
        "--operator-approval",
        "ExactToken",
        env=exact_env,
    )
    assert approved_provision.returncode == 1
    approved_provision_output = _combined_output(approved_provision)
    assert "Run as root" in approved_provision_output
    assert "Not a block device" not in approved_provision_output
    assert "ExactToken" not in approved_provision_output


def test_direct_provision_script_repacks_config_dirs_as_dot_hermes() -> None:
    section = _live_usb_section()
    provision_script = (LIVE_USB_DIR / "provision.sh").read_text(encoding="utf-8")

    assert 'config=".agentcyber-home"' in section
    assert "repack" in section.lower()
    assert "prebuilt tarballs must already contain a `.hermes/` top-level directory" in section
    assert 'tar cf - -C "$CONFIG_DIR" .' in provision_script
    assert 'tar xf - -C "${TMP_CFG}/.hermes"' in provision_script
    assert 'tar czf "${MNT}/hermes-config.tar.gz" \\' in provision_script
    assert '-C "${TMP_CFG}" ".hermes"' in provision_script
    assert 'TMP_CFG=""' in provision_script
    assert 'if [[ -n "${TMP_CFG:-}" ]]' in provision_script
    assert 'TMP_CFG=""\n  echo "✓  Config dir packed from ${CONFIG_DIR} as .hermes"' in provision_script
    assert '-C "$(dirname "$CONFIG_DIR")" "$(basename "$CONFIG_DIR")"' not in provision_script


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
    assert "hermes_agentcyber_live_usb_approval" in lowered
    assert "--operator-approval" in lowered
    assert "canonical whole removable /dev disk" in lowered
    assert "write to usb:  sudo ./write_usb.sh" not in lowered


def test_firstboot_forensic_mode_skips_host_mounts_and_gateway_startup() -> None:
    firstboot = FIRSTBOOT.read_text(encoding="utf-8")
    grub_cfg = GRUB_CFG.read_text(encoding="utf-8")
    gateway_service = GATEWAY_SERVICE.read_text(encoding="utf-8")
    readme_section = _live_usb_section().lower()

    assert "Hermes AgentCyber Live (Forensic" in grub_cfg
    assert "noautomount noswap nopersistent" in grub_cfg
    assert "HERMES_LIVE_MODE=forensic" in grub_cfg
    assert "ConditionKernelCommandLine=!HERMES_LIVE_MODE=forensic" in gateway_service

    forensic_guard = firstboot.index("if _is_forensic_mode; then")
    pypi_install = firstboot.index("if [[ -f /opt/hermes-install-on-firstboot ]]", forensic_guard)
    config_autoload = firstboot.index("CONFIG_PART=\"$(_hermescfg_partition || true)\"", pypi_install)
    wizard = firstboot.index("Interactive first-boot wizard", config_autoload)
    gateway_start = firstboot.index("systemctl start hermes-gateway.service", config_autoload)

    assert forensic_guard < pypi_install < config_autoload < wizard
    assert forensic_guard < gateway_start
    assert "skipping config auto-load, wizard, and gateway startup" in firstboot
    assert "agentcyber first boot skips config auto-load, setup wizard, and gateway startup" in readme_section
    assert "does not scan or mount host/provision block devices" in readme_section


def test_firstboot_loads_provisioned_config_only_from_hermescfg_label() -> None:
    firstboot = FIRSTBOOT.read_text(encoding="utf-8")

    assert "/dev/disk/by-label/HERMESCFG" in firstboot
    assert "blkid -L HERMESCFG" in firstboot
    assert "CONFIG_PART=\"$(_hermescfg_partition || true)\"" in firstboot
    assert "mount -o ro \"$CONFIG_PART\" \"$TMP_MNT\"" in firstboot
    assert "Found provisioned config on HERMESCFG" in firstboot
    assert "Do not glob common" in firstboot

    obsolete_host_disk_scans = (
        "for part in /dev/sd?3 /dev/nvme?n?p3",
        "mount -o ro \"$part\"",
        "[[ -b \"$part\" ]]",
    )
    for obsolete_scan in obsolete_host_disk_scans:
        assert obsolete_scan not in firstboot
