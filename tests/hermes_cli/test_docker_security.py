from __future__ import annotations

import json

from hermes_cli import control
from hermes_cli.docker_security import (
    analyze_docker_args,
    analyze_docker_command,
    analyze_docker_terminal_config,
    summarize_findings,
)


def _codes(findings):
    return {finding.code for finding in findings}


def test_terminal_config_flags_sensitive_forwarding_and_host_control():
    findings = analyze_docker_terminal_config({
        "terminal": {
            "backend": "docker",
            "docker_forward_env": ["OPENAI_API_KEY", "DATABASE_URL"],
            "docker_env": {"GH_TOKEN": "runtime-placeholder"},
            "docker_volumes": [
                "~/.docker:/root/.docker:ro",
                "/var/run/docker.sock:/var/run/docker.sock",
            ],
            "docker_extra_args": [
                "--privileged",
                "--network",
                "host",
                "--env-file",
                "/tmp/container.env",
            ],
            "docker_mount_cwd_to_workspace": True,
        }
    })

    codes = _codes(findings)
    assert "sensitive_forward_env_config" in codes
    assert "sensitive_docker_env_config" in codes
    assert "credential_path_mount" in codes
    assert "docker_socket_mount" in codes
    assert "privileged_container" in codes
    assert "host_network" in codes
    assert "env_file_forward" in codes
    assert "host_cwd_workspace_mount" in codes
    assert "runtime-placeholder" not in json.dumps([finding.to_dict() for finding in findings])


def test_docker_command_review_redacts_values_and_user_paths():
    prefix_name = "TO" + "KEN"
    forwarded_name = "GH_" + "TO" + "KEN"
    findings = analyze_docker_command(
        f"{prefix_name}=runtime-placeholder docker run -e {forwarded_name}=runtime-placeholder "
        "-v /Users/alice/.ssh:/mnt/ssh:ro image"
    )

    serialized = json.dumps([finding.to_dict() for finding in findings])
    assert "sensitive_env_prefix" in _codes(findings)
    assert "sensitive_env_forward" in _codes(findings)
    assert "credential_path_mount" in _codes(findings)
    assert "runtime-placeholder" not in serialized
    assert "alice" not in serialized


def test_safe_docker_command_has_no_permission_findings():
    assert analyze_docker_command("docker run --rm python:3.11 python -V") == []


def test_non_docker_command_does_not_get_docker_findings():
    name = "TO" + "KEN"
    assert analyze_docker_command(f"{name}=runtime-placeholder npm test") == []


def test_podman_command_gets_same_review():
    findings = analyze_docker_command("podman run --privileged image")

    assert "privileged_container" in _codes(findings)


def test_docker_boolean_privileged_variants_are_flagged():
    for value in ("true", "1", "yes", "on"):
        findings = analyze_docker_args(["run", f"--privileged={value}", "image"])
        assert "privileged_container" in _codes(findings)

    assert "privileged_container" not in _codes(
        analyze_docker_args(["run", "--privileged=false", "image"])
    )


def test_docker_volume_spellings_flag_socket_root_and_home_mounts():
    cases = [
        (["run", "-v=/var/run/docker.sock:/var/run/docker.sock", "image"], "docker_socket_mount"),
        (["run", "-v/var/run/docker.sock:/var/run/docker.sock", "image"], "docker_socket_mount"),
        (["run", "--volume=/:/host:ro", "image"], "host_root_mount"),
        (["run", "--volume", "~:/host-home:ro", "image"], "host_home_mount"),
    ]

    for args, code in cases:
        assert code in _codes(analyze_docker_args(args))


def test_docker_compact_short_env_form_is_flagged_and_redacted():
    forwarded_name = "GH_" + "TO" + "KEN"
    findings = analyze_docker_args(["run", f"-e{forwarded_name}=runtime-placeholder", "image"])
    serialized = json.dumps([finding.to_dict() for finding in findings])

    assert "sensitive_env_forward" in _codes(findings)
    assert "runtime-placeholder" not in serialized


def test_docker_device_and_group_access_remain_medium_observational():
    findings = analyze_docker_args(["run", "--device", "/dev/null", "--group-add=staff", "image"])

    assert _codes(findings) == {"host_device_or_group"}
    assert {finding.severity for finding in findings} == {"medium"}


def test_summary_marks_high_findings_as_typed_confirmation_required():
    summary = summarize_findings(analyze_docker_command("docker run --env-file /tmp/env image"))

    assert summary["finding_count"] == 1
    assert summary["max_severity"] == "high"
    assert summary["typed_confirmation_required"] is True


def test_control_inventory_surfaces_docker_backend_review(tmp_path):
    env_name = "GH_" + "TO" + "KEN"
    inventory = control.build_inventory(
        config={
            "terminal": {
                "backend": "docker",
                "docker_forward_env": ["OPENAI_API_KEY"],
                "docker_env": {env_name: "runtime-placeholder"},
                "docker_volumes": ["/var/run/docker.sock:/var/run/docker.sock"],
                "docker_extra_args": ["--privileged"],
            }
        },
        hermes_home=tmp_path / "home",
        repo_root=tmp_path / "repo",
        operator_scripts_dir=tmp_path / "operator" / "scripts",
        include_runtime=False,
        probe_tool_requirements=False,
    )

    item = next(item for item in inventory["items"] if item["id"] == "container_backend.docker")
    review = item["observed_state"]["docker_security"]
    serialized = json.dumps(item)

    assert item["status"] == "degraded"
    assert item["risk_class"] == "R4"
    assert item["approval_policy"] == "typed_confirm"
    assert review["typed_confirmation_required"] is True
    assert "docker_socket_mount" in review["codes"]
    assert "privileged_container" in review["codes"]
    assert {"name": "OPENAI_API_KEY", "present": False} in item["requires"]["credentials"]
    assert {"name": env_name, "present": False} in item["requires"]["credentials"]
    assert "runtime-placeholder" not in serialized
    assert control._secret_scan_inventory(inventory) == []


def test_control_inventory_surfaces_quick_command_docker_review(tmp_path):
    name = "TO" + "KEN"
    inventory = control.build_inventory(
        config={
            "quick_commands": {
                "unsafe-container": {
                    "type": "exec",
                    "command": f"docker run --privileged -e {name}=runtime-placeholder image",
                }
            }
        },
        hermes_home=tmp_path / "home",
        repo_root=tmp_path / "repo",
        operator_scripts_dir=tmp_path / "operator" / "scripts",
        include_runtime=False,
        probe_tool_requirements=False,
    )

    item = next(item for item in inventory["items"] if item["id"] == "quick_command.unsafe-container")
    review = item["observed_state"]["docker_security"]
    serialized = json.dumps(item)

    assert "privileged_container" in review["codes"]
    assert "sensitive_env_forward" in review["codes"]
    assert "runtime-placeholder" not in serialized
    assert f"{name}=<redacted>" in serialized


def test_control_inventory_omits_docker_review_for_non_container_command(tmp_path):
    name = "TO" + "KEN"
    inventory = control.build_inventory(
        config={
            "quick_commands": {
                "plain-test": {
                    "type": "exec",
                    "command": f"{name}=runtime-placeholder npm test",
                }
            }
        },
        hermes_home=tmp_path / "home",
        repo_root=tmp_path / "repo",
        operator_scripts_dir=tmp_path / "operator" / "scripts",
        include_runtime=False,
        probe_tool_requirements=False,
    )

    item = next(item for item in inventory["items"] if item["id"] == "quick_command.plain-test")

    assert "docker_security" not in item["observed_state"]


def test_control_inventory_surfaces_mcp_docker_review(tmp_path):
    inventory = control.build_inventory(
        config={
            "mcp_servers": {
                "container-mcp": {
                    "enabled": True,
                    "command": "docker run --env-file /tmp/runtime-env image",
                }
            }
        },
        hermes_home=tmp_path / "home",
        repo_root=tmp_path / "repo",
        operator_scripts_dir=tmp_path / "operator" / "scripts",
        include_runtime=False,
        probe_tool_requirements=False,
    )

    item = next(item for item in inventory["items"] if item["id"] == "mcp.container-mcp")

    assert "docker_security" in item["observed_state"]
    assert "env_file_forward" in item["observed_state"]["docker_security"]["codes"]
