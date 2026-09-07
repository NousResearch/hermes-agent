from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from hermes_cli import secure_worker_cli
from hermes_cli.secure_worker import BoundaryReport


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    secure_worker_cli.register_cli(parser)
    return parser


def test_cli_registers_required_commands() -> None:
    parser = _parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    assert set(subparsers.choices) == {
        "audit",
        "run",
        "pack",
        "verify",
        "profile-render",
        "attest",
        "destroy",
    }


def test_remote_audit_parser_accepts_manifest_and_policy_proofs(tmp_path: Path) -> None:
    parser = _parser()
    args = parser.parse_args(
        [
            "audit",
            "--config",
            str(tmp_path / "profile.yaml"),
            "--pack",
            str(tmp_path / "pack"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--policy",
            str(tmp_path / "policy.json"),
            "--image-lock",
            str(tmp_path / "image-lock.json"),
            "--receipt",
            str(tmp_path / "admission.json"),
        ]
    )
    assert args.manifest.endswith("manifest.json")
    assert args.policy.endswith("policy.json")
    assert args.receipt.endswith("admission.json")


def test_run_parser_requires_admission_receipt_and_fixed_hermes_args(tmp_path: Path) -> None:
    args = _parser().parse_args(
        [
            "run",
            "--config", str(tmp_path / "profile.yaml"),
            "--pack", str(tmp_path / "pack"),
            "--manifest", str(tmp_path / "manifest.json"),
            "--policy", str(tmp_path / "policy.json"),
            "--attestation", str(tmp_path / "attestation.json"),
            "--image-lock", str(tmp_path / "image-lock.json"),
            "--receipt", str(tmp_path / "admission.json"),
            "--", "review", "the proposal",
        ]
    )
    assert args.hermes_args == ["--", "review", "the proposal"]


def test_run_refuses_mutated_runtime_before_constructing_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    original = BoundaryReport(
        allowed=True,
        reasons=(),
        manifest_sha256="a" * 64,
        source_commit="b" * 40,
        config_sha256="c" * 64,
        policy_sha256="d" * 64,
        worker_image="worker@sha256:" + "e" * 64,
        broker_executable_sha256="f" * 64,
        broker_module_sha256="1" * 64,
    )
    receipt = tmp_path / "admission.json"
    from hermes_cli.secure_worker import admission_receipt_from_report

    receipt.write_text(
        json.dumps(asdict(admission_receipt_from_report(original))), encoding="utf-8"
    )
    mutated = replace(original, config_sha256="2" * 64)
    monkeypatch.setattr(secure_worker_cli, "_audit_from_args", lambda args: ({}, mutated))
    monkeypatch.setattr(
        secure_worker_cli.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("process must not be constructed"),
    )
    args = argparse.Namespace(
        hermes_args=["--", "review the proposal"],
        pack=str(pack),
        receipt=str(receipt),
        config=str(tmp_path / "profile.yaml"),
    )
    assert secure_worker_cli.cmd_run(args) == 1


def test_attest_requires_explicit_privacy_confirmation(tmp_path: Path) -> None:
    parser = _parser()
    output = tmp_path / "attestation.json"
    args = parser.parse_args(["attest", "--output", str(output)])
    assert secure_worker_cli.cmd_attest(args) == 1
    assert not output.exists()


def test_attest_writes_bounded_machine_readable_attestation(tmp_path: Path) -> None:
    parser = _parser()
    output = tmp_path / "attestation.json"
    args = parser.parse_args(
        [
            "attest",
            "--output",
            str(output),
            "--confirm-privacy-mode",
            "--ttl-minutes",
            "30",
        ]
    )
    assert secure_worker_cli.cmd_attest(args) == 0
    row = json.loads(output.read_text())
    assert row["privacy_mode"] is True
    assert row["provider"] == "nous"


def test_attest_rejects_excessive_ttl(tmp_path: Path) -> None:
    parser = _parser()
    output = tmp_path / "attestation.json"
    args = parser.parse_args(
        [
            "attest",
            "--output",
            str(output),
            "--confirm-privacy-mode",
            "--ttl-minutes",
            "121",
        ]
    )
    assert secure_worker_cli.cmd_attest(args) == 1
    assert not output.exists()


def test_profile_render_refuses_existing_destination(tmp_path: Path) -> None:
    parser = _parser()
    output = tmp_path / "profile.yaml"
    output.write_text("owned by user\n")
    args = parser.parse_args(
        [
            "profile-render",
            "--kind",
            "local-safe",
            "--cwd",
            str(tmp_path),
            "--output",
            str(output),
            "--worker-image",
            "worker@sha256:" + "a" * 64,
        ]
    )
    assert secure_worker_cli.cmd_profile_render(args) == 1
    assert output.read_text() == "owned by user\n"
