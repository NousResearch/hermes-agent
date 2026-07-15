from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[3]
MODULE_PATH = ROOT / "ops/muncho/runtime/mechanical_job_rail.py"
RUNTIME = MODULE_PATH.parent
SPEC = importlib.util.spec_from_file_location("mechanical_job_rail_test", MODULE_PATH)
assert SPEC and SPEC.loader
rail = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rail
SPEC.loader.exec_module(rail)

REVISION = "a" * 40


def _host_facts() -> dict[str, object]:
    unsigned = {
        "schema": rail.HOST_FACTS_SCHEMA,
        "collected_at": "2026-07-14T12:00:00Z",
        "github_cli": {
            "path": str(rail.GH_PATH),
            "regular": True,
            "nlink": 1,
            "uid": 0,
            "gid": 0,
            "mode": "0755",
            "group_or_other_writable": False,
            "sha256": "4" * 64,
        },
        "git": {
            "path": str(rail.GIT_PATH),
            "regular": True,
            "nlink": 1,
            "uid": 0,
            "gid": 0,
            "mode": "0755",
            "group_or_other_writable": False,
            "sha256": "5" * 64,
        },
        "github_credential": {
            "path": str(rail.CREDENTIAL_SOURCE),
            "regular": True,
            "nlink": 1,
            "uid": 0,
            "gid": 0,
            "mode": "0400",
            "content_recorded": False,
            "size_recorded": False,
            "digest_recorded": False,
        },
        "provider_or_model_credential_observed": False,
        "discord_credential_observed": False,
    }
    return {
        **unsigned,
        "host_facts_sha256": hashlib.sha256(rail._canonical(unsigned)).hexdigest(),
    }


def _package(**kwargs):
    facts = _host_facts()
    return rail.build_package(
        revision=REVISION,
        host_facts=facts,
        expected_host_facts_sha256=facts["host_facts_sha256"],
        **kwargs,
    )


def _load_sync_routine():
    name = "fork_upstream_auto_sync_pr_routine_mechanical_rail_test"
    spec = importlib.util.spec_from_file_location(
        name,
        RUNTIME / "fork_upstream_auto_sync_pr_routine.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    releases = tmp_path / "releases"
    monkeypatch.setattr(rail, "RELEASES_ROOT", releases)
    release = releases / f"hermes-agent-{REVISION[:12]}"
    for relative in (
        rail.RAIL_RELATIVE,
        rail.ROUTINE_RELATIVE,
        rail.HARDENING_RELATIVE,
    ):
        target = release / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source = ROOT / relative
        shutil.copyfile(source, target)
    marker = release / rail.SOURCE_MARKER_RELATIVE
    marker.write_text(REVISION + "\n", encoding="ascii")
    interpreter = release / ".venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"python-placeholder\n")
    interpreter.chmod(0o755)
    return release


def test_package_is_exact_hardened_delayed_and_has_one_allowlisted_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = _release(tmp_path, monkeypatch)
    package = _package()
    manifest = json.loads(package.manifest_bytes)
    service = package.service_bytes.decode()
    timer = package.timer_bytes.decode()

    assert package.release_root == release
    assert manifest["job_allowlist"] == [
        {
            "job_id": rail.JOB_ID,
            "argv": ["--execute"],
            "routine": str(release / rail.ROUTINE_RELATIVE),
            "routine_sha256": package.routine_sha256,
            "hardening": str(release / rail.HARDENING_RELATIVE),
            "hardening_sha256": package.hardening_sha256,
            "fork_repository": "lomliev/hermes-agent",
            "upstream_repository_read_only": "NousResearch/hermes-agent",
            "auto_merge_or_deploy_enabled": False,
        }
    ]
    assert manifest["provider_or_model_dependency"] is False
    assert manifest["discord_dependency"] is False
    assert manifest["credential_value_recorded"] is False
    assert manifest["timer_started_by_package"] is False
    assert rail.package_public_manifest(package) == manifest
    assert rail.validate_package_manifest(
        manifest,
        revision=REVISION,
        host_facts_sha256=manifest["host_facts_sha256"],
    ) == manifest
    assert package.service_sha256 == hashlib.sha256(package.service_bytes).hexdigest()
    assert package.timer_sha256 == hashlib.sha256(package.timer_bytes).hexdigest()

    for required in (
        "Type=oneshot",
        "DynamicUser=yes",
        "LoadCredential=github-token:/etc/muncho/fork-auto-sync/github-token",
        "NoNewPrivileges=yes",
        "CapabilityBoundingSet=",
        "ProtectSystem=strict",
        "ProtectHome=yes",
        "PrivateDevices=yes",
        "RestrictNamespaces=yes",
        "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6",
        "IPAddressDeny=169.254.169.254/32",
        "StandardOutput=null",
    ):
        assert required in service
    assert "Restart=" not in service
    assert "OnFailure=" not in service
    assert "OPENAI" not in service
    assert "DISCORD" not in service
    assert "HERMES_HOME=" not in service
    assert "EnvironmentFile=" not in service
    assert "PassEnvironment=" not in service
    assert str(release) in service
    assert package.rail_sha256 in service
    assert package.routine_sha256 in service
    assert package.hardening_sha256 in service

    assert f"Unit={rail.SERVICE_UNIT}" in timer
    assert "OnActiveSec=30m" in timer
    assert "OnUnitActiveSec=3h" in timer
    assert "Persistent=false" in timer
    assert "OnBootSec=" not in timer

    tampered = json.loads(json.dumps(manifest))
    tampered["timer_started_by_package"] = True
    with pytest.raises(
        rail.MechanicalJobRailError,
        match="package_manifest_invalid",
    ):
        rail.validate_package_manifest(
            tampered,
            revision=REVISION,
            host_facts_sha256=manifest["host_facts_sha256"],
        )


def test_host_facts_validator_is_pure_exact_and_credential_redaction_safe() -> None:
    facts = _host_facts()
    assert rail.validate_host_facts(
        facts,
        expected_sha256=facts["host_facts_sha256"],
    ) == facts
    assert rail.validate_host_facts(facts) == facts
    encoded = json.dumps(facts)
    assert "content_recorded\": false" in encoded
    assert "size_recorded\": false" in encoded
    assert "digest_recorded\": false" in encoded
    assert "GH_TOKEN" not in encoded

    tampered = json.loads(json.dumps(facts))
    tampered["github_credential"]["mode"] = "0440"
    with pytest.raises(rail.MechanicalJobRailError, match="host_facts_invalid"):
        rail.validate_host_facts(
            tampered,
            expected_sha256=facts["host_facts_sha256"],
        )


def test_package_digest_changes_when_reviewed_routine_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = _release(tmp_path, monkeypatch)
    first = _package()
    with (release / rail.ROUTINE_RELATIVE).open("ab") as stream:
        stream.write(b"\n# reviewed change\n")
    second = _package()
    assert first.routine_sha256 != second.routine_sha256
    assert first.service_sha256 != second.service_sha256
    assert first.manifest_sha256 != second.manifest_sha256


def test_package_staging_is_byte_exact_and_does_not_install_or_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _release(tmp_path, monkeypatch)
    package = _package()
    staging = tmp_path / "staged"
    receipt = rail.write_package(package, output_root=staging)

    assert (staging / rail.SERVICE_UNIT).read_bytes() == package.service_bytes
    assert (staging / rail.TIMER_UNIT).read_bytes() == package.timer_bytes
    assert (staging / "manifest.json").read_bytes() == package.manifest_bytes
    assert all(path.stat().st_mode & 0o777 == 0o444 for path in staging.iterdir())
    assert receipt["installed"] is False
    assert receipt["timer_enabled"] is False
    assert receipt["timer_started"] is False
    assert receipt["job_executed"] is False
    assert rail.verify_package(package, output_root=staging) == receipt

    (staging / rail.TIMER_UNIT).chmod(0o644)
    with pytest.raises(
        rail.MechanicalJobRailError,
        match="artifact_drifted",
    ):
        rail.verify_package(package, output_root=staging)


def test_run_rejects_every_non_allowlisted_job() -> None:
    args = argparse.Namespace(
        job_id="legacy-script-from-jobs-json",
        revision=REVISION,
        rail_sha256="1" * 64,
        routine_sha256="2" * 64,
        hardening_sha256="3" * 64,
        gh_sha256="4" * 64,
        git_sha256="5" * 64,
    )
    with pytest.raises(rail.MechanicalJobRailError, match="not_allowlisted"):
        rail.run_job(args)


def test_run_has_no_provider_discord_or_output_delivery_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "state"
    runtime = tmp_path / "run"
    credential_dir = tmp_path / "credentials"
    credential_dir.mkdir()
    token = "github_pat_" + "x" * 32
    (credential_dir / rail.CREDENTIAL_NAME).write_text(token, encoding="ascii")
    gh = tmp_path / "gh"
    git = tmp_path / "git"
    gh.write_text("binary", encoding="ascii")
    git.write_text("binary", encoding="ascii")
    routine = tmp_path / "routine.py"
    routine.write_text("# routine", encoding="ascii")

    monkeypatch.setattr(rail, "STATE_ROOT", state)
    monkeypatch.setattr(rail, "RUNTIME_ROOT", runtime)
    monkeypatch.setattr(rail, "GH_PATH", gh)
    monkeypatch.setattr(rail, "GIT_PATH", git)
    monkeypatch.setattr(
        rail,
        "_attest_release",
        lambda **_kwargs: (tmp_path / "release", routine, tmp_path / "hardening.py"),
    )
    monkeypatch.setattr(rail, "_attest_host_binaries", lambda **_kwargs: None)
    monkeypatch.setenv("CREDENTIALS_DIRECTORY", str(credential_dir))
    monkeypatch.setenv("INVOCATION_ID", "1" * 32)
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["environment"] = dict(kwargs["env"])
        kwargs["stdout"].write(b"safe summary\n")
        kwargs["stderr"].write(b"provider-shaped text that stays private\n")
        report = state / "routine-state/auto-sync-pr-latest.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text('{"status":"blocked_merge_conflicts"}\n')
        return subprocess.CompletedProcess(command, 2)

    monkeypatch.setattr(rail.subprocess, "run", fake_run)
    args = argparse.Namespace(
        job_id=rail.JOB_ID,
        revision=REVISION,
        rail_sha256="1" * 64,
        routine_sha256="2" * 64,
        hardening_sha256="3" * 64,
        gh_sha256="4" * 64,
        git_sha256="5" * 64,
    )
    assert rail.run_job(args) == 0
    receipt = json.loads((state / "latest.json").read_text())
    encoded = json.dumps(receipt)
    environment = observed["environment"]

    assert receipt["outcome"] == "blocked_receipt_recorded"
    assert receipt["provider_or_model_invoked"] is False
    assert receipt["discord_delivery_attempted"] is False
    assert receipt["secret_material_recorded"] is False
    assert receipt["stdout"]["content_recorded"] is False
    assert receipt["stderr"]["content_recorded"] is False
    assert token not in encoded
    assert "safe summary" not in encoded
    assert "provider-shaped" not in encoded
    assert environment["GH_TOKEN"] == token
    assert environment["FORK_UPSTREAM_AUTO_SYNC_EXECUTE_APPROVED"] == "1"
    assert "FORK_UPSTREAM_AUTO_SYNC_AUTO_MERGE_DEPLOY_APPROVED" not in environment
    assert not any(
        name.startswith(("OPENAI", "DISCORD", "HERMES_")) for name in environment
    )
    assert observed["command"][-1] == "--execute"


def test_routine_push_target_is_exact_fork_and_never_upstream() -> None:
    source = (ROOT / rail.ROUTINE_RELATIVE).read_text(encoding="utf-8")
    assert 'FORK_GIT_URL = "https://github.com/lomliev/hermes-agent.git"' in source
    assert 'UPSTREAM_GIT_URL = "https://github.com/NousResearch/hermes-agent.git"' in source
    push = source[source.index('"push",\n            FORK_GIT_URL') - 200 :]
    assert '"push",\n            FORK_GIT_URL' in push
    assert '"push",\n            UPSTREAM_GIT_URL' not in source
    assert '"pr",\n                    "create",\n                    "--repo",\n                    FORK_REPO' in source


def test_existing_pr_without_separate_merge_gate_is_inert_and_receipted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routine = _load_sync_routine()
    state = tmp_path / "state"
    reports = tmp_path / "reports"
    reports.mkdir()
    monkeypatch.setattr(routine, "STATE_DIR", state)
    monkeypatch.setattr(routine, "REPORT_DIR", reports)
    monkeypatch.setattr(
        routine, "BLOCKER_DEDUPE_STATE", state / "blocker-dedupe.json"
    )
    monkeypatch.setenv(routine.EXECUTE_ENV, "1")
    monkeypatch.delenv(routine.AUTO_MERGE_DEPLOY_ENV, raising=False)
    monkeypatch.setattr(
        routine,
        "build_plan",
        lambda _args: {
            "created_at_utc": "2026-07-14T12:00:00Z",
            "fresh_refs": {"behind_by": 31},
            "open_sync_prs": [
                {
                    "number": 101,
                    "url": "https://github.com/lomliev/hermes-agent/pull/101",
                }
            ],
        },
    )
    monkeypatch.setattr(
        routine,
        "cleanup_stale_sync_prs",
        lambda _prs, _fresh: {"closed": [], "kept": []},
    )
    monkeypatch.setattr(routine, "cleanup_old_auto_sync_worktrees", lambda: [])
    monkeypatch.setattr(
        routine,
        "auto_merge_sync_pr_and_start_deploy",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("merge/deploy gate must remain inert")
        ),
    )

    assert routine.execute(argparse.Namespace(execute=True)) == 0
    receipt = json.loads((state / "auto-sync-pr-latest.json").read_text())
    assert receipt["status"] == "open_sync_pr_exists_review_required_no_action"
    assert receipt["pr_number"] == 101
    assert not (state / "deploy_queue").exists()


def test_routine_redacts_exact_and_token_shaped_github_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routine = _load_sync_routine()
    secret = "github_pat_" + "a" * 40
    monkeypatch.setenv("GH_TOKEN", secret)
    value = f"first={secret} second=ghp_{'b' * 40}"
    redacted = routine.redact_command_output(value)
    assert secret not in redacted
    assert "ghp_" not in redacted
    assert redacted.count("[REDACTED]") == 2


def test_module_compiles_under_isolated_stdlib() -> None:
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(MODULE_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr

    routine = ROOT / rail.ROUTINE_RELATIVE
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(routine), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
