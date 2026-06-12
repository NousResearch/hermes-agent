from __future__ import annotations

import json
from dataclasses import dataclass

from plugins.mobile_bug_agent.config import MonicaConfig, ProofConfig, RuntimeConfig
from plugins.mobile_bug_agent.proof import ProofRunner


@dataclass(frozen=True)
class FakeRun:
    id: str = "run-123"
    linear_identifier: str = "ENG-123"
    branch_name: str = "monica/ENG-123-proof"


def _mark_git_worktree(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / ".git").write_text("gitdir: /tmp/fake-worktree-git-dir", encoding="utf-8")
    return path


def _config(tmp_path, **proof_overrides):
    proof_values = {
        "enabled": True,
        "required_for_done": True,
        "commands": ("capture-proof",),
        "platform_order": ("ios",),
        "artifact_dir": "proof",
    }
    proof_values.update(proof_overrides)
    return MonicaConfig(
        proof=ProofConfig(**proof_values),
        runtime=RuntimeConfig(home_subdir=str(tmp_path / "monica-runtime")),
    )


def test_proof_runner_blocks_when_commands_are_empty(tmp_path):
    runner = ProofRunner(config=_config(tmp_path, commands=()))

    result = runner.run(run=FakeRun(), worktree=_mark_git_worktree(tmp_path / "worktree"))

    assert result.passed is False
    assert result.summary == "Proof blocked: proof.commands is empty."
    assert result.artifacts == ()


def test_proof_runner_passes_when_command_creates_artifact(tmp_path):
    calls = []

    def run(command, cwd, timeout, env):
        calls.append(
            (
                command,
                cwd,
                timeout,
                env["MONICA_PROOF_DIR"],
                env["MONICA_DEV_CLIENT_SCHEME"],
                env["MONICA_IOS_BUNDLE_ID"],
                env["MONICA_ANDROID_SERIAL"],
                env["MONICA_ANDROID_AVD"],
                env["MONICA_ANDROID_PACKAGE"],
            )
        )
        proof_path = tmp_path / "monica-runtime" / "proof" / "run-123" / "ios-screenshot.png"
        proof_path.write_text("image bytes", encoding="utf-8")
        return 0, "captured", ""

    runner = ProofRunner(
        config=_config(
            tmp_path,
            timeout_minutes=3,
            dev_client_scheme="elixir-card",
            ios_bundle_id="com.elixir.card",
            android_serial="emulator-5554",
            android_avd="MonicaPixel",
            android_package="com.joinelixir.elixirclub",
        ),
        run_command=run,
    )

    result = runner.run(run=FakeRun(), worktree=_mark_git_worktree(tmp_path / "worktree"))

    assert result.passed is True
    assert result.summary == "Proof captured."
    proof_dir = tmp_path / "monica-runtime" / "proof" / "run-123"
    manifest_path = proof_dir / "monica-proof-manifest.json"
    assert result.artifacts == (
        str(manifest_path),
        str(proof_dir / "ios-screenshot.png"),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "run-123"
    assert manifest["linear_identifier"] == "ENG-123"
    assert manifest["branch_name"] == "monica/ENG-123-proof"
    assert manifest["worktree"] == str(tmp_path / "worktree")
    assert manifest["platforms"] == ["ios"]
    assert manifest["proof_artifacts"] == [str(proof_dir / "ios-screenshot.png")]
    assert calls == [
        (
            "capture-proof",
            tmp_path / "worktree",
            180,
            str(tmp_path / "monica-runtime" / "proof" / "run-123"),
            "elixir-card",
            "com.elixir.card",
            "emulator-5554",
            "MonicaPixel",
            "com.joinelixir.elixirclub",
        )
    ]


def test_proof_runner_blocks_when_command_produces_no_artifacts(tmp_path):
    runner = ProofRunner(
        config=_config(tmp_path),
        run_command=lambda _cmd, _cwd, _timeout, _env: (0, "ok", ""),
    )

    result = runner.run(run=FakeRun(), worktree=_mark_git_worktree(tmp_path / "worktree"))

    assert result.passed is False
    assert result.summary == "Proof blocked: proof commands produced no artifacts."
    assert result.artifacts == ()


def test_proof_runner_blocks_when_required_platform_artifact_is_missing(tmp_path):
    def run(_command, _cwd, _timeout, _env):
        proof_path = tmp_path / "monica-runtime" / "proof" / "run-123" / "android-screenshot.png"
        proof_path.write_text("android image bytes", encoding="utf-8")
        return 0, "captured android", ""

    runner = ProofRunner(
        config=_config(tmp_path, platform_order=("ios", "android")),
        run_command=run,
    )

    result = runner.run(run=FakeRun(), worktree=_mark_git_worktree(tmp_path / "worktree"))

    assert result.passed is False
    assert result.summary == "Proof blocked: missing required platform artifacts: ios."
    assert result.artifacts == (
        str(tmp_path / "monica-runtime" / "proof" / "run-123" / "android-screenshot.png"),
    )


def test_proof_runner_passes_only_after_all_required_platform_artifacts_exist(tmp_path):
    def run(_command, _cwd, _timeout, _env):
        proof_dir = tmp_path / "monica-runtime" / "proof" / "run-123"
        (proof_dir / "android-screenshot.png").write_text("android image bytes", encoding="utf-8")
        (proof_dir / "ios-screenshot.png").write_text("ios image bytes", encoding="utf-8")
        return 0, "captured both", ""

    runner = ProofRunner(
        config=_config(tmp_path, platform_order=("ios", "android")),
        run_command=run,
    )

    result = runner.run(run=FakeRun(), worktree=_mark_git_worktree(tmp_path / "worktree"))

    proof_dir = tmp_path / "monica-runtime" / "proof" / "run-123"
    assert result.passed is True
    assert result.artifacts == (
        str(proof_dir / "monica-proof-manifest.json"),
        str(proof_dir / "android-screenshot.png"),
        str(proof_dir / "ios-screenshot.png"),
    )


def test_proof_runner_preserves_partial_artifacts_on_command_failure(tmp_path):
    def run(_command, _cwd, _timeout, env):
        proof_path = tmp_path / "monica-runtime" / "proof" / "run-123" / "before-failure.txt"
        proof_path.write_text(env["MONICA_LINEAR_IDENTIFIER"], encoding="utf-8")
        return 1, "", "simulator failed"

    runner = ProofRunner(config=_config(tmp_path), run_command=run)

    result = runner.run(run=FakeRun(), worktree=_mark_git_worktree(tmp_path / "worktree"))

    assert result.passed is False
    assert result.summary == "Proof blocked: capture-proof"
    assert result.artifacts == (str(tmp_path / "monica-runtime" / "proof" / "run-123" / "before-failure.txt"),)
    assert "simulator failed" in result.output
