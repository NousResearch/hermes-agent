from __future__ import annotations

import pytest

from hermes_cli.config import DEFAULT_CONFIG
from plugins.mobile_bug_agent.config import config_from_mapping, runtime_root


def test_config_defaults_are_safe():
    cfg = config_from_mapping({})

    assert cfg.enabled is False
    assert cfg.rollout_mode == "dry_run"
    assert cfg.dry_run is True
    assert cfg.loop.require_fix_approval is True
    assert cfg.repo.branch_prefix == "monica"
    assert cfg.runtime.home_subdir == "agents/monica"
    assert cfg.runtime.worker_session_prefix == "monica"
    assert cfg.runtime.skip_memory is True
    assert cfg.worker.backend == "codex_cli"
    assert cfg.worker.codex_command == "codex"
    assert cfg.worker.codex_sandbox == "workspace-write"
    assert cfg.worker.codex_approval_policy == "never"
    assert cfg.proof.enabled is False
    assert cfg.proof.required_for_done is False
    assert cfg.proof.platform_order == ("ios", "android")
    assert cfg.proof.artifact_dir == "proof"
    assert cfg.proof.commands == ()
    assert cfg.proof.dev_client_scheme == ""


def test_default_config_shape_matches_monica_parser():
    cfg = config_from_mapping(DEFAULT_CONFIG["mobile_bug_agent"])

    assert cfg.enabled is False
    assert cfg.rollout_mode == "dry_run"
    assert cfg.slack.download_attachments is True
    assert cfg.loop.create_linear is True
    assert cfg.verification.commands == ()
    assert cfg.runtime.home_subdir == "agents/monica"
    assert cfg.worker.backend == "codex_cli"
    assert cfg.proof.required_for_done is False


def test_config_loads_nested_values():
    cfg = config_from_mapping(
        {
            "enabled": True,
            "dry_run": False,
            "slack": {
                "allowed_channels": ["C123"],
                "bot_user_ids": ["U999"],
                "approver_user_ids": ["U_APPROVER"],
                "download_attachments": False,
            },
            "loop": {
                "max_thread_messages": 25,
                "max_attachment_bytes": 1234,
                "require_fix_approval": True,
            },
            "linear": {
                "team_id": "team",
                "project_id": "project",
                "label_ids": ["label"],
            },
            "repo": {
                "url": "git@github.com:org/mobile.git",
                "default_branch": "main",
            },
            "verification": {"commands": ["npm test", "npm run lint"]},
            "runtime": {
                "home_subdir": "agents/mobile-monica",
                "worker_session_prefix": "mobile-monica",
                "skip_memory": False,
            },
            "worker": {
                "backend": "internal_agent",
                "codex_command": "/opt/codex/bin/codex",
                "codex_model": "gpt-5-codex",
                "codex_profile": "monica",
                "codex_sandbox": "danger-full-access",
                "codex_approval_policy": "on-request",
                "timeout_minutes": 12,
            },
            "proof": {
                "enabled": True,
                "required_for_done": True,
                "platform_order": ["ios"],
                "artifact_dir": "monica-proof",
                "commands": ["npm run proof:ios"],
                "dev_client_scheme": "elixir-card",
                "ios_simulator_udid": "SIM-UDID",
                "ios_bundle_id": "com.elixir.card",
                "android_serial": "emulator-5554",
                "android_avd": "MonicaPixel",
                "android_package": "com.joinelixir.elixirclub",
                "timeout_minutes": 8,
            },
        }
    )

    assert cfg.enabled is True
    assert cfg.rollout_mode == "linear_only"
    assert cfg.dry_run is False
    assert cfg.slack.allowed_channels == ("C123",)
    assert cfg.slack.approver_user_ids == ("U_APPROVER",)
    assert cfg.slack.download_attachments is False
    assert cfg.loop.max_thread_messages == 25
    assert cfg.loop.max_attachment_bytes == 1234
    assert cfg.linear.label_ids == ("label",)
    assert cfg.repo.url == "git@github.com:org/mobile.git"
    assert cfg.verification.commands == ("npm test", "npm run lint")
    assert cfg.runtime.home_subdir == "agents/mobile-monica"
    assert cfg.runtime.worker_session_prefix == "mobile-monica"
    assert cfg.runtime.skip_memory is False
    assert cfg.worker.backend == "internal_agent"
    assert cfg.worker.codex_command == "/opt/codex/bin/codex"
    assert cfg.worker.codex_model == "gpt-5-codex"
    assert cfg.worker.codex_profile == "monica"
    assert cfg.worker.codex_sandbox == "danger-full-access"
    assert cfg.worker.codex_approval_policy == "on-request"
    assert cfg.worker.timeout_minutes == 12
    assert cfg.proof.enabled is True
    assert cfg.proof.required_for_done is True
    assert cfg.proof.platform_order == ("ios",)
    assert cfg.proof.artifact_dir == "monica-proof"
    assert cfg.proof.commands == ("npm run proof:ios",)
    assert cfg.proof.dev_client_scheme == "elixir-card"
    assert cfg.proof.ios_simulator_udid == "SIM-UDID"
    assert cfg.proof.ios_bundle_id == "com.elixir.card"
    assert cfg.proof.android_serial == "emulator-5554"
    assert cfg.proof.android_avd == "MonicaPixel"
    assert cfg.proof.android_package == "com.joinelixir.elixirclub"
    assert cfg.proof.timeout_minutes == 8


def test_runtime_root_is_profile_relative(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-home"))
    cfg = config_from_mapping({"runtime": {"home_subdir": "agents/monica"}})

    assert runtime_root(cfg) == tmp_path / "profile-home" / "agents" / "monica"


def test_runtime_root_rejects_relative_parent_traversal(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-home"))
    cfg = config_from_mapping({"runtime": {"home_subdir": "../outside-runtime"}})

    with pytest.raises(ValueError, match="must stay inside HERMES_HOME"):
        runtime_root(cfg)


def test_runtime_root_rejects_chandler_named_subdir(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-home"))
    cfg = config_from_mapping({"runtime": {"home_subdir": "agents/chandler"}})

    with pytest.raises(ValueError, match="must not point at a Chandler runtime path"):
        runtime_root(cfg)


def test_runtime_root_allows_absolute_override(tmp_path):
    cfg = config_from_mapping({"runtime": {"home_subdir": str(tmp_path / "monica")}})

    assert runtime_root(cfg) == tmp_path / "monica"
