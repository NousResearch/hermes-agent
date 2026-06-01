"""Tests for the safe Bitwarden migration helpers and CLI wrappers."""

from __future__ import annotations

import stat
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_cli import bitwarden_migration as bwm  # noqa: E402
from hermes_cli import secrets_cli  # noqa: E402


@pytest.fixture
def make_profile(tmp_path: Path):
    def _make_profile(name: str, env_text: str, config_text: str) -> Path:
        profile_home = tmp_path / name
        profile_home.mkdir(parents=True, exist_ok=True)
        (profile_home / ".env").write_text(env_text, encoding="utf-8")
        (profile_home / "config.yaml").write_text(config_text, encoding="utf-8")
        return profile_home

    return _make_profile


class TestInventory:
    def test_all_profiles_redacts_values_and_marks_bootstrap_secret(
        self,
        make_profile,
        monkeypatch: pytest.MonkeyPatch,
        capsys,
    ) -> None:
        coding_home = make_profile(
            "coding",
            """BWS_ACCESS_TOKEN=0.coding-token
OPENAI_API_KEY=sk-coding-secret
PATH=/usr/local/bin
""",
            """secrets:
  bitwarden:
    enabled: true
    access_token_env: BWS_ACCESS_TOKEN
    project_id: coding-project
""",
        )
        ops_home = make_profile(
            "ops",
            """BWS_ACCESS_TOKEN=0.ops-token
DISCORD_CHANNEL_ID=1234567890
GITHUB_TOKEN=ghp_ops_secret
""",
            """secrets:
  bitwarden:
    enabled: true
    access_token_env: BWS_ACCESS_TOKEN
    project_id: ops-project
""",
        )

        monkeypatch.setattr(
            bwm,
            "list_profiles",
            lambda: [
                SimpleNamespace(name="coding", path=coding_home),
                SimpleNamespace(name="ops", path=ops_home),
            ],
        )

        rc = secrets_cli.cmd_inventory(Namespace(profile=None, all_profiles=True))
        out = capsys.readouterr().out

        assert rc == 0
        assert "PROFILE | SURFACE | CLASS | STATE | KEY" in out
        assert "coding | .env | bootstrap-secret | set | BWS_ACCESS_TOKEN" in out
        assert "ops | .env | bootstrap-secret | set | BWS_ACCESS_TOKEN" in out
        assert "coding | .env | secret | set | OPENAI_API_KEY" in out
        assert "coding | .env | non-secret | set | PATH" in out
        assert "ops | .env | non-secret | set | DISCORD_CHANNEL_ID" in out
        assert "sk-coding-secret" not in out
        assert "0.coding-token" not in out
        assert "ghp_ops_secret" not in out
        assert "0.ops-token" not in out


class TestPrunePlanning:
    def test_disabled_bitwarden_config_stays_read_only(
        self,
        make_profile,
    ) -> None:
        profile_home = make_profile(
            "coding",
            """BWS_ACCESS_TOKEN=0.coding-token
OPENAI_API_KEY=sk-coding-secret
PATH=/usr/local/bin
""",
            """secrets:
  bitwarden:
    enabled: false
    access_token_env: BWS_ACCESS_TOKEN
    project_id: coding-project
""",
        )

        called = {"count": 0}

        def fake_fetch(**_kwargs):
            called["count"] += 1
            return ({"OPENAI_API_KEY": "sk-coding-secret"}, [])

        plan = bwm.prune_plan_for_profile(
            "coding",
            profile_home,
            fetcher=fake_fetch,
        )

        assert called["count"] == 0
        assert plan.verification_error == "Bitwarden is disabled in config.yaml"
        assert plan.removable_keys() == []
        actions = {row.key: row.action for row in plan.rows}
        assert actions["BWS_ACCESS_TOKEN"] == "keep"
        assert actions["OPENAI_API_KEY"] == "keep"
        assert actions["PATH"] == "keep"


class TestPruneApply:
    @pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX mode bits not enforced on Windows")
    def test_apply_prune_plan_creates_0600_backup_and_leaves_config_untouched(
        self,
        make_profile,
    ) -> None:
        profile_home = make_profile(
            "coding",
            """BWS_ACCESS_TOKEN=0.coding-token
OPENAI_API_KEY=sk-coding-secret
PATH=/usr/local/bin
DISCORD_CHANNEL_ID=1234567890
GITHUB_TOKEN=ghp_ops_secret
UNVERIFIED_SECRET=keep-me
""",
            """secrets:
  bitwarden:
    enabled: true
    access_token_env: BWS_ACCESS_TOKEN
    project_id: coding-project
    server_url: https://vault.bitwarden.eu
""",
        )
        env_path = profile_home / ".env"
        config_path = profile_home / "config.yaml"
        original_config = config_path.read_text(encoding="utf-8")

        def fake_fetch(**kwargs):
            assert kwargs["access_token"] == "0.coding-token"
            assert kwargs["project_id"] == "coding-project"
            assert kwargs["server_url"] == "https://vault.bitwarden.eu"
            assert kwargs["use_cache"] is False
            assert kwargs["home_path"] == profile_home
            return (
                {
                    "OPENAI_API_KEY": "sk-coding-secret",
                    "GITHUB_TOKEN": "ghp_ops_secret",
                },
                ["verified project"],
            )

        plan = bwm.prune_plan_for_profile(
            "coding",
            profile_home,
            fetcher=fake_fetch,
        )
        assert plan.verification_error is None
        assert plan.warnings == ["verified project"]
        assert [row.action for row in plan.rows if row.key == "OPENAI_API_KEY"] == ["remove"]
        assert [row.action for row in plan.rows if row.key == "GITHUB_TOKEN"] == ["remove"]
        assert [row.action for row in plan.rows if row.key == "UNVERIFIED_SECRET"] == ["keep"]

        result = bwm.apply_prune_plan(plan)

        assert result.changed is True
        assert result.removed_keys == ["OPENAI_API_KEY", "GITHUB_TOKEN"]
        assert result.backup_path is not None
        assert result.backup_path.exists()
        assert result.backup_path.name.startswith(".env.bak-pre-bitwarden-prune-")
        assert stat.S_IMODE(result.backup_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(env_path.stat().st_mode) == 0o600
        assert result.backup_path.read_text(encoding="utf-8") == (
            "BWS_ACCESS_TOKEN=0.coding-token\n"
            "OPENAI_API_KEY=sk-coding-secret\n"
            "PATH=/usr/local/bin\n"
            "DISCORD_CHANNEL_ID=1234567890\n"
            "GITHUB_TOKEN=ghp_ops_secret\n"
            "UNVERIFIED_SECRET=keep-me\n"
        )
        rewritten = env_path.read_text(encoding="utf-8")
        assert "OPENAI_API_KEY=" not in rewritten
        assert "GITHUB_TOKEN=" not in rewritten
        assert "BWS_ACCESS_TOKEN=0.coding-token" in rewritten
        assert "PATH=/usr/local/bin" in rewritten
        assert "DISCORD_CHANNEL_ID=1234567890" in rewritten
        assert "UNVERIFIED_SECRET=keep-me" in rewritten
        assert config_path.read_text(encoding="utf-8") == original_config


class TestCliWrappers:
    def test_cmd_prune_defaults_to_dry_run_without_applying(
        self,
        make_profile,
        monkeypatch: pytest.MonkeyPatch,
        capsys,
    ) -> None:
        profile_home = make_profile(
            "coding",
            """BWS_ACCESS_TOKEN=0.coding-token
OPENAI_API_KEY=sk-coding-secret
PATH=/usr/local/bin
""",
            """secrets:
  bitwarden:
    enabled: true
    access_token_env: BWS_ACCESS_TOKEN
    project_id: coding-project
""",
        )
        env_path = profile_home / ".env"
        original_env = env_path.read_text(encoding="utf-8")
        config_path = profile_home / "config.yaml"

        plan = bwm.PrunePlan(
            profile="coding",
            profile_home=profile_home,
            env_path=env_path,
            config_path=config_path,
            settings=bwm.BitwardenSettings(
                enabled=True,
                access_token_env="BWS_ACCESS_TOKEN",
                project_id="coding-project",
                server_url="",
            ),
            rows=[
                bwm.PruneRow(
                    profile="coding",
                    surface=".env",
                    env_path=env_path,
                    key="OPENAI_API_KEY",
                    classification="secret",
                    state="set",
                    action="remove",
                    reason="verified in Bitwarden project",
                    bws_resolved=True,
                ),
                bwm.PruneRow(
                    profile="coding",
                    surface=".env",
                    env_path=env_path,
                    key="PATH",
                    classification="non-secret",
                    state="set",
                    action="keep",
                    reason="non-secret config stays in .env",
                    bws_resolved=None,
                ),
            ],
            warnings=[],
            verification_error=None,
        )

        monkeypatch.setattr(
            "hermes_cli.profiles.resolve_profile_env",
            lambda _profile_name: str(profile_home),
        )
        monkeypatch.setattr(bwm, "prune_plan_for_profile", lambda *_args, **_kwargs: plan)

        def _should_not_apply(_plan):
            raise AssertionError("cmd_prune applied changes in dry-run mode")

        monkeypatch.setattr(bwm, "apply_prune_plan", _should_not_apply)

        rc = secrets_cli.cmd_prune(Namespace(profile="coding", apply=False))
        out = capsys.readouterr().out

        assert rc == 0
        assert "Dry-run only: add --apply to rewrite the .env file after Bitwarden verification." in out
        assert "OPENAI_API_KEY" in out
        assert "PATH" in out
        assert env_path.read_text(encoding="utf-8") == original_env
        assert not list(profile_home.glob("*.bak-pre-bitwarden-prune-*"))
