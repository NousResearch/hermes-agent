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


ACCESS_TOKEN_SENTINEL = "ACCESS_TOKEN_SENTINEL_12345"
SECRET_VALUE_SENTINEL = "SECRET_VALUE_SENTINEL_12345"
STDERR_SENTINEL = "STDERR_SENTINEL_12345"
EXCEPTION_SENTINEL = "EXCEPTION_SENTINEL_12345"


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
            f"""BWS_ACCESS_TOKEN={ACCESS_TOKEN_SENTINEL}
OPENAI_API_KEY={SECRET_VALUE_SENTINEL}
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
        assert SECRET_VALUE_SENTINEL not in out
        assert ACCESS_TOKEN_SENTINEL not in out
        assert "ghp_ops_secret" not in out
        assert "0.ops-token" not in out


class TestPrunePlanning:
    @pytest.mark.parametrize(("apply", "expected_rc"), [(False, 0), (True, 1)])
    def test_verification_exception_is_redacted_and_never_mutates_env(
        self,
        apply: bool,
        expected_rc: int,
        make_profile,
        monkeypatch: pytest.MonkeyPatch,
        capsys,
    ) -> None:
        profile_home = make_profile(
            "coding",
            f"""BWS_ACCESS_TOKEN={ACCESS_TOKEN_SENTINEL}
OPENAI_API_KEY={SECRET_VALUE_SENTINEL}
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

        def failing_fetch(**kwargs):
            assert kwargs["access_token"] == ACCESS_TOKEN_SENTINEL
            raise RuntimeError(
                f"{EXCEPTION_SENTINEL}: backend stderr {STDERR_SENTINEL}; "
                f"token={ACCESS_TOKEN_SENTINEL}; value={SECRET_VALUE_SENTINEL}"
            )

        monkeypatch.setattr(bwm, "fetch_bitwarden_secrets", failing_fetch)
        monkeypatch.setattr(
            "hermes_cli.profiles.resolve_profile_env",
            lambda _profile_name: str(profile_home),
        )

        rc = secrets_cli.cmd_prune(Namespace(profile="coding", apply=apply))
        out = capsys.readouterr().out

        assert rc == expected_rc
        assert (
            "Bitwarden verification failed; no plaintext secrets were removed. "
            "Check Bitwarden configuration and credentials, then retry."
        ) in out
        assert EXCEPTION_SENTINEL not in out
        assert STDERR_SENTINEL not in out
        assert ACCESS_TOKEN_SENTINEL not in out
        assert SECRET_VALUE_SENTINEL not in out
        assert env_path.read_text(encoding="utf-8") == original_env
        assert not list(profile_home.glob(".env.bak-pre-bitwarden-prune-*"))

    def test_config_parse_exception_is_redacted_and_never_mutates_env(
        self,
        make_profile,
        monkeypatch: pytest.MonkeyPatch,
        capsys,
    ) -> None:
        profile_home = make_profile(
            "coding",
            f"""BWS_ACCESS_TOKEN={ACCESS_TOKEN_SENTINEL}
OPENAI_API_KEY={SECRET_VALUE_SENTINEL}
""",
            "secrets: ignored-by-failing-parser\n",
        )
        env_path = profile_home / ".env"
        original_env = env_path.read_text(encoding="utf-8")

        def failing_yaml_load(_contents):
            raise RuntimeError(
                f"{EXCEPTION_SENTINEL}: {STDERR_SENTINEL}; "
                f"token={ACCESS_TOKEN_SENTINEL}; value={SECRET_VALUE_SENTINEL}"
            )

        monkeypatch.setattr(bwm.yaml, "safe_load", failing_yaml_load)
        monkeypatch.setattr(
            "hermes_cli.profiles.resolve_profile_env",
            lambda _profile_name: str(profile_home),
        )

        rc = secrets_cli.cmd_prune(Namespace(profile="coding", apply=True))
        out = capsys.readouterr().out

        assert rc == 1
        assert "could not read config.yaml; fix its YAML syntax and retry" in out
        assert EXCEPTION_SENTINEL not in out
        assert STDERR_SENTINEL not in out
        assert ACCESS_TOKEN_SENTINEL not in out
        assert SECRET_VALUE_SENTINEL not in out
        assert env_path.read_text(encoding="utf-8") == original_env
        assert not list(profile_home.glob(".env.bak-pre-bitwarden-prune-*"))

    @pytest.mark.parametrize("apply", [False, True])
    def test_fetch_warnings_are_redacted_from_prune_output(
        self,
        apply: bool,
        make_profile,
        monkeypatch: pytest.MonkeyPatch,
        capsys,
    ) -> None:
        profile_home = make_profile(
            "coding",
            f"""BWS_ACCESS_TOKEN={ACCESS_TOKEN_SENTINEL}
OPENAI_API_KEY={SECRET_VALUE_SENTINEL}
PATH=/usr/local/bin
""",
            """secrets:
  bitwarden:
    enabled: true
    access_token_env: BWS_ACCESS_TOKEN
    project_id: coding-project
""",
        )

        def warning_fetch(**kwargs):
            assert kwargs["access_token"] == ACCESS_TOKEN_SENTINEL
            return (
                {"OPENAI_API_KEY": SECRET_VALUE_SENTINEL},
                [f"backend stderr {STDERR_SENTINEL}: {SECRET_VALUE_SENTINEL}"],
            )

        monkeypatch.setattr(bwm, "fetch_bitwarden_secrets", warning_fetch)
        monkeypatch.setattr(
            "hermes_cli.profiles.resolve_profile_env",
            lambda _profile_name: str(profile_home),
        )

        rc = secrets_cli.cmd_prune(Namespace(profile="coding", apply=apply))
        out = capsys.readouterr().out

        assert rc == 0
        assert (
            "warning: Bitwarden verification returned 1 warning(s); "
            "backend details were redacted."
        ) in out
        assert STDERR_SENTINEL not in out
        assert ACCESS_TOKEN_SENTINEL not in out
        assert SECRET_VALUE_SENTINEL not in out
        rewritten = (profile_home / ".env").read_text(encoding="utf-8")
        if apply:
            assert "OPENAI_API_KEY=" not in rewritten
        else:
            assert f"OPENAI_API_KEY={SECRET_VALUE_SENTINEL}" in rewritten

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

    def test_shell_only_bootstrap_token_verifies_target_profile(
        self,
        make_profile,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        profile_home = make_profile(
            "coding",
            "OPENAI_API_KEY=plaintext-placeholder\n",
            """secrets:
  bitwarden:
    enabled: true
    access_token_env: BWS_ACCESS_TOKEN
    project_id: coding-project
""",
        )
        monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.shell-token")

        def fake_fetch(**kwargs):
            assert kwargs["access_token"] == "0.shell-token"
            assert kwargs["project_id"] == "coding-project"
            assert kwargs["home_path"] == profile_home
            return ({"OPENAI_API_KEY": "resolved-value"}, [])

        plan = bwm.prune_plan_for_profile(
            "coding",
            profile_home,
            fetcher=fake_fetch,
        )

        assert plan.verification_error is None
        assert plan.removable_keys() == ["OPENAI_API_KEY"]


class TestPruneApply:
    @pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX mode bits not enforced on Windows")
    def test_apply_prune_plan_preserves_env_mode_and_creates_0600_backup(
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
        env_path.chmod(0o640)
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
        assert plan.warnings == [
            "Bitwarden verification returned 1 warning(s); backend details were redacted."
        ]
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
        assert stat.S_IMODE(env_path.stat().st_mode) == 0o640
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

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX mode bits not enforced on Windows")
    def test_apply_prunes_complete_multiline_assignment_atomically(
        self,
        make_profile,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original_env = """# keep leading comment
BWS_ACCESS_TOKEN=0.coding-token
API_URL=https://api.example.test/v1
PRIVATE_KEY="line-one
line-two"
# keep middle comment
PORT=8443
CHANNEL_ID=1234567890
UNVERIFIED_SECRET=keep-me
"""
        profile_home = make_profile(
            "coding",
            original_env,
            """secrets:
  bitwarden:
    enabled: true
    access_token_env: BWS_ACCESS_TOKEN
    project_id: coding-project
""",
        )
        env_path = profile_home / ".env"
        original_mode = stat.S_IMODE(env_path.stat().st_mode)

        def fake_fetch(**_kwargs):
            return ({"PRIVATE_KEY": "line-one\nline-two"}, [])

        atomic_calls: list[tuple[Path, Path]] = []
        real_atomic_replace = bwm.atomic_replace

        def tracking_atomic_replace(source, destination):
            source_path = Path(source)
            destination_path = Path(destination)
            assert source_path.exists()
            assert source_path.parent == env_path.parent
            atomic_calls.append((source_path, destination_path))
            real_atomic_replace(source, destination)

        monkeypatch.setattr(bwm, "atomic_replace", tracking_atomic_replace)

        plan = bwm.prune_plan_for_profile(
            "coding",
            profile_home,
            fetcher=fake_fetch,
        )
        result = bwm.apply_prune_plan(plan)

        assert plan.verification_error is None
        assert plan.removable_keys() == ["PRIVATE_KEY"]
        assert result.changed is True
        assert result.removed_keys == ["PRIVATE_KEY"]
        assert len(atomic_calls) == 1
        assert atomic_calls[0][1] == env_path
        assert result.backup_path is not None
        assert result.backup_path.read_text(encoding="utf-8") == original_env
        assert stat.S_IMODE(result.backup_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(env_path.stat().st_mode) == original_mode
        assert env_path.read_text(encoding="utf-8") == (
            "# keep leading comment\n"
            "BWS_ACCESS_TOKEN=0.coding-token\n"
            "API_URL=https://api.example.test/v1\n"
            "# keep middle comment\n"
            "PORT=8443\n"
            "CHANNEL_ID=1234567890\n"
            "UNVERIFIED_SECRET=keep-me\n"
        )

    def test_unterminated_multiline_assignment_fails_closed(
        self,
        make_profile,
    ) -> None:
        original_env = """BWS_ACCESS_TOKEN=0.coding-token
PRIVATE_KEY="line-one
line-two
OPENAI_API_KEY=must-not-be-removed
"""
        profile_home = make_profile(
            "coding",
            original_env,
            """secrets:
  bitwarden:
    enabled: true
    access_token_env: BWS_ACCESS_TOKEN
    project_id: coding-project
""",
        )

        plan = bwm.prune_plan_for_profile(
            "coding",
            profile_home,
            fetcher=lambda **_kwargs: (
                {"PRIVATE_KEY": "resolved", "OPENAI_API_KEY": "resolved"},
                [],
            ),
        )
        result = bwm.apply_prune_plan(plan)

        assert plan.verification_error == (
            "Could not safely parse .env; no plaintext secrets were removed. "
            "Fix unterminated quoted values, then retry."
        )
        assert plan.removable_keys() == []
        assert result.changed is False
        assert (profile_home / ".env").read_text(encoding="utf-8") == original_env
        assert not list(profile_home.glob(".env.bak-pre-bitwarden-prune-*"))


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
