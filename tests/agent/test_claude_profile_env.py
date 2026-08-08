"""The two directory variables a Claude Code child process needs.

Hermes never hands a token to the child. It names a directory, and the
``claude`` program reads its own secret from that directory. With fewer than
two profiles configured, nothing changes: the child inherits exactly the
environment it inherits today.
"""

import os
from pathlib import Path

import yaml

from agent import claude_cli_profiles as ccp
from hermes_constants import apply_claude_profile_env
from tools.environments.local import _sanitize_subprocess_env, hermes_subprocess_env


def write_config(section):
    path = Path(os.environ["HERMES_HOME"]) / "config.yaml"
    path.write_text(yaml.dump({"claude_cli_profiles": section} if section else {}))
    return path


def configure(tmp_path, count=2):
    write_config({"profiles": [
        {"name": name, "config_dir": str(tmp_path / name)}
        for name in ("work", "spare")[:count]
    ]})


class TestApplyClaudeProfileEnv:
    def test_no_profiles_configured_leaves_the_environment_untouched(self):
        write_config(None)
        env = {"PATH": "/usr/bin"}
        apply_claude_profile_env(env)
        assert env == {"PATH": "/usr/bin"}

    def test_one_profile_configured_leaves_the_environment_untouched(self, tmp_path):
        configure(tmp_path, count=1)
        ccp.record_active("work")
        env = {"PATH": "/usr/bin"}
        apply_claude_profile_env(env)
        assert env == {"PATH": "/usr/bin"}

    def test_the_recorded_profile_sets_both_directory_variables(self, tmp_path):
        configure(tmp_path)
        ccp.record_active("spare")
        env = {}
        apply_claude_profile_env(env)
        assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "spare")
        assert env["CLAUDE_SECURESTORAGE_CONFIG_DIR"] == str(tmp_path / "spare")

    def test_a_recorded_profile_a_person_removed_changes_nothing(self, tmp_path):
        configure(tmp_path)
        ccp.record_active("retired")
        env = {"PATH": "/usr/bin"}
        apply_claude_profile_env(env)
        assert env == {"PATH": "/usr/bin"}

    def test_an_explicit_selection_wins_over_the_recorded_profile(self, tmp_path):
        configure(tmp_path)
        ccp.record_active("work")
        env = {}
        apply_claude_profile_env(
            env, profile_env={"CLAUDE_CONFIG_DIR": "/tmp/chosen",
                              "CLAUDE_SECURESTORAGE_CONFIG_DIR": "/tmp/chosen"}
        )
        assert env["CLAUDE_CONFIG_DIR"] == "/tmp/chosen"

    def test_the_inherited_oauth_token_is_removed_when_a_profile_is_selected(self, tmp_path):
        """That variable overrides the profile's own login. If both are
        present the profile choice does nothing and the run quietly bills the
        wrong account."""
        configure(tmp_path)
        ccp.record_active("work")
        env = {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-inherited"}
        apply_claude_profile_env(env)
        assert "CLAUDE_CODE_OAUTH_TOKEN" not in env

    def test_the_inherited_oauth_token_stays_when_no_profile_is_selected(self):
        """Hermes deliberately passes this variable to an agent-started
        ``claude``. Removing it made the child fall back to the shared store
        and sign the person out. Keep that behaviour when the switcher is
        off."""
        write_config(None)
        env = {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-inherited"}
        apply_claude_profile_env(env)
        assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "sk-ant-oat01-inherited"

    def test_a_config_change_takes_effect_without_a_restart(self, tmp_path):
        """The identity check reads the file every time, so a person who
        edits config.yaml does not have to restart a service."""
        configure(tmp_path, count=1)
        ccp.record_active("work")
        first = {}
        apply_claude_profile_env(first)
        assert first == {}

        configure(tmp_path, count=2)
        second = {}
        apply_claude_profile_env(second)
        assert second["CLAUDE_CONFIG_DIR"] == str(tmp_path / "work")

    def test_the_environment_holds_no_token(self, tmp_path):
        configure(tmp_path)
        ccp.record_active("work")
        env = {}
        apply_claude_profile_env(env)
        assert not any("sk-ant" in value for value in env.values())
        assert set(env) == {"CLAUDE_CONFIG_DIR", "CLAUDE_SECURESTORAGE_CONFIG_DIR"}


class TestSpawnSurfaces:
    def test_the_worker_and_delegation_environment_carries_both_variables(self, tmp_path):
        configure(tmp_path)
        ccp.record_active("spare")
        env = hermes_subprocess_env(inherit_credentials=True)
        assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "spare")
        assert env["CLAUDE_SECURESTORAGE_CONFIG_DIR"] == str(tmp_path / "spare")

    def test_the_terminal_environment_carries_both_variables(self, tmp_path):
        configure(tmp_path)
        ccp.record_active("spare")
        env = _sanitize_subprocess_env({"PATH": "/usr/bin"})
        assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "spare")
        assert env["CLAUDE_SECURESTORAGE_CONFIG_DIR"] == str(tmp_path / "spare")

    def test_the_worker_environment_is_unchanged_with_one_profile(self, tmp_path):
        configure(tmp_path, count=1)
        env = hermes_subprocess_env(inherit_credentials=True)
        assert "CLAUDE_CONFIG_DIR" not in env
        assert "CLAUDE_SECURESTORAGE_CONFIG_DIR" not in env

    def test_the_terminal_environment_is_unchanged_with_one_profile(self, tmp_path):
        configure(tmp_path, count=1)
        env = _sanitize_subprocess_env({"PATH": "/usr/bin"})
        assert "CLAUDE_CONFIG_DIR" not in env


class TestSafeFailureAndRollback:
    def test_an_unreadable_state_file_does_not_stop_a_spawn(self, tmp_path):
        configure(tmp_path)
        ccp.state_path().write_text("{ not json")
        env = hermes_subprocess_env(inherit_credentials=True)
        assert "CLAUDE_CONFIG_DIR" not in env
        assert env.get("PATH")

    def test_an_unreadable_config_file_does_not_stop_a_spawn(self, tmp_path):
        (Path(os.environ["HERMES_HOME"]) / "config.yaml").write_text("{[not: yaml")
        env = hermes_subprocess_env(inherit_credentials=True)
        assert "CLAUDE_CONFIG_DIR" not in env
        assert env.get("PATH")

    def test_a_failure_inside_the_switcher_leaves_the_environment_alone(
        self, tmp_path, monkeypatch
    ):
        configure(tmp_path)
        ccp.record_active("work")
        monkeypatch.setattr(
            ccp, "active_profile_env",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        env = {"PATH": "/usr/bin", "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-inherited"}
        apply_claude_profile_env(env)
        assert env == {"PATH": "/usr/bin", "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-inherited"}

    def test_clearing_the_state_returns_the_spawn_to_the_inherited_account(self, tmp_path):
        configure(tmp_path)
        ccp.record_active("spare")
        assert "CLAUDE_CONFIG_DIR" in hermes_subprocess_env(inherit_credentials=True)

        ccp.clear_state()
        assert "CLAUDE_CONFIG_DIR" not in hermes_subprocess_env(inherit_credentials=True)

    def test_removing_the_second_profile_returns_to_the_old_behaviour(self, tmp_path):
        configure(tmp_path)
        ccp.record_active("work")
        assert "CLAUDE_CONFIG_DIR" in hermes_subprocess_env(inherit_credentials=True)

        configure(tmp_path, count=1)
        assert "CLAUDE_CONFIG_DIR" not in hermes_subprocess_env(inherit_credentials=True)
