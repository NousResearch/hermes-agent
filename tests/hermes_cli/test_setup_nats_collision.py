"""Tests for NATS profile-lock collision detection in the setup wizard."""
from pathlib import Path

import pytest

from hermes_cli import setup as setup_mod


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    """Isolated home with a default + named-profiles tree.

    Active profile is ``play`` unless overridden in the test.
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    (default_home / "profiles").mkdir()
    play = default_home / "profiles" / "play"
    play.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(play))
    return tmp_path


def _write_profile_config(tmp_path: Path, profile: str, body: str) -> Path:
    """Write a config.yaml under <home>/.hermes/profiles/<profile>/ (or default)."""
    if profile == "default":
        path = tmp_path / ".hermes"
    else:
        path = tmp_path / ".hermes" / "profiles" / profile
        path.mkdir(exist_ok=True)
    cfg = path / "config.yaml"
    cfg.write_text(body)
    return cfg


def _write_profile_env(tmp_path: Path, profile: str, **env_vars: str) -> Path:
    """Write a .env under <home>/.hermes/profiles/<profile>/ (or default)."""
    if profile == "default":
        path = tmp_path / ".hermes"
    else:
        path = tmp_path / ".hermes" / "profiles" / profile
        path.mkdir(exist_ok=True)
    env = path / ".env"
    env.write_text("\n".join(f"{k}={v}" for k, v in env_vars.items()) + "\n")
    return env


class TestFindNatsProfileCollisions:
    def test_no_other_profiles_returns_empty(self, profile_env):
        assert setup_mod._find_nats_profile_collisions("hermes", "alice", "demo") == []

    def test_other_profile_with_disabled_nats_does_not_collide(self, profile_env):
        _write_profile_config(profile_env, "work", """
platforms:
  nats:
    enabled: false
    extra:
      owner: alice
      session_name: demo
""")
        assert setup_mod._find_nats_profile_collisions("hermes", "alice", "demo") == []

    def test_other_profile_with_different_triple_does_not_collide(self, profile_env):
        _write_profile_config(profile_env, "work", """
platforms:
  nats:
    enabled: true
    extra:
      owner: alice
      session_name: prod
""")
        assert setup_mod._find_nats_profile_collisions("hermes", "alice", "demo") == []

    def test_matching_triple_in_named_profile_is_flagged(self, profile_env):
        _write_profile_config(profile_env, "work", """
platforms:
  nats:
    enabled: true
    extra:
      owner: alice
      session_name: demo
""")
        result = setup_mod._find_nats_profile_collisions("hermes", "alice", "demo")
        assert len(result) == 1
        assert result[0]["profile"] == "work"
        assert result[0]["owner"] == "alice"
        assert result[0]["session_name"] == "demo"
        assert result[0]["agent"] == "hermes"
        assert "work/config.yaml" in result[0]["path"]

    def test_matching_triple_in_default_is_flagged(self, profile_env):
        _write_profile_config(profile_env, "default", """
platforms:
  nats:
    enabled: true
    extra:
      owner: alice
      session_name: demo
""")
        result = setup_mod._find_nats_profile_collisions("hermes", "alice", "demo")
        assert [c["profile"] for c in result] == ["default"]

    def test_active_profile_is_excluded_from_collision_scan(
        self, profile_env, monkeypatch
    ):
        # Active profile == 'work', and 'work' has the same triple. Must not flag.
        play = profile_env / ".hermes" / "profiles" / "play"
        work = profile_env / ".hermes" / "profiles" / "work"
        work.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(work))
        _write_profile_config(profile_env, "work", """
platforms:
  nats:
    enabled: true
    extra:
      owner: alice
      session_name: demo
""")
        # Sanity: 'play' has nothing, so only 'work' could collide — but it's active.
        assert setup_mod._find_nats_profile_collisions("hermes", "alice", "demo") == []

    def test_multiple_matching_profiles_all_returned(self, profile_env):
        _write_profile_config(profile_env, "work", """
platforms:
  nats:
    enabled: true
    extra:
      owner: alice
      session_name: demo
""")
        _write_profile_config(profile_env, "default", """
platforms:
  nats:
    enabled: true
    extra:
      owner: alice
      session_name: demo
""")
        result = setup_mod._find_nats_profile_collisions("hermes", "alice", "demo")
        names = sorted(c["profile"] for c in result)
        assert names == ["default", "work"]

    def test_unreadable_sibling_config_is_skipped(self, profile_env):
        # Malformed YAML in 'work' must not block setup; collision check should
        # silently skip it and still detect an unrelated real collision.
        _write_profile_config(profile_env, "work", "platforms:\n  nats:\n    enabled: [unclosed")
        _write_profile_config(profile_env, "side", """
platforms:
  nats:
    enabled: true
    extra:
      owner: alice
      session_name: demo
""")
        result = setup_mod._find_nats_profile_collisions("hermes", "alice", "demo")
        assert [c["profile"] for c in result] == ["side"]

    def test_matches_only_when_all_three_components_match(self, profile_env):
        # 'work' has same owner/session but a different (explicit) agent token.
        _write_profile_config(profile_env, "work", """
platforms:
  nats:
    enabled: true
    extra:
      agent: notes
      owner: alice
      session_name: demo
""")
        # Default agent is "hermes" — different agent token must not collide.
        assert setup_mod._find_nats_profile_collisions("hermes", "alice", "demo") == []
        # ...but matches when the user picks the same explicit agent.
        result = setup_mod._find_nats_profile_collisions("notes", "alice", "demo")
        assert [c["profile"] for c in result] == ["work"]

    # -----------------------------------------------------------------------
    # .env-driven sibling profiles (the wizard's primary output)
    # -----------------------------------------------------------------------

    def test_matching_triple_in_sibling_dotenv_is_flagged(self, profile_env):
        _write_profile_env(
            profile_env,
            "work",
            HERMES_NATS_OWNER="alice",
            HERMES_NATS_SESSION_NAME="demo",
        )
        result = setup_mod._find_nats_profile_collisions("hermes", "alice", "demo")
        assert len(result) == 1
        assert result[0]["profile"] == "work"
        assert result[0]["owner"] == "alice"
        assert result[0]["session_name"] == "demo"
        assert result[0]["agent"] == "hermes"
        assert "work/.env" in result[0]["path"]

    def test_dotenv_with_only_url_marks_profile_as_using_default_agent(
        self, profile_env
    ):
        # _apply_env_overrides treats any NATS env var as an implicit enable.
        # Setting only NATS_URL with no identity vars means the profile WILL
        # try to register at agents.prompt.hermes.<missing>.<missing> — but
        # it can't actually run, since owner/session_name are required by
        # the adapter. So a "URL-only" sibling cannot collide with anyone
        # — no matching identity triple to compare.
        _write_profile_env(
            profile_env,
            "work",
            NATS_URL="nats://demo.nats.io",
        )
        # Owner=None, session=None → triple is ("hermes", None, None) which
        # can't match a real triple ("hermes", "alice", "demo").
        assert setup_mod._find_nats_profile_collisions("hermes", "alice", "demo") == []

    def test_env_overrides_yaml_for_collision_check(self, profile_env):
        # YAML says (hermes, bob, prod); .env overrides owner+session to
        # (alice, demo). Effective triple (env-wins-per-key) collides with
        # ours.
        _write_profile_config(profile_env, "work", """
platforms:
  nats:
    enabled: true
    extra:
      owner: bob
      session_name: prod
""")
        _write_profile_env(
            profile_env,
            "work",
            HERMES_NATS_OWNER="alice",
            HERMES_NATS_SESSION_NAME="demo",
        )
        result = setup_mod._find_nats_profile_collisions("hermes", "alice", "demo")
        assert len(result) == 1
        assert result[0]["profile"] == "work"
        assert result[0]["owner"] == "alice"      # env, not yaml's "bob"
        assert result[0]["session_name"] == "demo"  # env, not yaml's "prod"

    def test_yaml_only_sibling_still_detected_after_dotenv_refactor(
        self, profile_env
    ):
        # Backward-compat: a sibling profile that pre-dates the wizard
        # refactor (or was hand-edited) still uses config.yaml only. Must
        # still be detected.
        _write_profile_config(profile_env, "work", """
platforms:
  nats:
    enabled: true
    extra:
      owner: alice
      session_name: demo
""")
        # No .env in 'work'. Should still flag from the YAML side.
        result = setup_mod._find_nats_profile_collisions("hermes", "alice", "demo")
        assert [c["profile"] for c in result] == ["work"]

    def test_explicit_agent_in_dotenv_is_honored(self, profile_env):
        _write_profile_env(
            profile_env,
            "work",
            HERMES_NATS_AGENT="notes",
            HERMES_NATS_OWNER="alice",
            HERMES_NATS_SESSION_NAME="demo",
        )
        # Default-agent caller doesn't collide — different agent token.
        assert setup_mod._find_nats_profile_collisions("hermes", "alice", "demo") == []
        # Same explicit agent collides.
        result = setup_mod._find_nats_profile_collisions("notes", "alice", "demo")
        assert [c["profile"] for c in result] == ["work"]
