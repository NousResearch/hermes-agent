"""Secret-source caches use the canonical home identity (regression for #154609)."""

import os
from pathlib import Path

from agent.secret_sources import registry
from agent.secret_sources.base import FetchResult, SecretSource
from hermes_cli import env_loader
from hermes_constants import hermes_home_key


class _HomeKeyProbe(SecretSource):
    name = "homekeyprobe"
    label = "Home key probe"
    shape = "bulk"

    def fetch(self, cfg: dict, home_path: Path) -> FetchResult:
        return FetchResult(secrets={"HOME_KEY_PROBE_TOKEN": "scoped-value"})


def test_hydration_reset_and_snapshot_share_canonical_home_key(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text(
        "secrets:\n  homekeyprobe:\n    enabled: true\n", encoding="utf-8"
    )
    alias = tmp_path / "alternate-spelling"
    canonical = hermes_home_key(home)
    # Exercise the consumer contract independent of host filesystem case rules.
    monkeypatch.setattr(env_loader, "hermes_home_key", lambda _home: canonical, raising=False)
    monkeypatch.delenv("HOME_KEY_PROBE_TOKEN", raising=False)
    registry._reset_registry_for_tests()
    registry.register_source(_HomeKeyProbe(), replace=True)
    env_loader.reset_secret_source_cache()
    try:
        expected = {"HOME_KEY_PROBE_TOKEN": "scoped-value"}
        assert env_loader.hydrate_profile_secret_sources(home) == expected
        assert env_loader.get_secret_source_values(alias) == expected
        assert env_loader.hydrate_profile_secret_sources(alias) == expected
        env_loader.reset_secret_source_cache(alias)
        assert env_loader.get_secret_source_values(home) == {}
        assert canonical not in env_loader._APPLIED_HOMES
    finally:
        env_loader.reset_secret_source_cache()
        registry._reset_registry_for_tests()


def test_dotenv_apply_restore_and_revoke_share_canonical_home_key(tmp_path, monkeypatch):
    home, alias = tmp_path / "profile", tmp_path / "alternate-spelling"
    for path in (home, alias):
        path.mkdir()
        (path / "config.yaml").write_text(
            "secrets:\n  homekeyprobe:\n    enabled: true\n    override_existing: true\n", encoding="utf-8")
    canonical = hermes_home_key(home)
    monkeypatch.setattr(env_loader, "hermes_home_key", lambda _home: canonical)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HOME_KEY_PROBE_TOKEN", raising=False)
    fetches = []

    class _CountingProbe(_HomeKeyProbe):
        def fetch(self, cfg: dict, home_path: Path) -> FetchResult:
            fetches.append(home_path)
            return super().fetch(cfg, home_path)

    registry._reset_registry_for_tests()
    registry.register_source(_CountingProbe(), replace=True)
    env_loader.reset_secret_source_cache()
    try:
        env_loader.load_hermes_dotenv(hermes_home=home)
        assert os.environ["HOME_KEY_PROBE_TOKEN"] == "scoped-value"
        # Same home, other spelling: the once-per-home guard holds and the source beats a stale .env line.
        (alias / ".env").write_text("HOME_KEY_PROBE_TOKEN=stale\n", encoding="utf-8")
        env_loader.load_hermes_dotenv(hermes_home=alias)
        assert len(fetches) == 1
        assert os.environ["HOME_KEY_PROBE_TOKEN"] == "scoped-value"
        # Source removed: a load under the other spelling takes back the earlier write.
        (alias / ".env").unlink()
        (alias / "config.yaml").write_text("model: {}\n", encoding="utf-8")
        env_loader.reset_secret_source_cache(alias)
        env_loader.load_hermes_dotenv(hermes_home=alias)
        assert "HOME_KEY_PROBE_TOKEN" not in os.environ
    finally:
        env_loader.reset_secret_source_cache()
        registry._reset_registry_for_tests()
