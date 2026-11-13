"""Secret-source caches use the canonical home identity (regression for #154609)."""

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
