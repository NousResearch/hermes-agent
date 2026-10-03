"""Bootstrap ownership survives source disablement (regression for #77483)."""
from pathlib import Path

import pytest

from agent.secret_sources import registry as reg
from agent.secret_sources.base import FetchResult, SecretSource
from agent.secret_sources.bitwarden import BitwardenSource


class SupplyingSource(SecretSource):
    name = "fixture"
    shape = "mapped"

    def fetch(self, cfg, home_path):
        return FetchResult(secrets=cfg["values"])

    def override_existing(self, cfg):
        return True


@pytest.fixture(autouse=True)
def clean_registry():
    reg._reset_registry_for_tests()
    yield
    reg._reset_registry_for_tests()


@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("existing", [None, "", "bootstrap"])
def test_disabled_builtin_keeps_bootstrap_ownership(tmp_path, monkeypatch, configured, existing):
    token = "CUSTOM_BOOTSTRAP" if configured else "BWS_ACCESS_TOKEN"
    cfg = {"fixture": {"enabled": True, "values": {token: "replacement", "API_KEY": "usable"}},
           "bitwarden": {"enabled": False}}
    if configured:
        cfg["bitwarden"]["access_token_env"] = token
    def forbidden_fetch(*args):
        pytest.fail("disabled Bitwarden must not fetch")
    monkeypatch.setattr(BitwardenSource, "fetch", forbidden_fetch)
    reg.list_sources(scope=str(tmp_path))  # Real bundled discovery, not an empty registry stand-in.
    assert reg.register_source(SupplyingSource())
    env = {} if existing is None else {token: existing}
    report = reg.apply_all(cfg, tmp_path, environ=env)
    assert env.get(token) == existing
    assert env["API_KEY"] == "usable"
    assert [s.name for s in report.sources] == ["fixture"]
    assert report.sources[0].skipped_protected == [token]
    assert token not in report.provenance


def test_disabled_scoped_source_protection_is_home_local(tmp_path):
    class DisabledSource(SecretSource):
        name = "scoped"
        token_env_key = "token_env"
        default_token_env = "SCOPED_TOKEN"

        def fetch(self, cfg, home_path):
            pytest.fail("disabled scoped source must not fetch")

    home_a, home_b = tmp_path / "a", tmp_path / "b"
    home_a.mkdir()
    home_b.mkdir()
    reg.list_sources(scope=str(home_a))
    assert reg.register_source(DisabledSource(), scope=str(home_a))
    assert reg.register_source(SupplyingSource())
    cfg = {"fixture": {"enabled": True, "values": {"SCOPED_TOKEN": "replacement"}}}
    for home in (home_a, home_b, home_a):
        env = {"SCOPED_TOKEN": "bootstrap"}
        report = reg.apply_all(cfg, Path(home), environ=env)
        if home == home_a:
            assert env["SCOPED_TOKEN"] == "bootstrap"
            assert report.sources[0].skipped_protected == ["SCOPED_TOKEN"]
        else:
            assert env["SCOPED_TOKEN"] == "replacement"
            assert report.sources[0].applied == ["SCOPED_TOKEN"]
