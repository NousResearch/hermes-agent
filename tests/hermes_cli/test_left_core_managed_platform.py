"""Migration sees administrator-pinned platform credentials across YAML shapes."""
import json

import pytest

from hermes_agent.hermes_cli import left_core_migration as lcm
from hermes_agent.hermes_cli import managed_scope


@pytest.mark.parametrize("shape", [
    "root", "root-extra", "gateway-platforms", "platforms", "gateway", "root-and-gateway", "unmanaged",
])
def test_managed_credentials_outweigh_stale_user_extras(tmp_path, monkeypatch, shape):
    credentials = {"app_id": "fixture-id", "app_secret": "fixture-secret"}
    layers = {
        "root": {"yuanbao": credentials},
        "root-extra": {"yuanbao": {"extra": credentials}},
        "gateway-platforms": {"gateway": {"platforms": {"yuanbao": {"extra": credentials}}}},
        "platforms": {"platforms": {"yuanbao": {"extra": credentials}}},
        "gateway": {"gateway": {"yuanbao": {"extra": credentials}}},
        "root-and-gateway": {"yuanbao": {"app_id": "", "app_secret": ""},
                             "gateway": {"yuanbao": {"extra": credentials}}},
        "unmanaged": {},
    }
    home, managed = tmp_path / "home", tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    config = {"platforms": {"yuanbao": {"extra": {"app_id": "", "app_secret": ""}}}}
    if shape == "unmanaged":
        config["yuanbao"] = credentials
    config_path = home / "config.yaml"
    config_path.write_text(json.dumps(config))
    (managed / "config.yaml").write_text(json.dumps(layers[shape]))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    for key in ("YUANBAO_APP_ID", "YUANBAO_APP_KEY", "YUANBAO_APP_SECRET"):
        monkeypatch.delenv(key, raising=False)
    managed_scope.invalidate_managed_cache()
    try:
        assert lcm.yuanbao_in_use(home) is (shape != "unmanaged")
        assert json.loads(config_path.read_text()) == config
    finally:
        managed_scope.invalidate_managed_cache()
