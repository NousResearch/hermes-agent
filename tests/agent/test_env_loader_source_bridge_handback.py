"""A secret source that took over a YAML-bridged value hands it back to the bridge when removed.

Otherwise the restored value reads as the operator's, and a later config.yaml tightening of the same
allowlist is ignored: the runner's authorization keeps admitting the removed user.
"""

import json
import os

import pytest

from gateway.config_loader import load_yaml_layer
from hermes_cli import env_loader
from tests.agent.test_env_loader_reload_restore import _register_fake_source

KEY = "TELEGRAM_ALLOWED_USERS"
MARKER = "HERMES_BRIDGED_" + KEY


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    from agent.secret_sources import registry

    registry._reset_registry_for_tests()
    env_loader.reset_secret_source_cache()
    for name in ("_LOADED_DOTENV_KEYS", "_MANAGED_DOTENV_KEYS"):
        monkeypatch.setattr(env_loader, name, set())
    monkeypatch.setattr(env_loader, "_DOTENV_PUBLISHED", {})
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for key in (KEY, MARKER):  # setenv first so teardown also removes what the loaders write
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    yield
    registry._reset_registry_for_tests()
    env_loader.reset_secret_source_cache()


def _config(home, users, source=False):
    text = "telegram:\n  allow_from: " + json.dumps(users) + "\n"
    if source:
        text += "secrets:\n  fakebulk:\n    enabled: true\n"
    (home / "config.yaml").write_text(text, encoding="utf-8")


@pytest.mark.parametrize("source_value, override", [("333", True), ("111,222", False)],
                         ids=["source_overrode", "source_skipped_equal"])
def test_removed_source_returns_the_list_to_yaml(tmp_path, source_value, override):
    from gateway.authz_mixin import GatewayAuthorizationMixin
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource

    _config(tmp_path, ["111", "222"])
    load_yaml_layer(tmp_path, {})
    _register_fake_source({KEY: source_value}, override_existing=override)
    _config(tmp_path, ["111", "222"], source=True)
    env_loader.load_hermes_dotenv(hermes_home=tmp_path)
    assert os.environ[KEY] == source_value and MARKER not in os.environ

    _config(tmp_path, ["111"])  # source removed, list tightened in the same edit
    env_loader.reset_secret_source_cache(tmp_path)
    env_loader.load_hermes_dotenv(hermes_home=tmp_path)
    load_yaml_layer(tmp_path, {})

    runner = GatewayAuthorizationMixin()
    runner.config, runner.adapters, runner.pairing_store = GatewayConfig(), {}, None
    removed = SessionSource(platform=Platform.TELEGRAM, user_id="222", chat_id="222", chat_type="dm")
    assert os.environ[KEY] == "111"
    assert runner._is_user_authorized(removed) is False
