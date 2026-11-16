"""A legacy ``custom_providers:`` entry's ``key_cmd`` reaches auxiliary clients, as a ``providers:`` one does.

Auxiliary tasks (vision, titles, compression) resolve a named custom provider through
``_get_named_custom_provider`` and mint the key from its ``key_cmd``. The legacy matcher dropped the
field, so every auxiliary call sent the ``no-key-required`` placeholder and got a 401 (#156334).
"""

from __future__ import annotations

import hermes_yaml as yaml
import pytest

ENDPOINT = "http://127.0.0.1:18080/v1"
TOKEN = "minted-token-0123456789"


def _write_config(tmp_path, monkeypatch, layout):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    entry = {"base_url": ENDPOINT, "key_cmd": f"echo {TOKEN}"}
    config = {"model": {"default": "m", "provider": "custom:example"}}
    if layout == "legacy":
        config["custom_providers"] = [{"name": "example", "model": "m", **entry}]
    else:
        config["providers"] = {"example": {"default_model": "m", **entry}}
    (hermes_home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


@pytest.mark.parametrize("layout", ["legacy", "providers"])
def test_auxiliary_key_comes_from_key_cmd(tmp_path, monkeypatch, layout):
    _write_config(tmp_path, monkeypatch, layout)
    from agent.auxiliary_client import _named_custom_api_key
    from hermes_cli.runtime_provider import _get_named_custom_provider

    entry = _get_named_custom_provider("custom:example")
    assert entry["key_cmd"] == f"echo {TOKEN}"
    key = _named_custom_api_key(entry, "custom:example", entry["base_url"])
    assert key != "no-key-required"
    assert (key() if callable(key) else key) == TOKEN
