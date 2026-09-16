"""The dashboard's Honcho save path shares the plugin's write invariants: it refuses to rewrite a
honcho.json that exists but does not parse, and it merges into a parseable one."""

import json

import pytest

from hermes_cli.web_routers import memory_providers as mp
from plugins.memory.honcho.config_schema import CONFIG_SCHEMA


def _point_at(monkeypatch, path):
    from plugins.memory.honcho.client import _host_block
    monkeypatch.setattr(mp, "_honcho_resolvers", lambda: (lambda: "hermes", lambda: path, _host_block))


@pytest.mark.parametrize("corrupt", [True, False], ids=["unparseable-file-is-left-alone", "parseable-file-is-merged"])
def test_web_save_never_replaces_an_unparseable_honcho_json(tmp_path, monkeypatch, corrupt):
    path = tmp_path / "honcho.json"
    before = "{not json" if corrupt else json.dumps({"hosts": {"other": {"apiKey": "keep-me"}}})
    path.write_text(before, encoding="utf-8")
    _point_at(monkeypatch, path)

    if corrupt:
        with pytest.raises(ValueError):
            mp._write_provider_honcho(CONFIG_SCHEMA, {"recallMode": "tools"})
        assert path.read_text(encoding="utf-8") == before
        return
    mp._write_provider_honcho(CONFIG_SCHEMA, {"recallMode": "tools"})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["hosts"]["other"]["apiKey"] == "keep-me"
    assert data["hosts"]["hermes"]["recallMode"] == "tools"
