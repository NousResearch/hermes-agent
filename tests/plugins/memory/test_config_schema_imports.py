"""Regression for #147384: schema helpers resolve inside each provider directory."""

import os
from pathlib import Path

import pytest

from plugins.memory.config_schema import get_provider_config_schema


def _provider(home, isolation, label):
    directory = home / "plugins" / "schema-probe"
    directory.mkdir(parents=True)
    (home / "config.yaml").write_text(
        f"plugins:\n  isolation: {isolation}\n", encoding="utf-8"
    )
    (directory / "__init__.py").write_text(
        '"""register_memory_provider"""\n'
        'raise AssertionError("provider runtime imported while reading schema")\n',
        encoding="utf-8",
    )
    (directory / "metadata.py").write_text(f"LABEL = {label!r}\n", encoding="utf-8")
    (directory / "schema_data").mkdir()
    (directory / "schema_data" / "__init__.py").write_text("", encoding="utf-8")
    (directory / "schema_data" / "fields.py").write_text(
        "import os\n"
        "from dataclasses import dataclass\n"
        "from ..metadata import LABEL\n"
        "from plugins.memory.config_schema import ProviderConfigSchema, ProviderField\n"
        "@dataclass\n"
        "class Defaults:\n"
        "    pid: str = str(os.getpid())\n"
        'CONFIG_SCHEMA = ProviderConfigSchema(name="schema-probe", label=LABEL,\n'
        '    fields=(ProviderField(key="pid", label="PID", default=Defaults().pid),))\n',
        encoding="utf-8",
    )
    (directory / "config_schema.py").write_text(
        "from .schema_data.fields import CONFIG_SCHEMA\n", encoding="utf-8"
    )
    return directory


@pytest.mark.platforms("any")
@pytest.mark.parametrize("isolation", ["in_process", "host"])
def test_schema_helpers_are_profile_local_and_leave_runtime_unloaded(tmp_path, monkeypatch, isolation):
    from hermes_cli.plugin_isolation import user_plugin_host
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app
    from starlette.testclient import TestClient

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    homes = [tmp_path / label for label in ("first", "second")]
    for home in homes:
        _provider(home, isolation, home.name)

    hosts = []
    try:
        with TestClient(app, headers={_SESSION_HEADER_NAME: _SESSION_TOKEN}) as client:
            for home in [*homes, homes[0]]:
                monkeypatch.setenv("HERMES_HOME", str(home))
                host = user_plugin_host()
                if host is not None:
                    hosts.append(host)
                schema = get_provider_config_schema("schema-probe")
                assert schema is not None
                assert schema.label == home.name
                pid = int(schema.fields[0].default)
                if host is not None:
                    assert pid != os.getpid()
                else:
                    assert pid == os.getpid()
                response = client.get(
                    "/api/memory/providers/schema-probe/config", params={"surface": "declared"}
                )
                assert response.status_code == 200, response.text
                assert response.json()["fields"][0]["value"] == str(pid)
                assert get_provider_config_schema("schema-probe") is schema
    finally:
        for host in hosts:
            host.shutdown()


@pytest.mark.platforms("any")
@pytest.mark.parametrize("isolation", ["in_process", "host"])
def test_failed_schema_load_retries_its_helpers(tmp_path, monkeypatch, isolation):
    from hermes_cli.plugin_isolation import user_plugin_host

    home = tmp_path / "retry"
    directory = _provider(home, isolation, "Recovered")
    marker = home / "state"
    marker.write_text("before", encoding="utf-8")
    (directory / "early.py").write_text(
        f"from pathlib import Path\nSTATE = Path({str(marker)!r}).read_text(encoding='utf-8')\n",
        encoding="utf-8",
    )
    (directory / "config_schema.py").write_text(
        "from .early import STATE\n"
        "from .repaired import READY\n"
        'assert STATE == "after", "cached helper survived a failed schema load"\n'
        "from .schema_data.fields import CONFIG_SCHEMA\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    host = user_plugin_host()
    try:
        assert get_provider_config_schema("schema-probe") is None
        marker.write_text("after", encoding="utf-8")
        (directory / "repaired.py").write_text("READY = True\n", encoding="utf-8")
        schema = get_provider_config_schema("schema-probe")
        assert schema is not None
        assert schema.label == "Recovered"
    finally:
        if host is not None:
            host.shutdown()
