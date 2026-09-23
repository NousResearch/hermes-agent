"""A config.yaml that does not parse must be repairable from the dashboard.

Every config save refuses to replace a file it cannot parse (the fail-closed guard). The raw YAML
editor sends the whole document and reads nothing back, so it may replace such a file; every other
save still refuses, and the refusal's fix-it message must reach the page instead of a bare 500.
"""

import pytest
import yaml

from hermes_cli.config import get_config_path
from hermes_cli.config_backups import list_config_backups

BROKEN_YAML = "model:\n  default: my-model\n  provider: [unclosed\n"


@pytest.fixture
def client(_isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")
    from hermes_cli import web_server

    client = TestClient(web_server.app, raise_server_exceptions=False)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return client


@pytest.mark.parametrize("broken", [BROKEN_YAML, "- a list\n- is not settings\n"])
def test_raw_editor_replaces_a_config_that_does_not_parse(client, broken):
    config_path = get_config_path()
    config_path.write_text(broken, encoding="utf-8")

    resp = client.put("/api/config/raw", json={"yaml_text": "model:\n  default: repaired\n"})

    assert resp.status_code == 200, resp.text
    assert yaml.safe_load(config_path.read_text(encoding="utf-8"))["model"]["default"] == "repaired"
    backups = list_config_backups(config_path, "corrupt")
    assert backups and backups[0].read_text(encoding="utf-8") == broken, "the replaced bytes must stay recoverable"


@pytest.mark.parametrize(("path", "body", "on_disk", "reason"), [
    ("/api/config", {"config": {"model": {"default": "changed"}}}, "broken", "has a formatting error"),
    ("/api/dashboard/theme", {"name": "default"}, "broken", "has a formatting error"),
    # Only a file that parses badly is replaceable: one that cannot be read is refused even here.
    ("/api/config/raw", {"yaml_text": "model:\n  default: repaired\n"}, "unreadable", "cannot be read"),
])
def test_other_saves_refuse_with_the_fix_it_message(client, path, body, on_disk, reason):
    config_path = get_config_path()
    if on_disk == "broken":
        config_path.write_text(BROKEN_YAML, encoding="utf-8")
    else:
        config_path.mkdir()

    resp = client.put(path, json=body)

    assert resp.status_code == 409, resp.text
    assert reason in resp.json()["detail"]
    if on_disk == "broken":
        assert config_path.read_text(encoding="utf-8") == BROKEN_YAML
    else:
        assert config_path.is_dir()
