"""Route editor writes replace routes but preserve plugin-owned preset snapshots."""
from copy import deepcopy

import hermes_yaml as yaml

pytest_plugins = ("tests.hermes_cli.test_web_delegation_routing",)


def test_route_edit_preserves_saved_snapshots_and_does_not_return_them(client, homes):
    _, work = homes
    path = work / "config.yaml"
    data = yaml.safe_load(path.read_text())
    block = data["delegation"]["backends"]["model_map"]
    snapshot = {"name": "saved", "options": deepcopy(block["options"]), "default": block["default"]}
    block.update(presets=[snapshot], active_preset="saved", future_metadata={"keep": True})
    path.write_text(yaml.safe_dump(data))

    response = client.put("/api/delegation/backends/model_map?profile=work", json={
        "options": [{"name": "only", "description": "New", "provider": "openrouter", "model": "new"}],
        "default": None,
    })
    assert response.status_code == 200, response.text
    stored = yaml.safe_load(path.read_text())["delegation"]["backends"]["model_map"]
    assert stored["presets"] == [snapshot]
    assert stored["active_preset"] == "saved"
    assert stored["future_metadata"] == {"keep": True}
    assert "default" not in stored
    assert [o["name"] for o in stored["options"]] == ["only"]
    assert "presets" not in response.json()["model_map"]
    assert "presets" not in client.get("/api/delegation/routing?profile=work").json()["model_map"]
