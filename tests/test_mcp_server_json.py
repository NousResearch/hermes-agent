import json
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_server_json_launches_hermes_mcp_serve_from_pypi_extra():
    data = json.loads((ROOT / "server.json").read_text())
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]


    assert data["$schema"] == "https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json"
    assert data["name"] == "io.github.nousresearch/hermes-agent"
    assert data["repository"] == {
        "url": "https://github.com/NousResearch/hermes-agent",
        "source": "github",
        "id": "1024554267",
    }

    assert data["version"] == project["version"]
    assert len(data["packages"]) == 1
    package = data["packages"][0]
    assert package["registryType"] == "pypi"
    assert package["registryBaseUrl"] == "https://pypi.org"
    assert package["identifier"] == f"{project['name']}[mcp]"
    assert package["version"] == project["version"]
    assert package["runtimeHint"] == "uvx"
    assert package["transport"] == {"type": "stdio"}
    assert package["packageArguments"] == [
        {"type": "positional", "value": "mcp"},
        {"type": "positional", "value": "serve"},
    ]


def test_server_json_registry_metadata_stays_small_and_allowed():
    data = json.loads((ROOT / "server.json").read_text())

    assert set(data["_meta"]) == {"io.modelcontextprotocol.registry/publisher-provided"}
    publisher_meta = data["_meta"]["io.modelcontextprotocol.registry/publisher-provided"]
    assert len(json.dumps(publisher_meta).encode()) < 4096
