"""Diagnostic/regression test: legacy `custom_providers` entries must be
surfaced by GET /api/providers/custom-endpoints.

Root cause of the Provider Manager "edit opens Add modal" + "Unknown custom
provider: custom:…" bugs: the desktop manager sources custom providers from
/api/providers/custom-endpoints, but that endpoint only iterated the keyed
`providers:` dict. A provider stored in the legacy `custom_providers` list
(which the model-options catalog still surfaces, with a `custom:<name>` slug)
was therefore invisible to the manager's edit / test-connection / enable logic.
"""
import pytest


@pytest.fixture
def client(tmp_path, monkeypatch, _isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    # A legacy custom_providers entry (pre-v12 schema) — no `providers:` block.
    (home / "config.yaml").write_text(
        "custom_providers:\n"
        "  - name: Legacy Lab\n"
        "    base_url: https://legacy.example/v1\n"
        "    api_mode: chat_completions\n"
        "    models:\n"
        "      legacy-model: {}\n",
        encoding="utf-8",
    )
    (home / ".env").write_text("", encoding="utf-8")

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")
    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def test_legacy_custom_provider_is_surfaced(client):
    """A legacy custom_providers entry appears in the endpoints list."""
    resp = client.get("/api/providers/custom-endpoints")

    assert resp.status_code == 200
    endpoints = resp.json()["endpoints"]
    ids = [e["id"] for e in endpoints]
    # _custom_endpoint_id("Legacy Lab") -> "legacy-lab"
    assert "legacy-lab" in ids

    legacy = next(e for e in endpoints if e["id"] == "legacy-lab")
    assert legacy["base_url"] == "https://legacy.example/v1"
    assert "legacy-model" in legacy["models"]
    assert legacy["enabled"] is True


def test_legacy_and_providers_dedup_by_id(client, tmp_path):
    """When the same id exists in both schemas, the providers: row wins (no dup)."""
    from hermes_constants import get_hermes_home

    (get_hermes_home() / "config.yaml").write_text(
        "providers:\n"
        "  legacy-lab:\n"
        "    name: Legacy Lab\n"
        "    base_url: https://new.example/v1\n"
        "custom_providers:\n"
        "  - name: Legacy Lab\n"
        "    base_url: https://legacy.example/v1\n",
        encoding="utf-8",
    )

    resp = client.get("/api/providers/custom-endpoints")
    endpoints = resp.json()["endpoints"]
    matching = [e for e in endpoints if e["id"] == "legacy-lab"]

    assert len(matching) == 1
    # The providers: row (source=providers) takes precedence.
    assert matching[0]["base_url"] == "https://new.example/v1"
    assert matching[0]["source"] == "providers"
