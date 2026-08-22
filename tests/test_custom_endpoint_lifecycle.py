"""Custom endpoint lifecycle: exact-key resolution and duplicate guards.

The Desktop settings panel round-trips the literal ``providers`` key, which
for older installs contains characters the slugger rewrites (e.g.
``custom:sakiko-dev``). Slugging that id on the way back in silently targeted
a *different* key — deletes and activates 404'd, and an edit wrote a brand
new ``custom-sakiko-dev`` entry alongside the original, splitting the API key
onto the twin and producing duplicate rows in the panel.

These tests pin the behaviour contract:

- an id that exactly matches an existing providers key resolves to that key
  (edit/delete/activate), and only unknown ids fall through to the slugger;
- creating an endpoint whose ``base_url`` is already configured is rejected
  with 409 instead of forking a twin;
- the duplicate guard never blocks editing the entry that owns the URL.
"""

import os

import pytest
import yaml

from hermes_cli import web_server


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with one colon-prefixed and one plain endpoint."""
    config = {
        "model": {"provider": "gyz", "default": "gyz-model"},
        "providers": {
            "custom:sakiko-dev": {
                "name": "Sakiko Dev",
                "base_url": "https://sakiko.dev/v1",
                "model": "tokenrhythm/deepseek-v4-flash-0731",
                "api_key": "sk-test-plaintext-key",
                "models": {"tokenrhythm/deepseek-v4-flash-0731": {}},
            },
            "gyz": {
                "name": "gyz",
                "base_url": "https://api.bailan.store",
                "model": "gyz-model",
                "api_key": "sk-gyz-key",
                "models": {"gyz-model": {}},
            },
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _load_cfg(home):
    return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))


def _body(**kwargs):
    defaults = dict(
        name="Sakiko Dev", base_url="https://sakiko.dev/v1",
        model="tokenrhythm/deepseek-v4-flash-0731",
        discover_models=True, make_default=False,
    )
    defaults.update(kwargs)
    return web_server.CustomEndpointUpdate(**defaults)


def test_edit_preserves_colon_prefixed_key(hermes_home):
    """Editing custom:sakiko-dev must update that entry, not fork a twin."""
    cfg = _load_cfg(hermes_home)
    endpoint_id, _entry = web_server._write_custom_endpoint(cfg, _body(id="custom:sakiko-dev"))
    assert endpoint_id == "custom:sakiko-dev"
    assert "custom-sakiko-dev" not in cfg["providers"]


def test_edit_merges_models_onto_original_entry(hermes_home):
    cfg = _load_cfg(hermes_home)
    web_server._write_custom_endpoint(cfg, _body(id="custom:sakiko-dev", model="agy/gemini-3.6-flash-high"))
    models = cfg["providers"]["custom:sakiko-dev"]["models"]
    assert "agy/gemini-3.6-flash-high" in models


def test_delete_resolves_exact_key(hermes_home):
    """Deleting a colon-prefixed endpoint removes it (previously a 404)."""
    web_server.delete_custom_endpoint("custom:sakiko-dev")
    providers = _load_cfg(hermes_home)["providers"]
    assert "custom:sakiko-dev" not in providers
    assert "gyz" in providers  # siblings untouched


def test_delete_plain_key_still_works(hermes_home):
    web_server.delete_custom_endpoint("gyz")
    assert "gyz" not in _load_cfg(hermes_home)["providers"]


def test_activate_resolves_exact_key(hermes_home):
    """Activating a colon-prefixed endpoint works (previously a 404)."""
    result = web_server.activate_custom_endpoint("custom:sakiko-dev")
    assert result["ok"] is True
    assert result["provider"] == "custom:sakiko-dev"
    assert _load_cfg(hermes_home)["model"]["provider"] == "custom:sakiko-dev"


def test_unknown_id_still_404s(hermes_home):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        web_server.delete_custom_endpoint("no-such-endpoint")
    assert excinfo.value.status_code == 404


def test_fresh_names_still_get_slugged():
    providers = {"custom:sakiko-dev": {}}
    assert web_server._resolve_provider_key(providers, "My New Endpoint!") == "my-new-endpoint"
    assert web_server._resolve_provider_key(providers, "custom:sakiko-dev") == "custom:sakiko-dev"


def test_duplicate_base_url_rejected_on_create(hermes_home):
    """Creating a second endpoint for an existing base_url -> 409, no twin."""
    from fastapi import HTTPException

    cfg = _load_cfg(hermes_home)
    with pytest.raises(HTTPException) as excinfo:
        web_server._write_custom_endpoint(cfg, _body(name="sakiko-twin"))
    assert excinfo.value.status_code == 409
    assert "custom:sakiko-dev" in excinfo.value.detail
    assert set(cfg["providers"]) == {"custom:sakiko-dev", "gyz"}


def test_duplicate_base_url_normalized(hermes_home):
    """Case and trailing-slash variants of an existing URL also 409."""
    from fastapi import HTTPException

    cfg = _load_cfg(hermes_home)
    with pytest.raises(HTTPException) as excinfo:
        web_server._write_custom_endpoint(cfg, _body(name="twin2", base_url="HTTPS://SAKIKO.DEV/v1/"))
    assert excinfo.value.status_code == 409


def test_duplicate_guard_does_not_block_editing_owner(hermes_home):
    """Saving an endpoint that keeps its own URL is an edit, not a duplicate."""
    cfg = _load_cfg(hermes_home)
    endpoint_id, _ = web_server._write_custom_endpoint(cfg, _body(id="custom:sakiko-dev"))
    assert endpoint_id == "custom:sakiko-dev"


def test_create_with_fresh_url_succeeds(hermes_home):
    cfg = _load_cfg(hermes_home)
    endpoint_id, _ = web_server._write_custom_endpoint(
        cfg, _body(name="fresh-endpoint", base_url="http://127.0.0.1:9999/v1", model="m"))
    assert endpoint_id == "fresh-endpoint"
    assert "fresh-endpoint" in cfg["providers"]
