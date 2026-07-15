"""Dashboard config surface for ``model.allowlist`` (the /model switch gate).

The dashboard's editable schema is built from ``DEFAULT_CONFIG``, where
``model`` is deliberately a flat string — so dict-only model subkeys are
surfaced as virtual top-level fields through the normalize/denormalize cycle
(the same mechanism as ``model_context_length``). These tests pin:

  - ``model_allowlist`` is registered in CONFIG_SCHEMA (list field, rendered
    adjacent to ``model``), so the supported dashboard config UI can edit it
  - the explicit no-op default: a config without ``model.allowlist`` (dict or
    bare-string ``model``) normalizes to ``[]``, meaning "no restriction"
  - GET normalization surfaces ``model.allowlist`` → ``model_allowlist``
  - PUT denormalization writes it back into the ``model`` dict, clears it on
    empty, preserves it when the payload omits the field, upgrades a
    bare-string ``model`` to the dict form, and never drops sibling subkeys
"""

import pytest

from hermes_cli.web_server import (
    CONFIG_SCHEMA,
    _denormalize_config_from_web,
    _normalize_config_for_web,
    _sanitize_model_allowlist,
)


# ---------------------------------------------------------------------------
# Schema registration
# ---------------------------------------------------------------------------


def test_model_allowlist_registered_in_config_schema():
    """The dashboard schema must expose model_allowlist as an editable list."""
    entry = CONFIG_SCHEMA.get("model_allowlist")
    assert entry is not None, "model_allowlist missing from CONFIG_SCHEMA"
    assert entry["type"] == "list"
    assert entry["category"] == "general"


def test_model_allowlist_renders_adjacent_to_model():
    """Virtual model fields sit right after ``model`` so the frontend groups
    the three fields that edit the same on-disk ``model:`` section."""
    keys = list(CONFIG_SCHEMA.keys())
    i_model = keys.index("model")
    assert keys[i_model + 1] == "model_context_length"
    assert keys[i_model + 2] == "model_allowlist"


# ---------------------------------------------------------------------------
# Sanitizer + the explicit no-op default
# ---------------------------------------------------------------------------


def test_sanitize_model_allowlist_defaults_and_malformed():
    assert _sanitize_model_allowlist(None) == []
    assert _sanitize_model_allowlist("not-a-list") == []
    assert _sanitize_model_allowlist([]) == []
    # trims, drops blanks + non-strings, de-dupes, preserves order + case
    assert _sanitize_model_allowlist([" a/b ", "", 5, None, "c/d", "a/b"]) == [
        "a/b",
        "c/d",
    ]


def test_normalize_defaults_to_empty_allowlist_for_dict_model():
    """No ``model.allowlist`` on disk → the field still renders, as [] (no
    restriction) — the explicit no-op default."""
    out = _normalize_config_for_web(
        {"model": {"default": "a/b", "provider": "openrouter"}}
    )
    assert out["model"] == "a/b"
    assert out["model_allowlist"] == []


def test_normalize_defaults_to_empty_allowlist_for_string_model():
    out = _normalize_config_for_web({"model": "a/b"})
    assert out["model"] == "a/b"
    assert out["model_allowlist"] == []


def test_normalize_surfaces_allowlist_from_model_dict():
    out = _normalize_config_for_web(
        {"model": {"default": "a/b", "allowlist": ["x/y", " z/w "]}}
    )
    assert out["model"] == "a/b"
    assert out["model_allowlist"] == ["x/y", "z/w"]


# ---------------------------------------------------------------------------
# Denormalization (PUT /api/config write-back)
# ---------------------------------------------------------------------------


def _with_disk_config(monkeypatch, disk_config):
    monkeypatch.setattr(
        "hermes_cli.web_server.load_config", lambda: dict(disk_config)
    )


def test_denormalize_writes_allowlist_into_model_dict(monkeypatch):
    _with_disk_config(
        monkeypatch,
        {"model": {"default": "a/b", "provider": "openrouter", "base_url": "https://x"}},
    )
    out = _denormalize_config_from_web(
        {"model": "a/b", "model_allowlist": ["x/y", "z/w"]}
    )
    model = out["model"]
    assert isinstance(model, dict)
    assert model["allowlist"] == ["x/y", "z/w"]
    # sibling subkeys recovered from disk survive the round-trip
    assert model["provider"] == "openrouter"
    assert model["base_url"] == "https://x"
    # the virtual field never reaches config.yaml
    assert "model_allowlist" not in out


def test_denormalize_empty_allowlist_clears_disk_value(monkeypatch):
    """Present-but-empty means the user cleared the field → key removed (back
    to the unrestricted default)."""
    _with_disk_config(
        monkeypatch, {"model": {"default": "a/b", "allowlist": ["x/y"]}}
    )
    out = _denormalize_config_from_web({"model": "a/b", "model_allowlist": []})
    assert "allowlist" not in out["model"]


def test_denormalize_absent_allowlist_preserves_disk_value(monkeypatch):
    """A payload that omits the field (older frontend) must not clobber a
    hand-written allowlist."""
    _with_disk_config(
        monkeypatch, {"model": {"default": "a/b", "allowlist": ["x/y"]}}
    )
    out = _denormalize_config_from_web({"model": "a/b"})
    assert out["model"]["allowlist"] == ["x/y"]


def test_denormalize_upgrades_bare_string_model(monkeypatch):
    """Disk config with flat ``model: "a/b"`` upgrades to the dict form when
    an allowlist is set — mirroring the context_length upgrade path."""
    _with_disk_config(monkeypatch, {"model": "a/b"})
    out = _denormalize_config_from_web(
        {"model": "a/b", "model_allowlist": ["x/y"]}
    )
    assert out["model"] == {"default": "a/b", "allowlist": ["x/y"]}


def test_normalize_denormalize_roundtrip_is_stable(monkeypatch):
    disk = {"model": {"default": "a/b", "provider": "openrouter", "allowlist": ["x/y"]}}
    _with_disk_config(monkeypatch, disk)
    out = _denormalize_config_from_web(_normalize_config_for_web(dict(disk)))
    assert out["model"]["allowlist"] == ["x/y"]
    assert out["model"]["default"] == "a/b"
    assert out["model"]["provider"] == "openrouter"


# ---------------------------------------------------------------------------
# Endpoint-level round-trip (PUT /api/config)
# ---------------------------------------------------------------------------


class TestAllowlistEndpointRoundTrip:
    """Set AND clear must survive the real endpoint, not just the pure
    denormalize function. update_config deep-merges the denormalized payload
    over the raw disk config — which would silently resurrect a popped
    ``allowlist`` key — so the ``model`` section is replaced wholesale there;
    these tests pin that behavior end-to-end."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

        self.client = TestClient(app)
        self.client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    def _seed(self, allowlist=None):
        from hermes_cli.config import save_config

        model = {"default": "a/b", "provider": "openrouter"}
        if allowlist:
            model["allowlist"] = allowlist
        save_config({"model": model})

    def test_put_sets_allowlist_on_disk(self):
        from hermes_cli.config import load_config

        self._seed()
        web_config = self.client.get("/api/config").json()
        assert web_config["model_allowlist"] == []

        web_config["model_allowlist"] = ["x/y", "z/w"]
        resp = self.client.put("/api/config", json={"config": web_config})
        assert resp.status_code == 200

        after = load_config()
        assert after["model"]["allowlist"] == ["x/y", "z/w"]
        # sibling subkeys survive the write
        assert after["model"]["provider"] == "openrouter"
        assert after["model"]["default"] == "a/b"

    def test_put_clear_removes_allowlist_from_disk(self):
        from hermes_cli.config import load_config

        self._seed(allowlist=["x/y"])
        web_config = self.client.get("/api/config").json()
        assert web_config["model_allowlist"] == ["x/y"]

        web_config["model_allowlist"] = []
        resp = self.client.put("/api/config", json={"config": web_config})
        assert resp.status_code == 200

        after = load_config()
        assert "allowlist" not in (after.get("model") or {}), (
            "clearing the dashboard field must clear the restriction on disk "
            "— the endpoint's deep-merge must not resurrect the popped key"
        )
        # ...without collateral damage to the rest of the model section
        assert after["model"]["default"] == "a/b"
        assert after["model"]["provider"] == "openrouter"
