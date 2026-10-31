"""Tests for the MiniMax provider profile's live model listing.

MiniMax's anthropic_messages routes serve the model catalog at
``{base_url}/v1/models`` and authenticate with the ``x-api-key`` header. The
base ProviderProfile requests ``{base_url}/models`` with an ``Authorization``
header, which 404s on those routes — so the picker silently fell back to the
static catalog and a newly released model never appeared.

Run: pytest tests/providers/test_minimax_fetch_models.py -q
"""

import json
import sys

import providers

# Discovery imports the bundled provider plugins; that import is what registers
# the ``plugins.model_providers.<name>`` module under its stable alias.
providers._discover_providers()

_minimax = sys.modules["plugins.model_providers.minimax"]
MiniMaxProfile = _minimax.MiniMaxProfile
_minimax_models_url = _minimax._minimax_models_url


def test_anthropic_base_url_uses_v1_models_path():
    assert _minimax_models_url("https://api.minimaxi.com/anthropic") == "https://api.minimaxi.com/anthropic/v1/models"


def test_openai_base_url_uses_models_path():
    assert _minimax_models_url("https://api.minimax.io/v1") == "https://api.minimax.io/v1/models"


def test_trailing_slash_is_normalised():
    assert _minimax_models_url("https://api.minimaxi.com/anthropic/") == "https://api.minimaxi.com/anthropic/v1/models"


def test_empty_base_url_yields_empty_string():
    assert _minimax_models_url("") == ""
    assert _minimax_models_url(None) == ""


def test_fetch_models_without_api_key_returns_none():
    profile = MiniMaxProfile(name="minimax", base_url="https://api.minimaxi.com/anthropic")
    assert profile.fetch_models(api_key=None) is None


def test_fetch_models_parses_catalog(monkeypatch):
    """A successful response yields the model ids, de-duplicated and in order."""
    payload = {
        "data": [
            {"id": "MiniMax-M3.1-Flash"},
            {"id": "MiniMax-M3"},
            {"id": "MiniMax-M3.1-Flash"},
            {"not_an_id": "x"},
        ]
    }

    class _Resp:
        def read(self):
            return json.dumps(payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    captured = {}

    def _fake_open(req, *, timeout, **kwargs):
        captured["url"] = req.full_url
        captured["headers"] = {k.lower(): v for k, v in req.header_items()}
        return _Resp()

    monkeypatch.setattr(_minimax, "open_credentialed_url", _fake_open)

    profile = MiniMaxProfile(name="minimax", base_url="https://api.minimaxi.com/anthropic")
    models = profile.fetch_models(api_key="secret-key", base_url="https://api.minimaxi.com/anthropic")

    assert models == ["MiniMax-M3.1-Flash", "MiniMax-M3"]
    assert captured["url"] == "https://api.minimaxi.com/anthropic/v1/models"
    # MiniMax authenticates with x-api-key, not an Authorization header.
    assert captured["headers"].get("x-api-key") == "secret-key"
    assert "authorization" not in captured["headers"]


def test_fetch_models_returns_none_on_error(monkeypatch):
    """A failed request returns None so the caller falls back to the static list."""

    def _boom(req, *, timeout, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(_minimax, "open_credentialed_url", _boom)

    profile = MiniMaxProfile(name="minimax", base_url="https://api.minimaxi.com/anthropic")
    assert profile.fetch_models(api_key="secret-key") is None


def test_fetch_models_returns_none_on_empty_catalog(monkeypatch):
    """An empty catalog is treated as a failed fetch (falls back to static)."""

    class _Resp:
        def read(self):
            return json.dumps({"data": []}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(_minimax, "open_credentialed_url", lambda req, **kw: _Resp())

    profile = MiniMaxProfile(name="minimax", base_url="https://api.minimaxi.com/anthropic")
    assert profile.fetch_models(api_key="secret-key") is None
