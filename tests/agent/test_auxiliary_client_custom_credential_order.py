"""Credential order for explicit custom endpoints (regression for #145452).

A route's own credential — the same-origin main key or the ``providers.<id>`` entry owning the
URL — must beat an ambient ``OPENAI_API_KEY``, which stays only the documented last resort for a
URL with no credential of its own, and then only on an OpenAI host or the exact ``OPENAI_BASE_URL``
it is paired with (#91308). Real config.yaml under a temp ``HERMES_HOME``, resolved through the
task route; HTTP is faked.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import hermes_yaml as yaml

from agent.auxiliary_client import resolve_provider_client


def test_main_credential_beats_ambient_openai_key_on_the_same_origin(tmp_path, monkeypatch):
    """A live OPENAI_API_KEY is not the route's credential. The main endpoint's key for the
    same origin wins, so a self-hosted proxy never receives an OpenAI key (regression for
    #145452)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    base_url = "https://proxy.example.test/v1"
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "custom", "base_url": base_url,
                  "api_key": "sk-main-route-key", "default": "main-model"},
    }))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-openai")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    captured: dict = {}

    def _capture_create(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    import agent.auxiliary_client as ac

    with patch.object(ac, "_create_openai_client", side_effect=_capture_create):
        resolve_provider_client(
            "custom", model="main-model", explicit_base_url=base_url, explicit_api_key=None,
        )

    assert captured.get("api_key") == "sk-main-route-key", (
        "the route's own same-origin credential must beat the ambient OPENAI_API_KEY, got: "
        + repr(captured.get("api_key"))
    )


def test_declared_provider_key_env_beats_ambient_openai_key(tmp_path, monkeypatch):
    """A ``providers.<id>`` block owning the explicit base_url supplies the credential even
    when the main model lives at another origin and OPENAI_API_KEY is exported. The OpenAI
    key must never be sent to the third-party endpoint (regression for #145452)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    base_url = "https://proxy.example.test/v1"
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "litellm", "base_url": base_url, "default": "my-combo"},
        "providers": {
            "litellm": {
                "name": "my litellm", "base_url": base_url, "key_env": "ROUTE_KEY",
                "models": {"my-combo": {"context_length": 500000}},
            },
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("ROUTE_KEY", "sk-route-credential")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-openai")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    captured: dict = {}

    def _capture_create(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    import agent.auxiliary_client as ac

    with patch.object(ac, "_create_openai_client", side_effect=_capture_create):
        resolve_provider_client(
            "custom", model="my-combo", explicit_base_url=base_url, explicit_api_key=None,
        )

    assert captured.get("api_key") == "sk-route-credential"


@pytest.mark.parametrize(("paired_base", "expected"), [
    ("https://other.example.net/v1", "ambient-openai-key"),
    (None, "no-key-required"),
])
def test_route_without_credential_falls_back_to_openai_key_never_the_main_key(
    tmp_path, monkeypatch, paired_base, expected,
):
    """A cross-origin base_url with no credential of its own gets the documented
    OPENAI_API_KEY last resort only when paired with it via OPENAI_BASE_URL (#91308) —
    never the main route's key (regression for #145452)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "custom", "base_url": "https://main.example.test/v1",
                  "api_key": "main-route-key", "default": "main-model"},
    }))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-openai-key")
    if paired_base:
        monkeypatch.setenv("OPENAI_BASE_URL", paired_base)
    else:
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    captured: dict = {}

    def _capture_create(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    import agent.auxiliary_client as ac

    with patch.object(ac, "_create_openai_client", side_effect=_capture_create):
        resolve_provider_client(
            "custom", model="other-model",
            explicit_base_url="https://other.example.net/v1", explicit_api_key=None,
        )

    assert captured.get("api_key") == expected


def _declared_entry_home(tmp_path, monkeypatch, entry):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "openrouter", "default": "x/y"},
        "providers": {"litellm": {"name": "my litellm", "base_url": "https://proxy.example.test/v1", **entry}},
    }))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)


def test_cached_client_follows_a_rotated_declared_key(tmp_path, monkeypatch):
    """The declared entry's key joins the client cache key like the borrowed main key does: once
    its key_env value rotates, the client built with the old key must not keep serving."""
    import agent.auxiliary_client as ac

    _declared_entry_home(tmp_path, monkeypatch, {"key_env": "ROUTE_KEY"})
    monkeypatch.setenv("ROUTE_KEY", "route-key-v1")
    first, _ = ac._get_cached_client("custom", model="m", base_url="https://proxy.example.test/v1")
    monkeypatch.setenv("ROUTE_KEY", "route-key-v2")
    second, _ = ac._get_cached_client("custom", model="m", base_url="https://proxy.example.test/v1")

    assert (first.api_key, second.api_key) == ("route-key-v1", "route-key-v2")


def test_cached_client_is_reused_for_a_declared_key_cmd(tmp_path, monkeypatch):
    """A key_cmd entry yields a fresh token source per lookup; the cache keys on its stable
    command identity, so repeated calls reuse one client instead of building one per call."""
    import sys
    import agent.auxiliary_client as ac

    helper = tmp_path / "mint.py"
    helper.write_text("print('minted-token')\n", encoding="utf-8")
    _declared_entry_home(tmp_path, monkeypatch, {"key_cmd": f'"{sys.executable}" "{helper}"'})
    first, _ = ac._get_cached_client("custom", model="m", base_url="https://proxy.example.test/v1")
    second, _ = ac._get_cached_client("custom", model="m", base_url="https://proxy.example.test/v1")
    (tmp_path / ".hermes" / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "openrouter", "default": "x/y"},
        "providers": {"litellm": {"name": "my litellm", "base_url": "https://proxy.example.test/v1",
                                  "key_cmd": f'"{sys.executable}" "{helper}" --rotated'}},
    }))
    third, _ = ac._get_cached_client("custom", model="m", base_url="https://proxy.example.test/v1")

    assert first is not None and first is second
    assert third is not first  # an edited key_cmd builds a client around the new helper


def test_cached_client_lookup_never_selects_from_the_entry_pool(tmp_path, monkeypatch):
    """With key_env unset the entry's key comes from its credential pool. Only the client build
    selects from it; warm cache lookups must not advance rotation or re-select."""
    import agent.auxiliary_client as ac

    _declared_entry_home(tmp_path, monkeypatch, {"key_env": "UNSET_ROUTE_KEY"})
    monkeypatch.delenv("UNSET_ROUTE_KEY", raising=False)
    selects = []

    class _Pool:
        provider = "custom:my-litellm"

        def entries(self):  # an `hermes auth add` row the entry owns, bound to its origin (#130410)
            return [SimpleNamespace(source="manual", base_url="https://proxy.example.test/v1")]

        def has_credentials(self):
            return True

        def select(self, model=None):
            selects.append(model)
            return SimpleNamespace(runtime_api_key=f"pooled-{len(selects)}")

    monkeypatch.setattr(ac, "load_pool", lambda key: _Pool())
    clients = [ac._get_cached_client("custom", model="m", base_url="https://proxy.example.test/v1")[0] for _ in range(3)]

    assert clients[0].api_key == "pooled-1"
    assert clients[1] is clients[0] and clients[2] is clients[0]
    assert len(selects) == 1


def test_dual_surface_anthropic_entry_gets_its_own_key(tmp_path, monkeypatch):
    """An entry declaring an ``/anthropic`` URL owns that URL: the plain client's base is rewritten to
    ``/v1``, but ownership is looked up on the URL the route names, so the entry's key is sent."""
    import agent.auxiliary_client as ac

    url = "https://api.minimax.io/anthropic"
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "openrouter", "default": "x/y"},
        "providers": {"mm-plan": {"name": "mm plan", "base_url": url, "key_env": "MM_KEY"}},
    }))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("MM_KEY", "mm-route-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    captured: dict = {}

    def _capture_create(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    with patch.object(ac, "_create_openai_client", side_effect=_capture_create):
        resolve_provider_client("custom", model="m", explicit_base_url=url, explicit_api_key=None)
    assert captured.get("api_key") == "mm-route-key"

    monkeypatch.setenv("MM_KEY", "mm-route-key-v1")
    first, _ = ac._get_cached_client("custom", model="m", base_url=url)
    monkeypatch.setenv("MM_KEY", "mm-route-key-v2")
    second, _ = ac._get_cached_client("custom", model="m", base_url=url)
    assert first is not second  # the cache keys on the same raw-URL owner the build reads
