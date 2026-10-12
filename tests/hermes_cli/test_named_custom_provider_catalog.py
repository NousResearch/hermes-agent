"""A named custom provider's model catalog must be discoverable.

A *named* custom provider is ``providers.<name>`` in config.yaml with
``model.provider: custom:<name>``. It is a first-class, keyed entry — the
schema the setup flow writes and the one the docs point users at.

``provider_model_ids("custom:<name>")`` returned ``[]`` for such a provider:

1. The slug is not in ``_PROVIDER_CATALOG_FETCHERS`` (only the bare
   ``"custom"`` key is), so no fetcher matched and discovery fell through to
   an empty static catalog.
2. When a ``model.base_url`` is configured, the relay branch intercepted
   first and resolved the credential via ``_api_key_credentials(slug)``,
   which returns ``""`` for a named route — the probe ran unauthenticated
   and returned nothing.

The user-visible symptom is the worst kind: the model picker reports an
empty catalog, so the provider looks *deleted* after an update even though
its config is intact and its endpoint answers ``/v1/models``.

These tests pin the contract: a named custom provider discovers its catalog
from the same credential source runtime inference uses
(``providers.<name>.api_key`` / ``key_env``).
"""

from unittest.mock import patch

import pytest


@pytest.fixture
def named_custom_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a named custom provider and a relay base_url."""
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n"
        "  default: tier-a\n"
        "  provider: custom:acme\n"
        "  base_url: https://api.acme.test/v1\n"
        "  context_length: 1000000\n"
        "providers:\n"
        "  acme:\n"
        "    base_url: https://api.acme.test/v1\n"
        "    api_key: sk-acme-secret\n"
        "    default_model: tier-a\n"
        "    models:\n"
        "      - tier-a\n"
        "      - tier-b\n"
    )
    (home / ".env").write_text("")
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Discovery must not pick up an ambient credential from the developer's shell.
    for var in ("CUSTOM_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    return home


def _reload_models_module():
    """Re-import hermes_cli.models so config reads hit the fixture's HERMES_HOME."""
    import importlib

    from hermes_cli import models

    return importlib.reload(models)


class TestNamedCustomProviderCatalog:
    def test_named_custom_provider_resolves_its_catalog(self, named_custom_home):
        """The endpoint's live catalog is returned, not an empty list."""
        models = _reload_models_module()
        seen = {}

        def fake_fetch(api_key, base_url, api_mode=None):
            seen["api_key"] = api_key
            seen["base_url"] = base_url
            return ["tier-a", "tier-b"]

        with patch.object(models, "fetch_api_models", fake_fetch):
            result = models.provider_model_ids("custom:acme", force_refresh=True)

        assert result == ["tier-a", "tier-b"]
        assert seen["base_url"] == "https://api.acme.test/v1"

    def test_named_custom_provider_sends_its_configured_credential(
        self, named_custom_home
    ):
        """The probe authenticates with providers.<name>.api_key.

        ``_api_key_credentials("custom:acme")`` is empty, so a discovery path
        that only consults it probes unauthenticated and the endpoint answers
        401 — indistinguishable from "provider has no models".
        """
        models = _reload_models_module()
        seen = {}

        def fake_fetch(api_key, base_url, api_mode=None):
            seen["api_key"] = api_key
            return ["tier-a"]

        with patch.object(models, "fetch_api_models", fake_fetch):
            models.provider_model_ids("custom:acme", force_refresh=True)

        assert seen["api_key"] == "sk-acme-secret"

    def test_named_custom_provider_honours_key_env(self, tmp_path, monkeypatch):
        """A ``key_env`` entry resolves through the environment."""
        home = tmp_path / "hermes"
        home.mkdir()
        (home / "config.yaml").write_text(
            "model:\n"
            "  default: tier-a\n"
            "  provider: custom:acme\n"
            "  base_url: https://api.acme.test/v1\n"
            "providers:\n"
            "  acme:\n"
            "    base_url: https://api.acme.test/v1\n"
            "    key_env: ACME_TEST_KEY\n"
        )
        (home / ".env").write_text("")
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("ACME_TEST_KEY", "sk-from-env")
        for var in ("CUSTOM_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
            monkeypatch.delenv(var, raising=False)

        models = _reload_models_module()
        seen = {}

        def fake_fetch(api_key, base_url, api_mode=None):
            seen["api_key"] = api_key
            return ["tier-a"]

        with patch.object(models, "fetch_api_models", fake_fetch):
            models.provider_model_ids("custom:acme", force_refresh=True)

        assert seen["api_key"] == "sk-from-env"

    def test_anonymous_custom_still_resolves(self, tmp_path, monkeypatch):
        """The bare ``custom`` provider keeps working (no regression)."""
        home = tmp_path / "hermes"
        home.mkdir()
        (home / "config.yaml").write_text(
            "model:\n"
            "  default: local-model\n"
            "  provider: custom\n"
            "  base_url: http://127.0.0.1:8081/v1\n"
            "  api_key: sk-local\n"
        )
        (home / ".env").write_text("")
        monkeypatch.setenv("HERMES_HOME", str(home))
        for var in ("CUSTOM_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
            monkeypatch.delenv(var, raising=False)

        models = _reload_models_module()

        with patch.object(
            models, "fetch_api_models", lambda *a, **k: ["local-model"]
        ):
            assert models.provider_model_ids("custom", force_refresh=True) == [
                "local-model"
            ]

    def test_named_custom_provider_without_a_catalog_is_empty_not_an_error(
        self, named_custom_home
    ):
        """A failing probe degrades to an empty list, never an exception."""
        models = _reload_models_module()

        with patch.object(models, "fetch_api_models", lambda *a, **k: None):
            assert models.provider_model_ids("custom:acme", force_refresh=True) == []

    def test_named_provider_probes_its_own_endpoint_with_its_own_key(self, tmp_path, monkeypatch):
        """``custom:beta`` never sends beta's key to the active ``model.base_url`` host, and an
        ambient ``OPENAI_API_KEY`` never outranks the entry's own credential."""
        home = tmp_path / "hermes"
        home.mkdir()
        (home / "config.yaml").write_text(
            "model:\n"
            "  default: tier-a\n"
            "  provider: custom:acme\n"
            "  base_url: https://api.acme.test/v1\n"
            "providers:\n"
            "  acme:\n"
            "    base_url: https://api.acme.test/v1\n"
            "    api_key: sk-acme-secret\n"
            "  beta:\n"
            "    base_url: https://api.beta.test/v1\n"
            "    api_key: sk-beta-secret\n"
        )
        (home / ".env").write_text("")
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("OPENAI_API_KEY", "sk-ambient")
        models = _reload_models_module()
        probes = []

        def fake_fetch(api_key, base_url, api_mode=None):
            probes.append((base_url, api_key))
            return ["m"]

        with patch.object(models, "fetch_api_models", fake_fetch):
            models.provider_model_ids("custom:acme", force_refresh=True)
            models.provider_model_ids("custom:beta", force_refresh=True)

        assert probes == [
            ("https://api.acme.test/v1", "sk-acme-secret"),
            ("https://api.beta.test/v1", "sk-beta-secret"),
        ], probes
