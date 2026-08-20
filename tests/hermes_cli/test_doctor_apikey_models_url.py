"""Regression: hermes doctor must honor a provider profile's explicit
``models_url`` even when it declares no ``base_url``.

``ProviderProfile.models_url`` is the explicit models endpoint; ``base_url``
only supplies the ``{base_url}/models`` fallback (providers/base.py). In
``_build_apikey_providers_list`` the ``if base_url else None`` guard wrapped the
whole expression, so a profile with ``models_url`` but no ``base_url`` lost its
endpoint (``default_url=None``) and doctor probed ``None`` — a spurious yellow
"couldn't verify" even with a valid key and a good models endpoint.
"""

from __future__ import annotations


def test_build_apikey_providers_list_honors_models_url_without_base_url(monkeypatch):
    from hermes_cli import doctor
    from providers.base import ProviderProfile

    stub = ProviderProfile(
        name="foo-audit-only",          # not a known/dedicated provider name
        display_name="Foo AI",
        env_vars=("FOO_API_KEY",),
        models_url="https://api.foo.ai/v1/models",
        base_url="",                     # explicit models_url, no base_url
        auth_type="api_key",
    )
    monkeypatch.setattr("providers.list_providers", lambda: [stub])

    entries = doctor._build_apikey_providers_list()

    # Tuple shape: (display_name, env_vars, default_url, base_env, supports_health_check)
    foo = [e for e in entries if e[0] == "Foo AI"]
    assert foo, f"stub provider missing from list: {sorted(e[0] for e in entries)}"
    assert foo[0][2] == "https://api.foo.ai/v1/models", (
        "explicit models_url must be kept even when base_url is empty; "
        f"got default_url={foo[0][2]!r}"
    )


def test_build_apikey_providers_list_falls_back_to_base_url(monkeypatch):
    """When only base_url is set, the endpoint is still {base_url}/models."""
    from hermes_cli import doctor
    from providers.base import ProviderProfile

    stub = ProviderProfile(
        name="bar-audit-only",
        display_name="Bar AI",
        env_vars=("BAR_API_KEY",),
        base_url="https://api.bar.ai/v1",
        auth_type="api_key",
    )
    monkeypatch.setattr("providers.list_providers", lambda: [stub])

    entries = doctor._build_apikey_providers_list()
    bar = [e for e in entries if e[0] == "Bar AI"]
    assert bar and bar[0][2] == "https://api.bar.ai/v1/models"
