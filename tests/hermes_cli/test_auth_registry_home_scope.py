"""Auth registry rows and provider aliases belong to the profile home that declares them.

One process serves many profile homes. The auth registry and the catalog alias table are
process-wide, so a row rewritten for one home (endpoint, credential env names) or an alias claimed
first by one home leaked into every other home served by the same process.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _plugin(home: Path, name: str, url: str, env: tuple, aliases: tuple) -> None:
    plugin = home / "plugins" / "model-providers" / name
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text(textwrap.dedent(f"""
        from providers import register_provider
        from providers.base import ProviderProfile

        register_provider(ProviderProfile(
            name={name!r}, auth_type="api_key", env_vars={env!r}, base_url={url!r},
            aliases={aliases!r}, catalog_aliases=True))
    """), encoding="utf-8")


@pytest.fixture
def homes(tmp_path, monkeypatch):
    import providers
    from hermes_cli import auth_plugin_providers as app
    from hermes_cli.auth import PROVIDER_REGISTRY

    a, b, c = (tmp_path / n for n in "abc")
    _plugin(a, "acme-scope", "https://a.invalid/v1", ("ACME_SCOPE_A_KEY",), ("acs",))
    _plugin(a, "deepseek", "https://deepseek-a.invalid/v1", ("DEEPSEEK_API_KEY",), ())
    _plugin(b, "acme-scope", "https://b.invalid/v1", ("ACME_SCOPE_B_KEY",), ())
    _plugin(b, "zeta-scope", "https://z.invalid/v1", ("ZETA_SCOPE_KEY",), ("acs",))
    c.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(c))
    monkeypatch.setattr(providers, "_REGISTRY", dict(providers._REGISTRY))
    monkeypatch.setattr(providers, "_ALIASES", dict(providers._ALIASES))
    monkeypatch.setattr(providers, "_PROVIDER_LIST_CACHE", None)
    monkeypatch.setattr(providers, "_HOME_LAYERS", {})
    monkeypatch.setattr(app, "PLUGIN_MIRRORED_PROVIDERS", set(app.PLUGIN_MIRRORED_PROVIDERS))
    rows = list(dict.items(PROVIDER_REGISTRY))
    yield a, b, c
    PROVIDER_REGISTRY.clear()
    PROVIDER_REGISTRY.update(rows)
    for mod in [m for m in sys.modules if m.startswith("_hermes_user_provider")]:
        del sys.modules[mod]


def _view(home: Path):
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.auth_plugin_providers import registry_lookup
    from hermes_cli.models import normalize_provider

    token = set_hermes_home_override(home)
    try:
        acme = registry_lookup("acme-scope")
        alias = PROVIDER_REGISTRY.get("acs")
        return ((acme.inference_base_url, tuple(acme.api_key_env_vars)) if acme else None,
                alias.id if alias else None, normalize_provider("acs"),
                PROVIDER_REGISTRY["deepseek"].inference_base_url)
    finally:
        reset_hermes_home_override(token)


def test_each_home_reads_its_own_provider_rows_and_aliases(homes):
    a, b, c = homes
    builtin_deepseek = _view(c)[3]
    home_a = (("https://a.invalid/v1", ("ACME_SCOPE_A_KEY",)), "acme-scope", "acme-scope",
              "https://deepseek-a.invalid/v1")
    assert _view(a) == home_a
    assert _view(b) == (("https://b.invalid/v1", ("ACME_SCOPE_B_KEY",)), "zeta-scope", "zeta-scope",
                        builtin_deepseek)
    assert _view(a) == home_a
    assert _view(c) == (None, None, "acs", builtin_deepseek)
