"""``model_catalog.excluded_providers`` must also hide providers from the provider list served by
``config.get provider`` (``list_available_providers``) and from the unconfigured setup rows of
``model.options include_unconfigured`` (``_append_unconfigured_rows``), matching the ``/model``
and ``hermes model`` pickers. The configured-current provider keeps its row either way.
"""

import pytest


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / ".env").write_text("")
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _write_config(home, excluded=None):
    import yaml

    cfg = {"model": {"provider": "openrouter", "default": "x"}, "custom_providers": []}
    if excluded is not None:
        cfg["model_catalog"] = {"excluded_providers": excluded}
    (home / "config.yaml").write_text(yaml.safe_dump(cfg))


def _available_ids():
    from hermes_cli.models import list_available_providers

    return {row["id"] for row in list_available_providers()}


def _ctx(excluded, current=""):
    from hermes_cli.inventory import ConfigContext

    return ConfigContext(current_provider=current, current_model="", current_base_url="",
                         user_providers={}, custom_providers=[], excluded_providers=excluded)


def _setup_slugs(excluded, current=""):
    from hermes_cli.inventory import _append_unconfigured_rows

    return {row["slug"] for row in _append_unconfigured_rows([], _ctx(excluded, current))}


def test_available_providers_unchanged_without_exclusion(config_home):
    _write_config(config_home)
    ids = _available_ids()
    assert {"opencode-zen", "opencode-go", "custom"} <= ids


def test_available_providers_hide_excluded_slug_and_alias(config_home):
    # "opencode" is an alias of opencode-zen; "opencode-go" is a canonical slug.
    _write_config(config_home, excluded=["OpenCode", "opencode-go"])
    ids = _available_ids()
    assert "opencode-zen" not in ids
    assert "opencode-go" not in ids
    assert {"openrouter", "custom"} <= ids


def test_available_providers_ignore_string_value(config_home):
    # A quoted string is not a list; every other reader ignores it too.
    _write_config(config_home, excluded="opencode-go")
    assert "opencode-go" in _available_ids()


def test_setup_rows_skip_excluded_providers():
    assert {"opencode-zen", "opencode-go"} <= _setup_slugs([])
    hidden = _setup_slugs(["opencode", "go"])
    assert "opencode-zen" not in hidden
    assert "opencode-go" not in hidden
    assert "openrouter" in hidden


def test_setup_rows_keep_configured_current_provider_even_if_excluded():
    rows = _setup_slugs(["opencode-go"], current="opencode-go")
    assert "opencode-go" in rows
