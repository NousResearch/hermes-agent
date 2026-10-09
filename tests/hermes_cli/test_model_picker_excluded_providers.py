"""Tests that ``model_catalog.excluded_providers`` hides providers from the
interactive ``hermes model`` CLI picker.

The CLI picker (``hermes_cli.main.select_provider_and_model``) builds its
provider menu from the live provider catalog via ``group_providers`` — a
separate code path from ``list_authenticated_providers``. These tests
verify the exclusion config is honored there too, matching the
gateway/TUI picker behavior.
"""

from unittest.mock import patch

import pytest


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a minimal config."""
    home = tmp_path / "hermes"
    home.mkdir()
    config_yaml = home / "config.yaml"
    config_yaml.write_text("model: old-model\ncustom_providers: []\n")
    env_file = home / ".env"
    env_file.write_text("")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return home


def _write_config(home, **top_level):
    import hermes_yaml as yaml
    cfg = {"model": "old-model", "custom_providers": []}
    cfg.update(top_level)
    (home / "config.yaml").write_text(yaml.safe_dump(cfg))


def _capture_provider_labels(config_home):
    """Drive ``select_provider_and_model`` and return the provider-menu labels
    shown to the user (the first ``_prompt_provider_choice`` call). Cancels
    immediately after capturing."""
    from hermes_cli.main import select_provider_and_model

    captured: dict = {}

    def _capture_and_cancel(labels, default=0, title=None):
        # Only capture the top-level provider menu (the first call).
        if "labels" not in captured:
            captured["labels"] = list(labels)
        return None  # cancel

    with patch("hermes_cli.main._prompt_provider_choice",
               side_effect=_capture_and_cancel), \
         patch("builtins.print"):
        select_provider_and_model()

    return captured.get("labels", [])


def test_cli_picker_hides_excluded_provider(config_home):
    """``excluded_providers: [openrouter]`` must remove the OpenRouter row
    from the ``hermes model`` provider menu."""
    _write_config(config_home, **{"model_catalog": {"excluded_providers": ["openrouter"]}})

    labels = _capture_provider_labels(config_home)
    assert labels, "provider menu was empty"
    assert not any("OpenRouter" in lbl for lbl in labels), (
        f"OpenRouter should be hidden by excluded_providers, got: {labels}"
    )


def test_cli_picker_hides_excluded_provider_by_alias(config_home):
    """Exclusion by an alias (not the canonical slug) must also hide the
    provider, matching ``list_authenticated_providers``' matching against
    hermes_id / alias names."""
    from application_provider_groups import provider_group_for_slug
    from hermes_cli.provider_catalog import provider_catalog_by_slug
    from providers import list_providers

    # Find a canonical provider that has at least one alias and is a leaf
    # row (not folded into a multi-member group) so its label appears
    # directly. Pick the first such provider.
    target_slug = None
    target_alias = None
    catalog = provider_catalog_by_slug()
    for profile in list_providers():
        aliases = tuple(str(alias or "").strip() for alias in profile.aliases if str(alias or "").strip())
        if profile.name in catalog and aliases and not provider_group_for_slug(profile.name):
            target_slug = profile.name
            target_alias = aliases[0]
            break
    if target_slug is None:
        pytest.skip("no aliased canonical provider available to test")

    target_label_fragment = catalog[target_slug].description

    # Baseline: the provider appears without exclusion.
    _write_config(config_home)
    baseline = _capture_provider_labels(config_home)
    assert any(target_label_fragment in lbl for lbl in baseline), (
        f"sanity: {target_slug} ({target_label_fragment!r}) should appear by "
        f"default; labels={baseline}"
    )

    # Excluding by alias hides it.
    _write_config(
        config_home,
        **{"model_catalog": {"excluded_providers": [target_alias]}},
    )
    excluded_labels = _capture_provider_labels(config_home)
    assert not any(target_label_fragment in lbl for lbl in excluded_labels), (
        f"excluding alias {target_alias!r} should hide {target_slug}; "
        f"labels={excluded_labels}"
    )
