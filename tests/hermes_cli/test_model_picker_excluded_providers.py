"""Tests that ``model_catalog.excluded_providers`` hides providers from the
interactive ``hermes model`` CLI picker.

The CLI picker (``hermes_cli.main.select_provider_and_model``) builds its
provider menu from ``CANONICAL_PROVIDERS`` via ``group_providers`` — a
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

    with patch("hermes_cli.main._prompt_provider_choice",
               side_effect=_capture_and_cancel), \
         patch("builtins.print"):
        select_provider_and_model()

    return captured.get("labels", [])


def test_cli_picker_hides_excluded_provider(config_home):
    """``excluded_providers: [openrouter]`` must remove the OpenRouter row
    from the ``hermes model`` provider menu."""
    _write_config(config_home, model_catalog={"excluded_providers": ["openrouter"]})

    labels = _capture_provider_labels(config_home)
    assert labels, "provider menu was empty"
    assert not any("OpenRouter" in lbl for lbl in labels), (
        f"OpenRouter should be hidden by excluded_providers, got: {labels}"
    )


def test_cli_picker_hides_excluded_provider_by_alias(config_home):
    """Exclusion by an alias (not the canonical slug) must also hide the
    provider, matching ``list_authenticated_providers``' matching against
    hermes_id / alias names."""
    # 'openai' is an alias-style hermes id; ensure excluding it hides the
    # canonical openai provider row if present. Use the canonical slug's
    # alias from _PROVIDER_ALIASES to stay robust to renames.
    from hermes_cli.models import _PROVIDER_ALIASES, CANONICAL_PROVIDERS

    # Find a canonical provider that has at least one alias and is a leaf
    # row (not folded into a multi-member group) so its label appears
    # directly. Pick the first such provider.
    target_slug = None
    target_alias = None
    for alias, canon in _PROVIDER_ALIASES.items():
        if canon and any(p.slug == canon for p in CANONICAL_PROVIDERS):
            target_slug = canon
            target_alias = alias
            break
    if target_slug is None:
        pytest.skip("no aliased canonical provider available to test")

    from hermes_cli.models import _PROVIDER_LABELS
    target_label_fragment = _PROVIDER_LABELS.get(target_slug, target_slug)

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
        model_catalog={"excluded_providers": [target_alias]},
    )
    excluded_labels = _capture_provider_labels(config_home)
    assert not any(target_label_fragment in lbl for lbl in excluded_labels), (
        f"excluding alias {target_alias!r} should hide {target_slug}; "
        f"labels={excluded_labels}"
    )


def test_cli_picker_empty_excluded_is_noop(config_home):
    """An empty ``excluded_providers`` list must not change the menu."""
    _write_config(config_home, model_catalog={"excluded_providers": []})
    excluded_labels = _capture_provider_labels(config_home)

    _write_config(config_home)
    baseline_labels = _capture_provider_labels(config_home)

    assert excluded_labels == baseline_labels


# ─── include_unconfigured (in-session TUI ``/model``) path ────────────────────
# The TUI picker calls ``build_models_payload(include_unconfigured=True)``, which
# appends ``CANONICAL_PROVIDERS`` skeleton rows via ``_append_unconfigured_rows``.
# That loop must honor ``excluded_providers`` too, or excluded providers reappear
# in the TUI ``/model`` picker even though ``hermes model`` hides them (#68816).


def _picker_ctx(excluded=None, *, current_provider="", current_model=""):
    from hermes_cli.inventory import ConfigContext

    return ConfigContext(
        current_provider=current_provider,
        current_model=current_model,
        current_base_url="",
        user_providers={},
        custom_providers=[],
        excluded_providers=excluded,
    )


def test_unconfigured_rows_hide_excluded_provider():
    from hermes_cli.inventory import _append_unconfigured_rows

    baseline = {r["slug"].lower() for r in _append_unconfigured_rows([], _picker_ctx())}
    assert "openrouter" in baseline, "sanity: openrouter should be a canonical skeleton row"

    slugs = {
        r["slug"].lower()
        for r in _append_unconfigured_rows([], _picker_ctx(excluded=["openrouter"]))
    }
    assert "openrouter" not in slugs, "excluded provider must not be re-added as a skeleton row"
    assert slugs == baseline - {"openrouter"}, "only the excluded provider should be removed"


def test_unconfigured_rows_exclusion_is_case_insensitive():
    from hermes_cli.inventory import _append_unconfigured_rows

    slugs = {
        r["slug"].lower()
        for r in _append_unconfigured_rows([], _picker_ctx(excluded=["OpenRouter"]))
    }
    assert "openrouter" not in slugs


def test_unconfigured_rows_hide_excluded_provider_by_alias():
    """Excluding by an *alias* (not the canonical slug) must also drop the
    canonical skeleton row, matching ``hermes model`` and
    ``list_authenticated_providers``. Comparing the raw exclusion strings
    against ``entry.slug`` alone would leak the canonical row back in."""
    from hermes_cli.inventory import _append_unconfigured_rows
    from hermes_cli.models import CANONICAL_PROVIDERS, _PROVIDER_ALIASES

    # Pick an alias whose canonical target is an actual skeleton row.
    baseline = {r["slug"].lower() for r in _append_unconfigured_rows([], _picker_ctx())}
    target_slug = None
    target_alias = None
    for alias, canon in _PROVIDER_ALIASES.items():
        if canon and canon.lower() in baseline and any(p.slug == canon for p in CANONICAL_PROVIDERS):
            target_slug = canon
            target_alias = alias
            break
    if target_slug is None:
        pytest.skip("no aliased canonical provider present as an unconfigured row")

    slugs = {
        r["slug"].lower()
        for r in _append_unconfigured_rows([], _picker_ctx(excluded=[target_alias]))
    }
    assert target_slug.lower() not in slugs, (
        f"excluding alias {target_alias!r} should hide canonical {target_slug!r}; got {slugs}"
    )


def test_unconfigured_rows_exclude_current_provider_matches_cli():
    """``list_authenticated_providers`` drops an excluded provider even when it is
    the current one; the skeleton loop must not re-surface it as the
    ``configured-current`` warning row."""
    from hermes_cli.inventory import _append_unconfigured_rows

    rows = _append_unconfigured_rows(
        [], _picker_ctx(excluded=["openrouter"], current_provider="openrouter", current_model="some-model")
    )
    assert "openrouter" not in {r["slug"].lower() for r in rows}


def test_unconfigured_rows_empty_excluded_is_noop():
    from hermes_cli.inventory import _append_unconfigured_rows

    base = {r["slug"].lower() for r in _append_unconfigured_rows([], _picker_ctx())}
    empty = {r["slug"].lower() for r in _append_unconfigured_rows([], _picker_ctx(excluded=[]))}
    none = {r["slug"].lower() for r in _append_unconfigured_rows([], _picker_ctx(excluded=None))}
    assert base == empty == none


# ─── user-defined providers: / custom_providers: shapes (#142961) ─────────────
# ``providers:`` entries (section 3) never consulted ``excluded_providers`` and
# ``custom_providers:`` entries (section 4) surface as ``custom:<name>``, which
# never string-matches a bare config entry like ``back1``. Both shapes leaked
# into ``/model`` even when explicitly excluded.


@pytest.fixture
def _offline_picker(monkeypatch):
    """No network / no built-in catalog: only the user-defined rows below."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda *a, **k: {})
    monkeypatch.setattr("agent.models_dev.PROVIDER_TO_MODELS_DEV", {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.models.cached_provider_model_ids", lambda *a, **k: [])
    monkeypatch.setattr("hermes_cli.models.provider_model_ids", lambda *a, **k: [])


def _user_defined_slugs(excluded):
    from hermes_cli.model_switch_providers import list_authenticated_providers

    rows = list_authenticated_providers(
        current_provider="com",
        current_model="deepseek-flash",
        user_providers={
            "com": {"base_url": "http://127.0.0.1:56673/v1", "model": "deepseek-flash",
                    "discover_models": False},
            "back1": {"base_url": "http://127.0.0.1:52718/v1", "model": "some-model",
                      "discover_models": False},
            "back2": {"base_url": "https://example.internal/v1", "model": "other-model",
                      "discover_models": False},
        },
        custom_providers=[
            {"name": "back3", "base_url": "http://127.0.0.1:52719/v1",
             "model": "third-model", "discover_models": False},
        ],
        excluded_providers=excluded,
        probe_custom_providers=False,
    )
    return {str(r["slug"]).lower() for r in rows if r.get("is_user_defined")}


def test_user_defined_providers_baseline_listed(_offline_picker):
    slugs = _user_defined_slugs(None)
    assert {"com", "back1", "back2", "custom:back3"} <= slugs, (
        f"sanity: user-defined rows should list without exclusion; got {slugs}"
    )


def test_excluded_hides_providers_dict_entries(_offline_picker):
    slugs = _user_defined_slugs(["back1", "back2"])
    assert "back1" not in slugs and "back2" not in slugs, (
        f"excluded providers: entries must be hidden; got {slugs}"
    )
    assert "com" in slugs and "custom:back3" in slugs, (
        f"non-excluded rows must survive; got {slugs}"
    )


def test_excluded_bare_name_hides_custom_prefixed_row(_offline_picker):
    slugs = _user_defined_slugs(["back3"])
    assert "custom:back3" not in slugs, (
        f"bare 'back3' must hide the custom:back3 row; got {slugs}"
    )


def test_excluded_custom_spelling_and_case_insensitive(_offline_picker):
    slugs = _user_defined_slugs(["custom:back1", "BACK2"])
    assert "back1" not in slugs and "back2" not in slugs, (
        f"'custom:back1' / 'BACK2' spellings must also hide rows; got {slugs}"
    )
    assert _user_defined_slugs([]) == _user_defined_slugs(None), (
        "empty excluded_providers must be a no-op"
    )
