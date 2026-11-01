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


# ─── authed picker rows (list_authenticated_providers) ────────────────────────
# The skeleton loop and the CLI picker expand ``_PROVIDER_ALIASES`` before
# filtering, but the authed-row filters in ``model_switch_providers`` raw-matched
# the normalized entries only: excluding the alias ``moonshot`` left the
# authenticated ``kimi-coding`` row visible on the gateway/TUI/Desktop pickers
# while ``hermes model`` hid it. Alias parity for that path (#68816 contract,
# rebased from #94362).


def test_expanded_excluded_set_is_alias_aware():
    """An alias exclusion expands to its canonical slug (and siblings), a
    canonical exclusion expands to every alias that surfaces it."""
    from hermes_cli.model_switch_providers import _expanded_excluded_provider_set

    by_alias = _expanded_excluded_provider_set(["moonshot"])
    assert "moonshot" in by_alias
    assert "kimi-coding" in by_alias, "alias exclusion must hide the canonical slug"

    by_canonical = _expanded_excluded_provider_set(["kimi-coding"])
    assert "kimi-coding" in by_canonical
    assert "moonshot" in by_canonical and "kimi" in by_canonical, (
        "canonical exclusion must cover every key the provider surfaces as"
    )


def test_expanded_excluded_set_normalizes_case_and_whitespace():
    from hermes_cli.model_switch_providers import _expanded_excluded_provider_set

    for entry in ("Moonshot", "  moonshot  ", "  KIMI-CODING ", "MOONSHOT"):
        got = _expanded_excluded_provider_set([entry])
        assert "kimi-coding" in got, f"{entry!r} should hide kimi-coding; got {got}"


def test_expanded_excluded_set_empty_and_none_are_noop():
    from hermes_cli.model_switch_providers import _expanded_excluded_provider_set

    assert _expanded_excluded_provider_set([]) == set()
    assert _expanded_excluded_provider_set(None) == set()
    assert _expanded_excluded_provider_set(["", "  ", None]) == set()


def test_expanded_excluded_set_leaves_unrelated_providers_alone():
    from hermes_cli.model_switch_providers import _expanded_excluded_provider_set

    got = _expanded_excluded_provider_set(["moonshot"])
    assert "openrouter" not in got
    assert "kimi-coding-cn" not in got, "the CN sibling is a distinct provider"


def _stub_kimi_authed_discovery(monkeypatch):
    """Isolate ``list_authenticated_providers`` to the Kimi family, fully offline:
    stubbed models.dev map/catalog/overlays/canonical list plus a canned model-id
    fetch, so no network, no credentials and no user config are involved."""
    import agent.models_dev as md
    import hermes_cli.models as hm
    import hermes_cli.models_catalog_static as csm
    from hermes_cli import models_catalog_static

    monkeypatch.setattr(md, "PROVIDER_TO_MODELS_DEV", {
        "kimi": "kimi-for-coding",
        "kimi-coding": "kimi-for-coding",
        "moonshot": "kimi-for-coding",
        "kimi-coding-cn": "kimi-for-coding",
    })
    monkeypatch.setattr(md, "fetch_models_dev", lambda *a, **k: {
        "kimi-for-coding": {"name": "Kimi For Coding", "env": ["KIMI_API_KEY"]},
    })

    from agent.models_dev import ProviderInfo

    monkeypatch.setattr(md, "get_provider_info", lambda _pid: ProviderInfo(
        id="kimi-for-coding", name="Kimi For Coding", env=("KIMI_API_KEY",), api=""))
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
    canonical = [
        models_catalog_static.ProviderEntry("kimi-coding", "Kimi / Kimi Coding Plan", "desc"),
        models_catalog_static.ProviderEntry("kimi-coding-cn", "Kimi / Moonshot (China)", "desc"),
    ]
    monkeypatch.setattr(hm, "CANONICAL_PROVIDERS", canonical)
    monkeypatch.setattr(csm, "CANONICAL_PROVIDERS", canonical)
    monkeypatch.setattr(hm, "cached_provider_model_ids", lambda *a, **k: ["kimi-k2.6"])
    monkeypatch.setattr(hm, "clear_provider_models_cache", lambda *a, **k: None)
    monkeypatch.setenv("KIMI_API_KEY", "sk-test-kimi")
    # Separate credential so the CN row is not dedup-collapsed into the global one.
    monkeypatch.setenv("KIMI_CN_API_KEY", "sk-test-kimi-cn")


def _authed_slugs(monkeypatch, excluded, **kw):
    from hermes_cli import model_switch

    rows = model_switch.list_authenticated_providers(
        max_models=10, excluded_providers=excluded, **kw)
    return {str(r.get("slug", "")).lower() for r in rows}


def test_authed_row_hidden_by_alias_exclusion(monkeypatch):
    """The #68816 contract on the authed path: ``excluded_providers: [moonshot]``
    must hide the authenticated ``kimi-coding`` row — it used to leak through
    because the row filters never expanded aliases."""
    _stub_kimi_authed_discovery(monkeypatch)

    baseline = _authed_slugs(monkeypatch, [])
    assert "kimi-coding" in baseline, f"sanity: kimi-coding should list; got {baseline}"

    assert "kimi-coding" not in _authed_slugs(monkeypatch, ["moonshot"])
    assert "kimi-coding" not in _authed_slugs(monkeypatch, ["Moonshot"])
    assert "kimi-coding" not in _authed_slugs(monkeypatch, ["  moonshot  "])
    assert "kimi-coding" not in _authed_slugs(monkeypatch, ["kimi-coding"])


def test_authed_alias_exclusion_keeps_unrelated_rows(monkeypatch):
    """Excluding the global Kimi family must not touch the distinct CN provider."""
    _stub_kimi_authed_discovery(monkeypatch)

    baseline = _authed_slugs(monkeypatch, [])
    assert "kimi-coding-cn" in baseline, f"sanity: CN row should list; got {baseline}"

    slugs = _authed_slugs(monkeypatch, ["moonshot"])
    assert "kimi-coding" not in slugs
    assert "kimi-coding-cn" in slugs, "unrelated providers must remain visible"
    assert slugs == baseline - {"kimi-coding"}, "only the excluded provider should be removed"


def test_authed_alias_exclusion_empty_is_noop(monkeypatch):
    _stub_kimi_authed_discovery(monkeypatch)

    baseline = _authed_slugs(monkeypatch, [])
    assert _authed_slugs(monkeypatch, []) == baseline
    assert _authed_slugs(monkeypatch, None) == baseline
    assert _authed_slugs(monkeypatch, []) == baseline


def test_authed_alias_excluded_current_provider_stays_hidden(monkeypatch):
    """An excluded provider stays excluded even when it is the current provider
    (the existing picker behavior — expansion must not reintroduce it)."""
    _stub_kimi_authed_discovery(monkeypatch)

    baseline = _authed_slugs(monkeypatch, [], current_provider="kimi-coding")
    assert "kimi-coding" in baseline, f"sanity: current provider should list; got {baseline}"

    slugs = _authed_slugs(monkeypatch, ["moonshot"], current_provider="kimi-coding")
    assert "kimi-coding" not in slugs


def test_payload_alias_exclusion_hides_row_for_desktop(monkeypatch):
    """End-to-end over the Desktop path: ``build_models_payload`` with
    ``include_unconfigured=True`` (what ``GET /api/model/options`` serves) must
    emit no ``kimi-coding`` row when the exclusion names its alias — neither the
    authed row nor a canonical skeleton may resurrect it."""
    _stub_kimi_authed_discovery(monkeypatch)
    from hermes_cli.inventory import build_models_payload

    def _payload(excluded):
        p = build_models_payload(
            _picker_ctx(excluded=excluded), include_unconfigured=True,
            probe_custom_providers=False, non_blocking_catalogs=True)
        return {str(r.get("slug", "")).lower() for r in p["providers"]}

    baseline = _payload([])
    assert "kimi-coding" in baseline, f"sanity: kimi-coding should be in payload; got {baseline}"

    slugs = _payload(["moonshot"])
    assert "kimi-coding" not in slugs, "alias-excluded provider leaked into the payload"
    assert "kimi-coding-cn" in slugs, "unrelated providers must remain in the payload"
