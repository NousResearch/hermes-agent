"""Tests for ``_inject_aliases_into_rows`` — surfacing ``model_aliases:`` in
the picker.

Model aliases (``config.yaml`` ``model_aliases:``) are CLI-only by default and
never appear in the gateway/desktop picker, which renders the
``list_authenticated_providers`` rows verbatim. Aggregator rows (e.g.
``openrouter``) only show a curated subset, so an alias pointing at an
off-curation model is unreachable in the GUI. ``_inject_aliases_into_rows``
prepends the alias's resolved model id onto the matching provider row (so it
becomes selectable) and records the alias name in a per-row ``aliases`` map so
the UI can label the entry (e.g. ``qwen27b (qwen/qwen3.6-27b)``).

These tests exercise the helper in isolation by monkeypatching
``DIRECT_ALIASES``, so no config, network or auth state is required.
"""

from hermes_cli import model_switch
from hermes_cli.model_switch import DirectAlias


def _make_provider(slug, models, *, total_models=None, source="built-in"):
    """Build a dict shaped like ``list_authenticated_providers`` output."""
    return {
        "slug": slug,
        "name": slug.title(),
        "is_current": False,
        "is_user_defined": False,
        "models": list(models),
        "total_models": len(models) if total_models is None else total_models,
        "source": source,
    }


def _set_aliases(monkeypatch, aliases):
    """Force DIRECT_ALIASES to a fixed map and skip lazy config loading."""
    monkeypatch.setattr(model_switch, "DIRECT_ALIASES", dict(aliases))
    monkeypatch.setattr(model_switch, "_ensure_direct_aliases", lambda: None)


def test_offcuration_alias_model_prepended_and_labeled(monkeypatch):
    """An openrouter alias whose model is absent from the curated list is
    prepended as the real model id and labeled with the alias name."""
    rows = [_make_provider("openrouter", ["a/model", "b/model"], total_models=2)]
    _set_aliases(monkeypatch, {
        "qwen27b": DirectAlias(model="qwen/qwen3.6-27b",
                               provider="openrouter", base_url=""),
    })

    model_switch._inject_aliases_into_rows(rows)

    assert rows[0]["models"] == ["qwen/qwen3.6-27b", "a/model", "b/model"]
    assert rows[0]["aliases"] == {"qwen/qwen3.6-27b": "qwen27b"}
    # total_models reflects the added model so picker sort/labels stay in sync.
    assert rows[0]["total_models"] == 3


def test_alias_for_provider_without_row_is_skipped(monkeypatch):
    """An alias targeting a provider with no row leaves rows untouched."""
    rows = [_make_provider("openrouter", ["a/model"])]
    _set_aliases(monkeypatch, {
        "localqwen": DirectAlias(model="qwen3.6-27b",
                                 provider="llamaproxy", base_url=""),
    })

    model_switch._inject_aliases_into_rows(rows)

    assert rows[0]["models"] == ["a/model"]
    assert rows[0]["total_models"] == 1
    assert "aliases" not in rows[0]


def test_model_already_curated_is_labeled_not_duplicated(monkeypatch):
    """If the aliased model is already in the curated list, it is labeled but
    not added a second time."""
    rows = [_make_provider("openrouter", ["qwen/qwen3.6-35b-a3b", "a/model"])]
    _set_aliases(monkeypatch, {
        "qwen35b": DirectAlias(model="qwen/qwen3.6-35b-a3b",
                               provider="openrouter", base_url=""),
    })

    model_switch._inject_aliases_into_rows(rows)

    assert rows[0]["models"] == ["qwen/qwen3.6-35b-a3b", "a/model"]
    assert rows[0]["aliases"] == {"qwen/qwen3.6-35b-a3b": "qwen35b"}
    assert rows[0]["total_models"] == 2


def test_no_aliases_leaves_rows_unchanged(monkeypatch):
    """With no aliases configured, the rows pass through verbatim."""
    rows = [_make_provider("openrouter", ["a/model", "b/model"])]
    _set_aliases(monkeypatch, {})

    model_switch._inject_aliases_into_rows(rows)

    assert rows[0]["models"] == ["a/model", "b/model"]
    assert rows[0]["total_models"] == 2
    assert "aliases" not in rows[0]


def test_alias_matches_user_defined_provider_row(monkeypatch):
    """Slug matching also covers user-defined / custom provider rows; an
    unrelated row is left untouched."""
    rows = [
        _make_provider("openrouter", ["a/model"]),
        _make_provider("llamaproxy", ["gemma4-12b"], source="user-config"),
    ]
    _set_aliases(monkeypatch, {
        "bigqwen": DirectAlias(model="qwen3.6-72b",
                               provider="llamaproxy", base_url=""),
    })

    model_switch._inject_aliases_into_rows(rows)

    assert rows[0]["models"] == ["a/model"]  # unrelated row untouched
    assert "aliases" not in rows[0]
    assert rows[1]["models"] == ["qwen3.6-72b", "gemma4-12b"]
    assert rows[1]["aliases"] == {"qwen3.6-72b": "bigqwen"}
    assert rows[1]["total_models"] == 2
