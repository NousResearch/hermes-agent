"""Behavior contracts for the per-provider ``hidden_models:`` picker exclusion.

A vendor can keep a retired model alias in its live ``/models`` catalog long after it stops
being usable. Deleting the ID from ``custom_providers[].models`` does NOT hide it: every picker
row rebuilds its list from a live probe (see ``_models_config_is_allowlist``), so the next
``/model`` open adds the ID straight back. ``hidden_models`` is the explicit, probe-proof
exclusion.

These tests pin the contracts — not a snapshot of any provider's catalog:

- probed IDs named in ``hidden_models`` never reach a picker row;
- matching is case-insensitive, and works by slug and by endpoint URL;
- an absent ``hidden_models`` changes nothing;
- the model you are currently running stays visible even when hidden;
- the key survives config normalization (the two-place registration the real path needs).
"""

from unittest.mock import patch

import pytest

ENDPOINT = "http://127.0.0.1:9/v1"
PROBED = ["test-small", "test-retired", "test-large"]


def _entry(hidden=None, *, name="TestDS", url=ENDPOINT):
    entry = {"name": name, "base_url": url, "api_key": "sk-test", "api_mode": "chat_completions"}
    if hidden is not None:
        entry["hidden_models"] = hidden
    return entry


def _row(models=None, *, slug="testds", url=ENDPOINT, **extra):
    row = {
        "slug": slug,
        "name": slug,
        "is_current": False,
        "is_user_defined": True,
        "models": list(PROBED if models is None else models),
        "total_models": len(PROBED if models is None else models),
        "source": "user-config",
        "api_url": url,
    }
    row.update(extra)
    return row


def _finalize(rows, *, user_providers=None, custom_providers=None, current_model=""):
    from hermes_cli.model_switch_providers import _finalize_picker_rows

    return _finalize_picker_rows(rows, user_providers or {}, current_model, custom_providers)


def _by_slug(rows, slug):
    """Finalize sorts rows (current first, then by model count) — never index by position."""
    for row in rows:
        if row["slug"] == slug:
            return row
    raise AssertionError(f"no row for slug {slug!r}: {[r['slug'] for r in rows]}")


# ── the exclusion itself ──────────────────────────────────────────────────


def test_hidden_model_is_never_offered():
    """A probed ID listed in ``hidden_models`` is stripped from the row."""
    rows = _finalize([_row()], custom_providers=[_entry(["test-retired"])])

    assert _by_slug(rows, "testds")["models"] == ["test-small", "test-large"]


def test_hidden_count_is_consistent_with_the_visible_list():
    """``total_models`` must not claim a model the picker cannot show."""
    row = _by_slug(_finalize([_row()], custom_providers=[_entry(["test-retired"])]), "testds")

    assert row["total_models"] == len(row["models"])


def test_hidden_matching_is_case_insensitive():
    """Config case must not decide whether the exclusion takes effect."""
    rows = _finalize([_row()], custom_providers=[_entry(["TEST-Retired"])])

    assert "test-retired" not in _by_slug(rows, "testds")["models"]


def test_hidden_matches_by_endpoint_url_when_slug_differs():
    """The bare ``model:`` + ``base_url`` row is slugged ``custom`` — it must still be
    filtered, because the exclusion is anchored to the endpoint."""
    rows = _finalize(
        [_row(slug="custom", url=ENDPOINT)],
        custom_providers=[_entry(["test-retired"])],
    )

    assert "test-retired" not in _by_slug(rows, "custom")["models"]


def test_hidden_via_providers_dict():
    """``providers.<name>.hidden_models`` works like the ``custom_providers[]`` shape."""
    rows = _finalize([_row()], user_providers={"testds": _entry(["test-retired"])})

    assert "test-retired" not in _by_slug(rows, "testds")["models"]


def test_absent_hidden_models_leaves_the_probed_list_untouched():
    """Backward compat: configs without the key behave exactly as before."""
    row = _by_slug(_finalize([_row()], custom_providers=[_entry()]), "testds")

    assert row["models"] == PROBED
    assert "hidden_models" not in row


def test_unrelated_rows_are_not_touched():
    """A different endpoint must not inherit another provider's exclusion."""
    rows = _finalize(
        [_row(), _row(slug="other", url="http://127.0.0.1:8/v1")],
        custom_providers=[_entry(["test-retired"])],
    )

    assert _by_slug(rows, "other")["models"] == PROBED
    assert "test-retired" not in _by_slug(rows, "testds")["models"]


def test_current_model_stays_visible_when_hidden():
    """Never lie about the active model: if you are running a hidden ID, the current row
    still shows it (that is how you switch away)."""
    rows = _finalize(
        [_row(models=["test-small", "test-large"], is_current=True)],
        custom_providers=[_entry(["test-retired"])],
        current_model="test-retired",
    )
    row = _by_slug(rows, "testds")

    assert row["is_current"]
    assert row["models"][0] == "test-retired"
    assert row["total_models"] == 3


# ── config plumbing ───────────────────────────────────────────────────────


def test_hidden_models_survives_config_normalization(tmp_path, monkeypatch):
    """The key must survive ``_normalize_custom_provider_entry`` — the normalizer every real
    picker passes its rows through. A normalizer that drops the key makes the exclusion a no-op
    in production while direct-call unit tests still pass."""
    from hermes_cli.config_providers import (
        _normalize_custom_provider_entry,
        providers_dict_to_custom_providers,
    )

    normalized = _normalize_custom_provider_entry(_entry(["test-retired"]))
    assert normalized is not None
    assert normalized.get("hidden_models") == ["test-retired"]

    # Legacy list shape -> providers dict shape keeps it too.
    converted = providers_dict_to_custom_providers({"testds": _entry(["test-retired"])})
    assert converted and converted[0].get("hidden_models") == ["test-retired"]


def test_hidden_models_is_a_known_provider_key():
    """The key must be registered, otherwise config load logs 'unknown config keys ignored'
    and users get a warning for a documented setting."""
    from hermes_cli.config_providers import _KNOWN_PROVIDER_KEYS

    assert "hidden_models" in _KNOWN_PROVIDER_KEYS


def test_list_authenticated_providers_threads_custom_providers_into_finalize():
    """The exclusion lives in the finalize post-pass, so ``list_authenticated_providers``
    must hand it the custom providers. Wiring regression guard: dropping the argument would
    leave the feature silently dead in every UI while finalize's own unit tests still pass."""
    from hermes_cli import model_switch_providers

    captured: dict = {}

    def _spy(results, user_providers, current_model, custom_providers=None):
        captured["custom_providers"] = custom_providers
        return results

    entries = [_entry(["test-retired"])]
    with patch.object(model_switch_providers, "_finalize_picker_rows", side_effect=_spy), \
         patch("agent.models_dev.fetch_models_dev", return_value={}), \
         patch.object(model_switch_providers, "_build_curated_lists", return_value={}):
        model_switch_providers.list_authenticated_providers(
            custom_providers=entries, probe_custom_providers=False
        )

    assert captured.get("custom_providers") == entries


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
