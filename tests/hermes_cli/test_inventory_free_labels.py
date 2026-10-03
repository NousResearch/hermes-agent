"""Tests for inventory._apply_declared_free_labels — the config-declared free
tag that lets a provider with no registered pricing fetcher still light up
the pickers' green "免费" / "Free" badge (both keyed on `pricing[model].free`).

Without it, a model the user knows is free (free-tier account, promo window,
daily-quota plan) renders exactly like an unknown one: no price, no badge.
"""

from unittest.mock import patch

import hermes_cli.inventory as inv


def test_explicit_model_list_marks_only_named_models():
    rows = [{"slug": "acme", "models": ["a/free", "a/paid", "b/free"]}]
    with patch("hermes_cli.config.read_raw_config",
               return_value={"model_free_labels": {"acme": ["a/free", "b/free"]}}):
        inv._apply_declared_free_labels(rows)
    assert rows[0]["pricing"] == {
        "a/free": {"input": "free", "output": "free", "cache": None, "free": True},
        "b/free": {"input": "free", "output": "free", "cache": None, "free": True},
    }
    assert "a/paid" not in rows[0]["pricing"]


def test_star_marks_every_model_the_row_offers():
    """A wholly-free provider: newly discovered models stay labelled with no config edit."""
    rows = [{"slug": "acme", "models": ["one", "two", "three"]}]
    with patch("hermes_cli.config.read_raw_config",
               return_value={"model_free_labels": {"acme": "*"}}):
        inv._apply_declared_free_labels(rows)
    assert set(rows[0]["pricing"]) == {"one", "two", "three"}
    assert all(v["free"] is True for v in rows[0]["pricing"].values())


def test_true_is_accepted_as_star():
    rows = [{"slug": "acme", "models": ["one"]}]
    with patch("hermes_cli.config.read_raw_config",
               return_value={"model_free_labels": {"acme": True}}):
        inv._apply_declared_free_labels(rows)
    assert rows[0]["pricing"]["one"]["free"] is True


def test_unlisted_provider_is_untouched():
    rows = [{"slug": "other", "models": ["a"]}]
    with patch("hermes_cli.config.read_raw_config",
               return_value={"model_free_labels": {"acme": "*"}}):
        inv._apply_declared_free_labels(rows)
    assert "pricing" not in rows[0]


def test_model_ids_match_case_insensitively_as_a_fallback():
    """Mirrors models_dev's exact-then-lowered lookup so a casing change keeps the label."""
    rows = [{"slug": "acme", "models": ["Org/Model-X"]}]
    with patch("hermes_cli.config.read_raw_config",
               return_value={"model_free_labels": {"acme": ["org/model-x"]}}):
        inv._apply_declared_free_labels(rows)
    # Keyed by the ROW's spelling, not the declaration's.
    assert set(rows[0]["pricing"]) == {"Org/Model-X"}


def test_exact_match_wins_over_case_insensitive():
    """An id present verbatim is not re-matched through the lowered map."""
    rows = [{"slug": "acme", "models": ["Foo", "foo"]}]
    with patch("hermes_cli.config.read_raw_config",
               return_value={"model_free_labels": {"acme": ["Foo"]}}):
        inv._apply_declared_free_labels(rows)
    assert set(rows[0]["pricing"]) == {"Foo"}


def test_a_real_quote_is_never_overridden():
    """The whole point of the guard: declaring a PAID model free must not hide its price."""
    rows = [{
        "slug": "openrouter",
        "models": ["a/paid", "b/free"],
        "pricing": {"a/paid": {"input": "$3.00", "output": "$15.00", "cache": None, "free": False}},
    }]
    with patch("hermes_cli.config.read_raw_config",
               return_value={"model_free_labels": {"openrouter": "*"}}):
        inv._apply_declared_free_labels(rows)
    assert rows[0]["pricing"]["a/paid"] == {
        "input": "$3.00", "output": "$15.00", "cache": None, "free": False,
    }
    assert rows[0]["pricing"]["b/free"]["free"] is True


def test_fetcher_sourced_free_row_is_idempotent():
    """A provider that DOES report zero rates keeps its fetcher-written entry untouched."""
    rows = [{
        "slug": "nous",
        "models": ["a/free"],
        "pricing": {"a/free": {"input": "free", "output": "free", "cache": None, "free": True}},
    }]
    with patch("hermes_cli.config.read_raw_config",
               return_value={"model_free_labels": {"nous": "*"}}):
        inv._apply_declared_free_labels(rows)
    assert rows[0]["pricing"]["a/free"]["free"] is True


def test_row_is_matched_through_its_alias_set():
    """The GUI writes custom providers under `custom:<key>`; one entry must cover every
    spelling `_apply_custom_aliases` attached, or the label silently misses."""
    rows = [{"slug": "custom:acme", "name": "Acme", "aliases": ["acme"], "models": ["m"]}]
    with patch("hermes_cli.config.read_raw_config",
               return_value={"model_free_labels": {"acme": ["m"]}}):
        inv._apply_declared_free_labels(rows)
    assert rows[0]["pricing"]["m"]["free"] is True


def test_row_is_matched_by_display_name():
    rows = [{"slug": "custom:acme", "name": "Acme Cloud", "models": ["m"]}]
    with patch("hermes_cli.config.read_raw_config",
               return_value={"model_free_labels": {"Acme Cloud": ["m"]}}):
        inv._apply_declared_free_labels(rows)
    assert rows[0]["pricing"]["m"]["free"] is True


def test_absent_section_is_a_no_op():
    """Not configured at all — the overwhelmingly common case. Must not add a pricing key."""
    rows = [{"slug": "acme", "models": ["a"]}]
    with patch("hermes_cli.config.read_raw_config", return_value={}):
        inv._apply_declared_free_labels(rows)
    assert "pricing" not in rows[0]


def test_malformed_section_exists_in_silence():
    """A section that is simply absent is the normal case — no warning, no pricing key."""
    rows = [{"slug": "acme", "models": ["a"]}]
    with patch("hermes_cli.config.read_raw_config", return_value={"model_free_labels": None}):
        inv._apply_declared_free_labels(rows)
    assert "pricing" not in rows[0]


def test_malformed_section_is_a_no_op(caplog):
    """A hand-edited typo must not raise out of a picker open."""
    for bad in ("acme", ["acme"], 42, {"acme": {"m": True}}, {"acme": []}):
        rows = [{"slug": "acme", "models": ["a"]}]
        with patch("hermes_cli.config.read_raw_config",
                   return_value={"model_free_labels": bad}):
            inv._apply_declared_free_labels(rows)
        assert "pricing" not in rows[0], bad


# --- fail loud at the config boundary (CONTRIBUTING) -----------------------
# An unusable declaration must never be a silent no-op: it must not raise (a picker
# open cannot fail on a typo) but it must be logged with the offending value.

def test_non_mapping_section_warns(caplog):
    import logging
    rows = [{"slug": "acme", "models": ["a"]}]
    with caplog.at_level(logging.WARNING, logger="hermes_cli.inventory"):
        with patch("hermes_cli.config.read_raw_config",
                   return_value={"model_free_labels": ["acme"]}):
            inv._apply_declared_free_labels(rows)
    assert any("model_free_labels" in m and "list" in m for m in caplog.messages)
    assert "pricing" not in rows[0]


def test_wrong_per_provider_shape_warns_and_names_the_provider(caplog):
    import logging
    rows = [{"slug": "acme", "models": ["a"]}]
    with caplog.at_level(logging.WARNING, logger="hermes_cli.inventory"):
        with patch("hermes_cli.config.read_raw_config",
                   return_value={"model_free_labels": {"acme": {"a": True}}}):
            inv._apply_declared_free_labels(rows)
    assert any("model_free_labels.acme" in m for m in caplog.messages)
    assert "pricing" not in rows[0]


def test_non_string_entries_warn_but_strings_still_apply(caplog):
    import logging
    rows = [{"slug": "acme", "models": ["a", "b"]}]
    with caplog.at_level(logging.WARNING, logger="hermes_cli.inventory"):
        with patch("hermes_cli.config.read_raw_config",
                   return_value={"model_free_labels": {"acme": ["a", 7]}}):
            inv._apply_declared_free_labels(rows)
    assert any("model_free_labels.acme" in m for m in caplog.messages)
    assert set(rows[0]["pricing"]) == {"a"}


def test_declared_ids_that_match_nothing_warn(caplog):
    """Drift (a model renamed or withdrawn) is worth saying out loud, not swallowing."""
    import logging
    rows = [{"slug": "acme", "models": ["a"]}]
    with caplog.at_level(logging.WARNING, logger="hermes_cli.inventory"):
        with patch("hermes_cli.config.read_raw_config",
                   return_value={"model_free_labels": {"acme": ["gone", "also-gone"]}}):
            inv._apply_declared_free_labels(rows)
    assert any("none of the declared models" in m for m in caplog.messages)
    assert "pricing" not in rows[0]


def test_config_read_failure_warns_with_the_cause(caplog):
    import logging
    rows = [{"slug": "acme", "models": ["a"]}]
    with caplog.at_level(logging.WARNING, logger="hermes_cli.inventory"):
        with patch("hermes_cli.config.read_raw_config", side_effect=RuntimeError("boom")):
            inv._apply_declared_free_labels(rows)
    assert any("could not read config" in m and "boom" in m for m in caplog.messages)
    assert "pricing" not in rows[0]


def test_unknown_model_ids_are_ignored():
    """A declaration naming a model the row no longer offers (renamed/removed) adds nothing."""
    rows = [{"slug": "acme", "models": ["a"]}]
    with patch("hermes_cli.config.read_raw_config",
               return_value={"model_free_labels": {"acme": ["gone", "a"]}}):
        inv._apply_declared_free_labels(rows)
    assert set(rows[0]["pricing"]) == {"a"}


def test_empty_row_is_skipped():
    """An unconfigured skeleton row has no models; must not gain an empty pricing dict."""
    rows = [{"slug": "acme", "models": []}]
    with patch("hermes_cli.config.read_raw_config",
               return_value={"model_free_labels": {"acme": "*"}}):
        inv._apply_declared_free_labels(rows)
    assert "pricing" not in rows[0]


def test_multiple_providers_in_one_section():
    rows = [
        {"slug": "acme", "models": ["a1", "a2"]},
        {"slug": "other", "models": ["o1"]},
    ]
    with patch("hermes_cli.config.read_raw_config", return_value={
        "model_free_labels": {"acme": ["a1"], "other": "*"},
    }):
        inv._apply_declared_free_labels(rows)
    assert set(rows[0]["pricing"]) == {"a1"}
    assert set(rows[1]["pricing"]) == {"o1"}


def test_row_without_slug_or_name_is_skipped():
    rows = [{"models": ["a"]}]
    with patch("hermes_cli.config.read_raw_config",
               return_value={"model_free_labels": {"": ["a"]}}):
        inv._apply_declared_free_labels(rows)
    assert "pricing" not in rows[0]
