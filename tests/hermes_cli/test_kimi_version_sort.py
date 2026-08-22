"""Tests for Kimi `k`-prefixed version parsing in the model sort key (#78886).

`_model_sort_key` only recognized `v`/`V` as a version prefix, so Moonshot
Kimi ids (`kimi-k3`, `kimi-k2.6`, `kimi-k1.5`) fell back to plain string
comparison of the suffix. Because ``"k2.6" < "k3"`` lexicographically, the
catalog-search branch of alias resolution picked `kimi-k2.6` as the "latest"
model whenever a higher `k3.x` existed.
"""
from __future__ import annotations

from hermes_cli.model_switch import _model_sort_key

KIMI_PREFIX = "moonshotai/kimi"


def test_kimi_k_versions_parse_numerically():
    """k3 must outrank k2.6 regardless of lexicographic order."""
    models = ["moonshotai/kimi-k2.6", "moonshotai/kimi-k3"]
    models.sort(key=lambda m: _model_sort_key(m, KIMI_PREFIX))
    assert models[0] == "moonshotai/kimi-k3"


def test_kimi_k3_5_outranks_k3():
    """A higher minor version must win even when the string sorts lower."""
    models = ["moonshotai/kimi-k3", "moonshotai/kimi-k3.5"]
    models.sort(key=lambda m: _model_sort_key(m, KIMI_PREFIX))
    assert models[0] == "moonshotai/kimi-k3.5"


def test_kimi_full_family_order():
    """k1.5 < k2 < k2.6 < k3 with the whole family present."""
    models = [
        "moonshotai/kimi-k1.5",
        "moonshotai/kimi-k2",
        "moonshotai/kimi-k2.6",
        "moonshotai/kimi-k3",
    ]
    models.sort(key=lambda m: _model_sort_key(m, KIMI_PREFIX))
    assert models[0] == "moonshotai/kimi-k3"


def test_kimi_uppercase_k_prefix_parses():
    """Uppercase K must be recognized too (issue reports k/K)."""
    models = ["moonshotai/kimi-K2.6", "moonshotai/kimi-K3"]
    models.sort(key=lambda m: _model_sort_key(m, KIMI_PREFIX))
    assert models[0] == "moonshotai/kimi-K3"


def test_kimi_non_numeric_k_suffix_keeps_its_text():
    """k/K is a version marker only when it introduces a numeric version.

    Regression for review feedback on #78892: ``kimi-king`` and ``kimi-ing``
    must keep distinct sort keys — otherwise alias resolution can let catalog
    order pick the model.
    """
    king = _model_sort_key("moonshotai/kimi-king", KIMI_PREFIX)
    ing = _model_sort_key("moonshotai/kimi-ing", KIMI_PREFIX)
    assert king != ing
    # The leading k is not consumed as a version marker; both suffixes keep
    # their full text.
    assert king[-1] == "king"
    assert ing[-1] == "ing"


def test_kimi_k_marker_requires_following_digit():
    """k/K introduces a version only when the next char is a digit."""
    versioned = _model_sort_key("moonshotai/kimi-k3", KIMI_PREFIX)
    suffix = _model_sort_key("moonshotai/kimi-kx", KIMI_PREFIX)
    # k3 parses as a numeric version; kx keeps the k in the suffix.
    assert versioned[0] == -3.0
    assert suffix[-1] == "kx"


def test_kimi_k_suffix_after_version_keeps_text():
    """k/K after a version separator is not consumed as a new version marker."""
    models = ["moonshotai/kimi-k2.6-king", "moonshotai/kimi-k2.6-ing"]
    keys = [_model_sort_key(m, KIMI_PREFIX) for m in models]
    assert keys[0] != keys[1]
    assert keys[0][-1] == "king"
    assert keys[1][-1] == "ing"
