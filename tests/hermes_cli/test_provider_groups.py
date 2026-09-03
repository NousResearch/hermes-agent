"""Tests for provider-group folding (display-only picker grouping).

These are invariant tests, not catalog snapshots: they assert how
``group_providers`` folds a flat slug list and how member slugs relate to
``PROVIDER_GROUPS`` / ``CANONICAL_PROVIDERS`` — not the specific set of
vendors, which is expected to change over time.
"""

from types import SimpleNamespace

from hermes_cli.models import (
    CANONICAL_PROVIDERS,
    PROVIDER_GROUPS,
    fold_profile_groups,
    group_providers,
    provider_group_for_slug,
)


def _slugs(rows):
    """Flatten picker rows back to the concrete slugs they expose."""
    out = []
    for r in rows:
        if r["kind"] == "single":
            out.append(r["slug"])
        else:
            out.extend(r["members"])
    return out




def test_reverse_index_matches_groups():
    for gid, (_label, _desc, members) in PROVIDER_GROUPS.items():
        for m in members:
            assert provider_group_for_slug(m) == gid
    assert provider_group_for_slug("openrouter") == ""
    assert provider_group_for_slug("") == ""




def _profile(name, group=()):
    return SimpleNamespace(name=name, group=group)


def test_profile_declared_group_creates_a_group_row():
    """A provider plugin can declare its own group via ProviderProfile.group."""
    groups = {}
    profiles = [
        _profile("acme-eu", ("acme", "Acme", "EU & US endpoints")),
        _profile("acme-us", ("acme", "Acme", "EU & US endpoints")),
    ]
    fold_profile_groups(profiles, {"acme-eu", "acme-us"}, groups)
    assert groups == {"acme": ("Acme", "EU & US endpoints", ["acme-eu", "acme-us"])}


def test_profile_declared_group_joins_an_existing_group():
    """Naming an existing group_id appends to it, keeping its label/description."""
    groups = {"qwen": ("Qwen", "Qwen endpoints", ["alibaba"])}
    fold_profile_groups(
        [_profile("acme-eu", ("qwen", "Ignored", "Ignored"))], {"acme-eu"}, groups
    )
    assert groups["qwen"] == ("Qwen", "Qwen endpoints", ["alibaba", "acme-eu"])


def test_profile_declared_group_skips_bad_or_absent_declarations():
    """Malformed declarations and non-canonical slugs never reach the picker."""
    groups = {}
    fold_profile_groups(
        [
            _profile("no-group"),
            _profile("short-tuple", ("acme",)),
            _profile("blank-id", ("", "Acme", "desc")),
            _profile("not-canonical", ("acme", "Acme", "desc")),
        ],
        {"no-group", "short-tuple", "blank-id"},
        groups,
    )
    assert groups == {}


def test_multi_member_group_folds_to_one_row():
    rows = group_providers(["minimax", "minimax-oauth", "minimax-cn"])
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "group"
    assert row["group_id"] == "minimax"
    assert row["members"] == ["minimax", "minimax-oauth", "minimax-cn"]
    # group rows carry the short top-level description from PROVIDER_GROUPS
    assert row["description"] == PROVIDER_GROUPS["minimax"][1]
    assert row["description"]








