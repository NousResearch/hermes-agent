"""Tests for provider-group folding (display-only picker grouping).

These are invariant tests, not catalog snapshots: they assert how
``group_providers`` folds a flat slug list and how member slugs relate to
``PROVIDER_GROUPS`` / ``CANONICAL_PROVIDERS`` — not the specific set of
vendors, which is expected to change over time.
"""

from hermes_cli.models import (
    CANONICAL_PROVIDERS,
    PROVIDER_GROUPS,
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


def test_groups_reference_real_canonical_slugs():
    """Every group member must be an actual provider slug. Guards typos and
    stale group entries after a provider is renamed/removed."""
    canonical = {p.slug for p in CANONICAL_PROVIDERS}
    for gid, (label, desc, members) in PROVIDER_GROUPS.items():
        assert label, f"group {gid} has empty label"
        assert desc, f"group {gid} has empty description"
        assert len(members) >= 1
        for m in members:
            assert m in canonical, f"group {gid} member {m!r} is not a canonical slug"


def test_reverse_index_matches_groups():
    for gid, (_label, _desc, members) in PROVIDER_GROUPS.items():
        for m in members:
            assert provider_group_for_slug(m) == gid
    assert provider_group_for_slug("openrouter") == ""
    assert provider_group_for_slug("") == ""


def test_ungrouped_providers_pass_through_in_order():
    rows = group_providers(["nous", "openrouter", "deepseek"])
    assert all(r["kind"] == "single" for r in rows)
    assert [r["slug"] for r in rows] == ["nous", "openrouter", "deepseek"]


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


def test_group_appears_at_first_member_position():
    """The group row takes the slot of its earliest-listed present member,
    and later members do not re-emit."""
    rows = group_providers(["nous", "minimax", "deepseek", "minimax-cn"])
    kinds = [(r["kind"], r.get("group_id") or r.get("slug")) for r in rows]
    assert kinds == [
        ("single", "nous"),
        ("group", "minimax"),
        ("single", "deepseek"),
    ]
    # both minimax members folded into the single group row
    assert rows[1]["members"] == ["minimax", "minimax-cn"]


def test_duplicate_slugs_ignored():
    rows = group_providers(["nous", "nous", "minimax", "minimax"])
    assert [r.get("slug") or r["group_id"] for r in rows] == ["nous", "minimax"]


def test_fold_is_lossless_for_present_slugs():
    """Every input slug (deduped) must still be reachable through the folded
    rows — grouping hides nothing."""
    flat = [p.slug for p in CANONICAL_PROVIDERS]
    rows = group_providers(flat)
    assert set(_slugs(rows)) == set(flat)


