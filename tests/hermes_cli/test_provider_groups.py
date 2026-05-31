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
    for gid, (label, members) in PROVIDER_GROUPS.items():
        assert label, f"group {gid} has empty label"
        assert len(members) >= 1
        for m in members:
            assert m in canonical, f"group {gid} member {m!r} is not a canonical slug"


def test_member_slugs_are_unique_across_groups():
    """A slug may belong to at most one group."""
    seen = {}
    for gid, (_label, members) in PROVIDER_GROUPS.items():
        for m in members:
            assert m not in seen, f"{m!r} in both {seen[m]!r} and {gid!r}"
            seen[m] = gid


def test_reverse_index_matches_groups():
    for gid, (_label, members) in PROVIDER_GROUPS.items():
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


def test_single_present_member_degrades_to_single_row():
    """A group with only one present member shows no submenu."""
    rows = group_providers(["xai"])  # xai-oauth absent
    assert len(rows) == 1
    assert rows[0]["kind"] == "single"
    assert rows[0]["slug"] == "xai"


def test_member_order_follows_declaration_not_input():
    """Inside a folded group, members are ordered by PROVIDER_GROUPS, not by
    the order they appeared in the input list."""
    rows = group_providers(["minimax-cn", "minimax", "minimax-oauth"])
    assert rows[0]["members"] == ["minimax", "minimax-oauth", "minimax-cn"]


def test_duplicate_slugs_ignored():
    rows = group_providers(["nous", "nous", "minimax", "minimax"])
    assert [r.get("slug") or r["group_id"] for r in rows] == ["nous", "minimax"]


def test_fold_is_lossless_for_present_slugs():
    """Every input slug (deduped) must still be reachable through the folded
    rows — grouping hides nothing."""
    flat = [p.slug for p in CANONICAL_PROVIDERS]
    rows = group_providers(flat)
    assert set(_slugs(rows)) == set(flat)


def test_canonical_fold_row_count_shrinks():
    """Folding the full canonical list produces fewer top-level rows than the
    flat list (proves grouping actually consolidates)."""
    flat = [p.slug for p in CANONICAL_PROVIDERS]
    rows = group_providers(flat)
    assert len(rows) < len(flat)
