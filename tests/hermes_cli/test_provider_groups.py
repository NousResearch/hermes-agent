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




def test_reverse_index_matches_groups():
    for gid, (_label, _desc, members) in PROVIDER_GROUPS.items():
        for m in members:
            assert provider_group_for_slug(m) == gid
    assert provider_group_for_slug("openrouter") == ""
    assert provider_group_for_slug("") == ""




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








