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




def test_token_plan_tiers_fold_into_qwen_members():
    # Token Plan ships Personal and Team editions per region. The Team slugs
    # are registered by a user plugin rather than in-tree, so listing them as
    # Qwen group members is what keeps them folded into the Qwen row instead
    # of leaking out as top-level singles when the plugin is installed.
    team_slugs = ("alibaba-token-plan-team", "alibaba-token-plan-cn-team")
    personal_slugs = ("alibaba-token-plan", "alibaba-token-plan-cn")
    members = PROVIDER_GROUPS["qwen"][2]
    for slug in personal_slugs + team_slugs:
        assert slug in members, f"{slug} must be a declared Qwen member"

    # Feed an explicit input (not the group list) so a fold that wrongly
    # emitted a Team slug as a top-level single, or dropped it, is caught:
    # every Token Plan slug must land in the Qwen row's members, and none may
    # also appear as a standalone single row.
    rows = group_providers(list(personal_slugs) + list(team_slugs) + ["openrouter"])
    assert len(rows) == 2  # one Qwen group + the ungrouped openrouter single
    qwen = next(r for r in rows if r["kind"] == "group")
    singles = {r["slug"] for r in rows if r["kind"] == "single"}
    assert qwen["group_id"] == "qwen"
    for slug in personal_slugs + team_slugs:
        assert slug in qwen["members"], f"{slug} folded out of the Qwen row"
        assert slug not in singles, f"{slug} leaked as a top-level single"

    # Absent members are inert: with only the plugin uninstalled (no Team
    # slugs in the picker input), the Personal pair still folds and the Team
    # names add nothing.
    rows = group_providers([*personal_slugs, "openrouter"])
    qwen = next(r for r in rows if r["kind"] == "group")
    assert qwen["members"] == [*personal_slugs]


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








