"""Focused tests for the isolated LinkedIn manager contract.

Covers the deterministic daily package (exactly 1 PM insight + 2 AI/PM posts),
the on-demand user-material lane, the mandatory argument-pack requirement, the
approval-only / fail-closed behaviour, the PM-insight direct-link requirement,
and the isolation from the X manager, the blog pipeline, the Postiz bridge, and
the legacy ``drafts`` backlog.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ce_dir = Path(__file__).resolve().parent.parent
if str(ce_dir) not in sys.path:
    sys.path.insert(0, str(ce_dir))

import li_manager as li


def _pack(**kw) -> li.ArgumentPack:
    defaults = dict(
        claim="claim text",
        evidence="evidence text",
        mechanism="mechanism text",
        position="position text",
    )
    defaults.update(kw)
    return li.ArgumentPack(**defaults)


def _insight(source_url="https://sahil.blog/post/context-vs-model", **pack_kw) -> li.LiArtifact:
    return li.LiArtifact(
        id=li._new_id(li.KIND_PM_INSIGHT),
        lane=li.LANE_DAILY_PACKAGE,
        kind=li.KIND_PM_INSIGHT,
        body=("Most PMs still treat AI adoption as a model-selection decision. "
              "The stronger product question is whether the surrounding workflow "
              "provides reliable context, evaluation and ownership before release."),
        pack=_pack(**pack_kw),
        source_url=source_url,
    )


def _post(i: int = 0, **pack_kw) -> li.LiArtifact:
    body = " ".join([
        f"Independent evidence-backed AI product argument {i}.",
        "Teams often mistake a convincing prototype for a production-ready capability.",
        "The distinction sits in the surrounding workflow: source quality, evaluation, ownership, monitoring and recovery.",
        "A model can generate a plausible response while the product still fails its user because nobody defined what good looks like.",
        "Product managers should validate the operating system around the model, not only the model output shown in a demo.",
        "That means testing failure states, measuring user outcomes and assigning ownership before increasing reach.",
        "Capability matters, but reliable adoption depends on the decisions made around it.",
    ])
    return li.LiArtifact(
        id=li._new_id(li.KIND_AI_PM_POST),
        lane=li.LANE_DAILY_PACKAGE,
        kind=li.KIND_AI_PM_POST,
        body=body,
        pack=_pack(claim=f"claim {i}", evidence=f"evidence source {i}", **pack_kw),
    )


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Point the manager at a throwaway DB so tests never touch the live one."""
    db = tmp_path / "li_manager_test.db"
    monkeypatch.setattr(li, "DB_PATH", db)
    yield db


# ── Argument pack contract ────────────────────────────────────────────────


def test_pack_complete_when_all_fields_present():
    assert _pack().is_complete()


@pytest.mark.parametrize("field", li.REQUIRED_PACK_FIELDS)
def test_pack_incomplete_when_field_missing(field):
    pack = _pack(**{field: ""})
    assert not pack.is_complete()
    assert field in pack.missing_fields()


# ── Daily package: deterministic 1 + 2 cardinality ────────────────────────


def test_daily_package_requires_exactly_two_posts():
    insight = _insight()
    with pytest.raises(li.LiManagerError):
        li.build_daily_package(insight, [_post(0)])


def test_daily_package_requires_exactly_two_posts_not_three():
    insight = _insight()
    with pytest.raises(li.LiManagerError):
        li.build_daily_package(insight, [_post(0), _post(1), _post(2)])


def test_daily_package_rejects_non_pm_insight_kind():
    insight = _insight()
    insight.kind = li.KIND_AI_PM_POST
    with pytest.raises(li.LiManagerError):
        li.build_daily_package(insight, [_post(0), _post(1)])


def test_daily_package_rejects_post_on_wrong_kind():
    insight = _insight()
    bad = _post(0)
    bad.kind = li.KIND_PM_INSIGHT
    with pytest.raises(li.LiManagerError):
        li.build_daily_package(insight, [bad, _post(1)])


def test_daily_package_rejects_insight_missing_source_url():
    insight = _insight(source_url="")
    with pytest.raises(li.LiManagerError):
        li.build_daily_package(insight, [_post(0), _post(1)])


def test_daily_package_rejects_incomplete_pack():
    insight = _insight(claim="")
    with pytest.raises(li.LiManagerError):
        li.build_daily_package(insight, [_post(0), _post(1)])


def test_daily_package_rejects_duplicate_ids():
    insight = _insight()
    post = _post(0)
    # Force a duplicate id across insight and post.
    post.id = insight.id
    with pytest.raises(li.LiManagerError):
        li.build_daily_package(insight, [post, _post(1)])


def test_daily_package_accepts_valid_1_plus_2():
    insight = _insight()
    package = li.build_daily_package(insight, [_post(0), _post(1)])
    assert package.insight.kind == li.KIND_PM_INSIGHT
    assert [p.kind for p in package.posts] == [li.KIND_AI_PM_POST, li.KIND_AI_PM_POST]
    assert package.insight.source_url


def test_stage_daily_package_returns_three_cards_in_order():
    insight = _insight()
    p0, p1 = _post(0), _post(1)
    cards = li.stage_daily_package(insight, [p0, p1], channel_id="CH")
    assert len(cards) == 3
    assert cards[0].artifact_id == insight.id
    assert {c.artifact_id for c in cards} == {insight.id, p0.id, p1.id}
    assert all(c.channel_id == "CH" for c in cards)
    # All three artifacts persisted as pending.
    pending = li.list_artifacts(status=li.STATUS_PENDING)
    assert len(pending) == 3


# ── PM insight direct-link requirement ────────────────────────────────────


def test_pm_insight_requires_non_empty_source_url_to_stage():
    insight = _insight(source_url="")
    with pytest.raises(li.LiManagerError):
        li.stage_for_approval(insight)


def test_format_card_includes_full_post_link():
    insight = _insight(source_url="https://sahil.blog/post/x")
    card = li.stage_and_format_card(insight, channel_id="CH")
    assert "https://sahil.blog/post/x" in card.body
    assert "Full post" in card.body


# ── On-demand user-material lane ──────────────────────────────────────────


def test_on_demand_requires_text():
    with pytest.raises(li.LiManagerError):
        li.transform_user_material({"text": ""})


def test_on_demand_rejects_incomplete_pack():
    with pytest.raises(li.LiManagerError):
        li.transform_user_material(
            {"text": "a note"},
            pack=li.ArgumentPack(claim="", evidence="", mechanism="", position=""),
        )


def test_on_demand_returns_pending_artifact():
    art = li.transform_user_material(
        {"text": "a shared link", "source": "shared-link"},
        pack=_pack(),
    )
    assert art.lane == li.LANE_ON_DEMAND
    assert art.kind == li.KIND_USER_MATERIAL
    assert art.status == li.STATUS_PENDING
    assert art.pack.is_complete()


# ── Approval-only / fail-closed persistence ────────────────────────────────


def test_stage_writes_pending_and_rejects_incomplete():
    art = li.transform_user_material({"text": "note"}, pack=_pack())
    li.stage_for_approval(art)
    rows = li.list_artifacts(status=li.STATUS_PENDING)
    assert any(r["id"] == art.id for r in rows)
    assert all(r["status"] == li.STATUS_PENDING for r in rows)


def test_stage_fails_closed_on_incomplete_pack():
    art = li.LiArtifact(
        id="bad", lane=li.LANE_ON_DEMAND, kind=li.KIND_USER_MATERIAL, body="x",
        pack=li.ArgumentPack(claim="", evidence="", mechanism="", position=""),
    )
    with pytest.raises(li.LiManagerError):
        li.stage_for_approval(art)
    assert li.list_artifacts() == []


def test_decide_approve_and_reject():
    art = li.transform_user_material({"text": "note"}, pack=_pack())
    li.stage_for_approval(art)
    assert li.decide(art.id, li.STATUS_APPROVED, decided_by="sahil")
    assert li.list_artifacts(status=li.STATUS_APPROVED)[0]["id"] == art.id


def test_decide_rejects_unknown_action():
    art = li.transform_user_material({"text": "note"}, pack=_pack())
    li.stage_for_approval(art)
    with pytest.raises(li.LiManagerError):
        li.decide(art.id, "publish")


def test_no_publish_path_exists():
    """The manager must expose no publish function — approval only."""
    assert not hasattr(li, "publish")
    assert not hasattr(li, "post")
    assert not hasattr(li, "auto_post")
    assert not hasattr(li, "enqueue_postiz")


# ── Isolation: no legacy coupling ─────────────────────────────────────────


def test_manager_does_not_import_legacy_surfaces():
    """The manager must not import the X manager, blog pipeline, Postiz bridge,
    or legacy social generators."""
    src = Path(li.__file__).read_text()
    import_lines = [
        line for line in src.splitlines()
        if line.startswith(("import ", "from "))
    ]
    for banned in (
        "x_manager",
        "blog_router",
        "blog_generator",
        "blog_streams",
        "postiz_bridge",
        "postiz_publisher",
        "publish_to_postiz",
        "engagement_x_poster",
        "engagement_suggester",
        "llm_drafts",
        "content_engine",
    ):
        assert not any(banned in line for line in import_lines), (
            f"li_manager.py imports legacy {banned}"
        )


def test_manager_does_not_read_legacy_drafts_table(monkeypatch):
    """The manager persists only to li_manager_artifacts. It must never read
    the legacy ``drafts`` table that still holds the old rejected sahil_linkedin
    backlog — so those rows can never be resurfaced through this contract."""
    src = Path(li.__file__).read_text()
    assert "li_manager_artifacts" in src
    assert "FROM drafts" not in src
    assert "INSERT INTO drafts" not in src


def test_manager_uses_distinct_table_from_x_manager():
    """LinkedIn and X manager state must be separate tables, not shared."""
    src = Path(li.__file__).read_text()
    assert "li_manager_artifacts" in src
    assert "x_manager_artifacts" not in src
