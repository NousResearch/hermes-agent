"""Focused tests for the isolated X manager contract.

Covers the three lanes, the mandatory argument-pack requirement, the
approval-only / fail-closed behaviour, and the isolation from the legacy
x_scout / engagement auto-post pathways.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ce_dir = Path(__file__).resolve().parent.parent
if str(ce_dir) not in sys.path:
    sys.path.insert(0, str(ce_dir))

import x_manager as xm


def _pack(**kw) -> xm.ArgumentPack:
    defaults = dict(
        claim="claim text",
        evidence="evidence text",
        mechanism="mechanism text",
        position="position text",
    )
    defaults.update(kw)
    return xm.ArgumentPack(**defaults)


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Point the manager at a throwaway DB so tests never touch the live one."""
    db = tmp_path / "x_manager_test.db"
    monkeypatch.setattr(xm, "DB_PATH", db)
    yield db


# ── Argument pack contract ────────────────────────────────────────────────


def test_pack_complete_when_all_fields_present():
    assert _pack().is_complete()


@pytest.mark.parametrize("field", xm.REQUIRED_PACK_FIELDS)
def test_pack_incomplete_when_field_missing(field):
    pack = _pack(**{field: ""})
    assert not pack.is_complete()
    assert field in pack.missing_fields()


# ── Lane 1: user-material transform ───────────────────────────────────────


def test_transform_requires_text():
    with pytest.raises(xm.XManagerError):
        xm.transform_user_material({"text": ""})


def test_transform_rejects_incomplete_pack():
    with pytest.raises(xm.XManagerError):
        xm.transform_user_material(
            {"text": "a note"},
            pack=xm.ArgumentPack(claim="", evidence="", mechanism="", position=""),
        )


def test_transform_returns_pending_artifact():
    art = xm.transform_user_material(
        {"text": "a build note", "source": "build-log"},
        pack=_pack(),
    )
    assert art.lane == xm.LANE_TRANSFORM
    assert art.status == xm.STATUS_PENDING
    assert art.pack.is_complete()


# ── Lane 2: quote-tweet candidate scan ─────────────────────────────────────


def _candidate(tweet_id="1", author="alice", text="a tweet", quote_draft=None, **pack_kw):
    candidate = {
        "tweet_id": tweet_id,
        "author": author,
        "text": text,
        "pack": _pack(**pack_kw),
    }
    if quote_draft is None:
        quote_draft = f"A distinct response to tweet {tweet_id}"
    if quote_draft:
        candidate["quote_draft"] = quote_draft
    return candidate


def test_quote_scan_returns_3_to_5():
    cands = [_candidate(tweet_id=str(i), text=f"tweet {i}") for i in range(6)]
    arts = xm.scan_quote_tweet_candidates(cands)
    assert xm.QUOTE_SCAN_MIN <= len(arts) <= xm.QUOTE_SCAN_MAX


def test_quote_scan_drops_incomplete_packs():
    cands = [
        _candidate(tweet_id="1", text="ok"),
        _candidate(tweet_id="2", text="ok", evidence=""),
        _candidate(tweet_id="3", text="ok"),
        _candidate(tweet_id="4", text="ok"),
    ]
    # Only 3 complete packs survive -> exactly 3.
    arts = xm.scan_quote_tweet_candidates(cands)
    assert len(arts) == 3


def test_quote_scan_returns_empty_below_minimum():
    cands = [_candidate(tweet_id="1", text="ok"), _candidate(tweet_id="2", text="ok")]
    assert xm.scan_quote_tweet_candidates(cands) == []


def test_quote_scan_embeds_source_context():
    cands = [
        _candidate(tweet_id="42", author="bob", text="t1"),
        _candidate(tweet_id="43", author="carol", text="t2"),
        _candidate(tweet_id="44", author="dan", text="t3"),
    ]
    arts = xm.scan_quote_tweet_candidates(cands)
    assert len(arts) == 3
    first = arts[0]
    assert first.pack.context["tweet_id"] == "42"
    assert first.pack.context["author"] == "bob"
    assert "42" in first.pack.context["source_url"]


def test_quote_scan_uses_distinct_quote_draft_not_source_tweet():
    candidates = [
        _candidate(tweet_id=str(i), text="Source tweet", quote_draft=f"A distinct take {i}")
        for i in range(3)
    ]
    artifacts = xm.scan_quote_tweet_candidates(candidates)
    assert [artifact.body for artifact in artifacts] == [
        "A distinct take 0", "A distinct take 1", "A distinct take 2",
    ]


def test_quote_scan_drops_candidate_without_distinct_quote_draft():
    candidates = [
        _candidate(tweet_id="1", text="Source tweet", quote_draft="Source tweet"),
        _candidate(tweet_id="2", text="Source tweet", quote_draft="Distinct two"),
        _candidate(tweet_id="3", text="Source tweet", quote_draft="Distinct three"),
        _candidate(tweet_id="4", text="Source tweet", quote_draft="Distinct four"),
    ]
    artifacts = xm.scan_quote_tweet_candidates(candidates)
    assert len(artifacts) == 3


# ── Lane 3: morning article drafts ─────────────────────────────────────────


def test_article_requires_signals():
    with pytest.raises(xm.XManagerError):
        xm.morning_article_drafts([])


def test_article_requires_summary_text():
    with pytest.raises(xm.XManagerError):
        xm.morning_article_drafts([{"summary": ""}])


def test_article_rejects_incomplete_pack():
    with pytest.raises(xm.XManagerError):
        xm.morning_article_drafts(
            [{"summary": "a signal"}],
            pack=xm.ArgumentPack(claim="", evidence="", mechanism="", position=""),
        )


def test_article_is_text_first():
    art = xm.morning_article_drafts(
        [{"summary": "a signal"}],
        pack=_pack(),
        body="# Title\n\nBody text.",
    )
    assert art.lane == xm.LANE_ARTICLE
    assert art.body.startswith("# Title")
    assert art.pack.is_complete()


def test_stage_and_format_card_remains_pending_and_targets_manager_channel():
    artifact = xm.transform_user_material({"text": "note"}, pack=_pack())
    card = xm.stage_and_format_card(artifact)
    assert card.channel_id == xm.X_MANAGER_CHANNEL_ID
    assert card.artifact_id == artifact.id
    assert "PENDING APPROVAL" in card.body
    assert xm.list_artifacts(status=xm.STATUS_PENDING)[0]["id"] == artifact.id


def test_morning_package_stages_up_to_two_finished_articles():
    drafts = [
        {
            "signals": [{"summary": f"signal {i}"}],
            "pack": _pack(claim=f"claim {i}"),
            "body": f"# Article {i}\n\nBody",
        }
        for i in range(3)
    ]
    cards = xm.stage_morning_article_package(drafts)
    assert len(cards) == 2
    assert all(card.channel_id == xm.X_MANAGER_CHANNEL_ID for card in cards)


# ── Approval-only / fail-closed persistence ────────────────────────────────


def test_stage_writes_pending_and_rejects_incomplete():
    art = xm.transform_user_material({"text": "note"}, pack=_pack())
    xm.stage_for_approval(art)
    rows = xm.list_artifacts(status=xm.STATUS_PENDING)
    assert any(r["id"] == art.id for r in rows)
    assert all(r["status"] == xm.STATUS_PENDING for r in rows)


def test_stage_fails_closed_on_incomplete_pack():
    art = xm.XArtifact(
        id="bad", lane=xm.LANE_TRANSFORM, brand="sahil_twitter", body="x",
        pack=xm.ArgumentPack(claim="", evidence="", mechanism="", position=""),
    )
    with pytest.raises(xm.XManagerError):
        xm.stage_for_approval(art)
    assert xm.list_artifacts() == []


def test_decide_approve_and_reject():
    art = xm.transform_user_material({"text": "note"}, pack=_pack())
    xm.stage_for_approval(art)
    assert xm.decide(art.id, xm.STATUS_APPROVED, decided_by="sahil")
    assert xm.list_artifacts(status=xm.STATUS_APPROVED)[0]["id"] == art.id


def test_decide_rejects_unknown_action():
    art = xm.transform_user_material({"text": "note"}, pack=_pack())
    xm.stage_for_approval(art)
    with pytest.raises(xm.XManagerError):
        xm.decide(art.id, "publish")


def test_no_publish_path_exists():
    """The manager must expose no publish function — approval only."""
    assert not hasattr(xm, "publish")
    assert not hasattr(xm, "post")
    assert not hasattr(xm, "auto_post")


# ── Isolation from legacy auto-post pathways ───────────────────────────────


def test_manager_does_not_import_legacy_scout_or_poster():
    """The manager must not import the generic x_scout / engagement poster."""
    src = Path(xm.__file__).read_text()
    import_lines = [
        line for line in src.splitlines()
        if line.startswith(("import ", "from "))
    ]
    for banned in ("x_scout", "engagement_suggester", "engagement_x_poster"):
        assert not any(banned in line for line in import_lines), (
            f"x_manager.py imports legacy {banned}"
        )
