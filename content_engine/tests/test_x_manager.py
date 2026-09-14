"""Focused tests for the isolated X manager contract.

Covers the three lanes, the mandatory argument-pack requirement, the
approval-only / fail-closed behaviour, and the isolation from the legacy
x_scout / engagement auto-post pathways.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

ce_dir = Path(__file__).resolve().parent.parent
if str(ce_dir) not in sys.path:
    sys.path.insert(0, str(ce_dir))

import x_manager as xm


@pytest.fixture
def synthetic_voice_corpus(monkeypatch):
    """Explicit test evidence, not approval of any real account corpus."""
    import x_voice_gate
    corpus = [{"text": "Small queues keep failures visible.", "approved": True,
               "url": "https://example.test/synthetic-corpus",
               "provenance": {"kind": "synthetic_test_fixture"}}]
    monkeypatch.setattr(x_voice_gate, "load_voice_corpus", lambda: corpus)
    return corpus


def _source():
    identity = uuid4().hex
    return {"id": identity, "url": f"https://example.test/build-log/{identity}",
            "created_at": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
            "origin": "user_material"}


# One timestamp per module plus distinct sequence bits keeps X identities both
# time-consistent and unique, including multiple articles staged in one test.
_SOURCE_CREATED = datetime.now(timezone.utc) - timedelta(hours=1)


def _tweet_id(sequence=1):
    return str(((int(_SOURCE_CREATED.timestamp() * 1000) - 1288834974657) << 22) + sequence)


def _pack(**kw) -> xm.ArgumentPack:
    defaults: dict = dict(
        claim="claim text",
        evidence="evidence text",
        mechanism="mechanism text",
        position="position text",
        context={"sources": [_source()]},
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


def test_transform_rejects_incomplete_pack(synthetic_voice_corpus):
    with pytest.raises(xm.XManagerError, match="incomplete argument pack"):
        xm.transform_user_material(
            {"text": "a note"},
            pack=_pack(claim="", evidence="", mechanism="", position=""),
        )


def test_transform_returns_pending_artifact(synthetic_voice_corpus):
    art = xm.transform_user_material(
        {"text": "a build note", "source": "build-log"},
        pack=_pack(),
    )
    assert art.lane == xm.LANE_TRANSFORM
    assert art.status == xm.STATUS_PENDING
    assert art.pack.is_complete()


# ── Lane 2: quote-tweet candidate scan ─────────────────────────────────────


def _candidate(tweet_id=None, author="alice", text="a tweet", quote_draft=None, **pack_kw):
    tweet_id = tweet_id or _tweet_id()
    pack_kw.setdefault("context", {"sources": [{
        "id": tweet_id, "url": f"https://x.com/{author}/status/{tweet_id}",
        "created_at": _SOURCE_CREATED.isoformat(), "origin": "for_you",
    }]})
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


def test_quote_scan_returns_3_to_10(synthetic_voice_corpus):
    cands = [_candidate(tweet_id=_tweet_id(i), text=f"tweet {i}") for i in range(12)]
    arts = xm.scan_quote_tweet_candidates(cands)
    assert len(arts) == 10 == xm.QUOTE_SCAN_MAX


def test_quote_scan_keeps_conversations_without_complete_packs(synthetic_voice_corpus):
    cands = [
        _candidate(tweet_id=_tweet_id(1), text="ok"),
        _candidate(tweet_id=_tweet_id(2), text="ok", evidence=""),
        _candidate(tweet_id=_tweet_id(3), text="ok"),
        _candidate(tweet_id=_tweet_id(4), text="ok"),
    ]
    # Source/voice validation still runs; conversational packs are optional.
    arts = xm.scan_quote_tweet_candidates(cands)
    assert len(arts) == 4


def test_quote_scan_returns_empty_below_minimum(synthetic_voice_corpus):
    cands = [_candidate(tweet_id=_tweet_id(1), text="ok"), _candidate(tweet_id=_tweet_id(2), text="ok")]
    assert xm.scan_quote_tweet_candidates(cands) == []


def test_quote_scan_embeds_source_context(synthetic_voice_corpus):
    cands = [
        _candidate(tweet_id=_tweet_id(42), author="bob", text="t1"),
        _candidate(tweet_id=_tweet_id(43), author="carol", text="t2"),
        _candidate(tweet_id=_tweet_id(44), author="dan", text="t3"),
    ]
    arts = xm.scan_quote_tweet_candidates(cands)
    assert len(arts) == 3
    first = arts[0]
    assert first.pack.context["tweet_id"] == cands[0]["tweet_id"]
    assert first.pack.context["author"] == "bob"
    assert first.pack.context["source_url"].endswith("/" + cands[0]["tweet_id"])


def test_quote_scan_uses_distinct_quote_draft_not_source_tweet(synthetic_voice_corpus):
    candidates = [
        _candidate(tweet_id=_tweet_id(i), text="Source tweet", quote_draft=f"A distinct take {i}")
        for i in range(3)
    ]
    artifacts = xm.scan_quote_tweet_candidates(candidates)
    assert [artifact.body for artifact in artifacts] == [
        "A distinct take 0", "A distinct take 1", "A distinct take 2",
    ]


def test_quote_scan_drops_candidate_without_distinct_quote_draft(synthetic_voice_corpus):
    candidates = [
        _candidate(tweet_id=_tweet_id(1), text="Source tweet", quote_draft="Source tweet"),
        _candidate(tweet_id=_tweet_id(2), text="Source tweet", quote_draft="Distinct two"),
        _candidate(tweet_id=_tweet_id(3), text="Source tweet", quote_draft="Distinct three"),
        _candidate(tweet_id=_tweet_id(4), text="Source tweet", quote_draft="Distinct four"),
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


def test_article_rejects_incomplete_pack(synthetic_voice_corpus):
    with pytest.raises(xm.XManagerError, match="incomplete argument pack"):
        xm.morning_article_drafts(
            [{"summary": "a signal"}],
            pack=_pack(claim="", evidence="", mechanism="", position=""),
            body="# Title\n\nBody text.",
        )


def test_article_is_text_first(synthetic_voice_corpus):
    art = xm.morning_article_drafts(
        [{"summary": "a signal"}],
        pack=_pack(),
        body="# Title\n\nBody text.",
    )
    assert art.lane == xm.LANE_ARTICLE
    assert art.body.startswith("# Title")
    assert art.pack.is_complete()


def test_stage_and_format_card_remains_pending_and_targets_manager_channel(synthetic_voice_corpus):
    artifact = xm.transform_user_material({"text": "note"}, pack=_pack())
    card = xm.stage_and_format_card(artifact)
    assert card.channel_id == xm.X_MANAGER_CHANNEL_ID
    assert card.artifact_id == artifact.id
    assert "PENDING APPROVAL" in card.body
    assert xm.list_artifacts(status=xm.STATUS_PENDING)[0]["id"] == artifact.id


def test_morning_package_stages_up_to_two_finished_articles(synthetic_voice_corpus):
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


# ── Lane 4: reply drafts ───────────────────────────────────────────────────


def test_reply_artifact_requires_source_metadata():
    with pytest.raises(xm.XManagerError):
        xm.reply_draft_artifact(
            tweet_id="", author="a", source_text="src", body="body", pack=_pack()
        )


def test_reply_artifact_requires_body():
    with pytest.raises(xm.XManagerError):
        xm.reply_draft_artifact(
            tweet_id="t1", author="a", source_text="src", body="  ", pack=_pack()
        )


def test_reply_artifact_builds_pending_with_source_context(synthetic_voice_corpus):
    art = xm.reply_draft_artifact(
        tweet_id=_tweet_id(),
        author="author1",
        source_text="source text",
        body="this is a reply",
        pack=_pack(context={"sources": [{
            "id": _tweet_id(), "url": f"https://x.com/author1/status/{_tweet_id()}",
            "created_at": _SOURCE_CREATED.isoformat(), "origin": "mention",
        }]}),
    )
    assert art.lane == xm.LANE_REPLY
    assert art.status == xm.STATUS_PENDING
    assert art.body == "this is a reply"
    assert art.pack.context["tweet_id"] == _tweet_id()
    assert art.pack.context["author"] == "author1"
    assert art.pack.context["source_url"].endswith("/" + _tweet_id())


# ── Approval-only / fail-closed persistence ────────────────────────────────


def test_stage_writes_pending_and_rejects_incomplete(synthetic_voice_corpus):
    art = xm.transform_user_material({"text": "note"}, pack=_pack())
    xm.stage_for_approval(art)
    rows = xm.list_artifacts(status=xm.STATUS_PENDING)
    assert any(r["id"] == art.id for r in rows)
    assert all(r["status"] == xm.STATUS_PENDING for r in rows)


def test_stage_fails_closed_on_incomplete_pack(synthetic_voice_corpus):
    art = xm.XArtifact(
        id="bad", lane=xm.LANE_TRANSFORM, brand="sahil_twitter", body="x",
        pack=_pack(claim="", evidence="", mechanism="", position=""),
    )
    with pytest.raises(xm.XManagerError, match="incomplete argument pack"):
        xm.stage_for_approval(art)
    assert xm.list_artifacts() == []


def test_decide_approve_and_reject(synthetic_voice_corpus):
    art = xm.transform_user_material({"text": "note"}, pack=_pack())
    xm.stage_for_approval(art)
    assert xm.decide(art.id, xm.STATUS_APPROVED, decided_by="sahil")
    assert xm.list_artifacts(status=xm.STATUS_APPROVED)[0]["id"] == art.id


def test_decide_rejects_unknown_action(synthetic_voice_corpus):
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
