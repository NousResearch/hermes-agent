from __future__ import annotations

import sqlite3

import pytest

from TG_news.src.intelligence import NewsIntelligenceRepository
from communication_core.adapters import FakeCommunicationAdapter
from communication_core.errors import ScopeViolationError
from communication_core.repository import CommunicationRepository
from communication_core.service import CommunicationService
from communication_core.xdom import NewsCommunicationBridge


def _news_database(path):
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE sources(id TEXT PRIMARY KEY, name TEXT, url TEXT);
            CREATE TABLE articles(
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL REFERENCES sources(id),
                title TEXT,
                content TEXT,
                url TEXT
            );
            """
        )
        for index in range(1, 4):
            connection.execute(
                "INSERT INTO sources VALUES (?, ?, ?)",
                (f"s{index}", f"Source {index}", f"https://source{index}.invalid"),
            )
            connection.execute(
                "INSERT INTO articles VALUES (?, ?, ?, ?, ?)",
                (f"a{index}", f"s{index}", f"Article {index}", "public", f"https://source{index}.invalid/a"),
            )
        connection.commit()
    finally:
        connection.close()


def test_news_backlog_is_explainable_confirmed_and_archived(tmp_path):
    path = tmp_path / "news.db"
    _news_database(path)
    news = NewsIntelligenceRepository(path)
    news.initialize()

    news.upsert_story(
        story_id="story", title="Developing story", article_id="a1", source_id="s1",
        topics=["energy"], entities=["Org"], geography=["EU"], summary="first report",
        happened_at="2026-07-23T08:00:00Z",
    )
    news.upsert_story(
        story_id="story", title="Developing story", article_id="a2", source_id="s2",
        topics=["energy"], entities=["Org"], geography=["EU"], summary="second report",
        update_type="confirmation", happened_at="2026-07-23T09:00:00Z",
    )
    assert len(news.story_timeline("story")["updates"]) == 2

    news.add_claim_evidence(
        story_id="story", statement="The event happened", article_id="a1", source_id="s1", stance="supports", quote="yes"
    )
    news.add_claim_evidence(
        story_id="story", statement="The event happened", article_id="a2", source_id="s2", stance="disputes", quote="no"
    )
    matrix = news.contradiction_matrix("story")
    assert matrix[0]["contradiction"] is True
    assert {item["source_id"] for item in matrix[0]["sources"]} == {"s1", "s2"}

    reliability = news.record_reliability(
        "s1", outcome="confirmed", evidence_ref="claim:1", explanation="confirmed by later primary record"
    )
    assert reliability["score"] == 1.0
    watchlist = news.create_watchlist("Energy EU", topics=["energy"], geography=["EU"])
    assert news.preview_watchlist(watchlist["id"])[0]["story_id"] == "story"

    pending = news.confirm_breaking("story", source_id="s1", article_id="a1")
    confirmed = news.confirm_breaking("story", source_id="s2", article_id="a2")
    assert pending["status"] == "pending"
    assert confirmed["status"] == "confirmed"
    assert confirmed["distinct_sources"] == 2

    assert news.record_source_health("s3", success=False, detail_redacted="timeout", failure_threshold=2)["state"] == "degraded"
    assert news.record_source_health("s3", success=False, detail_redacted="timeout", failure_threshold=2)["state"] == "quarantined"
    with pytest.raises(ValueError, match="quarantined"):
        news.confirm_breaking("story", source_id="s3", article_id="a3")
    assert news.record_source_health("s3", success=True, detail_redacted="ok", failure_threshold=2)["outcome"] == "recovered"

    normalized = news.normalize_article(
        "a1", source_language="de", normalized_language="en", title="Normalized", summary="Summary", translator="fixture-v1"
    )
    assert normalized["source_language"] == "de"
    decision = news.explain_decision(
        "a1", decision="selected", reason_codes=["multi_source"], explanation="independently confirmed", evidence={"sources": ["s1", "s2"]}
    )
    assert decision["reason_codes"] == ["multi_source"]
    digest = news.build_digest(
        cadence="daily", period_start="2026-07-23T00:00:00Z", period_end="2026-07-24T00:00:00Z"
    )
    assert digest["stories"] == [{"id": "story", "title": "Developing story"}]
    assert digest["archive_id"].startswith("archive_")


def test_xdom_accepts_public_refs_only_and_creates_sourced_draft(tmp_path):
    repository = CommunicationRepository(tmp_path / "communication.db")
    repository.initialize()
    service = CommunicationService(repository, register_builtin_adapters=False)
    service.register_adapter(FakeCommunicationAdapter())
    source_account = repository.add_account(
        provider="fake", account_namespace="source", label="source", owner_profile="test"
    )
    target_account = repository.add_account(
        provider="fake", account_namespace="target", label="target", owner_profile="test"
    )
    person = repository.create_person("Person")
    _, source = repository.upsert_identity(
        connected_account_id=source_account["id"], external_id="source", display_name="Person", person_id=person["id"]
    )
    _, target = repository.upsert_identity(
        connected_account_id=target_account["id"], external_id="target", display_name="Person", person_id=person["id"]
    )
    repository.allow_account_link(
        source_account["id"], target_account["id"], allowed=True, actor="test", reason="test"
    )
    service.apply_route(
        person_id=person["id"], source_endpoint_id=source["id"], target_endpoint_id=target["id"]
    )
    bridge = NewsCommunicationBridge(repository)

    with pytest.raises(ScopeViolationError, match="public topic/entity"):
        bridge.suggest(
            person_id=person["id"],
            public_story={"story_id": "story", "source_urls": ["https://news.invalid/a"], "content": "must not cross"},
            contact_topics=["energy"],
        )

    suggestion = bridge.suggest(
        person_id=person["id"],
        public_story={
            "story_id": "story", "article_id": "a1", "title": "Public title",
            "topics": ["energy"], "entities": ["Org"], "source_urls": ["https://news.invalid/a"],
        },
        contact_topics=["energy"],
        contact_entities=["Other"],
    )
    assert suggestion["matched_topics"] == ["energy"]
    assert "Public title" not in repr(suggestion)
    draft = bridge.create_sourced_draft(
        suggestion["id"], service=service, source_endpoint_id=source["id"], text="Possible conversation starter"
    )
    assert "https://news.invalid/a" in draft["payload"]
    assert draft["status"] == "draft"
    assert bridge.list_suggestions(person["id"])[0]["status"] == "drafted"
