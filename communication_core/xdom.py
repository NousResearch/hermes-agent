"""Privacy-preserving bridge from public news references to private suggestions."""

from __future__ import annotations

from typing import Any, Iterable

from .errors import ScopeViolationError
from .repository import CommunicationRepository, json_text, json_value, stable_id, utc_now


_PUBLIC_STORY_KEYS = {
    "story_id",
    "article_id",
    "title",
    "topics",
    "entities",
    "source_urls",
}
_PRIVATE_KEYS = {"body", "content", "message", "messages", "conversation", "private_text"}


class NewsCommunicationBridge:
    """Match public topic/entity IDs without sending private text to News."""

    def __init__(self, repository: CommunicationRepository) -> None:
        self.repository = repository

    def suggest(
        self,
        *,
        person_id: str,
        public_story: dict[str, Any],
        contact_topics: Iterable[str] = (),
        contact_entities: Iterable[str] = (),
    ) -> dict[str, Any] | None:
        keys = set(public_story)
        if keys & _PRIVATE_KEYS or not keys.issubset(_PUBLIC_STORY_KEYS):
            raise ScopeViolationError(
                "XDOM accepts public topic/entity references only, never message or article bodies"
            )
        story_id = str(public_story.get("story_id") or "").strip()
        source_urls = sorted({str(value).strip() for value in public_story.get("source_urls", []) if str(value).strip()})
        if not story_id or not source_urls:
            raise ValueError("public story ID and source URLs are required")
        if self.repository.get_person(person_id) is None:
            raise KeyError(person_id)
        matched_topics = sorted(set(contact_topics) & set(public_story.get("topics", [])))
        matched_entities = sorted(set(contact_entities) & set(public_story.get("entities", [])))
        if not matched_topics and not matched_entities:
            return None
        suggestion_id = stable_id("xdom", person_id, story_id)
        rationale_parts = []
        if matched_topics:
            rationale_parts.append("shared public topics: " + ", ".join(matched_topics))
        if matched_entities:
            rationale_parts.append("shared public entities: " + ", ".join(matched_entities))
        now = utc_now()
        with self.repository.transaction() as connection:
            connection.execute(
                """INSERT INTO xdom_suggestions(
                       id, person_id, public_story_id, public_article_id,
                       matched_topics_json, matched_entities_json, source_urls_json,
                       rationale, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(person_id, public_story_id) DO UPDATE SET
                       public_article_id = excluded.public_article_id,
                       matched_topics_json = excluded.matched_topics_json,
                       matched_entities_json = excluded.matched_entities_json,
                       source_urls_json = excluded.source_urls_json,
                       rationale = excluded.rationale,
                       updated_at = excluded.updated_at""",
                (
                    suggestion_id,
                    person_id,
                    story_id,
                    public_story.get("article_id"),
                    json_text(matched_topics),
                    json_text(matched_entities),
                    json_text(source_urls),
                    "; ".join(rationale_parts),
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM xdom_suggestions WHERE id = ?", (suggestion_id,)
            ).fetchone()
        return self._decode(dict(row))

    @staticmethod
    def _decode(row: dict[str, Any]) -> dict[str, Any]:
        row["matched_topics"] = json_value(row.pop("matched_topics_json"), [])
        row["matched_entities"] = json_value(row.pop("matched_entities_json"), [])
        row["source_urls"] = json_value(row.pop("source_urls_json"), [])
        return row

    def list_suggestions(self, person_id: str | None = None) -> list[dict[str, Any]]:
        where = "WHERE person_id = ?" if person_id else ""
        params = (person_id,) if person_id else ()
        with self.repository.read_connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM xdom_suggestions {where} ORDER BY created_at DESC, id",
                params,
            ).fetchall()
        return [self._decode(dict(row)) for row in rows]

    def create_sourced_draft(
        self,
        suggestion_id: str,
        *,
        service: Any,
        source_endpoint_id: str,
        text: str,
    ) -> dict[str, Any]:
        with self.repository.read_connection() as connection:
            row = connection.execute(
                "SELECT * FROM xdom_suggestions WHERE id = ? AND status = 'suggested'",
                (suggestion_id,),
            ).fetchone()
        if row is None:
            raise KeyError(suggestion_id)
        sources = json_value(row["source_urls_json"], [])
        citation = "Sources: " + ", ".join(sources)
        payload = f"{text.rstrip()}\n\n{citation}"
        draft = service.create_draft(
            person_id=row["person_id"],
            source_endpoint_id=source_endpoint_id,
            payload=payload,
        )
        with self.repository.transaction() as connection:
            connection.execute(
                """UPDATE xdom_suggestions SET status = 'drafted', draft_id = ?,
                       updated_at = ? WHERE id = ? AND status = 'suggested'""",
                (draft["id"], utc_now(), suggestion_id),
            )
        return draft
