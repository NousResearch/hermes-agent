"""Native Matrix poll content and response aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


UNSTABLE = "org.matrix.msc3381.poll."
REQUESTER = "ai.hermes.poll.requester"
POLL_TYPES = frozenset(f"{prefix}{kind}" for prefix in ("m.poll.", UNSTABLE) for kind in ("start", "response", "end"))


def subtype(content: dict[str, Any], kind: str) -> Any:
    value = content.get(f"m.poll.{kind}", content.get(f"{UNSTABLE}{kind}"))
    if value is not None:
        return value
    if kind == "start":
        return content.get("m.poll")
    if kind == "response" and "m.selections" in content:
        return {"answers": content["m.selections"]}
    return None


def message_text(content: Any) -> str | None:
    if not isinstance(content, dict):
        return None
    for key in ("m.text", "org.matrix.msc1767.text"):
        text = content.get(key)
        if isinstance(text, str) and text.strip():
            return text
        if isinstance(text, list):
            for rendering in text:
                if (isinstance(rendering, dict) and rendering.get("mimetype", "text/plain") == "text/plain"
                        and isinstance(rendering.get("body"), str) and rendering["body"].strip()):
                    return rendering["body"]
    return None


@dataclass(frozen=True)
class PollAnswer:
    id: str
    text: str


@dataclass(frozen=True)
class MatrixPoll:
    event_id: str
    room_id: str
    creator: str
    question: str
    answers: tuple[PollAnswer, ...]
    max_selections: int
    disclosed: bool
    # Any sender can set this key, so it identifies an owner only on a poll that the bot created.
    requester: str | None = None

    @classmethod
    def from_event(cls, raw: dict[str, Any], room_id: str) -> MatrixPoll:
        if (raw.get("room_id", room_id) != room_id or raw.get("type") not in {"m.poll.start", f"{UNSTABLE}start"}
                or "state_key" in raw or raw.get("unsigned", {}).get("redacted_because")):
            raise ValueError("The target is not an available poll in this room")
        event_id, creator = raw.get("event_id"), raw.get("sender")
        if not isinstance(event_id, str) or not event_id.startswith("$") or not isinstance(creator, str) or not creator.startswith("@"):
            raise ValueError("The poll identity is invalid")
        content = raw.get("content")
        if not isinstance(content, dict) or not isinstance(poll := subtype(content, "start"), dict):
            raise ValueError("The poll start content is invalid")
        question = message_text(poll.get("question"))
        raw_answers = poll.get("answers")
        if question is None or not isinstance(raw_answers, list) or not raw_answers:
            raise ValueError("The poll requires a question and at least one answer")
        raw_answers = raw_answers[:20]
        answers = []
        for answer in raw_answers:
            text = message_text(answer)
            answer_id = answer.get("id", answer.get("m.id")) if isinstance(answer, dict) else None
            if text is None or not isinstance(answer_id, str) or not answer_id:
                raise ValueError("The poll contains an invalid answer")
            answers.append(PollAnswer(answer_id, text))
        if len({answer.id for answer in answers}) != len(answers):
            raise ValueError("The poll contains duplicate answer IDs")
        selections = poll.get("max_selections", 1)
        if not isinstance(selections, int) or isinstance(selections, bool) or selections <= 0:
            selections = 1
        requester = content.get(REQUESTER)
        return cls(event_id, room_id, creator, question, tuple(answers), selections,
                   poll.get("kind") in {"m.disclosed", "m.poll.disclosed", f"{UNSTABLE}disclosed"},
                   requester if isinstance(requester, str) else None)

    def valid_selection(self, selection: Any) -> bool:
        return (isinstance(selection, list) and all(isinstance(answer, str) for answer in selection)
                and len(selection) <= self.max_selections and len(set(selection)) == len(selection)
                and set(selection) <= {answer.id for answer in self.answers})


    def received_selection(self, selection: Any, *, truncate_first: bool = False) -> tuple[str, ...]:
        if truncate_first and isinstance(selection, list):
            selection = selection[:self.max_selections]
        if (not isinstance(selection, list) or any(not isinstance(answer, str) for answer in selection)
                or not set(selection) <= {answer.id for answer in self.answers}):
            return ()
        return tuple(dict.fromkeys(selection[:self.max_selections]))


def poll_results(
    poll: MatrixPoll, events: list[dict[str, Any]], *, moderators: set[str] | None = None,
    incomplete_reasons: list[str] | None = None,
) -> dict[str, Any]:
    reasons = list(incomplete_reasons or [])
    related = []
    for raw in events:
        if (raw.get("room_id", poll.room_id) != poll.room_id or "state_key" in raw
                or raw.get("unsigned", {}).get("redacted_because")):
            continue
        content = raw.get("content")
        relation = content.get("m.relates_to") if isinstance(content, dict) else None
        if not isinstance(relation, dict) or relation.get("rel_type") != "m.reference" or relation.get("event_id") != poll.event_id:
            continue
        if raw.get("type") not in {"m.poll.response", f"{UNSTABLE}response", "m.poll.end", f"{UNSTABLE}end"}:
            continue
        timestamp = raw.get("origin_server_ts")
        if (not isinstance(timestamp, int) or isinstance(timestamp, bool) or timestamp < 0
                or not isinstance(raw.get("event_id"), str) or not isinstance(raw.get("sender"), str)):
            reasons.append("invalid relation identity or timestamp")
            continue
        related.append(raw)
    related.sort(key=lambda raw: (raw["origin_server_ts"], raw["event_id"]))
    end = next((raw for raw in related if raw["type"].endswith(".end")
                and isinstance(subtype(raw["content"], "end"), dict)
                and raw["sender"] in {poll.creator, *(moderators or set())}), None)
    latest: dict[str, dict[str, Any]] = {}
    for raw in related:
        if raw["type"].endswith(".response") and (end is None or raw["origin_server_ts"] <= end["origin_server_ts"]):
            latest[raw["sender"]] = raw
    counts = dict.fromkeys((answer.id for answer in poll.answers), 0)
    voters = 0
    for raw in latest.values():
        response = subtype(raw["content"], "response")
        selection = response.get("answers") if isinstance(response, dict) else None
        selections = poll.received_selection(selection, truncate_first="m.selections" in raw["content"])
        if not selections:
            continue
        voters += 1
        for answer_id in selections:
            counts[answer_id] += 1
    complete = not reasons
    visible = poll.disclosed or (end is not None and complete)
    return {
        "poll_id": poll.event_id, "question": poll.question,
        "kind": "disclosed" if poll.disclosed else "undisclosed", "max_selections": poll.max_selections,
        "closed": end is not None, "end_event_id": end["event_id"] if end else None,
        "end_timestamp": end["origin_server_ts"] if end else None,
        "complete": complete, "incomplete_reasons": sorted(set(reasons)), "results_visible": visible,
        "answers": [{"id": answer.id, "text": answer.text, "votes": counts[answer.id] if visible and complete else None}
                    for answer in poll.answers],
        "voters": voters if visible and complete else None,
    }


def poll_context(content: dict[str, Any], event_type: str | None = None) -> str | None:
    start = subtype(content, "start")
    if isinstance(start, dict):
        question = message_text(start.get("question"))
        answers = start.get("answers")
        if question and isinstance(answers, list):
            labels = [f"{answer.get('id', answer.get('m.id'))}: {message_text(answer)}" for answer in answers[:20]
                      if isinstance(answer, dict) and message_text(answer)]
            return f"[poll: {question}; answers: {'; '.join(labels)}]"
        return "[poll content unavailable]"
    if subtype(content, "response") is not None:
        return "[poll response]"
    if subtype(content, "end") is not None or event_type == "m.poll.end":
        return "[poll end event; closure authority must be checked]"
    return None
