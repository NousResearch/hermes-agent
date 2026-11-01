"""In-process ownership of authentic, structured user corrections.

This rail is separate from legacy text /steer: user uploads are durable user
content, not transient tool screenshots or a mutation of a cached tool row.
"""
from dataclasses import dataclass, field
from functools import wraps
import json
import logging
import threading

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContentSteer:
    """Immutable content snapshot ticketed by its durable admission."""

    correction_id: str
    _content_json: str = field(repr=False)

    @property
    def content(self) -> list[dict]:
        return json.loads(self._content_json)


class ContentSteerPersistenceError(RuntimeError):
    """An uncommitted correction must not reach a model request."""


class ContentSteering:
    def __init__(self):
        self._lock = threading.Lock()
        self._settlement_lock = threading.Lock()
        self._open = False
        self._pending: list[ContentSteer] = []
        self._seen: set[str] = set()

    def open(self):
        with self._lock:
            if self._open or self._pending:
                raise RuntimeError("Previous content steering ownership is unresolved")
            self._seen.clear()
            self._open = True

    def accept(self, content, *, correction_id):
        if not isinstance(correction_id, str) or not correction_id.strip():
            return False
        if not isinstance(content, list) or not content:
            return False
        has_content = False
        for part in content:
            if not isinstance(part, dict):
                return False
            if part.get("type") == "text":
                if not isinstance(part.get("text"), str):
                    return False
                has_content = has_content or bool(part["text"].strip())
            elif part.get("type") == "image_url":
                image = part.get("image_url")
                if not isinstance(image, dict):
                    return False
                url = image.get("url")
                if not isinstance(url, str) or not url.startswith(("data:image/", "https://", "http://")):
                    return False
                has_content = True
            else:
                return False
        if not has_content:
            return False
        try:
            payload = ContentSteer(correction_id, json.dumps(content, allow_nan=False))
        except (TypeError, ValueError):
            return False
        with self._lock:
            if not self._open:
                return False
            if correction_id not in self._seen:
                self._pending.append(payload)
                self._seen.add(correction_id)
            return True

    def seal(self):
        with self._lock:
            self._open = False

    def withdraw(self, ticket):
        """True only for an accepted ticket the turn provably never committed."""
        with self._lock:
            for index, payload in enumerate(self._pending):
                if payload.correction_id == ticket:
                    del self._pending[index]
                    return True
        return False

    def finish_candidate(self, agent, messages, history, candidate, *, can_continue):
        # Empty-check and terminal admission close have ONE linearization point.
        with self._lock:
            if not can_continue or not self._pending:
                self._open = False
                return False
            pending = list(self._pending)
        if not self._append(agent, messages, history, pending, candidate):
            return False
        agent._emit_interim_assistant_message(candidate)
        agent._stream_needs_break = True
        return True

    def after_tools(self, agent, messages, history):
        if not self._pending or not messages:
            return False
        # The hook runs outside all executor segments. Still require a closed
        # tool protocol before introducing a real user message (error tails may
        # have only some of the results). Existing text/voice steers may already
        # follow the batch; leave their durable rows untouched.
        tail = len(messages)
        while tail and messages[tail - 1].get("role") == "user":
            message = messages[tail - 1]
            if message.get("display_kind") != "steer" and not (
                message.get("display_metadata") or {}
            ).get("content_steer_ids"):
                return False
            tail -= 1
        if not tail or messages[tail - 1].get("role") != "tool":
            return False
        answered = set()
        for message in reversed(messages[:tail]):
            if message.get("role") == "tool":
                answered.add(message.get("tool_call_id"))
                continue
            if message.get("role") != "assistant" or not message.get("tool_calls"):
                return False
            expected = {call.get("id") for call in message["tool_calls"]}
            if answered != expected:
                return False
            break
        else:
            return False
        with self._lock:
            pending = list(self._pending)
        if not pending:
            return False
        return self._append(agent, messages, history, pending)

    def _append(self, agent, messages, history, pending, candidate=None):
        with self._settlement_lock:
            with self._lock:
                if not self._open:
                    return False
            if (
                agent._interrupt_requested
                or agent.api_mode == "codex_app_server"
                or not agent._model_supports_vision()
            ):
                self.seal()
                return False
            self._commit(agent, messages, history, pending, candidate)
            return True

    def _commit(self, agent, messages, history, pending, candidate):
        from agent.message_metadata import append_message

        start = len(messages)
        if candidate is not None:
            append_message(messages, candidate)
        append_message(messages, {
            "role": "user",
            "content": [part for payload in pending for part in payload.content],
            "display_metadata": {"content_steer_ids": [p.correction_id for p in pending]},
        })
        try:
            if agent._flush_messages_to_session_db(messages, history) is False:
                raise ContentSteerPersistenceError("Could not persist user correction")
        except BaseException as exc:
            # The existing flush is transactional. Remove only our unsent
            # suffix so finalization cannot persist it AND return its original
            # event for the next turn. Previously persisted rows stay intact.
            del messages[start:]
            # A transport/storage error can be reported after COMMIT. Only a
            # successful read proving absence permits FIFO replay; uncertainty
            # is consumed/interrupted, just like an interrupted native claim.
            try:
                rows = agent._session_db.get_messages(agent.session_id)
                persisted = {ticket for row in rows for ticket in
                             (row.get("display_metadata") or {}).get("content_steer_ids", [])}
            except Exception:
                logger.warning("Cannot determine image correction commit; refusing replay", exc_info=True)
                persisted = {p.correction_id for p in pending}
            with self._lock:
                self._pending = [p for p in self._pending if p.correction_id not in persisted]
            self.seal()
            if not isinstance(exc, Exception):
                raise
            raise ContentSteerPersistenceError("Could not persist user correction") from exc
        with self._lock:
            del self._pending[:len(pending)]
        agent._session_messages = messages


def finish_content_candidate(agent, messages, history, candidate, api_call_count):
    steering = getattr(agent, "_content_steering", None)
    return steering is not None and steering.finish_candidate(
        agent, messages, history, candidate,
        can_continue=(not agent._interrupt_requested and api_call_count < agent.max_iterations
                      and agent.iteration_budget.remaining > 0),
    )


def content_steering_turn(run):
    """Seal every exit; the authority withdraws unread tickets at claim settlement."""
    @wraps(run)
    def wrapped(agent, *args, **kwargs):
        steering = getattr(agent, "_content_steering", None)
        if steering is None:
            steering = agent._content_steering = ContentSteering()
        steering.open()
        try:
            result = run(agent, *args, **kwargs)
        finally:
            steering.seal()
        return result
    return wrapped