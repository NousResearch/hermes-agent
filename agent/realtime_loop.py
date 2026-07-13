"""Low-latency realtime conversation loop primitives.

This module is intentionally independent of chat platforms and voice stacks. It
gives API-server clients a small "talk now, work in background" surface:

* a mutable per-session live context document,
* a no-tools talker/orchestrator LLM call backed by Hermes auxiliary routing,
* lightweight task records for background work orchestration, and
* bounded event fan-out for progress UIs.

Slow work is still expected to run through Hermes' normal AIAgent/run machinery;
the realtime loop coordinates the live conversation around that work.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


DEFAULT_REALTIME_CONTEXT: Dict[str, Any] = {
    "conversation_goal": "Help the user naturally and efficiently.",
    "current_user_need": "",
    "active_work": [],
    "known_facts": [],
    "pending_confirmations": [],
    "talker_guidance": [
        "Speak in one or two short, natural sentences.",
        "Acknowledge slow work immediately and keep the user updated.",
        "Do not claim you failed to hear the user unless the latest transcript is empty or unintelligible.",
    ],
    "guardrails": [
        "Confirm before destructive or external writes.",
        "Do not expose tool names, JSON, raw traces, or private reasoning.",
    ],
}

DEFAULT_TALKER_PROMPT = (
    "You are the realtime talker/orchestrator for a live conversation. "
    "Do not use tools. Do not do long private reasoning. Produce the next short "
    "spoken response and decide whether background work should be started. "
    "Return exactly one JSON object with keys: say, action, action_request, "
    "context_patch. action must be one of none, start_task, confirm, cancel."
)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_WORK_INTENT_RE = re.compile(
    r"\b(check|look up|lookup|find|search|read|review|open|send|write|"
    r"email|mail|calendar|invoice|research|summarize)\b",
    re.IGNORECASE,
)


@dataclass
class RealtimeTask:
    task_id: str
    session_key: str
    request: str
    status: str = "queued"
    source: str = "api"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    run_id: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None

    def to_public_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["object"] = "hermes.realtime.task"
        return data


class LiveContextStore:
    """In-memory realtime context store scoped by session key.

    The store is deliberately simple and bounded. It is not a replacement for
    Hermes long-term memory; it is the low-latency working prompt for an active
    realtime conversation.
    """

    def __init__(self, *, max_events_per_session: int = 200):
        self._contexts: Dict[str, Dict[str, Any]] = {}
        self._tasks: Dict[str, Dict[str, RealtimeTask]] = {}
        self._events: Dict[str, List[Dict[str, Any]]] = {}
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}
        self._max_events_per_session = max_events_per_session

    def get_context(self, session_key: str) -> Dict[str, Any]:
        if session_key not in self._contexts:
            self._contexts[session_key] = deepcopy(DEFAULT_REALTIME_CONTEXT)
        return deepcopy(self._contexts[session_key])

    def patch_context(self, session_key: str, patch: Dict[str, Any]) -> Dict[str, Any]:
        current = self.get_context(session_key)
        merged = _merge_context_patch(current, patch)
        self._contexts[session_key] = merged
        self.publish_event(session_key, "context.updated", {"patch": patch, "context": merged})
        return deepcopy(merged)

    def create_task(self, session_key: str, request: str, *, source: str = "api") -> RealtimeTask:
        task = RealtimeTask(
            task_id=f"rtask_{uuid.uuid4().hex}",
            session_key=session_key,
            request=request,
            source=source,
        )
        self._tasks.setdefault(session_key, {})[task.task_id] = task
        self.patch_context(
            session_key,
            {
                "active_work": {
                    "_append": [
                        {
                            "id": task.task_id,
                            "status": task.status,
                            "request": task.request,
                            "started_at": task.created_at,
                        }
                    ]
                }
            },
        )
        self.publish_event(session_key, "task.created", {"task": task.to_public_dict()})
        return task

    def update_task(self, session_key: str, task_id: str, **fields: Any) -> Optional[RealtimeTask]:
        task = self._tasks.get(session_key, {}).get(task_id)
        if task is None:
            return None
        for key, value in fields.items():
            if hasattr(task, key):
                setattr(task, key, value)
        task.updated_at = time.time()
        self.publish_event(session_key, "task.updated", {"task": task.to_public_dict()})
        return task

    def list_tasks(self, session_key: str) -> List[Dict[str, Any]]:
        return [task.to_public_dict() for task in self._tasks.get(session_key, {}).values()]

    def recent_events(self, session_key: str) -> List[Dict[str, Any]]:
        return list(self._events.get(session_key, []))

    def subscribe(self, session_key: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(session_key, []).append(queue)
        return queue

    def unsubscribe(self, session_key: str, queue: asyncio.Queue) -> None:
        subscribers = self._subscribers.get(session_key)
        if not subscribers:
            return
        try:
            subscribers.remove(queue)
        except ValueError:
            return

    def publish_event(self, session_key: str, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        event = {
            "object": "hermes.realtime.event",
            "event": event_type,
            "session_key": session_key,
            "timestamp": time.time(),
            **payload,
        }
        events = self._events.setdefault(session_key, [])
        events.append(event)
        if len(events) > self._max_events_per_session:
            del events[: len(events) - self._max_events_per_session]
        for queue in list(self._subscribers.get(session_key, [])):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass
        return event


class RealtimeLoop:
    """Realtime talker/orchestrator backed by Hermes auxiliary model routing."""

    def __init__(self, store: Optional[LiveContextStore] = None):
        self.store = store or LiveContextStore()

    async def handle_turn(
        self,
        *,
        session_key: str,
        user_text: str,
        transcript: Optional[List[Dict[str, Any]]] = None,
        context_patch: Optional[Dict[str, Any]] = None,
        base_prompt: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = 256,
        timeout: Optional[float] = 3.0,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if context_patch:
            self.store.patch_context(session_key, context_patch)

        context = self.store.get_context(session_key)
        messages = _build_talker_messages(
            base_prompt=base_prompt or DEFAULT_TALKER_PROMPT,
            user_text=user_text,
            context=context,
            transcript=transcript or [],
            tasks=self.store.list_tasks(session_key),
        )

        started = time.perf_counter()

        timeout_seconds = max(0.25, float(timeout or 3.0))
        degraded_error = ""
        if not _has_explicit_talker_provider(provider):
            degraded_error = "RuntimeError: realtime talker provider not configured"
            decision = _fallback_decision(user_text)
            self.store.publish_event(
                session_key,
                "turn.degraded",
                {
                    "reason": _short_error(degraded_error),
                    "fallback": decision,
                },
            )
        else:
            from agent.auxiliary_client import async_call_llm

            try:
                response = await asyncio.wait_for(
                    async_call_llm(
                        task="realtime_talker",
                        provider=provider,
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=timeout_seconds,
                        extra_body=extra_body,
                    ),
                    timeout=timeout_seconds,
                )
                text = _extract_response_text(response)
                decision = _parse_decision(text)
            except Exception as exc:
                degraded_error = f"{type(exc).__name__}: {exc}"
                decision = _fallback_decision(user_text)
                self.store.publish_event(
                    session_key,
                    "turn.degraded",
                    {
                        "reason": _short_error(degraded_error),
                        "fallback": decision,
                    },
                )
        elapsed = time.perf_counter() - started
        patch = decision.get("context_patch")
        if isinstance(patch, dict) and patch:
            context = self.store.patch_context(session_key, patch)

        task = None
        if decision.get("action") == "start_task":
            request = str(decision.get("action_request") or user_text).strip()
            if request:
                task = self.store.create_task(session_key, request, source="turn")

        result = {
            "object": "hermes.realtime.turn",
            "session_key": session_key,
            "say": str(decision.get("say") or "").strip(),
            "action": decision.get("action") or "none",
            "action_request": str(decision.get("action_request") or "").strip(),
            "context": context,
            "task": task.to_public_dict() if task else None,
            "latency_seconds": round(elapsed, 3),
            "degraded": bool(degraded_error),
        }
        self.store.publish_event(session_key, "turn.completed", result)
        return result


def _has_explicit_talker_provider(provider: Optional[str]) -> bool:
    """Return True only when callers intentionally route the talker LLM.

    The realtime endpoint is used in live voice loops, so provider auto-detection
    is too risky: auth discovery can block the event loop and create dead air.
    Clients that want the no-tools talker model should pass a concrete provider.
    """
    normalized = str(provider or "").strip().lower()
    return bool(normalized and normalized != "auto")


def _build_talker_messages(
    *,
    base_prompt: str,
    user_text: str,
    context: Dict[str, Any],
    transcript: List[Dict[str, Any]],
    tasks: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    user_payload = {
        "latest_user_text": user_text,
        "live_context": context,
        "active_tasks": tasks,
        "transcript": transcript[-30:],
    }
    return [
        {"role": "system", "content": base_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True, sort_keys=True)},
    ]


def _extract_response_text(response: Any) -> str:
    try:
        return str(response.choices[0].message.content or "")
    except Exception:
        pass
    if isinstance(response, str):
        return response
    return str(response or "")


def _fallback_decision(user_text: str) -> Dict[str, Any]:
    """Return a phone-safe response when the optional talker model is unavailable."""
    needs_work = bool(_WORK_INTENT_RE.search(user_text or ""))
    if needs_work:
        return {
            "say": "I am checking that now. This may take a moment, and I will keep you updated.",
            "action": "start_task",
            "action_request": user_text.strip(),
            "context_patch": {"current_user_need": user_text.strip()},
        }
    return {
        "say": "I heard you. Give me a moment to make sure I handle that correctly.",
        "action": "none",
        "action_request": "",
        "context_patch": {"current_user_need": user_text.strip()},
    }


def _short_error(text: str, limit: int = 240) -> str:
    return re.sub(r"[\r\n\t]+", " ", text).strip()[:limit]


def _parse_decision(text: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    match = _JSON_OBJECT_RE.search(text or "")
    if match:
        try:
            raw = json.loads(match.group(0))
            if isinstance(raw, dict):
                payload = raw
        except json.JSONDecodeError:
            payload = {}
    if not payload:
        payload = {"say": text.strip(), "action": "none", "action_request": "", "context_patch": {}}

    action = str(payload.get("action") or "none").strip().lower()
    if action not in {"none", "start_task", "confirm", "cancel"}:
        action = "none"
    payload["action"] = action
    payload["say"] = str(payload.get("say") or "").strip()
    payload["action_request"] = str(payload.get("action_request") or "").strip()
    if not isinstance(payload.get("context_patch"), dict):
        payload["context_patch"] = {}
    return payload


def _merge_context_patch(current: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(current)
    for key, value in patch.items():
        if isinstance(value, dict) and "_append" in value:
            existing = merged.get(key)
            if not isinstance(existing, list):
                existing = []
            additions = value.get("_append")
            if isinstance(additions, list):
                merged[key] = (existing + additions)[-50:]
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_context_patch(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged
