"""Durable Dev clarification sessions for planning-mode vision refinement."""

from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from agent.auxiliary_client import call_llm
from gateway.dev_control.acceptance_criteria import (
    ACCEPTANCE_CRITERION_JSON_SCHEMA,
    ALLOWED_VERIFICATION_COMMAND_SHAPES,
    normalize_acceptance_criteria,
    validate_and_downgrade_criteria,
)
from gateway.dev_control.project_scope import DEFAULT_PROJECT_ID, resolve_project_id
from gateway.dev_control.repo_grounding import collect_repo_grounding
from hermes_state import DEFAULT_DB_PATH, apply_wal_with_fallback


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS dev_clarification_sessions (
    clarification_id TEXT PRIMARY KEY,
    project_id TEXT,
    session_id TEXT,
    status TEXT NOT NULL,
    vision_brief TEXT NOT NULL,
    current_question_index INTEGER NOT NULL DEFAULT 0,
    questions TEXT NOT NULL,
    answers TEXT NOT NULL,
    clarified_brief TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    completed_at REAL,
    payload TEXT
);

CREATE INDEX IF NOT EXISTS idx_dev_clarification_sessions_updated_at
    ON dev_clarification_sessions(updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_dev_clarification_sessions_project_status
    ON dev_clarification_sessions(project_id, status, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_dev_clarification_sessions_session
    ON dev_clarification_sessions(session_id, updated_at DESC);
"""

DEFAULT_MAX_QUESTIONS = 5
MIN_TARGET_QUESTIONS = 3
MAX_QUESTION_LIMIT = 5
CLARIFICATION_STATUSES = {"active", "completed", "cancelled", "expired"}

QUESTION_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["questions"],
    "properties": {
        "questions": {
            "type": "array",
            "minItems": MIN_TARGET_QUESTIONS,
            "maxItems": MAX_QUESTION_LIMIT,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "question_id",
                    "prompt",
                    "recommended_option_id",
                    "allow_freeform",
                    "reason",
                    "options",
                ],
                "properties": {
                    "question_id": {"type": "string"},
                    "prompt": {"type": "string"},
                    "recommended_option_id": {"type": "string"},
                    "allow_freeform": {"type": "boolean"},
                    "reason": {"type": "string"},
                    "options": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 4,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["option_id", "label", "description"],
                            "properties": {
                                "option_id": {"type": "string"},
                                "label": {"type": "string"},
                                "description": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
    },
}

CLARIFIED_BRIEF_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "refined_vision",
        "goals",
        "non_goals",
        "constraints",
        "assumptions",
        "acceptance_criteria",
        "risk_notes",
        "open_questions",
        "suggested_next_action",
    ],
    "properties": {
        "refined_vision": {"type": "string"},
        "goals": {"type": "array", "items": {"type": "string"}},
        "non_goals": {"type": "array", "items": {"type": "string"}},
        "constraints": {"type": "array", "items": {"type": "string"}},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "acceptance_criteria": {
            "type": "array",
            "minItems": 1,
            "items": ACCEPTANCE_CRITERION_JSON_SCHEMA,
        },
        "risk_notes": {"type": "array", "items": {"type": "string"}},
        "open_questions": {"type": "array", "items": {"type": "string"}},
        "suggested_next_action": {"type": "string"},
    },
}


@dataclass
class DevClarificationStore:
    """Persistence for durable Dev clarification sessions."""

    db_path: Optional[Path] = None

    def __post_init__(self) -> None:
        self.db_path = self.db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        apply_wal_with_fallback(self._conn, db_label="state.db")
        with self._conn:
            self._conn.executescript(SCHEMA_SQL)

    def close(self) -> None:
        self._conn.close()

    def create(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        now = float(payload.get("created_at") or time.time())
        payload = dict(payload)
        payload["created_at"] = now
        payload["updated_at"] = now
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO dev_clarification_sessions (
                    clarification_id, project_id, session_id, status, vision_brief,
                    current_question_index, questions, answers, clarified_brief,
                    created_at, updated_at, completed_at, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _row_values(payload),
            )
        return self.get(payload["clarification_id"]) or payload

    def update(self, clarification_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        current = self.get(clarification_id)
        if not current:
            raise KeyError(f"Clarification session not found: {clarification_id}")
        payload = {**current, **updates, "updated_at": time.time()}
        with self._conn:
            self._conn.execute(
                """
                UPDATE dev_clarification_sessions
                SET project_id = ?, session_id = ?, status = ?, vision_brief = ?,
                    current_question_index = ?, questions = ?, answers = ?,
                    clarified_brief = ?, created_at = ?, updated_at = ?,
                    completed_at = ?, payload = ?
                WHERE clarification_id = ?
                """,
                (*_row_values(payload)[1:], clarification_id),
            )
        return self.get(clarification_id) or payload

    def get(self, clarification_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            """
            SELECT *
            FROM dev_clarification_sessions
            WHERE clarification_id = ?
            """,
            (str(clarification_id or "").strip(),),
        ).fetchone()
        return _row_to_payload(row) if row else None

    def list(
        self,
        *,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[Dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if project_id:
            clauses.append("project_id = ?")
            params.append(str(project_id).strip())
        if session_id:
            clauses.append("session_id = ?")
            params.append(str(session_id).strip())
        if status:
            clauses.append("status = ?")
            params.append(str(status).strip())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(int(limit or 50), 200)))
        rows = self._conn.execute(
            f"""
            SELECT *
            FROM dev_clarification_sessions
            {where}
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        return [_row_to_payload(row) for row in rows]


def start_clarification(
    *,
    store: DevClarificationStore,
    vision_brief: str,
    project_id: Optional[str] = None,
    session_id: Optional[str] = None,
    project_context: Optional[Dict[str, Any]] = None,
    max_questions: int = DEFAULT_MAX_QUESTIONS,
) -> Dict[str, Any]:
    brief = str(vision_brief or "").strip()
    if not brief:
        raise ValueError("vision_brief is required")
    question_count = max(MIN_TARGET_QUESTIONS, min(int(max_questions or DEFAULT_MAX_QUESTIONS), MAX_QUESTION_LIMIT))
    generation_mode = "llm"
    warning = None
    normalized_project_context = _normalize_project_context(project_context, project_id=project_id)
    resolved_project_id = resolve_project_id(
        project_id,
        (normalized_project_context or {}).get("project_id"),
    )
    grounding_result = collect_repo_grounding(
        repositories=(normalized_project_context or {}).get("repositories") or [],
        vision_brief=brief,
    )
    grounding = grounding_result.get("grounding") or {}
    grounding_provenance = grounding_result.get("provenance") or []
    grounding_warnings = grounding_result.get("warnings") or []
    try:
        questions = _generate_questions(
            brief,
            max_questions=question_count,
            project_context=normalized_project_context,
            grounding=grounding,
        )
    except Exception as exc:
        generation_mode = "fallback"
        warning = f"LLM question generation failed; using deterministic fallback questions: {exc}"
        questions = _fallback_questions(max_questions=question_count)
    payload = {
        "object": "hermes.dev_clarification",
        "clarification_id": f"devclar-{uuid.uuid4().hex[:10]}",
        "project_id": resolved_project_id,
        "session_id": session_id,
        "status": "active",
        "vision_brief": brief,
        "project_context": normalized_project_context,
        "current_question_index": 0,
        "questions": questions,
        "answers": [],
        "clarified_brief": None,
        "grounding": grounding,
        "grounding_provenance": grounding_provenance,
        "grounding_warnings": grounding_warnings,
        "completed_at": None,
        "generation_mode": generation_mode,
        "warning": warning,
    }
    return _with_current_question(store.create(payload))


def list_clarifications(
    *,
    store: DevClarificationStore,
    project_id: Optional[str] = None,
    session_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    data = [_with_current_question(item) for item in store.list(
        project_id=project_id,
        session_id=session_id,
        status=status,
        limit=limit,
    )]
    return {"object": "list", "data": data, "total": len(data)}


def get_clarification(*, store: DevClarificationStore, clarification_id: str) -> Dict[str, Any]:
    payload = store.get(clarification_id)
    if not payload:
        raise KeyError(f"Clarification session not found: {clarification_id}")
    return _with_current_question(payload)


def answer_clarification(
    *,
    store: DevClarificationStore,
    clarification_id: str,
    question_id: Optional[str] = None,
    option_id: Optional[str] = None,
    answer_text: Optional[str] = None,
    skipped: bool = False,
    back: bool = False,
) -> Dict[str, Any]:
    payload = get_clarification(store=store, clarification_id=clarification_id)
    if payload["status"] != "active":
        raise ValueError(f"Clarification session is {payload['status']}, not active")
    questions = payload.get("questions") or []
    current_index = int(payload.get("current_question_index") or 0)
    if back:
        return _with_current_question(store.update(clarification_id, {
            "current_question_index": max(0, current_index - 1),
        }))
    if not questions:
        raise ValueError("Clarification session has no questions")
    if current_index >= len(questions):
        current_index = len(questions) - 1
    question = questions[current_index]
    expected_question_id = question.get("question_id")
    if question_id and question_id != expected_question_id:
        raise ValueError(f"Answer targets {question_id}, but current question is {expected_question_id}")

    option = _find_option(question, option_id)
    answer = {
        "question_id": expected_question_id,
        "question_prompt": question.get("prompt"),
        "option_id": option.get("option_id") if option else option_id,
        "option_label": option.get("label") if option else None,
        "answer_text": str(answer_text or "").strip() or None,
        "skipped": bool(skipped),
        "answered_at": time.time(),
    }
    answers = [item for item in payload.get("answers") or [] if item.get("question_id") != expected_question_id]
    answers.append(answer)
    next_index = min(current_index + 1, len(questions))
    return _with_current_question(store.update(clarification_id, {
        "answers": answers,
        "current_question_index": next_index,
    }))


def complete_clarification(*, store: DevClarificationStore, clarification_id: str) -> Dict[str, Any]:
    payload = get_clarification(store=store, clarification_id=clarification_id)
    if payload["status"] not in {"active", "completed"}:
        raise ValueError(f"Clarification session is {payload['status']} and cannot be completed")
    clarified = _build_clarified_brief(payload)
    warning = _combine_warning(payload.get("warning"), clarified.get("warning"))
    return _with_current_question(store.update(clarification_id, {
        "status": "completed",
        "current_question_index": len(payload.get("questions") or []),
        "clarified_brief": clarified,
        "warning": warning,
        "completed_at": time.time(),
    }))


def cancel_clarification(
    *,
    store: DevClarificationStore,
    clarification_id: str,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    payload = get_clarification(store=store, clarification_id=clarification_id)
    if payload["status"] == "completed":
        raise ValueError("Completed clarification sessions cannot be cancelled")
    updated = store.update(clarification_id, {
        "status": "cancelled",
        "completed_at": time.time(),
        "warning": str(reason or "").strip() or None,
    })
    return _with_current_question(updated)


def _row_values(payload: Dict[str, Any]) -> tuple[Any, ...]:
    return (
        payload["clarification_id"],
        payload.get("project_id"),
        payload.get("session_id"),
        payload["status"],
        payload["vision_brief"],
        int(payload.get("current_question_index") or 0),
        json.dumps(payload.get("questions") or [], ensure_ascii=False),
        json.dumps(payload.get("answers") or [], ensure_ascii=False),
        json.dumps(payload.get("clarified_brief"), ensure_ascii=False) if payload.get("clarified_brief") is not None else None,
        float(payload["created_at"]),
        float(payload["updated_at"]),
        payload.get("completed_at"),
        json.dumps(payload, ensure_ascii=False),
    )


def _row_to_payload(row: sqlite3.Row) -> Dict[str, Any]:
    payload = json.loads(row["payload"] or "{}")
    payload.update({
        "object": "hermes.dev_clarification",
        "clarification_id": row["clarification_id"],
        "project_id": row["project_id"],
        "session_id": row["session_id"],
        "status": row["status"],
        "vision_brief": row["vision_brief"],
        "current_question_index": int(row["current_question_index"] or 0),
        "questions": json.loads(row["questions"] or "[]"),
        "answers": json.loads(row["answers"] or "[]"),
        "clarified_brief": json.loads(row["clarified_brief"]) if row["clarified_brief"] else None,
        "created_at": float(row["created_at"]),
        "updated_at": float(row["updated_at"]),
        "completed_at": row["completed_at"],
    })
    return payload


def _with_current_question(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(payload)
    questions = payload.get("questions") or []
    index = int(payload.get("current_question_index") or 0)
    payload["current_question"] = questions[index] if 0 <= index < len(questions) and payload.get("status") == "active" else None
    minimum_answers = min(len(questions), MIN_TARGET_QUESTIONS)
    answered_question_ids = {
        item.get("question_id")
        for item in (payload.get("answers") or [])
        if item.get("question_id")
    }
    payload["can_complete"] = len(answered_question_ids) >= minimum_answers or index >= minimum_answers
    return payload


def _generate_questions(
    vision_brief: str,
    *,
    max_questions: int,
    project_context: Optional[Dict[str, Any]] = None,
    grounding: Optional[Dict[str, Any]] = None,
) -> list[Dict[str, Any]]:
    messages = [
        {
            "role": "system",
            "content": (
                "You generate specific planning clarification questions for software/product work. "
                "Return only valid JSON. No markdown. No prose outside JSON. "
                "Questions must be tailored to the user's exact vision brief. Avoid generic project-management wording. "
                "Prefer questions that clarify product behavior, workflow boundaries, data/state needs, success criteria, and risks. "
                "The JSON schema is: "
                "{\"questions\":[{\"question_id\":\"q1\",\"prompt\":\"...\","
                "\"recommended_option_id\":\"a\",\"allow_freeform\":true,\"reason\":\"...\","
                "\"options\":[{\"option_id\":\"a\",\"label\":\"...\",\"description\":\"...\"}]}]}. "
                "Generate 3 to 5 questions. Each question must have 2 to 4 concrete options. "
                "Use short option labels and descriptions that are concrete enough for a plan draft."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Vision brief:\n{vision_brief}\n\n"
                f"Project context:\n{json.dumps(project_context or {}, ensure_ascii=False)}\n\n"
                f"Read-only repository grounding:\n{json.dumps(grounding or {}, ensure_ascii=False)}\n\n"
                f"Max questions: {max_questions}"
            ),
        },
    ]
    content = _call_question_llm(messages)
    try:
        return _validate_questions(_extract_json(content), max_questions=max_questions)
    except Exception:
        repair_messages = [
            messages[0],
            {"role": "user", "content": f"Repair this into valid schema JSON only:\n{content}"},
        ]
        repaired = _call_question_llm(repair_messages)
        return _validate_questions(_extract_json(repaired), max_questions=max_questions)


def _call_question_llm(messages: list[Dict[str, str]]) -> str:
    kwargs = {
        "task": "dev_clarification",
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 3000,
        "timeout": 55,
    }
    try:
        response = call_llm(
            **kwargs,
            extra_body={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "dev_clarification_questions",
                        "schema": QUESTION_JSON_SCHEMA,
                        "strict": True,
                    },
                },
            },
        )
    except Exception as exc:
        error_text = str(exc).lower()
        if "response_format" not in error_text and "json_schema" not in error_text and "unsupported" not in error_text:
            raise
        response = call_llm(**kwargs)
    return str(response.choices[0].message.content or "").strip()


def _extract_json(content: str) -> Dict[str, Any]:
    text = str(content or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
    return _loads_json_with_small_repairs(text)


def _loads_json_with_small_repairs(text: str) -> Dict[str, Any]:
    cleaned = re.sub(r",\s*([}\]])", r"\1", str(text or "").strip())
    decoder = json.JSONDecoder()
    try:
        parsed, _ = decoder.raw_decode(cleaned)
    except json.JSONDecodeError:
        parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON root must be an object")
    return parsed


def _fallback_questions(*, max_questions: int) -> list[Dict[str, Any]]:
    questions = [
        {
            "question_id": "q1",
            "prompt": "What should this idea primarily improve first?",
            "recommended_option_id": "a",
            "allow_freeform": True,
            "reason": "The first planning split should identify the main value path before implementation details.",
            "options": [
                {"option_id": "a", "label": "User workflow", "description": "Make the end-to-end user experience clearer and easier to complete."},
                {"option_id": "b", "label": "System reliability", "description": "Reduce failures, unclear states, and recovery friction."},
                {"option_id": "c", "label": "Developer leverage", "description": "Improve how Dev plans, delegates, verifies, or ships work."},
            ],
        },
        {
            "question_id": "q2",
            "prompt": "How much should the first version do?",
            "recommended_option_id": "a",
            "allow_freeform": True,
            "reason": "A bounded first version keeps planning useful without prematurely expanding scope.",
            "options": [
                {"option_id": "a", "label": "Narrow pilot", "description": "Build the smallest useful path and prove it with one workflow."},
                {"option_id": "b", "label": "Complete workflow", "description": "Cover the full expected user journey, but avoid advanced automation."},
                {"option_id": "c", "label": "System foundation", "description": "Prioritize durable architecture and APIs before richer UI behavior."},
            ],
        },
        {
            "question_id": "q3",
            "prompt": "What should count as success for this phase?",
            "recommended_option_id": "a",
            "allow_freeform": True,
            "reason": "Acceptance criteria let Dev turn the clarified brief into a concrete implementation plan later.",
            "options": [
                {"option_id": "a", "label": "Manual proof", "description": "Felipe can complete the flow once and confirm the result is useful."},
                {"option_id": "b", "label": "Automated tests", "description": "The core behavior is covered by backend and app tests."},
                {"option_id": "c", "label": "Operational evidence", "description": "The system records enough state and diagnostics to debug failures."},
            ],
        },
        {
            "question_id": "q4",
            "prompt": "What should stay out of scope for now?",
            "recommended_option_id": "a",
            "allow_freeform": True,
            "reason": "Explicit non-goals keep the next plan from turning into a broad platform rewrite.",
            "options": [
                {"option_id": "a", "label": "No automatic execution", "description": "Do not launch workers or mutate policies from this planning flow."},
                {"option_id": "b", "label": "No new UI surface", "description": "Reuse existing composer and sidebar surfaces where possible."},
                {"option_id": "c", "label": "No broad refactor", "description": "Keep changes scoped to the planning/control-plane path."},
            ],
        },
        {
            "question_id": "q5",
            "prompt": "What risk should Dev watch most closely?",
            "recommended_option_id": "a",
            "allow_freeform": True,
            "reason": "Risk selection helps shape verification and rollout strategy.",
            "options": [
                {"option_id": "a", "label": "Ambiguous output", "description": "The system produces a plan that sounds good but is not actionable."},
                {"option_id": "b", "label": "UI dead ends", "description": "The flow loses state, hides results, or gives no clear next action."},
                {"option_id": "c", "label": "Wrong automation", "description": "The system starts implementation before Felipe explicitly approves it."},
            ],
        },
    ]
    return questions[:max(MIN_TARGET_QUESTIONS, min(int(max_questions or DEFAULT_MAX_QUESTIONS), MAX_QUESTION_LIMIT))]


def _validate_questions(payload: Dict[str, Any], *, max_questions: int) -> list[Dict[str, Any]]:
    raw_questions = payload.get("questions")
    if not isinstance(raw_questions, list):
        raise ValueError("LLM response missing questions[]")
    questions: list[Dict[str, Any]] = []
    for idx, raw in enumerate(raw_questions[:max_questions], start=1):
        if not isinstance(raw, dict):
            continue
        prompt = str(raw.get("prompt") or "").strip()
        options = raw.get("options") or []
        if not prompt or not isinstance(options, list):
            continue
        normalized_options = []
        for option_idx, option in enumerate(options[:4], start=1):
            if not isinstance(option, dict):
                continue
            option_id = str(option.get("option_id") or chr(96 + option_idx)).strip()
            label = str(option.get("label") or "").strip()
            description = str(option.get("description") or "").strip()
            if label and description:
                normalized_options.append({
                    "option_id": option_id,
                    "label": label,
                    "description": description,
                })
        if len(normalized_options) < 2:
            continue
        recommended = str(raw.get("recommended_option_id") or normalized_options[0]["option_id"]).strip()
        if recommended not in {option["option_id"] for option in normalized_options}:
            recommended = normalized_options[0]["option_id"]
        questions.append({
            "question_id": str(raw.get("question_id") or f"q{idx}").strip(),
            "prompt": prompt,
            "recommended_option_id": recommended,
            "allow_freeform": bool(raw.get("allow_freeform", True)),
            "reason": str(raw.get("reason") or "").strip(),
            "options": normalized_options,
        })
    if len(questions) < MIN_TARGET_QUESTIONS:
        raise ValueError("LLM response did not contain at least three valid questions")
    return questions


def _find_option(question: Dict[str, Any], option_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not option_id:
        return None
    for option in question.get("options") or []:
        if option.get("option_id") == option_id:
            return option
    raise ValueError(f"Unknown option_id for current question: {option_id}")


def _build_clarified_brief(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        brief = _validate_clarified_brief(_generate_clarified_brief_with_llm(payload))
        return _validate_clarified_brief_criteria(brief, payload)
    except Exception as exc:
        fallback = _fallback_clarified_brief(payload)
        fallback["warning"] = f"LLM clarified brief synthesis failed; using deterministic fallback brief: {exc}"
        return fallback


def _generate_clarified_brief_with_llm(payload: Dict[str, Any]) -> Dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": (
                "Synthesize a clarified software/product spec brief from the user's vision, "
                "clarification answers, project context, and read-only repository grounding. "
                "Return only valid JSON matching the provided schema. "
                "Make acceptance criteria concrete and verifiable. verification_detail for any "
                "machine_checkable: true criterion MUST match one of these exact command shapes: "
                f"{'; '.join(ALLOWED_VERIFICATION_COMMAND_SHAPES)}. "
                "Reference only files/paths present in the provided repository grounding; do not invent file paths. "
                "Prefer a whole-suite or directory-level command when unsure of an exact file. "
                "If no allowlisted command fits, set machine_checkable: false and describe a manual check instead. "
                "Do not create worker tasks, launch work, approve execution, or claim implementation has started."
            ),
        },
        {
            "role": "user",
            "content": json.dumps({
                "vision_brief": payload.get("vision_brief"),
                "project_context": payload.get("project_context") or {},
                "answers": _answers_context(payload),
                "grounding": payload.get("grounding") or {},
                "grounding_provenance": payload.get("grounding_provenance") or [],
                "grounding_warnings": payload.get("grounding_warnings") or [],
                "repository_grounding_paths": _grounding_paths(payload.get("grounding_provenance")),
            }, ensure_ascii=False),
        },
    ]
    content = _call_brief_llm(messages)
    try:
        return _extract_json(content)
    except Exception:
        repair_messages = [
            messages[0],
            {"role": "user", "content": f"Repair this into valid clarified brief schema JSON only:\n{content}"},
        ]
        repaired = _call_brief_llm(repair_messages)
        return _extract_json(repaired)


def _call_brief_llm(messages: list[Dict[str, str]]) -> str:
    kwargs = {
        "task": "dev_clarification_synthesis",
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 3200,
        "timeout": 55,
    }
    try:
        response = call_llm(
            **kwargs,
            extra_body={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "dev_clarified_brief",
                        "schema": CLARIFIED_BRIEF_JSON_SCHEMA,
                        "strict": True,
                    },
                },
            },
        )
    except Exception as exc:
        error_text = str(exc).lower()
        if "response_format" not in error_text and "json_schema" not in error_text and "unsupported" not in error_text:
            raise
        response = call_llm(**kwargs)
    return str(response.choices[0].message.content or "").strip()


def _validate_clarified_brief(value: Dict[str, Any]) -> Dict[str, Any]:
    criteria = normalize_acceptance_criteria(value.get("acceptance_criteria"))
    if not criteria:
        raise ValueError("Clarified brief requires acceptance_criteria")
    return {
        "refined_vision": str(value.get("refined_vision") or "").strip(),
        "goals": _string_list(value.get("goals"))[:8],
        "non_goals": _string_list(value.get("non_goals"))[:8],
        "constraints": _string_list(value.get("constraints"))[:8],
        "assumptions": _string_list(value.get("assumptions"))[:8],
        "acceptance_criteria": criteria[:8],
        "risk_notes": _string_list(value.get("risk_notes"))[:8],
        "open_questions": _string_list(value.get("open_questions"))[:8],
        "suggested_next_action": str(value.get("suggested_next_action") or "").strip()
        or "Review the clarified brief, then draft a Dev execution plan if the direction is correct.",
    }


def _validate_clarified_brief_criteria(brief: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    criteria, warnings = validate_and_downgrade_criteria(
        brief.get("acceptance_criteria"),
        repo_roots=_repo_roots_from_project_context(payload.get("project_context") or {}),
    )
    if not warnings:
        return brief
    brief = dict(brief)
    brief["acceptance_criteria"] = criteria[:8]
    brief["warning"] = _combine_warning(
        brief.get("warning"),
        "Acceptance criteria downgraded: " + " ".join(warnings),
    )
    return brief


def _fallback_clarified_brief(payload: Dict[str, Any]) -> Dict[str, Any]:
    answers = payload.get("answers") or []
    project_context = payload.get("project_context") or {}
    answered = [item for item in answers if not item.get("skipped")]
    skipped = [item for item in answers if item.get("skipped")]
    goals = []
    constraints = []
    assumptions = []
    if project_context.get("project_name"):
        assumptions.append(f"Project: {project_context.get('project_name')}")
    if project_context.get("vision"):
        assumptions.append(f"Project vision: {project_context.get('vision')}")
    for item in answered:
        text = item.get("answer_text") or item.get("option_label")
        if text:
            goals.append(f"{item.get('question_prompt')}: {text}")
            assumptions.append(f"Felipe selected: {text}")
    for item in skipped:
        constraints.append(f"Clarify later: {item.get('question_prompt')}")
    return {
        "refined_vision": payload.get("vision_brief"),
        "goals": goals[:8],
        "non_goals": ["Do not create or launch a Dev execution plan from this clarification alone."],
        "constraints": constraints,
        "assumptions": assumptions[:8],
        "acceptance_criteria": [
            {
                "statement": "A human can turn this clarified brief into an implementation plan.",
                "verification_method": "manual",
                "verification_detail": "Review the clarified brief before creating a plan artifact.",
                "machine_checkable": False,
            },
            {
                "statement": "Open questions are explicit before any worker execution begins.",
                "verification_method": "manual",
                "verification_detail": "Confirm skipped questions are listed in open_questions.",
                "machine_checkable": False,
            },
        ],
        "risk_notes": ["This is planning guidance only; technical feasibility still needs codebase validation."],
        "open_questions": [item.get("question_prompt") for item in skipped],
        "suggested_next_action": "Review the clarified brief, then draft a Dev execution plan if the direction is correct.",
    }


def _answers_context(payload: Dict[str, Any]) -> list[Dict[str, Any]]:
    questions_by_id = {
        item.get("question_id"): item
        for item in payload.get("questions") or []
    }
    answers = []
    for item in payload.get("answers") or []:
        question = questions_by_id.get(item.get("question_id")) or {}
        answers.append({
            "question_id": item.get("question_id"),
            "question": item.get("question_prompt") or question.get("prompt"),
            "answer": item.get("answer_text") or item.get("option_label"),
            "skipped": bool(item.get("skipped")),
        })
    return answers


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _combine_warning(*values: Any) -> Optional[str]:
    parts = [str(value).strip() for value in values if str(value or "").strip()]
    return " ".join(parts) if parts else None


def _repo_roots_from_project_context(project_context: Dict[str, Any]) -> list[str]:
    roots: list[str] = []
    repositories = project_context.get("repositories")
    if isinstance(repositories, list):
        for repo in repositories:
            if isinstance(repo, dict):
                path = str(repo.get("path") or "").strip()
                if path:
                    roots.append(path)
    return roots


def _grounding_paths(provenance: Any) -> list[str]:
    paths: list[str] = []
    if not isinstance(provenance, list):
        return paths
    for item in provenance:
        if isinstance(item, str):
            path = item.strip()
            if path:
                paths.append(path)
            continue
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or item.get("repo_path") or "").strip()
        if path:
            paths.append(path)
    return paths


def _normalize_project_context(
    value: Optional[Dict[str, Any]],
    *,
    project_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    repositories = []
    for item in value.get("repositories") or []:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip()
        if not path:
            continue
        repositories.append({
            "label": str(item.get("label") or "").strip() or None,
            "path": path,
        })
    work_items = [
        str(item).strip()
        for item in (value.get("work_items") or [])
        if str(item).strip()
    ]
    context = {
        "project_id": str(value.get("project_id") or project_id or "").strip() or None,
        "project_name": str(value.get("project_name") or "").strip() or None,
        "vision": str(value.get("vision") or "").strip() or None,
        "coordinator_profile": str(value.get("coordinator_profile") or "").strip() or None,
        "repositories": repositories[:10],
        "work_items": work_items[:20],
    }
    return context if any(context.get(key) for key in ("project_id", "project_name", "vision", "coordinator_profile")) or repositories or work_items else None
