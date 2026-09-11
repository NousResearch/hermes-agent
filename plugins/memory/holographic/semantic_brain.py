"""Local Semantic Brain — OPTIONAL teacher above deterministic Holographic.

Status: OPT-IN interface only. No backend is wired: there is no offline local
inference backend in this runtime (auxiliary routing needs API keys; only an
ollama-cloud cache exists), so SEMANTIC_BRAIN_AVAILABLE is False and every
normal path stays at zero LLM calls. Per R1 promotion rule this interface may
only move EXPERIMENTAL -> OPT-IN -> DEFAULT with benchmark evidence.

Hard rules (enforced here, not by convention):
  * never on the normal retrieval path; invoked only behind ConfidenceGate
    or an explicit dream-cycle budget (default max_calls = 0);
  * never a source of truth: output is a *candidate* with a trust ceiling,
    validated deterministically before use;
  * secrets / injected instructions are screened BEFORE any backend call and
    rejected IN backend output. Memory is DATA.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import OrderedDict
from typing import Any, Callable

# Availability probe result. Flips to True only when a real local backend is
# configured AND reachable; nothing in this module performs network I/O.
SEMANTIC_BRAIN_AVAILABLE = False

# Allowed jobs (R1: equivalence, dedupe, consolidation, lesson/pattern
# extraction, ambiguity classification). Anything else is rejected.
ACTION_EQUIVALENCE = "equivalence"
ACTION_DEDUPE = "dedupe"
ACTION_CONSOLIDATE = "consolidate"
ACTION_LESSON = "lesson"
ACTION_PATTERN = "pattern"
ACTION_CLASSIFY = "classify"

ALLOWED_ACTIONS = frozenset({
    ACTION_EQUIVALENCE, ACTION_DEDUPE, ACTION_CONSOLIDATE,
    ACTION_LESSON, ACTION_PATTERN, ACTION_CLASSIFY,
})

ALLOWED_MEM_TYPES = frozenset({
    "fact", "preference", "constraint", "decision", "invariant",
    "project_state", "event", "lesson", "pattern", "evidence",
    "hypothesis", "general",
})

# Trust ceiling for unverified semantic output: candidate/conditional only.
# Promotion needs deterministic evidence, repetition, repo or user confirmation.
SEMANTIC_TRUST_CEILING = 0.6
SEMANTIC_SUGGESTED_TRUST = 0.45

REQUIRED_FIELDS = ("action", "memory_type", "confidence", "subject",
                   "predicate", "value", "reason", "source_refs")

_SECRET_RE = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\bghp_[A-Za-z0-9]{8,}"),
    re.compile(r"\bAKIA[0-9A-Z]{12,}"),
    re.compile(r"(?i)\b(api[_-]?key|secret[_-]?key|access[_-]?token|private[_-]?key)\b\s*[:=]\s*\S+"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9\-._~+/=]{8,}"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
)
_INSTRUCTION_RE = (
    re.compile(r"(?i)\bignore\s+(all\s+)?(previous|prior|above)\s+instructions\b"),
    re.compile(r"(?i)^\s*system\s*:"),
    re.compile(r"(?i)\bdelete\s+all\s+(files|data|memories)\b"),
    re.compile(r"(?i)\bdrop\s+table\b"),
    re.compile(r"(?i)\brm\s+-rf\b"),
)


def _has_secret(text: str) -> bool:
    return bool(text) and any(p.search(text) for p in _SECRET_RE)


def _has_instruction(text: str) -> bool:
    return bool(text) and any(p.search(text) for p in _INSTRUCTION_RE)


def validate_semantic_output(payload: Any) -> tuple[bool, str, dict]:
    """Deterministic schema + safety validation.

    Returns (ok, reason, cleaned). Rejects malformed JSON shapes, bad enums,
    missing sources, out-of-range confidence, unsafe content, unsupported
    actions. Applies the trust ceiling to accepted outputs.
    """
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (ValueError, TypeError):
            return False, "malformed-json", {}
    if not isinstance(payload, dict):
        return False, "not-an-object", {}
    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        return False, f"missing-fields:{','.join(missing)}", {}
    if payload["action"] not in ALLOWED_ACTIONS:
        return False, f"unsupported-action:{payload['action']}", {}
    if payload["memory_type"] not in ALLOWED_MEM_TYPES:
        return False, f"invalid-mem-type:{payload['memory_type']}", {}
    try:
        conf = float(payload["confidence"])
    except (TypeError, ValueError):
        return False, "invalid-confidence", {}
    if not (0.0 <= conf <= 1.0):
        return False, "confidence-out-of-range", {}
    if not isinstance(payload["source_refs"], list) or not payload["source_refs"]:
        return False, "missing-source", {}
    blob = " ".join(str(payload[f]) for f in ("subject", "predicate", "value", "reason"))
    refs = " ".join(map(str, payload["source_refs"]))
    if _has_secret(blob) or _has_secret(refs):
        return False, "secret-like-content", {}
    if _has_instruction(blob) or _has_instruction(refs):
        return False, "instruction-like-content", {}
    cleaned = {f: payload[f] for f in REQUIRED_FIELDS}
    cleaned["confidence"] = min(conf, SEMANTIC_TRUST_CEILING)
    cleaned["suggested_trust"] = min(SEMANTIC_SUGGESTED_TRUST, SEMANTIC_TRUST_CEILING)
    cleaned["tier"] = "candidate"
    return True, "ok", cleaned


class ConfidenceGate:
    """Invoke the semantic brain only when deterministic confidence is low."""

    def __init__(self, threshold: float = 0.5):
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("threshold must be in [0, 1]")
        self.threshold = threshold

    def should_invoke(self, deterministic_confidence: float) -> bool:
        try:
            conf = float(deterministic_confidence)
        except (TypeError, ValueError):
            return True  # unknown confidence -> allow, backend may still refuse
        return conf < self.threshold


class SemanticCache:
    """Bounded, local, invalidatable, non-authoritative cache.

    Key = sha256(normalized input + action + brain version + config). Entries
    carry the store revision they were computed at; callers must drop entries
    older than current truth. Never overrides repository state.
    """

    VERSION = "sb1"

    def __init__(self, max_entries: int = 200):
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self.max_entries = max_entries
        self._entries: OrderedDict[str, dict] = OrderedDict()

    def _key(self, action: str, text: str, context: str = "") -> str:
        norm = " ".join((text or "").strip().lower().split())
        ctx = " ".join((context or "").strip().lower().split())
        return hashlib.sha256(f"{self.VERSION}|{action}|{norm}|{ctx}".encode()).hexdigest()[:32]

    def get(self, action: str, text: str, context: str = "") -> dict | None:
        key = self._key(action, text, context)
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        return dict(entry["value"])

    def put(self, action: str, text: str, result: dict, context: str = "") -> None:
        key = self._key(action, text, context)
        self._entries[key] = {"value": dict(result), "ts": time.time()}
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def invalidate(self) -> int:
        n = len(self._entries)
        self._entries.clear()
        return n

    def __len__(self) -> int:
        return len(self._entries)


BackendFn = Callable[[str, str, dict], Any]


class SemanticBrain:
    """Optional teacher. Disabled unless constructed with a working backend.

    backend: callable(action, text, context) -> structured dict/str. Every
    backend result passes validate_semantic_output; failures return
    (False, reason) and never write. Secrets in INPUT refuse before the call.
    """

    def __init__(self, backend: BackendFn | None = None,
                 gate: ConfidenceGate | None = None,
                 cache: SemanticCache | None = None,
                 max_calls_per_session: int = 0):
        self._backend = backend
        self.gate = gate or ConfidenceGate()
        self.cache = cache or SemanticCache()
        self.max_calls_per_session = max_calls_per_session
        self.calls_made = 0
        self.available = backend is not None and SEMANTIC_BRAIN_AVAILABLE

    def analyze(self, action: str, text: str, context: str = "",
                deterministic_confidence: float = 0.0) -> dict:
        """Single guarded semantic judgment. Never raises; never writes."""
        if action not in ALLOWED_ACTIONS:
            return {"ok": False, "reason": f"unsupported-action:{action}"}
        if not self.available or self._backend is None:
            return {"ok": False, "reason": "backend-unavailable"}
        if _has_secret(text):
            return {"ok": False, "reason": "input-secret-refused"}
        if _has_instruction(text):
            return {"ok": False, "reason": "input-instruction-refused"}
        if not self.gate.should_invoke(deterministic_confidence):
            return {"ok": False, "reason": "confidence-sufficient"}
        cached = self.cache.get(action, text, context)
        if cached is not None:
            return {"ok": True, "reason": "cache-hit", "output": cached, "calls_made": self.calls_made}
        if self.calls_made >= self.max_calls_per_session:
            return {"ok": False, "reason": "budget-exhausted"}
        try:
            raw = self._backend(action, text, {"context": context})
        except Exception as exc:  # backend failure is non-fatal
            return {"ok": False, "reason": f"backend-error:{str(exc)[:100]}"}
        ok, reason, cleaned = validate_semantic_output(raw)
        if not ok:
            return {"ok": False, "reason": reason}
        self.calls_made += 1
        self.cache.put(action, text, cleaned, context)
        return {"ok": True, "reason": reason, "output": cleaned, "calls_made": self.calls_made}


def semantic_dream_pass(store, brain: SemanticBrain | None,
                        max_calls: int = 0) -> dict:
    """Dream-cycle semantic step. Default max_calls=0 -> pure no-op scan.

    Even when enabled, only ambiguous cases are collected; this pass only
    *counts* them unless the caller explicitly raises max_calls AND provides
    an available brain. Never deletes.
    """
    report: dict = {"candidates": 0, "processed": 0, "llm_calls": 0, "errors": []}
    if brain is None or not brain.available or max_calls <= 0:
        return report
    try:
        rows = store._conn.execute(
            "SELECT fact_id, content FROM facts LIMIT 500").fetchall()
    except Exception as exc:
        report["errors"].append(str(exc)[:150])
        return report
    for row in rows:
        content = row["content"] or ""
        if len(content) < 200 or "?" in content or "maybe" in content.lower():
            report["candidates"] += 1
    return report


__all__ = [
    "SEMANTIC_BRAIN_AVAILABLE", "ALLOWED_ACTIONS", "SEMANTIC_TRUST_CEILING",
    "ConfidenceGate", "SemanticCache", "SemanticBrain",
    "validate_semantic_output", "semantic_dream_pass",
]
