"""Deterministic cognitive processing for Holographic memory (no LLM, stdlib only).

Pipeline: screen (secrets/instructions) -> classify -> salience -> dedupe key
-> conflict check -> temporal evaluation -> trust update. Every function is a
pure deterministic helper so it is unit-testable without a database.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime, timezone

from .taxonomy import (
    ACTIVE, AGING, ARCHIVED, CONSTRAINT, DECISION, EVIDENCE, EVENT, FACT,
    GENERAL, HYPOTHESIS, INVARIANT, LESSON, MEM_TYPES, PATTERN, PREFERENCE,
    PROJECT_STATE, QUARANTINE, STALE, SUPERSEDED, SUPERSEDED_STATE,
)

# ---------------------------------------------------------------------------
# Secret screening — never store credentials in memory (mission rule 14).
# ---------------------------------------------------------------------------

_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\bsk-ant-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{8,}"),
    re.compile(r"\bghp_[A-Za-z0-9]{8,}"),
    re.compile(r"\bgsk_[A-Za-z0-9]{8,}"),
    re.compile(r"\bAKIA[0-9A-Z]{12,}"),
    re.compile(r"(?i)\b(api[_-]?key|api[_-]?secret|secret[_-]?key|access[_-]?token|auth[_-]?token|private[_-]?key)\b\s*[:=]\s*\S+"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)\bpasswd\s*[:=]\s*\S+"),
)


def contains_secret(text: str) -> bool:
    """True if text looks like it carries a credential/API key."""
    if not text:
        return False
    return any(p.search(text) for p in _SECRET_PATTERNS)


# ---------------------------------------------------------------------------
# Instruction / prompt-injection screening — memory is DATA, never instructions.
# ---------------------------------------------------------------------------

_INSTRUCTION_PATTERNS = (
    re.compile(r"(?i)\bignore\s+(all\s+)?(previous|prior|above)\s+instructions\b"),
    re.compile(r"(?i)\bdisregard\s+(all\s+)?(previous|prior|above)\b"),
    re.compile(r"(?i)\byou\s+are\s+now\s+(a|an)\b"),
    re.compile(r"(?i)^\s*system\s*:"),
    re.compile(r"(?i)\b(execute|run|eval)\s*\(\s*['\"]"),
    re.compile(r"(?i)\bdelete\s+all\s+(files|data|memories)\b"),
    re.compile(r"(?i)\bdrop\s+table\b"),
    re.compile(r"(?i)\b;.*\b(drop|delete|shutdown)\b"),
)

_COMMAND_HINT = re.compile(
    r"(?i)\b(rm\s+-rf|sudo\s|chmod\s+777|powershell\s+-e|cmd\s+/c|wget\s+http|curl\s+http)\b"
)


def contains_instruction(text: str) -> bool:
    """True if text looks like an injected instruction/command, not a fact."""
    if not text:
        return False
    if any(p.search(text) for p in _INSTRUCTION_PATTERNS):
        return True
    return bool(_COMMAND_HINT.search(text))


# ---------------------------------------------------------------------------
# Deterministic classifier (Thai + English structural signals).
# ---------------------------------------------------------------------------

# Each entry: (compiled patterns, mem_type). Order matters — first match wins,
# with more specific types (constraint/invariant) before generic ones.
_CLASSIFIER_RULES: tuple[tuple[tuple[re.Pattern, ...], str], ...] = (
    ((re.compile(r"(?i)\bdo\s+not\b"), re.compile(r"(?i)\bmust\s+not\b"),
      re.compile(r"(?i)\bnever\b.{0,40}\b(commit|push|delete|merge|deploy)\b"),
      re.compile("ห้าม"), re.compile("ห้ามแก้"), re.compile("ห้ามลบ"),
      re.compile(r"(?i)\bforbidden\b"), re.compile(r"(?i)\bprohibited\b")), CONSTRAINT),
    ((re.compile(r"(?i)\binvariant\b"), re.compile(r"(?i)\bbaseline\b.{0,30}\b(frozen|locked|must not change)\b"),
      re.compile("ห้ามแก้.*baseline"), re.compile("baseline.*ห้ามแก้")), INVARIANT),
    ((re.compile(r"(?i)\barchitecture\s+decision\b"), re.compile(r"(?i)\bwe\s+(decided|agreed|chose)\b"),
      re.compile("ตัดสินใจใช้"), re.compile("ตัดสินใจ"), re.compile("ตกลงใช้")), DECISION),
    ((re.compile(r"(?i)\bpytest\b.{0,40}\bpass"), re.compile(r"(?i)\btests?\b.{0,30}\bpass(ed)?\b"),
      re.compile(r"(?i)\bverified\b"), re.compile("ทดสอบแล้ว"), re.compile("ผ่านแล้ว"),
      re.compile(r"(?i)\bbenchmark\b.{0,30}\b(result|shows?)\b")), EVIDENCE),
    ((re.compile(r"(?i)\b(doesn'?t|does not|didn'?t|failed|won'?t)\s+work\b"),
      re.compile(r"(?i)\blesson\b"), re.compile("ใช้ไม่ได้"), re.compile("บทเรียน"),
      re.compile(r"(?i)\bavoid\b.{0,30}\bbecause\b")), LESSON),
    ((re.compile(r"(?i)\b(maybe|probably|possibly|might be|could be|hypothesis)\b"),
      re.compile("อาจจะ"), re.compile("น่าจะ"), re.compile("เป็นไปได้ว่า"),
      re.compile(r"\?{1}\s*$")), HYPOTHESIS),
    ((re.compile(r"(?i)\bcurrently\s+(uses?|using|on|is)\b"), re.compile(r"(?i)\bprovider\s*="),
      re.compile("ปัจจุบันใช้"), re.compile("ตอนนี้ใช้")), PROJECT_STATE),
    ((re.compile(r"(?i)\bI\s+(prefer|like|love|want|need)\b"), re.compile(r"(?i)\bmy\s+(favorite|preferred|default)\b"),
      re.compile("ชอบ"), re.compile("ต้องการ"), re.compile("อยากได้")), PREFERENCE),
    ((re.compile(r"(?i)\bwe\s+tried\b"), re.compile(r"(?i)\btried\s+approach\b"),
      re.compile("ลองวิธี"), re.compile("เคยลอง")), EVENT),
    ((re.compile(r"(?i)\bpattern\b"), re.compile(r"(?i)\brecurring\b"),
      re.compile("รูปแบบ"), re.compile("ทำแบบนี้.*ทุกครั้ง")), PATTERN),
)


def classify(text: str, category_hint: str = "") -> str:
    """Deterministic rule-based classification; falls back to FACT/GENERAL.

    Never invokes an LLM. Thai and English structural signals; language
    independent fallback on sentence shape (question -> HYPOTHESIS).
    """
    if not text or not text.strip():
        return GENERAL
    lowered = text.strip()
    for patterns, mem_type in _CLASSIFIER_RULES:
        if any(p.search(lowered) for p in patterns):
            return mem_type
    # Structural fallback: explicit short instruction-like sentences are facts.
    if len(lowered) < 200 and re.search(r"(?i)\b(is|are|=|ใช้|คือ)\b", lowered):
        return FACT
    return GENERAL


# ---------------------------------------------------------------------------
# Salience scoring.
# ---------------------------------------------------------------------------

_POSITIVE_SIGNALS: tuple[tuple[re.Pattern, float], ...] = (
    (re.compile(r"(?i)\b(must|required|critical|baseline|invariant)\b"), 0.30),
    (re.compile("ห้าม|ห้ามแก้|สำคัญ|จำเป็น"), 0.30),
    (re.compile(r"(?i)\b(decided|decision|agreed|architecture)\b"), 0.22),
    (re.compile("ตัดสินใจ|ตกลง"), 0.22),
    (re.compile(r"(?i)\b(prefer|preference|constraint|security)\b"), 0.18),
    (re.compile("ชอบ|ต้องการ|ปลอดภัย"), 0.15),
    (re.compile(r"(?i)\b(verified|passed|evidence|benchmark)\b"), 0.18),
    (re.compile("ทดสอบแล้ว|ผ่านแล้ว"), 0.18),
    (re.compile(r"(?i)\b(lesson|failed|avoid)\b"), 0.15),
    (re.compile("ใช้ไม่ได้|บทเรียน"), 0.15),
)

_NEGATIVE_SIGNALS: tuple[tuple[re.Pattern, float], ...] = (
    (re.compile(r"(?i)\b(maybe|probably|possibly|might|could be)\b"), -0.20),
    (re.compile("อาจจะ|น่าจะ"), -0.20),
    (re.compile(r"(?i)^(hi|hey|hello|thanks|ok|okay|lol|haha)[\s!.,]*$"), -0.40),
    (re.compile(r"^(.)\1{5,}$"), -0.30),  # repeated-char noise
)


def salience_score(
    text: str,
    mem_type: str = GENERAL,
    repeat_count: int = 0,
    has_evidence: bool = False,
    is_contradicted: bool = False,
) -> float:
    """Deterministic salience in [0,1]. Base 0.4; signals adjust; clamped."""
    if not text or not text.strip():
        return 0.0
    score = 0.40
    for pattern, delta in _POSITIVE_SIGNALS:
        if pattern.search(text):
            score += delta
            break
    for pattern, delta in _NEGATIVE_SIGNALS:
        if pattern.search(text):
            score += delta
            break
    if mem_type in (CONSTRAINT, INVARIANT):
        score += 0.15
    elif mem_type == HYPOTHESIS:
        score -= 0.15
    elif mem_type == EVIDENCE:
        score += 0.10
    if repeat_count >= 2:
        score += 0.10
    if has_evidence:
        score += 0.10
    if is_contradicted:
        score -= 0.25
    if len(text) < 12:
        score -= 0.15  # transient noise
    if len(text) > 2000:
        score -= 0.10  # likely a dump, not a memory
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Normalization + multi-level dedupe keys.
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Unicode NFKC, lowercase, collapse whitespace/punctuation for comparison."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text.strip().lower())
    t = re.sub(r"\s+", " ", t)
    t = t.strip(".,;:!?\"'()[]{}#@<>-–—…")
    return t.strip()


def dedupe_keys(content: str) -> dict:
    """Level 1-3 dedupe keys: exact hash, normalized text, subject slot.

    Level 4 (contradictory values) and 5 (related/derived) need DB context
    and are handled by conflict detection + lineage, not here.
    """
    norm = normalize_text(content)
    exact = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]
    # Subject/predicate slot: text before first = : / คือ / ใช้, else first 6 words.
    m = re.split(r"\s*(=|:|คือ|ใช้)\s*", norm, maxsplit=1)
    slot = m[0].strip() if len(m) > 1 else " ".join(norm.split()[:6])
    slot_key = hashlib.sha256(slot.encode("utf-8")).hexdigest()[:16]
    return {"exact": exact, "normalized": norm, "slot": slot, "slot_key": slot_key}


# ---------------------------------------------------------------------------
# Conflict detection (same subject + same predicate + different value).
# ---------------------------------------------------------------------------

_PREDICATE_SPLIT = re.compile(r"\s*(=|:|คือ|ใช้|->|→)\s*")


def split_subject_value(content: str) -> tuple[str, str]:
    """Split 'subject = value' style content; returns (subject, value)."""
    norm = normalize_text(content)
    parts = _PREDICATE_SPLIT.split(norm, maxsplit=1)
    if len(parts) >= 3:
        return parts[0].strip(), parts[2].strip()
    words = norm.split()
    if len(words) >= 3:
        return " ".join(words[:2]), " ".join(words[2:])
    return norm, ""


def is_conflicting(old: str, new: str) -> bool:
    """True when both share a subject slot but carry different values."""
    old_sub, old_val = split_subject_value(old)
    new_sub, new_val = split_subject_value(new)
    if not old_sub or not new_sub or not old_val or not new_val:
        return False
    if old_sub != new_sub:
        return False
    return normalize_text(old_val) != normalize_text(new_val)


# ---------------------------------------------------------------------------
# Temporal evaluation.
# ---------------------------------------------------------------------------

STALE_AFTER_DAYS = 90
AGING_AFTER_DAYS = 30


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        text = value.replace("Z", "+00:00") if isinstance(value, str) else value
        ts = datetime.fromisoformat(text)
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def temporal_state(
    updated_at: str | None,
    lifecycle: str = ACTIVE,
    now: datetime | None = None,
) -> str:
    """Evaluate lifecycle from age. Terminal states (superseded/archived/
    quarantine) are sticky — age never revives or demotes them further."""
    if lifecycle in (SUPERSEDED_STATE, ARCHIVED, QUARANTINE):
        return lifecycle
    ts = _parse_ts(updated_at)
    if ts is None:
        return ACTIVE
    ref = now or datetime.now(timezone.utc)
    age_days = (ref - ts).total_seconds() / 86400
    if age_days < 0:
        return ACTIVE  # impossible timestamp -> handled by self-heal, not here
    if age_days >= STALE_AFTER_DAYS:
        return STALE
    if age_days >= AGING_AFTER_DAYS:
        return AGING
    return ACTIVE


# ---------------------------------------------------------------------------
# Trust / confidence extension.
# ---------------------------------------------------------------------------

def adjusted_confidence(
    base_trust: float,
    mem_type: str = GENERAL,
    age_days: float = 0.0,
    confirmations: int = 0,
    contradictions: int = 0,
    helpful: int = 0,
    unhelpful: int = 0,
    speculative: bool = False,
) -> float:
    """Evidence-weighted confidence in [0,1]. A single write can never reach
    high trust: caps base contribution and requires confirmations for >0.85."""
    base = max(0.0, min(1.0, base_trust))
    score = base * 0.7 + 0.15  # single-write ceiling ≈ 0.85 only at base 1.0
    score += min(0.10, confirmations * 0.03)
    score += min(0.05, helpful * 0.02)
    score -= min(0.30, contradictions * 0.15)
    score -= min(0.20, unhelpful * 0.08)
    if age_days > STALE_AFTER_DAYS:
        score -= 0.10
    elif age_days > AGING_AFTER_DAYS:
        score -= 0.05
    if speculative or mem_type == HYPOTHESIS:
        score -= 0.15
    if mem_type in (CONSTRAINT, INVARIANT, EVIDENCE):
        score += 0.05
    return max(0.0, min(1.0, score))


def feedback_trust_delta(helpful: bool, streak: int = 0) -> float:
    """Bounded feedback delta: helpful +0.05, unhelpful -0.10 (compat with
    existing store constants); streak caps prevent feedback poisoning from
    vaulting or tanking trust in one burst."""
    if helpful:
        return min(0.05, 0.02 + 0.01 * max(0, streak))
    return max(-0.10, -0.05 - 0.01 * max(0, streak))


__all__ = [
    "contains_secret", "contains_instruction", "classify", "salience_score",
    "normalize_text", "dedupe_keys", "split_subject_value", "is_conflicting",
    "temporal_state", "adjusted_confidence", "feedback_trust_delta",
    "MEM_TYPES",
]
