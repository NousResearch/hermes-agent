"""Deterministic (non-LLM) secret scrubber for mem0 auto-capture (spec INV-4 / NB1).

The salience GATE (an LLM prompt) is NOT a reliable secret boundary — the gated experiment
leaked a bot token verbatim THROUGH the rubric. So a candidate extracted fact must pass a
DETERMINISTIC pattern scrubber before it is accepted for the store/recall surface. A fact that
trips a high-confidence secret pattern is DROPPED (returned as rejected), not stored.

This is intentionally conservative + fail-closed: a false positive drops one durable fact (the
deliberate `mem0_conclude` path is the belt-and-suspenders for anything auto missed); a false
negative writes a secret into a recalled store, which is the exfiltration risk we must not take.

Patterns mirror the fleet redaction set (doc-share privacy-scan + qmd_secrets lineage). Kept as
a standalone module so it is unit-testable and reusable by the A-full server-side seam too.
"""

from __future__ import annotations

import math
import re
from typing import List, Tuple

# High-confidence secret shapes — a hit here DROPS the fact (INV-4).
_SECRET_PATTERNS = [
    # OpenAI / Anthropic / OpenRouter style keys
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"), "openai_or_anthropic_key"),
    (re.compile(r"\bsk-ant-[A-Za-z0-9_-]{16,}\b"), "anthropic_key"),
    (re.compile(r"\bsk-or-v1-[A-Za-z0-9_-]{16,}\b"), "openrouter_key"),
    # GitHub tokens
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "github_token"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"), "github_pat"),
    # AWS
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "aws_access_key"),
    (re.compile(r"\bASIA[0-9A-Z]{16}\b"), "aws_sts_key"),
    # Google API key
    (re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), "google_api_key"),
    # Slack
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), "slack_token"),
    # Telegram bot token (digits:base64ish) — this is exactly what the experiment leaked
    (re.compile(r"\b\d{6,12}:[A-Za-z0-9_-]{30,}\b"), "telegram_bot_token"),
    # JWT (three base64url segments)
    (re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"), "jwt"),
    # PEM private key blocks
    (re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----"), "pem_private_key"),
    # Bearer header carrying a real-looking token
    (re.compile(r"\bBearer\s+[A-Za-z0-9._-]{20,}\b"), "bearer_token"),
    # Common connection strings with an inline password
    (re.compile(r"\b(?:postgres|postgresql|mysql|mongodb(?:\+srv)?|redis|amqp)://[^\s:@/]+:[^\s:@/]+@"), "conn_string_with_password"),
    # AWS secret-access-key-shaped assignment (40 char base64) next to a secret-y label
    (re.compile(r"(?i)\b(?:secret|token|passwd|password|api[_-]?key)\b\s*[:=]\s*['\"]?[A-Za-z0-9/+=_-]{20,}"), "labeled_secret_assignment"),
]

# A 1Password reference (op://...) is SAFE-BY-DESIGN — it's a pointer, not a secret. Never drop it.
_OP_REF = re.compile(r"\bop://[^\s'\"]+")


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    from collections import Counter
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in Counter(s).values())


def _looks_high_entropy_secret(token: str) -> bool:
    """A bare high-entropy blob (>=24 chars, mixed classes, entropy>~3.5) that isn't a word/path.
    Used only as a LOW-confidence supplementary signal, gated on multiple class requirements to
    avoid nuking normal identifiers/hashes-in-prose."""
    if len(token) < 24:
        return False
    if not re.search(r"[a-z]", token) or not re.search(r"[A-Z0-9]", token):
        return False
    if "/" in token or " " in token:  # paths / phrases are not bare secrets
        return False
    # must have digits AND letters AND (a symbol OR be long)
    has_digit = bool(re.search(r"\d", token))
    has_alpha = bool(re.search(r"[A-Za-z]", token))
    has_sym = bool(re.search(r"[_+=/-]", token))
    if not (has_digit and has_alpha and (has_sym or len(token) >= 32)):
        return False
    return _shannon_entropy(token) >= 3.5


def scan(text: str, *, entropy_check: bool = False) -> List[str]:
    """Return a list of matched secret-pattern names in `text`. Empty = clean.
    op:// references are stripped before scanning so they never count as a hit."""
    if not text:
        return []
    scrubbed = _OP_REF.sub("", text)
    hits: List[str] = []
    for pat, name in _SECRET_PATTERNS:
        if pat.search(scrubbed):
            hits.append(name)
    if entropy_check:
        for tok in re.findall(r"[A-Za-z0-9._+=/-]{24,}", scrubbed):
            if _looks_high_entropy_secret(tok):
                hits.append("high_entropy_blob")
                break
    return hits


def is_secret(text: str, *, entropy_check: bool = False) -> bool:
    return bool(scan(text, entropy_check=entropy_check))


def filter_facts(facts: List[str], *, entropy_check: bool = False) -> Tuple[List[str], List[dict]]:
    """Split extracted facts into (kept, dropped). A dropped fact carries the matched pattern
    names so the drop is auditable (logged as a count + reason, never the secret itself)."""
    kept: List[str] = []
    dropped: List[dict] = []
    for f in facts:
        hits = scan(f, entropy_check=entropy_check)
        if hits:
            dropped.append({"reason": ",".join(sorted(set(hits))), "len": len(f)})
        else:
            kept.append(f)
    return kept, dropped
