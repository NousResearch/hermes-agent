"""Structural egress guardrails (library engine).

NOTE: this engine and its ``DEFAULT_RULES`` serve *standalone* embeddings of
the durability layer (the ``OutboxWorker`` guardrail hook). Hermes's own
outbound platform boundary intentionally does NOT use these rules — it runs
``agent.redact.redact_sensitive_text`` via ``hermes_durability.egress`` so
there is exactly one authoritative secret-pattern set in the repo.

A mandatory middleware chain evaluated on every outbound envelope before any
network send: parse -> normalize -> detect -> redact/block/drop -> audit.

Normalization defeats the classic bypasses:
  * ANSI escape "glue" hiding secrets across styling sequences
  * zero-width characters splitting tokens
  * Unicode full-width / CJK-compatibility forms (NFKC folding)
Detection runs on the normalized text but redaction is applied to the
original string via span mapping, so legitimate formatting survives.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Optional

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
ZERO_WIDTH_RE = re.compile("[​‌‍⁠﻿]")

DEFAULT_RULES = [
    # (policy_id, pattern, action)
    ("secret.openai-key", r"sk-[A-Za-z0-9_-]{20,}", "redact"),
    ("secret.anthropic-key", r"sk-ant-[A-Za-z0-9_-]{20,}", "redact"),
    ("secret.aws-access-key", r"\bAKIA[0-9A-Z]{16}\b", "redact"),
    ("secret.github-token", r"\bgh[pousr]_[A-Za-z0-9]{20,}\b", "redact"),
    ("secret.slack-token", r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b", "redact"),
    ("secret.private-key-block", r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "block"),
    ("secret.jwt", r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b", "redact"),
    ("secret.cookie-header", r"(?im)^(?:set-)?cookie\s*:\s*\S+", "redact"),
    ("secret.env-assignment",
     r"(?im)\b(?:[A-Z][A-Z0-9_]*_)?(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?)\s*=\s*['\"]?[^\s'\"]{8,}", "redact"),
    ("secret.bearer", r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{16,}=*", "redact"),
]

REDACTION = "[REDACTED:{policy}]"


@dataclass
class Envelope:
    session_id: str
    channel: str
    payload: dict          # must contain "text" (str) for text channels
    outbox_id: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class Verdict:
    action: str            # "allow" | "redact" | "block" | "drop"
    envelope: Optional[Envelope]
    matched_policies: list[str] = field(default_factory=list)


class Rule:
    def __init__(self, policy_id: str, pattern: str, action: str):
        assert action in ("redact", "block", "drop")
        self.policy_id = policy_id
        self.pattern = re.compile(pattern)
        self.action = action


class Guardrail:
    """Pure evaluation engine plus durable audit hook.

    audit_sink(payload_id, session_id, action, policy_id, envelope_hash)
    is called (and must persist) BEFORE the verdict is returned, so the
    audit trail exists before any send is attempted.
    """

    def __init__(self, rules: Optional[list[Rule]] = None,
                 audit_sink: Optional[Callable[..., None]] = None):
        self.rules = rules if rules is not None else [Rule(*r) for r in DEFAULT_RULES]
        self.audit_sink = audit_sink

    def load_policy_dicts(self, dicts: list[dict]) -> None:
        """Hot-reload: replace the rule set atomically."""
        self.rules = [Rule(d["id"], d["pattern"], d["action"]) for d in dicts]

    # -- normalization -----------------------------------------------------
    @staticmethod
    def normalize(text: str) -> tuple[str, list[int]]:
        """Strip ANSI + zero-width chars and NFKC-fold, returning the
        normalized text and a map: normalized index -> original index."""
        kept: list[tuple[str, int]] = []
        i = 0
        while i < len(text):
            m = ANSI_RE.match(text, i) or ZERO_WIDTH_RE.match(text, i)
            if m:
                i = m.end()
                continue
            kept.append((text[i], i))
            i += 1
        norm_chars: list[str] = []
        idx_map: list[int] = []
        for ch, orig_i in kept:
            folded = unicodedata.normalize("NFKC", ch)
            for fch in folded:
                norm_chars.append(fch)
                idx_map.append(orig_i)
        return "".join(norm_chars), idx_map

    # -- evaluation --------------------------------------------------------
    def evaluate(self, envelope: Envelope) -> Verdict:
        text = envelope.payload.get("text")
        if not isinstance(text, str):
            text = json.dumps(envelope.payload, ensure_ascii=False)
            structured = True
        else:
            structured = False

        norm, idx_map = self.normalize(text)
        matches: list[tuple[Rule, int, int]] = []
        for rule in self.rules:
            for m in rule.pattern.finditer(norm):
                matches.append((rule, m.start(), m.end()))

        matched_ids = sorted({r.policy_id for r, _, _ in matches})
        envelope_hash = hashlib.sha256(text.encode()).digest()

        def audit(action: str, policy_id: str) -> None:
            if self.audit_sink:
                self.audit_sink(envelope.outbox_id or "-", envelope.session_id,
                                action, policy_id, envelope_hash)

        for rule, _, _ in matches:
            if rule.action == "block":
                audit("block", rule.policy_id)
                return Verdict("block", None, matched_ids)
        for rule, _, _ in matches:
            if rule.action == "drop":
                audit("drop", rule.policy_id)
                return Verdict("drop", None, matched_ids)

        if not matches:
            return Verdict("allow", envelope, [])

        # Redact: map normalized spans back to original offsets.
        spans: list[tuple[int, int, str]] = []
        for rule, s, e in matches:
            if rule.action != "redact":
                continue
            orig_s = idx_map[s] if s < len(idx_map) else len(text)
            orig_e = (idx_map[e - 1] + 1) if e - 1 < len(idx_map) else len(text)
            spans.append((orig_s, orig_e, rule.policy_id))
        spans.sort()
        merged: list[tuple[int, int, str]] = []
        for s, e, pid in spans:
            if merged and s <= merged[-1][1]:
                ps, pe, ppid = merged[-1]
                merged[-1] = (ps, max(pe, e), ppid)
            else:
                merged.append((s, e, pid))
        out: list[str] = []
        cursor = 0
        for s, e, pid in merged:
            out.append(text[cursor:s])
            out.append(REDACTION.format(policy=pid))
            audit("redact", pid)
            cursor = e
        out.append(text[cursor:])
        redacted_text = "".join(out)

        new_payload = dict(envelope.payload)
        if structured:
            new_payload = {"text": redacted_text}
        else:
            new_payload["text"] = redacted_text
        return Verdict("redact",
                       Envelope(envelope.session_id, envelope.channel,
                                new_payload, envelope.outbox_id, envelope.meta),
                       matched_ids)
