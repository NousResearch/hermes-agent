"""Stock write-path + prefetch screening: secrets and injected instructions.

Memory content is DATA. Automatic memory formation (auto_extract, mirror)
must never turn a credential or an injected instruction into trusted memory,
and the context firewall must keep such rows out of model context.

Deliberate explicit writes (MemoryStore.add_fact / fact_store add tool) are
left untouched for backward compatibility; screening gates the AUTOMATIC
paths and the prefetch injection path.

False-positive policy: generic mentions ("JWT คือ JSON Web Token", "API key
goes in .env") must NOT match. Every secret pattern requires a token-like
shape (known prefix + entropy, PEM armor, or KEY=VALUE assignment).
"""

from __future__ import annotations

import re

_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\bsk-ant-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{8,}"),
    re.compile(r"\bghp_[A-Za-z0-9]{8,}"),
    re.compile(r"\bgsk_[A-Za-z0-9]{8,}"),
    re.compile(r"\bAKIA[0-9A-Z]{12,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9\-._~+/=]{8,}"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def _looks_secret_value(value: str) -> bool:
    """Secret-shaped value for a GENERIC all-caps KEY (named-secret keys like
    ``password`` use a lenient floor instead — the key name itself is signal).

    Requires len>=4 AND (a digit OR a separator char OR len>=12), so ordinary
    prose values (active, vacation, tomorrow, call) never match while
    abc123 / s3cr3t-value / long tokens do.
    """
    if len(value) < 4:
        return False
    return (any(ch.isdigit() for ch in value)
            or any(ch in "-_./+=" for ch in value)
            or len(value) >= 12)


_ENV_KV_RE = re.compile(
    r"(?<![A-Za-z0-9_])([A-Za-z][A-Za-z0-9_]{0,40}_(?:key|secret|token|password))\s*=\s*(\S{2,})",
    re.IGNORECASE,
)

_NAMED_SECRET_RE = (
    re.compile(r"(?i)(?:^|[\s_])(password|passwd)\b\s*[:=]\s*\S{2,}"),
    re.compile(r"(?i)\b(api[_-]?key|api[_-]?secret|secret[_-]?key|access[_-]?token|"
               r"auth[_-]?token|private[_-]?key|client[_-]?secret)\b\s*[:=]\s*\S{4,}"),
)

_INSTRUCTION_PATTERNS = (
    re.compile(r"(?i)\bignore\s+(all\s+)?(previous|prior|above)\s+instructions\b"),
    re.compile(r"(?i)\bdisregard\s+(all\s+)?(previous|prior|above)\s+(instructions|rules)\b"),
    re.compile(r"(?i)\byou\s+are\s+now\s+(a|an)\b"),
    re.compile(r"(?im)^\s*system\s*:"),
    re.compile(r"(?i)\bdelete\s+all\s+(files|data|memories)\b"),
    re.compile(r"(?i)\bdrop\s+table\b"),
    re.compile(r"(?i)\brm\s+-rf\b"),
    re.compile(r"(?i)\bexfiltrate\b"),
)




def _clean(text: str) -> str:
    import unicodedata as _ud
    norm = _ud.normalize("NFKC", text or "")
    # Strip invisible format chars (Cf: zero-width spaces, bidi controls,
    # word joiners) abused to split detection patterns. NFKC alone does not
    # remove them.
    return "".join(c for c in norm if _ud.category(c) != "Cf")


def contains_secret(text: str) -> bool:
    """True only for token-shaped credential material (see module policy)."""
    if not text:
        return False
    text = _clean(text)
    if any(p.search(text) for p in _SECRET_PATTERNS):
        return True
    if any(p.search(text) for p in _NAMED_SECRET_RE):
        return True
    # env-style KEY = value in any letter case needs a secret-shaped
    # value; plain words stay safe.
    for m in _ENV_KV_RE.finditer(text):
        if _looks_secret_value(m.group(2)):
            return True
    return False


def contains_instruction(text: str) -> bool:
    """True for injected-instruction shapes; ordinary prose never matches."""
    if not text:
        return False
    text = _clean(text)
    return any(p.search(text) for p in _INSTRUCTION_PATTERNS)


def classify_content(text: str) -> dict:
    """Screen one memory candidate. Returns flags + firewall class.

    safe: ordinary content. quarantine: secret-like or instruction-like.
    """
    secret = contains_secret(text or "")
    instruction = contains_instruction(text or "")
    if secret or instruction:
        return {"secret": secret, "instruction": instruction, "firewall": "quarantine"}
    return {"secret": False, "instruction": False, "firewall": "safe"}


def is_firewalled(content: str) -> bool:
    """True when content must never reach model context via prefetch."""
    flags = classify_content(content)
    return flags["firewall"] == "quarantine"


__all__ = ["contains_secret", "contains_instruction", "classify_content", "is_firewalled"]
