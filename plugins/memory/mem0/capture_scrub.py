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

# ── Natural-language credential disclosure (the 2026-07-07 prune-leak class) ──────────────────
# The pattern set above only catches secrets with a `:`/`=` right after a label or a structurally
# distinctive token. Human sentences — "the sudo password is 'Zxcv...'", "password djt!K9x",
# "passphrase is 'correct horse battery staple'" — sailed through, and 22 such rows (plaintext
# RDP/sudo/usenet passwords, a recovery passphrase) were found live in the store on 2026-07-07.
# _scan_nl_credentials catches a credential WORD sitting near a value that looks like a real secret.
# Original author: Kyzcreig (fork PR #214); re-landed on current main with the Greptile review
# fixes folded in (wider value window, 2-word passphrase, 'reset to' connector).
_CRED_WORD = re.compile(r"(?i)\b(?:pass(?:word|code|phrase)|passwd|pwd)\b")
# The cred word as a NOUN MODIFIER ('password manager/app/field/policy/...') is not a disclosure.
# `reset(s)` is only a role noun when NOT followed by 'to' — 'the password reset flow' is a flow
# term, but 'the password reset to 84719263' is a phrased VALUE disclosure the _CRED_CONNECTOR
# must see (Greptile #393: a bare `reset` here silently negated the 'reset to' connector).
_CRED_ROLE_NOUN = re.compile(
    r"(?i)(?:manager|managers|app|apps|application|field|fields|box|prompt|step|screen|wall|"
    r"policy|policies|reset(?!\s+to\b)|resets(?!\s+to\b)|recovery|rotation|entry|entries|store|vault|hint|hints|"
    r"strength|length|requirement|requirements|protection|protected|authentication|form)\b"
)
_QUOTES = "'\"`\u2018\u2019\u201c\u201d"
_QUOTED_VAL = re.compile(r"[" + _QUOTES + r"]([^" + _QUOTES + r"]{5,64})[" + _QUOTES + r"]")
_BARE_TOKEN = re.compile(r"[^\s'\"`,;]{8,64}")
# A connector signalling a value follows ('password is X', 'password: X', 'reset to X', '= X').
# Greptile #214 (scrub:198) fix: include 'reset to'/'changed to'/'now'/'set to' so a phrased
# reset ('the router password has been reset to 84719263') is caught, not just 'is'/'was'/':'/'='.
_CRED_CONNECTOR = re.compile(
    r"(?i)(?:\bis\b|\bwas\b|\bset to\b|\breset to\b|\bchanged to\b|\bnow\b|[:=])\s*$"
)
# Quoted values that are product/label names, not secrets — never a disclosure.
_CRED_NONSECRET_WORDS = {
    "1password", "bitwarden", "lastpass", "dashlane", "keepass", "keepassxc", "keeper",
    "nordpass", "proton pass", "enpass", "vaultwarden", "authy", "none", "empty", "n/a",
}
_CRED_STOPWORDS = {"password", "passcode", "passphrase", "passwd", "pwd", "the", "a", "an"}


def _looks_like_password_value(token: str) -> bool:
    """A bare token that looks like a real password value: has a letter, and has a
    digit OR a strong symbol (so a plain dictionary word is NOT flagged, but 'djt!K9x' is).

    Called on two token shapes with different length floors: _BARE_TOKEN matches
    are already >=8 chars; individual WORDS of a quoted multi-word value can be
    shorter, so the >=6 floor here is the binding one for those (e.g. 'K9x!7a').
    """
    core = token.strip(".,;:!?)('\"`\u2018\u2019\u201c\u201d")
    if len(core) < 6:
        return False
    # Path-like tokens (op:// leftovers, ~/.ssh/id_ed25519, URLs) are never bare passwords.
    if "/" in core or "\\" in core:
        return False
    # Known product names ('1Password') are not secrets even though digit+letters.
    if core.lower() in _CRED_NONSECRET_WORDS:
        return False
    if not re.search(r"[A-Za-z]", core):
        return False
    # Require a digit or a STRONG symbol — a bare '-' is an identifier hyphen, not password-shaped
    # ('Plex-owned', 'nexus-command' must stay clean; 'FALCON-9-ORION-2287' has digits anyway).
    if not re.search(r"[0-9!@#$%^&*_+=]", core):
        return False
    # A hyphenated pure-alphabetic word in any case ('Plex-owned') is not a password.
    if re.fullmatch(r"[A-Za-z]+(?:-[A-Za-z]+)+", core):
        return False
    return True


def _scan_nl_credentials(text: str) -> List[str]:
    """Detect natural-language credential disclosure: a credential word near a value that looks
    like a real secret. Value shapes caught: a QUOTED value or a BARE secret-shaped token within a
    window after the cred word, plus a pure-digit PIN after a value connector.

    Greptile #214 review fixes folded in:
      - scrub:167 distant value: window widened 60 -> 160 chars (qualifier phrases like 'password
        for the backup router's admin console on the secondary VLAN is <secret>').
      - scrub:182 two-word passphrase: a quoted value of >=2 words counts (was >=3).
      - scrub:198 'reset to' connector: _CRED_CONNECTOR now matches phrased resets.

    Guards against false positives: cred word as a noun modifier ('password manager/field/...') and
    a quoted product name ('1Password', 'Bitwarden') are not disclosures.
    """
    hits: List[str] = []
    for m in _CRED_WORD.finditer(text):
        after = text[m.end(): m.end() + 12].lstrip()
        if _CRED_ROLE_NOUN.match(after):
            continue
        window = text[m.end(): m.end() + 160]
        matched = False
        # 1) quoted value in the window
        for qm in _QUOTED_VAL.finditer(window):
            val = qm.group(1).strip()
            low = val.lower()
            if low in _CRED_STOPWORDS or low in _CRED_NONSECRET_WORDS:
                continue
            words = val.split()
            # A multi-word quoted value (>=2 words) after a cred word is a passphrase; a
            # secret-shaped single token also counts; a lone quoted >=6-char word counts (quoting
            # a value right after 'password' is itself the disclosure signal).
            if (len(words) >= 2
                    or any(_looks_like_password_value(w) for w in words)
                    or (len(words) == 1 and len(val) >= 6)):
                hits.append("nl_credential_quoted")
                matched = True
                break
        if matched:
            continue
        # 2) bare secret-shaped token, or a pure-digit PIN after a value connector
        for bm in _BARE_TOKEN.finditer(window):
            tok = bm.group(0)
            if _looks_like_password_value(tok):
                hits.append("nl_credential_bare")
                break
            core = tok.strip(".,;:!?)('\"`\u2018\u2019\u201c\u201d")
            if core.isdigit() and len(core) >= 6:
                pre = window[: bm.start()]
                if _CRED_CONNECTOR.search(pre[-16:]):
                    hits.append("nl_credential_bare")
                    break
    return hits


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
    hits.extend(_scan_nl_credentials(scrubbed))
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
