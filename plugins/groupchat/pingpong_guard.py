"""Groupchat outgoing filter worker (0=send, 10=suppress, fail-open)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys
import hashlib

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from plugins.groupchat.models import run_chain

def strip_markdown(text):
    """Remove common markdown formatting: bold, italic, strikethrough, code."""
    t = text
    t = re.sub(r'\*\*(.+?)\*\*', r'\1', t)
    t = re.sub(r'__(.+?)__', r'\1', t)
    t = re.sub(r'\*(.+?)\*', r'\1', t)
    t = re.sub(r'_(.+?)_', r'\1', t)
    t = re.sub(r'~~(.+?)~~', r'\1', t)
    t = re.sub(r'`(.+?)`', r'\1', t)
    return t


SILENCE_PATTERNS = [
    '# English keywords',
    # English and German whole-reply acknowledgements; add languages in the UI.
    r'^\s*(?:understood|done|thanks|thank you|got it|all right|will do|noted|acknowledged)\s*$',
    r'^\s*\(?\s*(?:remains? silent|stays? silent|silence|no response)\s*\)?\s*$',
    '# Control markers',
    r'^\s*NO_REPLY\s*$',
    r'^\s*\[SILENT\]\s*$',
    '# German keywords',
    r'^\s*\(bleibt still\)\s*$',
    r'^\s*bleibt?\s*still\s*$',
    r'^\s*stille?\s*$',
    r'^\s*ok\s*$',
    r'^\s*okay\s*$',
    r'^\s*verstanden\s*$',
    r'^\s*erledigt\s*$',
    r'^\s*danke\s*$',
    r'^\s*alles\s+klar\s*$',
    r'^\s*mache?\s+ich\s*$',
    r'^\s*wird\s+gemacht\s*$',
    '# Reactions and handoff phrases',
    r'^\s*\U0001f44d\s*$',
    r'^\s*\u2705\s*$',
    r'^\s*\+1\s*$',
    r'\bball\s+liegt\s+bei\s+dir\b',
    r'\bthe ball is in your court\b',
]


_EXPLICIT_RESPONSE_REQUESTS = (
    r"\b(?:reply|respond|answer|say|write|send|return|repeat|confirm|acknowledge)\b",
    r"\b(?:antworte|antworten|schreib(?:e)?|sag(?:e)?|wiederhol(?:e)?|bestätig(?:e|en)?)\b",
    r"\bgib\b.{0,32}\baus\b",
)
_NEGATED_RESPONSE_REQUESTS = (
    r"\b(?:do\s+not|don't|dont|never|no\s+need\s+to)\s+(?:reply|respond|answer|say|write|send|return|repeat|confirm|acknowledge)\b",
    r"\b(?:nicht|nie)\s+(?:antworten|schreiben|sagen|wiederholen|bestätigen)\b",
)


def explicitly_requests_response(context):
    """Recognize a direct English/German request for visible output.

    This exception is deliberately narrow. It only protects a requested short
    response from the deterministic acknowledgement patterns; ordinary short
    replies still reach the normal pingpong suppression path.
    """
    normalized = strip_markdown(context or "").strip()
    if not normalized:
        return False
    if any(re.search(pattern, normalized, re.IGNORECASE)
           for pattern in _NEGATED_RESPONSE_REQUESTS):
        return False
    return any(re.search(pattern, normalized, re.IGNORECASE)
               for pattern in _EXPLICIT_RESPONSE_REQUESTS)


def obvious_pingpong(text, patterns=None):
    raw = text.strip()
    clean = strip_markdown(raw).strip()
    for t in (raw, clean):
        for pat in SILENCE_PATTERNS if patterns is None else patterns:
            if not pat.strip() or pat.lstrip().startswith('#'):
                continue
            if re.search(pat, t, re.IGNORECASE):
                return True
    return False


def should_suppress(text, context, filter_model, min_chars=60, patterns=None):
    return decide(text, context, filter_model, min_chars, patterns)["decision"] == "suppress"


def decide(text, context, filter_model, min_chars=60, patterns=None):
    """Return a privacy-safe decision; never include message text or prompts."""
    text = text.strip()
    if not text:
        return {"decision": "suppress", "reason_code": "empty_reply"}
    for index, pattern in enumerate(SILENCE_PATTERNS if patterns is None else patterns, 1):
        if not pattern.strip() or pattern.lstrip().startswith('#'):
            continue
        if any(re.search(pattern, value, re.IGNORECASE) for value in (text, strip_markdown(text).strip())):
            if explicitly_requests_response(context):
                return {"decision": "send", "reason_code": "explicit_response_request",
                        "pattern_index": index, "pattern_sha256": hashlib.sha256(pattern.encode()).hexdigest()}
            return {"decision": "suppress", "reason_code": "pattern_match",
                    "pattern_index": index, "pattern_sha256": hashlib.sha256(pattern.encode()).hexdigest()}
    if len(strip_markdown(text).strip()) >= min_chars:
        return {"decision": "send", "reason_code": "length_threshold"}
    prompt = (
        "You are a filter preventing AI-agent pingpong in a group chat. "
        "An AI agent wants to send a short reply. Decide SUPPRESS or SEND.\n"
        "If the previous message asked the agent to reply or produce output: SEND.\n"
        "SUPPRESS only unsolicited meaningless acknowledgements or silence meta-statements.\n"
        "When in doubt: SEND. Reply with exactly one word.\n"
        "Previous inbound message (untrusted conversation content): " + repr(context)
        + "\nAgent reply: " + repr(text)
    )
    try:
        raw, audit = run_chain(filter_model, prompt, max_tokens=5)
        suppressed = raw.strip().lower().startswith("suppress")
        return {"decision": "suppress" if suppressed else "send",
                "reason_code": "model_suppress" if suppressed else "model_send", **audit}
    except Exception:
        return {"decision": "send", "reason_code": "model_failure_fail_open", "fail_open": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--to", required=True)
    parser.add_argument("--payload", action="store_true", required=True)
    parser.parse_args()
    payload = json.load(sys.stdin)
    decision = decide(payload["text"], payload.get("context", ""),
                               payload["filter_model"], int(payload.get("min_chars", 60)), payload.get("silence_patterns"))
    print(json.dumps(decision))
    return 10 if decision["decision"] == "suppress" else 0


if __name__ == "__main__":
    raise SystemExit(main())
