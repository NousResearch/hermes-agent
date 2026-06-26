"""Export redaction pipeline tests — the security-critical layer.

Invariants:
  * Secrets ALWAYS stripped, every mode, no flag disables it.
  * Content gated by telemetry.trajectories, not a redaction mode.
  * PII stripped in 'pii' mode; structure preserved (codec-aware).
"""

from __future__ import annotations

import json

from agent.telemetry import redaction as R


# ── secrets are always redacted ─────────────────────────────────────────────
def test_secrets_stripped_in_none_mode():
    text = "here is sk-ant-api03-SECRETKEY123 and a token"
    out = R.redact_for_export(text, content_mode=R.CONTENT_NONE)
    assert "SECRETKEY123" not in out


def test_secrets_stripped_in_pii_mode():
    text = "Authorization: Bearer abcdef123456789secret"
    out = R.redact_for_export(text, content_mode=R.CONTENT_PII)
    assert "abcdef123456789secret" not in out


def test_secret_redactor_fails_closed(monkeypatch):
    # If the underlying redactor raises, we must NOT return the raw string.
    import agent.redact as ar
    monkeypatch.setattr(ar, "redact_sensitive_text", lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    out = R.redact_for_export("sk-secret-value", content_mode=R.CONTENT_NONE)
    assert "sk-secret-value" not in out
    assert out == "[redaction-unavailable]"


# ── PII ─────────────────────────────────────────────────────────────────────
def test_pii_mode_strips_email_and_phone():
    text = "contact alice@example.com or +1 415 555 1234"
    out = R.redact_for_export(text, content_mode=R.CONTENT_PII)
    assert "alice@example.com" not in out
    assert "[email]" in out
    assert "555" not in out or "[phone]" in out


def test_none_mode_keeps_nonsecret_text_but_drops_via_message_path():
    # redact_for_export(none) scrubs secrets but doesn't strip ordinary words;
    # content *dropping* happens at the message layer (trajectories gate).
    out = R.redact_for_export("just ordinary words", content_mode=R.CONTENT_NONE)
    assert "ordinary" in out


# ── trajectories gate (content_export_enabled) ──────────────────────────────
def test_content_export_disabled_by_default():
    assert R.content_export_enabled({}) is False
    assert R.content_export_enabled({"telemetry": {}}) is False
    assert R.content_export_enabled({"telemetry": {"trajectories": {"enabled": False}}}) is False


def test_content_export_enabled_when_trajectories_on():
    assert R.content_export_enabled({"telemetry": {"trajectories": {"enabled": True}}}) is True


# ── codec-aware message redaction ───────────────────────────────────────────
def test_message_structural_only_when_content_excluded():
    msg = {"role": "user", "content": "my email is bob@x.com and key sk-12345"}
    out = R.redact_message(msg, include_content=False)
    assert out["role"] == "user"
    assert "content" not in out          # body dropped entirely
    assert out["content_chars"] == len(msg["content"])  # only the size remains
    assert "bob@x.com" not in json.dumps(out)


def test_message_content_included_is_redacted():
    msg = {"role": "user", "content": "email bob@x.com secret sk-ant-SECRET999"}
    out = R.redact_message(msg, content_mode=R.CONTENT_PII, include_content=True)
    assert "content" in out
    assert "SECRET999" not in out["content"]    # secret gone
    assert "bob@x.com" not in out["content"]    # pii gone
    assert "[email]" in out["content"]


def test_tool_calls_redacted_names_kept_args_scrubbed():
    msg = {
        "role": "assistant",
        "tool_calls": json.dumps([
            {"function": {"name": "web_search", "arguments": '{"q": "email me at z@z.com"}'}}
        ]),
    }
    out = R.redact_message(msg, content_mode=R.CONTENT_PII, include_content=True)
    tc = out["tool_calls"]
    assert tc[0]["name"] == "web_search"        # structure/name preserved
    assert "z@z.com" not in json.dumps(tc)      # arg pii scrubbed


def test_tool_calls_counted_when_content_excluded():
    msg = {
        "role": "assistant",
        "tool_calls": json.dumps([
            {"function": {"name": "a", "arguments": "{}"}},
            {"function": {"name": "b", "arguments": "{}"}},
        ]),
    }
    out = R.redact_message(msg, include_content=False)
    assert out["tool_call_count"] == 2
    assert "tool_calls" not in out


def test_content_mode_for_reads_config():
    assert R.content_mode_for({"telemetry": {"content_redaction": "pii"}}) == "pii"
    assert R.content_mode_for({"telemetry": {"content_redaction": "bogus"}}) == "none"
    assert R.content_mode_for({}) == "none"
