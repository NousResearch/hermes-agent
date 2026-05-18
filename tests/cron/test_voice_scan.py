"""Tests for the post-LLM voice-scan layer (Artemis B-0510-01 Phase 4b).

Voice-scan is a semantic enforcement layer: after the deterministic Phase 3
anti-pattern guard passes, scheduler calls a small OpenRouter LLM to judge
whether the briefing is in second-person voice or narrates the recipient in
third person. On a confident FAIL verdict the scheduler substitutes the
deterministic quiet-day fallback.

Tests monkeypatch the HTTP call — no real network. Two red cases drive the
fail path (verdict=FAIL → substitution); two green cases drive the pass path
(verdict=PASS → original text preserved). Fail-open paths (missing API key,
HTTP error, non-JSON response) round out the suite.
"""
import json

import pytest

from cron.scheduler import _voice_scan_check


# -----------------------------------------------------------------------------
# RED fixtures — third-person narration about the recipient. Must FAIL.
# -----------------------------------------------------------------------------

AMY_QUIET_DAY_NAME_THIRD = """Quiet day on the board — no roles match today's filter.

If Amy responds to the warm-intro question, I'll pivot the next briefing to
that thread."""

CRYSTAL_EXEC_THIRD_PERSON = """Crystal's positioning shift toward AI infra is
landing. Her CS + SWE positioning differentiates her from product-only
candidates — the technical depth she brings is the wedge.

Crystal should lead with the platform-engineering frame in her cover letter.
She requested the Andiamo packet last week and it's now drafted."""


# -----------------------------------------------------------------------------
# GREEN fixtures — second-person voice, recipient name in legit context. Must PASS.
# -----------------------------------------------------------------------------

MAGGIE_SECOND_PERSON_REAL_QUESTION = """The Andiamo intake closes Friday — let
me know if you're going to push the application through tonight or sleep on
it. I have your resume + the role brief ready either way."""

CRYSTAL_THIRD_PARTY_ENTITIES = """AIET 2026 in Zagreb is the highest-value
room for AI infra ICs in EU right now. Andiamo's Series B closed last week —
they're hiring 4 platform engineers. Reply with a yes and I'll draft the
outreach."""


# -----------------------------------------------------------------------------
# Helpers — monkeypatch urllib.request.urlopen to return a canned response.
# -----------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _make_fake_urlopen(verdict: str, offending: list[str] | None = None,
                       raise_exc: Exception | None = None,
                       content_override: str | None = None):
    def fake_urlopen(req, timeout=None):
        if raise_exc:
            raise raise_exc
        if content_override is not None:
            content = content_override
        else:
            content = json.dumps({
                "verdict": verdict,
                "offending_phrases": offending or [],
            })
        return _FakeResponse({
            "choices": [{"message": {"content": content}}],
        })
    return fake_urlopen


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_voice_scan_flags_amy_quiet_day_name_third(monkeypatch):
    """Amy 5/16 16:02 prod fixture — name in third-person conditional clause
    is the exact B-0510-01 Phase 4 day-1 regression."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    import urllib.request
    monkeypatch.setattr(
        urllib.request, "urlopen",
        _make_fake_urlopen("FAIL", ["if Amy responds"]),
    )
    clean, reason = _voice_scan_check(AMY_QUIET_DAY_NAME_THIRD, job_id="amy-test")
    assert clean is False
    assert "voice-scan FAIL" in reason
    assert "if Amy responds" in reason


def test_voice_scan_flags_crystal_executor_third_person(monkeypatch):
    """Crystal 5/16 Executor brief — possessive + third-person pronoun
    narration about the recipient."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    import urllib.request
    monkeypatch.setattr(
        urllib.request, "urlopen",
        _make_fake_urlopen("FAIL", ["Crystal's positioning shift", "Her CS + SWE", "She requested"]),
    )
    clean, reason = _voice_scan_check(CRYSTAL_EXEC_THIRD_PERSON, job_id="crystal-test")
    assert clean is False
    assert "voice-scan FAIL" in reason


def test_voice_scan_passes_second_person_with_recipient_name(monkeypatch):
    """Recipient name in legitimate second-person context — must NOT flag."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", _make_fake_urlopen("PASS"))
    clean, reason = _voice_scan_check(MAGGIE_SECOND_PERSON_REAL_QUESTION, job_id="maggie-test")
    assert clean is True
    assert reason == ""


def test_voice_scan_passes_third_party_entities(monkeypatch):
    """Third-party proper nouns (events, companies) — must NOT flag."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", _make_fake_urlopen("PASS"))
    clean, reason = _voice_scan_check(CRYSTAL_THIRD_PARTY_ENTITIES, job_id="crystal-clean-test")
    assert clean is True
    assert reason == ""


# -----------------------------------------------------------------------------
# Fail-open behavior — voice scan must NEVER block delivery on its own error.
# -----------------------------------------------------------------------------

def test_voice_scan_fail_open_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    clean, reason = _voice_scan_check("anything here", job_id="no-key-test")
    assert clean is True
    assert reason == ""


def test_voice_scan_fail_open_on_http_error(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    import urllib.request
    import urllib.error
    monkeypatch.setattr(
        urllib.request, "urlopen",
        _make_fake_urlopen("FAIL", raise_exc=urllib.error.URLError("connection refused")),
    )
    clean, reason = _voice_scan_check(AMY_QUIET_DAY_NAME_THIRD, job_id="http-err-test")
    assert clean is True
    assert reason == ""


def test_voice_scan_fail_open_on_non_json_content(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    import urllib.request
    monkeypatch.setattr(
        urllib.request, "urlopen",
        _make_fake_urlopen("FAIL", content_override="sorry, I cannot judge this"),
    )
    clean, reason = _voice_scan_check(AMY_QUIET_DAY_NAME_THIRD, job_id="bad-json-test")
    assert clean is True
    assert reason == ""


def test_voice_scan_disabled_via_env(monkeypatch):
    """VOICE_SCAN_ENABLED=0 short-circuits the check before any HTTP call."""
    monkeypatch.setenv("VOICE_SCAN_ENABLED", "0")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    # Intentionally no urlopen patch — would raise if called.
    clean, reason = _voice_scan_check(AMY_QUIET_DAY_NAME_THIRD, job_id="disabled-test")
    assert clean is True
    assert reason == ""


def test_voice_scan_empty_text_passes():
    clean, reason = _voice_scan_check("", job_id="empty-test")
    assert clean is True
    assert reason == ""
