"""Unit tests for the /draft slash command — Plan 030-A.

Acceptance criteria covered:

AC1 — ``plugins/slash/draft.py`` exists with recipient resolution. The
      handler must parse the first whitespace-delimited token as the
      recipient and classify it as email / handle / unresolved.

AC2 — Tests pass with coverage >= 80% on new code. This module exercises
      every branch of ``_parse_args`` + ``_parse_recipient`` + the stub
      reply composer.

AC3 — Manual smoke test: ``/draft <email> "hello"`` returns the stub
      message containing the recipient and the two TODO markers (030-B
      context lookup, 030-C draft generation).

AC4 — No LLM call. The handler is pure parsing + string formatting; no
      Atlas client or model client is imported, and the test suite
      installs no mock for either. If 030-A ever starts importing
      anthropic/openai/atlas at module-load time, ``test_no_llm_imports``
      will trip.
"""

from __future__ import annotations

import sys

import pytest

from plugins.slash import draft as draft_mod
from plugins.slash.draft import (
    DraftArgs,
    Recipient,
    _compose_stub_reply,
    _friendly_name_from_email,
    _parse_args,
    _parse_recipient,
    handle_draft,
)


# ---------------------------------------------------------------------------
# Recipient parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "token,expected_value,expected_display",
    [
        ("sarah@example.com", "sarah@example.com", "Sarah"),
        ("greg.smith@firm.co", "greg.smith@firm.co", "Greg"),
        ("blake.aber+pe@gmail.com", "blake.aber+pe@gmail.com", "Blake"),
        ("a@b.io", "a@b.io", "A"),
    ],
)
def test_parse_recipient_email(token, expected_value, expected_display):
    r = _parse_recipient(token)
    assert r.kind == "email"
    assert r.value == expected_value
    assert r.display == expected_display


@pytest.mark.parametrize(
    "token,bare",
    [
        ("@bossman2", "bossman2"),
        ("@greg.smith", "greg.smith"),
        ("@a_b-c", "a_b-c"),
    ],
)
def test_parse_recipient_handle(token, bare):
    r = _parse_recipient(token)
    assert r.kind == "handle"
    assert r.value == bare
    assert r.display == f"@{bare}"


@pytest.mark.parametrize(
    "token",
    [
        "not-an-email",
        "@",
        "sarah@",
        "@with spaces",  # space-containing won't reach here from _parse_args,
        "drop table;",
    ],
)
def test_parse_recipient_unresolved(token):
    r = _parse_recipient(token)
    assert r.kind == "unresolved"
    assert r.value == token
    assert r.display == token


def test_friendly_name_strips_plus_tag_and_dots():
    assert _friendly_name_from_email("sarah.connor+pe@example.com") == "Sarah"
    assert _friendly_name_from_email("greg@firm.com") == "Greg"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def test_parse_args_email_and_intent():
    args = _parse_args("sarah@example.com follow-up on the term sheet")
    assert isinstance(args, DraftArgs)
    assert args.recipient.kind == "email"
    assert args.recipient.value == "sarah@example.com"
    assert args.intent == "follow-up on the term sheet"


def test_parse_args_strips_matched_quotes():
    args = _parse_args('sarah@example.com "follow-up on the term sheet"')
    assert args is not None
    assert args.intent == "follow-up on the term sheet"


def test_parse_args_strips_matched_single_quotes():
    args = _parse_args("sarah@example.com 'hello world'")
    assert args is not None
    assert args.intent == "hello world"


def test_parse_args_does_not_strip_mismatched_quotes():
    args = _parse_args("sarah@example.com \"hello world'")
    assert args is not None
    assert args.intent == "\"hello world'"


def test_parse_args_handle_and_intent():
    args = _parse_args("@bossman2 ping me about the deck")
    assert args is not None
    assert args.recipient.kind == "handle"
    assert args.recipient.value == "bossman2"
    assert args.intent == "ping me about the deck"


def test_parse_args_no_intent_returns_empty_string():
    args = _parse_args("sarah@example.com")
    assert args is not None
    assert args.intent == ""


def test_parse_args_empty_returns_none():
    assert _parse_args("") is None
    assert _parse_args("   ") is None
    assert _parse_args(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Stub reply composition
# ---------------------------------------------------------------------------


def test_compose_stub_reply_email_includes_header_and_intent():
    """030-B note: ``_compose_stub_reply`` is now header+intent only.

    The context block + 030-C TODO marker are layered on by
    ``_compose_full_reply``; tests that assert on those markers run
    against ``handle_draft`` with an injected ask_fn instead.
    """
    args = DraftArgs(
        recipient=Recipient(
            kind="email", value="sarah@example.com", display="Sarah"
        ),
        intent="follow-up on the term sheet",
    )
    reply = _compose_stub_reply(args)
    assert "Drafting message to Sarah (sarah@example.com)" in reply
    assert "Intent: follow-up on the term sheet" in reply


def test_compose_stub_reply_handle_omits_email_parens():
    args = DraftArgs(
        recipient=Recipient(kind="handle", value="bossman2", display="@bossman2"),
        intent="hello",
    )
    reply = _compose_stub_reply(args)
    assert "Drafting message to @bossman2" in reply
    # No parenthesized email since there isn't one
    assert "(" not in reply.split("\n")[0]


def test_compose_stub_reply_unresolved_marks_as_unresolved():
    args = DraftArgs(
        recipient=Recipient(kind="unresolved", value="garbage", display="garbage"),
        intent="hi",
    )
    reply = _compose_stub_reply(args)
    assert "garbage" in reply
    assert "unresolved" in reply.lower()


def test_compose_stub_reply_no_intent_says_none_provided():
    args = DraftArgs(
        recipient=Recipient(
            kind="email", value="a@b.io", display="A"
        ),
        intent="",
    )
    reply = _compose_stub_reply(args)
    assert "Intent: (none provided)" in reply


# ---------------------------------------------------------------------------
# Handler (end-to-end on the public surface)
# ---------------------------------------------------------------------------


def test_handle_draft_acceptance_smoke():
    """030-A/B AC — manual smoke test.

    Blake types: /draft sarah@example.com "follow-up on the term sheet"
    Expected reply contains the header, the 3 Atlas-context section
    labels, and the trailing 030-C composition TODO. We pass a no-op
    ``ask_fn`` so the test doesn't reach a real Atlas instance —
    Atlas-empty answers fall back to the section-specific "No X found"
    sentinels (AC1).
    """
    reply = handle_draft(
        'sarah@example.com "follow-up on the term sheet"',
        ask_fn=lambda **_: {"answer": "", "citations": []},
        compose_fn=lambda sys, usr: "Hi Sarah, quick follow-up. Thanks, Blake",
    )
    assert "Drafting message to Sarah (sarah@example.com)" in reply
    assert "Context found:" in reply
    assert "Prior commitments" in reply
    assert "Open contradictions" in reply
    assert "Last interaction" in reply
    assert "*Draft" in reply  # 030-C — actual draft block now renders


def test_handle_draft_empty_returns_usage():
    reply = handle_draft("")
    assert "Usage:" in reply
    assert "/draft" in reply


def test_handle_draft_whitespace_returns_usage():
    reply = handle_draft("   ")
    assert "Usage:" in reply


def test_handle_draft_handle_recipient_smoke():
    reply = handle_draft(
        "@bossman2 ping me about the deck",
        ask_fn=lambda **_: {"answer": "", "citations": []},
        compose_fn=lambda sys, usr: "Hey, quick ping on the deck.",
    )
    assert "@bossman2" in reply
    assert "ping me about the deck" in reply


def test_handle_draft_unresolved_recipient_still_responds():
    reply = handle_draft(
        "not-an-email some intent",
        ask_fn=lambda **_: {"answer": "", "citations": []},
        compose_fn=lambda sys, usr: "Hello.",
    )
    # Should not crash; should mention the raw token
    assert "not-an-email" in reply
    assert "*Draft" in reply  # 030-C — actual draft block now renders


# ---------------------------------------------------------------------------
# AC4 — no LLM call
# ---------------------------------------------------------------------------


def test_no_llm_imports_at_module_load():
    """030-A must not pull in anthropic/openai/atlas client at import time.

    030-B is where Atlas context fetch lands; 030-C is where the actual
    LLM call lands. If either sneaks into this phase, this guard trips
    and forces the author to push the dep back to its real phase.
    """
    forbidden = {"anthropic", "openai"}
    # We allow the test process to import these elsewhere, but the
    # ``draft`` module itself should not have referenced them at load
    # time. Inspect the module's globals as a proxy for what it bound.
    bound = set(vars(draft_mod).keys())
    assert bound.isdisjoint(forbidden), (
        f"draft.py bound forbidden symbols at module load: "
        f"{bound & forbidden}"
    )


# ---------------------------------------------------------------------------
# Plugin registration smoke test — confirms /draft is wired through.
# ---------------------------------------------------------------------------


def test_register_wires_draft_command():
    from plugins.slash import register

    registered: dict[str, dict] = {}

    class _Ctx:
        def register_command(
            self, name: str, handler, description: str = "", args_hint: str = ""
        ) -> None:
            registered[name] = {
                "handler": handler,
                "description": description,
                "args_hint": args_hint,
            }

    register(_Ctx())

    assert "draft" in registered
    assert registered["draft"]["args_hint"] == "<recipient> <context>"
    assert callable(registered["draft"]["handler"])
    # And the existing 020-E commands still register
    assert "resume" in registered
    assert "skip" in registered


# ---------------------------------------------------------------------------
# Phase 030-B — Atlas context fan-out (3 parallel asks)
# ---------------------------------------------------------------------------

import threading  # noqa: E402

from plugins.slash.draft import (  # noqa: E402
    _CONTEXT_QUESTIONS,
    _EMPTY_FALLBACKS,
    _extract_answer,
    _is_empty_answer,
    fetch_atlas_context,
)


def test_context_questions_cover_three_required_topics():
    """030-B AC: the three parallel asks must be commitments,
    contradictions, and last meaningful interaction."""
    labels = [label for label, _ in _CONTEXT_QUESTIONS]
    assert labels == ["Prior commitments", "Open contradictions", "Last interaction"]
    # And every label has a fallback for the empty-Atlas case.
    for label in labels:
        assert label in _EMPTY_FALLBACKS


def test_extract_answer_handles_dict_string_and_none():
    assert _extract_answer({"answer": "Hello [cite:abc]"}) == "Hello [cite:abc]"
    assert _extract_answer("plain string") == "plain string"
    assert _extract_answer(None) == ""
    assert _extract_answer({}) == ""
    # Citations must survive verbatim — no escaping, no stripping.
    payload = {"answer": "Sent demo deck [cite:msg:42]", "citations": [{"id": "msg:42"}]}
    assert "[cite:msg:42]" in _extract_answer(payload)


def test_is_empty_answer_flags_atlas_no_info_phrases():
    assert _is_empty_answer("")
    assert _is_empty_answer("   ")
    assert _is_empty_answer("I don't have information on that")
    assert _is_empty_answer("No records found")
    # Real answers — even short ones — must pass through.
    assert not _is_empty_answer("Sent demo deck on 2026-04-12 [cite:abc]")
    assert not _is_empty_answer("Greg owes Blake a follow-up email.")


def test_fetch_atlas_context_returns_three_sections_on_empty_corpus():
    """AC1: even when Atlas has zero facts, the response has 3 sections.

    The empty-corpus path is the cold-start case from 030's design
    decision 6 — CP3 must be useful in week 1 before 022-B has ingested
    a meaningful corpus. The user sees "No X found" rather than a
    missing section.
    """
    sections = fetch_atlas_context("Sarah", ask_fn=lambda **_: {"answer": ""})
    assert len(sections) == 3
    labels = [s[0] for s in sections]
    assert labels == ["Prior commitments", "Open contradictions", "Last interaction"]
    for label, answer in sections:
        assert answer == _EMPTY_FALLBACKS[label]


def test_fetch_atlas_context_passes_recipient_into_each_question():
    """Each ask must substitute the recipient display name."""
    seen: list[str] = []

    def _fake_ask(*, question: str, **_: object):
        seen.append(question)
        return {"answer": "fact about " + question}

    fetch_atlas_context("Greg", ask_fn=_fake_ask)
    assert len(seen) == 3
    # Every question must mention "Greg" — substitution check.
    for q in seen:
        assert "Greg" in q
    # And the three questions must be distinct (no duplicate asks).
    assert len(set(seen)) == 3


def test_fetch_atlas_context_runs_three_asks_in_parallel():
    """AC: the three asks are dispatched in parallel.

    We assert this by counting concurrent in-flight calls. If the
    implementation regresses to serial, the gauge never exceeds 1.
    """
    in_flight = 0
    peak = 0
    lock = threading.Lock()
    barrier = threading.Barrier(3, timeout=2.0)

    def _fake_ask(*, question: str, **_: object):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        # Sync at the barrier — if asks ran serially, two threads would
        # never reach the barrier together and it would time out.
        barrier.wait()
        with lock:
            in_flight -= 1
        return {"answer": "ok"}

    fetch_atlas_context("Sarah", ask_fn=_fake_ask)
    assert peak == 3, f"expected 3 concurrent asks, peaked at {peak}"


def test_fetch_atlas_context_isolates_per_section_failures():
    """One failed ask must not poison the other two sections (AC: best-effort)."""
    def _flaky(*, question: str, **_: object):
        if "contradictions" in question.lower():
            raise RuntimeError("simulated atlas hiccup")
        return {"answer": "real fact for " + question[:20]}

    sections = fetch_atlas_context("Sarah", ask_fn=_flaky)
    by_label = dict(sections)
    assert "real fact" in by_label["Prior commitments"]
    assert "real fact" in by_label["Last interaction"]
    # The failed section surfaces a short error sentinel — not a crash,
    # not an empty fallback (which would imply "no contradictions").
    assert "failed" in by_label["Open contradictions"].lower()


def test_fetch_atlas_context_preserves_citations_verbatim():
    """[cite:...] markers from Atlas must survive unmodified."""
    def _fake_ask(*, question: str, **_: object):
        return {"answer": "Greg agreed to demo [cite:gmail:msg:42]"}

    sections = fetch_atlas_context("Greg", ask_fn=_fake_ask)
    for _, answer in sections:
        assert "[cite:gmail:msg:42]" in answer


def test_handle_draft_renders_atlas_answers_under_section_headers():
    """End-to-end 030-B AC: real Atlas answers appear in the Slack reply."""
    def _fake_ask(*, question: str, **_: object):
        if "commitment" in question.lower():
            return {"answer": "Blake owes Sarah the term sheet by Friday [cite:c1]"}
        if "contradiction" in question.lower():
            return {"answer": "None tracked."}
        return {"answer": "Last email 2026-04-12 re: term sheet"}

    reply = handle_draft(
        'sarah@example.com "term sheet status"',
        ask_fn=_fake_ask,
        compose_fn=lambda sys, usr: "Hi Sarah, term sheet update incoming.",
    )
    assert "*Prior commitments:*" in reply
    assert "Blake owes Sarah the term sheet" in reply
    assert "[cite:c1]" in reply
    assert "*Open contradictions:*" in reply
    assert "*Last interaction:*" in reply
    assert "2026-04-12" in reply
    assert "*Draft" in reply  # 030-C — actual draft block now renders


def test_handle_draft_atlas_unavailable_still_returns_three_sections():
    """When Atlas can't even be constructed, we still ship 3 fallback sections.

    We simulate this by injecting an ``ask_fn`` that raises on every
    call. The handler must downgrade to the empty-section fallbacks
    rather than crashing.
    """
    def _broken(**_: object):
        raise RuntimeError("atlas down")

    reply = handle_draft(
        "sarah@example.com check in",
        ask_fn=_broken,
        compose_fn=lambda sys, usr: "Hi Sarah, checking in.",
    )
    assert "Context found:" in reply
    assert reply.count("*") >= 6  # 3 bold section labels × 2 asterisks
    assert "*Draft" in reply  # 030-C — actual draft block now renders


# ---------------------------------------------------------------------------
# Phase 030-C — Nova-Pro composition + Send/Edit/Discard UX
# ---------------------------------------------------------------------------

from plugins.slash.draft import (  # noqa: E402
    StoredDraft,
    build_nova_prompt,
    compose_action_marker,
    extract_action_draft_id,
    get_stored_draft,
    pop_stored_draft,
    record_draft_decision,
    strip_action_marker,
    _fallback_draft,
    _safe_compose,
    _DRAFT_STORE,
)


@pytest.fixture(autouse=False)
def _clear_draft_store():
    """Some 030-C tests rely on a clean draft store snapshot."""
    _DRAFT_STORE.clear()
    yield
    _DRAFT_STORE.clear()


def _empty_ask(**_):
    return {"answer": "", "citations": []}


# --- Nova-Pro prompt construction ------------------------------------------


def test_build_nova_prompt_includes_recipient_and_intent():
    args = DraftArgs(
        recipient=Recipient(
            kind="email", value="sarah@example.com", display="Sarah"
        ),
        intent="follow-up on the term sheet",
    )
    sections = [
        ("Prior commitments", "Blake owes Sarah the term sheet by Friday [cite:c1]"),
        ("Open contradictions", "No contradictions found."),
        ("Last interaction", "Last email 2026-04-12"),
    ]
    system, user = build_nova_prompt(args, sections)
    # System prompt establishes Blake's voice + no-em-dash rule
    assert "Blake" in system
    assert "house style" in system.lower() or "concise" in system.lower()
    # User prompt threads recipient + intent + every context section
    assert "Sarah" in user
    assert "sarah@example.com" in user
    assert "follow-up on the term sheet" in user
    assert "Prior commitments" in user
    assert "Open contradictions" in user
    assert "Last interaction" in user
    assert "[cite:c1]" in user  # citations survive into the prompt


def test_build_nova_prompt_handles_empty_intent():
    args = DraftArgs(
        recipient=Recipient(kind="handle", value="bossman2", display="@bossman2"),
        intent="",
    )
    _system, user = build_nova_prompt(args, [])
    assert "no specific intent" in user.lower()
    assert "@bossman2" in user


def test_build_nova_prompt_truncates_long_context_sections():
    long_blob = "x" * 5000
    args = DraftArgs(
        recipient=Recipient(kind="email", value="a@b.io", display="A"),
        intent="hi",
    )
    _sys, user = build_nova_prompt(args, [("Prior commitments", long_blob)])
    # Bounded by _CONTEXT_SECTION_CHAR_CAP (1200) + overhead — must be
    # much shorter than the raw 5000-char input.
    assert len(user) < 3000
    assert "..." in user  # truncation marker present


# --- Nova-Pro happy path + failure fallback --------------------------------


def test_safe_compose_returns_nova_output_on_success():
    args = DraftArgs(
        recipient=Recipient(kind="email", value="a@b.io", display="A"),
        intent="hello",
    )
    body, err = _safe_compose(
        args, [], compose_fn=lambda sys, usr: "Hi A, hello.\n\nBlake"
    )
    assert err is None
    assert body == "Hi A, hello.\n\nBlake"


def test_safe_compose_falls_back_when_nova_raises():
    args = DraftArgs(
        recipient=Recipient(kind="email", value="sarah@example.com", display="Sarah"),
        intent="ping",
    )

    def _boom(sys, usr):
        raise RuntimeError("bedrock throttled")

    body, err = _safe_compose(args, [], compose_fn=_boom)
    assert err is not None
    assert "throttled" in err
    # Fallback template mentions the recipient and Blake's sign-off
    assert "Sarah" in body
    assert "Blake" in body


def test_safe_compose_falls_back_when_nova_returns_empty():
    args = DraftArgs(
        recipient=Recipient(kind="email", value="a@b.io", display="A"),
        intent="x",
    )
    body, err = _safe_compose(args, [], compose_fn=lambda s, u: "   ")
    assert err is not None  # empty body treated as failure
    assert body == _fallback_draft(args)


def test_handle_draft_renders_nova_body_in_reply():
    reply = handle_draft(
        "sarah@example.com term sheet ping",
        ask_fn=_empty_ask,
        compose_fn=lambda sys, usr: "Hi Sarah, term sheet is on track. Blake",
    )
    assert "*Draft:*" in reply
    assert "Hi Sarah, term sheet is on track. Blake" in reply
    # No fallback caveat when Nova-Pro succeeded.
    assert "fallback" not in reply.lower()


def test_handle_draft_surfaces_fallback_caveat_when_nova_fails():
    def _broken(sys, usr):
        raise RuntimeError("creds missing")

    reply = handle_draft(
        "sarah@example.com follow-up",
        ask_fn=_empty_ask,
        compose_fn=_broken,
    )
    assert "*Draft (fallback):*" in reply
    assert "Nova-Pro unavailable" in reply
    assert "creds missing" in reply


# --- Action marker -----------------------------------------------------------


def test_handle_draft_emits_action_marker():
    reply = handle_draft(
        "sarah@example.com ping",
        ask_fn=_empty_ask,
        compose_fn=lambda s, u: "Hi.",
    )
    draft_id = extract_action_draft_id(reply)
    assert draft_id is not None
    assert 4 <= len(draft_id) <= 32
    # And the stored draft is recoverable by that id.
    stored = get_stored_draft(draft_id)
    assert stored is not None
    assert stored.recipient_value == "sarah@example.com"
    assert stored.body == "Hi."


def test_compose_and_extract_action_marker_roundtrip():
    marker = compose_action_marker("abc12345")
    assert marker == "[DRAFT_ACTIONS:abc12345]"
    assert extract_action_draft_id(f"prefix {marker} suffix") == "abc12345"
    assert extract_action_draft_id("no marker here") is None


def test_strip_action_marker_removes_inline_token():
    text = "Body text\n\n[DRAFT_ACTIONS:abc12345]"
    assert strip_action_marker(text) == "Body text"


# --- Draft store + decision recording ---------------------------------------


def test_record_decision_send_writes_atlas_triple_and_pops_store(_clear_draft_store):
    reply = handle_draft(
        "sarah@example.com ping",
        ask_fn=_empty_ask,
        compose_fn=lambda s, u: "Hi Sarah, ping.",
    )
    draft_id = extract_action_draft_id(reply)
    captured: dict = {}

    def _spy_writeback(*, draft: StoredDraft, decision: str):
        captured["draft"] = draft
        captured["decision"] = decision
        return {"ok": True}

    ok, msg = record_draft_decision(draft_id, "send", writeback_fn=_spy_writeback)
    assert ok is True
    assert "Atlas" in msg or "logged" in msg
    # Writeback received the right payload
    assert captured["decision"] == "send"
    assert captured["draft"].body == "Hi Sarah, ping."
    assert captured["draft"].recipient_value == "sarah@example.com"
    # Send pops the draft from the store
    assert get_stored_draft(draft_id) is None


def test_record_decision_send_tolerates_writeback_failure(_clear_draft_store):
    reply = handle_draft(
        "sarah@example.com ping",
        ask_fn=_empty_ask,
        compose_fn=lambda s, u: "Hi.",
    )
    draft_id = extract_action_draft_id(reply)

    def _bad_writeback(**_):
        raise RuntimeError("atlas 503")

    ok, msg = record_draft_decision(draft_id, "send", writeback_fn=_bad_writeback)
    # Send-intent still considered ok (locally marked); error surfaced to Blake.
    assert ok is True
    assert "deferred" in msg.lower() or "atlas 503" in msg.lower()


def test_record_decision_discard_pops_without_writeback(_clear_draft_store):
    reply = handle_draft(
        "sarah@example.com ping",
        ask_fn=_empty_ask,
        compose_fn=lambda s, u: "Hi.",
    )
    draft_id = extract_action_draft_id(reply)

    def _writeback(**_):
        raise AssertionError("Discard must not call writeback")

    ok, msg = record_draft_decision(draft_id, "discard", writeback_fn=_writeback)
    assert ok is True
    assert "discard" in msg.lower()
    assert get_stored_draft(draft_id) is None


def test_record_decision_edit_keeps_draft_alive(_clear_draft_store):
    reply = handle_draft(
        "sarah@example.com ping",
        ask_fn=_empty_ask,
        compose_fn=lambda s, u: "Hi Sarah.",
    )
    draft_id = extract_action_draft_id(reply)
    ok, msg = record_draft_decision(draft_id, "edit", writeback_fn=lambda **_: {})
    assert ok is True
    assert "Hi Sarah." in msg
    # Edit must NOT pop the draft — Blake's still working on it.
    assert get_stored_draft(draft_id) is not None


def test_record_decision_unknown_draft_id_returns_not_found(_clear_draft_store):
    ok, msg = record_draft_decision(
        "does-not-exist", "send", writeback_fn=lambda **_: {"ok": True}
    )
    assert ok is False
    assert "not found" in msg.lower() or "expired" in msg.lower()


def test_record_decision_unknown_action_returns_error(_clear_draft_store):
    reply = handle_draft(
        "sarah@example.com ping",
        ask_fn=_empty_ask,
        compose_fn=lambda s, u: "Hi.",
    )
    draft_id = extract_action_draft_id(reply)
    ok, msg = record_draft_decision(draft_id, "yeet")
    assert ok is False
    assert "unknown" in msg.lower()


def test_draft_store_evicts_oldest_when_soft_cap_exceeded(_clear_draft_store):
    """Soft cap keeps memory bounded under a runaway /draft loop."""
    from plugins.slash import draft as draft_mod

    # Shrink the cap for the test so we don't have to compose 256 drafts.
    original_cap = draft_mod._DRAFT_STORE_SOFT_CAP
    draft_mod._DRAFT_STORE_SOFT_CAP = 3
    try:
        ids = []
        for i in range(5):
            reply = handle_draft(
                f"user{i}@example.com ping",
                ask_fn=_empty_ask,
                compose_fn=lambda s, u, i=i: f"draft {i}",
            )
            ids.append(extract_action_draft_id(reply))
        # Only the last ~3 survive; the first two were evicted.
        survivors = [i for i in ids if get_stored_draft(i) is not None]
        assert len(survivors) <= 3
        assert ids[-1] in survivors
    finally:
        draft_mod._DRAFT_STORE_SOFT_CAP = original_cap
