"""Tests for the output-side anti-pattern guard (Artemis B-0510-01 Phase 3).

Guard scans Coach briefing output for reasoning leaks / template-bypass shapes
(A, A', A'' sub-symptoms in the invest). On hit, the scheduler substitutes a
deterministic quiet-day fallback. See
docs/plans/investigations/coach-briefing-output-fidelity.md (Artemis repo) for
the failure taxonomy and verbatim fixtures.
"""
from cron.scheduler import _scan_briefing_anti_patterns, _quiet_day_fallback


# -----------------------------------------------------------------------------
# RED fixtures — verbatim or near-verbatim prod leak samples. Must be flagged.
# -----------------------------------------------------------------------------

JAMES_5_9_A = """Now let me assess the current date: Saturday, May 09, 2026 (or possibly later - this is a cron job). The strategy was last updated May 8, 2026. The follow-ups mention June 12, June 15, June 18. Today is May 9, which means no follow-ups are due today yet.

The emotional context shows the user is 11 days post-graduation with zero engagement, high stress, avoidance patterns. The `do_not_do` says don't surface job listings until resume content exists, and today is May 9 — no June events are due yet (the June 12 follow-up is more than 2 days away). No resume exists (`no_resume` status).

This is a quiet day with heavy emotional context. I should send a quiet-day briefing that acknowledges the situation without pushing.

Quiet day — no action items are due today."""

GARWIN_5_2_A_PRIME = """The matched roles are all entry-level — well below Garwin's C-suite level. Per strategy direction, public job boards produce zero relevant matches for him. I'll skip New Roles entirely.

Garwin is in acute ambiguity fatigue with the May 5 forced default in 3 days. I'll acknowledge the weight, surface only the 2 highest-leverage actions, and keep it binary.

Scanning for executive-level signals — the job boards are quiet for C-suite today, which is expected at this level.

Quiet morning on the executive boards. The May 5 default sits 3 days out; here are the two highest-leverage moves to consider today."""

JAMES_5_12_A_DOUBLEPRIME = """Here is the situation:

- User is 11+ days post-graduation (June 2026, now May 12 2026 — actually user hasn't graduated yet! Wait - the profile says "graduating June 2026" and today is May 12, 2026. So the user hasn't graduated yet, they have about a month to go. The emotional context says "11 days post-graduation" which seems wrong — that might have been written assuming a later run date).
- Status: `no_resume` — no job matches available.
- Emotional context: heavy stress, zero engagement, avoidance behavior.

Let me check the `do_not_do` list: skip surfacing job listings until resume exists.

Wait — the strategy is stale (updated May 8, and the staleness threshold is 48 hours).

Actually, let me reconsider. The user is still a student graduating in June.

Let me write a quiet-day note."""


# -----------------------------------------------------------------------------
# GREEN fixtures — clean briefs that previously rendered correctly. Must pass.
# -----------------------------------------------------------------------------

JAMES_5_10_CLEAN = "No action needed today — finals come first, and the job search is paused on my end too. We'll pick back up when you're through. Reply any time if something shifts."

GARWIN_CLEAN = """Quiet morning on the executive boards — your team is holding the line through the May 5 default. Two moves on the table.

```
📌 Follow-ups
───────────
⭐ TODAY · Send the binary check-in (#1 of 2)
···  May 7 · Final ping if no reply
```

💬 **Coach's Take:** The binary check-in is ready — two yes/no questions, under three sentences. If nothing comes back by May 7 I'll send the final ping then go quiet until June."""

AMY_CLEAN_QUIET = "Nothing urgent on the board today — your team is in a watching pattern. I'll keep scanning roles and tracking your follow-ups in the background. Reply any time if something comes up."


# -----------------------------------------------------------------------------
# Scanner tests
# -----------------------------------------------------------------------------

def test_a_class_reasoning_prefix_flagged():
    clean, reason = _scan_briefing_anti_patterns(JAMES_5_9_A)
    assert not clean
    assert reason  # non-empty reason string


def test_a_prime_class_third_person_prefix_flagged():
    clean, reason = _scan_briefing_anti_patterns(GARWIN_5_2_A_PRIME)
    assert not clean
    assert reason


def test_a_doubleprime_template_bypass_flagged():
    clean, reason = _scan_briefing_anti_patterns(JAMES_5_12_A_DOUBLEPRIME)
    assert not clean
    assert reason


def test_clean_quiet_day_passes():
    clean, reason = _scan_briefing_anti_patterns(JAMES_5_10_CLEAN)
    assert clean, f"expected clean, got reason={reason!r}"


def test_clean_content_brief_passes():
    clean, reason = _scan_briefing_anti_patterns(GARWIN_CLEAN)
    assert clean, f"expected clean, got reason={reason!r}"


def test_clean_amy_quiet_passes():
    clean, reason = _scan_briefing_anti_patterns(AMY_CLEAN_QUIET)
    assert clean, f"expected clean, got reason={reason!r}"


def test_empty_returns_clean():
    # Empty / whitespace passthrough — the broader scheduler already drops
    # empty deliveries via bool() check upstream.
    clean, _ = _scan_briefing_anti_patterns("")
    assert clean


def test_leading_wait_em_dash_flagged():
    text = "Wait — actually the situation is quieter than it looked. No action today; reply if anything shifts."
    clean, _ = _scan_briefing_anti_patterns(text)
    assert not clean


def test_mid_content_let_me_write_flagged():
    text = ("Nothing urgent today on the board. Let me write a quiet-day note "
            "for you. Reply any time.")
    clean, _ = _scan_briefing_anti_patterns(text)
    assert not clean


# -----------------------------------------------------------------------------
# "Looking at <topic>" opener — leading "looking at" was removed from
# _BRIEFING_LEADING_REASONING so legitimate briefing openers pass layer 1.
# Reasoning-shape variants stay in _BRIEFING_MIDCONTENT_REASONING and still
# trip layer 2. See Artemis S-0511-07 § Architecture.
# -----------------------------------------------------------------------------

def test_topic_leading_looking_at_opener_passes():
    text = ("Looking at backend roles across the Bay — five strong matches "
            "surfaced today.\n\n"
            "```\n📌 Follow-ups\n───────────\n⭐ TODAY  Send the Waymo app\n```\n\n"
            "💬 **Coach's Take:** Apply today.")
    clean, reason = _scan_briefing_anti_patterns(text)
    assert clean, f"expected clean, got reason={reason!r}"


def test_topic_leading_looking_at_series_b_passes():
    text = "Looking at Series B data science openings this morning."
    clean, reason = _scan_briefing_anti_patterns(text)
    assert clean, f"expected clean, got reason={reason!r}"


def test_looking_at_the_strategy_flagged():
    text = ("Looking at the strategy, the user has not shared their resume yet "
            "so I'll keep the surface small.")
    clean, _ = _scan_briefing_anti_patterns(text)
    assert not clean


def test_looking_at_the_user_flagged():
    text = ("Quiet day on the board. Looking at the user's emotional context, "
            "they need rest.")
    clean, _ = _scan_briefing_anti_patterns(text)
    assert not clean


def test_looking_at_the_emotional_context_flagged():
    text = ("Looking at the emotional context, the user is in finals overload — "
            "let me keep this short.")
    clean, _ = _scan_briefing_anti_patterns(text)
    assert not clean


def test_looking_at_session_flagged():
    text = ("Quiet day. Looking at session history, the user mentioned a "
            "deadline last week.")
    clean, _ = _scan_briefing_anti_patterns(text)
    assert not clean


def test_content_with_user_quote_not_flagged():
    # Quoted user speech inside a brief that begins with a clean opener should
    # not trip the guard. The anti-patterns are detected only at the leading
    # clause or as standalone mid-content reasoning fragments.
    text = ('Quiet day on your end — last week you said "let me think about it" '
            'and then went heads down. Holding the surface small until you surface. '
            'Reply any time.')
    clean, reason = _scan_briefing_anti_patterns(text)
    assert clean, f"expected clean, got reason={reason!r}"


# -----------------------------------------------------------------------------
# Fallback template
# -----------------------------------------------------------------------------

def test_fallback_template_is_second_person():
    msg = _quiet_day_fallback()
    assert msg
    # No third-person leakage / reasoning markers
    clean, _ = _scan_briefing_anti_patterns(msg)
    assert clean


def test_fallback_template_is_short():
    msg = _quiet_day_fallback()
    assert len(msg) < 400
