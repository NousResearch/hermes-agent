"""Session previews must show content, not Hermes' own machine-authored wrappers.

The cron run hint (``cron.scheduler_prompt._CRON_HINT``) and the bot-chat delivery banner
(``cron.scheduler_delivery._deliver_to_bot_chat``) frame a message for a model that is not the
user. Display surfaces must cut that framing; ``agent.system_wrappers`` holds the recognizers.
These tests cover the recognizer behaviour and the real listing-SQL path against a temp
SessionDB; the build-site byte contract lives in ``tests/cron/test_system_wrapper_contract.py``.
"""

from pathlib import Path

import pytest

from agent.system_wrappers import (BOT_CHAT_DELIVERY_PREFIX, CRON_HINT_PREFIX,
    WRAPPER_END, strip_system_wrappers)
from hermes_state import SessionDB


# ── recognizer behaviour ─────────────────────────────────────────────────────

def test_strip_recovers_the_job_prompt_from_the_real_cron_hint():
    from cron import scheduler_prompt
    prompt = "Summarize what landed in the vault today."
    assert strip_system_wrappers(scheduler_prompt._CRON_HINT + prompt) == prompt


def test_strip_survives_flattening():
    """Listing queries flatten newlines to spaces; the flattened terminator still anchors."""
    from cron import scheduler_prompt
    flat = scheduler_prompt._CRON_HINT.replace("\n", " ").replace("\r", " ")
    assert strip_system_wrappers(flat + "Summarize the week.") == "Summarize the week."


def test_strip_never_cuts_mid_wrapper():
    """The hint quotes [SILENT]/[CRON_FAILURE] inside itself — the first ']' is NOT the end."""
    from cron import scheduler_prompt
    assert "[SILENT]" in scheduler_prompt._CRON_HINT
    assert strip_system_wrappers(scheduler_prompt._CRON_HINT + "Real content.") == "Real content."


def test_unterminated_wrapper_is_left_alone():
    """A truncated excerpt must not lose its whole message to a guess."""
    truncated = CRON_HINT_PREFIX + " DELIVERY: your final response will be automat"
    assert strip_system_wrappers(truncated) == truncated


def test_stacked_wrappers_cut_in_sequence():
    banner = (f'{BOT_CHAT_DELIVERY_PREFIX}"digest" output — scheduled job, not the user. '
              "Review it and summarize.]" + WRAPPER_END)
    stacked = banner + CRON_HINT_PREFIX + " body.]  " + WRAPPER_END + "inner content"
    assert strip_system_wrappers(stacked) == "inner content"


def test_plain_text_untouched():
    assert strip_system_wrappers("just a normal message") == "just a normal message"
    assert strip_system_wrappers("") == ""


# ── behaviour: the real listing SQL path ─────────────────────────────────────

_LONG_BODY = "X" * 9000  # pushes the skill body past the head+tail excerpt windows


@pytest.fixture
def db(tmp_path):
    handle = SessionDB(Path(tmp_path) / "state.db")
    yield handle
    handle.close()


def _preview(db, session_id):
    return next(s["preview"] for s in db.list_sessions_rich(limit=20) if s["id"] == session_id)


def test_cron_run_session_previews_the_job_prompt(db):
    from cron import scheduler_prompt
    db.create_session("cronrun", source="cron")
    db.append_message("cronrun", role="user",
                      content=scheduler_prompt._CRON_HINT + "You are a one-shot delivery watcher for the monitor script.")
    preview = _preview(db, "cronrun")
    assert preview.startswith("You are a one-shot delivery watcher")
    assert "IMPORTANT" not in preview


def test_bot_chat_mirror_previews_the_content(db):
    db.create_session("botchat", source="telegram")
    db.append_message("botchat", role="user", content=(
        f'{BOT_CHAT_DELIVERY_PREFIX}"personal-brain-daily-digest" output — scheduled job, not '
        "the user. Review it, act on anything that needs action, and summarize for the chat.]"
        + WRAPPER_END + "PERSONAL BRAIN FETCH FAILED: iMessage fetch exited with code 1"))
    preview = _preview(db, "botchat")
    assert preview.startswith("PERSONAL BRAIN FETCH FAILED")
    assert "Cronjob" not in preview


def test_cron_skill_session_previews_the_job_prompt_not_the_slug(db):
    """A skills-job embeds hint+prompt behind the skill body; the row must not stop at /slug."""
    from agent.skill_commands import _SINGLE_SKILL_INSTRUCTION
    from cron import scheduler_prompt
    db.create_session("cronskill", source="cron")
    db.append_message("cronskill", role="user", content=(
        '[IMPORTANT: The user has invoked the "personal-brain-vault" skill, indicating they '
        "want you to follow its instructions. The full skill content is loaded below.]\n\n"
        + _LONG_BODY + "\n" + _SINGLE_SKILL_INSTRUCTION
        + scheduler_prompt._CRON_HINT + "Summarize what landed in the vault today and what changed."))
    preview = _preview(db, "cronskill")
    assert preview.startswith("Summarize what landed in the vault")
    assert "personal-brain-vault" not in preview


def test_cron_skill_session_with_context_block_previews_the_job_prompt(db):
    """The real weekly-review shape: skill body, instruction, cron hint, THEN a fenced
    previous-run block ahead of the job prompt. The preview must land on the prompt."""
    from agent.skill_commands import _SINGLE_SKILL_INSTRUCTION
    from cron import scheduler_prompt
    block = scheduler_prompt._prepend_context_block(
        "Summarize what landed in the vault this week.", "Your previous run's output",
        scheduler_prompt._SELF_CONTEXT_INTRO, "the archived previous run's output")
    db.create_session("cronskillctx", source="cron")
    db.append_message("cronskillctx", role="user", content=(
        '[IMPORTANT: The user has invoked the "personal-brain-vault" skill, indicating they '
        "want you to follow its instructions. The full skill content is loaded below.]\n\n"
        + _LONG_BODY + "\n" + _SINGLE_SKILL_INSTRUCTION + scheduler_prompt._CRON_HINT + block))
    preview = _preview(db, "cronskillctx")
    assert preview.startswith("Summarize what landed in the vault this week")
    assert "previous run" not in preview.lower()
    assert "personal-brain-vault" not in preview


def test_interactive_skill_session_keeps_its_described_preview(db):
    """Control: the cron-skill cut is scoped to cron-hint rows; interactive /skill previews
    keep the existing ``/slug — typed instruction`` shape."""
    from agent.skill_commands import _SINGLE_SKILL_INSTRUCTION
    db.create_session("interskill", source="cli")
    db.append_message("interskill", role="user", content=(
        '[IMPORTANT: The user has invoked the "work" skill, indicating they want you to '
        "follow its instructions. The full skill content is loaded below.]\n\n"
        + _LONG_BODY + "\n" + _SINGLE_SKILL_INSTRUCTION + "Fix the CI gate before the release"))
    assert _preview(db, "interskill") == "/work — Fix the CI gate before the release"


def test_roster_latest_message_preview_strips_banner(db):
    """The desktop roster's latest-message preview (tui_gateway.methods_profiles) cuts the
    banner BEFORE collapsing whitespace (the collapse would merge the flattened terminator)."""
    from tui_gateway.methods_profiles import _latest_message_preview
    db.create_session("roster", source="telegram")
    db.append_message("roster", role="user", content=(
        f'{BOT_CHAT_DELIVERY_PREFIX}"digest" output — scheduled job, not the user. Review it, '
        "act on anything that needs action, and summarize for the chat.]" + WRAPPER_END +
        "Quiet night in the mailbox: six emails, all correctly skipped."))
    preview = _latest_message_preview(db, "roster")
    assert preview.startswith("Quiet night in the mailbox")
    assert "Cronjob" not in preview
