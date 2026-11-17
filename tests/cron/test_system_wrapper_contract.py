"""Contract: the wrapper recognizers in ``agent.system_wrappers`` stay byte-identical to the
build sites. A reworded cron hint or bot-chat delivery banner must fail HERE, not silently stop
being stripped from session previews and roster rows.

Both assertions run the REAL build sites: the hint constant is imported, and the delivery banner
is the actual text ``bot_chat_message`` builds for the admitted Bot Chat turn.
"""

from agent.system_wrappers import (BOT_CHAT_DELIVERY_PREFIX, CONTEXT_BLOCK_HEADING,
    CRON_HINT_PREFIX, WRAPPER_END, strip_system_wrappers)
from cron import scheduler_delivery as sched_delivery


def test_cron_hint_recognizer_matches_the_build_site():
    from cron import scheduler_prompt
    hint = scheduler_prompt._CRON_HINT
    assert hint.startswith(CRON_HINT_PREFIX)
    # The paragraph-terminator rule: the hint is ONE bracketed paragraph ending at the first
    # blank line, so the recognizer can cut at the flattened ``]  `` without landing mid-wrapper.
    first_para, sep, _ = hint.partition("\n\n")
    assert sep and first_para.endswith("]")
    assert strip_system_wrappers(hint + "the job prompt") == "the job prompt"


def test_context_block_recognizer_matches_the_build_site():
    """``_prepend_context_block`` frames a previous run's output AHEAD of the job prompt; the
    recognizer must reduce the real frame to exactly the prompt, both raw and flattened."""
    from cron import scheduler_prompt
    prompt = "Summarize the week."
    block = scheduler_prompt._prepend_context_block(
        prompt, "Your previous run's output",
        scheduler_prompt._SELF_CONTEXT_INTRO, "the archived previous output")
    assert block.startswith(CONTEXT_BLOCK_HEADING)
    assert strip_system_wrappers(block) == prompt
    flat = block.replace("\n", " ").replace("\r", " ")
    assert strip_system_wrappers(flat) == prompt
    # The stacked shape a skills-job persists: hint, then block, then prompt.
    stacked = scheduler_prompt._CRON_HINT + block
    assert strip_system_wrappers(stacked) == prompt
    flat_stacked = stacked.replace("\n", " ").replace("\r", " ")
    assert strip_system_wrappers(flat_stacked) == prompt


def test_bot_chat_banner_recognizer_matches_the_build_site():
    message = sched_delivery.bot_chat_message({"id": "j1", "name": "Daily digest"}, "the payload")
    assert message.startswith(BOT_CHAT_DELIVERY_PREFIX)
    # Whatever the build site produces, the recognizer must reduce it to exactly the payload.
    assert strip_system_wrappers(message) == "the payload"
    first_para, sep, rest = message.partition(WRAPPER_END)
    assert sep == WRAPPER_END
    assert first_para.startswith(BOT_CHAT_DELIVERY_PREFIX) and first_para.endswith("chat.")
    assert rest == "the payload"


def test_cron_run_transcript_shows_only_the_job_prompt():
    """A cron run's first user row (hint + run-time line + prompt, plain or skill-backed) renders as the
    job prompt on the TUI/Desktop resume projection and the Desktop REST projection (kimi-code#4780)."""
    from agent.skill_commands import _SINGLE_SKILL_INSTRUCTION
    from cron import scheduler_prompt
    from hermes_cli.web_routers.sessions import _project_for_display
    from tui_gateway import server

    prompt = "Summarize what landed in the vault today."
    plain = scheduler_prompt._CRON_HINT + scheduler_prompt._run_time_line() + prompt
    skilled = ('[IMPORTANT: The user has invoked the "vault" skill, indicating they want you to follow its '
               "instructions. The full skill content is loaded below.]\n\nSKILL BODY\n" + _SINGLE_SKILL_INSTRUCTION + plain)
    for content, expected in ((plain, prompt), (skilled, "/vault " + prompt)):
        rows = server._history_to_messages([{"role": "user", "content": content}])
        assert rows[0]["text"] == expected
    assert _project_for_display([{"role": "user", "content": plain}])[0]["display_content"] == prompt
    # Control: a Bot Chat delivery keeps its attribution banner in the transcript.
    banner = sched_delivery.bot_chat_message({"id": "j1", "name": "digest"}, "payload")
    assert server._history_to_messages([{"role": "user", "content": banner}])[0]["text"] == banner


def test_recognizers_are_importable_without_cron():
    """The leaf must stay dependency-free (state listing queries import it without cron)."""
    import importlib
    import agent.system_wrappers as sw
    importlib.reload(sw)
    assert BOT_CHAT_DELIVERY_PREFIX in sw.SYSTEM_WRAPPER_PREFIXES and CRON_HINT_PREFIX in sw.SYSTEM_WRAPPER_PREFIXES
