"""Contract: the wrapper recognizers in ``agent.system_wrappers`` stay byte-identical to the
build sites. A reworded cron hint or bot-chat delivery banner must fail HERE, not silently stop
being stripped from session previews and roster rows.

Both assertions run the REAL build sites: the hint constant is imported, and the delivery banner
is captured from the actual message ``_deliver_to_bot_chat`` writes to the query file (same mock
lane as ``test_deliver_message_carries_cron_attribution``).
"""

import os
import subprocess
from unittest import mock

from agent.system_wrappers import (BOT_CHAT_DELIVERY_PREFIX, CONTEXT_BLOCK_HEADING,
    CRON_HINT_PREFIX, WRAPPER_END, strip_system_wrappers)
from cron import scheduler_delivery as sched_delivery
from cron.scheduler_delivery import _deliver_to_bot_chat


def _completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


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
    captured = {}

    def fake_run(argv, env, report_path, timeout):
        qf = argv[argv.index("--query-file") + 1]
        with open(qf, encoding="utf-8") as fh:
            captured["message"] = fh.read()
        return _completed()

    with mock.patch.object(sched_delivery, "_run_bot_chat_turn", side_effect=fake_run), \
         mock.patch.object(sched_delivery.shutil, "which", return_value="/usr/bin/hermes"):
        _deliver_to_bot_chat({"id": "j1", "name": "Daily digest"}, "the payload", "")

    message = captured["message"]
    assert message.startswith(BOT_CHAT_DELIVERY_PREFIX)
    # Whatever the build site produces, the recognizer must reduce it to exactly the payload.
    assert strip_system_wrappers(message) == "the payload"
    # The wrapper paragraph ends at the first blank line; the closing bracket belongs to the
    # terminator itself (``]`` + blank line), so the text before it ends with the banner prose.
    first_para, sep, rest = message.partition(WRAPPER_END)
    assert sep == WRAPPER_END
    assert first_para.startswith(BOT_CHAT_DELIVERY_PREFIX) and first_para.endswith("chat.")
    assert rest == "the payload"


def test_recognizers_are_importable_without_cron():
    """The leaf must stay dependency-free (state listing queries import it without cron)."""
    import importlib
    import agent.system_wrappers as sw
    importlib.reload(sw)
    assert sw.SYSTEM_WRAPPER_PREFIXES == (CRON_HINT_PREFIX, BOT_CHAT_DELIVERY_PREFIX)
