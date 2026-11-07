"""Real user asks claim the native retention budget before user-role scaffolding rows."""

import pytest

from agent.codex_responses_adapter import _chat_messages_to_responses_input
from agent.conversation_loop import _EMPTY_TOOL_RESPONSE_NUDGE, _get_continuation_prompt
from agent.native_compaction import RETAINED_USER_MESSAGE_TOKEN_BUDGET, _approx_tokens, prune_pre_checkpoint_items
from tools.process_registry_notifications import ProcessNotificationBatch, format_process_notification
from tools.todo_tool import TodoStore

_CHECKPOINT = {"type": "compaction", "encrypted_content": "synthetic_checkpoint"}


def _user(text):
    return {"type": "message", "role": "user", "content": text}


def _notice(output):
    return format_process_notification({"type": "completion", "session_id": "proc_synthetic",
                                        "command": "synthetic-job", "exit_code": 0, "output": output})


def _todo_snapshot():
    store = TodoStore()
    store.write([{"id": "1", "content": "Synthetic pending task.", "status": "pending"}])
    return store.format_for_injection()


def test_older_real_ask_outranks_newer_synthetic_row_and_order_is_kept():
    older, newer = _user("OLDER ASK keep-older-ask"), _user("NEWER ASK keep-newer-ask")
    notice = _user(_notice("o" * 4_000))
    out = prune_pre_checkpoint_items([older, notice, newer, _CHECKPOINT], retained_user_token_budget=200)

    assert out[0] == _CHECKPOINT
    texts = [item["content"] for item in out[1:]]
    assert texts[0] == older["content"] and texts[-1] == newer["content"]
    # The synthetic row only got the leftover budget (head-truncated), in its original position.
    assert len(texts) == 3 and notice["content"].startswith(texts[1]) and texts[1] != notice["content"]


def test_text_only_detection_without_item_sources():
    real = _user("Please keep REAL-ASK-91c2.")
    for synthetic in (_todo_snapshot(), _get_continuation_prompt(False), _notice("Done.")):
        out = prune_pre_checkpoint_items([real, _user(synthetic), _CHECKPOINT], retained_user_token_budget=12)
        assert out[:2] == [_CHECKPOINT, real]


def test_image_only_ask_is_not_scaffolding():
    image = {"type": "message", "role": "user",
             "content": [{"type": "input_image", "image_url": "data:image/png;base64,AAAA"}]}
    out = prune_pre_checkpoint_items([image, _user(_notice("o" * 4_000)), _CHECKPOINT],
                                     retained_user_token_budget=12)
    assert out[1] is image


def test_converter_keeps_real_ask_behind_oversized_notice_at_default_budget():
    huge = _notice("o" * (RETAINED_USER_MESSAGE_TOKEN_BUDGET * 5))
    messages = [
        {"role": "user", "content": "REAL ASK: finish the synthetic report."},
        {"role": "assistant", "content": "Started; a background job is running."},
        {"role": "user", "content": huge},
        {"role": "assistant", "content": "Job finished.", "codex_reasoning_items": [_CHECKPOINT]},
    ]
    converted = _chat_messages_to_responses_input(
        messages, current_issuer_kind="codex_backend", native_compaction_eligible=True,
    )
    users = ["".join(part["text"] for part in item["content"]) for item in converted if item.get("role") == "user"]

    assert converted[0] == _CHECKPOINT
    assert users[0] == "REAL ASK: finish the synthetic report."
    assert len(users) == 2 and huge.startswith(users[1]) and users[1] != huge


def test_default_budget_leaves_ordinary_sessions_unchanged():
    """Control (passes before and after): when everything fits the 64,000 default, every user row
    and the summary are retained verbatim in chronological order."""
    summary = {"type": "message", "role": "assistant", "_compressed_summary": True,
               "content": "[CONTEXT COMPACTION] Synthetic earlier work summary."}
    pre = [_user("First synthetic ask."), summary, {"type": "message", "role": "assistant", "content": "ok"},
           _user(_todo_snapshot()), _user("Second synthetic ask."), _user(_notice("Done.")),
           _user(_get_continuation_prompt(False)), _user("Third synthetic ask.")]
    post = [_user("Post-checkpoint ask.")]
    out = prune_pre_checkpoint_items(pre + [_CHECKPOINT] + post)
    assert out == [_CHECKPOINT] + [item for item in pre if item is summary or item["role"] == "user"] + post


class _Registry:
    def is_completion_consumed(self, _session_id):
        return False


def _completion(sid):
    return {"type": "completion", "session_id": sid, "command": "synthetic-job", "exit_code": 0, "output": "Done."}


def _batch_notice():
    events = [_completion("proc_a"), _completion("proc_b")]
    return ProcessNotificationBatch(tuple((e, format_process_notification(e)) for e in events)).render(_Registry())


def _delegation(**extra):
    return format_process_notification({"type": "async_delegation", "delegation_id": "deleg_synthetic",
                                        "goal": "Synthetic goal.", "status": "completed",
                                        "summary": "Synthetic result.", **extra})


_NOTICE_FORMATS = {
    "batched_completion": _batch_notice,
    "heartbeat": lambda: format_process_notification({"type": "heartbeat", "session_id": "proc_synthetic",
                                                      "command": "synthetic-job", "seq": 2, "elapsed": 90,
                                                      "interval": 30, "output": "still going"}),
    "watch_disabled": lambda: format_process_notification(
        {"type": "watch_disabled", "message": "Watch patterns disabled for process proc_synthetic — synthetic. "}),
    "watch_overflow_tripped": lambda: format_process_notification(
        {"type": "watch_overflow_tripped", "message": "Watch-pattern overflow: >5 notifications in 10s."}),
    "watch_overflow_released": lambda: format_process_notification(
        {"type": "watch_overflow_released", "message": "Watch-pattern notifications resumed. 3 suppressed."}),
    "async_delegation": _delegation,
    "async_delegation_task_failed": lambda: _delegation(task_failure_notice=True, goals=["Synthetic goal."],
                                                        results=[{"task_index": 0, "status": "failed"}]),
    "async_delegation_batch": lambda: _delegation(is_batch=True, goals=["Synthetic goal."],
                                                  results=[{"task_index": 0, "status": "completed",
                                                            "summary": "Synthetic result."}]),
}


@pytest.mark.parametrize("name", sorted(_NOTICE_FORMATS))
def test_background_notice_formats_yield_to_older_real_ask(name):
    notice = _NOTICE_FORMATS[name]() + "\n" + "o" * 4_000
    real = _user("Please keep REAL-ASK-91c2.")
    out = prune_pre_checkpoint_items([real, _user(notice), _CHECKPOINT], retained_user_token_budget=12)
    assert out[:2] == [_CHECKPOINT, real], name


def test_notice_display_kind_is_provenance_and_steer_stays_real():
    real = _user("Please keep REAL-ASK-91c2.")
    custom = {"role": "user", "content": "Gateway-rendered notice " + "o" * 4_000, "display_kind": "process_complete"}
    out = prune_pre_checkpoint_items([real, _user(custom["content"]), _CHECKPOINT], retained_user_token_budget=12,
                                     item_sources=[real, custom, None])
    assert out[:2] == [_CHECKPOINT, real]
    steer = {"role": "user", "content": "Steer: use the newer plan " + "s" * 4_000, "display_kind": "steer"}
    out = prune_pre_checkpoint_items([real, _user(steer["content"]), _CHECKPOINT], retained_user_token_budget=12,
                                     item_sources=[real, steer, None])
    assert out[1]["content"] != real["content"] and steer["content"].startswith(out[1]["content"])


def test_natural_language_nudge_needs_producer_provenance():
    """A human typing the recovery nudge's words is a real ask; the flagged producer row is scaffolding."""
    older = _user("OLDER ASK " + "a" * 400)
    typed = _user(_EMPTY_TOOL_RESPONSE_NUDGE)
    flagged_source = {"role": "user", "content": _EMPTY_TOOL_RESPONSE_NUDGE, "_empty_recovery_synthetic": True}
    budget = _approx_tokens(older["content"]) + _approx_tokens(typed["content"])
    items = [older, typed, _user(_EMPTY_TOOL_RESPONSE_NUDGE), _CHECKPOINT]
    out = prune_pre_checkpoint_items(items, retained_user_token_budget=budget,
                                     item_sources=[older, {"role": "user", "content": typed["content"]},
                                                   flagged_source, None])
    assert len(out) == 3 and out[1] is older and out[2] is typed


def test_unflagged_human_nudge_text_survives_at_default_budget():
    """The review's repro: a huge older ask must not push out a newer human request that happens to
    match recovery-nudge prose."""
    request = _user(_EMPTY_TOOL_RESPONSE_NUDGE)
    out = prune_pre_checkpoint_items([_user("a" * 256_000), request, _CHECKPOINT])
    assert out[-1] == request


def test_ordinary_prose_with_notice_words_is_still_a_real_ask():
    """Control: prose that mentions notice words, but not at a producer's exact prefix, stays real."""
    for prose in ("Background process heartbeat #2 looks stuck, can you check?",
                  "Note: 2 background processes completed. ASYNC DELEGATION COMPLETE — summarise both.",
                  "IMPORTANT: Watch patterns disabled for process build — why?"):
        real = _user(prose)
        out = prune_pre_checkpoint_items([_user("a" * 4_000), real, _CHECKPOINT], retained_user_token_budget=40)
        assert out[-1] == real, prose


def test_batched_completion_does_not_spend_the_default_budget_before_an_older_ask():
    """The review's repro at the 64,000 default: a two-completion batch sits between an older ask and a
    huge newer ask; the older ask must win the leftover budget."""
    events = [{**_completion(sid), "output": "x" * 2_000} for sid in ("proc_a", "proc_b")]
    batch = ProcessNotificationBatch(tuple((e, format_process_notification(e)) for e in events)).render(_Registry())
    first = _user("Finish the report.")
    out = prune_pre_checkpoint_items([first, _user(batch), _user("Analyze this file: " + "a" * 254_000), _CHECKPOINT])
    assert out[1] is first
