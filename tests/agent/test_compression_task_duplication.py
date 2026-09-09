"""Compression must keep one actionable copy of a large unfinished request.

Related to #100818: preserving the task after the handoff must not duplicate
the entire protected user input. Summary generation is stubbed; assembly is real.

#106889 review: the head-copy replacement must (a) select the *active* in-flight
row (the most recent equal-text user row before the carrier), not an earlier
completed-history row with the same text, and (b) drop the stale
``api_content`` sidecar so the provider wire carries the pointer, not the
full original request.
"""

import socket
from unittest.mock import patch

import pytest


@pytest.fixture
def compress_task(monkeypatch):
    def deny_network(*args, **kwargs):
        raise AssertionError("This synthetic reproduction must not use the network")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_network)
    monkeypatch.setattr(socket, "getaddrinfo", deny_network)

    from agent.context_compressor import ContextCompressor, SUMMARY_PREFIX

    def run(
        protect_first_n=3,
        completed=False,
        history_equal_task=False,
        with_api_content=False,
    ):
        task = ("Audit this synthetic document.\n" + "Synthetic policy material. " * 30000).rstrip()
        messages = []
        if history_equal_task:
            # A completed earlier turn that happens to share the verbatim text
            # of the active in-flight request. It must be left intact (#106889
            # review: text equality alone must not select this row).
            messages.extend([
                {"role": "user", "content": task},
                {"role": "assistant", "content": "Earlier turn done."},
            ])
        messages.append({"role": "user", "content": task})
        if with_api_content:
            # The provider wire representation of the active request; replaying
            # it after the head copy is replaced would resend the full text.
            messages[0 if not history_equal_task else 2]["api_content"] = task
        for index in range(10):
            call_id = f"call_{index}"
            messages.extend([
                {"role": "assistant", "content": "", "tool_calls": [{
                    "id": call_id, "type": "function",
                    "function": {"name": "synthetic_submit", "arguments": "{}"},
                }]},
                {"role": "tool", "tool_call_id": call_id, "content": "accepted"},
            ])
        if completed:
            messages.append({"role": "assistant", "content": "Finished."})
        compressor = ContextCompressor(
            "synthetic-model", threshold_percent=0.8,
            protect_first_n=protect_first_n, protect_last_n=20,
            config_context_length=229376, max_tokens=8192, quiet_mode=True,
        )
        with patch.object(
            compressor, "_generate_summary",
            return_value=SUMMARY_PREFIX + "\nSynthetic handoff.",
        ) as summary:
            result = compressor.compress(messages, current_tokens=180000, force=True)
        assert summary.call_count == 1, "The fixture must cross a summary boundary"
        return task, result

    return run


@pytest.mark.parametrize("repeat", [1, 2])
def test_first_compaction_keeps_one_actionable_task_copy(compress_task, repeat):
    from agent.context_compressor import _SUMMARY_END_MARKER

    task, messages = compress_task()
    texts = [message.get("content", "") for message in messages]
    boundary = next(index for index, text in enumerate(texts) if _SUMMARY_END_MARKER in text)
    after_handoff = [texts[boundary].split(_SUMMARY_END_MARKER, 1)[1], *texts[boundary + 1:]]
    assert any(task in text for text in after_handoff), "Unfinished task must remain actionable"
    copies = sum(text.count(task) for text in texts)
    assert copies == 1, f"repeat={repeat}: full task copies after compression={copies}"


@pytest.mark.parametrize("repeat", [1, 2])
@pytest.mark.parametrize("protect_first_n,completed", [(0, False), (3, True)])
def test_task_copy_controls(compress_task, protect_first_n, completed, repeat):
    task, messages = compress_task(protect_first_n=protect_first_n, completed=completed)
    copies = sum(message.get("content", "").count(task) for message in messages)
    assert copies == 1, f"repeat={repeat}: control full task copies={copies}"


def test_completed_equal_text_history_row_not_replaced(compress_task):
    """The active in-flight copy is the most recent equal-text user row before
    the carrier; an earlier completed turn with the same text must be left
    intact (regression for the wrong-row selection, #106889 review)."""
    from agent.context_compressor import _INFLIGHT_HEAD_REPLACED_NOTICE

    task, messages = compress_task(history_equal_task=True)

    pointer_rows = [
        idx for idx, m in enumerate(messages)
        if _INFLIGHT_HEAD_REPLACED_NOTICE in str(m.get("content", ""))
    ]
    assert len(pointer_rows) == 1, (
        f"expected exactly one head copy replaced with the pointer, got {pointer_rows}"
    )

    # The completed-history row is the user turn immediately followed by the
    # "Earlier turn done." assistant marker; it must retain the full task text.
    completed_retained = False
    for idx, m in enumerate(messages):
        if m.get("role") != "user":
            continue
        nxt = messages[idx + 1] if idx + 1 < len(messages) else {}
        if nxt.get("role") == "assistant" and "earlier turn done" in str(nxt.get("content", "")).lower():
            assert task in str(m.get("content", "")), (
                "completed equal-text history row was wrongly replaced with the pointer"
            )
            completed_retained = True
    assert completed_retained, "fixture did not surface the completed-history row for assertion"

    # The active in-flight copy was replaced; the full text now lives exactly
    # once after the boundary (the replay), so the only full-text occurrence is
    # the replay (the completed-history row shares the text but is a distinct,
    # completed turn — not a duplicate of the in-flight request).
    assert sum(str(m.get("content", "")).count(task) for m in messages) == 2, (
        "expected full text on exactly the completed-history row + the replay"
    )


def test_replaced_head_copy_drops_stale_api_content(compress_task):
    """Replacing the head copy's content must also drop the stale
    ``api_content`` sidecar so ``substitute_api_content()`` cannot restore the
    full original request on the provider wire (#106889 review)."""
    from agent.context_compressor import _INFLIGHT_HEAD_REPLACED_NOTICE

    _task, messages = compress_task(with_api_content=True)

    replaced = [
        m for m in messages
        if _INFLIGHT_HEAD_REPLACED_NOTICE in str(m.get("content", ""))
    ]
    assert replaced, "expected the head copy to be replaced with the pointer"
    for m in replaced:
        assert "api_content" not in m, (
            "replaced head copy must drop the stale api_content sidecar"
        )
