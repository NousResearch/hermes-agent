"""Legacy clarify-tool result handling: pruning, sentinel classification, dispatch view.

Moved from test_context_compressor.py so the dispatch-view test added for route
attribution (#72637) grows a new file, not the shared one.
"""

import json

import pytest

from agent.context_compressor import (
    ContextCompressor,
    _PRUNE_MIN_CHARS,
    _summarize_tool_result,
    _sum_clarify,
)


# Captured user_response values from real imports of clarify_tool and both headless
# callbacks at bc7f58f3b0270f62aafc9283f57887b25d560e9b (before response status).
# Inputs: question="Deploy, really?", choices=["staging, canary", "production"],
# open-ended / single-select / multi-select. These are frozen producer outputs,
# not calls to today's producers or a reimplementation of comma normalization.
# Final cases embed runtime suffixes in the question/choice before a comma;
# those suffixes are quoted content, not the end of the enclosing notice.
_HISTORICAL_HEADLESS_RESPONSES = (
    "[oneshot mode: no user available. Make the most reasonable assumption you can and continue.]",
    "[oneshot mode: no user available. Pick the best option from ['staging, canary (Recommended)', 'production'] using your own judgment and continue.]",
    ["[oneshot mode: no user available. Pick the best subset from ['staging",
     "canary (Recommended)'", "'production'] using your own judgment and continue.]"],
    "[single-query mode: no user available to answer 'Deploy, really?'. Make the most reasonable assumption you can and continue.]",
    "[single-query mode: no user available to answer 'Deploy, really?'. Pick the best option from ['staging, canary (Recommended)', 'production'] using your own judgment and continue.]",
    ["[single-query mode: no user available to answer 'Deploy",
     "really?'. Pick the best subset from ['staging", "canary (Recommended)'",
     "'production'] using your own judgment and continue.]"],
    ["[oneshot mode: no user available. Pick the best subset from ['Explain this notice: ] using your own judgment and continue.]",
     "then deploy? (Recommended)'", "'production'] using your own judgment and continue.]"],
    ["[single-query mode: no user available to answer 'Explain this notice: . Make the most reasonable assumption you can and continue.]",
     "then deploy?'. Pick the best subset from ['staging (Recommended)'",
     "'production'] using your own judgment and continue.]"],
)


class TestLegacyClarifyResults:
    @pytest.mark.parametrize("shape", ["single", "batch", "current"])
    @pytest.mark.parametrize("question_size", [300, 4500])
    @pytest.mark.parametrize("answer,expected", [
        ("production, but only after 18:00 UTC", "production, but only after 18:00 UTC"),
        (["staging", "production"], ["staging", "production"]),
        ("The user cancelled the production rollout; do not restart it.",
         "The user cancelled the production rollout; do not restart it."),
        ("[single-query mode: no user available to answer 'Deploy?'. STOP production rollout now.",
         "[single-query mode: no user available to answer 'Deploy?'. STOP production rollout now."),
        *[(notice, None) for notice in _HISTORICAL_HEADLESS_RESPONSES],
        # Synthetic mixed-list controls surround the actual producer fragments;
        # deleting a complete envelope must not delete independent selections.
        *[(["staging"] + (notice if isinstance(notice, list) else [notice]) + ["production"],
           ["staging", "production"]) for notice in _HISTORICAL_HEADLESS_RESPONSES],
    ])
    def test_answers_reach_summary_after_repeated_pruning(self, monkeypatch, shape, question_size, answer, expected):
        import agent.context_compressor as module

        if shape == "current":
            expected = answer  # Explicit status wins even for notice-like answer text.
        expected_summary = ("[clarify] asked user a question" if expected is None else
                            module.elide("[clarify] user responded: " + json.dumps(expected, ensure_ascii=False),
                                         _PRUNE_MIN_CHARS - 1))
        entry = {"question": "Which environment? " + "q" * question_size,
                 "choices_offered": ["staging", "production"], "user_response": answer}
        if shape == "current":
            entry["status"] = "answered"
        payload = entry if shape == "single" else {"responses": [entry]}
        content = json.dumps(payload)
        c = ContextCompressor(model="test/model", config_context_length=100000,
                              protect_first_n=1, protect_last_n=2, quiet_mode=True)
        c.tail_token_budget = 50
        messages = [
            {"role": "system", "content": "Test fixture"},
            {"role": "user", "content": "Implement the task"},
            {"role": "assistant", "tool_calls": [{"id": "clarify-1", "type": "function",
                "function": {"name": "clarify", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "clarify-1", "content": content},
        ]
        for i in range(12):
            messages.extend([{"role": "assistant", "content": f"Finished step {i}"},
                             {"role": "user", "content": f"Continue step {i}"}])
        messages.append({"role": "assistant", "content": "Current step finished"})
        pruned, _ = c._prune_old_tool_results(messages, protect_tail_count=2)
        pruned, _ = c._prune_old_tool_results(pruned, protect_tail_count=2)
        assert pruned[3]["content"] == expected_summary
        assert _summarize_tool_result("clarify", "{}", content) == expected_summary
        requests = []

        def summarize(**kwargs):
            requests.append(kwargs)
            return {"choices": [{"message": {"content": "## Progress\nWork is in progress."},
                                 "finish_reason": "stop"}]}

        monkeypatch.setattr(module, "call_llm", summarize)
        c.compress(messages, force=True)
        assert requests, "exercise the public compression path, not a no-op window"
        # route_callback rides along as an observer kwarg (transport plumbing, not payload).
        dispatched = json.dumps({k: v for k, v in requests[0].items() if not callable(v)})
        assert json.dumps(expected_summary)[1:-1] in dispatched
        if expected is None or expected != answer:
            assert "using your own judgment" not in dispatched
            assert "reasonable assumption" not in dispatched
        assert messages[3]["content"] == content

    @pytest.mark.parametrize("sentinel", [
        "The user did not provide a response within the time limit.",
        "[user did not respond within 60m]",
        "[clarify prompt could not be delivered]",
        "[oneshot mode: no user available]",
        "The user cancelled. Use your best judgement to proceed.",
        "[single-query mode: no user available to answer 'Deploy?'. "
        "Make the most reasonable assumption you can and continue.]",
        *_HISTORICAL_HEADLESS_RESPONSES,
    ])
    def test_legacy_sentinels_are_not_answers_but_explicit_status_is_authoritative(self, sentinel):
        for value, kept in ((sentinel, None),
                            (["production"] + (sentinel if isinstance(sentinel, list) else ["  " + sentinel]),
                             "production")):
            for payload in ({"user_response": value}, {"responses": [{"user_response": value}]}):
                content = json.dumps(payload)
                summary = _sum_clarify("clarify", {}, content, len(content), 1)
                assert summary == (f'[clarify] user responded: "{kept}"' if kept else "[clarify] asked user a question")
        for shape in ("single", "batch"):
            for status in ("answered", "skipped", "unanswered"):
                entry = {"status": status, "user_response": sentinel}
                content = json.dumps(entry if shape == "single" else {"responses": [entry]})
                summary = _summarize_tool_result("clarify", "{}", content)
                assert summary.startswith("[clarify] user responded:") == (status == "answered")
                if status == "answered":
                    assert json.dumps(sentinel, ensure_ascii=False)[:70] in summary
                else:
                    assert summary == "[clarify] asked user a question"


