from __future__ import annotations

import json

from claude_selfimprove import llm


def _eligible(cid="cid-1", title="Never force push", body="Never force push to main."):
    return [{"id": cid, "title": title, "body": body}]


def test_no_conflict_when_model_says_so(sandbox):
    def runner(prompt, *, model, provider, timeout):
        return True, json.dumps([{"index": 1, "conflicts": False, "reason": ""}])

    result = llm.check_conflicts(_eligible(), [], runner=runner)
    assert result["cid-1"] == (False, "")


def test_conflict_detected_with_reason(sandbox):
    def runner(prompt, *, model, provider, timeout):
        return True, json.dumps(
            [{"index": 1, "conflicts": True, "reason": "contradicts existing always-push rule"}]
        )

    result = llm.check_conflicts(
        _eligible(),
        [{"title": "Always push directly", "body": "Always push directly to main."}],
        runner=runner,
    )
    conflicts, reason = result["cid-1"]
    assert conflicts is True
    assert "contradicts" in reason


def test_fails_closed_on_runner_failure(sandbox):
    def runner(prompt, *, model, provider, timeout):
        return False, "boom"

    result = llm.check_conflicts(_eligible(), [], runner=runner)
    assert result["cid-1"][0] is True  # fail closed = treated as a conflict


def test_fails_closed_on_unparseable_output(sandbox):
    def runner(prompt, *, model, provider, timeout):
        return True, "not json"

    result = llm.check_conflicts(_eligible(), [], runner=runner)
    assert result["cid-1"][0] is True


def test_fails_closed_for_items_missing_from_response(sandbox):
    # Two eligible candidates, model only answers for the first.
    eligible = _eligible(cid="cid-1") + _eligible(cid="cid-2", title="Other", body="Other body.")

    def runner(prompt, *, model, provider, timeout):
        return True, json.dumps([{"index": 1, "conflicts": False, "reason": ""}])

    result = llm.check_conflicts(eligible, [], runner=runner)
    assert result["cid-1"] == (False, "")
    assert result["cid-2"][0] is True  # not answered -> fail closed


def test_empty_eligible_list_short_circuits():
    calls = []

    def runner(prompt, *, model, provider, timeout):
        calls.append(1)
        return True, "[]"

    assert llm.check_conflicts([], [], runner=runner) == {}
    assert calls == []
