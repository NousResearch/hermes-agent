import json
from types import SimpleNamespace

from hermes_cli.machine_result import capture_usage, emit_machine_result, usage_delta


def test_usage_delta_excludes_resumed_session_history():
    before = {"inputTokens": 100, "outputTokens": 20, "cachedInputTokens": 50,
              "cacheWriteTokens": 2, "reasoningTokens": 5, "estimatedCostUsd": 0.1}
    agent = SimpleNamespace(
        session_input_tokens=140, session_output_tokens=27,
        session_cache_read_tokens=80, session_cache_write_tokens=2,
        session_reasoning_tokens=8, session_estimated_cost_usd=0.13)

    result = usage_delta(before, capture_usage(agent))

    assert result["usage"] == {
        "inputTokens": 40, "outputTokens": 7, "cachedInputTokens": 30,
        "cacheWriteTokens": 0, "reasoningTokens": 3,
    }
    assert result["costUsd"] == 0.03
    assert result["costStatus"] == "estimated"


def test_machine_result_rejects_corrupt_cost_and_is_json(capsys):
    before = {key: 0 for key in (
        "inputTokens", "outputTokens", "cachedInputTokens",
        "cacheWriteTokens", "reasoningTokens", "estimatedCostUsd")}
    after = dict(before, inputTokens=12, estimatedCostUsd=-809556)
    result = usage_delta(before, after)
    emit_machine_result("session-1", result)

    line = capsys.readouterr().err.strip()
    payload = json.loads(line.removeprefix("hermes_result:"))
    assert payload["usage"]["inputTokens"] == 12
    assert payload["costUsd"] is None
    assert payload["costStatus"] == "unpriced"
