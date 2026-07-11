"""
Regression test for issue #62695.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, "/tmp/hermes-pr-work-60859/hermes-agent")


def test_prepare_inbound_uses_resolve_session_agent_runtime():
    src = Path("/tmp/hermes-pr-work-60859/hermes-agent/gateway/run.py").read_text()
    marker = "preprocess_context_references_async("
    idx = src.find(marker)
    assert idx > 0
    window = src[max(0, idx-1500):idx+200]
    assert "_resolve_session_agent_runtime" in window, (
        "Fix not detected: the @ context reference expansion block should "
        "call self._resolve_session_agent_runtime"
    )
    call_match = re.search(r'get_model_context_length_async\([^)]*\)', window, re.DOTALL)
    if call_match:
        call_text = call_match.group(0)
        assert "self._model" not in call_text, (
            f"self._model still in call: {call_text[:200]}"
        )
        assert "self._base_url" not in call_text, (
            f"self._base_url still in call: {call_text[:200]}"
        )


def test_warning_log_level_for_expansion_failure():
    src = Path("/tmp/hermes-pr-work-60859/hermes-agent/gateway/run.py").read_text()
    marker = "@ context reference expansion failed"
    idx = src.find(marker)
    assert idx > 0, "Could not find @ context reference expansion failed log"
    line_before = src[max(0, idx-200):idx]
    last_logger = line_before.rfind("logger.")
    assert last_logger >= 0
    logger_call = line_before[last_logger:idx]
    assert "logger.warning" in logger_call, (
        f"Swallow log should be logger.warning, got: {logger_call!r}"
    )


def test_works_on_gateway_runner_no_attribute_error():
    src = Path("/tmp/hermes-pr-work-60859/hermes-agent/gateway/run.py").read_text()
    prep_start = src.find("async def _prepare_inbound_message_text(")
    prep_end = src.find("\n    async def ", prep_start + 100)
    if prep_end < 0:
        prep_end = len(src)
    body = src[prep_start:prep_end]
    # self._model and self._base_url should not appear as call args
    # (only in comments explaining the fix)
    import re
    # Find any call like get_model_context_length_async(...)
    calls = re.findall(r'get_model_context_length_async\([^)]*\)', body, re.DOTALL)
    for c in calls:
        assert "self._model" not in c, f"self._model in call: {c[:200]}"
        assert "self._base_url" not in c, f"self._base_url in call: {c[:200]}"


if __name__ == "__main__":
    test_prepare_inbound_uses_resolve_session_agent_runtime()
    print("PASS: test_prepare_inbound_uses_resolve_session_agent_runtime")
    test_warning_log_level_for_expansion_failure()
    print("PASS: test_warning_log_level_for_expansion_failure")
    test_works_on_gateway_runner_no_attribute_error()
    print("PASS: test_works_on_gateway_runner_no_attribute_error")
