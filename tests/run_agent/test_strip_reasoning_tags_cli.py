"""Tests for cli.py::_strip_reasoning_tags — specifically the tool-call
XML stripping added in openclaw/openclaw#67318 port.

The CLI has its own copy of the stripper because it needs to run on the
final displayed assistant text (after streaming) without depending on the
AIAgent instance. It must stay in sync with run_agent.py::_strip_think_blocks
for tool-call tag coverage."""


from cli import _strip_reasoning_tags


class TestToolCallStripping:
    def test_tool_call_block_stripped(self):
        text = '<tool_call>{"name": "x"}</tool_call>result'
        result = _strip_reasoning_tags(text)
        assert "<tool_call>" not in result
        assert "result" in result

    def test_function_calls_block_stripped(self):
        text = '<function_calls>[{}]</function_calls>\nanswer'
        result = _strip_reasoning_tags(text)
        assert "<function_calls>" not in result
        assert "answer" in result

    def test_gemma_function_name_block_stripped(self):
        text = (
            'Reading.\n'
            '<function name="r"><parameter name="p">/tmp/x</parameter></function>\n'
            'Done.'
        )
        result = _strip_reasoning_tags(text)
        assert '<function name="r">' not in result
        assert "/tmp/x" not in result
        assert "Reading." in result
        assert "Done." in result

    def test_prose_mention_of_function_preserved(self):
        text = "Use <function> declarations in JavaScript."
        result = _strip_reasoning_tags(text)
        assert "JavaScript" in result

    def test_reasoning_still_stripped(self):
        """Regression: make sure existing think-tag stripping still works."""
        text = "<think>reasoning</think> answer"
        result = _strip_reasoning_tags(text)
        assert "reasoning" not in result
        assert "answer" in result

    def test_mixed_reasoning_and_tool_call(self):
        text = '<think>plan</think><tool_call>{"x":1}</tool_call>final'
        result = _strip_reasoning_tags(text)
        assert "plan" not in result
        assert "<tool_call>" not in result
        assert "final" in result

    def test_stray_function_close(self):
        text = "visible</function> tail"
        result = _strip_reasoning_tags(text)
        assert "</function>" not in result
        assert "visible" in result
        assert "tail" in result

    def test_empty_string(self):
        assert _strip_reasoning_tags("") == ""

    def test_plain_text_unchanged(self):
        assert _strip_reasoning_tags("just text") == "just text"


class TestPseudoToolCallLogging:
    """#56461: a model (e.g. a local Ollama model that doesn't reliably
    support native function calling) can emit its tool call as plain text
    instead of the structured tool_calls field. The call never executes,
    but the model's narration ("Got it, saved!") survives untouched --
    silently stripping the pseudo-call with no trace made this
    undiagnosable. A warning must fire so it's visible in logs."""

    def test_stripped_tool_call_logs_warning(self, caplog):
        import logging
        text = 'Got it, saved!\n<tool_call>{"name": "memory"}</tool_call>'
        with caplog.at_level(logging.WARNING):
            result = _strip_reasoning_tags(text)
        assert "Got it, saved!" in result
        assert "<tool_call>" not in result
        assert any("Stripped a <tool_call>" in r.message for r in caplog.records)

    def test_stripped_function_name_block_logs_warning(self, caplog):
        import logging
        text = 'Reading.\n<function name="r"><parameter name="p">/tmp/x</parameter></function>'
        with caplog.at_level(logging.WARNING):
            _strip_reasoning_tags(text)
        assert any("Stripped a <function>" in r.message for r in caplog.records)

    def test_reasoning_tag_strip_does_not_log(self, caplog):
        """Only tool-call-shaped blocks are undiagnosable failures -- plain
        reasoning-tag stripping is normal, expected behavior and must not
        spam the log."""
        import logging
        with caplog.at_level(logging.WARNING):
            _strip_reasoning_tags("<think>plan</think> answer")
        assert not caplog.records

    def test_plain_text_does_not_log(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            _strip_reasoning_tags("just a normal reply")
        assert not caplog.records
