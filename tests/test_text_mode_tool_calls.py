"""Unit tests for text_mode_tool_calls.maybe_synthesize_tool_calls.

Run with: python -m unittest test_text_mode_tool_calls
"""
import json
import sys
import unittest
from types import SimpleNamespace

# Make the helper importable when tests run from this directory
sys.path.insert(0, ".")
from text_mode_tool_calls import (  # noqa: E402
    maybe_synthesize_tool_calls,
    _model_is_text_mode,
    _normalize_to_tool_call_shape,
    _extract_candidates,
)


def _msg(content, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


class TestModelAllowlist(unittest.TestCase):
    def test_groq_in_allowlist(self):
        self.assertTrue(_model_is_text_mode("groq/llama-3.3-70b-versatile"))
        self.assertTrue(_model_is_text_mode("groq-fast"))
        self.assertTrue(_model_is_text_mode("GROQ/llama-3.1-8b"))  # case-insensitive

    def test_anthropic_not_in_allowlist(self):
        self.assertFalse(_model_is_text_mode("anthropic/claude-sonnet-4-6"))
        self.assertFalse(_model_is_text_mode("claude-haiku"))

    def test_openai_not_in_allowlist(self):
        self.assertFalse(_model_is_text_mode("openai/gpt-4o-mini"))

    def test_cerebras_in_allowlist(self):
        self.assertTrue(_model_is_text_mode("cerebras-llama"))

    def test_qwen_in_allowlist(self):
        self.assertTrue(_model_is_text_mode("qwen3-5-flash-or"))


class TestNormalize(unittest.TestCase):
    def test_canonical_shape_unchanged(self):
        out = _normalize_to_tool_call_shape({"name": "terminal", "arguments": {"cmd": "ls"}})
        self.assertEqual(out["name"], "terminal")
        self.assertEqual(json.loads(out["arguments"]), {"cmd": "ls"})

    def test_mistral_qwen_shape(self):
        out = _normalize_to_tool_call_shape({"tool": "memory", "args": {"query": "x"}})
        self.assertEqual(out["name"], "memory")
        self.assertEqual(json.loads(out["arguments"]), {"query": "x"})

    def test_string_arguments_passthrough(self):
        out = _normalize_to_tool_call_shape({"name": "X", "arguments": '{"k": 1}'})
        self.assertEqual(json.loads(out["arguments"]), {"k": 1})

    def test_missing_name_returns_none(self):
        self.assertIsNone(_normalize_to_tool_call_shape({"arguments": {}}))

    def test_empty_args_ok(self):
        out = _normalize_to_tool_call_shape({"name": "ping"})
        self.assertEqual(json.loads(out["arguments"]), {})


class TestFormatA_TagWrapped(unittest.TestCase):
    """<tool_call>{...}</tool_call> — the groq llama 3.3 format."""
    def test_single_tagged_call(self):
        content = 'Calling tool: <tool_call>{"name": "terminal", "arguments": {"cmd": "ls"}}</tool_call>'
        cands = _extract_candidates(content)
        self.assertEqual(len(cands), 1)
        self.assertEqual(cands[0]["name"], "terminal")

    def test_multiple_tagged_calls(self):
        content = (
            '<tool_call>{"name": "terminal", "arguments": {"cmd": "curl A"}}</tool_call>'
            ' and then '
            '<tool_call>{"name": "terminal", "arguments": {"cmd": "curl B"}}</tool_call>'
        )
        cands = _extract_candidates(content)
        self.assertEqual(len(cands), 2)


class TestFormatB_FencedJSON(unittest.TestCase):
    """Markdown-fenced JSON tool calls."""
    def test_fenced_with_json_marker(self):
        content = '''Here you go:
```json
{"name": "memory", "arguments": {"query": "test"}}
```
'''
        cands = _extract_candidates(content)
        self.assertEqual(len(cands), 1)
        self.assertEqual(cands[0]["name"], "memory")

    def test_fenced_without_marker(self):
        content = '''```
{"tool": "terminal", "args": {"cmd": "echo ok"}}
```'''
        cands = _extract_candidates(content)
        self.assertEqual(len(cands), 1)

    def test_fenced_arbitrary_json_skipped(self):
        # Fenced JSON that isn't tool-shaped should NOT be synthesized
        content = '''```json
{"answer": 42, "explanation": "..."}
```'''
        cands = _extract_candidates(content)
        self.assertEqual(len(cands), 0)


class TestFormatC_BareJSON(unittest.TestCase):
    """Content IS the JSON, no prose."""
    def test_bare_top_level_call(self):
        content = '{"name": "terminal", "arguments": {"cmd": "ls"}}'
        cands = _extract_candidates(content)
        self.assertEqual(len(cands), 1)

    def test_bare_with_whitespace(self):
        content = '\n  {"name": "X", "arguments": {}}  \n'
        cands = _extract_candidates(content)
        self.assertEqual(len(cands), 1)

    def test_bare_list_of_calls(self):
        content = '[{"name": "A", "arguments": {}}, {"name": "B", "arguments": {}}]'
        cands = _extract_candidates(content)
        self.assertEqual(len(cands), 2)


class TestFalsePositiveGuards(unittest.TestCase):
    def test_plain_text_no_synthesis(self):
        msg = _msg("Hello world. The answer is 42.")
        result = maybe_synthesize_tool_calls(msg, model="groq/llama-3.3-70b-versatile")
        self.assertFalse(result)
        self.assertIsNone(getattr(msg, "tool_calls", None))

    def test_anthropic_skipped_even_if_text_json(self):
        # Even if content has tool-call shape, don't synthesize for anthropic
        content = '<tool_call>{"name": "terminal", "arguments": {"cmd": "ls"}}</tool_call>'
        msg = _msg(content)
        result = maybe_synthesize_tool_calls(msg, model="anthropic/claude-sonnet-4-6")
        self.assertFalse(result)

    def test_idempotence_existing_tool_calls(self):
        existing = [SimpleNamespace(id="real", function=SimpleNamespace(name="X", arguments="{}"))]
        msg = _msg('<tool_call>{"name":"Y","arguments":{}}</tool_call>', tool_calls=existing)
        result = maybe_synthesize_tool_calls(msg, model="groq/llama-3.3-70b-versatile")
        self.assertFalse(result)
        # tool_calls unchanged
        self.assertEqual(len(msg.tool_calls), 1)
        self.assertEqual(msg.tool_calls[0].id, "real")

    def test_empty_content_no_synthesis(self):
        msg = _msg("")
        result = maybe_synthesize_tool_calls(msg, model="groq/llama-3.3-70b-versatile")
        self.assertFalse(result)

    def test_arbitrary_structured_json_no_synthesis(self):
        # Output looks like JSON but doesn't have tool keys
        content = '{"final_answer": "42", "confidence": 0.95}'
        msg = _msg(content)
        result = maybe_synthesize_tool_calls(msg, model="groq/llama-3.3-70b-versatile")
        self.assertFalse(result)


class TestEndToEndSynthesis(unittest.TestCase):
    def test_synthesis_attaches_duck_typed_tool_calls(self):
        content = '<tool_call>{"name": "terminal", "arguments": {"cmd": "curl /health"}}</tool_call>'
        msg = _msg(content)
        result = maybe_synthesize_tool_calls(msg, model="groq/llama-3.3-70b-versatile")
        self.assertTrue(result)
        tcs = msg.tool_calls
        self.assertEqual(len(tcs), 1)
        tc = tcs[0]
        # Required fields the agent dispatcher reads
        self.assertTrue(tc.id.startswith("call_synth_"))
        self.assertEqual(tc.id, tc.call_id)
        self.assertIsNone(tc.response_item_id)
        self.assertEqual(tc.type, "function")
        self.assertEqual(tc.function.name, "terminal")
        self.assertEqual(json.loads(tc.function.arguments), {"cmd": "curl /health"})
        # Provenance markers
        self.assertTrue(getattr(msg, "_synthesized_tool_calls", False))
        self.assertEqual(getattr(msg, "_synthesized_format_count", 0), 1)

    def test_double_call_run_is_noop(self):
        """Running synthesizer twice must not double tool_calls."""
        content = '<tool_call>{"name": "X", "arguments": {}}</tool_call>'
        msg = _msg(content)
        first = maybe_synthesize_tool_calls(msg, model="groq/llama-3.3-70b-versatile")
        self.assertTrue(first)
        first_count = len(msg.tool_calls)
        second = maybe_synthesize_tool_calls(msg, model="groq/llama-3.3-70b-versatile")
        self.assertFalse(second)  # idempotent — already has tool_calls
        self.assertEqual(len(msg.tool_calls), first_count)

    def test_deterministic_ids(self):
        """Same content + name + idx → same call_id (so retries hash identically)."""
        msg1 = _msg('<tool_call>{"name": "X", "arguments": {}}</tool_call>')
        msg2 = _msg('<tool_call>{"name": "X", "arguments": {}}</tool_call>')
        maybe_synthesize_tool_calls(msg1, model="groq/llama-3.3-70b-versatile")
        maybe_synthesize_tool_calls(msg2, model="groq/llama-3.3-70b-versatile")
        self.assertEqual(msg1.tool_calls[0].id, msg2.tool_calls[0].id)


class TestMalformedInput(unittest.TestCase):
    def test_malformed_json_in_tags_skipped(self):
        content = '<tool_call>{"name": "terminal", "arguments":}</tool_call>'  # syntax error
        cands = _extract_candidates(content)
        self.assertEqual(len(cands), 0)

    def test_single_quote_repair(self):
        # Some models emit single-quoted JSON. The light repair pass should handle simple cases.
        content = "<tool_call>{'name': 'terminal', 'arguments': {'cmd': 'ls'}}</tool_call>"
        cands = _extract_candidates(content)
        # We only do single→double quote substitution if strict json.loads fails
        self.assertEqual(len(cands), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
