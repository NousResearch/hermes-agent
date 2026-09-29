"""Unit tests for AIAgent._coerce_api_error_detail.

The static method normalises structured provider error fields (dicts,
lists, nested objects) into a display-safe string for user-facing error
messages.  These tests cover the direct coercion logic in isolation,
independent of the full _summarize_api_error pipeline.
"""

from run_agent import AIAgent


# ── String passthrough ───────────────────────────────────────────────────


class TestCoerceStringPassthrough:
    def test_plain_string_returned_unchanged(self):
        assert AIAgent._coerce_api_error_detail("plain error") == "plain error"

    def test_empty_string_returned_as_is(self):
        assert AIAgent._coerce_api_error_detail("") == ""

    def test_whitespace_only_string_returned_as_is(self):
        # Caller strips / decides significance; coercion is passthrough.
        assert AIAgent._coerce_api_error_detail("  ") == "  "


# ── None ────────────────────────────────────────────────────────────────


class TestCoerceNone:
    def test_none_returns_empty_string(self):
        assert AIAgent._coerce_api_error_detail(None) == ""


# ── Dict: first-priority string keys ────────────────────────────────────


class TestCoerceDictFirstPriority:
    """Dicts whose first-priority key holds a non-empty string."""

    def test_message_key(self):
        assert AIAgent._coerce_api_error_detail({"message": "err"}) == "err"

    def test_detail_key(self):
        assert AIAgent._coerce_api_error_detail({"detail": "det"}) == "det"

    def test_error_key(self):
        assert AIAgent._coerce_api_error_detail({"error": "err"}) == "err"

    def test_code_key(self):
        assert AIAgent._coerce_api_error_detail({"code": "ERR001"}) == "ERR001"

    def test_type_key(self):
        assert AIAgent._coerce_api_error_detail({"type": "invalid_request"}) == "invalid_request"

    def test_message_takes_priority_over_detail(self):
        result = AIAgent._coerce_api_error_detail({"message": "msg", "detail": "det"})
        assert result == "msg"

    def test_empty_string_value_is_skipped(self):
        # Empty/whitespace values are skipped; falls through to json fallback.
        result = AIAgent._coerce_api_error_detail({"message": ""})
        assert result == '{"message": ""}'

    def test_whitespace_only_value_preserved_via_recursion(self):
        # Whitespace-only strings are skipped in the first loop (strip() is
        # falsy), but the second loop recurses into the raw value.  Since
        # _coerce_api_error_detail("  ") returns the string as-is, the
        # whitespace value surfaces rather than falling to json.dumps.
        result = AIAgent._coerce_api_error_detail({"message": "  "})
        assert result == "  "


# ── Dict: recursive descent ─────────────────────────────────────────────


class TestCoerceDictRecursive:
    """Dicts whose first-priority keys hold non-string values trigger recursion."""

    def test_nested_message_under_error(self):
        result = AIAgent._coerce_api_error_detail({"error": {"message": "nested"}})
        assert result == "nested"

    def test_deeply_nested_message(self):
        result = AIAgent._coerce_api_error_detail(
            {"error": {"message": {"message": "deep"}}}
        )
        assert result == "deep"

    def test_nested_dict_with_only_int_values(self):
        # First loop skips int, second loop recurses; int -> str via fallback.
        result = AIAgent._coerce_api_error_detail({"error": {"code": 400}})
        assert result == "400"

    def test_integer_code_at_top_level(self):
        # code key with int value: first loop skips (not str), second loop recurses.
        result = AIAgent._coerce_api_error_detail({"code": 400})
        assert result == "400"


# ── Dict: json fallback ─────────────────────────────────────────────────


class TestCoerceDictJsonFallback:
    """Dicts with no extractable string fall through to json.dumps."""

    def test_empty_dict(self):
        result = AIAgent._coerce_api_error_detail({})
        assert result == "{}"

    def test_dict_with_only_empty_string_values(self):
        result = AIAgent._coerce_api_error_detail({"message": "", "detail": ""})
        # Both first-priority keys have empty-string values -> json fallback.
        assert '"message": ""' in result
        assert '"detail": ""' in result

    def test_dict_with_non_string_values(self):
        result = AIAgent._coerce_api_error_detail({"code": 400, "retry_after": 60})
        # First loop: code=400 not str, retry_after=60 not str -> skip.
        # Second loop: code=400 -> _coerce(400) -> "400" (truthy) -> return.
        assert result == "400"


# ── List / tuple ────────────────────────────────────────────────────────


class TestCoerceListTuple:
    def test_list_of_strings(self):
        result = AIAgent._coerce_api_error_detail(["err1", "err2"])
        assert result == "err1; err2"

    def test_tuple_of_strings(self):
        result = AIAgent._coerce_api_error_detail(("a", "b"))
        assert result == "a; b"

    def test_list_of_dicts_with_message(self):
        result = AIAgent._coerce_api_error_detail(
            [{"message": "e1"}, {"message": "e2"}]
        )
        assert result == "e1; e2"

    def test_list_with_empty_items_filtered(self):
        result = AIAgent._coerce_api_error_detail(["", "err", ""])
        assert result == "err"

    def test_empty_list_returns_empty_string(self):
        assert AIAgent._coerce_api_error_detail([]) == ""

    def test_single_item_list(self):
        assert AIAgent._coerce_api_error_detail(["only"]) == "only"


# ── Scalar non-string types ─────────────────────────────────────────────


class TestCoerceScalars:
    def test_integer(self):
        assert AIAgent._coerce_api_error_detail(42) == "42"

    def test_float(self):
        assert AIAgent._coerce_api_error_detail(3.14) == "3.14"

    def test_boolean_true(self):
        assert AIAgent._coerce_api_error_detail(True) == "True"

    def test_boolean_false(self):
        assert AIAgent._coerce_api_error_detail(False) == "False"


# ── Real-world provider shapes ──────────────────────────────────────────


class TestCoerceRealWorldShapes:
    def test_openai_style_error_dict(self):
        result = AIAgent._coerce_api_error_detail({
            "error": {
                "message": "This model's maximum context length is 262144 tokens.",
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
            }
        })
        assert result == "This model's maximum context length is 262144 tokens."

    def test_hf_router_nested_message(self):
        result = AIAgent._coerce_api_error_detail({
            "error": {
                "message": {
                    "type": "Bad Request",
                    "code": "context_length_exceeded",
                    "message": "Please reduce the length of the messages.",
                }
            }
        })
        assert result == "Please reduce the length of the messages."

    def test_flat_message_dict(self):
        result = AIAgent._coerce_api_error_detail({
            "message": "Rate limit exceeded"
        })
        assert result == "Rate limit exceeded"

    def test_list_of_errors(self):
        result = AIAgent._coerce_api_error_detail([
            {"message": "error one"},
            {"message": "error two"},
        ])
        assert result == "error one; error two"
