from gateway.response_filters import (
    is_autonomous_silence_response,
    is_intentional_silence_agent_result,
    is_intentional_silence_response,
    sanitize_dsml_markers,
)


def test_exact_silence_tokens_are_intentional_silence():
    for token in ("[SILENT]", " SILENT ", "NO_REPLY", "no reply"):
        assert is_intentional_silence_response(token)


def test_autonomous_silence_accepts_marker_with_own_line_note():
    """The loose rule for cron/webhook lanes: marker + explanation suppresses."""
    assert is_autonomous_silence_response("[SILENT]")
    assert is_autonomous_silence_response("[SILENT]\n\nNothing new this tick.")
    assert is_autonomous_silence_response("2 deals filtered\n\n[SILENT]")
    assert is_autonomous_silence_response("no_reply\nduplicate inbound, already handled")
    assert is_autonomous_silence_response("[SILENT] No changes detected")


# ── DSML sanitizer regression tests ──────────────────────────────────────────
_DSML = "\uFF5C" + "DSML" + "\uFF5C"


def test_dsml_strips_balanced_tool_calls_envelope():
    env = (
        "<" + _DSML + "tool_calls>\n"
        "<" + _DSML + 'invoke name="foo">\n'
        "<" + _DSML + 'parameter name="x" string="false">1<' + "/" + _DSML + "parameter>\n"
        "</" + _DSML + "invoke>\n"
        "</" + _DSML + "tool_calls>\n"
        "Visible prose"
    )
    out = sanitize_dsml_markers(env)
    assert out == "\nVisible prose"


def test_dsml_strips_orphaned_opener_to_end_of_string():
    leak = "prefix <" + _DSML + "tool_calls>\n<" + _DSML + 'invoke name="bar"'
    assert sanitize_dsml_markers(leak) == "prefix "


def test_dsml_strips_complete_tags():
    assert sanitize_dsml_markers("text </" + _DSML + "parameter> more") == "text  more"
    assert sanitize_dsml_markers('text <' + _DSML + 'invoke name="x"> body') == "text  body"


def test_dsml_strips_truncated_tag_without_gt():
    """Regression: max_tokens truncation emits a partial tag with no '>' (e.g.
    </|DSML|parame). The old regex required a trailing '>', so these leaked to
    the chat surface. Fixed Aug 6 2026 by making '>' optional."""
    assert sanitize_dsml_markers("stopping the flow:  </" + _DSML + "parame") == "stopping the flow:  "
    assert sanitize_dsml_markers("text <" + _DSML + "invo") == "text "


