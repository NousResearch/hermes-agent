"""Snowflake IDs in ``GET /api/config`` must be strings (#135122).

Split out of ``test_web_server.py``, which is over the file-size cap.
"""


def test_normalize_stringifies_ints_beyond_js_safe_range():
    """Snowflake IDs (> 2**53) must reach the SPA as strings: a browser parses a JSON
    number as a double, and the Settings autosave PUT then writes the rounded value
    back (``1535369580753592401`` -> ``1535369580753592300``), silently breaking the
    Discord free-response channel match. Small ints, bools and nested lists are kept."""
    from hermes_cli.web_server_config import _normalize_config_for_web

    result = _normalize_config_for_web({
        "model": "anthropic/claude-sonnet-4",
        "platforms": {"discord": {"extra": {
            "free_response_channels": 1535369580753592401,
            "ignored_channels": [1535369580753592401, 42],
            "require_mention": True,
            "rate_limit_retry_base_seconds": 120,
        }}},
    })
    extra = result["platforms"]["discord"]["extra"]
    assert extra["free_response_channels"] == "1535369580753592401"
    assert extra["ignored_channels"] == ["1535369580753592401", 42]
    assert extra["require_mention"] is True
    assert extra["rate_limit_retry_base_seconds"] == 120
