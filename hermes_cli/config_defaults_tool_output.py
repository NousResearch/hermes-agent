"""Tool-output defaults, kept separate from the main configuration table."""

TOOL_OUTPUT_DEFAULTS = {
    "max_bytes": 50000,
    "max_lines": 2000,
    "max_line_length": 2000,
    "json_compaction": {
        "mode": "off",
        "min_chars": 1024,
        "min_savings_ratio": 0.1,
        # File reads may be used to inspect exact formatting.
        "exclude_tools": ["read_file"],
    },
}
