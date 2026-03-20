"""Extract tool calls from text content when structured tool_calls are absent."""

from typing import List, Optional, Tuple


def extract_text_tool_calls(
    content: Optional[str],
    existing_tool_calls: Optional[list],
) -> Tuple[Optional[str], Optional[list]]:
    """Parse <tool_call> XML from content when no structured tool_calls exist.

    Returns (cleaned_content, parsed_tool_calls) or (original_content, None).
    """
    # Skip if structured tool calls already present
    if existing_tool_calls:
        return content, None

    # Skip if no content to parse
    if not content or "<tool_call>" not in content:
        return content, None

    from environments.tool_call_parsers.hermes_parser import HermesToolCallParser

    parser = HermesToolCallParser()
    cleaned, tool_calls = parser.parse(content)

    if tool_calls:
        # Strip whitespace from cleaned content
        if cleaned:
            cleaned = cleaned.strip()
        return cleaned, tool_calls

    return content, None
