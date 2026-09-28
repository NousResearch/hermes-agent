"""Command reply formatting for messaging clients."""

import re
from typing import Any


_MATRIX_CODE_COMMAND_RE = re.compile(r"^/([A-Za-z][A-Za-z0-9_-]*)(?=\s|$)")
_MATRIX_CODE_SPAN_RE = re.compile(
    r"(?<![`\\])(`+)(?!`)((?:[^\n]|\n(?![ \t]*\r?\n))+?)(?<!`)\1(?!`)"
)


def _platformize_command_mentions(text: str, platform: Any) -> str:
    """Use client command syntax in generated help and directory replies."""
    from gateway.run import _telegramize_command_mentions

    rendered = _telegramize_command_mentions(text, platform)
    platform_value = getattr(platform, "value", platform)
    if platform_value != "matrix":
        return rendered

    from agent.skill_commands import get_skill_commands
    from hermes_cli.commands import is_gateway_known_command
    from markdown.extensions.fenced_code import FencedBlockPreprocessor

    skill_command_names = {
        str(command).removeprefix("/") for command in get_skill_commands()
    }

    def _replace_code_span(span: re.Match[str]) -> str:
        delimiter, content = span.groups()
        match = _MATRIX_CODE_COMMAND_RE.match(content)
        if len(delimiter) == 1 and match:
            command_name = match.group(1)
            if (
                is_gateway_known_command(command_name)
                or command_name in skill_command_names
            ):
                content = f"!{command_name}{content[match.end() :]}"
        return f"{delimiter}{content}{delimiter}"

    normalized_lines: list[str] = []
    line_offsets = [0]
    for line in rendered.splitlines(keepends=True):
        normalized_lines.append(line.rstrip("\r\n").expandtabs(4))
        line_offsets.append(line_offsets[-1] + len(line))
    normalized = "\n".join(normalized_lines)

    parts: list[str] = []
    cursor = 0
    for fence in FencedBlockPreprocessor.FENCED_BLOCK_RE.finditer(normalized):
        start = line_offsets[normalized.count("\n", 0, fence.start())]
        end = line_offsets[normalized.count("\n", 0, fence.end()) + 1]
        parts.append(
            _MATRIX_CODE_SPAN_RE.sub(_replace_code_span, rendered[cursor:start])
        )
        parts.append(rendered[start:end])
        cursor = end
    parts.append(_MATRIX_CODE_SPAN_RE.sub(_replace_code_span, rendered[cursor:]))
    return "".join(parts)
