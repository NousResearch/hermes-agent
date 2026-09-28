"""Command reply formatting for messaging clients."""

import re
from typing import Any


_MATRIX_CODE_COMMAND_RE = re.compile(r"^/([A-Za-z][A-Za-z0-9_-]*)(?=\s|$)")


def _platformize_command_mentions(text: str, platform: Any) -> str:
    """Use client command syntax in generated help and directory replies."""
    from gateway.run import _telegramize_command_mentions

    rendered = _telegramize_command_mentions(text, platform)
    platform_value = getattr(platform, "value", platform)
    if platform_value != "matrix":
        return rendered

    from agent.skill_commands import get_skill_commands
    from hermes_cli.commands import is_gateway_known_command

    skill_command_names = {
        str(command).removeprefix("/") for command in get_skill_commands()
    }

    def _replace_single_backtick_spans(line: str) -> str:
        parts: list[str] = []
        cursor = 0
        while cursor < len(line):
            opening = line.find("`", cursor)
            if opening < 0:
                parts.append(line[cursor:])
                break
            parts.append(line[cursor:opening])
            run_end = opening
            while run_end < len(line) and line[run_end] == "`":
                run_end += 1
            delimiter = line[opening:run_end]
            closing = line.find(delimiter, run_end)
            if closing < 0:
                parts.append(line[opening:])
                break

            content = line[run_end:closing]
            match = _MATRIX_CODE_COMMAND_RE.match(content)
            if len(delimiter) == 1 and match:
                command_name = match.group(1)
                if (
                    is_gateway_known_command(command_name)
                    or command_name in skill_command_names
                ):
                    content = f"!{command_name}{content[match.end() :]}"
            parts.extend((delimiter, content, delimiter))
            cursor = closing + len(delimiter)
        return "".join(parts)

    lines: list[str] = []
    fence: tuple[str, int] | None = None
    for line in rendered.splitlines(keepends=True):
        stripped = line.lstrip(" \t")
        if fence is not None:
            lines.append(line)
            marker, minimum_length = fence
            marker_length = len(stripped) - len(stripped.lstrip(marker))
            if marker_length >= minimum_length and not stripped[marker_length:].strip():
                fence = None
            continue

        opening_fence = re.match(r"(`{3,}|~{3,})", stripped)
        if opening_fence:
            marker = opening_fence.group(1)
            fence = (marker[0], len(marker))
            lines.append(line)
            continue
        lines.append(_replace_single_backtick_spans(line))

    return "".join(lines)
