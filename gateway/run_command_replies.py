"""Command reply formatting for messaging clients."""

import re
from typing import Any
from xml.etree.ElementTree import Element


_MATRIX_CODE_COMMAND_RE = re.compile(r"^/([A-Za-z][A-Za-z0-9_-]*)(?=\s|$)")
_MATRIX_COMMAND_CANDIDATE_RE = re.compile(r"(?<=`)/([A-Za-z][A-Za-z0-9_-]*)(?=\s|`|$)")


def _platformize_command_mentions(text: str, platform: Any) -> str:
    """Use client command syntax in generated help and directory replies."""
    from gateway.run import _telegramize_command_mentions

    rendered = _telegramize_command_mentions(text, platform)
    platform_value = getattr(platform, "value", platform)
    if platform_value != "matrix":
        return rendered

    from agent.skill_commands import get_skill_commands
    from hermes_cli.commands import is_gateway_known_command
    from markdown import Markdown
    from markdown.inlinepatterns import BACKTICK_RE, BacktickInlineProcessor

    skill_command_names = {
        str(command).removeprefix("/") for command in get_skill_commands()
    }

    marker = "HERMESCOMMAND"
    while marker in rendered:
        marker += "_"
    marker_re = re.compile(rf"^/{marker}(\d+):")
    command_offsets: set[int] = set()

    def _mark_command(match: re.Match[str]) -> str:
        command_name = match.group(1)
        if (
            is_gateway_known_command(command_name)
            or command_name in skill_command_names
        ):
            return f"/{marker}{match.start()}:{match.group(0)}"
        return match.group(0)

    class CommandBacktickProcessor(BacktickInlineProcessor):
        def handleMatch(
            self, m: re.Match[str], data: str
        ) -> tuple[Element | str, int, int]:
            if m.group(2) == "`":
                command = marker_re.match(m.group(3))
                if command and _MATRIX_CODE_COMMAND_RE.match(
                    m.group(3)[command.end() :]
                ):
                    command_offsets.add(int(command.group(1)))
            return super().handleMatch(m, data)

    # Markers link parsed spans to the original reply because Markdown
    # normalises whitespace and discards source positions.
    md = Markdown(extensions=["fenced_code", "tables", "nl2br", "sane_lists"])
    md.preprocessors.deregister("html_block")
    md.inlinePatterns.register(CommandBacktickProcessor(BACKTICK_RE), "backtick", 190)
    md.convert(_MATRIX_COMMAND_CANDIDATE_RE.sub(_mark_command, rendered))
    return "".join(
        "!" if index in command_offsets else character
        for index, character in enumerate(rendered)
    )
