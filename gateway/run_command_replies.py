"""Command reply formatting for messaging clients."""

import re
from typing import Any
from xml.etree.ElementTree import Element


_MATRIX_CODE_COMMAND_RE = re.compile(r"^/([A-Za-z][A-Za-z0-9_-]*)(?=\s|$)")
_MATRIX_COMMAND_CANDIDATE_RE = re.compile(
    r"(?<=`)/(?P<code>[A-Za-z][A-Za-z0-9_-]*)(?=\s|`|$)"
    r"|(?<![A-Za-z0-9_./:~`<\\@=-])/(?P<plain>[A-Za-z][A-Za-z0-9_-]*)"
    r"(?![A-Za-z0-9_/-]|\.[A-Za-z0-9])"
)
_MATRIX_HELP_ROW_RE = re.compile(r"^`[^`\n]+` (?:--|—) ", re.MULTILINE)
_MATRIX_LITERAL_TAGS = frozenset({"a", "code", "pre"})


def _matrix_description_spans(text: str, skill_commands: dict) -> list[range]:
    """The positions of the command and skill descriptions in help rows."""
    from hermes_cli.commands import COMMAND_REGISTRY

    descriptions = {command.describe() for command in COMMAND_REGISTRY}
    for info in skill_commands.values():
        description = str(info.get("description", ""))
        descriptions.update({description, description.strip()})
    descriptions.discard("")
    spans = []
    for row in _MATRIX_HELP_ROW_RE.finditer(text):
        start = row.end()
        length = max(
            (len(item) for item in descriptions if text.startswith(item, start)),
            default=0,
        )
        spans.append(range(start, start + length))
    return spans


def _platformize_command_mentions(text: str, platform: Any) -> str:
    """Use client command syntax in generated help and directory replies."""
    from gateway.run import _telegramize_command_mentions

    rendered = _telegramize_command_mentions(text, platform)
    platform_value = getattr(platform, "value", platform)
    if platform_value != "matrix":
        return rendered

    from agent.skill_commands import get_platform_skill_commands
    from hermes_cli.commands import is_gateway_known_command
    from markdown import Markdown
    from markdown.inlinepatterns import BACKTICK_RE, BacktickInlineProcessor
    from markdown.treeprocessors import Treeprocessor
    from plugins.platforms.matrix.adapter import _prepare_matrix_markdown

    skill_commands = get_platform_skill_commands(platform_value)
    skill_command_names = {str(command).removeprefix("/") for command in skill_commands}
    description_spans = _matrix_description_spans(rendered, skill_commands)

    marker = "HERMESCOMMAND"
    while marker in rendered:
        marker += "_"
    marker_re = re.compile(rf"^/{marker}(\d+):")
    any_marker_re = re.compile(rf"/{marker}(\d+):")
    command_offsets: set[int] = set()

    def _in_description(offset: int) -> bool:
        return any(offset in span for span in description_spans)

    def _should_mark(match: re.Match[str]) -> bool:
        start = match.start()
        command_name = match.group("code") or match.group("plain")
        if not (
            is_gateway_known_command(command_name)
            or command_name in skill_command_names
        ):
            return False
        if match.group("code"):
            return True
        return _in_description(start)

    def _mark(match: re.Match[str]) -> str:
        if not _should_mark(match):
            return match.group(0)
        return f"/{marker}{match.start()}:{match.group(0)}"

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

    class PlainTextCommandProcessor(Treeprocessor):
        def run(self, root: Element) -> None:
            self._collect(root)

        def _collect(self, element: Element) -> None:
            if element.tag in _MATRIX_LITERAL_TAGS:
                return
            self._record(element.text)
            for child in element:
                self._collect(child)
                self._record(child.tail)

        @staticmethod
        def _record(value: str | None) -> None:
            command_offsets.update(
                int(found.group(1)) for found in any_marker_re.finditer(value or "")
            )

    # Markers link parsed spans to the original reply because the Matrix
    # renderer's preprocessing and Markdown both rewrite the text and discard
    # source positions.
    md = Markdown(extensions=["fenced_code", "tables", "nl2br", "sane_lists"])
    md.preprocessors.deregister("html_block")
    md.inlinePatterns.register(CommandBacktickProcessor(BACKTICK_RE), "backtick", 190)
    md.treeprocessors.register(PlainTextCommandProcessor(md), "plain_commands", 15)
    marked, _ = _prepare_matrix_markdown(
        _MATRIX_COMMAND_CANDIDATE_RE.sub(_mark, rendered)
    )
    md.convert(marked)
    return "".join(
        "!" if index in command_offsets else character
        for index, character in enumerate(rendered)
    )
