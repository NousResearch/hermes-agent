"""Command reply presentation contracts for Matrix clients. Regression for #96361."""

import json
import re
from html import escape
from pathlib import Path

import pytest

from agent.i18n import SUPPORTED_LANGUAGES, get_language, reset_language_cache, t
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from hermes_cli.commands import (
    GATEWAY_KNOWN_COMMANDS,
    gateway_help_lines,
    is_gateway_known_command,
)
from hermes_cli.commands_platforms import _sanitize_telegram_name
from hermes_cli.slash_exec import CommandContext, execute_command


_DESCRIPTION = (
    "Try `/help skills`; paths `/tmp`, `/tmp/file`, `/model/log`, `/unregistered`.\n"
    "Keep ``/commands``, `` `/commands` ``, ```/commands```, `echo /commands`, "
    "`https://example.org/commands`, plain /commands and https://example.org/commands.\n"
    "```text\n`/commands`\n```\n"
    "~~~text\n`/help`\n~~~"
)


@pytest.fixture
def installed_skill_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, dict[str, str]]:
    from agent import skill_commands

    home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(skill_commands, "_skill_commands", {})
    expected = {}
    for index in range(12):
        name = f"research-{index:02d}"
        description = _DESCRIPTION if index == 0 else f"Research task {index:02d}"
        directory = home / "skills" / name
        directory.mkdir(parents=True)
        skill_md = directory / "SKILL.md"
        skill_md.write_text(
            f"---\nname: {name}\ndescription: {json.dumps(description)}\n---\n\nResearch.\n",
            encoding="utf-8",
        )
        expected[f"/{name}"] = {
            "name": name,
            "description": description,
            "skill_md_path": str(skill_md),
            "skill_dir": str(directory),
        }

    registered = skill_commands.scan_skill_commands()
    assert registered == expected
    return registered


def _write_skill(home: Path, name: str, description: str = "Research") -> None:
    directory = home / "skills" / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {json.dumps(description)}\n---\n\nResearch.\n",
        encoding="utf-8",
    )


def _event(text: str, platform: Platform) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=platform,
            chat_id="chat-1",
            user_id="user-1",
            chat_type="dm",
        ),
    )


def _matrix_registry_descriptions() -> dict[str, str]:
    """Registry descriptions as Matrix help shows them."""
    changes = {
        "help": [("/help", "!help"), ("<text>", "&lt;text&gt;")],
        "save": [("/save", "!save")],
        "pause": [("/pause", "!pause")],
        "bundles": [("<name>", "&lt;name&gt;")],
    }
    native = {}
    for command, replacements in changes.items():
        description = t(f"slash.{command}.description")
        native[description] = description
        for old, new in replacements:
            native[description] = native[description].replace(old, new)
    return native


def _expected_reply(
    canonical: str, platform: Platform, skills: dict[str, dict[str, str]]
) -> str:
    commands = GATEWAY_KNOWN_COMMANDS | {key.removeprefix("/") for key in skills}
    parts = canonical.split(_DESCRIPTION)
    for command in sorted(commands):
        display = f"/{command}"
        if platform == Platform.MATRIX:
            display = f"!{command}"
        if platform == Platform.TELEGRAM:
            display = f"/{_sanitize_telegram_name(command)}"
        for delimiter in ("`", " "):
            parts = [
                part.replace(f"`/{command}{delimiter}", f"`{display}{delimiter}")
                for part in parts
            ]
    description = _DESCRIPTION
    if platform == Platform.MATRIX:
        for registry, native in _matrix_registry_descriptions().items():
            parts = [part.replace(registry, native) for part in parts]
        description = description.replace("`/help skills`", "`!help skills`").replace(
            "plain /commands", "plain !commands"
        )
    return description.join(parts)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "platform", [Platform.MATRIX, Platform.DISCORD, Platform.TELEGRAM]
)
@pytest.mark.parametrize("command", ["help", "commands"])
async def test_command_catalog_replies_use_client_tokens(
    platform: Platform,
    command: str,
    installed_skill_commands: dict[str, dict[str, str]],
):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    handler = getattr(runner, f"_handle_{command}_command")
    page_size = 15 if platform == Platform.TELEGRAM else 20
    if command == "help":
        contexts = [CommandContext(surface="gateway")]
    else:
        entry_count = len(gateway_help_lines()) + 2 + len(installed_skill_commands)
        pages = (entry_count + page_size - 1) // page_size
        contexts = [
            CommandContext(
                surface="gateway", args=str(page), options={"page_size": page_size}
            )
            for page in [*range(1, pages + 1), 999, "invalid"]
        ]

    actual = [
        await handler(_event(f"/{command} {context.args}".rstrip(), platform))
        for context in contexts
    ]
    expected = [
        _expected_reply(
            execute_command(command, context).text, platform, installed_skill_commands
        )
        for context in contexts
    ]
    assert actual == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "platform", [Platform.MATRIX, Platform.DISCORD, Platform.TELEGRAM]
)
async def test_skills_help_lists_all_installed_commands(
    platform: Platform, installed_skill_commands: dict[str, dict[str, str]]
):
    from agent.skill_commands import skill_command_collision_note
    from gateway.run import GatewayRunner
    from hermes_constants import get_hermes_home

    _write_skill(get_hermes_home(), "model")
    canonical = "\n".join([
        t("gateway.help.skill_header", count=len(installed_skill_commands)),
        *[
            f"`{command}` — {info['description']}"
            for command, info in sorted(installed_skill_commands.items())
        ],
        f"⚠ {skill_command_collision_note('model')}",
    ])
    actual = await object.__new__(GatewayRunner)._handle_help_command(
        _event("/help skills", platform)
    )
    assert actual == _expected_reply(canonical, platform, installed_skill_commands)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("platform", "disabled"),
    [
        (Platform.MATRIX, ["matrix-off", "model"]),
        (Platform.DISCORD, ["discord-off"]),
        (Platform.TELEGRAM, []),
    ],
)
async def test_catalogues_omit_skills_disabled_for_the_platform(
    platform: Platform,
    disabled: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    installed = ["research", "matrix-off", "discord-off", "model"]
    descriptions = {"research": "Pair with `/matrix-off` or `/discord-off`."}
    pages = range(1, len(gateway_help_lines()) // 10 + 2)

    async def catalogues(home: Path, skills: list[str], config: str) -> list[str]:
        monkeypatch.setenv("HERMES_HOME", str(home))
        home.mkdir()
        (home / "config.yaml").write_text(config, encoding="utf-8")
        for name in skills:
            _write_skill(home, name, descriptions.get(name, "Research"))
        requests = [
            ("help", ""),
            ("help", "skills"),
            *[("commands", str(page)) for page in pages],
        ]
        return [
            await getattr(runner, f"_handle_{command}_command")(
                _event(f"/{command} {args}".rstrip(), platform)
            )
            for command, args in requests
        ]

    monkeypatch.chdir(tmp_path)
    actual = await catalogues(
        tmp_path / "configured",
        installed,
        "skills:\n"
        "  platform_disabled:\n"
        "    matrix: [matrix-off, model]\n"
        "    discord: [discord-off]\n",
    )
    expected = await catalogues(
        tmp_path / "reference",
        [name for name in installed if name not in disabled],
        "{}\n",
    )
    assert actual == expected


@pytest.fixture
def display_language(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HERMES_LANGUAGE", request.param)
    reset_language_cache()
    yield request.param
    reset_language_cache()


@pytest.mark.asyncio
@pytest.mark.parametrize("display_language", SUPPORTED_LANGUAGES, indirect=True)
async def test_matrix_command_help_spells_every_command_with_a_bang(
    display_language: str,
):
    from gateway.run import GatewayRunner

    reply = await object.__new__(GatewayRunner)._handle_help_command(
        _event("/help", Platform.MATRIX)
    )
    mentions = re.findall(r"(?<![A-Za-z0-9_./~-])/([A-Za-z][A-Za-z0-9_-]*)", reply)
    leftovers = [name for name in mentions if is_gateway_known_command(name)]
    assert (get_language(), leftovers) == (display_language, [])


@pytest.mark.asyncio
@pytest.mark.parametrize("display_language", SUPPORTED_LANGUAGES, indirect=True)
async def test_matrix_command_help_shows_argument_placeholders(display_language: str):
    from gateway.run import GatewayRunner
    from plugins.platforms.matrix.adapter import MatrixAdapter

    canonical = execute_command("help", CommandContext(surface="gateway")).text
    reply = await object.__new__(GatewayRunner)._handle_help_command(
        _event("/help", Platform.MATRIX)
    )
    html = object.__new__(MatrixAdapter)._markdown_to_html(reply)
    placeholders = set(re.findall(r"<[^<>\n]+>", canonical))
    rendered = {
        placeholder: html.count(escape(placeholder, quote=False))
        for placeholder in placeholders
    }
    expected = {
        placeholder: canonical.count(placeholder) for placeholder in placeholders
    }
    assert (get_language(), rendered) == (display_language, expected)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("description", "native_description"),
    [
        (
            "Use ``literal ```/help` literal`` here. Try `/help skills`.",
            "Use ``literal ```/help` literal`` here. Try `!help skills`.",
        ),
        (
            "Use ``literal `/help` literal`` here. Try `/help skills`.",
            "Use ``literal `/help` literal`` here. Try `!help skills`.",
        ),
        (
            "Example:\n```/help```\nTry `/help skills`.",
            "Example:\n```/help```\nTry `!help skills`.",
        ),
        (
            "Example:\n   ```/help```\nTry `/help skills`.",
            "Example:\n   ```/help```\nTry `!help skills`.",
        ),
        (
            "Use ``literal\n`/help`\nliteral`` here. Try `/help skills`.",
            "Use ``literal\n`/help`\nliteral`` here. Try `!help skills`.",
        ),
        (
            "Try `/help\nskills`.",
            "Try `!help\nskills`.",
        ),
        (
            "Example:\n ```text\n`/help`\n ```\nTry `/help skills`.",
            "Example:\n ```text\n`/help`\n ```\nTry `!help skills`.",
        ),
        (
            "Example:\n```text\n`/help`\n```\nTry `/help skills`.",
            "Example:\n```text\n`/help`\n```\nTry `!help skills`.",
        ),
        (
            "Example:\n~~~text\n`/help`\n~~~\nTry `/help skills`.",
            "Example:\n~~~text\n`/help`\n~~~\nTry `!help skills`.",
        ),
        (
            "Example:\r\n~~~text\r\n`/help`\r\n~~~\r\nTry `/help skills`.",
            "Example:\r\n~~~text\r\n`/help`\r\n~~~\r\nTry `!help skills`.",
        ),
        (
            "Example:\n~~~text\t\n`/help`\n~~~\t\nTry `/help skills`.",
            "Example:\n~~~text\t\n`/help`\n~~~\t\nTry `!help skills`.",
        ),
        (
            "Example:\n````text\n```\n`/help`\n`````\n````\nTry `/help skills`.",
            "Example:\n````text\n```\n`/help`\n`````\n````\nTry `!help skills`.",
        ),
        (
            "Example:\n```{.text #example}\n`/help`\n```\nTry `/help skills`.",
            "Example:\n```{.text #example}\n`/help`\n```\nTry `!help skills`.",
        ),
        (
            r"Use \\`echo /help` here. Try `/help skills`.",
            r"Use \\`echo /help` here. Try `!help skills`.",
        ),
        (
            r"Use \\\\``literal `/help` literal`` here. Try `/help skills`.",
            r"Use \\\\``literal `/help` literal`` here. Try `!help skills`.",
        ),
        (
            r"Try \\`/help skills`.",
            r"Try \\`!help skills`.",
        ),
        (
            r"Use \`/help\` here. Try `/help skills`.",
            r"Use \`/help\` here. Try `!help skills`.",
        ),
        (
            "Example:\n\n    `/help`\n\nTry `/help skills`.",
            "Example:\n\n    `/help`\n\nTry `!help skills`.",
        ),
        (
            "Example:\r\n\r\n\t`/help`\r\n\r\nTry `/help skills`.",
            "Example:\r\n\r\n\t`/help`\r\n\r\nTry `!help skills`.",
        ),
        (
            "Example:\n\n>     `/help`\n\nTry `/help skills`.",
            "Example:\n\n>     `/help`\n\nTry `!help skills`.",
        ),
        (
            "Example:\n\n* Literal:\n\n        `/help`\n\nTry `/help skills`.",
            "Example:\n\n* Literal:\n\n        `/help`\n\nTry `!help skills`.",
        ),
        (
            "Example:\n    `/help`\nTry `/help skills`.",
            "Example:\n    `!help`\nTry `!help skills`.",
        ),
        (
            "Use `/help`` literal` here. Try `/help skills`.",
            "Use `/help`` literal` here. Try `!help skills`.",
        ),
        (
            "Use $`$ and `/help`.",
            "Use $`$ and `!help`.",
        ),
        (
            "Use $`/help`$ here. Try `/help skills`.",
            "Use $`/help`$ here. Try `!help skills`.",
        ),
        (
            "Use <script>`</script> and `/help`.",
            "Use <script>`</script> and `!help`.",
        ),
        (
            "Send /help skills, (/save) or '/pause off'; 直接输入/save查看; **/help**.",
            "Send !help skills, (!save) or '!pause off'; 直接输入!save查看; **!help**.",
        ),
        (
            r"Keep https://example.org/help, /usr/bin, ~/help, ./help, a/help, "
            r"/help/x, /help.txt, \/help, `echo /help`, ` /help` and /unregistered.",
            r"Keep https://example.org/help, /usr/bin, ~/help, ./help, a/help, "
            r"/help/x, /help.txt, \/help, `echo /help`, ` /help` and /unregistered.",
        ),
        (
            "See [/help](https://example.org/help), [docs](/help), "
            "<https://example.org/help> and $x /help y$, then /help.",
            "See [/help](https://example.org/help), [docs](/help), "
            "<https://example.org/help> and $x /help y$, then !help.",
        ),
        (
            "Open https://app.example.org/#/status or https://example.org/a;/help, "
            "then #/status and /status.",
            "Open https://app.example.org/#/status or https://example.org/a;/help, "
            "then #/status and !status.",
        ),
        (
            'Raw <a href="https://example.org">/help</a>, <code>echo /help</code> '
            "and <pre>/help</pre>, then /help.",
            'Raw <a href="https://example.org">/help</a>, <code>echo /help</code> '
            "and <pre>/help</pre>, then !help.",
        ),
    ],
    ids=[
        "double-with-triple-and-single",
        "double-with-single",
        "inline-triple",
        "indented-inline-triple",
        "multiline-double",
        "multiline-command",
        "indented-multiline-span",
        "backtick-fence",
        "tilde-fence",
        "crlf-fence",
        "fence-trailing-tabs",
        "exact-fence-close",
        "fence-attributes",
        "even-backslashes-single",
        "even-backslashes-double",
        "even-backslashes-command",
        "odd-backslashes-literal",
        "indented-code",
        "tab-indented-code-crlf",
        "blockquote-indented-code",
        "list-indented-code",
        "indented-paragraph-continuation",
        "literal-backticks-after-token",
        "math-backtick-before-command",
        "command-inside-math",
        "removed-script-backtick",
        "plain-commands",
        "plain-paths-urls-code-and-unknown-names",
        "plain-links-and-maths",
        "plain-url-fragments-and-punctuation",
        "plain-raw-html",
    ],
)
async def test_matrix_catalogues_preserve_literal_spans_and_later_commands(
    description: str,
    native_description: str,
    installed_skill_commands: dict[str, dict[str, str]],
):
    from gateway.run import GatewayRunner
    from plugins.platforms.matrix.adapter import MatrixAdapter

    installed_skill_commands["/research-00"]["description"] = description
    runner = object.__new__(GatewayRunner)
    adapter = object.__new__(MatrixAdapter)
    entry_count = len(gateway_help_lines()) + 2 + len(installed_skill_commands)
    pages = (entry_count + 19) // 20
    requests = [
        ("help", ""),
        ("help", "skills"),
        *[("commands", str(page)) for page in range(1, pages + 1)],
    ]
    actual = []
    expected = []
    for command, args in requests:
        reply = await getattr(runner, f"_handle_{command}_command")(
            _event(f"/{command} {args}".rstrip(), Platform.MATRIX)
        )
        canonical = execute_command(
            command, CommandContext(surface="gateway", args=args)
        ).text
        native = native_description.join(
            _expected_reply(part, Platform.MATRIX, installed_skill_commands)
            for part in canonical.split(description)
        )
        actual.append((reply, adapter._markdown_to_html(reply)))
        expected.append((native, adapter._markdown_to_html(native)))

    assert actual == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("unsupported_slug", ["123-research", "研究"])
async def test_matrix_help_only_advertises_native_bangs_for_invocable_skill_slugs(
    unsupported_slug: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from agent.skill_commands import scan_skill_commands
    from gateway.run import GatewayRunner
    from plugins.platforms.matrix.adapter import _normalize_matrix_bang_command

    home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    for slug in ["research", unsupported_slug]:
        directory = home / "skills" / slug
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text(
            f"---\nname: {slug}\ndescription: Research\n---\n\nResearch.\n",
            encoding="utf-8",
        )
    skills = scan_skill_commands()
    assert set(skills) == {"/research", f"/{unsupported_slug}"}
    assert (
        _normalize_matrix_bang_command(f"!{unsupported_slug}") == f"!{unsupported_slug}"
    )
    assert _normalize_matrix_bang_command("!research") == "/research"

    actual = await object.__new__(GatewayRunner)._handle_help_command(
        _event("/help skills", Platform.MATRIX)
    )
    assert actual == "\n".join([
        t("gateway.help.skill_header", count=len(skills)),
        *[
            f"`{'!research' if command == '/research' else command}` — Research"
            for command in sorted(skills)
        ],
    ])
