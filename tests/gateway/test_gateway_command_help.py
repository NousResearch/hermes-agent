"""Gateway command help rendering tests."""

import re
from pathlib import Path

import pytest

from agent.i18n import t
from agent.skill_commands import get_skill_commands, skill_command_collision_note
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import _telegramize_command_mentions
from gateway.session import SessionSource
from hermes_cli.commands import gateway_help_lines, resolve_command
from hermes_constants import get_hermes_home


def _make_event(text: str, platform: Platform) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=platform,
            chat_id="chat-1",
            user_id="user-1",
            user_name="tester",
            chat_type="dm",
        ),
    )


def _make_runner():
    from gateway.run import GatewayRunner

    return object.__new__(GatewayRunner)


@pytest.mark.asyncio
async def test_help_sanitizes_slash_command_mentions_for_telegram(monkeypatch):
    """Telegram help output must not expose invalid uppercase/hyphenated slashes."""
    monkeypatch.setattr(
        "agent.skill_commands.get_skill_commands",
        lambda: {
            "/Linear": {"description": "Open Linear"},
            "/Custom-Thing": {"description": "Run a custom thing"},
        },
    )

    result = await _make_runner()._handle_help_command(
        _make_event("/help", Platform.TELEGRAM)
    )

    assert "`/linear`" in result
    assert "`/custom_thing`" in result
    assert "`/Linear`" not in result
    assert "`/Custom-Thing`" not in result


@pytest.mark.asyncio
async def test_commands_sanitizes_slash_command_mentions_for_telegram(monkeypatch):
    """Paginated Telegram /commands output uses Telegram-valid slash mentions."""
    monkeypatch.setattr(
        "agent.skill_commands.get_skill_commands",
        lambda: {"/Linear": {"description": "Open Linear"}},
    )

    result = await _make_runner()._handle_commands_command(
        _make_event("/commands 999", Platform.TELEGRAM)
    )

    assert "`/linear`" in result
    assert "`/Linear`" not in result


def _write_skill(home: Path, name: str, description: str) -> None:
    directory = home / "skills" / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\nBody.\n",
        encoding="utf-8",
    )


@pytest.fixture
def skill_home() -> Path:
    """Eleven skill commands, one skill that mentions models, and a skill named
    like the built-in ``/model``."""
    home = get_hermes_home()
    for index in range(10):
        _write_skill(home, f"research-{index:02d}", f"Research task {index:02d}")
    _write_skill(home, "compare-models", "Compare model outputs")
    _write_skill(home, "model", "Shadowed by the built-in")
    return home


def _help_runner(gated: bool, platform: Platform = Platform.DISCORD):
    """A runner whose ``user-1`` is an ungated caller, or a non-admin under slash gating."""
    runner = _make_runner()
    if gated:
        runner.config = GatewayConfig(platforms={
            platform: PlatformConfig(enabled=True, extra={"allow_admin_from": ["admin"]}),
        })
    return runner


_GATED_ALLOWED = {"help", "whoami"}


@pytest.mark.asyncio
@pytest.mark.parametrize("args", ["skills", "SKILL"])
@pytest.mark.parametrize("gated", [False, True])
async def test_help_skills_lists_every_skill_command(skill_home, args, gated):
    """Gated non-admins can run no skill command, so they get neither the list nor the notes."""
    if gated:
        expected = t("cli.help.no_skill_commands")
    else:
        skills = get_skill_commands()
        expected = "\n".join([
            t("gateway.help.skill_header", count=len(skills)),
            *[f"`{cmd}` — {info['description']}" for cmd, info in sorted(skills.items())],
            f"⚠ {skill_command_collision_note('model')}",
        ])

    reply = await _help_runner(gated)._handle_help_command(
        _make_event(f"/help {args}", Platform.DISCORD)
    )

    assert reply == expected


def _command_matches(row: str, query: str) -> bool:
    """Whether *query* occurs in the name, an alias, the argument hint or the description of
    the command in *row*, ignoring case and reading ``_`` as ``-``."""
    match = re.match(r"`/([^` ]+)", row)
    assert match is not None
    cmd = resolve_command(match.group(1))
    assert cmd is not None
    fields = [f"/{cmd.name}", *(f"/{alias}" for alias in cmd.aliases), cmd.args_hint, cmd.describe()]
    needle = query.lower().replace("_", "-")
    return any(needle in field.lower().replace("_", "-") for field in fields)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "platform", "gated", "matches_commands", "skill_rows"),
    [
        ("MoDeL", Platform.DISCORD, False, True, ["`/compare-models` — Compare model outputs"]),
        ("MoDeL", Platform.DISCORD, True, False, []),
        ("reload_mcp", Platform.TELEGRAM, False, True, []),
        ("`", Platform.DISCORD, False, True, []),
        ("no-such-command", Platform.DISCORD, False, False, []),
    ],
)
async def test_help_text_filters_commands_and_skills(
    skill_home, query, platform, gated, matches_commands, skill_rows
):
    all_rows = gateway_help_lines(_GATED_ALLOWED if gated else None)
    command_rows = [row for row in all_rows if _command_matches(row, query)]
    skill_section = ["", t("gateway.commands.skill_header"), *skill_rows] if skill_rows else []
    expected = _telegramize_command_mentions("\n".join([
        t("gateway.help.header"),
        *command_rows,
        *skill_section,
        t("gateway.help.filtered_by", query=query),
    ]), platform)

    reply = await _help_runner(gated, platform)._handle_help_command(
        _make_event(f"/help {query}", platform)
    )

    assert (bool(command_rows), len(command_rows) < len(all_rows), reply) == (
        matches_commands, True, expected
    )


@pytest.mark.asyncio
async def test_help_uses_matrix_safe_bang_command_mentions(monkeypatch):
    monkeypatch.setattr(
        "agent.skill_commands.get_skill_commands",
        lambda: {"/arxiv": {"description": "Search arXiv"}},
    )

    result = await _make_runner()._handle_help_command(
        _make_event("/help", Platform.MATRIX)
    )

    assert "`!model" in result
    assert "`!commands" in result
    assert "`!arxiv`" in result
    assert "`/model" not in result
    assert "`/commands" not in result
    assert "`/arxiv`" not in result


@pytest.mark.asyncio
async def test_commands_uses_matrix_safe_bang_entries_and_navigation(monkeypatch):
    monkeypatch.setattr(
        "agent.skill_commands.get_skill_commands",
        lambda: {"/arxiv": {"description": "Search arXiv"}},
    )

    result = await _make_runner()._handle_commands_command(
        _make_event("/commands 1", Platform.MATRIX)
    )

    assert "`!" in result
    assert "`!commands 2`" in result
    assert "`/commands 2`" not in result


@pytest.mark.asyncio
async def test_commands_keeps_slash_prefix_on_non_matrix_gateway(monkeypatch):
    monkeypatch.setattr("agent.skill_commands.get_skill_commands", lambda: {})

    result = await _make_runner()._handle_commands_command(
        _make_event("/commands 1", Platform.DISCORD)
    )

    assert "`/start" in result
    assert "`!start" not in result


def test_matrix_command_rendering_preserves_paths_and_multi_backtick_code():
    from gateway.run import _platformize_command_mentions

    source = (
        "Use `/tmp` and `/tmp/file`; keep ``/commands`` and "
        "`` `/commands` `` unchanged.\n"
        "```text\n`/commands`\n```\n"
        "```/commands``` also stays unchanged."
    )

    assert _platformize_command_mentions(source, Platform.MATRIX) == source


