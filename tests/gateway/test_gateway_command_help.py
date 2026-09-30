"""Gateway command help rendering tests."""

from pathlib import Path

import pytest

from agent.i18n import t
from agent.skill_commands import get_skill_commands, skill_command_collision_note
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from hermes_cli.commands import gateway_help_lines
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


def _help_runner(gated: bool):
    """A runner whose ``user-1`` is an ungated caller, or a non-admin under slash gating."""
    runner = _make_runner()
    if gated:
        runner.config = GatewayConfig(platforms={
            Platform.DISCORD: PlatformConfig(enabled=True, extra={"allow_admin_from": ["admin"]}),
        })
    return runner


_GATED_ALLOWED = {"help", "whoami"}


@pytest.mark.asyncio
@pytest.mark.parametrize("args", ["skills", "SKILL"])
@pytest.mark.parametrize("gated", [False, True])
async def test_help_skills_lists_every_skill_command(skill_home, args, gated):
    skills = {} if gated else get_skill_commands()
    notes = [] if gated else [f"⚠ {skill_command_collision_note('model')}"]
    expected = "\n".join([
        t("gateway.help.skill_header", count=len(skills)),
        *[f"`{cmd}` — {info['description']}" for cmd, info in sorted(skills.items())],
        *notes,
    ])

    reply = await _help_runner(gated)._handle_help_command(
        _make_event(f"/help {args}", Platform.DISCORD)
    )

    assert reply == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "gated", "skill_rows"),
    [
        ("MoDeL", False, ["`/compare-models` — Compare model outputs"]),
        ("MoDeL", True, []),
        ("no-such-command", False, []),
    ],
)
async def test_help_text_filters_commands_and_skills(skill_home, query, gated, skill_rows):
    allowed = _GATED_ALLOWED if gated else None
    command_rows = [line for line in gateway_help_lines(allowed) if query.lower() in line.lower()]
    skill_section = [t("gateway.help.skill_header", count=len(skill_rows)), *skill_rows] if skill_rows else []
    expected = "\n".join([
        t("gateway.help.header"),
        *command_rows,
        *skill_section,
        t("gateway.help.filtered_by", query=query),
    ])

    reply = await _help_runner(gated)._handle_help_command(
        _make_event(f"/help {query}", Platform.DISCORD)
    )

    assert reply == expected
