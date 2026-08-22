"""Regression coverage for profile-specific cwd in a multiplexed gateway."""

import asyncio

from pathlib import Path

import pytest

from agent.runtime_cwd import resolve_agent_cwd
from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionContext, SessionSource
import tools.terminal_tool as terminal_tool
from tools.file_tools import _resolve_path_for_task


def _context(profile: str, session_key: str) -> SessionContext:
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id=f"{profile}-chat",
        user_id=f"{profile}-user",
        profile=profile,
    )
    return SessionContext(
        source=source,
        connected_platforms=[],
        home_channels={},
        session_key=session_key,
    )


@pytest.mark.asyncio
async def test_multiplex_sessions_seed_cwd_from_their_routed_profile(
    tmp_path, monkeypatch
):
    sales = tmp_path / "Sales"
    general = tmp_path / "General"
    sales.mkdir()
    general.mkdir()

    homes = {}
    for profile, cwd in (("bellie", sales), ("boop", general)):
        home = tmp_path / "profiles" / profile
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(
            f"terminal:\n  cwd: {cwd}\n", encoding="utf-8"
        )
        homes[profile] = home

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.adapters = {}
    monkeypatch.setattr(
        runner,
        "_resolve_profile_home_for_source",
        lambda source: homes[source.profile],
    )
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    monkeypatch.setenv("TERMINAL_CWD", str(general))

    bellie = _context("bellie", "agent:bellie:discord:dm:bellie-chat")
    boop = _context("boop", "agent:boop:discord:dm:boop-chat")

    ready = 0
    both_ready = asyncio.Event()

    async def bind_and_observe(context):
        nonlocal ready
        tokens = runner._set_session_env(context)
        try:
            ready += 1
            if ready == 2:
                both_ready.set()
            await both_ready.wait()
            return resolve_agent_cwd(), terminal_tool.get_session_cwd(
                context.session_key
            )
        finally:
            runner._clear_session_env(tokens)

    bellie_result, boop_result = await asyncio.gather(
        bind_and_observe(bellie), bind_and_observe(boop)
    )
    assert bellie_result == (sales, str(sales))
    assert boop_result == (general, str(general))
    assert _resolve_path_for_task("notes.md", bellie.session_key) == sales / "notes.md"
    assert _resolve_path_for_task("notes.md", boop.session_key) == general / "notes.md"

    # A session-local `cd` remains authoritative on later turns; profile config
    # only seeds sessions that do not have a live cwd record yet.
    sales_subdir = sales / "Q3"
    sales_subdir.mkdir()
    terminal_tool.record_session_cwd(bellie.session_key, str(sales_subdir))

    tokens = runner._set_session_env(bellie)
    try:
        assert resolve_agent_cwd() == sales_subdir
        assert terminal_tool.get_session_cwd(bellie.session_key) == str(sales_subdir)
    finally:
        runner._clear_session_env(tokens)
