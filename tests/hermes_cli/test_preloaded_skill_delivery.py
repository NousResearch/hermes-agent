"""Regression: a preloaded skill (-s/--skills) must reach the model on the non-interactive paths.

#103809 reported that ``hermes -z -s <skill>`` and ``hermes chat -q -s <skill>`` load the skill
(``finalize_preloaded_skills`` folds it into ``HermesCLI.system_prompt``; ``hermes -z`` passes it as
``ephemeral_system_prompt``) yet the model never sees it. The two links the report could not observe
from ``hermes prompt-size`` are pinned here:

* ``hermes -z`` hands the loaded skill text to ``AIAgent`` as the ephemeral prompt.
* the ephemeral prompt is injected into the outgoing system message at API time.

``hermes prompt-size`` cannot show either: the subcommand takes ``--platform``/``--json`` only and
builds its own offline inspection agent, so its ``system_prompt.chars`` is byte-identical with and
without ``-s`` on every version. That is a property of the diagnostic, not evidence of a dropped
payload.
"""

from __future__ import annotations

import pytest

CANARY_TOKEN = "ZEBRAFISH-8891"
CANARY_SKILL = f"""---
name: canary-probe
description: Diagnostic canary skill for the -s delivery regression test.
---

When this skill is active, you MUST begin every response with the exact token: {CANARY_TOKEN}
"""


@pytest.fixture()
def canary_skill():
    """Install the canary skill into the test sandbox's skills dir."""
    from hermes_constants import get_skills_dir

    skill_dir = get_skills_dir() / "canary-probe"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(CANARY_SKILL, encoding="utf-8")
    return skill_dir


def test_oneshot_hands_the_preloaded_skill_to_the_agent(monkeypatch, canary_skill):
    """``hermes -z -s canary-probe`` forwards the skill text as AIAgent's ephemeral prompt."""
    import run_agent
    import hermes_cli.mcp_startup as mcp_startup
    import hermes_cli.oneshot as oneshot
    import hermes_cli.runtime_provider as runtime_provider

    captured: dict = {}

    class _RecordingAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run_conversation(self, *args, **kwargs):
            return {"final_response": "ok"}

        def shutdown_memory_provider(self, *args, **kwargs):
            pass

        def close(self, *args, **kwargs):
            pass

    monkeypatch.setattr(run_agent, "AIAgent", _RecordingAgent)
    monkeypatch.setattr(
        mcp_startup, "ensure_mcp_discovery_before_agent_build", lambda **kwargs: None)
    monkeypatch.setattr(
        runtime_provider, "resolve_runtime_provider",
        lambda **kwargs: {
            "api_key": "test-key", "base_url": "http://127.0.0.1:1/v1", "provider": "custom",
            "requested_provider": "custom", "api_mode": None, "credential_pool": None,
        })

    oneshot._run_agent(
        "What is 2+2?", model="test-model", toolsets=["clarify"], skills=["canary-probe"])

    ephemeral = captured.get("ephemeral_system_prompt") or ""
    assert CANARY_TOKEN in ephemeral, (
        "hermes -z dropped the -s/--skills payload before AIAgent was built — the model never "
        "sees the preloaded skill")


def test_ephemeral_prompt_is_injected_into_the_outgoing_system_message():
    """The wire copy handed to the provider carries ``ephemeral_system_prompt`` (API-time injection)."""
    from agent.turn_context import build_api_messages
    from run_agent import AIAgent

    agent = AIAgent(
        model="inspect-only", api_key="inspect-only", base_url="http://127.0.0.1:1/v1",
        quiet_mode=True, skip_context_files=True, skip_memory=True, platform="cli",
        enabled_toolsets=["clarify"],
        ephemeral_system_prompt=f"PRELOADED SKILL BODY {CANARY_TOKEN}",
    )

    api_messages, _effective = build_api_messages(
        agent, [{"role": "user", "content": "What is 2+2?"}],
        current_turn_user_idx=0, ext_prefetch_cache=None, plugin_user_context=None,
        moa_config=None, active_system_prompt=agent._cached_system_prompt or "",
    )

    assert api_messages and api_messages[0]["role"] == "system", (
        "no system message on the wire — ephemeral prompt had nowhere to go")
    assert CANARY_TOKEN in api_messages[0]["content"], (
        "the outgoing system message dropped ephemeral_system_prompt — a preloaded skill "
        "(and any -s payload) would be invisible to the model")


def _stub_oneshot_runtime(monkeypatch):
    """Stub provider/host bits so the oneshot path runs offline with a recording agent."""
    import run_agent
    import hermes_cli.mcp_startup as mcp_startup
    import hermes_cli.runtime_provider as runtime_provider

    class _RecordingAgent:
        def __init__(self, **kwargs):
            pass

        def run_conversation(self, *args, **kwargs):
            return {"final_response": "ok"}

        def shutdown_memory_provider(self, *args, **kwargs):
            pass

        def close(self, *args, **kwargs):
            pass

    monkeypatch.setattr(run_agent, "AIAgent", _RecordingAgent)
    monkeypatch.setattr(
        mcp_startup, "ensure_mcp_discovery_before_agent_build", lambda **kwargs: None)
    monkeypatch.setattr(
        runtime_provider, "resolve_runtime_provider",
        lambda **kwargs: {
            "api_key": "test-key", "base_url": "http://127.0.0.1:1/v1", "provider": "custom",
            "requested_provider": "custom", "api_mode": None, "credential_pool": None,
        })


def test_oneshot_reports_the_entry_it_dropped(capsys, monkeypatch, canary_skill):
    """``hermes -z -s ghost,canary-probe`` must name the entry it skipped, not drop it silently."""
    import logging

    import hermes_cli.oneshot as oneshot

    _stub_oneshot_runtime(monkeypatch)

    try:
        rc = oneshot.run_oneshot(
            "What is 2+2?", model="test-model", toolsets=["clarify"],
            skills=["ghost-skill-xyz", "canary-probe"])
    finally:
        # run_oneshot disables stdlib logging process-wide; do not leak that into other tests.
        logging.disable(logging.NOTSET)

    err = capsys.readouterr().err
    assert rc == 0
    assert "ghost-skill-xyz" in err, (
        "hermes -z silently discarded an unknown -s/--skills entry — the user is never told")
    assert "canary-probe" in err, "the notice must say which skills did load"


def test_chat_q_reports_the_entry_it_dropped(capsys):
    """``hermes chat -q -s ghost,canary-probe`` must name the skipped entry on stderr."""
    import cli as cli_mod

    finalize = cli_mod.HermesCLI.__dict__["finalize_preloaded_skills"]

    class _JoinedThread:
        def join(self, timeout=None):
            pass

    class _DummyCLI:
        system_prompt = "base prompt"
        preloaded_skills: list = []
        _preload_skills_finalized = False
        _preload_skills_thread = _JoinedThread()
        _preload_skills_error = None
        _preload_skills_result = ("SKILL BODY", ["canary-probe"], ["ghost-skill-xyz"])

    finalize(_DummyCLI)

    captured = capsys.readouterr()
    assert "ghost-skill-xyz" in captured.err, (
        "hermes chat -q silently discarded an unknown -s/--skills entry — the user is never told")
    assert "ghost-skill-xyz" not in captured.out, (
        "the dropped-skill notice must stay off stdout so `$(hermes chat -q ...)` stays clean")

