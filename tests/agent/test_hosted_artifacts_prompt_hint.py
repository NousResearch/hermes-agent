"""The hosted artifacts prompt hint: present only when the session's tools include them, per surface."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.prompt_builder import (
    HOSTED_ARTIFACTS_GUIDANCE,
    HOSTED_ARTIFACTS_LINK_HINT,
    HOSTED_ARTIFACTS_SURFACE_HINTS,
)
from agent.system_prompt import _tool_guidance_block, build_system_prompt_parts
from tools.hosted_artifacts import contract_tool_names

ARTIFACT_TOOLS = set(contract_tool_names())


def _agent(names, platform="cli"):
    return SimpleNamespace(
        load_soul_identity=False, skip_context_files=True, valid_tool_names=set(names),
        _task_completion_guidance=False, _parallel_tool_call_guidance=False, _tool_use_enforcement=False,
        _execution_guidance=False, _environment_probe=False, _bot_mode_protocol=False,
        _kanban_worker_guidance="", _memory_store=None, _memory_manager=None, model="", provider="",
        platform=platform, pass_session_id=False, session_id="", _emit_status=lambda *_a, **_k: None,
    )


def _full_prompt(agent) -> str:
    with patch("agent.prompt_builder.load_soul_md", return_value=""), \
         patch("agent.prompt_builder.build_environment_hints", return_value=""):
        return "\n\n".join(build_system_prompt_parts(agent).values())


@pytest.mark.parametrize("platform, surface", [
    ("desktop", HOSTED_ARTIFACTS_SURFACE_HINTS["desktop"]),
    ("cli", HOSTED_ARTIFACTS_SURFACE_HINTS["cli"]),
    ("tui", HOSTED_ARTIFACTS_SURFACE_HINTS["tui"]),
    ("telegram", HOSTED_ARTIFACTS_LINK_HINT),
    ("slack", HOSTED_ARTIFACTS_LINK_HINT),
])
def test_hint_names_the_surface_when_attached(platform, surface):
    block = _tool_guidance_block(_agent(ARTIFACT_TOOLS | {"terminal"}, platform))
    assert block is not None and surface in block and HOSTED_ARTIFACTS_GUIDANCE in block
    others = (set(HOSTED_ARTIFACTS_SURFACE_HINTS.values()) | {HOSTED_ARTIFACTS_LINK_HINT}) - {surface}
    assert not any(other in block for other in others)


def test_hint_teaches_the_update_and_link_rules():
    text = HOSTED_ARTIFACTS_GUIDANCE
    for needle in ("artifact_update", "base_version", "stale_base", "viewer_url", "content_url"):
        assert needle in text
    assert "never its content_url" in text


@pytest.mark.parametrize("names", [
    set(), {"terminal", "memory"}, {"mcp__other__artifact_create", "mcp__hermes_artifact__artifact_show"},
])
def test_hint_absent_without_the_artifact_tools(names):
    assert HOSTED_ARTIFACTS_GUIDANCE not in (_tool_guidance_block(_agent(names)) or "")
    assert HOSTED_ARTIFACTS_GUIDANCE not in _full_prompt(_agent(names))


def test_hint_rides_the_built_prompt_and_is_byte_stable():
    agent = _agent(ARTIFACT_TOOLS, "telegram")
    first = _full_prompt(agent)
    assert HOSTED_ARTIFACTS_LINK_HINT in first and HOSTED_ARTIFACTS_GUIDANCE in first
    assert _full_prompt(agent) == first
