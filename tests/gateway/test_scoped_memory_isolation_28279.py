"""End-to-end gateway isolation test for scoped memory (issue #28279, PR #71224).

Simulates the exact scenario from the issue: one Hermes instance serving a
private Telegram DM and a public Telegram group. A secret saved from the DM
with scope='current' must be invisible to the group session through BOTH
disclosure surfaces:

  1. the system-prompt memory block (snapshot filtering), and
  2. every memory-tool result path (access-boundary filtering) — error
     inventories and replace/remove targeting.

Agents are constructed the same way ``gateway/run.py`` builds them (real
``AIAgent`` with ``platform`` + ``chat_id`` from the message source, memory
enabled via ``$HERMES_HOME/config.yaml``), so this exercises the full chain:
gateway identity → agent_init._build_memory_session_scopes → MemoryStore.
"""

import json

import pytest

from run_agent import AIAgent


class _FakeOpenAI:
    def __init__(self, **kw):
        self.api_key = kw.get("api_key", "test")
        self.base_url = kw.get("base_url", "http://test")

    def close(self):
        pass


DM_CHAT_ID = "555000111"
GROUP_CHAT_ID = "999888777"
SECRET = "prod DB password hunter2-XYZZY"


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with memory enabled, as a gateway deploy has."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "memory:\n  memory_enabled: true\n  user_profile_enabled: true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _gateway_agent(monkeypatch, chat_id: str) -> AIAgent:
    """Build an agent the way GatewayRunner does for a Telegram message."""
    monkeypatch.setattr("run_agent.get_tool_definitions", lambda **kw: [])
    monkeypatch.setattr("run_agent.check_toolset_requirements", lambda: {})
    monkeypatch.setattr("run_agent.OpenAI", _FakeOpenAI)
    return AIAgent(
        api_key="test-key",
        base_url="http://test",
        provider="openrouter",
        api_mode="chat_completions",
        max_iterations=1,
        quiet_mode=True,
        skip_context_files=True,
        platform="telegram",
        user_id="42",
        chat_id=chat_id,
        enabled_toolsets=["memory"],
    )


def _memory_tool_call(agent: AIAgent, **args) -> dict:
    """Invoke the memory tool exactly as the tool registry dispatches it."""
    from tools.memory_tool import memory_tool

    return json.loads(memory_tool(store=agent._memory_store, **args))


class TestGatewayScopedMemoryIsolation:
    def test_dm_secret_isolated_from_group_session(self, hermes_home, monkeypatch):
        # ── 1. DM session saves a secret scoped to the current chat ──
        dm = _gateway_agent(monkeypatch, chat_id=DM_CHAT_ID)
        assert dm._memory_store is not None
        assert f"telegram:{DM_CHAT_ID}" in dm._memory_store.session_scopes

        result = _memory_tool_call(
            dm, action="add", target="memory",
            content=SECRET, scope="current",
        )
        assert result["success"] is True
        # Server-side resolution pinned it to the DM chat, no model-invented ID.
        on_disk = (hermes_home / "memories" / "MEMORY.md").read_text(encoding="utf-8")
        assert f"[scope: telegram:{DM_CHAT_ID}] {SECRET}" in on_disk

        # An unscoped (global) entry for contrast.
        assert _memory_tool_call(
            dm, action="add", target="memory",
            content="team stand-up is at 10am",
        )["success"] is True

        # ── 2. Group session on the SAME instance loads the same store ──
        group = _gateway_agent(monkeypatch, chat_id=GROUP_CHAT_ID)
        assert f"telegram:{GROUP_CHAT_ID}" in group._memory_store.session_scopes

        # Surface 1: system-prompt snapshot must not contain the secret.
        snap = group._memory_store.format_for_system_prompt("memory") or ""
        assert SECRET not in snap
        assert "team stand-up is at 10am" in snap  # global entry still shared

        # Surface 2a: error inventories must not leak it.
        err = _memory_tool_call(group, action="remove", target="memory")
        assert err["success"] is False
        assert SECRET not in json.dumps(err, ensure_ascii=False)
        assert any("stand-up" in e for e in err["current_entries"])

        # Surface 2b: the group session cannot target the entry by substring.
        steal = _memory_tool_call(
            group, action="replace", target="memory",
            old_text="XYZZY", content="stolen",
        )
        assert steal["success"] is False
        assert SECRET not in json.dumps(steal, ensure_ascii=False)

        # And the secret is still intact on disk for the DM session.
        on_disk = (hermes_home / "memories" / "MEMORY.md").read_text(encoding="utf-8")
        assert SECRET in on_disk

    def test_dm_session_keeps_full_access_to_its_secret(self, hermes_home, monkeypatch):
        dm = _gateway_agent(monkeypatch, chat_id=DM_CHAT_ID)
        _memory_tool_call(dm, action="add", target="memory",
                          content=SECRET, scope="current")

        # A NEW session of the same DM chat (fresh agent, same chat_id) sees
        # the secret in its snapshot (marker stripped)…
        dm2 = _gateway_agent(monkeypatch, chat_id=DM_CHAT_ID)
        snap = dm2._memory_store.format_for_system_prompt("memory") or ""
        assert SECRET in snap
        assert "[scope:" not in snap

        # …and can consolidate (replace) and delete it.
        assert _memory_tool_call(
            dm2, action="replace", target="memory",
            old_text="hunter2-XYZZY", content=f"[scope: telegram:{DM_CHAT_ID}] rotated",
        )["success"] is True
        assert _memory_tool_call(
            dm2, action="remove", target="memory", old_text="rotated",
        )["success"] is True
        on_disk = (hermes_home / "memories" / "MEMORY.md").read_text(encoding="utf-8")
        assert SECRET not in on_disk

    def test_cli_session_isolated_from_chat_scoped_entries(self, hermes_home, monkeypatch):
        """A local CLI session (no gateway platform) must not see chat-scoped
        secrets either — its identity is ['cli', ...], not a telegram scope."""
        dm = _gateway_agent(monkeypatch, chat_id=DM_CHAT_ID)
        _memory_tool_call(dm, action="add", target="memory",
                          content=SECRET, scope="current")

        monkeypatch.setattr("run_agent.get_tool_definitions", lambda **kw: [])
        monkeypatch.setattr("run_agent.check_toolset_requirements", lambda: {})
        monkeypatch.setattr("run_agent.OpenAI", _FakeOpenAI)
        cli = AIAgent(
            api_key="test-key", base_url="http://test",
            provider="openrouter", api_mode="chat_completions",
            max_iterations=1, quiet_mode=True, skip_context_files=True,
            enabled_toolsets=["memory"],
        )
        snap = cli._memory_store.format_for_system_prompt("memory") or ""
        assert SECRET not in snap
        err = _memory_tool_call(cli, action="remove", target="memory")
        assert SECRET not in json.dumps(err, ensure_ascii=False)
