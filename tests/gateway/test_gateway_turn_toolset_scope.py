"""Gateway turns cannot run tools outside the session's toolset grant (#121089).

The field report is a Telegram session saved as ``[file, skills, vision]`` with
``terminal`` disabled that still reached host shell execution. This module drives
that boundary end to end instead of calling the dispatcher directly: an isolated
profile home writes the reporter's config, the turn grant is resolved by the real
``GatewayRunner._resolve_turn_toolsets``, a real ``AIAgent`` is built from that
grant, and a model-emitted ``terminal`` call is fed through the real
``validate_tool_calls`` + ``AIAgent._execute_tool_calls`` executor — with an
in-scope ``read_file`` call as the control. ``agent/tool_executor.py`` forwards
``agent.enabled_toolsets`` / ``agent.disabled_toolsets`` into
``model_tools.handle_function_call``, so this is the same chain a gateway turn
runs.

Covered gate: the gateway agent path, fresh and cached. Not covered (pinned as a
characterization in ``test_absent_platform_key_grants_every_toolset``): a session
whose profile config never reaches the gateway, which is a separate decision.
"""

import json
from types import SimpleNamespace

import pytest

import gateway.run as gateway_run
from gateway.run import GatewayRunner, _gateway_config_home, _load_gateway_config

REPORTER_CONFIG = """platform_toolsets:
  telegram: [file, skills, vision]
agent:
  disabled_toolsets: [terminal]
"""


class _Source:
    platform = "telegram"
    chat_id = "telegram:4242"
    user_id = "4242"
    user_id_alt = None
    user_name = "reporter"
    chat_name = "reporter"
    chat_type = "private"
    thread_id = None


@pytest.fixture
def isolated_profile(tmp_path, monkeypatch):
    """A profile home the gateway can actually read, isolated per test.

    ``gateway.run`` caches ``_hermes_home`` at import time; production resolves the
    effective config through it, so the gateway sees a different profile whenever
    it is wrong — assert on that identity before asserting on tool grants.
    """
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path, raising=False)
    return tmp_path


def _runner():
    runner = object.__new__(GatewayRunner)
    runner._delivery_adapter_for = lambda source: None
    return runner


def _write_profile(home, text=REPORTER_CONFIG):
    (home / "config.yaml").write_text(text, encoding="utf-8")


def _turn_grant(home, text=REPORTER_CONFIG):
    _write_profile(home, text)
    return GatewayRunner._resolve_turn_toolsets(
        _runner(), _load_gateway_config(), _Source(), "telegram",
    )


def _tool_call(name, arguments, call_id):
    return SimpleNamespace(
        id=call_id, type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _scoped_agent(enabled, disabled):
    from run_agent import AIAgent

    return AIAgent(
        api_key="test-key", base_url="http://127.0.0.1:9/v1", model="test-model",
        provider="custom", quiet_mode=True, skip_context_files=True, skip_memory=True,
        enabled_toolsets=list(enabled), disabled_toolsets=list(disabled) if disabled else None,
        max_iterations=1,
    )


def test_turn_grant_binds_to_the_profile_home_and_excludes_terminal(isolated_profile):
    _write_profile(isolated_profile)

    # Identity read-back: the gateway's effective config home is the profile just
    # written, so a wrong-home binding cannot masquerade as a grant fix.
    assert _gateway_config_home() == isolated_profile
    loaded = _load_gateway_config()
    assert loaded["platform_toolsets"]["telegram"] == ["file", "skills", "vision"]
    assert loaded["agent"]["disabled_toolsets"] == ["terminal"]

    enabled, disabled = GatewayRunner._resolve_turn_toolsets(
        _runner(), loaded, _Source(), "telegram",
    )
    assert enabled == ["file", "skills", "vision"]
    assert disabled == ["terminal"]


def test_scoped_turn_refuses_model_emitted_terminal_call(isolated_profile, monkeypatch):
    enabled, disabled = _turn_grant(isolated_profile)
    agent = _scoped_agent(enabled, disabled)
    assert "terminal" not in agent.valid_tool_names
    assert "read_file" in agent.valid_tool_names  # control is admitted

    dispatched, reached_dispatcher = [], []
    import model_tools
    import tools.registry as registry_module

    real_handle = model_tools.handle_function_call

    def _recording_handle(name, args, *a, **kw):
        reached_dispatcher.append(name)
        return real_handle(name, args, *a, **kw)

    monkeypatch.setattr(model_tools, "handle_function_call", _recording_handle)
    monkeypatch.setattr(
        registry_module.registry, "dispatch",
        lambda name, args, **kw: (dispatched.append(name), '{"ok": true}')[1],
    )

    from agent.conversation_loop import _invalid_tool_name_error_content
    from agent.message_metadata import append_message
    from agent.message_sanitization import coalesce_tool_call_id
    from agent.turn_tool_validation import validate_tool_calls

    assistant_message = SimpleNamespace(content=None, model="test-model", tool_calls=[
        _tool_call("terminal", {"command": "id"}, "call_terminal"),
        _tool_call("read_file", {"path": "notes.txt"}, "call_control"),
    ])
    messages = [{"role": "user", "content": "run id"}]

    verdict = validate_tool_calls(
        agent, assistant_message, "tool_calls", messages=messages,
        conversation_history=[], api_call_count=1, effective_task_id="task",
    )
    # A mixed batch proceeds with the invalid call error-resulted and dropped —
    # same handling agent/turn_tool_round.py::run_tool_round applies before execution.
    assert verdict.action == "ok" and verdict.mixed_invalid_batch is True
    invalid = [tc for tc in assistant_message.tool_calls if tc.function.name not in agent.valid_tool_names]
    assert [tc.function.name for tc in invalid] == ["terminal"]
    for tc in invalid:
        append_message(messages, {
            "role": "tool", "name": tc.function.name, "tool_call_id": coalesce_tool_call_id(tc),
            "content": _invalid_tool_name_error_content(tc.function.name, agent.valid_tool_names),
        })
    assistant_message.tool_calls = [
        tc for tc in assistant_message.tool_calls if tc.function.name in agent.valid_tool_names
    ]

    agent._execute_tool_calls(assistant_message, messages, "task", 1)

    assert dispatched == ["read_file"], f"executor dispatched {dispatched}"
    assert "terminal" not in reached_dispatcher, "the refusal must happen before dispatch"
    refusal = next(m["content"] for m in messages if m.get("name") == "terminal")
    # On a freshly built agent the admission layer refuses first and names the tool —
    # the dispatcher guard's message below is what a session that admitted the name
    # (stale cached agent, restored transcript) sees instead.
    assert "terminal" in refusal and "does not exist" in refusal

    # The dispatcher guard is the second layer: the same grant, the call past
    # admission, and still no registry dispatch.
    dispatched.clear()
    direct = json.loads(real_handle(
        "terminal", {"command": "id"}, "task",
        enabled_tools=list(agent.valid_tool_names),
        skip_pre_tool_call_hook=True, skip_tool_request_middleware=True,
        skip_tool_execution_middleware=True,
        enabled_toolsets=agent.enabled_toolsets, disabled_toolsets=agent.disabled_toolsets,
    ))
    assert dispatched == []
    assert "not available in this session" in direct["error"]


def test_cached_agent_grant_is_invalidated_when_disabled_toolsets_change():
    """A cached agent is only rebuilt when the cache key changes.

    ``AIAgent`` freezes ``valid_tool_names`` (and its own toolset fields) at
    construction, so a grant that narrowed via ``disabled_toolsets`` must not reuse
    the agent cached under the wider grant — otherwise a tool the config disabled
    stays admitted for the life of the session.
    """
    runtime = {"api_key": "test-key", "base_url": "http://127.0.0.1:9/v1", "provider": "custom"}
    enabled = ["file", "skills", "vision", "terminal"]

    wide = GatewayRunner._agent_config_signature("test-model", runtime, enabled, "")
    narrow = GatewayRunner._agent_config_signature(
        "test-model", runtime, enabled, "", disabled_toolsets=["terminal"],
    )
    assert wide != narrow

    stable = GatewayRunner._agent_config_signature(
        "test-model", runtime, enabled, "", disabled_toolsets=["terminal"],
    )
    assert stable == narrow  # unchanged grant ⇒ reuse, no needless rebuild


def test_absent_platform_key_grants_every_toolset(isolated_profile):
    """Characterization, not endorsement.

    With nothing to read, ``platform_toolsets`` resolution returns every toolset —
    ``terminal`` included — so a session whose profile config never reaches the
    gateway is granted shell execution no matter what this fix does downstream.
    ``disabled_toolsets`` still subtracts (see the second half), which is why the
    reported config would be refused at every layer once it is actually read.
    """
    _write_profile(isolated_profile, "model:\n  default: test-model\n")
    enabled, disabled = _turn_grant(isolated_profile, "model:\n  default: test-model\n")
    assert disabled is None
    assert "terminal" in enabled
    import model_tools

    assert "terminal" in model_tools._select_tool_names(enabled, disabled, quiet_mode=True)

    _write_profile(isolated_profile, "agent:\n  disabled_toolsets: [terminal]\n")
    enabled_disabled, disabled = _turn_grant(isolated_profile, "agent:\n  disabled_toolsets: [terminal]\n")
    # _get_platform_tools already subtracts the disabled set from the platform list,
    # so the disabled half holds even when no platform_toolsets entry was readable.
    assert disabled == ["terminal"]
    assert "terminal" not in model_tools._select_tool_names(enabled_disabled, disabled, quiet_mode=True)
