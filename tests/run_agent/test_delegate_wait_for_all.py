"""Behavioral coverage for delegation.wait_for_all model dispatch."""

from __future__ import annotations

import copy
import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from run_agent import AIAgent
from tools.delegate_tool import _build_dynamic_schema_overrides
from tools.registry import registry


def _write_delegation_config(home, monkeypatch, **delegation):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        yaml.safe_dump({"delegation": delegation}), encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(home / "missing-managed"))
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    from hermes_cli import config

    config._LOAD_CONFIG_CACHE.clear()
    config._RAW_CONFIG_CACHE.clear()


@pytest.mark.parametrize(
    ("wait_for_all", "depth", "hidden_background", "expected_background"),
    [
        (False, 0, False, True),
        (False, 0, True, True),
        (True, 0, False, False),
        (True, 0, True, False),
        (False, 1, True, False),
        (True, 1, False, False),
    ],
)
def test_model_dispatch_policy_is_shared_by_live_and_registry_paths(
    tmp_path, monkeypatch, wait_for_all, depth, hidden_background, expected_background
):
    """Both model entrypoints read the active profile and ignore the hidden flag."""
    _write_delegation_config(
        tmp_path / "profile", monkeypatch, wait_for_all=wait_for_all
    )
    parent = AIAgent.__new__(AIAgent)
    setattr(parent, "_delegate_depth", depth)
    captured = []

    def fake_delegate_task(**kwargs):
        captured.append(kwargs["background"])
        return "{}"

    args = {
        "tasks": [{"goal": "policy probe"}],
        "background": hidden_background,
    }
    with patch("tools.delegate_tool.delegate_task", side_effect=fake_delegate_task):
        AIAgent._dispatch_delegate_task(parent, args)
        registry.dispatch("delegate_task", args, parent_agent=parent)

    assert captured == [expected_background, expected_background]


def test_model_dispatch_policy_follows_active_profile_and_defaults_disabled(
    tmp_path, monkeypatch
):
    parent = AIAgent.__new__(AIAgent)
    setattr(parent, "_delegate_depth", 0)
    captured = []

    def fake_delegate_task(**kwargs):
        captured.append(kwargs["background"])
        return "{}"

    with patch("tools.delegate_tool.delegate_task", side_effect=fake_delegate_task):
        _write_delegation_config(
            tmp_path / "joined-profile", monkeypatch, wait_for_all=True
        )
        AIAgent._dispatch_delegate_task(parent, {"tasks": [{"goal": "joined"}]})

        _write_delegation_config(tmp_path / "default-profile", monkeypatch)
        AIAgent._dispatch_delegate_task(parent, {"tasks": [{"goal": "async"}]})

        _write_delegation_config(
            tmp_path / "joined-profile", monkeypatch, wait_for_all=True
        )
        AIAgent._dispatch_delegate_task(parent, {"tasks": [{"goal": "joined again"}]})

    assert captured == [False, True, False]


@pytest.mark.parametrize(
    ("delegation", "expected_policy"),
    [
        ({}, {"wait_for_all": False, "model_tasks": "asynchronous"}),
        ({"wait_for_all": True}, {"wait_for_all": True, "model_tasks": "joined"}),
    ],
)
def test_public_policy_query_matches_effective_profile_and_model_dispatch(
    tmp_path, monkeypatch, delegation, expected_policy
):
    """The public query and both model dispatch paths share one effective policy."""
    _write_delegation_config(tmp_path / "profile", monkeypatch, **delegation)
    from tools.delegate_tool_config import get_delegation_execution_policy

    parent = AIAgent.__new__(AIAgent)
    setattr(parent, "_delegate_depth", 0)
    captured = []

    def fake_delegate_task(**kwargs):
        captured.append(kwargs["background"])
        return "{}"

    args = {"tasks": [{"goal": "policy query probe"}], "background": True}
    with patch("tools.delegate_tool.delegate_task", side_effect=fake_delegate_task):
        AIAgent._dispatch_delegate_task(parent, args)
        registry.dispatch("delegate_task", args, parent_agent=parent)

    assert get_delegation_execution_policy() == expected_policy
    expected_background = expected_policy["model_tasks"] == "asynchronous"
    assert captured == [expected_background, expected_background]


def test_joined_schema_describes_blocking_mode_and_hides_group(
    tmp_path, monkeypatch
):
    _write_delegation_config(
        tmp_path / "profile",
        monkeypatch,
        wait_for_all=True,
        independent_completions=True,
    )

    function = registry.get_definitions({"delegate_task"})[0]["function"]
    direct = _build_dynamic_schema_overrides()

    for definition in (function, direct):
        description = definition["description"]
        task_properties = definition["parameters"]["properties"]["tasks"][
            "items"
        ]["properties"]
        assert "waits for all" in description
        assert "run in parallel" in description
        assert "input order" in description
        assert "user can still interrupt" in description
        assert "returns immediately" not in description
        assert "END YOUR TURN" not in description
        assert "group" not in task_properties


def test_direct_python_callers_keep_explicit_background_control(tmp_path, monkeypatch):
    """The model policy must not override direct in-process callers."""
    _write_delegation_config(tmp_path / "profile", monkeypatch, wait_for_all=True)
    import tools.delegate_tool as delegate_tool

    parent = SimpleNamespace(_delegate_depth=0, model="test/parent")
    credentials = {
        "model": "test/child",
        "provider": None,
        "base_url": None,
        "api_key": None,
        "api_mode": None,
    }
    observed = []

    def capture(_batch, background):
        observed.append(background)
        return json.dumps({"background": background})

    with (
        patch.object(delegate_tool, "_resolve_delegation_credentials", return_value=credentials),
        patch.object(delegate_tool, "_announce_batch"),
        patch.object(delegate_tool, "_build_children", return_value=([(0, {"goal": "probe"}, object())], None)),
        patch.object(delegate_tool, "_run_batch", side_effect=capture),
        patch(
            "tools.delegation_live_log.create_live_transcripts",
            return_value=("deleg_direct", [], []),
        ),
    ):
        async_result = delegate_tool.delegate_task(
            tasks=[{"goal": "probe"}], background=True, parent_agent=parent
        )
        joined_result = delegate_tool.delegate_task(
            tasks=[{"goal": "probe"}], background=False, parent_agent=parent
        )

    assert json.loads(async_result) == {"background": True}
    assert json.loads(joined_result) == {"background": False}
    assert observed == [True, False]


def _tool_call(name, arguments, call_id):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _response(*, content="", tool_calls=None, finish_reason="tool_calls"):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        model="test/model",
        usage=None,
    )


def _make_agent():
    definitions = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} test tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in ("delegate_task", "terminal")
    ]
    with (
        patch("model_tools.get_tool_definitions", return_value=definitions),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    setattr(agent, "client", MagicMock())
    setattr(agent, "_cached_system_prompt", "stable prompt")
    setattr(agent, "_use_prompt_caching", False)
    setattr(agent, "compression_enabled", False)
    setattr(agent, "save_trajectories", False)
    return agent


class _BlockingChild:
    def __init__(self, index, rendezvous, release, *, fail=False):
        self.index = index
        self.rendezvous = rendezvous
        self.release = release
        self.fail = fail
        self.started = threading.Event()
        self.finished = threading.Event()
        self.stop_received = threading.Event()
        self._interrupt_requested = False
        self._credential_pool = None
        self._delegate_role = "leaf"
        self._subagent_id = None
        self.tool_progress_callback = None
        self.session_id = ""
        self.model = "test/child"
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = "estimated"

    def run_conversation(self, **_kwargs):
        self.started.set()
        self.rendezvous.wait(timeout=5)
        if self.fail:
            self.finished.set()
            return {
                "final_response": "child A failed honestly",
                "completed": False,
                "failed": True,
                "error": "child A failed honestly",
                "api_calls": 1,
                "messages": [],
            }
        assert self.release.wait(timeout=5)
        self.finished.set()
        if self._interrupt_requested:
            return {
                "final_response": "interrupted",
                "completed": False,
                "interrupted": True,
                "api_calls": 0,
                "messages": [],
            }
        return {
            "final_response": "child B complete",
            "completed": True,
            "api_calls": 1,
            "messages": [],
        }

    def hard_interrupt(self, _message=None, **_kwargs):
        self._interrupt_requested = True
        self.stop_received.set()
        self.release.set()

    def interrupt(self, message=None):
        self.hard_interrupt(message)

    def get_activity_summary(self):
        return {"api_call_count": 0, "current_tool": None, "last_activity_ts": 0}

    def close(self):
        return None


def test_child_attached_after_parent_stop_inherits_hard_cancel():
    """A Stop that wins the registration race must reach a late-built child."""
    from tools.delegate_tool_child_run import _attach_child

    hard_stop = threading.Event()
    hard_stop.set()
    parent = SimpleNamespace(
        _active_children=[],
        _active_children_lock=threading.Lock(),
        _interrupt_requested=True,
        _hard_interrupt_requested=hard_stop,
        _interrupt_message="explicit parent stop",
        _tool_interrupt_reason="explicit stop requested",
    )
    child = _BlockingChild(0, threading.Barrier(1), threading.Event())

    _attach_child(parent, child)

    assert parent._active_children == [child]
    assert child.stop_received.is_set()
    assert child._interrupt_requested is True


def _fake_child_builder(children_out, children_built, rendezvous, release, *, fail_first):
    def build(task_list, _task_schemas, _creds, *, parent_agent, **_kwargs):
        from tools.delegate_tool_child_run import _attach_child

        children = [
            _BlockingChild(i, rendezvous, release, fail=fail_first and i == 0)
            for i in range(len(task_list))
        ]
        for child in children:
            _attach_child(parent_agent, child)
        children_out.extend(children)
        children_built.set()
        return [(i, task, children[i]) for i, task in enumerate(task_list)], None

    return build


def _start_scripted_parent(agent, provider_calls, probe_started, outcome):
    delegate_call = _tool_call(
        "delegate_task",
        {"tasks": [{"goal": "Run child task A"}, {"goal": "Run child task B"}]},
        "delegate-1",
    )
    trailing_probe = _tool_call(
        "terminal", {"command": "probe"}, "probe-1"
    )
    first = _response(tool_calls=[delegate_call, trailing_probe])
    second = _response(content="parent done", tool_calls=None, finish_reason="stop")
    provider_lock = threading.Lock()

    def provider(**kwargs):
        with provider_lock:
            provider_calls.append(copy.deepcopy(kwargs["messages"]))
            call_number = len(provider_calls)
        if call_number == 1:
            return first
        if call_number == 2:
            return second
        raise AssertionError("unexpected provider request")

    def terminal(_name, _args, _task_id, **_kwargs):
        probe_started.set()
        return json.dumps({"probe": "ran"})

    agent.client.chat.completions.create.side_effect = provider

    def run():
        try:
            outcome["result"] = agent.run_conversation("delegate both")
        except BaseException as exc:  # surfaced in the test thread
            outcome["error"] = exc

    return run, terminal


def test_wait_for_all_joins_parallel_children_before_parent_advances(
    tmp_path, monkeypatch
):
    _write_delegation_config(
        tmp_path / "profile",
        monkeypatch,
        wait_for_all=True,
        independent_completions=True,
    )
    agent = _make_agent()
    rendezvous = threading.Barrier(2, timeout=5)
    release = threading.Event()
    children = []
    children_built = threading.Event()
    provider_calls = []
    probe_started = threading.Event()
    outcome = {}
    run, terminal = _start_scripted_parent(
        agent, provider_calls, probe_started, outcome
    )

    with (
        patch(
            "tools.delegate_tool._build_children",
            side_effect=_fake_child_builder(
                children, children_built, rendezvous, release, fail_first=True
            ),
        ),
        patch("model_tools.handle_function_call", side_effect=terminal),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        parent_thread = threading.Thread(target=run)
        parent_thread.start()
        assert children_built.wait(timeout=5)
        assert len(children) == 2
        assert all(child.started.wait(timeout=5) for child in children)
        assert children[0].finished.wait(timeout=5)

        assert parent_thread.is_alive()
        assert not probe_started.is_set()
        assert len(provider_calls) == 1

        release.set()
        parent_thread.join(timeout=5)

    assert not parent_thread.is_alive()
    assert "error" not in outcome
    assert outcome["result"]["final_response"] == "parent done"
    assert probe_started.is_set()
    assert len(provider_calls) == 2

    second_request = provider_calls[1]
    assistant_index = next(
        i for i, message in enumerate(second_request)
        if message.get("role") == "assistant" and message.get("tool_calls")
    )
    tool_rows = second_request[assistant_index + 1 : assistant_index + 3]
    assert [row["role"] for row in tool_rows] == ["tool", "tool"]
    assert [row["tool_call_id"] for row in tool_rows] == ["delegate-1", "probe-1"]

    delegation_result = json.loads(tool_rows[0]["content"])
    assert [entry["task_index"] for entry in delegation_result["results"]] == [0, 1]
    assert [entry["status"] for entry in delegation_result["results"]] == [
        "failed",
        "completed",
    ]
    assert "mode" not in delegation_result
    assert "delegation_id" not in delegation_result


def test_wait_for_all_hard_interrupt_stops_children_and_skips_parent_advance(
    tmp_path, monkeypatch
):
    _write_delegation_config(
        tmp_path / "profile", monkeypatch, wait_for_all=True
    )
    agent = _make_agent()
    rendezvous = threading.Barrier(2, timeout=5)
    release = threading.Event()
    children = []
    children_built = threading.Event()
    provider_calls = []
    probe_started = threading.Event()
    outcome = {}
    run, terminal = _start_scripted_parent(
        agent, provider_calls, probe_started, outcome
    )

    with (
        patch(
            "tools.delegate_tool._build_children",
            side_effect=_fake_child_builder(
                children, children_built, rendezvous, release, fail_first=False
            ),
        ),
        patch("model_tools.handle_function_call", side_effect=terminal),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        parent_thread = threading.Thread(target=run)
        parent_thread.start()
        assert children_built.wait(timeout=5)
        assert all(child.started.wait(timeout=5) for child in children)

        agent.hard_interrupt("explicit parent stop")
        parent_thread.join(timeout=5)

    assert not parent_thread.is_alive()
    assert "error" not in outcome
    assert all(child.stop_received.is_set() for child in children)
    assert not probe_started.is_set()
    assert len(provider_calls) == 1
