"""Contract tests for the bounded Otto counterpoint plugin.

These tests exercise the plugin controller at the hook boundary.  Model calls are
injected as a deterministic workflow callable; the real Hermes child is covered
by the bounded-package smoke, not by a unit test that would spend provider quota.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import types
from pathlib import Path
from types import SimpleNamespace


_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "otto-counterpoint"
if "hermes_plugins" not in sys.modules:
    namespace = types.ModuleType("hermes_plugins")
    namespace.__path__ = []
    sys.modules["hermes_plugins"] = namespace
if "hermes_plugins.otto_counterpoint" not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        "hermes_plugins.otto_counterpoint",
        _PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(_PLUGIN_DIR)],
    )
    _module = importlib.util.module_from_spec(_spec)
    _module.__package__ = "hermes_plugins.otto_counterpoint"
    _module.__path__ = [str(_PLUGIN_DIR)]
    sys.modules["hermes_plugins.otto_counterpoint"] = _module
    _spec.loader.exec_module(_module)

from hermes_plugins.otto_counterpoint.controller import CounterpointController
from hermes_plugins.otto_counterpoint.counterpoint import (
    DispatchDecision,
    HermesAgentClient,
    HermesRuntime,
)
from scripts.otto_counterpoint_plugin_smoke import _workflow_smoke_passed


class _State:
    def __init__(self, root):
        self.data_dir = root


def _route() -> dict[str, object]:
    return {
        "vendor": "anthropic",
        "family": "anthropic",
        "provider": "nous",
        "model": "anthropic/claude-sonnet-5",
        "reasoning_effort": "high",
        "authenticated": True,
        "accessible": True,
        "smoke_tested": True,
        "relative_load": 1.0,
    }


def _pre(controller, *, turn_id="turn-1", user_message="hello", model="gpt-5.6-luna"):
    controller.on_pre_llm_call(
        session_id="session-1",
        task_id="task-1",
        turn_id=turn_id,
        user_message=user_message,
        conversation_history=[],
        is_first_turn=True,
        model=model,
        provider="openai-codex",
        platform="discord",
    )


def test_discover_uses_launcher_runtime_and_checkout_root(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    launcher = hermes_home / "tools" / "python-3.14.7" / "bin" / "python3"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(Path(sys.executable))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    runtime = HermesRuntime.discover()

    assert runtime.runtime_python == launcher
    assert runtime.source_root == Path(__file__).resolve().parents[2]
    assert runtime.worker_script == _PLUGIN_DIR / "hermes_counterpoint_worker.py"


def test_child_environment_is_allowlisted_for_selected_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("NOUS_API_KEY", "selected-provider-secret")
    monkeypatch.setenv("TYPESAFE_API_KEY", "unrelated-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "unrelated-secret")
    runtime = HermesRuntime(
        runtime_python=Path(sys.executable),
        source_root=Path(__file__).resolve().parents[2],
        worker_script=_PLUGIN_DIR / "hermes_counterpoint_worker.py",
        hermes_home=tmp_path,
    )

    environment = HermesAgentClient(runtime)._child_environment("nous")

    assert environment["NOUS_API_KEY"] == "selected-provider-secret"
    assert "TYPESAFE_API_KEY" not in environment
    assert "AWS_SECRET_ACCESS_KEY" not in environment
    assert environment["HERMES_COUNTERPOINT_CHILD"] == "1"


    blocked = SimpleNamespace(status="blocked", decision=SimpleNamespace(verdict="block"))
    accepted = SimpleNamespace(status="succeeded", decision=SimpleNamespace(verdict="accept_local"))

    assert _workflow_smoke_passed(blocked) is False
    assert _workflow_smoke_passed(accepted) is True


def test_low_risk_turn_is_admitted_directly_without_tool_block(tmp_path):
    controller = CounterpointController(
        config={"mode": "blocking"},
        state=_State(tmp_path),
    )

    _pre(controller)

    assert controller.on_pre_tool_call(
        tool_name="read_file",
        args={"path": "README.md"},
        session_id="session-1",
        task_id="task-1",
        turn_id="turn-1",
    ) is None
    assert controller.on_transform_llm_output(
        response_text="answer",
        session_id="session-1",
        model="gpt-5.6-luna",
        platform="discord",
        turn_id="turn-1",
    ) is None
    events = controller.ledger.list_run("cp-turn-1")
    assert events[-1]["event_type"] == "run.finished"
    assert events[-1]["terminal_state"] == "succeeded"
    payload = events[-1]["payload"]
    assert "hello" not in str(payload)
    assert "answer" not in str(payload)


def test_high_risk_without_independent_route_blocks_tools_and_hands_off(tmp_path):
    controller = CounterpointController(
        config={"mode": "blocking"},
        state=_State(tmp_path),
    )

    _pre(controller, user_message="deploy this to production")

    directive = controller.on_pre_tool_call(
        tool_name="terminal",
        args={"command": "deploy"},
        session_id="session-1",
        task_id="task-1",
        turn_id="turn-1",
    )
    assert directive["action"] == "block"

    transformed = controller.on_transform_llm_output(
        response_text="I would deploy it now.",
        session_id="session-1",
        model="gpt-5.6-luna",
        platform="discord",
        turn_id="turn-1",
    )
    assert transformed is not None
    assert "revisão humana" in transformed.lower()
    events = controller.ledger.list_run("cp-turn-1")
    assert events[-1]["terminal_state"] == "cancelled"
    assert events[-1]["payload"]["decision"]["verdict"] == "human_review"


def test_counterpoint_shadow_runs_once_but_preserves_original_response(tmp_path):
    calls = []

    def workflow_runner(*, pending, response_text):
        calls.append((pending.envelope.run_id, response_text))
        return SimpleNamespace(
            status="succeeded",
            decision=SimpleNamespace(verdict="accept_local"),
        )

    controller = CounterpointController(
        config={"mode": "shadow", "counterpoint_route": _route()},
        state=_State(tmp_path),
        workflow_runner=workflow_runner,
    )

    _pre(controller, user_message="research the architecture and compare the evidence")

    assert controller.on_transform_llm_output(
        response_text="bounded answer",
        session_id="session-1",
        model="gpt-5.6-luna",
        platform="discord",
        turn_id="turn-1",
    ) is None
    assert calls == [("cp-turn-1", "bounded answer")]
    assert controller.on_transform_llm_output(
        response_text="duplicate callback",
        session_id="session-1",
        model="gpt-5.6-luna",
        platform="discord",
        turn_id="turn-1",
    ) is None
    assert len(calls) == 1


def test_shadow_admission_review_blocks_tools_and_replaces_response(tmp_path):
    controller = CounterpointController(
        config={"mode": "shadow"},
        state=_State(tmp_path),
    )

    _pre(controller, user_message="deploy this to production")

    directive = controller.on_pre_tool_call(
        tool_name="terminal",
        args={"command": "must not run"},
        session_id="session-1",
        task_id="task-1",
        turn_id="turn-1",
    )
    assert directive["action"] == "block"
    transformed = controller.on_transform_llm_output(
        response_text="I would deploy it now.",
        session_id="session-1",
        model="gpt-5.6-luna",
        platform="discord",
        turn_id="turn-1",
    )
    assert transformed is not None
    assert "revisão humana" in transformed.lower()


def test_shadow_workflow_failure_replaces_response_with_review(tmp_path):
    def workflow_runner(*, pending, response_text):
        raise RuntimeError("provider detail must stay private")

    controller = CounterpointController(
        config={"mode": "shadow", "counterpoint_route": _route()},
        state=_State(tmp_path),
        workflow_runner=workflow_runner,
    )
    _pre(controller, user_message="research the architecture and compare the evidence")

    transformed = controller.on_transform_llm_output(
        response_text="unreviewed answer",
        session_id="session-1",
        model="gpt-5.6-luna",
        platform="discord",
        turn_id="turn-1",
    )
    assert transformed is not None
    assert "revisão humana" in transformed.lower()
    assert "provider detail" not in transformed


def test_unselected_canary_is_not_recorded_as_direct_acceptance(tmp_path):
    controller = CounterpointController(
        config={"mode": "canary", "canary_percent": 0, "counterpoint_route": _route()},
        state=_State(tmp_path),
    )

    _pre(controller, user_message="research the architecture and compare the evidence")

    pending = controller.pending("turn-1")
    assert pending.outcome.decision == DispatchDecision.COUNTERPOINT
    assert pending.canary_selected is False
    assert pending.canary_excluded is True
    assert controller.ledger.list_run("cp-turn-1") == []


def test_blocking_counterpoint_blocks_tools_until_local_acceptance(tmp_path):
    calls = []

    def workflow_runner(*, pending, response_text):
        calls.append(pending.envelope.run_id)
        return SimpleNamespace(
            status="succeeded",
            decision=SimpleNamespace(verdict="accept_local"),
        )

    controller = CounterpointController(
        config={"mode": "blocking", "counterpoint_route": _route()},
        state=_State(tmp_path),
        workflow_runner=workflow_runner,
    )
    _pre(controller, user_message="research the architecture and compare the evidence")

    directive = controller.on_pre_tool_call(
        tool_name="terminal",
        args={"command": "not executed in this test"},
        session_id="session-1",
        task_id="task-1",
        turn_id="turn-1",
    )
    assert directive["action"] == "block"
    assert controller.on_transform_llm_output(
        response_text="bounded answer",
        session_id="session-1",
        model="gpt-5.6-luna",
        platform="discord",
        turn_id="turn-1",
    ) is None
    assert calls == ["cp-turn-1"]


def test_admission_errors_fail_closed_to_human_review(tmp_path):
    controller = CounterpointController(
        config={"mode": "blocking"},
        state=_State(tmp_path),
    )

    def _explode(**_kwargs):
        raise ValueError("synthetic admission failure")

    controller._generator_route = _explode
    _pre(controller, user_message="ordinary request")

    directive = controller.on_pre_tool_call(
        tool_name="terminal",
        args={"command": "must not run"},
        session_id="session-1",
        task_id="task-1",
        turn_id="turn-1",
    )
    assert directive["action"] == "block"
    transformed = controller.on_transform_llm_output(
        response_text="unsafe continuation",
        session_id="session-1",
        model="gpt-5.6-luna",
        platform="discord",
        turn_id="turn-1",
    )
    assert "revisão humana" in transformed.lower()
    assert controller.ledger.list_run("cp-turn-1")[-1]["terminal_state"] == "cancelled"


def test_request_hash_is_stable_and_never_stores_raw_text(tmp_path):
    controller = CounterpointController(
        config={"mode": "shadow"},
        state=_State(tmp_path),
    )
    prompt = "analyze this exact request"
    _pre(controller, user_message=prompt)

    pending = controller.pending("turn-1")
    assert pending.envelope.request_sha256 == hashlib.sha256(prompt.encode()).hexdigest()
    assert prompt not in str(controller.ledger.list_events())


def test_plugin_loads_through_real_plugin_manager(tmp_path, monkeypatch):
    from hermes_cli import plugins as plugins_mod

    hermes_home = tmp_path / ".hermes"
    shutil.copytree(_PLUGIN_DIR, hermes_home / "plugins" / "otto-counterpoint")
    (hermes_home / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - otto-counterpoint\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    manager = plugins_mod.PluginManager()
    manager.discover_and_load()

    loaded = manager._plugins["otto-counterpoint"]
    assert loaded.enabled is True
    assert {"pre_llm_call", "pre_tool_call", "transform_llm_output"} <= set(loaded.hooks_registered)
    assert manager.invoke_hook(
        "pre_llm_call",
        session_id="s1",
        task_id="t1",
        turn_id="turn-real-loader",
        user_message="hello",
        conversation_history=[],
        is_first_turn=True,
        model="gpt-5.6-luna",
        provider="openai-codex",
        platform="discord",
    ) == []


def test_default_workflow_executes_counterpoint_and_persists_bounded_ledger(tmp_path):
    calls = []

    class _Client:
        def complete(self, route, prompt):
            calls.append((route.model, prompt))
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "status": "no_material_finding",
                        "findings": [],
                        "coverage": {"criteria": ["Answer the request accurately."]},
                        "evidence_refs": ["request-sha256"],
                    }
                ),
                provider=route.provider,
                model=route.model,
            )

    controller = CounterpointController(
        config={"mode": "shadow", "counterpoint_route": _route()},
        state=_State(tmp_path),
        client_factory=lambda: _Client(),
    )
    _pre(controller, user_message="research the architecture and compare the evidence")

    assert controller.on_transform_llm_output(
        response_text="bounded answer",
        session_id="session-1",
        model="gpt-5.6-luna",
        platform="discord",
        turn_id="turn-1",
    ) is None
    assert len(calls) == 1
    events = controller.ledger.list_run("cp-turn-1")
    assert [event["event_type"] for event in events] == [
        "run.created",
        "run.started",
        "run.finished",
    ]
    assert events[-1]["terminal_state"] == "succeeded"
    serialized = str(events)
    assert "bounded answer" not in serialized
    assert "research the architecture" not in serialized
