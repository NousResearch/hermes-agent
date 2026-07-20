from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import hermes_cli.plugins as plugin_runtime
import pytest

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

from agent.turn_finalizer import finalize_turn


REPO_ROOT = Path(__file__).resolve().parents[3]
PLUGIN_DIR = REPO_ROOT / "plugins" / "truth-ledger"


def _load_truth_plugin_init():
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.truth_ledger",
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec is not None
    assert spec.loader is not None
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hermes_plugins.truth_ledger"] = mod
    spec.loader.exec_module(mod)
    return mod


def _spool_payloads(hermes_home: Path) -> list[dict]:
    pending_dir = hermes_home / "truth-ledger" / "spool" / "pending"
    payloads: list[dict] = []
    for rec_path in sorted(pending_dir.glob("*.json")):
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        payload_path = Path(rec["payload_path"])
        payloads.append(json.loads(payload_path.read_text(encoding="utf-8")))
    return payloads


class _StubAgent:
    def __init__(self, **overrides):
        self.max_iterations = 3
        self.iteration_budget = SimpleNamespace(remaining=10, used=1, max_total=3)
        self.quiet_mode = True
        self.model = "stub-model"
        self.provider = "stub-provider"
        self.base_url = ""
        self.session_id = "sess-1"
        self.platform = "cli"

        self.context_compressor = SimpleNamespace(last_prompt_tokens=0)
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_estimated_cost_usd = 0
        self.session_cost_status = "ok"
        self.session_cost_source = "stub"

        self._tool_guardrail_halt_decision = None
        self._interrupt_message = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self.valid_tool_names = []

        self._delegate_depth = 0
        self._parent_session_id = None
        self._user_id = None
        self._chat_id = None
        self._chat_type = None
        self._thread_id = None
        self._gateway_session_key = None

        for key, value in overrides.items():
            setattr(self, key, value)

    def _handle_max_iterations(self, _messages, _api_call_count):
        return "iteration summary"

    def _emit_status(self, *_args, **_kwargs):
        pass

    def _safe_print(self, *_args, **_kwargs):
        pass

    def _save_trajectory(self, *_args, **_kwargs):
        pass

    def _cleanup_task_resources(self, *_args, **_kwargs):
        pass

    def _drop_trailing_empty_response_scaffolding(self, _messages):
        pass

    def _persist_session(self, _messages, _conversation_history):
        pass

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **_kwargs):
        pass


def _install_truth_ledger_plugin_manager(monkeypatch, plugin, *, profile_name: str) -> PluginManager:
    manager = PluginManager()
    monkeypatch.setattr(plugin_runtime, "_plugin_manager", manager)
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: profile_name)
    plugin.register(PluginContext(PluginManifest(name="truth-ledger"), manager))
    return manager


def _run_finalize(agent: _StubAgent):
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "final"}]
    return finalize_turn(
        agent,
        final_response="final",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="hi",
        original_user_message="hi",
        _should_review_memory=False,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )


def test_runtime_compatibility_requires_plugin_manager_dispatch(
    tmp_path,
    monkeypatch,
):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    plugin = _load_truth_plugin_init()
    plugin._SEEN_ENVELOPES.clear()

    manager = _install_truth_ledger_plugin_manager(monkeypatch, plugin, profile_name="default")
    manager_calls: list[dict] = []
    original_invoke_hook = manager.invoke_hook

    def _probe_invoke_hook(hook_name: str, **kwargs):
        if hook_name == "post_llm_call":
            manager_calls.append(kwargs)
        return original_invoke_hook(hook_name, **kwargs)

    monkeypatch.setattr(manager, "invoke_hook", _probe_invoke_hook)
    _run_finalize(_StubAgent(session_id="sess-dispatch", platform="cli"))

    assert len(manager_calls) == 1
    assert len(_spool_payloads(hermes_home)) == 1


@pytest.mark.parametrize(
    "scenario,agent_overrides",
    [
        (
            {
                "name": "interactive_cli",
                "platform": "cli",
                "conversation_id": None,
                "chat_id": None,
                "thread_id": None,
                "chat_type": None,
                "speaker_id": None,
            },
            {},
        ),
        (
            {
                "name": "gateway_shaped",
                "platform": "telegram",
                "conversation_id": "agent:main:telegram:group:chat-123:thread-77",
                "chat_id": "chat-123",
                "thread_id": "thread-77",
                "chat_type": "group",
                "speaker_id": "user-42",
            },
            {
                "_user_id": "user-42",
                "_chat_id": "chat-123",
                "_thread_id": "thread-77",
                "_chat_type": "group",
                "_gateway_session_key": "agent:main:telegram:group:chat-123:thread-77",
            },
        ),
    ],
)
def test_runtime_compatibility_matrix_capture_via_turn_finalizer_entry_path(
    tmp_path,
    monkeypatch,
    scenario,
    agent_overrides,
):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    plugin = _load_truth_plugin_init()
    _install_truth_ledger_plugin_manager(monkeypatch, plugin, profile_name="default")
    plugin._SEEN_ENVELOPES.clear()

    _run_finalize(
        _StubAgent(
            session_id=f"sess-{scenario['name']}",
            platform=scenario["platform"],
            **agent_overrides,
        )
    )

    payloads = _spool_payloads(hermes_home)
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["profile"] == "default"
    assert payload["origin"]["platform"] == scenario["platform"]
    assert payload["origin"]["conversation_id"] == scenario["conversation_id"]
    assert payload["origin"]["chat_id"] == scenario["chat_id"]
    assert payload["origin"]["thread_id"] == scenario["thread_id"]
    assert payload["origin"]["chat_type"] == scenario["chat_type"]
    assert payload["origin"]["speaker_id"] == scenario["speaker_id"]


@pytest.mark.parametrize(
    "agent_overrides,kanban_task_id",
    [
        (
            {"session_id": "sess-kanban", "platform": "cli"},
            "t_worker",
        ),
        (
            {
                "session_id": "sess-subagent",
                "platform": "subagent",
                "_delegate_depth": 1,
                "_parent_session_id": "parent-1",
            },
            None,
        ),
    ],
)
def test_runtime_compatibility_exclusions_via_turn_finalizer_entry_path(
    tmp_path,
    monkeypatch,
    agent_overrides,
    kanban_task_id,
):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    if kanban_task_id:
        monkeypatch.setenv("HERMES_KANBAN_TASK", kanban_task_id)
    else:
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    plugin = _load_truth_plugin_init()
    _install_truth_ledger_plugin_manager(monkeypatch, plugin, profile_name="automation-operator")
    plugin._SEEN_ENVELOPES.clear()

    _run_finalize(_StubAgent(**agent_overrides))
    assert _spool_payloads(hermes_home) == []


def test_runtime_compatibility_profile_isolation_between_homes_via_turn_finalizer(
    tmp_path,
    monkeypatch,
):
    plugin = _load_truth_plugin_init()
    plugin._SEEN_ENVELOPES.clear()

    home_a = tmp_path / "home-a"
    home_b = tmp_path / "home-b"
    home_a.mkdir(parents=True, exist_ok=True)
    home_b.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("HERMES_HOME", str(home_a))
    _install_truth_ledger_plugin_manager(monkeypatch, plugin, profile_name="default")
    _run_finalize(_StubAgent(session_id="sess-a", platform="cli"))

    monkeypatch.setenv("HERMES_HOME", str(home_b))
    _install_truth_ledger_plugin_manager(monkeypatch, plugin, profile_name="profile-b")
    _run_finalize(_StubAgent(session_id="sess-b", platform="cli"))

    payloads_a = _spool_payloads(home_a)
    payloads_b = _spool_payloads(home_b)

    assert len(payloads_a) == 1
    assert len(payloads_b) == 1
    assert payloads_a[0]["session_id"] == "sess-a"
    assert payloads_b[0]["session_id"] == "sess-b"
    assert payloads_a[0]["profile"] == "default"
    assert payloads_b[0]["profile"] == "profile-b"


def test_runtime_compatibility_uninstalled_neutral_status(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "truth_ledger_commands_under_test",
        PLUGIN_DIR / "commands.py",
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    missing_root = tmp_path / "missing" / "truth-ledger"
    report = mod.status_report(missing_root)
    assert report["ok"] is True
    assert report["enabled"] is False