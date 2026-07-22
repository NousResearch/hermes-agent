"""Behavioral tests for the contrib advisor plugin."""

from __future__ import annotations

import builtins
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
IN_TREE_PLUGIN_DIR = ROOT / "plugins" / "advisor"
PLUGIN_DIR = (
    IN_TREE_PLUGIN_DIR
    if IN_TREE_PLUGIN_DIR.exists()
    else ROOT / "contrib" / "hermes-advisor-plugin"
)


def _load_plugin(package_name: str = "_advisor_test_plugin"):
    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            sys.modules.pop(module_name)
    spec = importlib.util.spec_from_file_location(
        package_name,
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def advisor(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return _load_plugin()


class _Result:
    def __init__(self, text: str):
        self.text = text
        self.provider = "test-provider"
        self.model = "test-model"
        self.usage = None


class _Llm:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return _Result(next(self.responses))


class _Context:
    def __init__(self, responses=()):
        self.llm = _Llm(responses)
        self.injected = []
        self.selection_callback = None

    def inject_message(self, content, role="user"):
        self.injected.append((content, role))
        return True

    def request_model_selection(self, callback, **_kwargs):
        self.selection_callback = callback
        return True


def _submit(runtime, turn_id: str):
    runtime.on_post_llm_call(
        turn_id=turn_id,
        user_message="Fix it",
        assistant_response="Done",
        conversation_history=[{"role": "user", "content": "Fix it"}],
        model="primary-model",
    )
    assert runtime.wait_for_idle()


def test_concern_is_held_then_delivered_after_restart(advisor, tmp_path):
    context = _Context(["[CONCERN] Missing regression test"])
    runtime = advisor.AdvisorRuntime(context)

    _submit(runtime, "turn-1")

    assert context.injected == []
    state_paths = list((tmp_path / "advisor" / "sessions").glob("*.json"))
    assert len(state_paths) == 1
    assert json.loads(state_paths[0].read_text())["held_notes"] == [
        {"note": "Missing regression test", "severity": "concern"}
    ]

    restarted_context = _Context(["[CONCERN]  missing   REGRESSION test"])
    restarted = advisor.AdvisorRuntime(restarted_context)
    _submit(restarted, "turn-2")

    assert len(restarted_context.injected) == 1
    assert "[CONCERN] missing   REGRESSION test" in restarted_context.injected[0][0]


def test_silence_drops_persisted_held_advice(advisor):
    context = _Context(["[BLOCKER] Unsafe write", "Nothing to flag."])
    runtime = advisor.AdvisorRuntime(context)

    _submit(runtime, "turn-1")
    assert runtime._session_state("default").has_held()
    _submit(runtime, "turn-2")

    assert not runtime._session_state("default").has_held()
    assert context.injected == []


def test_held_advice_is_scoped_to_session(advisor):
    context = _Context(
        [
            "[CONCERN] Session-specific issue",
            "[CONCERN] Session-specific issue",
            "[CONCERN] Session-specific issue",
        ]
    )
    runtime = advisor.AdvisorRuntime(context)

    for session_id, turn_id in (
        ("session-a", "turn-1"),
        ("session-b", "turn-1"),
        ("session-a", "turn-2"),
    ):
        runtime.on_post_llm_call(
            session_id=session_id,
            turn_id=turn_id,
            conversation_history=[{"role": "user", "content": "check"}],
        )
        assert runtime.wait_for_idle()

    assert len(context.injected) == 1


def test_post_llm_hook_does_not_wait_for_completion(advisor):
    started = threading.Event()
    release = threading.Event()

    class BlockingLlm(_Llm):
        def complete(self, **kwargs):
            self.calls.append(kwargs)
            started.set()
            assert release.wait(2)
            return _Result("Nothing to flag.")

    context = _Context()
    context.llm = BlockingLlm(())
    runtime = advisor.AdvisorRuntime(context)

    before = time.monotonic()
    runtime.on_post_llm_call(
        turn_id="turn-1",
        conversation_history=[{"role": "user", "content": "hello"}],
    )
    elapsed = time.monotonic() - before

    assert elapsed < 0.1
    assert started.wait(1)
    release.set()
    assert runtime.wait_for_idle()


def test_finalize_wait_is_bounded(advisor, monkeypatch):
    runtime = advisor.AdvisorRuntime(_Context())
    runtime._idle.clear()
    runtime_module = sys.modules[advisor.AdvisorRuntime.__module__]
    monkeypatch.setattr(runtime_module, "SHUTDOWN_GRACE_SECONDS", 0.01)

    before = time.monotonic()
    runtime.on_session_finalize()

    assert time.monotonic() - before < 0.1


def test_pending_queue_keeps_only_newest_turn(advisor):
    started = threading.Event()
    release = threading.Event()

    class RecordingLlm(_Llm):
        def complete(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                started.set()
                assert release.wait(2)
            return _Result("Nothing to flag.")

    context = _Context()
    context.llm = RecordingLlm(())
    runtime = advisor.AdvisorRuntime(context)

    runtime.on_post_llm_call(
        turn_id="turn-1",
        user_message="one",
        conversation_history=[{"role": "user", "content": "one"}],
    )
    assert started.wait(1)
    for turn_id, text in (("turn-2", "two"), ("turn-3", "three")):
        runtime.on_post_llm_call(
            turn_id=turn_id,
            user_message=text,
            conversation_history=[{"role": "user", "content": text}],
        )
    release.set()
    assert runtime.wait_for_idle()

    prompts = [call["messages"][-1]["content"] for call in context.llm.calls]
    assert len(prompts) == 2
    assert "one" in prompts[0]
    assert "three" in prompts[1]
    assert "two" not in prompts[1]


def test_transcript_includes_native_tool_calls_and_results(advisor):
    transcript = advisor.AdvisorRuntime._format_history(
        user_message="Inspect",
        response="Finished",
        history=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "terminal",
                            "arguments": '{"command":"git status"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "name": "terminal",
                "tool_call_id": "call-1",
                "content": "working tree clean",
            },
        ],
    )

    assert "tool `terminal`" in transcript
    assert "git status" in transcript
    assert "`terminal` result: working tree clean" in transcript


def test_native_selector_updates_only_advisor_state(advisor):
    context = _Context()
    runtime = advisor.AdvisorRuntime(context)

    assert runtime.handle_command("model") is None
    message = context.selection_callback(
        SimpleNamespace(
            success=True,
            new_model="review-model",
            target_provider="review-provider",
            provider_label="Review Provider",
            error_message="",
        )
    )

    assert runtime.state.model == "review-model"
    assert runtime.state.provider == "review-provider"
    assert "review-model" in message


def test_import_does_not_require_posix_modules(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"pty", "termios"}:
            raise ImportError(f"{name} is unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    module = _load_plugin("_advisor_windows_import_test")

    assert module.AdvisorRuntime is not None


def test_plugin_context_model_selection_uses_native_modal(monkeypatch):
    from hermes_cli.plugins import PluginContext

    opened = {}

    class FakeCli:
        provider = "primary-provider"
        model = "primary-model"
        base_url = ""

        def _open_model_picker(self, *args, **kwargs):
            opened["args"] = args
            opened["kwargs"] = kwargs

    context_data = SimpleNamespace(
        user_providers={"review-provider": {}},
        custom_providers=[],
        with_overrides=lambda **_kwargs: context_data,
    )
    monkeypatch.setattr(
        "hermes_cli.inventory.load_picker_context", lambda: context_data
    )
    monkeypatch.setattr(
        "hermes_cli.inventory.build_models_payload",
        lambda _context: {
            "providers": [
                {"slug": "review-provider", "name": "Review", "models": ["review-model"]}
            ]
        },
    )
    manager = SimpleNamespace(_cli_ref=FakeCli())
    plugin_context = object.__new__(PluginContext)
    plugin_context._manager = manager
    callback = lambda _result: "selected"

    assert plugin_context.request_model_selection(
        callback,
        current_provider="review-provider",
        current_model="review-model",
    )
    assert opened["kwargs"]["on_selected"] is callback
    assert manager._cli_ref.provider == "primary-provider"
    assert manager._cli_ref.model == "primary-model"


def test_plugin_model_selection_dispatches_callback_without_primary_switch(monkeypatch):
    import cli as cli_module
    from hermes_cli.model_switch import ModelSwitchResult

    result = ModelSwitchResult(
        success=True,
        new_model="review-model",
        target_provider="review-provider",
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model", lambda **_kwargs: result
    )
    callback = lambda _result: "advisor selected"
    dispatched = []
    primary_applied = []
    cli = SimpleNamespace(
        _app=None,
        _model_picker_state={
            "stage": "model",
            "provider_data": {"slug": "review-provider"},
            "model_list": ["review-model"],
            "selected": 0,
            "user_provs": None,
            "custom_provs": None,
            "on_selected": callback,
        },
        provider="primary-provider",
        model="primary-model",
        base_url="",
        api_key="",
        _restore_modal_input_snapshot=lambda: None,
        _invalidate=lambda **_kwargs: None,
        _confirm_and_dispatch_model_selection=lambda selected, selected_callback: (
            dispatched.append((selected, selected_callback))
        ),
        _confirm_and_apply_model_switch_result=lambda *_args: primary_applied.append(True),
    )
    cli._close_model_picker = cli_module.HermesCLI._close_model_picker.__get__(
        cli, type(cli)
    )

    cli_module.HermesCLI._handle_model_picker_selection(cli)

    assert dispatched == [(result, callback)]
    assert primary_applied == []
    assert cli.provider == "primary-provider"
    assert cli.model == "primary-model"


def test_model_picker_open_marshals_from_plugin_thread_to_app_loop():
    import cli as cli_module

    scheduled = []
    setup_threads = []

    class Loop:
        def call_soon_threadsafe(self, callback):
            scheduled.append(callback)

    cli = SimpleNamespace(
        _app=SimpleNamespace(loop=Loop()),
        _capture_modal_input_snapshot=lambda: setup_threads.append(
            threading.current_thread()
        ),
        _invalidate=lambda **_kwargs: None,
        _model_picker_state=None,
    )

    worker = threading.Thread(
        target=lambda: cli_module.HermesCLI._open_model_picker(
            cli,
            [{"slug": "review", "is_current": True}],
            "review-model",
            "review-provider",
        )
    )
    worker.start()
    worker.join()

    assert len(scheduled) == 1
    assert setup_threads == []
    scheduled[0]()
    assert setup_threads == [threading.current_thread()]
    assert cli._model_picker_state["current_model"] == "review-model"
