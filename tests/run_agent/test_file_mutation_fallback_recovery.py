"""Regression matrix for stale file-mutation failure recovery.

A failed native mutation remains actionable unless a later successful foreground
terminal/execute_code call changes that exact backend target during its own
execution window.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from agent.turn_finalizer import finalize_turn
from run_agent import AIAgent
from tools.file_operations import ExecuteResult, PatchResult, WriteResult
import tools.code_execution_tool as code_execution_tool
import tools.file_operations as file_operations
import tools.file_tools as file_tools


class _FailingPatchOps:
    def patch_replace(self, path, old_string, new_string, replace_all=False):
        return PatchResult(error=f"Could not find {old_string!r} in {path}")


class _RemoteFailingPatchOps(_FailingPatchOps):
    def __init__(self, files: dict[str, bytes], target: str):
        self.files = files
        self.target = target
        self.patch_targets: list[str] = []
        self.fingerprint_commands: list[str] = []
        self.raise_on_fingerprint = False

    def patch_replace(self, path, old_string, new_string, replace_all=False):
        self.patch_targets.append(path)
        return super().patch_replace(path, old_string, new_string, replace_all)

    def _exec(self, command, cwd=None, timeout=None, stdin_data=None):
        self.fingerprint_commands.append(command)
        if self.raise_on_fingerprint:
            raise RuntimeError("backend fingerprint failed")
        content = self.files.get(self.target)
        if content is None:
            marker = "__HERMES_FILE_CONTENT_MISSING__"
        else:
            digest = hashlib.sha256(content).hexdigest()
            marker = f"__HERMES_FILE_CONTENT_SHA256__={digest}"
        return ExecuteResult(stdout=marker + "\n", exit_code=0)


class _FailingWriteOps(_FailingPatchOps):
    def write_file(self, path, content):
        return WriteResult(error=f"write denied for {path}")


class _RecordingWriteOps:
    def __init__(self):
        self.targets: list[str] = []
        self.env: object | None = None

    def write_file(self, path, content):
        self.targets.append(path)
        return WriteResult(bytes_written=len(content))


class _FailingV4AOps(_FailingPatchOps):
    def __init__(self):
        self.resolved_paths = None

    def patch_v4a(self, patch_content):
        return self.patch_v4a_resolved(patch_content, resolved_paths=None)

    def patch_v4a_resolved(self, patch_content, resolved_paths):
        self.resolved_paths = resolved_paths
        return PatchResult(error="Patch validation failed")


class _InheritedFailingV4AOps(_FailingV4AOps):
    """Resolved-path support inherited rather than defined on the leaf class."""


class _RecordingV4AOps(_FailingPatchOps):
    def __init__(self):
        self.resolved_paths = None
        self.calls = 0

    def patch_v4a_resolved(self, patch_content, resolved_paths):
        self.calls += 1
        self.resolved_paths = resolved_paths
        return PatchResult(success=True)


class _DynamicCapabilityMeta(type):
    def __getattr__(cls, name):
        if name != "patch_v4a_resolved":
            raise AttributeError(name)

        def fabricated(instance, patch_content, resolved_paths):
            instance.fabricated_calls += 1
            return PatchResult(error="fabricated capability used")

        return fabricated


class _DynamicCapabilityOps(_FailingPatchOps, metaclass=_DynamicCapabilityMeta):
    def __init__(self):
        self.fabricated_calls = 0
        self.legacy_calls = 0

    def patch_v4a(self, patch_content):
        self.legacy_calls += 1
        return PatchResult(error="legacy V4A validation failed")


class _LegacyV4AOps(_FailingPatchOps):
    """Backend/plugin implementation that predates resolved-path plumbing."""

    def __init__(self):
        self.calls = 0
        self.patch_content = None

    def patch_v4a(self, patch_content):
        self.calls += 1
        self.patch_content = patch_content
        return PatchResult(error="legacy V4A validation failed")


class _FinalizeAgent(AIAgent):
    def __init__(self):
        # Deliberately skip AIAgent.__init__; this fixture exercises only the
        # mutation-verifier and finalizer seams.
        self._turn_failed_file_mutations = {}
        self._turn_file_mutation_paths = set()
        self._turn_file_mutation_lock = threading.Lock()
        self._turn_file_mutation_generation = 0
        self._turn_file_mutation_epoch = object()
        self.max_iterations = 90
        self.iteration_budget = SimpleNamespace(remaining=10, used=1, max_total=90)
        self.quiet_mode = True
        self.model = "test-model"
        self.provider = "test-provider"
        self.base_url = ""
        self.session_id = "sess-test"
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
        self.session_cost_status = "unknown"
        self.session_cost_source = "test"
        self._tool_guardrail_halt_decision = None
        self._interrupt_message = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self.valid_tool_names = []

    def _handle_max_iterations(self, *_args):
        raise AssertionError("not expected")

    def _emit_status(self, *_args, **_kwargs):
        pass

    def _safe_print(self, *_args, **_kwargs):
        pass

    def _save_trajectory(self, *_args, **_kwargs):
        pass

    def _cleanup_task_resources(self, *_args, **_kwargs):
        pass

    def _drop_trailing_empty_response_scaffolding(self, *_args, **_kwargs):
        pass

    def _persist_session(self, *_args, **_kwargs):
        pass

    def _file_mutation_verifier_enabled(self):
        return True

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **_kwargs):
        pass


@pytest.fixture
def agent() -> _FinalizeAgent:
    return _FinalizeAgent()


def _reset_verifier_turn(agent: _FinalizeAgent) -> None:
    """Mirror the production per-turn verifier reset with a fresh identity."""

    agent._turn_failed_file_mutations = {}
    agent._turn_file_mutation_paths = set()
    agent._turn_file_mutation_lock = threading.Lock()
    agent._turn_file_mutation_generation = 0
    agent._turn_file_mutation_epoch = object()


@pytest.fixture
def failing_patch_ops(monkeypatch):
    ops = _FailingPatchOps()
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda _task_id="default": ops)
    monkeypatch.setattr(file_tools, "_check_file_staleness", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(file_tools, "_path_resolution_warning", lambda *_args, **_kwargs: None)
    return ops


def _prepare(agent: AIAgent, tool_name: str, args: dict, task_id: str) -> None:
    """Use the new pre-call seam when present; current main has no such seam."""
    prepare = getattr(agent, "_prepare_file_mutation_verifier_call", None)
    if prepare is not None:
        prepare(tool_name, args, task_id)


def _prepare_with_active_event(
    agent: AIAgent,
    tool_name: str,
    args: dict,
    task_id: str,
    active_event: threading.Event,
) -> None:
    """Backward-compatible probe for revocable concurrent-call evidence."""

    prepare = getattr(agent, "_prepare_file_mutation_verifier_call", None)
    if prepare is None:
        return
    try:
        prepare(tool_name, args, task_id, active_event=active_event)
    except TypeError:
        # Baseline/candidate before the cancellation contract existed.
        prepare(tool_name, args, task_id)


def _prepare_with_execution_bounds(
    agent: AIAgent,
    tool_name: str,
    args: dict,
    task_id: str,
    *,
    deadline: float | None = None,
    cancel_check=None,
) -> None:
    """Prepare verifier evidence with worker-local execution boundaries."""

    active_event = threading.Event()
    active_event.set()
    agent._prepare_file_mutation_verifier_call(
        tool_name,
        args,
        task_id,
        active_event=active_event,
        deadline=deadline,
        cancel_check=cancel_check,
    )


def _record_failed_patch(
    agent: AIAgent,
    path: str,
    *,
    task_id: str = "default",
    error_old_string: str = "missing",
) -> tuple[dict, str]:
    args = {
        "mode": "replace",
        "path": path,
        "old_string": error_old_string,
        "new_string": "replacement",
    }
    _prepare(agent, "patch", args, task_id)
    result = file_tools.patch_tool(
        mode="replace",
        path=path,
        old_string=error_old_string,
        new_string="replacement",
        task_id=task_id,
    )
    assert json.loads(result).get("error"), result
    agent._record_file_mutation_result("patch", args, result, is_error=True)
    return args, result


def _record_successful_external(
    agent: AIAgent,
    mutate,
    *,
    task_id: str = "default",
    tool_name: str = "terminal",
    args: dict | None = None,
) -> None:
    if args is None:
        args = {"command": "external fallback"}
    _prepare(agent, tool_name, args, task_id)
    mutate()
    if tool_name == "execute_code":
        result = json.dumps({"status": "success", "output": "ok"})
    else:
        result = json.dumps({"output": "ok", "exit_code": 0})
    agent._record_file_mutation_result(tool_name, args, result, is_error=False)


def _finalize(agent: _FinalizeAgent, monkeypatch) -> str:
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_args, **_kwargs: [])
    result = finalize_turn(
        agent,
        final_response="Done.",
        api_call_count=2,
        interrupted=False,
        failed=False,
        messages=[{"role": "user", "content": "edit it"}],
        conversation_history=[],
        effective_task_id="default",
        turn_id="turn",
        user_message="edit it",
        original_user_message="edit it",
        _should_review_memory=False,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )
    return result["final_response"]


def test_absolute_existing_file_external_rewrite_suppresses_failure(
    agent, failing_patch_ops, tmp_path
):
    target = tmp_path / "app.txt"
    target.write_text("before\n")

    _record_failed_patch(agent, str(target))
    _record_successful_external(agent, lambda: target.write_text("after!\n"))

    assert agent._turn_failed_file_mutations == {}
    assert agent._turn_file_mutation_paths == {str(target.resolve())}


def test_write_file_failure_external_rewrite_suppresses(
    agent, tmp_path, monkeypatch
):
    target = tmp_path / "app.txt"
    target.write_text("before\n")
    ops = _FailingWriteOps()
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda *_args: ops)
    args = {"path": str(target), "content": "intended\n"}

    _prepare(agent, "write_file", args, "default")
    result = file_tools.write_file_tool(
        path=str(target),
        content="intended\n",
        task_id="default",
    )
    assert json.loads(result).get("error")
    agent._record_file_mutation_result(
        "write_file",
        args,
        result,
        is_error=True,
    )
    _record_successful_external(agent, lambda: target.write_text("intended\n"))

    assert agent._turn_failed_file_mutations == {}
    assert agent._turn_file_mutation_paths == {str(target.resolve())}


def test_policy_blocked_write_does_not_fingerprint_denied_target(
    agent, tmp_path, monkeypatch
):
    target = tmp_path / "sensitive.txt"
    target.write_text("secret\n")
    args = {"path": str(target), "content": "replacement\n"}
    fingerprinted = []

    _prepare(agent, "write_file", args, "default")
    monkeypatch.setattr(
        file_tools,
        "_check_sensitive_path",
        lambda *_args, **_kwargs: "blocked sensitive path",
    )
    monkeypatch.setattr(
        file_tools,
        "_fingerprint_resolved_file_content",
        lambda path, task_id="default": fingerprinted.append((path, task_id)),
        raising=False,
    )
    result = file_tools.write_file_tool(
        path=str(target),
        content="replacement\n",
        task_id="default",
    )
    assert json.loads(result).get("error")
    agent._record_file_mutation_result(
        "write_file",
        args,
        result,
        is_error=True,
    )

    assert fingerprinted == []
    assert str(target) in agent._turn_failed_file_mutations
    assert agent._turn_failed_file_mutations[str(target)].get("fingerprint") is None


def test_policy_blocked_write_defers_fingerprint_until_external_fallback(
    agent, tmp_path, monkeypatch
):
    target = tmp_path / "sensitive.txt"
    target.write_text("before\n")
    args = {"path": str(target), "content": "replacement\n"}

    _prepare(agent, "write_file", args, "default")
    monkeypatch.setattr(
        file_tools,
        "_check_sensitive_path",
        lambda *_args, **_kwargs: "blocked sensitive path",
    )
    result = file_tools.write_file_tool(
        path=str(target),
        content="replacement\n",
        task_id="default",
    )
    assert json.loads(result).get("error")
    agent._record_file_mutation_result(
        "write_file",
        args,
        result,
        is_error=True,
    )

    state = agent._turn_failed_file_mutations[str(target.resolve())]
    assert state.get("fingerprint") is None
    _record_successful_external(agent, lambda: target.write_text("replacement\n"))

    assert agent._turn_failed_file_mutations == {}
    assert agent._turn_file_mutation_paths == {str(target.resolve())}


@pytest.mark.parametrize("tool_kind", ["write", "v4a"])
def test_policy_checks_the_exact_captured_mutation_identity(
    agent, tmp_path, monkeypatch, tool_kind
):
    protected = str((tmp_path / "protected-config.yaml").resolve())
    safe = str((tmp_path / "safe.yaml").resolve())
    resolver_calls = 0

    def changing_resolver(_raw, _task_id="default"):
        nonlocal resolver_calls
        resolver_calls += 1
        return Path(protected if resolver_calls == 1 else safe)

    monkeypatch.setattr(file_tools, "_resolve_path_for_task", changing_resolver)
    monkeypatch.setattr(file_tools, "_get_hermes_config_resolved", lambda: protected)
    monkeypatch.setattr(file_tools, "_check_cross_profile_path", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        file_tools,
        "_fingerprint_resolved_file_content",
        lambda *_args, **_kwargs: None,
    )

    if tool_kind == "write":
        ops = _RecordingWriteOps()
        monkeypatch.setattr(file_tools, "_get_file_ops", lambda *_args: ops)
        result = json.loads(
            file_tools.write_file_tool(
                path="relative-config.yaml",
                content="replacement\n",
                task_id="default",
            )
        )
        calls = len(ops.targets)
    else:
        ops = _RecordingV4AOps()
        monkeypatch.setattr(file_tools, "_get_file_ops", lambda *_args: ops)
        stars = "***"
        body = (
            f"{stars}Begin Patch\n"
            f"{stars}Update File: relative-config.yaml\n"
            "@@\n"
            "-before\n"
            "+after\n"
            f"{stars}End Patch\n"
        )
        result = json.loads(
            file_tools.patch_tool(mode="patch", patch=body, task_id="default")
        )
        calls = ops.calls

    assert "error" in result
    assert "Hermes config" in result["error"]
    assert calls == 0


def test_cross_profile_policy_uses_the_exact_captured_mutation_identity(
    tmp_path, monkeypatch
):
    from agent import file_safety

    protected = str((tmp_path / "profiles" / "other" / "skills" / "x.md").resolve())
    safe = str((tmp_path / "safe.md").resolve())
    resolver_calls = 0

    def changing_resolver(_raw, _task_id="default"):
        nonlocal resolver_calls
        resolver_calls += 1
        return Path(protected if resolver_calls == 1 else safe)

    ops = _RecordingWriteOps()
    monkeypatch.setattr(file_tools, "_resolve_path_for_task", changing_resolver)
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda *_args: ops)
    monkeypatch.setattr(file_tools, "_get_hermes_config_resolved", lambda: None)
    monkeypatch.setattr(
        file_tools,
        "_fingerprint_resolved_file_content",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        file_safety,
        "get_cross_profile_warning",
        lambda resolved: "cross-profile identity blocked" if resolved == protected else None,
    )
    monkeypatch.setattr(file_safety, "get_sandbox_mirror_warning", lambda _resolved: None)
    monkeypatch.setattr(
        file_safety,
        "get_container_mirror_warning",
        lambda _resolved, mirror_prefix=None: None,
    )

    result = json.loads(
        file_tools.write_file_tool(
            path="relative.md",
            content="replacement\n",
            task_id="default",
        )
    )

    assert result.get("error") == "cross-profile identity blocked"
    assert resolver_calls == 1
    assert ops.targets == []


@pytest.mark.skipif(
    not code_execution_tool.SANDBOX_AVAILABLE,
    reason="execute_code sandbox unavailable",
)
def test_real_local_execute_code_rewrite_suppresses(
    agent, failing_patch_ops, tmp_path, monkeypatch
):
    """The real local sandbox shares absolute target identity with file tools."""

    target = tmp_path / "app.txt"
    target.write_text("before\n")
    _record_failed_patch(agent, str(target))

    code = (
        "from pathlib import Path\n"
        f"Path({str(target)!r}).write_text('sandbox-after\\n', encoding='utf-8')\n"
        "print('done')\n"
    )
    args = {"code": code}
    agent._prepare_file_mutation_verifier_call(
        "execute_code",
        args,
        "default",
        defer_probe_activation=True,
    )
    monkeypatch.setattr(
        "tools.terminal_tool._get_env_config",
        lambda: {"env_type": "local"},
    )
    monkeypatch.setattr(
        "tools.terminal_tool._docker_has_host_access",
        lambda _config: False,
    )
    monkeypatch.setattr(
        "tools.approval.check_execute_code_guard",
        lambda *_args, **_kwargs: {"approved": True},
    )
    monkeypatch.setattr(
        code_execution_tool,
        "_load_config",
        lambda: {"timeout": 15, "max_tool_calls": 1},
    )
    monkeypatch.setattr(code_execution_tool, "_get_execution_mode", lambda: "strict")

    result = code_execution_tool.execute_code(
        code=code,
        task_id="default",
        enabled_tools=[],
    )
    payload = json.loads(result)
    assert payload.get("status") == "success", payload
    agent._record_file_mutation_result(
        "execute_code",
        args,
        result,
        is_error=False,
    )

    assert target.read_text() == "sandbox-after\n"
    assert agent._turn_failed_file_mutations == {}
    assert agent._turn_file_mutation_paths == {str(target.resolve())}


@pytest.mark.parametrize("fallback_tool", ["terminal", "execute_code"])
def test_denied_real_fallback_gate_does_not_activate_deferred_fingerprint(
    agent, tmp_path, monkeypatch, fallback_tool
):
    target = tmp_path / "sensitive.txt"
    target.write_text("secret\n")
    write_args = {"path": str(target), "content": "replacement\n"}
    _prepare(agent, "write_file", write_args, "default")
    monkeypatch.setattr(
        file_tools,
        "_check_sensitive_path",
        lambda *_args, **_kwargs: "blocked sensitive path",
    )
    write_result = file_tools.write_file_tool(
        path=str(target),
        content="replacement\n",
        task_id="default",
    )
    agent._record_file_mutation_result(
        "write_file",
        write_args,
        write_result,
        is_error=True,
    )
    state = agent._turn_failed_file_mutations[str(target.resolve())]
    assert state.get("fingerprint_deferred") is True

    fingerprinted: list[str] = []
    monkeypatch.setattr(
        file_tools,
        "_fingerprint_resolved_file_content",
        lambda path, _task_id="default": fingerprinted.append(path),
    )

    if fallback_tool == "execute_code":
        fallback_args = {"code": "print('denied')"}
        agent._prepare_file_mutation_verifier_call(
            fallback_tool,
            fallback_args,
            "default",
            defer_probe_activation=True,
        )
        monkeypatch.setattr(
            "tools.terminal_tool._get_env_config",
            lambda: {"env_type": "local"},
        )
        monkeypatch.setattr(
            "tools.terminal_tool._docker_has_host_access",
            lambda _config: False,
        )
        monkeypatch.setattr(
            "tools.approval.check_execute_code_guard",
            lambda *_args, **_kwargs: {
                "approved": False,
                "message": "denied by execute_code guard",
            },
        )
        fallback_result = code_execution_tool.execute_code(
            code=fallback_args["code"],
            task_id="default",
            enabled_tools=[],
        )
    else:
        import tools.terminal_tool as terminal_tool

        fallback_args = {"command": "denied fallback"}
        agent._prepare_file_mutation_verifier_call(
            fallback_tool,
            fallback_args,
            "default",
            defer_probe_activation=True,
        )
        monkeypatch.setattr(
            terminal_tool,
            "_get_env_config",
            lambda: {"env_type": "local", "cwd": str(tmp_path), "timeout": 30},
        )
        monkeypatch.setattr(terminal_tool, "resolve_task_overrides", lambda *_args: {})
        monkeypatch.setattr(terminal_tool, "_resolve_container_task_id", lambda *_args: "default")
        monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
        monkeypatch.setitem(
            terminal_tool._active_environments,
            "default",
            SimpleNamespace(cwd=str(tmp_path)),
        )
        monkeypatch.setattr(
            terminal_tool,
            "_check_all_guards",
            lambda *_args, **_kwargs: {
                "approved": False,
                "status": "blocked",
                "message": "denied by terminal guard",
            },
        )
        fallback_result = terminal_tool.terminal_tool(
            command=fallback_args["command"],
            task_id="default",
        )

    assert json.loads(fallback_result).get("error")
    assert fingerprinted == []
    assert state.get("fingerprint") is None
    assert state.get("fingerprint_deferred") is True
    agent._prepare_file_mutation_verifier_call("read_file", {}, "default")


def test_absolute_unchanged_file_retains_failure(agent, failing_patch_ops, tmp_path):
    target = tmp_path / "app.txt"
    target.write_text("before\n")

    _record_failed_patch(agent, str(target))
    _record_successful_external(agent, lambda: None)

    assert str(target.resolve()) in agent._turn_failed_file_mutations


def test_unlanded_native_result_cannot_clear_prior_failure(
    agent, failing_patch_ops, tmp_path
):
    """A malformed success classification is not proof that a patch landed."""

    target = tmp_path / "app.txt"
    target.write_text("before\n")
    args, _ = _record_failed_patch(agent, str(target))

    _prepare(agent, "patch", args, "default")
    agent._record_file_mutation_result(
        "patch",
        args,
        json.dumps({"success": False}),
        is_error=False,
    )

    assert str(target.resolve()) in agent._turn_failed_file_mutations
    assert agent._turn_file_mutation_paths == set()


@pytest.mark.parametrize("source", ["live_cwd", "workspace_override"])
def test_relative_target_uses_effective_task_workspace_not_process_cwd(
    agent, failing_patch_ops, tmp_path, monkeypatch, source
):
    workspace = tmp_path / "workspace"
    process_cwd = tmp_path / "process-cwd"
    workspace.mkdir()
    process_cwd.mkdir()
    intended = workspace / "app.txt"
    decoy = process_cwd / "app.txt"
    intended.write_text("workspace-before\n")
    decoy.write_text("decoy-before\n")
    monkeypatch.chdir(process_cwd)
    task_id = "session-worktree"

    if source == "live_cwd":
        monkeypatch.setattr(
            file_tools,
            "_get_live_tracking_cwd",
            lambda incoming="default": str(workspace) if incoming == task_id else None,
        )
        monkeypatch.setattr(file_tools, "_registered_task_cwd_override", lambda *_args: None)
    else:
        monkeypatch.setattr(file_tools, "_get_live_tracking_cwd", lambda *_args: None)
        monkeypatch.setattr(
            file_tools,
            "_registered_task_cwd_override",
            lambda incoming="default": str(workspace) if incoming == task_id else None,
        )

    _record_failed_patch(agent, "app.txt", task_id=task_id)
    _record_successful_external(
        agent,
        lambda: intended.write_text("workspace-after!\n"),
        task_id=task_id,
    )

    assert agent._turn_failed_file_mutations == {}
    assert agent._turn_file_mutation_paths == {str(intended.resolve())}
    assert decoy.read_text() == "decoy-before\n"


def test_v4a_uses_captured_resolved_identity_for_backend_calls_and_recovery(
    agent, tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    process_cwd = tmp_path / "process-cwd"
    workspace.mkdir()
    process_cwd.mkdir()
    target = workspace / "app.txt"
    target.write_text("before\n")
    monkeypatch.chdir(process_cwd)
    task_id = "v4a-worktree"
    monkeypatch.setattr(
        file_tools,
        "_get_live_tracking_cwd",
        lambda incoming="default": str(workspace) if incoming == task_id else None,
    )
    monkeypatch.setattr(file_tools, "_registered_task_cwd_override", lambda *_args: None)
    monkeypatch.setattr(file_tools, "_check_file_staleness", lambda *_args: None)
    monkeypatch.setattr(file_tools, "_path_resolution_warning", lambda *_args: None)
    ops = _InheritedFailingV4AOps()
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda *_args: ops)
    stars = "***"
    body = (
        f"{stars}Begin Patch\n"
        f"{stars}Update File: app.txt\n"
        "@@\n"
        "-missing\n"
        "+replacement\n"
        f"{stars}End Patch\n"
    )
    args = {"mode": "patch", "patch": body}

    _prepare(agent, "patch", args, task_id)
    result = file_tools.patch_tool(
        mode="patch",
        patch=body,
        task_id=task_id,
    )
    assert json.loads(result).get("error")
    agent._record_file_mutation_result("patch", args, result, is_error=True)

    expected = str(target.resolve())
    assert ops.resolved_paths == {"app.txt": expected}
    assert any(
        info["display_path"] == expected
        for info in agent._turn_failed_file_mutations.values()
    )

    _record_successful_external(
        agent,
        lambda: target.write_text("after!\n"),
        task_id=task_id,
    )
    assert agent._turn_failed_file_mutations == {}


def test_v4a_captures_resolved_identities_for_add_delete_move_and_update(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ops = _RecordingV4AOps()

    def resolve_in_workspace(raw: str, task_id: str = "default"):
        candidate = Path(raw)
        return candidate if candidate.is_absolute() else workspace / candidate

    monkeypatch.setattr(file_tools, "_resolve_path_for_task", resolve_in_workspace)
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda *_args: ops)
    monkeypatch.setattr(file_tools, "_check_sensitive_path", lambda *_args: None)
    monkeypatch.setattr(file_tools, "_check_cross_profile_path", lambda *_args: None)
    monkeypatch.setattr(file_tools, "_check_file_staleness", lambda *_args: None)
    monkeypatch.setattr(file_tools, "_path_resolution_warning", lambda *_args: None)
    monkeypatch.setattr(
        file_tools,
        "_fingerprint_resolved_file_content",
        lambda *_args, **_kwargs: None,
    )
    stars = "***"
    body = (
        f"{stars}Begin Patch\n"
        f"{stars}Update File: update.txt\n"
        "@@\n-old\n+new\n"
        f"{stars}Add File: add.txt\n"
        "+new file\n"
        f"{stars}Delete File: delete.txt\n"
        f"{stars}Move File: source.txt -> moved.txt\n"
        f"{stars}End Patch\n"
    )

    payload = json.loads(file_tools.patch_tool(mode="patch", patch=body))

    assert payload.get("success") is True
    assert ops.resolved_paths == {
        name: str(workspace / name)
        for name in (
            "update.txt",
            "add.txt",
            "delete.txt",
            "source.txt",
            "moved.txt",
        )
    }


def test_v4a_legacy_backend_signature_remains_compatible(
    agent, tmp_path, monkeypatch
):
    target = tmp_path / "app.txt"
    target.write_text("before\n")
    ops = _LegacyV4AOps()
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda *_args: ops)
    monkeypatch.setattr(file_tools, "_check_file_staleness", lambda *_args: None)
    monkeypatch.setattr(file_tools, "_path_resolution_warning", lambda *_args: None)
    stars = "***"
    body = (
        f"{stars}Begin Patch\n"
        f"{stars}Update File: {target}\n"
        "@@\n"
        "-missing\n"
        "+replacement\n"
        f"{stars}End Patch\n"
    )

    args = {"mode": "patch", "patch": body}
    _prepare(agent, "patch", args, "default")
    result = file_tools.patch_tool(
        mode="patch",
        patch=body,
        task_id="default",
    )
    payload = json.loads(result)

    assert payload.get("error") == "legacy V4A validation failed"
    assert ops.calls == 1

    # A legacy backend does not expose the capability needed to prove that it
    # used the tool layer's resolved identity. Preserve compatibility, but keep
    # recovery fail-closed instead of clearing a warning for a possibly
    # different target.
    agent._record_file_mutation_result("patch", args, result, is_error=True)
    _record_successful_external(agent, lambda: target.write_text("after!\n"))
    assert agent._turn_failed_file_mutations
    assert agent._turn_file_mutation_paths == set()


def test_legacy_v4a_rewrites_relative_header_to_captured_identity(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "app.txt"
    target.write_text("before\n")
    task_id = "legacy-worktree"
    ops = _LegacyV4AOps()
    monkeypatch.setattr(
        file_tools,
        "_resolve_path_for_task",
        lambda raw, _task_id="default": workspace / raw,
    )
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda *_args: ops)
    monkeypatch.setattr(file_tools, "_check_sensitive_path", lambda *_args: None)
    monkeypatch.setattr(file_tools, "_check_cross_profile_path", lambda *_args: None)
    monkeypatch.setattr(file_tools, "_check_file_staleness", lambda *_args: None)
    monkeypatch.setattr(file_tools, "_path_resolution_warning", lambda *_args: None)
    stars = "***"
    body = (
        f"{stars}Begin Patch\n"
        f"{stars}Update File: app.txt\n"
        "@@\n"
        "-before\n"
        "+after\n"
        f"{stars}Add File: add.txt\n"
        "+new\n"
        f"{stars}Delete File: delete.txt\n"
        f"{stars}Move File: source.txt -> moved.txt\n"
        f"{stars}End Patch\n"
    )

    payload = json.loads(
        file_tools.patch_tool(
            mode="patch",
            patch=body,
            task_id=task_id,
        )
    )

    assert payload.get("error") == "legacy V4A validation failed"
    assert ops.patch_content is not None
    expected_headers = (
        f"{stars}Update File: {workspace / 'app.txt'}",
        f"{stars}Add File: {workspace / 'add.txt'}",
        f"{stars}Delete File: {workspace / 'delete.txt'}",
        (
            f"{stars}Move File: {workspace / 'source.txt'} -> "
            f"{workspace / 'moved.txt'}"
        ),
    )
    assert all(header in ops.patch_content for header in expected_headers)
    assert f"{stars}Update File: app.txt\n" not in ops.patch_content


def test_resolved_v4a_adapter_rejects_missing_identity_mapping():
    delegate = object.__new__(file_operations.ShellFileOperations)
    adapter = file_operations._ResolvedPathFileOperations(
        delegate,
        {"captured.txt": "/workspace/captured.txt"},
    )

    with pytest.raises(ValueError, match="No captured resolved identity"):
        adapter.read_file_raw("uncaptured.txt")


def test_v4a_capability_detection_ignores_dynamic_metaclass_attributes(
    tmp_path, monkeypatch
):
    target = tmp_path / "app.txt"
    target.write_text("before\n")
    ops = _DynamicCapabilityOps()
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda *_args: ops)
    monkeypatch.setattr(file_tools, "_check_sensitive_path", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(file_tools, "_check_cross_profile_path", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(file_tools, "_check_file_staleness", lambda *_args: None)
    monkeypatch.setattr(file_tools, "_path_resolution_warning", lambda *_args: None)
    stars = "***"
    body = (
        f"{stars}Begin Patch\n"
        f"{stars}Update File: {target}\n"
        "@@\n"
        "-before\n"
        "+after\n"
        f"{stars}End Patch\n"
    )

    payload = json.loads(file_tools.patch_tool(mode="patch", patch=body))

    assert payload.get("error") == "legacy V4A validation failed"
    assert ops.legacy_calls == 1
    assert ops.fabricated_calls == 0


def test_ssh_resolution_uses_remote_home_and_never_dereferences_host_symlinks(
    tmp_path, monkeypatch
):
    task_id = "ssh-task"
    env = SimpleNamespace(
        _remote_home="/home/remote-user",
        cwd="/worktree",
        cwd_owner=task_id,
    )
    monkeypatch.setattr(file_tools, "_terminal_env_type_for_task", lambda *_args: "ssh")
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda *_args: SimpleNamespace(env=env))
    monkeypatch.setattr(file_tools, "_get_live_tracking_cwd", lambda *_args: "/worktree")

    assert file_tools._resolve_path_for_task("~/notes.txt", task_id) == PurePosixPath(
        "/home/remote-user/notes.txt"
    )

    host_target = tmp_path / "host-target.txt"
    host_target.write_text("host\n")
    host_link = tmp_path / "remote-link.txt"
    host_link.symlink_to(host_target)
    assert file_tools._resolve_path_for_task(str(host_link), task_id) == PurePosixPath(
        str(host_link)
    )


def test_ssh_tilde_sensitive_policy_uses_captured_remote_identity(
    monkeypatch,
):
    task_id = "ssh-sensitive"
    env = SimpleNamespace(
        _remote_home="/home/remote-user",
        cwd="/worktree",
        cwd_owner=task_id,
    )
    ops = _RecordingWriteOps()
    ops.env = env
    monkeypatch.setattr(file_tools, "_terminal_env_type_for_task", lambda *_args: "ssh")
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda *_args: ops)
    monkeypatch.setattr(file_tools, "_get_live_tracking_cwd", lambda *_args: "/worktree")
    monkeypatch.setattr(
        file_tools,
        "_get_hermes_config_resolved",
        lambda: "/root/.hermes/config.yaml",
    )
    monkeypatch.setattr(file_tools, "_check_cross_profile_path", lambda *_args: None)
    monkeypatch.setattr(file_tools, "_check_file_staleness", lambda *_args: None)
    monkeypatch.setattr(file_tools, "_path_resolution_warning", lambda *_args: None)
    monkeypatch.setattr(
        file_tools,
        "_fingerprint_resolved_file_content",
        lambda *_args, **_kwargs: None,
    )

    payload = json.loads(
        file_tools.write_file_tool(
            path="~/.hermes/config.yaml",
            content="remote-only\n",
            task_id=task_id,
        )
    )

    expected = "/home/remote-user/.hermes/config.yaml"
    assert payload.get("error") is None
    assert payload.get("resolved_path") == expected
    assert ops.targets == [expected]


def test_first_docker_mutation_resolves_after_backend_cwd_is_initialized(
    monkeypatch,
):
    task_id = "docker-first-call"
    live = False
    ops = _RecordingWriteOps()

    def get_ops(*_args):
        nonlocal live
        live = True
        return ops

    monkeypatch.setattr(file_tools, "_terminal_env_type_for_task", lambda *_args: "docker")
    monkeypatch.setattr(
        file_tools,
        "_get_live_tracking_cwd",
        lambda *_args: "/workspace" if live else None,
    )
    monkeypatch.setattr(
        file_tools,
        "_registered_task_cwd_override",
        lambda *_args: "/host/users/josh/project",
    )
    monkeypatch.setattr(file_tools, "_last_known_cwd_for", lambda *_args: None)
    monkeypatch.setattr(file_tools, "_configured_terminal_cwd", lambda: None)
    monkeypatch.setattr(file_tools, "_get_file_ops", get_ops)
    monkeypatch.setattr(file_tools, "_check_sensitive_path", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(file_tools, "_check_cross_profile_path", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(file_tools, "_check_file_staleness", lambda *_args: None)
    monkeypatch.setattr(file_tools, "_path_resolution_warning", lambda *_args: None)
    monkeypatch.setattr(
        file_tools,
        "_fingerprint_resolved_file_content",
        lambda *_args, **_kwargs: None,
    )

    payload = json.loads(
        file_tools.write_file_tool(
            path="app.txt",
            content="content\n",
            task_id=task_id,
        )
    )

    assert payload.get("resolved_path") == "/workspace/app.txt"
    assert ops.targets == ["/workspace/app.txt"]


@pytest.mark.parametrize("backend", ["docker", "ssh"])
def test_nonlocal_target_is_fingerprinted_through_backend_not_host(
    agent, monkeypatch, backend
):
    target = "/workspace/app.txt"
    remote_files = {target: b"before\n"}
    ops = _RemoteFailingPatchOps(remote_files, target)

    def resolve_remote(raw: str, task_id: str = "default"):
        candidate = PurePosixPath(raw)
        return candidate if candidate.is_absolute() else PurePosixPath("/workspace") / candidate

    monkeypatch.setattr(file_tools, "_terminal_env_type_for_task", lambda *_args: backend)
    monkeypatch.setattr(file_tools, "_resolve_path_for_task", resolve_remote)
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda *_args: ops)
    monkeypatch.setattr(file_tools, "_check_sensitive_path", lambda *_args: None)
    monkeypatch.setattr(file_tools, "_check_cross_profile_path", lambda *_args: None)
    monkeypatch.setattr(file_tools, "_check_file_staleness", lambda *_args: None)
    monkeypatch.setattr(file_tools, "_path_resolution_warning", lambda *_args: None)

    _record_failed_patch(agent, "app.txt", task_id="remote-task")
    _record_successful_external(
        agent,
        lambda: remote_files.__setitem__(target, b"after!\n"),
        task_id="remote-task",
    )

    assert agent._turn_failed_file_mutations == {}
    assert agent._turn_file_mutation_paths == {target}
    assert ops.patch_targets == [target]
    assert ops.fingerprint_commands
    assert all(target in command for command in ops.fingerprint_commands)


def test_same_resolved_path_in_different_tasks_keeps_independent_failures(
    agent, monkeypatch
):
    target = "/workspace/app.txt"
    ops_by_task = {
        "task-a": _RemoteFailingPatchOps({target: b"before-a\n"}, target),
        "task-b": _RemoteFailingPatchOps({target: b"before-b\n"}, target),
    }

    monkeypatch.setattr(file_tools, "_terminal_env_type_for_task", lambda *_args: "docker")
    monkeypatch.setattr(
        file_tools,
        "_resolve_path_for_task",
        lambda _raw, task_id="default": PurePosixPath(target),
    )
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda task_id="default": ops_by_task[task_id])
    monkeypatch.setattr(file_tools, "_check_sensitive_path", lambda *_args: None)
    monkeypatch.setattr(file_tools, "_check_cross_profile_path", lambda *_args: None)
    monkeypatch.setattr(file_tools, "_check_file_staleness", lambda *_args: None)
    monkeypatch.setattr(file_tools, "_path_resolution_warning", lambda *_args: None)

    _record_failed_patch(agent, "app.txt", task_id="task-a")
    _record_failed_patch(agent, "app.txt", task_id="task-b")

    assert len(agent._turn_failed_file_mutations) == 2
    assert {info["task_id"] for info in agent._turn_failed_file_mutations.values()} == {
        "task-a",
        "task-b",
    }

    _record_successful_external(
        agent,
        lambda: ops_by_task["task-a"].files.__setitem__(target, b"after-a\n"),
        task_id="task-a",
    )

    assert len(agent._turn_failed_file_mutations) == 1
    remaining = next(iter(agent._turn_failed_file_mutations.values()))
    assert remaining["task_id"] == "task-b"


def test_metadata_only_touch_and_chmod_do_not_suppress(
    agent, failing_patch_ops, tmp_path
):
    target = tmp_path / "app.txt"
    target.write_text("same content\n")
    original_mode = target.stat().st_mode & 0o777

    _record_failed_patch(agent, str(target))

    def metadata_only():
        os.utime(target, None)
        target.chmod(original_mode ^ 0o100)

    _record_successful_external(agent, metadata_only)

    assert str(target.resolve()) in agent._turn_failed_file_mutations


def test_same_size_rewrite_with_preserved_metadata_suppresses(
    agent, failing_patch_ops, tmp_path
):
    target = tmp_path / "app.txt"
    target.write_bytes(b"AAAA\n")
    before = target.stat()

    _record_failed_patch(agent, str(target))

    def rewrite_preserving_metadata():
        target.write_bytes(b"BBBB\n")
        target.chmod(before.st_mode & 0o777)
        os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))

    _record_successful_external(agent, rewrite_preserving_metadata)

    assert agent._turn_failed_file_mutations == {}


def test_failure_fallback_then_later_failure_is_new_actionable_generation(
    agent, failing_patch_ops, tmp_path
):
    target = tmp_path / "app.txt"
    target.write_text("first\n")

    _record_failed_patch(agent, str(target), error_old_string="missing-first")
    _record_successful_external(agent, lambda: target.write_text("fallback\n"))
    assert agent._turn_failed_file_mutations == {}

    _record_failed_patch(agent, str(target), error_old_string="missing-second")

    info = agent._turn_failed_file_mutations[str(target.resolve())]
    assert info["generation"] >= 2
    assert "missing-second" in info["error_preview"]


def test_successful_external_tool_before_failure_does_not_suppress(
    agent, failing_patch_ops, tmp_path
):
    target = tmp_path / "app.txt"
    target.write_text("before\n")

    _record_successful_external(agent, lambda: target.write_text("earlier\n"))
    _record_failed_patch(agent, str(target))

    assert str(target.resolve()) in agent._turn_failed_file_mutations


@pytest.mark.parametrize("background", [True, 1, "true"])
def test_background_terminal_launch_does_not_suppress_later_watcher_change(
    agent, failing_patch_ops, tmp_path, background
):
    target = tmp_path / "app.txt"
    target.write_text("before\n")
    _record_failed_patch(agent, str(target))

    _record_successful_external(
        agent,
        lambda: target.write_text("watcher-change\n"),
        args={"command": "watcher", "background": background},
    )

    assert str(target.resolve()) in agent._turn_failed_file_mutations


def test_unresolvable_path_retains_warning(agent):
    malformed = "bad\x00path.txt"
    args = {
        "mode": "replace",
        "path": malformed,
        "old_string": "x",
        "new_string": "y",
    }
    _prepare(agent, "patch", args, "default")
    agent._record_file_mutation_result(
        "patch",
        args,
        json.dumps({"error": "malformed path"}),
        is_error=True,
    )
    _record_successful_external(agent, lambda: None)

    assert agent._turn_failed_file_mutations
    assert agent._turn_file_mutation_paths == set()


def test_backend_fingerprint_exception_retains_warning(agent, monkeypatch):
    target = "/workspace/app.txt"
    remote_files = {target: b"before\n"}
    ops = _RemoteFailingPatchOps(remote_files, target)
    ops.raise_on_fingerprint = True

    monkeypatch.setattr(file_tools, "_terminal_env_type_for_task", lambda *_args: "docker")
    monkeypatch.setattr(
        file_tools,
        "_resolve_path_for_task",
        lambda raw, task_id="default": PurePosixPath(target),
    )
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda *_args: ops)
    monkeypatch.setattr(file_tools, "_check_sensitive_path", lambda *_args: None)
    monkeypatch.setattr(file_tools, "_check_cross_profile_path", lambda *_args: None)
    monkeypatch.setattr(file_tools, "_check_file_staleness", lambda *_args: None)
    monkeypatch.setattr(file_tools, "_path_resolution_warning", lambda *_args: None)

    _record_failed_patch(agent, "app.txt", task_id="remote-task")
    _record_successful_external(
        agent,
        lambda: remote_files.__setitem__(target, b"after!\n"),
        task_id="remote-task",
    )

    assert any(
        info["display_path"] == target
        for info in agent._turn_failed_file_mutations.values()
    )
    assert agent._turn_file_mutation_paths == set()


@pytest.mark.parametrize(
    ("tool_name", "args", "result"),
    [
        ("terminal", {"command": "fallback"}, json.dumps({"output": "ok"})),
        (
            "execute_code",
            {"code": "print('fallback')"},
            json.dumps({"status": "unknown", "output": "ok"}),
        ),
    ],
)
def test_malformed_external_success_result_cannot_suppress(
    agent, failing_patch_ops, tmp_path, tool_name, args, result
):
    target = tmp_path / "app.txt"
    target.write_text("before\n")
    _record_failed_patch(agent, str(target))

    _prepare(agent, tool_name, args, "default")
    target.write_text("after!\n")
    agent._record_file_mutation_result(
        tool_name,
        args,
        result,
        is_error=False,
    )

    assert str(target.resolve()) in agent._turn_failed_file_mutations
    assert agent._turn_file_mutation_paths == set()


def test_revoked_concurrent_fallback_evidence_cannot_suppress(
    agent, failing_patch_ops, tmp_path
):
    target = tmp_path / "app.txt"
    target.write_text("before\n")
    _record_failed_patch(agent, str(target))
    args = {"command": "late concurrent fallback"}
    active_event = threading.Event()
    active_event.set()

    _prepare_with_active_event(agent, "terminal", args, "default", active_event)
    active_event.clear()  # dispatcher synthesized timeout/cancellation
    target.write_text("late-change\n")
    agent._record_file_mutation_result(
        "terminal",
        args,
        json.dumps({"output": "ok", "exit_code": 0}),
        is_error=False,
    )

    assert str(target.resolve()) in agent._turn_failed_file_mutations
    assert agent._turn_file_mutation_paths == set()


@pytest.mark.parametrize("boundary", ["deadline", "interrupt"])
def test_expired_or_interrupted_fallback_context_cannot_suppress(
    agent, failing_patch_ops, tmp_path, boundary
):
    target = tmp_path / "app.txt"
    target.write_text("before\n")
    _record_failed_patch(agent, str(target))
    args = {"command": "bounded fallback"}
    interrupted = threading.Event()
    deadline = time.monotonic() + 0.2 if boundary == "deadline" else None

    _prepare_with_execution_bounds(
        agent,
        "terminal",
        args,
        "default",
        deadline=deadline,
        cancel_check=interrupted.is_set,
    )
    if boundary == "deadline":
        time.sleep(0.25)
    else:
        interrupted.set()
    target.write_text("late-change\n")
    agent._record_file_mutation_result(
        "terminal",
        args,
        json.dumps({"output": "ok", "exit_code": 0}),
        is_error=False,
    )

    assert str(target.resolve()) in agent._turn_failed_file_mutations
    assert agent._turn_file_mutation_paths == set()


def test_detached_fallback_from_old_turn_cannot_clear_replacement_turn(
    agent, failing_patch_ops, tmp_path
):
    target = tmp_path / "app.txt"
    target.write_text("before\n")
    _record_failed_patch(agent, str(target), error_old_string="old-turn")
    args = {"command": "detached old-turn fallback"}
    _prepare(agent, "terminal", args, "default")

    _reset_verifier_turn(agent)
    errors = []

    def record_new_turn_failure():
        try:
            _record_failed_patch(agent, str(target), error_old_string="new-turn")
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    thread = threading.Thread(target=record_new_turn_failure)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert errors == []

    target.write_text("late-old-turn-change\n")
    agent._record_file_mutation_result(
        "terminal",
        args,
        json.dumps({"output": "ok", "exit_code": 0}),
        is_error=False,
    )

    info = agent._turn_failed_file_mutations[str(target.resolve())]
    assert "new-turn" in info["error_preview"]
    assert agent._turn_file_mutation_paths == set()


def test_detached_native_success_from_old_turn_cannot_clear_replacement_turn(
    agent, failing_patch_ops, tmp_path
):
    target = tmp_path / "app.txt"
    target.write_text("before\n")
    old_args = {"path": str(target), "content": "old-turn-content\n"}
    _prepare(agent, "write_file", old_args, "default")
    file_tools._capture_mutation_target_identities([str(target)], "default")
    old_result = json.dumps(
        {
            "bytes_written": len(old_args["content"]),
            "resolved_path": str(target.resolve()),
            "files_modified": [str(target.resolve())],
        }
    )

    _reset_verifier_turn(agent)
    errors = []

    def record_new_turn_failure():
        try:
            _record_failed_patch(agent, str(target), error_old_string="new-turn")
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    thread = threading.Thread(target=record_new_turn_failure)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert errors == []

    agent._record_file_mutation_result(
        "write_file",
        old_args,
        old_result,
        is_error=False,
    )

    info = agent._turn_failed_file_mutations[str(target.resolve())]
    assert "new-turn" in info["error_preview"]
    assert agent._turn_file_mutation_paths == set()


def test_concurrent_fallback_started_before_failure_cannot_suppress(
    agent, failing_patch_ops, tmp_path
):
    target = tmp_path / "app.txt"
    target.write_text("before\n")
    fallback_started = threading.Event()
    finish_fallback = threading.Event()

    def fallback_worker():
        args = {"command": "concurrent fallback"}
        _prepare(agent, "terminal", args, "default")
        fallback_started.set()
        assert finish_fallback.wait(timeout=5)
        target.write_text("after!\n")
        agent._record_file_mutation_result(
            "terminal",
            args,
            json.dumps({"output": "ok", "exit_code": 0}),
            is_error=False,
        )

    thread = threading.Thread(target=fallback_worker)
    thread.start()
    assert fallback_started.wait(timeout=5)
    _record_failed_patch(agent, str(target))
    finish_fallback.set()
    thread.join(timeout=5)
    assert not thread.is_alive()

    assert str(target.resolve()) in agent._turn_failed_file_mutations


def test_finalize_turn_omits_footer_after_ordered_content_change(
    agent, failing_patch_ops, tmp_path, monkeypatch
):
    target = tmp_path / "app.txt"
    target.write_text("before\n")
    _record_failed_patch(agent, str(target))
    _record_successful_external(agent, lambda: target.write_text("after!\n"))

    response = _finalize(agent, monkeypatch)

    assert "File-mutation verifier" not in response


def test_finalize_turn_retains_footer_when_target_unchanged(
    agent, failing_patch_ops, tmp_path, monkeypatch
):
    target = tmp_path / "app.txt"
    target.write_text("before\n")
    _record_failed_patch(agent, str(target))
    _record_successful_external(agent, lambda: None)

    response = _finalize(agent, monkeypatch)

    assert "File-mutation verifier" in response
    assert str(target.resolve()) in response
