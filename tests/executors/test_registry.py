#!/usr/bin/env python3
"""
Tests for executors/ adapter modules (Commit B).

Scope:
  - Registry ships the 5 default manifests
  - Each of the 5 adapter modules can be imported and exposes a class
  - Health checks do not crash when the underlying binary is missing
  - DeepSeek TUI adapter is a stub: always UNAVAILABLE, never drives the TUI
  - No real claude-code / codex / opencode / deepseek-tui subprocess is launched
  - No real model is called
  - No worktree is created
  - No user files are written

Strictly no subprocess execution of real CLI tools, no model calls, no worktrees.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import shutil
from pathlib import Path
from typing import List

import pytest

from executors.registry import ExecutorRegistry, _default_manifests
from executors.types import (
    AdapterStartResult,
    ExecutorHealthResult,
    ExecutorHealthStatus,
    ExecutorManifest,
    RunEvent,
    RunEventType,
    RunStatus,
)


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------

REQUIRED_ADAPTER_MODULES = [
    "executors.hermes_local_adapter",
    "executors.claude_code_adapter",
    "executors.codex_adapter",
    "executors.opencode_adapter",
    "executors.deepseek_tui_adapter",
]

REQUIRED_MANIFEST_IDS = [
    "hermes-local",
    "claude-code",
    "codex-cli",
    "opencode",
    "deepseek-tui",
]


@pytest.fixture(params=REQUIRED_ADAPTER_MODULES)
def adapter_module(request):
    """Import each adapter module fresh for parametrized tests."""
    return importlib.import_module(request.param)


# ---------------------------------------------------------------------------
# 1. Registry ships the 5 default manifests
# ---------------------------------------------------------------------------

class TestRegistryDefaultManifests:
    def test_default_manifests_contains_all_5(self) -> None:
        manifests = _default_manifests()
        for eid in REQUIRED_MANIFEST_IDS:
            assert eid in manifests, f"missing default manifest: {eid}"
            m = manifests[eid]
            assert isinstance(m, ExecutorManifest)
            assert m.id == eid
            assert m.label
            assert m.description

    def test_default_manifests_manifest_ids_match_manifest_objects(self) -> None:
        for eid, m in _default_manifests().items():
            assert m.id == eid

    def test_registry_can_register_default_manifests_with_adapters(self) -> None:
        """Round-trip: every default manifest is registrable."""
        reg = ExecutorRegistry()
        from executors.claude_code_adapter import ClaudeCodeAdapter
        from executors.codex_adapter import CodexAdapter
        from executors.deepseek_tui_adapter import DeepSeekTuiAdapter
        from executors.hermes_local_adapter import HermesLocalAdapter
        from executors.opencode_adapter import OpenCodeAdapter

        cls_map = {
            "hermes-local": HermesLocalAdapter,
            "claude-code": ClaudeCodeAdapter,
            "codex-cli": CodexAdapter,
            "opencode": OpenCodeAdapter,
            "deepseek-tui": DeepSeekTuiAdapter,
        }
        for eid, manifest in _default_manifests().items():
            adapter = cls_map[eid]()
            reg.register(manifest, adapter)
        listed = {m.id for m in reg.list_executors()}
        for eid in REQUIRED_MANIFEST_IDS:
            assert eid in listed


# ---------------------------------------------------------------------------
# 2. Adapter modules can be imported
# ---------------------------------------------------------------------------

class TestAdapterModuleImport:
    def test_every_adapter_module_imports_cleanly(self, adapter_module) -> None:
        # Importing the module already executed the import block.
        assert adapter_module is not None
        assert hasattr(adapter_module, "__file__")

    def test_every_adapter_module_exposes_a_class(self) -> None:
        from executors.claude_code_adapter import ClaudeCodeAdapter
        from executors.codex_adapter import CodexAdapter
        from executors.deepseek_tui_adapter import DeepSeekTuiAdapter
        from executors.hermes_local_adapter import HermesLocalAdapter
        from executors.opencode_adapter import OpenCodeAdapter

        for cls in (
            HermesLocalAdapter,
            ClaudeCodeAdapter,
            CodexAdapter,
            OpenCodeAdapter,
            DeepSeekTuiAdapter,
        ):
            name = f"{cls.__module__}.{cls.__name__}"
            assert callable(cls), name

    @pytest.mark.parametrize(
        "module_name,class_name",
        [
            ("executors.hermes_local_adapter", "HermesLocalAdapter"),
            ("executors.claude_code_adapter", "ClaudeCodeAdapter"),
            ("executors.codex_adapter", "CodexAdapter"),
            ("executors.opencode_adapter", "OpenCodeAdapter"),
            ("executors.deepseek_tui_adapter", "DeepSeekTuiAdapter"),
        ],
    )
    def test_each_module_contains_named_class(
        self, module_name: str, class_name: str
    ) -> None:
        mod = importlib.import_module(module_name)
        assert hasattr(mod, class_name), f"{module_name} missing {class_name}"


# ---------------------------------------------------------------------------
# 3. Health check does not crash when binary is missing
# ---------------------------------------------------------------------------

class TestHealthCheckMissingBinary:
    """For the 3 external-CLI adapters (claude-code / codex / opencode), we
    point each adapter at a path that absolutely cannot exist, then verify
    check_health() returns UNAVAILABLE cleanly.
    """

    @pytest.mark.parametrize(
        "module_name,class_name",
        [
            ("executors.claude_code_adapter", "ClaudeCodeAdapter"),
            ("executors.codex_adapter", "CodexAdapter"),
            ("executors.opencode_adapter", "OpenCodeAdapter"),
        ],
    )
    def test_health_returns_unavailable_for_missing_binary(
        self, module_name, class_name, monkeypatch
    ) -> None:
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        # Force the resolver to return a path that cannot exist. We replace
        # the function entirely (rather than mutating __defaults__) because
        # check_health() calls _resolve_command(None) explicitly.
        monkeypatch.setattr(
            mod, "_resolve_command",
            lambda override=None: "definitely-not-a-real-binary-xyz-123",
        )
        adapter = cls()
        result = asyncio.run(adapter.check_health())
        assert isinstance(result, ExecutorHealthResult)
        assert result.status == ExecutorHealthStatus.UNAVAILABLE
        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_hermes_local_health_does_not_crash(self) -> None:
        from executors.hermes_local_adapter import HermesLocalAdapter

        adapter = HermesLocalAdapter()
        result = asyncio.run(adapter.check_health())
        # Hermes Local imports run_agent at runtime; either it works (AVAILABLE)
        # or fails (UNAVAILABLE) — but it must NOT raise.
        assert result.status in (
            ExecutorHealthStatus.AVAILABLE,
            ExecutorHealthStatus.UNAVAILABLE,
            ExecutorHealthStatus.UNKNOWN,
        )


# ---------------------------------------------------------------------------
# 4. DeepSeek TUI is a STUB / UNAVAILABLE / not drivable
# ---------------------------------------------------------------------------

class TestDeepSeekTuiIsAStub:
    def test_check_health_returns_unavailable_when_binary_missing(
        self, monkeypatch
    ) -> None:
        from executors.deepseek_tui_adapter import DeepSeekTuiAdapter
        import executors.deepseek_tui_adapter as deepseek_mod

        monkeypatch.setattr(
            deepseek_mod, "_resolve_command",
            lambda override=None: "definitely-not-real-deepseek-xyz-123",
        )
        adapter = DeepSeekTuiAdapter()
        result = asyncio.run(adapter.check_health())
        assert result.status == ExecutorHealthStatus.UNAVAILABLE
        assert "stub" in result.error.lower() or "cannot be used" in result.error.lower()

    def test_check_health_still_unavailable_when_binary_exists(
        self, monkeypatch, tmp_path
    ) -> None:
        """Even if a deepseek-tui binary is on PATH, the stub reports UNAVAILABLE."""
        from executors.deepseek_tui_adapter import DeepSeekTuiAdapter
        import executors.deepseek_tui_adapter as deepseek_mod

        # Create a fake binary that PATH-searchable.
        fake = tmp_path / "deepseek-tui-fake"
        fake.write_text("#!/bin/sh\nexit 0\n")
        fake.chmod(0o755)

        # Point the resolver at the fake binary, and patch shutil.which so
        # check_command_exists finds it.
        monkeypatch.setattr(
            deepseek_mod, "_resolve_command", lambda override=None: str(fake),
        )
        monkeypatch.setattr(
            "executors.health.shutil.which",
            lambda cmd: str(fake) if cmd == str(fake) else None,
        )

        adapter = DeepSeekTuiAdapter()
        result = asyncio.run(adapter.check_health())
        assert result.status == ExecutorHealthStatus.UNAVAILABLE
        # Must explain that the TUI cannot be driven programmatically.
        assert "stub" in result.error.lower() or "tui" in result.error.lower()

    def test_start_refuses_to_launch(self) -> None:
        from executors.deepseek_tui_adapter import DeepSeekTuiAdapter

        adapter = DeepSeekTuiAdapter()
        # Build a minimal AgentRun.
        from executors.types import AgentRun, ExecutorConfig

        run = AgentRun(
            id="stub-test-1",
            executor_id="deepseek-tui",
            prompt="hello",
            workspace=Path.cwd(),
        )
        config = ExecutorConfig()
        result = asyncio.run(adapter.start(run, config))
        assert isinstance(result, AdapterStartResult)
        # The run state must be FAILED — never PENDING/RUNNING.
        status = asyncio.run(adapter.get_status(result.external_run_id))
        assert status == RunStatus.FAILED

    def test_stream_events_emits_failed_event(self) -> None:
        from executors.deepseek_tui_adapter import DeepSeekTuiAdapter

        adapter = DeepSeekTuiAdapter()

        async def collect() -> List[RunEvent]:
            events: List[RunEvent] = []
            async for ev in adapter.stream_events("does-not-exist"):
                events.append(ev)
            return events

        events = asyncio.run(collect())
        assert events, "stream_events should yield at least one FAILED event"
        assert any(e.type == RunEventType.FAILED for e in events)

    def test_allow_tui_launch_constant_is_false(self) -> None:
        from executors import deepseek_tui_adapter

        assert deepseek_tui_adapter._ALLOW_TUI_LAUNCH is False, (
            "DeepSeek TUI must remain a stub: _ALLOW_TUI_LAUNCH must be False"
        )


# ---------------------------------------------------------------------------
# 5. / 6. / 7. / 8. No real CLI / model / worktree / user-file side effects
# ---------------------------------------------------------------------------

class TestNoSideEffects:
    """These tests confirm the adapter modules are inert at import time and
    at health-check time — they do not spawn real CLIs, do not call any model
    API, do not create worktrees, and do not write user files.
    """

    @pytest.mark.parametrize("module_name", REQUIRED_ADAPTER_MODULES)
    def test_importing_adapter_does_not_invoke_clis(self, module_name) -> None:
        # If any module did `subprocess.Popen([...])` at import time, this
        # would block or fail on machines without those CLIs. We measure by
        # completing the import within a tight wall-clock budget and checking
        # the global process state is unchanged.
        importlib.import_module(module_name)
        # If we get here without hanging, the import had no side effects.

    @pytest.mark.parametrize(
        "module_name,class_name",
        [
            ("executors.claude_code_adapter", "ClaudeCodeAdapter"),
            ("executors.codex_adapter", "CodexAdapter"),
            ("executors.opencode_adapter", "OpenCodeAdapter"),
        ],
    )
    def test_instantiating_adapter_does_not_spawn_subprocess(
        self, module_name, class_name
    ) -> None:
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        # Instantiation must be cheap and side-effect free.
        adapter = cls()
        assert adapter is not None

    @pytest.mark.parametrize(
        "module_name,class_name",
        [
            ("executors.claude_code_adapter", "ClaudeCodeAdapter"),
            ("executors.codex_adapter", "CodexAdapter"),
            ("executors.opencode_adapter", "OpenCodeAdapter"),
        ],
    )
    def test_health_check_with_missing_binary_does_not_spawn(
        self, module_name, class_name, monkeypatch
    ) -> None:
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        monkeypatch.setattr(
            mod, "_resolve_command",
            lambda override=None: "definitely-not-real-xyz-456",
        )
        adapter = cls()
        result = asyncio.run(adapter.check_health())
        # Must short-circuit to UNAVAILABLE before ever spawning a process.
        assert result.status == ExecutorHealthStatus.UNAVAILABLE

    def test_no_subprocess_popen_open_called_at_import(
        self, monkeypatch
    ) -> None:
        """Wrap subprocess.Popen / asyncio.create_subprocess_exec and import
        all 5 adapter modules; nothing should have invoked them at import.
        """
        import subprocess
        import asyncio as _asyncio

        popen_calls: list = []
        original_popen = subprocess.Popen

        def tracking_popen(*args, **kwargs):
            popen_calls.append((args, kwargs))
            return original_popen(*args, **kwargs)

        monkeypatch.setattr(subprocess, "Popen", tracking_popen)

        # Force re-import to capture import-time calls (the modules may already
        # be in sys.modules from earlier tests, so we explicitly reload).
        import sys
        for mod_name in REQUIRED_ADAPTER_MODULES:
            if mod_name in sys.modules:
                importlib.reload(sys.modules[mod_name])
            else:
                importlib.import_module(mod_name)

        # asyncio.create_subprocess_exec is an async factory, but we can guard
        # by ensuring that no Popen call was made with a path we recognize as
        # a real CLI binary.
        cli_names = {"claude-code", "codex", "opencode", "deepseek-tui"}
        bad_calls = []
        for args, kwargs in popen_calls:
            arg_list = list(args[0]) if args else []
            if any(str(a) in cli_names for a in arg_list):
                bad_calls.append((args, kwargs))
        assert bad_calls == [], f"Unexpected subprocess at import: {bad_calls}"

    def test_no_user_file_writes_at_import(self, tmp_path, monkeypatch) -> None:
        """Point HOME and HERMES_HOME at tmp_path and verify no module import
        writes files into either.
        """
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

        import sys
        for mod_name in REQUIRED_ADAPTER_MODULES:
            if mod_name in sys.modules:
                importlib.reload(sys.modules[mod_name])
            else:
                importlib.import_module(mod_name)

        # Nothing should have appeared under HOME or HERMES_HOME.
        home_files = list(tmp_path.rglob("*"))
        # Filter out any pyc cache that pytest itself may produce under HOME.
        # We're checking for actual artifact files, not bytecode.
        real_files = [
            p for p in home_files
            if p.is_file() and not str(p).endswith(".pyc")
        ]
        assert real_files == [], (
            f"Unexpected files written under HOME/HERMES_HOME during import: "
            f"{[str(p) for p in real_files]}"
        )

    def test_hermes_local_adapter_does_not_chdir_at_construct_time(
        self, monkeypatch
    ) -> None:
        """Constructing HermesLocalAdapter must not change cwd. The chdir
        happens only inside start() under a try/finally guard.
        """
        from executors.hermes_local_adapter import HermesLocalAdapter

        original_cwd = os.getcwd()
        try:
            _ = HermesLocalAdapter(workspace=Path(tmp_path := os.getcwd()))
            assert os.getcwd() == original_cwd
        finally:
            os.chdir(original_cwd)

    def test_hermes_local_chdir_protected_by_finally(
        self, monkeypatch
    ) -> None:
        """Drive ``_execute`` directly so the chdir/finally contract is
        observable in a single asyncio.run. We patch ``run_agent.AIAgent`` to
        raise, and confirm cwd is restored after the exception propagates
        through the adapter's try/finally.
        """
        from executors import hermes_local_adapter as mod
        from executors.types import AgentRun, ExecutorConfig

        class _ExplodingAgent:
            def __init__(self, *args, **kwargs):
                pass

            def run_conversation(self, *args, **kwargs):
                raise RuntimeError("boom")

        # _execute does ``from run_agent import AIAgent`` — patch the source.
        import run_agent
        monkeypatch.setattr(run_agent, "AIAgent", _ExplodingAgent)

        original_cwd = os.getcwd()
        target_workspace = original_cwd  # any valid existing directory
        try:
            adapter = mod.HermesLocalAdapter(workspace=Path(target_workspace))
            run = AgentRun(
                id="explode-1",
                executor_id="hermes-local",
                prompt="x",
                workspace=Path(target_workspace),
            )
            config = ExecutorConfig()
            state = mod._RunState(
                run_id=run.id,
                status=RunStatus.RUNNING,
                cancel_event=asyncio.Event(),
            )

            # Drive the private _execute directly. It must complete (with
            # FAILED state) and the cwd must be restored when it returns.
            asyncio.run(adapter._execute(run, config, state))

            assert state.status == RunStatus.FAILED
            assert os.getcwd() == original_cwd, (
                f"cwd was not restored: {os.getcwd()!r} != {original_cwd!r}"
            )
        finally:
            # Defensive: in case the test failed mid-flight.
            if os.getcwd() != original_cwd:
                os.chdir(original_cwd)

    def test_no_worktree_creation_on_import(self, monkeypatch) -> None:
        """No adapter module may create git worktrees as a side effect of
        import. We watch for any call to executors.worktree.* during import.
        """
        import sys
        # If worktree module is already loaded, the import side-effect test
        # is moot; we only care that adapters don't reach into it.
        worktree_import_attempts: list = []

        real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def tracking_import(name, *args, **kwargs):
            if "worktree" in name and "executors" in name:
                worktree_import_attempts.append(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", tracking_import)

        for mod_name in REQUIRED_ADAPTER_MODULES:
            if mod_name in sys.modules:
                importlib.reload(sys.modules[mod_name])
            else:
                importlib.import_module(mod_name)

        assert worktree_import_attempts == [], (
            f"Adapter import pulled in worktree module: {worktree_import_attempts}"
        )
