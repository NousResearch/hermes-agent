#!/usr/bin/env python3
"""
Comprehensive Modal Integration Test Suite

Tests both:
1. terminal_tool.py Modal backend (CLI/agent use case)
2. atropos/backends/modal_backend.py (RL training use case)

Run with:
    # All tests (requires Modal account)
    python tests/test_modal_integration.py

    # Dry run (no Modal, tests config/logic only)
    python tests/test_modal_integration.py --dry-run

    # Specific test category
    python tests/test_modal_integration.py --category terminal
    python tests/test_modal_integration.py --category atropos
    python tests/test_modal_integration.py --category profiles
"""

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Atropos Import Helper
# =============================================================================

def try_import_atropos_backend():
    """
    Try to import atropos backend directly, bypassing the atroposlib check.
    Returns (ModalToolBackend, ModalSandboxConfig, Slot, SlotState) or raises ImportError.
    """
    try:
        # Try direct import first (works if atroposlib is installed)
        from atropos.backends.modal_backend import ModalToolBackend, ModalSandboxConfig
        from atropos.slots.slot import Slot, SlotState
        return ModalToolBackend, ModalSandboxConfig, Slot, SlotState
    except (ImportError, ModuleNotFoundError):
        # Try importing the module directly without going through atropos/__init__.py
        import importlib.util
        
        backend_path = Path(__file__).parent.parent / "atropos" / "backends" / "modal_backend.py"
        slot_path = Path(__file__).parent.parent / "atropos" / "slots" / "slot.py"
        executor_path = Path(__file__).parent.parent / "atropos" / "slots" / "executor.py"
        base_path = Path(__file__).parent.parent / "atropos" / "backends" / "base.py"
        
        if not backend_path.exists():
            raise ImportError(f"modal_backend.py not found at {backend_path}")
        
        # Load slot module first
        spec = importlib.util.spec_from_file_location("atropos_slots_slot", slot_path)
        slot_module = importlib.util.module_from_spec(spec)
        sys.modules["atropos.slots.slot"] = slot_module
        spec.loader.exec_module(slot_module)
        
        # Load executor module
        spec = importlib.util.spec_from_file_location("atropos_slots_executor", executor_path)
        executor_module = importlib.util.module_from_spec(spec)
        sys.modules["atropos.slots.executor"] = executor_module
        spec.loader.exec_module(executor_module)
        
        # Load base module
        spec = importlib.util.spec_from_file_location("atropos_backends_base", base_path)
        base_module = importlib.util.module_from_spec(spec)
        sys.modules["atropos.backends.base"] = base_module
        spec.loader.exec_module(base_module)
        
        # Now load modal_backend
        spec = importlib.util.spec_from_file_location("atropos_backends_modal_backend", backend_path)
        backend_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(backend_module)
        
        return (
            backend_module.ModalToolBackend,
            backend_module.ModalSandboxConfig,
            slot_module.Slot,
            slot_module.SlotState,
        )


# =============================================================================
# Test Configuration
# =============================================================================

@dataclass
class TestConfig:
    dry_run: bool = False
    verbose: bool = True
    category: Optional[str] = None  # None = all, or "terminal", "atropos", "profiles"


# =============================================================================
# Test Results Tracking
# =============================================================================

class TestResults:
    def __init__(self):
        self.passed: List[str] = []
        self.failed: List[tuple] = []  # (name, error)
        self.skipped: List[tuple] = []  # (name, reason)
    
    def record_pass(self, name: str):
        self.passed.append(name)
        print(f"  ✅ {name}")
    
    def record_fail(self, name: str, error: str):
        self.failed.append((name, error))
        print(f"  ❌ {name}: {error}")
    
    def record_skip(self, name: str, reason: str):
        self.skipped.append((name, reason))
        print(f"  ⏭️  {name}: {reason}")
    
    def summary(self):
        total = len(self.passed) + len(self.failed) + len(self.skipped)
        print(f"\n{'='*60}")
        print(f"TEST RESULTS: {len(self.passed)}/{total} passed")
        print(f"  Passed:  {len(self.passed)}")
        print(f"  Failed:  {len(self.failed)}")
        print(f"  Skipped: {len(self.skipped)}")
        
        if self.failed:
            print(f"\nFailed tests:")
            for name, error in self.failed:
                print(f"  - {name}: {error}")
        
        return len(self.failed) == 0


results = TestResults()


# =============================================================================
# CATEGORY 1: Profile Configuration Tests
# =============================================================================

def test_profile_loading_from_env():
    """Test ModalProfile.from_env() loads environment variables correctly."""
    from tools.terminal_tool import ModalProfile
    
    # Set test environment variables
    # Note: The prefix is TERMINAL_MODAL_PROFILE_{profile_name}_ where profile_name is used as-is
    os.environ["TERMINAL_MODAL_PROFILE_testenv_IMAGE"] = "python:3.12"
    os.environ["TERMINAL_MODAL_PROFILE_testenv_GPU"] = "A100"
    os.environ["TERMINAL_MODAL_PROFILE_testenv_CPU"] = "4.0"
    os.environ["TERMINAL_MODAL_PROFILE_testenv_MEMORY"] = "32768"
    os.environ["TERMINAL_MODAL_PROFILE_testenv_SECRETS"] = "secret1,secret2"
    os.environ["TERMINAL_MODAL_PROFILE_testenv_ENV_VARS"] = "KEY1=val1;KEY2=val2"
    
    try:
        profile = ModalProfile.from_env("testenv")
        
        assert profile.name == "testenv", f"Expected name 'testenv', got '{profile.name}'"
        assert profile.image == "python:3.12", f"Expected image 'python:3.12', got '{profile.image}'"
        assert profile.gpu == "A100", f"Expected GPU 'A100', got '{profile.gpu}'"
        assert profile.cpu == 4.0, f"Expected CPU 4.0, got {profile.cpu}"
        assert profile.memory == 32768, f"Expected memory 32768, got {profile.memory}"
        assert profile.secrets == ["secret1", "secret2"], f"Secrets mismatch: {profile.secrets}"
        assert profile.env_vars == {"KEY1": "val1", "KEY2": "val2"}, f"Env vars mismatch: {profile.env_vars}"
        
        results.record_pass("test_profile_loading_from_env")
    except Exception as e:
        results.record_fail("test_profile_loading_from_env", str(e))
    finally:
        # Cleanup
        for key in list(os.environ.keys()):
            if key.startswith("TERMINAL_MODAL_PROFILE_testenv_"):
                del os.environ[key]


def test_profile_loading_from_yaml():
    """Test ModalProfile.load_profiles() from YAML file."""
    from tools.terminal_tool import ModalProfile, YAML_AVAILABLE
    
    if not YAML_AVAILABLE:
        results.record_skip("test_profile_loading_from_yaml", "PyYAML not installed")
        return
    
    yaml_content = """
profiles:
  test-yaml:
    image: pytorch/pytorch:2.0
    gpu: T4
    cpu: 2.0
    memory: 8192
    min_pool: 1
    max_pool: 3
    secrets:
      - hf-token
    env_vars:
      CUDA_VISIBLE_DEVICES: "0"
  test-yaml-2:
    image: node:20
    cpu: 1.0
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name
    
    try:
        profiles = ModalProfile.load_profiles(yaml_path)
        
        assert "test-yaml" in profiles, f"Profile 'test-yaml' not found in {list(profiles.keys())}"
        assert "test-yaml-2" in profiles, f"Profile 'test-yaml-2' not found"
        
        p1 = profiles["test-yaml"]
        assert p1.image == "pytorch/pytorch:2.0"
        assert p1.gpu == "T4"
        assert p1.cpu == 2.0
        assert p1.memory == 8192
        assert p1.secrets == ["hf-token"]
        assert p1.env_vars == {"CUDA_VISIBLE_DEVICES": "0"}
        
        results.record_pass("test_profile_loading_from_yaml")
    except Exception as e:
        results.record_fail("test_profile_loading_from_yaml", str(e))
    finally:
        os.unlink(yaml_path)


def test_profile_defaults():
    """Test ModalProfile uses correct defaults."""
    from tools.terminal_tool import ModalProfile
    
    try:
        profile = ModalProfile(name="minimal")
        
        assert profile.image == "python:3.11"
        assert profile.gpu is None
        assert profile.cpu == 1.0
        assert profile.memory == 2048
        assert profile.min_pool == 1
        assert profile.max_pool == 5
        assert profile.idle_timeout == 120
        assert profile.secrets == []
        assert profile.env_vars == {}
        
        results.record_pass("test_profile_defaults")
    except Exception as e:
        results.record_fail("test_profile_defaults", str(e))


def test_atropos_config_with_app_name():
    """Test ModalSandboxConfig.with_app_name() method."""
    try:
        # Try direct import first
        try:
            from atropos.backends.modal_backend import ModalSandboxConfig
        except (ImportError, ModuleNotFoundError):
            # Try importing module directly without atropos/__init__.py
            ModalToolBackend, ModalSandboxConfig, _, _ = try_import_atropos_backend()
        
        config = ModalSandboxConfig(
            name="test-convert",
            image="python:3.10",
            gpu="A10G",
            cpu=2.0,
            memory=4096,
            secrets=["secret1"],
            env_vars={"FOO": "bar"},
        )
        
        config_with_app = config.with_app_name("my-app")
        
        assert config_with_app.app_name == "my-app-test-convert"
        assert config_with_app.image == "python:3.10"
        assert config_with_app.gpu == "A10G"
        assert config_with_app.cpu == 2.0
        assert config_with_app.memory == 4096
        assert config_with_app.secrets == ["secret1"]
        assert config_with_app.env_vars == {"FOO": "bar"}
        
        results.record_pass("test_atropos_config_with_app_name")
    except ImportError as e:
        results.record_skip("test_atropos_config_with_app_name", f"Requires atroposlib: pip install -e '.[atropos]'")
    except Exception as e:
        results.record_fail("test_atropos_config_with_app_name", str(e))


# =============================================================================
# CATEGORY 2: Terminal Tool Modal Tests
# =============================================================================

def test_terminal_modal_pool_manager_singleton():
    """Test _ModalPoolManager is a proper singleton."""
    from tools.terminal_tool import _ModalPoolManager
    
    try:
        # Reset singleton for test
        _ModalPoolManager._instance = None
        
        manager1 = _ModalPoolManager.get_instance()
        manager2 = _ModalPoolManager.get_instance()
        
        assert manager1 is manager2, "Pool manager should be singleton"
        
        results.record_pass("test_terminal_modal_pool_manager_singleton")
    except Exception as e:
        results.record_fail("test_terminal_modal_pool_manager_singleton", str(e))


def test_terminal_create_environment_modal():
    """Test _create_environment creates Modal environment correctly."""
    from tools.terminal_tool import _create_environment
    
    try:
        env = _create_environment(
            env_type="modal",
            image="python:3.11",
            cwd="/workspace",
            timeout=60,
            task_id="test-task-123",
            profile="default",
        )
        
        # Check it's the right type
        assert env.__class__.__name__ == "_ModalSandboxEnvironment"
        assert env.profile == "default"
        assert env.task_id == "test-task-123"
        
        results.record_pass("test_terminal_create_environment_modal")
    except Exception as e:
        results.record_fail("test_terminal_create_environment_modal", str(e))


def test_terminal_tool_profile_parameter(config: TestConfig):
    """Test terminal_tool() accepts profile parameter."""
    if config.dry_run:
        results.record_skip("test_terminal_tool_profile_parameter", "Dry run mode")
        return
    
    from tools.terminal_tool import terminal_tool, cleanup_vm
    
    # Save original env
    original_env = os.environ.get("TERMINAL_ENV")
    
    try:
        os.environ["TERMINAL_ENV"] = "modal"
        task_id = f"test-profile-param-{int(time.time())}"
        
        # This should work without error (profile passed through)
        result = terminal_tool(
            "echo 'Hello from Modal'",
            task_id=task_id,
            profile="default",
        )
        
        result_data = json.loads(result)
        # terminal_tool returns {"output", "exit_code", "error"} not {"success"}
        assert result_data.get("exit_code") == 0, f"Command failed: {result_data}"
        assert "Hello from Modal" in result_data.get("output", "")
        
        cleanup_vm(task_id)
        results.record_pass("test_terminal_tool_profile_parameter")
    except Exception as e:
        results.record_fail("test_terminal_tool_profile_parameter", str(e))
    finally:
        if original_env:
            os.environ["TERMINAL_ENV"] = original_env
        elif "TERMINAL_ENV" in os.environ:
            del os.environ["TERMINAL_ENV"]


def test_terminal_modal_execute_simple(config: TestConfig):
    """Test basic command execution in Modal sandbox."""
    if config.dry_run:
        results.record_skip("test_terminal_modal_execute_simple", "Dry run mode")
        return
    
    from tools.terminal_tool import terminal_tool, cleanup_vm
    
    original_env = os.environ.get("TERMINAL_ENV")
    
    try:
        os.environ["TERMINAL_ENV"] = "modal"
        task_id = f"test-simple-{int(time.time())}"
        
        # Test echo
        result = json.loads(terminal_tool("echo 'test123'", task_id=task_id))
        assert result["exit_code"] == 0, f"Echo failed: {result}"
        assert "test123" in result["output"]
        
        # Test pwd
        result = json.loads(terminal_tool("pwd", task_id=task_id))
        assert result["exit_code"] == 0, f"pwd failed: {result}"
        
        # Test file creation and reading
        result = json.loads(terminal_tool("echo 'content' > test.txt && cat test.txt", task_id=task_id))
        assert result["exit_code"] == 0, f"File ops failed: {result}"
        assert "content" in result["output"]
        
        cleanup_vm(task_id)
        results.record_pass("test_terminal_modal_execute_simple")
    except Exception as e:
        results.record_fail("test_terminal_modal_execute_simple", str(e))
    finally:
        if original_env:
            os.environ["TERMINAL_ENV"] = original_env
        elif "TERMINAL_ENV" in os.environ:
            del os.environ["TERMINAL_ENV"]


def test_terminal_modal_persistence(config: TestConfig):
    """Test state persists within same task_id."""
    if config.dry_run:
        results.record_skip("test_terminal_modal_persistence", "Dry run mode")
        return
    
    from tools.terminal_tool import terminal_tool, cleanup_vm
    
    original_env = os.environ.get("TERMINAL_ENV")
    
    try:
        os.environ["TERMINAL_ENV"] = "modal"
        task_id = f"test-persist-{int(time.time())}"
        
        # Create a file
        result1 = json.loads(terminal_tool("echo 'persistent data' > /workspace/persist.txt", task_id=task_id))
        assert result1["exit_code"] == 0, f"Create file failed: {result1}"
        
        # Read it in separate call (same task_id)
        result2 = json.loads(terminal_tool("cat /workspace/persist.txt", task_id=task_id))
        assert result2["exit_code"] == 0, f"Read file failed: {result2}"
        assert "persistent data" in result2["output"]
        
        cleanup_vm(task_id)
        results.record_pass("test_terminal_modal_persistence")
    except Exception as e:
        results.record_fail("test_terminal_modal_persistence", str(e))
    finally:
        if original_env:
            os.environ["TERMINAL_ENV"] = original_env
        elif "TERMINAL_ENV" in os.environ:
            del os.environ["TERMINAL_ENV"]


def test_terminal_modal_isolation(config: TestConfig):
    """Test different task_ids are isolated."""
    if config.dry_run:
        results.record_skip("test_terminal_modal_isolation", "Dry run mode")
        return
    
    from tools.terminal_tool import terminal_tool, cleanup_vm
    
    original_env = os.environ.get("TERMINAL_ENV")
    
    try:
        os.environ["TERMINAL_ENV"] = "modal"
        task_id_1 = f"test-iso-1-{int(time.time())}"
        task_id_2 = f"test-iso-2-{int(time.time())}"
        
        # Create file in task 1
        result1 = json.loads(terminal_tool("echo 'task1' > /workspace/iso.txt", task_id=task_id_1))
        assert result1["exit_code"] == 0, f"Task 1 create failed: {result1}"
        
        # Create different file in task 2
        result2 = json.loads(terminal_tool("echo 'task2' > /workspace/iso.txt", task_id=task_id_2))
        assert result2["exit_code"] == 0, f"Task 2 create failed: {result2}"
        
        # Verify task 1 still has its own content
        result3 = json.loads(terminal_tool("cat /workspace/iso.txt", task_id=task_id_1))
        assert result3["exit_code"] == 0, f"Task 1 read failed: {result3}"
        assert "task1" in result3["output"], f"Task 1 content corrupted: {result3['output']}"
        
        # Verify task 2 has its content
        result4 = json.loads(terminal_tool("cat /workspace/iso.txt", task_id=task_id_2))
        assert result4["exit_code"] == 0, f"Task 2 read failed: {result4}"
        assert "task2" in result4["output"], f"Task 2 content corrupted: {result4['output']}"
        
        cleanup_vm(task_id_1)
        cleanup_vm(task_id_2)
        results.record_pass("test_terminal_modal_isolation")
    except Exception as e:
        results.record_fail("test_terminal_modal_isolation", str(e))
    finally:
        if original_env:
            os.environ["TERMINAL_ENV"] = original_env
        elif "TERMINAL_ENV" in os.environ:
            del os.environ["TERMINAL_ENV"]


# =============================================================================
# CATEGORY 3: Atropos Modal Backend Tests
# =============================================================================

async def test_atropos_backend_lifecycle(config: TestConfig):
    """Test ModalToolBackend start/stop lifecycle."""
    if config.dry_run:
        results.record_skip("test_atropos_backend_lifecycle", "Dry run mode")
        return
    
    try:
        try:
            from atropos.backends.modal_backend import ModalToolBackend, ModalSandboxConfig
        except (ImportError, ModuleNotFoundError):
            ModalToolBackend, ModalSandboxConfig, _, _, _ = try_import_atropos_backend()
        
        config_obj = ModalSandboxConfig(
            app_name="test-lifecycle",
            min_sandboxes=1,
            max_sandboxes=2,
            slots_per_sandbox=3,
        )
        
        backend = ModalToolBackend(config_obj)
        
        # Start
        await backend.start()
        
        status = backend.get_status()
        assert status["sandboxes"] >= 1, f"Expected at least 1 sandbox, got {status}"
        assert status["slots_per_sandbox"] == 3
        
        # Stop
        await backend.stop(purge=True)
        
        results.record_pass("test_atropos_backend_lifecycle")
    except ImportError as e:
        results.record_skip("test_atropos_backend_lifecycle", f"Requires atroposlib: pip install -e '.[atropos]'")
    except Exception as e:
        results.record_fail("test_atropos_backend_lifecycle", str(e))


async def test_atropos_slot_acquire_release(config: TestConfig):
    """Test slot acquisition and release."""
    if config.dry_run:
        results.record_skip("test_atropos_slot_acquire_release", "Dry run mode")
        return
    
    try:
        try:
            from atropos.backends.modal_backend import ModalToolBackend, ModalSandboxConfig
        except (ImportError, ModuleNotFoundError):
            ModalToolBackend, ModalSandboxConfig, _, _, _ = try_import_atropos_backend()
        
        config_obj = ModalSandboxConfig(
            app_name="test-slots",
            min_sandboxes=1,
            max_sandboxes=2,
            slots_per_sandbox=5,
        )
        
        backend = ModalToolBackend(config_obj)
        await backend.start()
        
        try:
            # Acquire slot
            slot = await backend.acquire("trajectory-1")
            
            assert slot is not None
            assert slot.trajectory_id == "trajectory-1"
            assert "/data/" in slot.workspace_dir
            
            # Check status shows slot in use
            status = backend.get_status()
            assert status["available_slots"] < status["total_slots"]
            
            # Release slot
            await backend.release(slot)
            
            # Check slot is available again
            status = backend.get_status()
            # Note: might need small delay for status update
            
            results.record_pass("test_atropos_slot_acquire_release")
        finally:
            await backend.stop(purge=True)
    except ImportError as e:
        results.record_skip("test_atropos_slot_acquire_release", f"Requires atroposlib: pip install -e '.[atropos]'")
    except Exception as e:
        results.record_fail("test_atropos_slot_acquire_release", str(e))


async def test_atropos_execute_in_slot(config: TestConfig):
    """Test command execution in acquired slot."""
    if config.dry_run:
        results.record_skip("test_atropos_execute_in_slot", "Dry run mode")
        return
    
    try:
        try:
            from atropos.backends.modal_backend import ModalToolBackend, ModalSandboxConfig
        except (ImportError, ModuleNotFoundError):
            ModalToolBackend, ModalSandboxConfig, _, _, _ = try_import_atropos_backend()
        
        config_obj = ModalSandboxConfig(
            app_name="test-execute",
            min_sandboxes=1,
            max_sandboxes=1,
            slots_per_sandbox=3,
        )
        
        backend = ModalToolBackend(config_obj)
        await backend.start()
        
        try:
            slot = await backend.acquire("test-exec")
            
            # Execute bash command
            results_list = await backend.execute_batch([
                (slot, "bash", {"command": "echo 'hello world'"})
            ])
            
            assert len(results_list) == 1
            result = results_list[0]
            assert result.success, f"Command failed: {result.error}"
            assert "hello world" in result.output
            
            await backend.release(slot)
            results.record_pass("test_atropos_execute_in_slot")
        finally:
            await backend.stop(purge=True)
    except ImportError as e:
        results.record_skip("test_atropos_execute_in_slot", f"Requires atroposlib: pip install -e '.[atropos]'")
    except Exception as e:
        results.record_fail("test_atropos_execute_in_slot", str(e))


async def test_atropos_batched_execution(config: TestConfig):
    """Test batched parallel execution across multiple slots."""
    if config.dry_run:
        results.record_skip("test_atropos_batched_execution", "Dry run mode")
        return
    
    try:
        try:
            from atropos.backends.modal_backend import ModalToolBackend, ModalSandboxConfig
        except (ImportError, ModuleNotFoundError):
            ModalToolBackend, ModalSandboxConfig, _, _, _ = try_import_atropos_backend()
        
        config_obj = ModalSandboxConfig(
            app_name="test-batch",
            min_sandboxes=1,
            max_sandboxes=2,
            slots_per_sandbox=5,
        )
        
        backend = ModalToolBackend(config_obj)
        await backend.start()
        
        try:
            # Acquire multiple slots
            slots = []
            for i in range(3):
                slot = await backend.acquire(f"batch-{i}")
                slots.append(slot)
            
            # Execute batch of commands
            start_time = time.time()
            results_list = await backend.execute_batch([
                (slots[0], "bash", {"command": "sleep 1 && echo 'slot0'"}),
                (slots[1], "bash", {"command": "sleep 1 && echo 'slot1'"}),
                (slots[2], "bash", {"command": "sleep 1 && echo 'slot2'"}),
            ])
            elapsed = time.time() - start_time
            
            # All should succeed
            assert len(results_list) == 3
            for i, result in enumerate(results_list):
                assert result.success, f"Slot {i} failed: {result.error}"
                assert f"slot{i}" in result.output
            
            # Should be parallel - with Modal overhead, allow up to 5s for 3x 1-second sleeps
            # (If sequential, would take > 3s just for the sleeps)
            assert elapsed < 5.0, f"Batch execution took {elapsed}s, expected < 5.0s (parallel)"
            
            for slot in slots:
                await backend.release(slot)
            
            results.record_pass("test_atropos_batched_execution")
        finally:
            await backend.stop(purge=True)
    except ImportError as e:
        results.record_skip("test_atropos_batched_execution", f"Requires atroposlib: pip install -e '.[atropos]'")
    except Exception as e:
        results.record_fail("test_atropos_batched_execution", str(e))


async def test_atropos_slot_workspace_isolation(config: TestConfig):
    """Test workspace isolation between slots."""
    if config.dry_run:
        results.record_skip("test_atropos_slot_workspace_isolation", "Dry run mode")
        return
    
    try:
        try:
            from atropos.backends.modal_backend import ModalToolBackend, ModalSandboxConfig
        except (ImportError, ModuleNotFoundError):
            ModalToolBackend, ModalSandboxConfig, _, _, _ = try_import_atropos_backend()
        
        config_obj = ModalSandboxConfig(
            app_name="test-isolation",
            min_sandboxes=1,
            max_sandboxes=1,
            slots_per_sandbox=3,
        )
        
        backend = ModalToolBackend(config_obj)
        await backend.start()
        
        try:
            slot1 = await backend.acquire("iso-1")
            slot2 = await backend.acquire("iso-2")
            
            # Write different content to each slot
            await backend.execute_batch([
                (slot1, "bash", {"command": "echo 'content1' > test.txt"}),
                (slot2, "bash", {"command": "echo 'content2' > test.txt"}),
            ])
            
            # Read back and verify isolation
            results_list = await backend.execute_batch([
                (slot1, "bash", {"command": "cat test.txt"}),
                (slot2, "bash", {"command": "cat test.txt"}),
            ])
            
            assert "content1" in results_list[0].output, f"Slot 1 content wrong: {results_list[0].output}"
            assert "content2" in results_list[1].output, f"Slot 2 content wrong: {results_list[1].output}"
            
            await backend.release(slot1)
            await backend.release(slot2)
            
            results.record_pass("test_atropos_slot_workspace_isolation")
        finally:
            await backend.stop(purge=True)
    except ImportError as e:
        results.record_skip("test_atropos_slot_workspace_isolation", f"Requires atroposlib: pip install -e '.[atropos]'")
    except Exception as e:
        results.record_fail("test_atropos_slot_workspace_isolation", str(e))


async def test_atropos_workspace_reset(config: TestConfig):
    """Test workspace reset on slot release."""
    if config.dry_run:
        results.record_skip("test_atropos_workspace_reset", "Dry run mode")
        return
    
    try:
        try:
            from atropos.backends.modal_backend import ModalToolBackend, ModalSandboxConfig
        except (ImportError, ModuleNotFoundError):
            ModalToolBackend, ModalSandboxConfig, _, _, _ = try_import_atropos_backend()
        
        config_obj = ModalSandboxConfig(
            app_name="test-reset",
            min_sandboxes=1,
            max_sandboxes=1,
            slots_per_sandbox=2,
        )
        
        backend = ModalToolBackend(config_obj)
        await backend.start()
        
        try:
            # Acquire, create file, release with reset
            slot = await backend.acquire("reset-test")
            slot_id = slot.slot_id
            
            await backend.execute_batch([
                (slot, "bash", {"command": "echo 'should be deleted' > test.txt"}),
            ])
            
            await backend.release(slot, reset_workspace=True)
            
            # Re-acquire (might get same slot)
            slot2 = await backend.acquire("reset-test-2")
            
            # Check file doesn't exist (or we got different slot)
            result = await backend.execute_batch([
                (slot2, "bash", {"command": "cat test.txt 2>/dev/null || echo 'file not found'"}),
            ])
            
            # Either file not found OR different slot
            output = result[0].output
            if slot2.slot_id == slot_id:
                assert "file not found" in output or not result[0].success, f"File should be deleted: {output}"
            
            await backend.release(slot2)
            results.record_pass("test_atropos_workspace_reset")
        finally:
            await backend.stop(purge=True)
    except ImportError as e:
        results.record_skip("test_atropos_workspace_reset", f"Requires atroposlib: pip install -e '.[atropos]'")
    except Exception as e:
        results.record_fail("test_atropos_workspace_reset", str(e))


async def test_atropos_multi_profile(config: TestConfig):
    """Test multi-profile support with different resources."""
    if config.dry_run:
        results.record_skip("test_atropos_multi_profile", "Dry run mode")
        return
    
    try:
        try:
            from atropos.backends.modal_backend import ModalToolBackend, ModalSandboxConfig
        except (ImportError, ModuleNotFoundError):
            ModalToolBackend, ModalSandboxConfig, _, _ = try_import_atropos_backend()
        
        # Create backend with multiple profiles
        backend = ModalToolBackend.with_profiles(
            app_name="test-multiprofile",
            profiles={
                "default": ModalSandboxConfig(
                    name="default",
                    image="python:3.11",
                    cpu=1.0,
                    memory=2048,
                    min_sandboxes=1,
                    max_sandboxes=2,
                    slots_per_sandbox=3,
                ),
                "compute": ModalSandboxConfig(
                    name="compute",
                    image="python:3.11",
                    cpu=2.0,
                    memory=4096,
                    min_sandboxes=0,  # Start on demand
                    max_sandboxes=1,
                    slots_per_sandbox=2,
                ),
            },
            default_profile="default",
        )
        
        await backend.start(profiles_to_start=["default"])
        
        try:
            # List profiles
            profiles = backend.list_profiles()
            assert "default" in profiles
            assert "compute" in profiles
            assert profiles["default"]["active"] == True
            assert profiles["compute"]["active"] == False  # Not started yet
            
            # Acquire from default profile
            slot1 = await backend.acquire("traj-1", profile="default")
            assert slot1 is not None
            
            # Acquire from compute profile (should start it on demand)
            slot2 = await backend.acquire("traj-2", profile="compute")
            assert slot2 is not None
            
            # Execute on both
            results_list = await backend.execute_batch([
                (slot1, "bash", {"command": "python --version"}),
                (slot2, "bash", {"command": "python --version"}),
            ])
            
            assert results_list[0].success
            assert results_list[1].success
            
            await backend.release(slot1)
            await backend.release(slot2)
            
            # Check status shows both profiles
            status = backend.get_status()
            assert "default" in status["pools"]
            assert "compute" in status["pools"]
            
            results.record_pass("test_atropos_multi_profile")
        finally:
            await backend.stop(purge=True)
    except ImportError as e:
        results.record_skip("test_atropos_multi_profile", f"Requires atroposlib: pip install -e '.[atropos]'")
    except Exception as e:
        results.record_fail("test_atropos_multi_profile", str(e))


async def test_atropos_cross_profile_batch(config: TestConfig):
    """Test batched execution across different profiles."""
    if config.dry_run:
        results.record_skip("test_atropos_cross_profile_batch", "Dry run mode")
        return
    
    try:
        try:
            from atropos.backends.modal_backend import ModalToolBackend, ModalSandboxConfig
        except (ImportError, ModuleNotFoundError):
            ModalToolBackend, ModalSandboxConfig, _, _ = try_import_atropos_backend()
        
        backend = ModalToolBackend.with_profiles(
            app_name="test-crossprofile",
            profiles={
                "profile-a": ModalSandboxConfig(
                    name="profile-a",
                    min_sandboxes=1,
                    max_sandboxes=1,
                    slots_per_sandbox=2,
                ),
                "profile-b": ModalSandboxConfig(
                    name="profile-b",
                    min_sandboxes=1,
                    max_sandboxes=1,
                    slots_per_sandbox=2,
                ),
            },
            default_profile="profile-a",
        )
        
        await backend.start(profiles_to_start=["profile-a", "profile-b"])
        
        try:
            slot_a = await backend.acquire("traj-a", profile="profile-a")
            slot_b = await backend.acquire("traj-b", profile="profile-b")
            
            # Batch execute across profiles
            results_list = await backend.execute_batch([
                (slot_a, "bash", {"command": "echo 'from-a'"}),
                (slot_b, "bash", {"command": "echo 'from-b'"}),
            ])
            
            assert len(results_list) == 2
            assert "from-a" in results_list[0].output
            assert "from-b" in results_list[1].output
            
            await backend.release(slot_a)
            await backend.release(slot_b)
            
            results.record_pass("test_atropos_cross_profile_batch")
        finally:
            await backend.stop(purge=True)
    except ImportError as e:
        results.record_skip("test_atropos_cross_profile_batch", f"Requires atroposlib: pip install -e '.[atropos]'")
    except Exception as e:
        results.record_fail("test_atropos_cross_profile_batch", str(e))


async def test_atropos_artifact_helpers(config: TestConfig):
    """Test read_artifact, list_artifacts, archive_artifacts."""
    if config.dry_run:
        results.record_skip("test_atropos_artifact_helpers", "Dry run mode")
        return
    
    try:
        try:
            from atropos.backends.modal_backend import ModalToolBackend, ModalSandboxConfig
        except (ImportError, ModuleNotFoundError):
            ModalToolBackend, ModalSandboxConfig, _, _, _ = try_import_atropos_backend()
        
        config_obj = ModalSandboxConfig(
            app_name="test-artifacts",
            min_sandboxes=1,
            max_sandboxes=1,
            slots_per_sandbox=2,
        )
        
        backend = ModalToolBackend(config_obj)
        await backend.start()
        
        try:
            slot = await backend.acquire("artifact-test")
            
            # Create test files
            await backend.execute_batch([
                (slot, "bash", {"command": "echo 'hello' > file1.txt && echo 'world' > file2.txt && mkdir subdir && echo 'nested' > subdir/file3.txt"}),
            ])
            
            # Test read_artifact
            content = await backend.read_artifact(slot, "file1.txt")
            assert content["success"]
            assert "hello" in content["content"]
            
            # Test list_artifacts
            listing = await backend.list_artifacts(slot, ".", recursive=False)
            assert listing["success"]
            assert "file1.txt" in listing["entries"] or any("file1" in e for e in listing["entries"])
            
            # Test archive_artifacts
            archive = await backend.archive_artifacts(slot, ".", archive_format="tar.gz")
            assert archive["success"]
            assert len(archive["archive_base64"]) > 0
            
            await backend.release(slot)
            results.record_pass("test_atropos_artifact_helpers")
        finally:
            await backend.stop(purge=True)
    except ImportError as e:
        results.record_skip("test_atropos_artifact_helpers", f"Requires atroposlib: pip install -e '.[atropos]'")
    except Exception as e:
        results.record_fail("test_atropos_artifact_helpers", str(e))


# =============================================================================
# Test Runner
# =============================================================================

def run_sync_tests(config: TestConfig):
    """Run synchronous tests."""
    print("\n" + "="*60)
    print("SYNCHRONOUS TESTS")
    print("="*60)
    
    if config.category in (None, "profiles"):
        print("\n--- Profile Configuration Tests ---")
        test_profile_loading_from_env()
        test_profile_loading_from_yaml()
        test_profile_defaults()
        test_atropos_config_with_app_name()
    
    if config.category in (None, "terminal"):
        print("\n--- Terminal Tool Modal Tests ---")
        test_terminal_modal_pool_manager_singleton()
        test_terminal_create_environment_modal()
        test_terminal_tool_profile_parameter(config)
        test_terminal_modal_execute_simple(config)
        test_terminal_modal_persistence(config)
        test_terminal_modal_isolation(config)


async def run_async_tests(config: TestConfig):
    """Run asynchronous tests."""
    print("\n" + "="*60)
    print("ASYNCHRONOUS TESTS (Atropos Backend)")
    print("="*60)
    
    if config.category in (None, "atropos"):
        print("\n--- Backend Lifecycle Tests ---")
        await test_atropos_backend_lifecycle(config)
        
        print("\n--- Slot Management Tests ---")
        await test_atropos_slot_acquire_release(config)
        await test_atropos_execute_in_slot(config)
        await test_atropos_batched_execution(config)
        await test_atropos_slot_workspace_isolation(config)
        await test_atropos_workspace_reset(config)
        
        print("\n--- Multi-Profile Tests ---")
        await test_atropos_multi_profile(config)
        await test_atropos_cross_profile_batch(config)
        
        print("\n--- Artifact Helper Tests ---")
        await test_atropos_artifact_helpers(config)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Modal Integration Test Suite")
    parser.add_argument("--dry-run", action="store_true", help="Skip tests requiring Modal")
    parser.add_argument("--category", choices=["terminal", "atropos", "profiles"], help="Run specific category")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()
    
    config = TestConfig(
        dry_run=args.dry_run,
        verbose=args.verbose,
        category=args.category,
    )
    
    print("="*60)
    print("MODAL INTEGRATION TEST SUITE")
    print("="*60)
    print(f"Mode: {'DRY RUN' if config.dry_run else 'LIVE'}")
    print(f"Category: {config.category or 'ALL'}")
    
    # Run sync tests
    run_sync_tests(config)
    
    # Run async tests
    asyncio.run(run_async_tests(config))
    
    # Summary
    success = results.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
