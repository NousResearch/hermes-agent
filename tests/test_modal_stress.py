#!/usr/bin/env python3
"""
Modal Integration Stress Tests & Full Integration Tests

This test suite includes:
1. Stress tests for Modal sandbox pools (concurrent load, scaling)
2. Atropos backend tests (requires atroposlib)
3. mini-swe-agent integration tests

Prerequisites:
    # Install dev dependencies
    pip install -e '.[dev,modal]'
    
    # Install atroposlib for Atropos tests
    pip install -e '.[atropos]'
    
    # Clone mini-swe-agent (if not present)
    git clone https://github.com/anthropics/mini-swe-agent.git mini-swe-agent
    # Or as submodule:
    git submodule add https://github.com/anthropics/mini-swe-agent.git mini-swe-agent
    
Run with:
    # All tests
    python tests/test_modal_stress.py
    
    # Stress tests only
    python tests/test_modal_stress.py --category stress
    
    # Atropos tests only
    python tests/test_modal_stress.py --category atropos
    
    # Mini-swe-agent tests only
    python tests/test_modal_stress.py --category miniswe
    
    # Dry run (no Modal calls)
    python tests/test_modal_stress.py --dry-run
"""

import asyncio
import json
import os
import sys
import time
import random
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Test Configuration
# =============================================================================

@dataclass
class StressTestConfig:
    dry_run: bool = False
    verbose: bool = True
    category: Optional[str] = None
    # Stress test parameters (reduced defaults for faster first-run)
    concurrent_tasks: int = 3  # Start small - Modal cold starts are slow
    total_operations: int = 10
    max_sandboxes: int = 3
    slots_per_sandbox: int = 3


# =============================================================================
# Test Results Tracking
# =============================================================================

class TestResults:
    def __init__(self):
        self.passed: List[str] = []
        self.failed: List[Tuple[str, str]] = []
        self.skipped: List[Tuple[str, str]] = []
        self.metrics: Dict[str, Any] = {}
    
    def record_pass(self, name: str, metrics: Optional[Dict] = None):
        self.passed.append(name)
        if metrics:
            self.metrics[name] = metrics
        print(f"  ✅ {name}")
        if metrics:
            for k, v in metrics.items():
                print(f"     📊 {k}: {v}")
    
    def record_fail(self, name: str, error: str):
        self.failed.append((name, error))
        print(f"  ❌ {name}: {error}")
    
    def record_skip(self, name: str, reason: str):
        self.skipped.append((name, reason))
        print(f"  ⏭️  {name}: {reason}")
    
    def summary(self):
        total = len(self.passed) + len(self.failed) + len(self.skipped)
        print(f"\n{'='*70}")
        print(f"STRESS TEST RESULTS: {len(self.passed)}/{total} passed")
        print(f"  Passed:  {len(self.passed)}")
        print(f"  Failed:  {len(self.failed)}")
        print(f"  Skipped: {len(self.skipped)}")
        
        if self.failed:
            print(f"\nFailed tests:")
            for name, error in self.failed:
                print(f"  - {name}: {error}")
        
        if self.metrics:
            print(f"\nPerformance Metrics:")
            for test, metrics in self.metrics.items():
                print(f"  {test}:")
                for k, v in metrics.items():
                    print(f"    - {k}: {v}")
        
        return len(self.failed) == 0


results = TestResults()


# =============================================================================
# Helper: Atropos Import
# =============================================================================

def try_import_atropos():
    """Try importing Atropos backend components."""
    try:
        from atropos.backends.modal_backend import (
            ModalToolBackend, ModalSandboxConfig,
            _ModalMultiProfileManager
        )
        from atropos.slots.slot import Slot, SlotState
        return ModalToolBackend, ModalSandboxConfig, Slot, SlotState
    except (ImportError, ModuleNotFoundError) as e:
        return None


def try_import_miniswe():
    """Try importing mini-swe-agent components."""
    try:
        # Check if mini-swe-agent path exists and has content
        mini_swe_path = Path(__file__).parent.parent / "mini-swe-agent" / "src"
        if mini_swe_path.exists() and list(mini_swe_path.iterdir()):
            sys.path.insert(0, str(mini_swe_path))
            import minisweagent
            return minisweagent
        return None
    except (ImportError, ModuleNotFoundError) as e:
        return None


# =============================================================================
# CATEGORY 1: Stress Tests (Terminal Tool)
# =============================================================================

def test_stress_concurrent_tasks(config: StressTestConfig):
    """Stress test: Multiple concurrent task_ids hitting the pool."""
    if config.dry_run:
        results.record_skip("test_stress_concurrent_tasks", "Dry run mode")
        return
    
    from tools.terminal_tool import terminal_tool, cleanup_vm
    
    original_env = os.environ.get("TERMINAL_ENV")
    os.environ["TERMINAL_ENV"] = "modal"
    
    try:
        num_tasks = config.concurrent_tasks
        task_ids = [f"stress-concurrent-{i}-{int(time.time())}" for i in range(num_tasks)]
        
        start_time = time.time()
        errors = []
        successes = 0
        
        def run_task(task_id: str) -> Tuple[bool, str]:
            try:
                result = json.loads(terminal_tool(
                    f"echo 'Hello from {task_id}' && sleep 0.5",
                    task_id=task_id,
                ))
                success = result["exit_code"] == 0
                
                # IMPORTANT: Clean up immediately after task completes
                # This releases the sandbox back to the pool for other tasks
                try:
                    cleanup_vm(task_id)
                except:
                    pass
                
                if success:
                    return True, ""
                # Include more details for debugging
                error_detail = result.get("error", "no error message")
                output = result.get("output", "")[:100]  # First 100 chars
                return False, f"Exit code: {result['exit_code']}, error: {error_detail}, output: {output}"
            except Exception as e:
                # Clean up even on failure
                try:
                    cleanup_vm(task_id)
                except:
                    pass
                import traceback
                return False, f"Exception: {str(e)}\n{traceback.format_exc()}"
        
        # Run all tasks concurrently using threads
        with ThreadPoolExecutor(max_workers=num_tasks) as executor:
            futures = {executor.submit(run_task, tid): tid for tid in task_ids}
            
            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    success, error = future.result(timeout=60)
                    if success:
                        successes += 1
                    else:
                        errors.append(f"{task_id}: {error}")
                except Exception as e:
                    errors.append(f"{task_id}: {str(e)}")
        
        elapsed = time.time() - start_time
        
        # No need for cleanup here - each task cleans up immediately
        
        # Report
        success_rate = successes / num_tasks * 100
        
        if success_rate >= 90:  # Allow 10% failure rate for stress test
            results.record_pass("test_stress_concurrent_tasks", {
                "concurrent_tasks": num_tasks,
                "successes": successes,
                "failures": len(errors),
                "success_rate": f"{success_rate:.1f}%",
                "total_time": f"{elapsed:.2f}s",
                "avg_time_per_task": f"{elapsed/num_tasks:.2f}s",
            })
        else:
            results.record_fail(
                "test_stress_concurrent_tasks",
                f"Success rate {success_rate:.1f}% < 90%. Errors: {errors[:3]}"
            )
    
    except Exception as e:
        results.record_fail("test_stress_concurrent_tasks", str(e))
    finally:
        if original_env:
            os.environ["TERMINAL_ENV"] = original_env
        elif "TERMINAL_ENV" in os.environ:
            del os.environ["TERMINAL_ENV"]


def test_stress_rapid_fire(config: StressTestConfig):
    """Stress test: Rapid sequential commands to same task_id."""
    if config.dry_run:
        results.record_skip("test_stress_rapid_fire", "Dry run mode")
        return
    
    from tools.terminal_tool import terminal_tool, cleanup_vm
    
    original_env = os.environ.get("TERMINAL_ENV")
    os.environ["TERMINAL_ENV"] = "modal"
    
    try:
        task_id = f"stress-rapid-{int(time.time())}"
        num_commands = config.total_operations
        
        start_time = time.time()
        successes = 0
        errors = []
        
        for i in range(num_commands):
            try:
                result = json.loads(terminal_tool(f"echo {i}", task_id=task_id))
                if result["exit_code"] == 0 and str(i) in result["output"]:
                    successes += 1
                else:
                    errors.append(f"Command {i}: unexpected result")
            except Exception as e:
                errors.append(f"Command {i}: {str(e)}")
        
        elapsed = time.time() - start_time
        cleanup_vm(task_id)
        
        success_rate = successes / num_commands * 100
        commands_per_second = num_commands / elapsed
        
        if success_rate >= 95:
            results.record_pass("test_stress_rapid_fire", {
                "total_commands": num_commands,
                "successes": successes,
                "success_rate": f"{success_rate:.1f}%",
                "total_time": f"{elapsed:.2f}s",
                "commands_per_second": f"{commands_per_second:.1f}",
            })
        else:
            results.record_fail(
                "test_stress_rapid_fire",
                f"Success rate {success_rate:.1f}% < 95%"
            )
    
    except Exception as e:
        results.record_fail("test_stress_rapid_fire", str(e))
    finally:
        if original_env:
            os.environ["TERMINAL_ENV"] = original_env
        elif "TERMINAL_ENV" in os.environ:
            del os.environ["TERMINAL_ENV"]


def test_stress_pool_scaling(config: StressTestConfig):
    """Stress test: Force pool to scale up and down by running tasks in batches."""
    if config.dry_run:
        results.record_skip("test_stress_pool_scaling", "Dry run mode")
        return
    
    from tools.terminal_tool import terminal_tool, cleanup_vm, _ModalPoolManager
    
    original_env = os.environ.get("TERMINAL_ENV")
    os.environ["TERMINAL_ENV"] = "modal"
    
    try:
        # Run tasks in batches matching max_sandboxes to test pool reuse
        # This verifies sandboxes can be acquired, used, released, and reused
        batch_size = config.max_sandboxes
        num_batches = 3
        total_tasks = batch_size * num_batches
        
        start_time = time.time()
        successes = 0
        
        for batch in range(num_batches):
            task_ids = [f"stress-scale-{batch}-{i}-{int(time.time())}" for i in range(batch_size)]
            
            def run_task(task_id: str):
                try:
                    result = json.loads(terminal_tool(
                        "echo done",  # Fast command to test scaling
                        task_id=task_id,
                    ))
                    success = result["exit_code"] == 0
                    try:
                        cleanup_vm(task_id)
                    except:
                        pass
                    return success
                except:
                    try:
                        cleanup_vm(task_id)
                    except:
                        pass
                    return False
            
            # Run batch concurrently
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                batch_results = list(executor.map(run_task, task_ids))
            successes += sum(batch_results)
        
        elapsed = time.time() - start_time
        
        # Check pool status
        try:
            manager = _ModalPoolManager.get_instance()
            pool_status = manager.get_status() if hasattr(manager, 'get_status') else {}
        except:
            pool_status = {}
        
        success_rate = successes / total_tasks * 100
        
        if success_rate >= 80:  # Allow some tolerance
            results.record_pass("test_stress_pool_scaling", {
                "total_tasks": total_tasks,
                "num_batches": num_batches,
                "batch_size": batch_size,
                "successes": successes,
                "success_rate": f"{success_rate:.1f}%",
                "total_time": f"{elapsed:.2f}s",
                "pool_status": pool_status,
            })
        else:
            results.record_fail(
                "test_stress_pool_scaling",
                f"Success rate {success_rate:.1f}% < 80%"
            )
    
    except Exception as e:
        results.record_fail("test_stress_pool_scaling", str(e))
    finally:
        if original_env:
            os.environ["TERMINAL_ENV"] = original_env
        elif "TERMINAL_ENV" in os.environ:
            del os.environ["TERMINAL_ENV"]


def test_stress_large_output(config: StressTestConfig):
    """Stress test: Commands producing large output."""
    if config.dry_run:
        results.record_skip("test_stress_large_output", "Dry run mode")
        return
    
    from tools.terminal_tool import terminal_tool, cleanup_vm
    
    original_env = os.environ.get("TERMINAL_ENV")
    os.environ["TERMINAL_ENV"] = "modal"
    
    try:
        task_id = f"stress-large-{int(time.time())}"
        
        # First verify basic connectivity with simple command
        warmup = json.loads(terminal_tool("echo warmup", task_id=task_id))
        if warmup["exit_code"] != 0:
            results.record_fail(
                "test_stress_large_output",
                f"Warmup failed: {warmup.get('error', 'unknown')}"
            )
            return
        
        # Generate output - use seq which is more portable
        start_time = time.time()
        result = json.loads(terminal_tool(
            'seq 1 500 | while read i; do echo "Line $i: This is test content for large output"; done',
            task_id=task_id,
            timeout=60,
        ))
        elapsed = time.time() - start_time
        
        cleanup_vm(task_id)
        
        output_size = len(result.get("output", ""))
        error_msg = result.get("error", "")
        
        if result["exit_code"] == 0 and output_size > 5000:
            results.record_pass("test_stress_large_output", {
                "output_size": f"{output_size:,} bytes",
                "time": f"{elapsed:.2f}s",
                "throughput": f"{output_size/elapsed/1024:.1f} KB/s" if elapsed > 0 else "N/A",
            })
        else:
            results.record_fail(
                "test_stress_large_output",
                f"Exit code: {result['exit_code']}, output size: {output_size}, error: {error_msg}"
            )
    
    except Exception as e:
        import traceback
        results.record_fail("test_stress_large_output", f"{str(e)}\n{traceback.format_exc()}")
    finally:
        try:
            cleanup_vm(task_id)
        except:
            pass
        if original_env:
            os.environ["TERMINAL_ENV"] = original_env
        elif "TERMINAL_ENV" in os.environ:
            del os.environ["TERMINAL_ENV"]


def test_stress_error_recovery(config: StressTestConfig):
    """Stress test: Commands that fail and verify sandbox continues working."""
    if config.dry_run:
        results.record_skip("test_stress_error_recovery", "Dry run mode")
        return
    
    from tools.terminal_tool import terminal_tool, cleanup_vm
    
    original_env = os.environ.get("TERMINAL_ENV")
    os.environ["TERMINAL_ENV"] = "modal"
    
    try:
        task_id = f"stress-error-{int(time.time())}"
        
        # Run some failing commands
        failing_commands = [
            "exit 1",
            "false",
            "cat /nonexistent/file",
            "command_that_does_not_exist",
        ]
        
        for cmd in failing_commands:
            result = json.loads(terminal_tool(cmd, task_id=task_id))
            # These should fail but not crash
            assert result["exit_code"] != 0 or result.get("error"), f"Expected failure for: {cmd}"
        
        # Now run a command that should succeed
        result = json.loads(terminal_tool("echo 'recovery success'", task_id=task_id))
        
        cleanup_vm(task_id)
        
        if result["exit_code"] == 0 and "recovery success" in result["output"]:
            results.record_pass("test_stress_error_recovery", {
                "failed_commands": len(failing_commands),
                "recovery": "success",
            })
        else:
            results.record_fail(
                "test_stress_error_recovery",
                f"Recovery failed: {result}"
            )
    
    except Exception as e:
        results.record_fail("test_stress_error_recovery", str(e))
    finally:
        if original_env:
            os.environ["TERMINAL_ENV"] = original_env
        elif "TERMINAL_ENV" in os.environ:
            del os.environ["TERMINAL_ENV"]


# =============================================================================
# CATEGORY 2: Atropos Backend Stress Tests
# =============================================================================

async def test_atropos_stress_slot_churn(config: StressTestConfig):
    """Atropos stress test: Rapid slot acquire/release cycles."""
    if config.dry_run:
        results.record_skip("test_atropos_stress_slot_churn", "Dry run mode")
        return
    
    imports = try_import_atropos()
    if imports is None:
        results.record_skip("test_atropos_stress_slot_churn", "Requires atroposlib")
        return
    
    ModalToolBackend, ModalSandboxConfig, _, _ = imports
    
    try:
        backend_config = ModalSandboxConfig(
            app_name=f"stress-churn-{int(time.time())}",
            min_sandboxes=1,
            max_sandboxes=3,
            slots_per_sandbox=5,
        )
        
        backend = ModalToolBackend(backend_config)
        await backend.start()
        
        try:
            num_cycles = config.total_operations
            start_time = time.time()
            successes = 0
            
            for i in range(num_cycles):
                try:
                    slot = await backend.acquire(f"churn-{i}")
                    
                    # Quick command
                    results_list = await backend.execute_batch([
                        (slot, "bash", {"command": f"echo {i}"})
                    ])
                    
                    if results_list[0].success:
                        successes += 1
                    
                    await backend.release(slot, reset_workspace=(i % 5 == 0))
                except Exception as e:
                    pass  # Count as failure
            
            elapsed = time.time() - start_time
            success_rate = successes / num_cycles * 100
            
            if success_rate >= 90:
                results.record_pass("test_atropos_stress_slot_churn", {
                    "cycles": num_cycles,
                    "successes": successes,
                    "success_rate": f"{success_rate:.1f}%",
                    "total_time": f"{elapsed:.2f}s",
                    "cycles_per_second": f"{num_cycles/elapsed:.1f}",
                })
            else:
                results.record_fail(
                    "test_atropos_stress_slot_churn",
                    f"Success rate {success_rate:.1f}% < 90%"
                )
        
        finally:
            await backend.stop(purge=True)
    
    except Exception as e:
        results.record_fail("test_atropos_stress_slot_churn", str(e))


async def test_atropos_stress_parallel_batches(config: StressTestConfig):
    """Atropos stress test: Multiple parallel batch executions."""
    if config.dry_run:
        results.record_skip("test_atropos_stress_parallel_batches", "Dry run mode")
        return
    
    imports = try_import_atropos()
    if imports is None:
        results.record_skip("test_atropos_stress_parallel_batches", "Requires atroposlib")
        return
    
    ModalToolBackend, ModalSandboxConfig, _, _ = imports
    
    try:
        backend_config = ModalSandboxConfig(
            app_name=f"stress-batch-{int(time.time())}",
            min_sandboxes=2,
            max_sandboxes=4,
            slots_per_sandbox=5,
        )
        
        backend = ModalToolBackend(backend_config)
        await backend.start()
        
        try:
            num_slots = 10
            slots = []
            
            # Acquire multiple slots
            for i in range(num_slots):
                slot = await backend.acquire(f"batch-{i}")
                slots.append(slot)
            
            # Run multiple batches in parallel
            start_time = time.time()
            num_batches = 5
            
            async def run_batch(batch_id: int):
                requests = [
                    (slot, "bash", {"command": f"echo 'batch{batch_id}-slot{i}'"})
                    for i, slot in enumerate(slots)
                ]
                return await backend.execute_batch(requests)
            
            batch_tasks = [run_batch(i) for i in range(num_batches)]
            all_results = await asyncio.gather(*batch_tasks)
            
            elapsed = time.time() - start_time
            
            # Count successes
            total_commands = num_batches * num_slots
            successes = sum(
                1 for batch_result in all_results
                for r in batch_result
                if r.success
            )
            
            # Release slots
            for slot in slots:
                await backend.release(slot)
            
            success_rate = successes / total_commands * 100
            
            if success_rate >= 90:
                results.record_pass("test_atropos_stress_parallel_batches", {
                    "batches": num_batches,
                    "slots": num_slots,
                    "total_commands": total_commands,
                    "successes": successes,
                    "success_rate": f"{success_rate:.1f}%",
                    "total_time": f"{elapsed:.2f}s",
                    "commands_per_second": f"{total_commands/elapsed:.1f}",
                })
            else:
                results.record_fail(
                    "test_atropos_stress_parallel_batches",
                    f"Success rate {success_rate:.1f}% < 90%"
                )
        
        finally:
            await backend.stop(purge=True)
    
    except Exception as e:
        results.record_fail("test_atropos_stress_parallel_batches", str(e))


async def test_atropos_stress_multi_profile_load(config: StressTestConfig):
    """Atropos stress test: Load across multiple profiles."""
    if config.dry_run:
        results.record_skip("test_atropos_stress_multi_profile_load", "Dry run mode")
        return
    
    imports = try_import_atropos()
    if imports is None:
        results.record_skip("test_atropos_stress_multi_profile_load", "Requires atroposlib")
        return
    
    ModalToolBackend, ModalSandboxConfig, _, _ = imports
    
    try:
        backend = ModalToolBackend.with_profiles(
            app_name=f"stress-multiprofile-{int(time.time())}",
            profiles={
                "cpu-light": ModalSandboxConfig(
                    name="cpu-light",
                    cpu=0.5,
                    memory=1024,
                    min_sandboxes=1,
                    max_sandboxes=2,
                    slots_per_sandbox=5,
                ),
                "cpu-heavy": ModalSandboxConfig(
                    name="cpu-heavy",
                    cpu=2.0,
                    memory=4096,
                    min_sandboxes=0,
                    max_sandboxes=2,
                    slots_per_sandbox=3,
                ),
            }
        )
        
        await backend.start(profiles_to_start=["cpu-light", "cpu-heavy"])
        
        try:
            num_tasks_per_profile = 5
            slots = []
            
            # Acquire from both profiles
            for i in range(num_tasks_per_profile):
                light_slot = await backend.acquire(f"light-{i}", profile="cpu-light")
                heavy_slot = await backend.acquire(f"heavy-{i}", profile="cpu-heavy")
                slots.append((light_slot, "cpu-light"))
                slots.append((heavy_slot, "cpu-heavy"))
            
            # Execute batch across all profiles
            start_time = time.time()
            
            requests = [
                (slot, "bash", {"command": f"echo 'profile={profile}'"})
                for slot, profile in slots
            ]
            
            batch_results = await backend.execute_batch(requests)
            elapsed = time.time() - start_time
            
            successes = sum(1 for r in batch_results if r.success)
            
            # Release all
            for slot, _ in slots:
                await backend.release(slot)
            
            status = backend.get_status()
            
            success_rate = successes / len(slots) * 100
            
            if success_rate >= 90:
                results.record_pass("test_atropos_stress_multi_profile_load", {
                    "profiles": 2,
                    "tasks_per_profile": num_tasks_per_profile,
                    "total_tasks": len(slots),
                    "successes": successes,
                    "success_rate": f"{success_rate:.1f}%",
                    "time": f"{elapsed:.2f}s",
                    "status": status,
                })
            else:
                results.record_fail(
                    "test_atropos_stress_multi_profile_load",
                    f"Success rate {success_rate:.1f}% < 90%"
                )
        
        finally:
            await backend.stop(purge=True)
    
    except Exception as e:
        results.record_fail("test_atropos_stress_multi_profile_load", str(e))


# =============================================================================
# CATEGORY 3: Mini-SWE-Agent Integration Tests
# =============================================================================

def test_miniswe_environment_available():
    """Check if mini-swe-agent is properly set up."""
    mini_swe_path = Path(__file__).parent.parent / "mini-swe-agent" / "src"
    
    if not mini_swe_path.exists():
        results.record_skip(
            "test_miniswe_environment_available",
            "mini-swe-agent not found. Run: git clone https://github.com/anthropics/mini-swe-agent.git mini-swe-agent"
        )
        return
    
    if not list(mini_swe_path.iterdir()):
        results.record_skip(
            "test_miniswe_environment_available",
            "mini-swe-agent directory is empty. Run: git submodule update --init"
        )
        return
    
    miniswe = try_import_miniswe()
    if miniswe is None:
        results.record_fail(
            "test_miniswe_environment_available",
            "Failed to import minisweagent module"
        )
        return
    
    results.record_pass("test_miniswe_environment_available", {
        "path": str(mini_swe_path),
        "module": miniswe.__name__,
    })


def test_miniswe_modal_backend(config: StressTestConfig):
    """Test mini-swe-agent with Modal backend."""
    if config.dry_run:
        results.record_skip("test_miniswe_modal_backend", "Dry run mode")
        return
    
    miniswe = try_import_miniswe()
    if miniswe is None:
        results.record_skip(
            "test_miniswe_modal_backend",
            "mini-swe-agent not available"
        )
        return
    
    try:
        # Check if ModalEnvironment exists in minisweagent
        if not hasattr(miniswe, 'ModalEnvironment'):
            results.record_skip(
                "test_miniswe_modal_backend",
                "minisweagent.ModalEnvironment not found"
            )
            return
        
        # Create Modal environment
        env = miniswe.ModalEnvironment(
            image="python:3.11",
            timeout=60,
        )
        
        # Execute a command
        result = env.execute("echo 'Hello from mini-swe-agent Modal'")
        
        env.cleanup()
        
        if "Hello from mini-swe-agent Modal" in str(result):
            results.record_pass("test_miniswe_modal_backend")
        else:
            results.record_fail(
                "test_miniswe_modal_backend",
                f"Unexpected result: {result}"
            )
    
    except Exception as e:
        results.record_fail("test_miniswe_modal_backend", str(e))


# =============================================================================
# Test Runner
# =============================================================================

def run_sync_tests(config: StressTestConfig):
    """Run synchronous tests."""
    if config.category in (None, "stress"):
        print("\n" + "="*70)
        print("STRESS TESTS (Terminal Tool)")
        print("="*70)
        
        test_stress_concurrent_tasks(config)
        test_stress_rapid_fire(config)
        test_stress_pool_scaling(config)
        test_stress_large_output(config)
        test_stress_error_recovery(config)
    
    if config.category in (None, "miniswe"):
        print("\n" + "="*70)
        print("MINI-SWE-AGENT INTEGRATION TESTS")
        print("="*70)
        
        test_miniswe_environment_available()
        test_miniswe_modal_backend(config)


async def run_async_tests(config: StressTestConfig):
    """Run asynchronous tests."""
    if config.category in (None, "atropos"):
        print("\n" + "="*70)
        print("ATROPOS BACKEND STRESS TESTS")
        print("="*70)
        
        await test_atropos_stress_slot_churn(config)
        await test_atropos_stress_parallel_batches(config)
        await test_atropos_stress_multi_profile_load(config)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Modal Stress Test Suite")
    parser.add_argument("--dry-run", action="store_true", help="Skip tests requiring Modal")
    parser.add_argument("--category", choices=["stress", "atropos", "miniswe"], help="Run specific category")
    parser.add_argument("--concurrent", type=int, default=10, help="Number of concurrent tasks")
    parser.add_argument("--operations", type=int, default=50, help="Total operations for stress tests")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()
    
    config = StressTestConfig(
        dry_run=args.dry_run,
        verbose=args.verbose,
        category=args.category,
        concurrent_tasks=args.concurrent,
        total_operations=args.operations,
    )
    
    print("="*70)
    print("MODAL STRESS & INTEGRATION TEST SUITE")
    print("="*70)
    print(f"Mode: {'DRY RUN' if config.dry_run else 'LIVE'}")
    print(f"Category: {config.category or 'ALL'}")
    print(f"Concurrent tasks: {config.concurrent_tasks}")
    print(f"Total operations: {config.total_operations}")
    
    # Run sync tests
    run_sync_tests(config)
    
    # Run async tests
    asyncio.run(run_async_tests(config))
    
    # Summary
    success = results.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
