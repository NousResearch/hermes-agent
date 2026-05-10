#!/usr/bin/env python3
"""
Test script to verify checkpoint behavior in batch_runner.py

This script simulates batch processing with intentional failures to test:
1. Whether checkpoints are saved incrementally during processing
2. Whether resume functionality works correctly after interruption
3. Whether data integrity is maintained across checkpoint cycles

Usage:
    # Test current implementation
    python tests/test_checkpoint_resumption.py --test_current

    # Test after fix is applied
    python tests/test_checkpoint_resumption.py --test_fixed

    # Run full comparison
    python tests/test_checkpoint_resumption.py --compare
"""

import pytest
pytestmark = pytest.mark.integration

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import traceback

# Add project root to path to import batch_runner
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def create_test_dataset(num_prompts: int = 20) -> Path:
    """Create a small test dataset for checkpoint testing."""
    test_data_dir = Path("tests/test_data")
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_file = test_data_dir / "checkpoint_test_dataset.jsonl"
    
    with open(dataset_file, 'w', encoding='utf-8') as f:
        for i in range(num_prompts):
            entry = {
                "prompt": f"Test prompt {i}: What is 2+2? Just answer briefly.",
                "test_id": i
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"✅ Created test dataset: {dataset_file} ({num_prompts} prompts)")
    return dataset_file


def monitor_checkpoint_during_run(checkpoint_file: Path, duration: int = 30) -> List[Dict[str, Any]]:
    """
    Monitor checkpoint file during a batch run to see when it gets updated.
    
    Args:
        checkpoint_file: Path to checkpoint file to monitor
        duration: How long to monitor (seconds)
    
    Returns:
        List of checkpoint snapshots with timestamps
    """
    snapshots = []
    start_time = time.time()
    last_mtime = None
    
    print(f"\n🔍 Monitoring checkpoint file: {checkpoint_file}")
    print(f"   Duration: {duration}s")
    print("-" * 70)
    
    while time.time() - start_time < duration:
        if checkpoint_file.exists():
            current_mtime = checkpoint_file.stat().st_mtime
            
            # Check if file was modified
            if last_mtime is None or current_mtime != last_mtime:
                elapsed = time.time() - start_time
                
                try:
                    with open(checkpoint_file, 'r') as f:
                        checkpoint_data = json.load(f)
                    
                    snapshot = {
                        "elapsed_seconds": round(elapsed, 2),
                        "completed_count": len(checkpoint_data.get("completed_prompts", [])),
                        "completed_prompts": checkpoint_data.get("completed_prompts", [])[:5],  # First 5 for display
                        "timestamp": checkpoint_data.get("last_updated")
                    }
                    
                    snapshots.append(snapshot)
                    
                    print(f"[{elapsed:6.2f}s] Checkpoint updated: {snapshot['completed_count']} prompts completed")
                    
                except Exception as e:
                    print(f"[{elapsed:6.2f}s] Error reading checkpoint: {e}")
                
                last_mtime = current_mtime
        else:
            if len(snapshots) == 0:
                print(f"[{time.time() - start_time:6.2f}s] Checkpoint file not yet created...")
        
        time.sleep(0.5)  # Check every 0.5 seconds
    
    return snapshots


def _cleanup_test_artifacts(*paths):
    """Remove test-generated files and directories."""
    for p in paths:
        p = Path(p)
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.is_file():
            p.unlink(missing_ok=True)


def test_current_implementation():
    """Test the current checkpoint implementation."""
    print("\n" + "=" * 70)
    print("TEST 1: Current Implementation - Checkpoint Timing")
    print("=" * 70)
    print("\n📝 Testing whether checkpoints are saved incrementally during run...")
    
    # Setup
    dataset_file = create_test_dataset(num_prompts=12)
    run_name = "checkpoint_test_current"
    output_dir = Path("data") / run_name
    
    # Clean up any existing test data
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    # Import here to avoid issues if module changes
    from batch_runner import BatchRunner
    
    checkpoint_file = output_dir / "checkpoint.json"
    
    # Start monitoring in a separate process would be ideal, but for simplicity
    # we'll just check before and after
    print(f"\n▶️  Starting batch run...")
    print(f"   Dataset: {dataset_file}")
    print(f"   Batch size: 3 (4 batches total)")
    print(f"   Workers: 2")
    print(f"   Expected behavior: If incremental, checkpoint should update during run")
    
    start_time = time.time()
    
    try:
        runner = BatchRunner(
            dataset_file=str(dataset_file),
            batch_size=3,
            run_name=run_name,
            distribution="default",
            max_iterations=3,  # Keep it short
            model="claude-opus-4-20250514",
            num_workers=2,
            verbose=False
        )
        
        # Run with monitoring
        import threading
        snapshots = []
        
        def monitor():
            nonlocal snapshots
            snapshots = monitor_checkpoint_during_run(checkpoint_file, duration=60)
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        
        runner.run(resume=False)
        
        monitor_thread.join(timeout=2)
        
    except Exception as e:
        print(f"❌ Error during run: {e}")
        traceback.print_exc()
        return False
    finally:
        _cleanup_test_artifacts(dataset_file, output_dir)
    
    elapsed = time.time() - start_time
    
    # Analyze results
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS")
    print("=" * 70)
    print(f"Total run time: {elapsed:.2f}s")
    print(f"Checkpoint updates observed: {len(snapshots)}")
    
    if len(snapshots) == 0:
        print("\n❌ ISSUE: No checkpoint updates observed during run")
        print("   This suggests checkpoints are only saved at the end")
        return False
    elif len(snapshots) == 1:
        print("\n⚠️  WARNING: Only 1 checkpoint update (likely at the end)")
        print("   This confirms the bug - no incremental checkpointing")
        return False
    else:
        print(f"\n✅ GOOD: Multiple checkpoint updates ({len(snapshots)}) observed")
        print("   Checkpointing appears to be incremental")
        
        # Show timeline
        print("\n📈 Checkpoint Timeline:")
        for i, snapshot in enumerate(snapshots, 1):
            print(f"   {i}. [{snapshot['elapsed_seconds']:6.2f}s] "
                  f"{snapshot['completed_count']} prompts completed")
        
        return True


def test_interruption_and_resume():
    """Test that resume actually works after interruption."""
    print("\n" + "=" * 70)
    print("TEST 2: Interruption and Resume")
    print("=" * 70)
    print("\n📝 Testing whether resume works after manual interruption...")
    
    # Setup
    dataset_file = create_test_dataset(num_prompts=15)
    run_name = "checkpoint_test_resume"
    output_dir = Path("data") / run_name
    
    # Clean up any existing test data
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    from batch_runner import BatchRunner
    
    checkpoint_file = output_dir / "checkpoint.json"
    
    print(f"\n▶️  Starting first run (will process 5 prompts, then simulate interruption)...")
    
    temp_dataset = Path("tests/test_data/checkpoint_test_resume_partial.jsonl")
    try:
        # Create a modified dataset with only first 5 prompts for initial run
        with open(dataset_file, 'r') as f:
            lines = f.readlines()[:5]
        with open(temp_dataset, 'w') as f:
            f.writelines(lines)
        
        runner = BatchRunner(
            dataset_file=str(temp_dataset),
            batch_size=2,
            run_name=run_name,
            distribution="default",
            max_iterations=3,
            model="claude-opus-4-20250514",
            num_workers=1,
            verbose=False
        )
        
        runner.run(resume=False)
        
        # Check checkpoint after first run
        if not checkpoint_file.exists():
            print("❌ ERROR: Checkpoint file not created after first run")
            return False
        
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        
        initial_completed = len(checkpoint_data.get("completed_prompts", []))
        print(f"✅ First run completed: {initial_completed} prompts saved to checkpoint")
        
        # Now try to resume with full dataset
        print(f"\n▶️  Starting resume run with full dataset (15 prompts)...")
        
        runner2 = BatchRunner(
            dataset_file=str(dataset_file),
            batch_size=2,
            run_name=run_name,
            distribution="default",
            max_iterations=3,
            model="claude-opus-4-20250514",
            num_workers=1,
            verbose=False
        )
        
        runner2.run(resume=True)
        
        # Check final checkpoint
        with open(checkpoint_file, 'r') as f:
            final_checkpoint = json.load(f)
        
        final_completed = len(final_checkpoint.get("completed_prompts", []))
        
        print("\n" + "=" * 70)
        print("📊 TEST RESULTS")
        print("=" * 70)
        print(f"Initial completed: {initial_completed}")
        print(f"Final completed: {final_completed}")
        print(f"Expected: 15")
        
        if final_completed == 15:
            print("\n✅ PASS: Resume successfully completed all prompts")
            return True
        else:
            print(f"\n❌ FAIL: Expected 15 completed, got {final_completed}")
            return False
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        traceback.print_exc()
        return False
    finally:
        _cleanup_test_artifacts(dataset_file, temp_dataset, output_dir)


def test_simulated_crash():
    """Test behavior when process crashes mid-execution."""
    print("\n" + "=" * 70)
    print("TEST 3: Simulated Crash During Execution")
    print("=" * 70)
    print("\n📝 This test would require running in a subprocess and killing it...")
    print("   Skipping for safety - manual testing recommended")
    return None


def print_test_plan():
    """Print the detailed test and fix plan."""
    print("\n" + "=" * 70)
    print("CHECKPOINT FIX - DETAILED PLAN")
    print("=" * 70)
    
    print("""
📋 PROBLEM SUMMARY
------------------
Current implementation uses pool.map() which blocks until ALL batches complete.
Checkpoint is only saved after all batches finish (line 558-559).

If process crashes during batch processing:
- All progress is lost
- Resume does nothing (no incremental checkpoint was saved)

📋 PROPOSED SOLUTION
--------------------
Replace pool.map() with pool.imap_unordered() to get results as they complete.
Save checkpoint after EACH batch completes using a multiprocessing Lock.

Key changes:
1. Use Manager().Lock() for thread-safe checkpoint writes
2. Replace pool.map() with pool.imap_unordered()
3. Update checkpoint after each batch result
4. Maintain backward compatibility with existing checkpoints

📋 IMPLEMENTATION STEPS
-----------------------
1. Add Manager and Lock initialization before Pool creation
2. Pass shared checkpoint data and lock to workers (via Manager)
3. Replace pool.map() with pool.imap_unordered()
4. In result loop: save checkpoint after each batch
5. Add error handling for checkpoint write failures

📋 RISKS & MITIGATIONS
----------------------
Risk: Checkpoint file corruption if two processes write simultaneously
→ Mitigation: Use multiprocessing.Lock() for exclusive access

Risk: Performance impact from frequent checkpoint writes
→ Mitigation: Checkpoint writes are fast (small JSON), negligible impact

Risk: Breaking existing runs that are already checkpointed
→ Mitigation: Maintain checkpoint format, only change timing

Risk: Bugs in multiprocessing lock/manager code
→ Mitigation: Thorough testing with this test script

📋 TESTING STRATEGY
-------------------
1. Run test_current_implementation() - Confirm bug exists
2. Apply fix to batch_runner.py
3. Run test_current_implementation() again - Should see incremental updates
4. Run test_interruption_and_resume() - Verify resume works
5. Manual test: Start run, kill process mid-batch, resume

📋 ROLLBACK PLAN
----------------
If issues arise:
1. Git revert the changes
2. Original code is working (just missing incremental checkpoint)
3. No data corruption risk - checkpoints are write-only
""")


def main(
    test_current: bool = False,
    test_resume: bool = False,
    test_crash: bool = False,
    compare: bool = False,
    show_plan: bool = False
):
    """
    Run checkpoint behavior tests.
    
    Args:
        test_current: Test current implementation checkpoint timing
        test_resume: Test interruption and resume functionality
        test_crash: Test simulated crash scenario (manual)
        compare: Run all tests and compare
        show_plan: Show detailed fix plan
    """
    if show_plan or (not any([test_current, test_resume, test_crash, compare])):
        print_test_plan()
        return
    
    results = {}
    
    if test_current or compare:
        results['current'] = test_current_implementation()
    
    if test_resume or compare:
        results['resume'] = test_interruption_and_resume()
    
    if test_crash or compare:
        results['crash'] = test_simulated_crash()
    
    # Summary
    if results:
        print("\n" + "=" * 70)
        print("OVERALL TEST SUMMARY")
        print("=" * 70)
        for test_name, result in results.items():
            if result is None:
                status = "⏭️  SKIPPED"
            elif result:
                status = "✅ PASS"
            else:
                status = "❌ FAIL"
            print(f"{status} - {test_name}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)

