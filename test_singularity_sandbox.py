#!/usr/bin/env python3
"""
Test script for Singularity/Apptainer sandbox integration.

This tests the SlotPool with driver="singularity" using the raw_exec Nomad driver.
"""

import asyncio
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from atropos.slots.pool import SlotPool, SlotPoolConfig


async def test_singularity_sandbox():
    """Test the Singularity sandbox deployment and basic execution."""
    
    # Configure for Singularity
    config = SlotPoolConfig(
        nomad_address="http://localhost:4646",
        job_id="atropos-sandbox-singularity",
        driver="singularity",
        singularity_image="/root/Hermes-Agent/atropos/atropos-sandbox.sif",
        slots_per_container=5,
        min_containers=1,
        max_containers=2,
        cpu=500,
        memory=512,
        purge_job_on_start=True,  # Clean start for testing
    )
    
    print(f"Testing Singularity sandbox with config:")
    print(f"  driver: {config.driver}")
    print(f"  singularity_image: {config.singularity_image}")
    print(f"  job_id: {config.job_id}")
    print()
    
    pool = SlotPool(config)
    
    try:
        print("Starting SlotPool...")
        await pool.start()
        
        stats = pool.get_stats()
        print(f"Pool started! Stats: {stats}")
        print()
        
        # Acquire a slot
        print("Acquiring slot...")
        slot = await pool.acquire("test-trajectory-001")
        print(f"Acquired slot: {slot.slot_id} (alloc={slot.alloc_id[:8]})")
        print()
        
        # Execute a simple command
        print("Executing 'echo hello from singularity'...")
        result = await pool.execute(
            slot,
            "bash",
            {"command": "echo 'Hello from Singularity sandbox!' && uname -a"}
        )
        print(f"Result: {result}")
        print()
        
        # Test file write
        print("Testing file write...")
        write_result = await pool.execute(
            slot,
            "write_file",
            {"path": "test.txt", "content": "Test file from Singularity!"}
        )
        print(f"Write result: {write_result}")
        
        # Test file read
        print("Testing file read...")
        read_result = await pool.execute(
            slot,
            "read_file",
            {"path": "test.txt"}
        )
        print(f"Read result: {read_result}")
        print()
        
        # Release slot
        print("Releasing slot...")
        await pool.release(slot)
        
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        print("\nStopping pool...")
        await pool.stop(purge_job=True)
        print("Pool stopped.")
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_singularity_sandbox())
    sys.exit(0 if success else 1)
