#!/usr/bin/env python3
"""
Test script for Singularity sandbox job creation.

This tests the create_sandbox_job function with driver="singularity".
"""

import asyncio
import sys
import json
import importlib.util

# Load atropos.nomad.client directly to bypass __init__.py
spec = importlib.util.spec_from_file_location(
    "nomad_client",
    "/root/Hermes-Agent/atropos/nomad/client.py"
)
nomad_client = importlib.util.module_from_spec(spec)
sys.modules["nomad_client"] = nomad_client
spec.loader.exec_module(nomad_client)

NomadClient = nomad_client.NomadClient
create_sandbox_job = nomad_client.create_sandbox_job


async def test_singularity_job():
    """Test Singularity job creation and submission to Nomad."""
    
    job_id = "test-singularity-sandbox"
    sif_path = "/root/Hermes-Agent/atropos/atropos-sandbox.sif"
    
    print("=== Singularity Sandbox Job Test ===\n")
    
    # Create job spec for Singularity
    print("Creating Singularity job spec...")
    job_spec = create_sandbox_job(
        job_id=job_id,
        driver="singularity",
        singularity_image=sif_path,
        slots_per_container=5,
        count=1,
        cpu=500,
        memory=512,
    )
    
    # Print task driver and config
    task = job_spec["TaskGroups"][0]["Tasks"][0]
    print(f"  Driver: {task['Driver']}")
    print(f"  Config: {json.dumps(task['Config'], indent=4)}")
    print()
    
    # Test submission to Nomad
    print("Connecting to Nomad...")
    client = NomadClient(address="http://localhost:4646")
    
    try:
        # Check health
        healthy = await client.is_healthy()
        print(f"  Nomad healthy: {healthy}")
        
        if not healthy:
            print("❌ Nomad is not reachable!")
            return False
        
        # Purge any existing job
        print(f"\nPurging existing job '{job_id}'...")
        await client.stop_job(job_id, purge=True)
        
        # Submit job
        print(f"Submitting Singularity job '{job_id}'...")
        result = await client.submit_job(job_spec)
        print(f"  Result: {result}")
        
        if "error" in result:
            print(f"❌ Job submission failed: {result}")
            return False
        
        # Wait for allocation
        print("\nWaiting for allocation (10 seconds)...")
        await asyncio.sleep(10)
        
        # Check allocations
        allocs = await client.get_job_allocations(job_id)
        print(f"Allocations: {len(allocs)}")
        for alloc in allocs:
            print(f"  - {alloc.id[:8]} status={alloc.status.value} http={alloc.http_address}")
            
            # Get detailed info
            detail = await client.get_allocation(alloc.id)
            if detail:
                task_states = detail.get("TaskStates", {})
                for task_name, state in task_states.items():
                    events = state.get("Events", [])[-3:]
                    print(f"    Task '{task_name}': {[e.get('Type') for e in events]}")
        
        # Check if any are running
        running = [a for a in allocs if a.status.value == "running"]
        if running:
            print(f"\n✅ Job running! {len(running)} allocation(s)")
            
            # Try to reach the sandbox server
            if running[0].http_address:
                import aiohttp
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"{running[0].http_address}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                            print(f"  Health check: {resp.status} - {await resp.text()}")
                except Exception as e:
                    print(f"  Health check failed: {e}")
        else:
            print("\n⚠️ No running allocations yet (may still be starting)")
            
        return True
        
    finally:
        # Don't cleanup - leave running for debugging
        print(f"\n[Leaving job '{job_id}' running for debugging]")
        print(f"  View logs: nomad alloc logs -job {job_id}")
        print(f"  Cleanup: nomad job stop -purge {job_id}")
        await client.close()
        print("Done!")


if __name__ == "__main__":
    success = asyncio.run(test_singularity_job())
    sys.exit(0 if success else 1)
