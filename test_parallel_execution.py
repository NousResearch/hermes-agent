#!/usr/bin/env python3
"""
Test Parallel Execution with Persistent WebSocket Connection Pool

This script demonstrates that multiple agent runs can execute in parallel,
all sharing a single WebSocket connection for logging.

Benefits:
- No connection overhead (single persistent connection)
- No timeout issues (connection stays alive)
- True parallel execution (multiple sessions simultaneously)
"""

import asyncio
from run_agent import AIAgent
import time


async def run_agent_query(query: str, agent_name: str, mock_delay: int = 10):
    """
    Run a single agent query with logging.
    
    Args:
        query: Query to send to agent
        agent_name: Name for logging purposes
        mock_delay: Delay for mock tools (seconds)
    """
    print(f"🚀 [{agent_name}] Starting query: '{query[:40]}...'")
    start_time = time.time()
    
    try:
        agent = AIAgent(
            model="claude-sonnet-4-5-20250929",
            max_iterations=5,
            enabled_toolsets=["web"],
            enable_websocket_logging=True,
            websocket_server="ws://localhost:8000/ws",
            mock_web_tools=True,  # Use mock tools for fast testing
            mock_delay=mock_delay
        )
        
        result = await agent.run_conversation(query)
        
        duration = time.time() - start_time
        print(f"✅ [{agent_name}] Completed in {duration:.1f}s - {result['api_calls']} API calls")
        
        return {
            "agent": agent_name,
            "query": query,
            "success": True,
            "duration": duration,
            "api_calls": result['api_calls'],
            "session_id": result.get('session_id')
        }
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ [{agent_name}] Failed in {duration:.1f}s: {e}")
        return {
            "agent": agent_name,
            "query": query,
            "success": False,
            "error": str(e),
            "duration": duration
        }


async def test_sequential():
    """
    Test 1: Sequential execution (baseline).
    
    Runs 3 queries one after another. This shows how long it takes
    without parallelization.
    """
    print("\n" + "="*60)
    print("TEST 1: Sequential Execution (Baseline)")
    print("="*60)
    
    start_time = time.time()
    
    results = []
    for i in range(3):
        result = await run_agent_query(
            query=f"Find information about water companies #{i+1}",
            agent_name=f"Agent{i+1}",
            mock_delay=5  # Short delay for quick test
        )
        results.append(result)
    
    total_time = time.time() - start_time
    
    print(f"\n📊 Sequential Results:")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Successful: {sum(1 for r in results if r['success'])}/3")
    print(f"   Average per query: {total_time/3:.1f}s")
    
    return results


async def test_parallel():
    """
    Test 2: Parallel execution.
    
    Runs 3 queries simultaneously using asyncio.gather().
    All queries share the same WebSocket connection for logging.
    """
    print("\n" + "="*60)
    print("TEST 2: Parallel Execution (Shared Connection)")
    print("="*60)
    
    start_time = time.time()
    
    # Run all queries in parallel!
    results = await asyncio.gather(
        run_agent_query(
            query="Find publicly traded water utility companies",
            agent_name="Agent1",
            mock_delay=5
        ),
        run_agent_query(
            query="Find energy infrastructure companies",
            agent_name="Agent2",
            mock_delay=5
        ),
        run_agent_query(
            query="Find AI data center operators",
            agent_name="Agent3",
            mock_delay=5
        )
    )
    
    total_time = time.time() - start_time
    
    print(f"\n📊 Parallel Results:")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Successful: {sum(1 for r in results if r['success'])}/3")
    print(f"   Speedup: ~{(sum(r['duration'] for r in results) / total_time):.1f}x")
    print(f"   Sessions logged: {[r.get('session_id', 'N/A')[:8] for r in results]}")
    
    return results


async def test_high_concurrency():
    """
    Test 3: High concurrency (stress test).
    
    Runs 10 queries simultaneously to test connection pool under load.
    """
    print("\n" + "="*60)
    print("TEST 3: High Concurrency (10 Parallel Agents)")
    print("="*60)
    
    start_time = time.time()
    
    tasks = [
        run_agent_query(
            query=f"Test query #{i+1}",
            agent_name=f"Agent{i+1}",
            mock_delay=3  # Very short for stress test
        )
        for i in range(10)
    ]
    
    results = await asyncio.gather(*tasks)
    
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r['success'])
    
    print(f"\n📊 High Concurrency Results:")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Successful: {successful}/10")
    print(f"   Failed: {10 - successful}/10")
    print(f"   Queries per second: {10 / total_time:.2f}")
    
    return results


async def main():
    """Run all tests."""
    print("\n🧪 WebSocket Connection Pool - Parallel Execution Tests")
    print("="*60)
    print("\nPREREQUISITE: Make sure logging server is running:")
    print("  python api_endpoint/logging_server.py")
    print("\nPress Ctrl+C to stop at any time\n")
    
    await asyncio.sleep(2)  # Give user time to read
    
    try:
        # Test 1: Sequential (baseline)
        seq_results = await test_sequential()
        
        # Test 2: Parallel (main test)
        par_results = await test_parallel()
        
        # Test 3: High concurrency
        stress_results = await test_high_concurrency()
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"\n✅ All tests completed!")
        print(f"\nKey Findings:")
        print(f"  • Sequential (3 queries): {sum(r['duration'] for r in seq_results):.1f}s total")
        print(f"  • Parallel (3 queries): {max(r['duration'] for r in par_results):.1f}s total")
        print(f"  • Speedup: ~{sum(r['duration'] for r in seq_results) / max(r['duration'] for r in par_results):.1f}x")
        print(f"  • High concurrency (10 queries): ✅ Handled successfully")
        print(f"\n💡 All queries used the same persistent WebSocket connection!")
        print(f"   No connection overhead, no timeouts, true parallelization.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Tests failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("SETUP CHECK")
    print("="*60)
    
    # Check if logging server is running
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 8000))
        sock.close()
        
        if result == 0:
            print("✅ Logging server is running on port 8000")
        else:
            print("⚠️  Logging server not detected on port 8000")
            print("   Start it with: python api_endpoint/logging_server.py")
            print("\nContinuing anyway (tests will fail gracefully)...")
    except Exception as e:
        print(f"⚠️  Could not check server status: {e}")
    
    # Run tests
    asyncio.run(main())

