#!/usr/bin/env python3
#!/usr/bin/env python3
"""Argus Simulation Demo

Demonstrates the complete workflow:
1. Initialize dummy database
2. Generate test scenarios
3. Run detection validation
4. Display results

Paths configured by tests/argus/conftest.py when run via pytest.
For direct execution, ensure parent directory is in path.
"""

from dummy_db import reset_dummy_database
from argus_simulator import ArgusSimulator
from argus_test_harness import ArgusTestHarness


def main():
    print("=" * 60)
    print("ARGUS SIMULATION DEMO")
    print("=" * 60)
    
    # Step 1: Reset database
    print("\n[1] Initializing dummy database...")
    reset_dummy_database()
    
    # Step 2: Generate all test scenarios
    print("\n[2] Generating test scenarios...")
    sim = ArgusSimulator()
    
    scenarios = [
        ("repeat_tool_calls", "5 identical read_file calls"),
        ("repeat_commands", "4 identical 'ls -la' commands"),
        ("stuck_loop", "Repeating A,B,C pattern 2x"),
        ("no_file_changes", "3 writes + 3 patches with no changes"),
        ("error_cascade", "4 consecutive tool failures"),
    ]
    
    session_ids = []
    for scenario, desc in scenarios:
        print(f"    - {scenario}: {desc}")
        sim.run_scenario(scenario)
        session_ids.append(f"test_{scenario.replace('_', '')}" if scenario == 'stuck_loop' 
                          else f"test_{scenario.split('_')[0]}_{scenario.split('_')[1] if len(scenario.split('_')) > 1 else ''}".rstrip('_'))
    
    # Use simpler approach - just run all
    sim.close()
    reset_dummy_database()
    sim = ArgusSimulator()
    session_ids = sim.run_all_scenarios()
    sim.close()
    
    print(f"\n    Generated {len(session_ids)} test sessions")
    
    # Step 3: Validate detections
    print("\n[3] Running entropy detection...")
    harness = ArgusTestHarness()
    
    for sid in session_ids:
        detections = harness.detect_entropy_for_session(sid)
        if detections:
            print(f"\n    {sid}:")
            for d in detections:
                print(f"      [{d['severity']:8}] {d['entropy_type']}")
        else:
            print(f"    {sid}: No entropy detected")
    
    # Step 4: Summary
    print("\n[4] Summary")
    print("-" * 40)
    
    for sid in session_ids:
        summary = harness.get_session_summary(sid)
        print(f"  {summary['session_id'][:25]:25} | "
              f"Tools: {summary['tool_calls']:2} | "
              f"Cmds: {summary['terminal_commands']:2} | "
              f"Entropy: {summary['entropy_detections']}")
    
    harness.close()
    
    print("\n" + "=" * 60)
    print("Demo complete. Database: dummy_argus.db")
    print("=" * 60)


if __name__ == "__main__":
    main()
