#!/usr/bin/env python3
"""Test script to demonstrate timestamp validation integration."""

import json
import tempfile
from pathlib import Path
import sys

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from qualification_packet_timestamp_validator import (
    validate_qualification_packet_timestamps,
    ensure_qualification_packet_timestamps,
)


def test_complete_packet():
    """Test a qualification packet with complete timestamps."""
    print("Test 1: Complete qualification packet")
    packet = {
        "schema_version": "clawd.model_qualification_packet.v1",
        "qualification_id": "test_complete",
        "evaluated_at": "2026-04-03T00:00:00Z",
        "scorecard": {
            "scored_at": "2026-04-03T00:01:00Z",
            "cost": {
                "provider_evidence_updated_at": "2026-04-03T00:02:00Z"
            }
        }
    }
    
    is_valid, missing, invalid = validate_qualification_packet_timestamps(packet)
    print(f"  Valid: {is_valid}")
    print(f"  Missing: {missing}")
    print(f"  Invalid: {invalid}")
    
    if is_valid and not missing and not invalid:
        print("  ✓ PASS: Complete packet validated successfully")
    else:
        print("  ✗ FAIL: Complete packet should be valid")
        return False
    
    return True


def test_incomplete_packet():
    """Test a qualification packet with missing timestamps."""
    print("\nTest 2: Incomplete qualification packet (missing timestamps)")
    packet = {
        "schema_version": "clawd.model_qualification_packet.v1",
        "qualification_id": "test_incomplete",
        "evaluated_at": "2026-04-03T00:00:00Z"
        # Missing scorecard.scored_at and scorecard.cost.provider_evidence_updated_at
    }
    
    is_valid, missing, invalid = validate_qualification_packet_timestamps(packet)
    print(f"  Valid: {is_valid}")
    print(f"  Missing: {missing}")
    print(f"  Invalid: {invalid}")
    
    if not is_valid and missing and not invalid:
        print(f"  ✓ PASS: Incomplete packet correctly identified as missing {missing}")
    else:
        print("  ✗ FAIL: Incomplete packet should be invalid")
        return False
    
    return True


def test_backfill_allowed():
    """Test backfilling missing timestamps when allowed."""
    print("\nTest 3: Backfill missing timestamps (allowed)")
    packet = {
        "schema_version": "clawd.model_qualification_packet.v1",
        "qualification_id": "test_backfill_allowed",
        "evaluated_at": "2026-04-03T00:00:00Z"
    }
    
    try:
        result, was_backfilled, backfilled_fields = ensure_qualification_packet_timestamps(
            packet, allow_backfill=True
        )
        print(f"  Backfilled: {was_backfilled}")
        print(f"  Backfilled fields: {backfilled_fields}")
        
        # Validate the result
        is_valid, missing, invalid = validate_qualification_packet_timestamps(result)
        print(f"  Result valid: {is_valid}")
        print(f"  Result missing: {missing}")
        print(f"  Result invalid: {invalid}")
        
        if was_backfilled and is_valid and not missing and not invalid:
            print("  ✓ PASS: Backfill successful, packet now valid")
        else:
            print("  ✗ FAIL: Backfill should make packet valid")
            return False
            
    except Exception as e:
        print(f"  ✗ FAIL: Exception during backfill: {e}")
        return False
    
    return True


def test_backfill_not_allowed():
    """Test that backfill fails when not allowed."""
    print("\nTest 4: Backfill missing timestamps (not allowed)")
    packet = {
        "schema_version": "clawd.model_qualification_packet.v1",
        "qualification_id": "test_backfill_not_allowed",
        "evaluated_at": "2026-04-03T00:00:00Z"
    }
    
    try:
        result, was_backfilled, backfilled_fields = ensure_qualification_packet_timestamps(
            packet, allow_backfill=False
        )
        print(f"  ✗ FAIL: Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"  ✓ PASS: Correctly raised ValueError: {e}")
        return True
    except Exception as e:
        print(f"  ✗ FAIL: Wrong exception type: {type(e).__name__}: {e}")
        return False


def test_integration_with_model_rollout_gate_runner():
    """Test integration with model rollout gate runner concepts."""
    print("\nTest 5: Integration with gate runner concepts")
    
    # Simulate what the gate runner would do
    packet = {
        "schema_version": "clawd.model_qualification_packet.v1",
        "qualification_id": "test_integration",
        "evaluated_at": "2026-04-03T00:00:00Z",
        "model": {
            "model_key": "test/model",
            "provider": "test",
            "route_class": "TEST"
        },
        "qualification": {
            "checklist": [
                {"check_id": "schema_contract_valid", "status": "pass", "evidence_ref": "test"}
            ]
        }
    }
    
    # Case 1: Validate timestamps (fail-closed)
    print("  Case 1: Validate timestamps (fail-closed)")
    is_valid, missing, invalid = validate_qualification_packet_timestamps(packet)
    if not is_valid:
        print(f"    ✓ Packet would be blocked: missing {missing}")
    else:
        print(f"    ✗ Packet should be blocked")
        return False
    
    # Case 2: Allow backfill
    print("  Case 2: Allow backfill")
    try:
        result, was_backfilled, backfilled_fields = ensure_qualification_packet_timestamps(
            packet, allow_backfill=True
        )
        is_valid, missing, invalid = validate_qualification_packet_timestamps(result)
        if is_valid and was_backfilled:
            print(f"    ✓ Packet backfilled and now valid: {backfilled_fields}")
        else:
            print(f"    ✗ Packet should be valid after backfill")
            return False
    except Exception as e:
        print(f"    ✗ Exception during backfill: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Timestamp Validation Integration Tests")
    print("=" * 60)
    
    tests = [
        test_complete_packet,
        test_incomplete_packet,
        test_backfill_allowed,
        test_backfill_not_allowed,
        test_integration_with_model_rollout_gate_runner,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ TEST CRASHED: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())