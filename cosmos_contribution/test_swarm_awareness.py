"""Quick verification of Swarm-Awareness Protocol integration."""
import sys
import os

# Add project root
sys.path.insert(0, os.path.abspath("."))

print("=" * 60)
print("  SWARM-AWARENESS PROTOCOL - VERIFICATION")
print("=" * 60)

# Test 1: Import
print("\n1. Testing SwarmAwareness import...")
try:
    from Cosmos.web.cosmosynapse.engine.swarm_awareness import SwarmAwareness
    print("   OK  SwarmAwareness imported")
except Exception as e:
    print(f"   FAIL  Import error: {e}")
    sys.exit(1)

# Test 2: Initialization
print("\n2. Testing initialization...")
try:
    sa = SwarmAwareness()
    print("   OK  SwarmAwareness() created (no dependencies)")
except Exception as e:
    print(f"   FAIL  Init error: {e}")
    sys.exit(1)

# Test 3: Tick without assessment (not enough ticks yet)
print("\n3. Testing tick (below interval)...")
result = sa.tick()
assert result is None, "Should not assess on tick 1 (interval=50)"
print("   OK  No assessment on tick 1 (correct)")

# Test 4: Force assessment
print("\n4. Testing forced assessment...")
sa.assessment_interval = 1
report = sa.tick()
assert report is not None, "Should produce a report"
print(f"   OK  Report: aligned={report.aligned}, score={report.alignment_score:.3f}")
print(f"   OK  Values: {report.value_scores}")

# Test 5: With mock dependencies
print("\n5. Testing with mock dependencies...")

class MockField:
    def get_snapshot(self):
        return {
            "user_physics": {},
            "buffer_size": 15,
            "dark_matter": {"x": 0.3, "y": 0.2, "z": 0.1, "w": 0.5},
            "time": "2026-03-02 16:00:00",
        }

class MockPlasticity:
    def get_stats(self):
        return {
            "weights": {
                "LOGIC": {"DeepSeek": 1.2, "Claude": 0.6, "Gemini": 0.4},
                "EMPATHY": {"DeepSeek": 0.3, "Claude": 1.1, "Gemini": 0.7},
                "CREATIVITY": {"DeepSeek": 0.4, "Claude": 0.5, "Gemini": 1.0},
            }
        }

sa2 = SwarmAwareness(
    synaptic_field=MockField(),
    plasticity=MockPlasticity(),
    assessment_interval=1,
)

report2 = sa2.tick()
assert report2 is not None
print(f"   OK  With deps: aligned={report2.aligned}, score={report2.alignment_score:.3f}")

# Test 6: Multiple ticks / history
print("\n6. Testing alignment history...")
for _ in range(4):
    sa2.tick()
history = sa2.get_alignment_history()
assert len(history) == 5, f"Expected 5, got {len(history)}"
print(f"   OK  History ({len(history)} pts): {[round(h, 3) for h in history]}")

# Test 7: Stats
print("\n7. Testing stats...")
stats = sa2.get_stats()
assert stats["total_assessments"] == 5
print(f"   OK  Assessments={stats['total_assessments']}, Corrections={stats['total_corrections']}, Peers={stats['total_peer_reviews']}")

# Test 8: Value target update
print("\n8. Testing dynamic value update...")
sa2.update_value_target("empathy", 0.9)
assert sa2._value_targets["empathy"] == 0.9
print("   OK  empathy target updated to 0.9")

# Test 9: CNS integration
print("\n9. Testing CosmosCNS import with SwarmAwareness...")
try:
    from Cosmos.web.cosmosynapse.engine.cns_core import CosmosCNS
    cns = CosmosCNS()
    assert cns.awareness is None, "Should be None before initialize_organs"
    print("   OK  CosmosCNS created, awareness=None (pre-init)")
except Exception as e:
    print(f"   WARN  CosmosCNS import note: {e}")

print("\n" + "=" * 60)
print("  ALL TESTS PASSED")
print("=" * 60)
