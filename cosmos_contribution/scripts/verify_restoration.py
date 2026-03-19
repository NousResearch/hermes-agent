"""
Cosmos System Restoration — Smoke Test
========================================

Validates that all integration points from the restoration plan are
wired correctly.  Does NOT require Ollama, IBM Quantum, or a running
server.

Run:
    cd D:\\Cosmos
    python scripts/verify_restoration.py
"""

import sys
import os

# Ensure Cosmos package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  — {detail}")


# ──────────────────────────────────────────────
# 1. Planetary imports work
# ──────────────────────────────────────────────
print("\n=== 1. Planetary Package Imports ===")
try:
    from Cosmos.core.memory.planetary import PlanetaryMemory, PlanetaryAudioShard, get_audio_shard
    check("PlanetaryMemory importable", True)
    check("PlanetaryAudioShard importable", True)
except ImportError as e:
    check("PlanetaryMemory importable", False, str(e))
    check("PlanetaryAudioShard importable", False, str(e))

# ──────────────────────────────────────────────
# 2. PlanetaryMemory instantiation
# ──────────────────────────────────────────────
print("\n=== 2. PlanetaryMemory Instantiation ===")
try:
    pm = PlanetaryMemory(use_p2p=False)
    check("PlanetaryMemory(use_p2p=False)", pm is not None)
    check("Has local_skills dict", isinstance(pm.local_skills, dict))
except Exception as e:
    check("PlanetaryMemory(use_p2p=False)", False, str(e))

# ──────────────────────────────────────────────
# 3. MemorySystem has .planetary attribute
# ──────────────────────────────────────────────
print("\n=== 3. MemorySystem Planetary Integration ===")
try:
    from Cosmos.memory.memory_system import MemorySystem
    ms = MemorySystem(data_dir="./data")
    check("MemorySystem.planetary is not None", ms.planetary is not None,
          "PlanetaryMemory was not initialized in MemorySystem.__init__")

    stats = ms.get_stats()
    check("get_stats() includes 'planetary'", "planetary" in stats,
          f"Keys: {list(stats.keys())}")
except Exception as e:
    check("MemorySystem init", False, str(e))

# ──────────────────────────────────────────────
# 4. Quantum bridge singleton
# ──────────────────────────────────────────────
print("\n=== 4. Quantum Bridge Singleton ===")
try:
    from Cosmos.core.quantum_bridge import get_quantum_bridge
    bridge = get_quantum_bridge()
    check("get_quantum_bridge() returns instance", bridge is not None)
    check("Has entropy_buffer attribute", hasattr(bridge, "entropy_buffer"))
    check("Has _hermes_predict_buffer_demand", hasattr(bridge, "_hermes_predict_buffer_demand"))
except Exception as e:
    check("Quantum bridge import", False, str(e))

# ──────────────────────────────────────────────
# 5. EvolutionLoop has _entropy_heartbeat_loop
# ──────────────────────────────────────────────
print("\n=== 5. EvolutionLoop Entropy Heartbeat ===")
try:
    from Cosmos.core.evolution_loop import EvolutionLoop
    el = EvolutionLoop()
    check("EvolutionLoop._entropy_heartbeat_loop exists",
          hasattr(el, "_entropy_heartbeat_loop") and callable(el._entropy_heartbeat_loop))
except Exception as e:
    check("EvolutionLoop import", False, str(e))

# ──────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  RESULTS:  {PASS} passed,  {FAIL} failed")
print(f"{'='*50}")

sys.exit(1 if FAIL > 0 else 0)
