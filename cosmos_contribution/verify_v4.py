# -*- coding: utf-8 -*-
"""COSMOS V4.0 Architecture Verification"""
import sys
import os

# Ensure project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

passed = 0
failed = 0

def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  [PASS] {name}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        failed += 1

print("=" * 60)
print("  COSMOS V4.0 ARCHITECTURE VERIFICATION")
print("=" * 60)

# --- 1. Lyapunov Threshold ---
print("\n--- 1. Lyapunov Thresholds ---")
def t1a():
    from Cosmos.web.cosmosynapse.engine.lyapunov_lock import LYAPUNOV_STABILITY_THRESHOLD
    assert LYAPUNOV_STABILITY_THRESHOLD == 0.60, f"Got {LYAPUNOV_STABILITY_THRESHOLD}"
test("lyapunov_lock STABILITY threshold = 0.60 (response gate)", t1a)

def t1b():
    from Cosmos.web.cosmosynapse.engine.swarm_plasticity import LYAPUNOV_THRESHOLD
    assert LYAPUNOV_THRESHOLD == 0.45, f"Got {LYAPUNOV_THRESHOLD}"
test("swarm_plasticity LEARNING threshold = 0.45 (Hebbian gate)", t1b)

# --- 2. ChaosBuffer ---
print("\n--- 2. ChaosBuffer ---")
def t2a():
    from Cosmos.core.cross_agent_memory import ChaosBuffer
    cb = ChaosBuffer(max_size=50, drift_threshold=0.45)
    assert cb is not None
test("ChaosBuffer import & init", t2a)

def t2b():
    from Cosmos.core.cross_agent_memory import ChaosBuffer
    cb = ChaosBuffer(max_size=50, drift_threshold=0.45)
    for i in range(10):
        cb.inject_entropy(0.5 + i * 0.01)
    stats = cb.get_entropy_stats()
    assert stats["buffer_size"] == 10
    assert stats["drift_threshold"] == 0.45
    assert "current_drift" in stats
test("ChaosBuffer entropy injection & stats", t2b)

def t2c():
    from Cosmos.core.cross_agent_memory import ChaosBuffer
    cb = ChaosBuffer(max_size=10, drift_threshold=0.01)
    # Inject highly variable data to trigger recovery
    import random
    random.seed(42)
    for i in range(20):
        cb.inject_entropy(random.random())
    # With random data, drift should exceed 0.01
    assert cb.should_recover() == True, "Expected recovery trigger"
test("ChaosBuffer recovery trigger on high drift", t2c)

# --- 3. Dark Matter P2P Anchor ---
print("\n--- 3. Dark Matter P2P Anchor ---")
def t3a():
    from Cosmos.web.cosmosynapse.engine.dark_matter_lorenz import DarkMatterLorenz
    dm = DarkMatterLorenz()
    dm.update(0.01)
    anchor = dm.anchor_for_p2p()
    assert "x" in anchor and "y" in anchor and "z" in anchor and "w" in anchor
    assert "sigma" in anchor
test("anchor_for_p2p() serialization", t3a)

def t3b():
    from Cosmos.web.cosmosynapse.engine.dark_matter_lorenz import DarkMatterLorenz
    dm = DarkMatterLorenz()
    peer = {"x": 5.0, "y": 10.0, "z": 15.0, "w": 1.0}
    div = dm.get_divergence(peer)
    assert div > 0, "Divergence should be > 0"
test("get_divergence() calculation", t3b)

def t3c():
    from Cosmos.web.cosmosynapse.engine.dark_matter_lorenz import DarkMatterLorenz
    dm = DarkMatterLorenz()
    before = dm.get_current_state()
    dm.apply_peer_anchor({"x": 100.0, "y": 200.0, "z": 300.0, "w": 50.0}, trust=0.8)
    after = dm.get_current_state()
    assert after["x"] != before["x"], "State should change after merge"
test("apply_peer_anchor() merge", t3c)

# --- 4. Async Event-Driven Life Loop ---
print("\n--- 4. Async Event Loop ---")
def t4a():
    from Cosmos.web.cosmosynapse.engine.cns_core import EventType, CNSEvent
    assert EventType.QUANTUM_TICK.value == "quantum_tick"
    assert EventType.AUDIO_FRAME.value == "audio_frame"
    assert EventType.MEDIAPIPE_UPDATE.value == "mediapipe"
    assert EventType.SHUTDOWN.value == "shutdown"
    e = CNSEvent(event_type=EventType.QUANTUM_TICK, payload={"tick": 1})
    assert e.event_type == EventType.QUANTUM_TICK
test("EventType enum & CNSEvent dataclass", t4a)

def t4b():
    from Cosmos.web.cosmosynapse.engine.cns_core import CosmosCNS
    cns = CosmosCNS()
    assert cns._event_queue is None  # Not started yet
    assert hasattr(cns, 'push_event')
    assert hasattr(cns, '_async_life_loop')
    assert hasattr(cns, '_tick_generator')
    assert hasattr(cns, '_handle_event')
test("CosmosCNS has async event methods", t4b)

# --- 5. Hebbian Transfer Learning ---
print("\n--- 5. Hebbian Transfer Learning ---")
def t5a():
    from Cosmos.web.cosmosynapse.engine.swarm_plasticity import SwarmPlasticity
    sp = SwarmPlasticity(weights_path="test_synaptic_weights.json")
    exported = sp.export_weights()
    assert "weights" in exported
    assert "epoch" in exported
    assert "LOGIC" in exported["weights"]
test("export_weights() serialization", t5a)

def t5b():
    from Cosmos.web.cosmosynapse.engine.swarm_plasticity import SwarmPlasticity
    sp = SwarmPlasticity(weights_path="test_synaptic_weights.json")
    peer = {
        "weights": {
            "LOGIC": {"DeepSeek": 2.0, "Claude": 0.1, "Gemini": 1.5},
            "EMPATHY": {"DeepSeek": 1.0, "Claude": 1.0, "Gemini": 1.0},
            "CREATIVITY": {"DeepSeek": 1.0, "Claude": 1.0, "Gemini": 1.0},
        },
        "epoch": 42,
    }
    before = sp.export_weights()["weights"]["LOGIC"]["DeepSeek"]
    sp.import_peer_weights(peer, trust_factor=0.5)
    after = sp.export_weights()["weights"]["LOGIC"]["DeepSeek"]
    # Weight should have shifted toward peer's value
    assert after != before, "Weights should change after import"
test("import_peer_weights() phi-dampened merge", t5b)

# --- 6. Fault-Tolerant P2P ---
print("\n--- 6. Fault-Tolerant P2P ---")
def t6a():
    from Cosmos.core.swarm.p2p import StateCheckpoint, PeerInfo
    sc = StateCheckpoint(node_id="test", timestamp=0.0, epoch=1)
    assert sc.node_id == "test"
    pi = PeerInfo(id="p1", addr="127.0.0.1", port=9999, capabilities=["NLP"])
    assert pi.missed_pings == 0
test("StateCheckpoint & PeerInfo dataclasses", t6a)

def t6b():
    from Cosmos.core.swarm.p2p import SwarmFabric
    sf = SwarmFabric(node_id="test")
    assert hasattr(sf, 'broadcast_state_checkpoint')
test("SwarmFabric has broadcast_state_checkpoint", t6b)

# --- 7. 12D Knowledge Graph ---
print("\n--- 7. 12D Knowledge Graph ---")
def t7a():
    from Cosmos.memory.unified_knowledge_graph import UnifiedKnowledgeGraph, SynapticConnection
    sc = SynapticConnection(source_id="A", target_id="B", relation_type="related")
    assert sc.effective_weight > 0
    assert sc.resonance_score > 0
test("SynapticConnection properties", t7a)

def t7b():
    from Cosmos.memory.unified_knowledge_graph import UnifiedKnowledgeGraph
    ukg = UnifiedKnowledgeGraph(data_dir="./data/test_ukg_v4")
    physics = {"cst_physics": {"geometric_phase_rad": 0.78}, "dark_matter": {"w": 0.3}}
    c1 = ukg.add_synaptic_connection("AI", "Consciousness", "relates_to", physics, context="LOGIC")
    c2 = ukg.add_synaptic_connection("Consciousness", "Empathy", "requires", physics, context="EMPATHY")
    results = ukg.query_by_phase(phase_center=0.78, phase_range=0.5)
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
test("UnifiedKnowledgeGraph add + query_by_phase", t7b)

def t7c():
    from Cosmos.memory.unified_knowledge_graph import UnifiedKnowledgeGraph
    ukg = UnifiedKnowledgeGraph(data_dir="./data/test_ukg_v4b")
    physics = {"cst_physics": {"geometric_phase_rad": 0.78}}
    ukg.add_synaptic_connection("X", "Y", "links", physics)
    topo = ukg.get_12d_topology()
    assert topo["total_nodes"] == 2
    assert topo["total_edges"] == 1
test("get_12d_topology() export", t7c)

# --- 8. Web Search Integration ---
print("\n--- 8. Web Search for All Models ---")
def t8a():
    from Cosmos.tools.web_search import should_search, format_search_context
    assert should_search("search for quantum computing") == True
    assert should_search("hello") == False
    assert should_search("what is the latest news about AI") == True
    ctx = format_search_context([{"title": "Test", "snippet": "Snippet", "url": "http://x.com"}])
    assert "WEB SEARCH RESULTS" in ctx
test("should_search() detection + format_search_context()", t8a)

# --- Summary ---
print("\n" + "=" * 60)
total = passed + failed
print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
if failed == 0:
    print("  ALL COSMOS V4.0 VERIFICATION TESTS PASSED")
else:
    print("  SOME TESTS FAILED")
print("=" * 60)

# Cleanup test files
import shutil
for d in ["./data/test_ukg_v4", "./data/test_ukg_v4b"]:
    if os.path.exists(d):
        shutil.rmtree(d, ignore_errors=True)
if os.path.exists("cst_synaptic_weights.json"):
    os.remove("cst_synaptic_weights.json")
