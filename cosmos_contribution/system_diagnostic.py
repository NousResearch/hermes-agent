import os
import sys
import torch
from loguru import logger

# Add project root to path
sys.path.insert(0, r"D:\Cosmos")

def run_diagnostics():
    print("=" * 50)
    print("COSMOS 12D SYSTEM DIAGNOSTIC")
    print("=" * 50)

    # 1. GPU / Torch Check
    print("\n[1/3] HARDWARE ACCELERATION")
    print(f"Torch Version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"Active GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"VRAM Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    else:
        print("!! WARNING: System running on CPU fallback !!")

    # 2. Quantum Bridge Check
    print("\n[2/3] QUANTUM ENTANGLEMENT BRIDGE")
    try:
        # Import bridge
        from Cosmos.core.quantum_bridge import get_quantum_bridge
        from dotenv import load_dotenv
        load_dotenv(r"D:\Cosmos\.env")
        
        token = os.getenv("IBM_QUANTUM_TOKEN")
        bridge = get_quantum_bridge(token)
        
        print(f"Bridge Object Initialized: True")
        print(f"Connected to IBMQ: {bridge.connected}")
        
        if bridge.connected:
            print(f"Active Backend: {bridge.backend.name if bridge.backend else 'None'}")
            # Try a quick entropy pull
            entropy = bridge.get_entropy()
            print(f"Real-time Entropy Sample: {entropy:.4f}")
            print("Status: COHERENT")
        else:
            print("Status: RUNNING IN SIMULATION MODE (No active connection)")
    except Exception as e:
        print(f"!! ERROR: Quantum Bridge failure: {e} !!")

    # 3. Memory & Swarm Readiness
    print("\n[3/3] SWARM ORCHESTRATION")
    try:
        from Cosmos.web.cosmosynapse.engine.cosmos_swarm_orchestrator import CosmosSwarmOrchestrator
        orch = CosmosSwarmOrchestrator()
        print("Orchestrator Instance: Ready")
        print(f"Hebbian Weights Loaded: {len(orch.model_weights)} models tracked")
    except Exception as e:
        print(f"!! ERROR: Orchestrator failed to initialize: {e} !!")

    print("\n" + "=" * 50)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    run_diagnostics()
