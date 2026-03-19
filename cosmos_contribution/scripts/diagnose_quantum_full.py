
import sys
import os
import traceback
from pathlib import Path

# Setup Path - Need both project root and cosmos subdir
project_root = Path(__file__).parent.parent.parent
cosmos_root = project_root / "cosmos"
sys.path.append(str(project_root))
sys.path.append(str(cosmos_root))

print(f"Added to path: {project_root}")
print(f"Added to path: {cosmos_root}")

print("=== QUANTUM DIAGNOSTIC START ===")
print(f"Python: {sys.version}")
print(f"CWD: {os.getcwd()}")

# 1. Check Packages
print("\n--- Package Check ---")
try:
    import qiskit
    print(f"Qiskit version: {qiskit.__version__}")
except ImportError:
    print("Qiskit NOT INSTALLED")

try:
    import qiskit_ibm_runtime
    print(f"Qiskit Runtime version: {qiskit_ibm_runtime.__version__}")
except ImportError:
    print("Qiskit IBM Runtime NOT INSTALLED")

# 2. Check imports in Bridge
print("\n--- Import Check ---")
try:
    from cosmos.core.quantum_bridge import QuantumEntanglementBridge, get_quantum_bridge, QISKIT_AVAILABLE, QISKIT_ERROR
    print(f"Bridge Loaded. QISKIT_AVAILABLE={QISKIT_AVAILABLE}")
    print(f"QISKIT_ERROR={QISKIT_ERROR}")
except Exception as e:
    print(f"FAILED to import Bridge: {e}")
    traceback.print_exc()

# 3. Simulate Connection
print("\n--- Connection Simulation ---")
token = os.getenv("IBM_QUANTUM_TOKEN")
# Try .env if missing
if not token:
    env_path = Path(__file__).parent.parent.parent / "cosmos" / ".env" # Adjust path based on user structure?
    # Actually root is parent.parent.parent usually for scripts/
    # Let's try standard location
    env_path = Path(".env")
    if not env_path.exists():
         env_path = Path("cosmos/.env")
    
    if env_path.exists():
        print(f"Reading .env from {env_path}")
        with open(env_path, "r") as f:
            for line in f:
                if line.startswith("IBM_QUANTUM_TOKEN="):
                    token = line.strip().split("=", 1)[1]
                    break

print(f"Token to use: {token[:5]}..." if token else "Token: NONE")

try:
    # Use the getter
    bridge = get_quantum_bridge()
    print(f"Bridge Instance: {bridge}")
    print(f"Initial State: Connected={bridge.connected}, Error={bridge.last_error}")
    
    if token:
        print("Setting token and connecting...")
        bridge.api_token = token
        bridge._connect()
        print(f"Post-Connect State: Connected={bridge.connected}")
        print(f"Post-Connect Error: {bridge.last_error}")
        if bridge.backend:
            print(f"Backend: {bridge.backend.name}")
    else:
        print("Skipping connect (No Token)")

except Exception as e:
    print(f"Simulation Failed: {e}")
    traceback.print_exc()

print("=== QUANTUM DIAGNOSTIC END ===")
