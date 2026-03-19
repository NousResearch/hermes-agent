
import sys
import os
from dotenv import load_dotenv

# Load .env explicitly
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from cosmos.core.quantum_bridge import get_quantum_bridge, QISKIT_AVAILABLE, QISKIT_ERROR
    print(f"QISKIT_AVAILABLE: {QISKIT_AVAILABLE}")
    if not QISKIT_AVAILABLE:
        print(f"QISKIT_ERROR: {QISKIT_ERROR}")
    
    # Try to connect with a dummy token if none exists, or checks env
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    print(f"Token present in env: {bool(token)}")
    
    bridge = get_quantum_bridge(token)
    print(f"Bridge initialized. Connected: {bridge.connected}")
    if bridge.last_error:
        print(f"Bridge Last Error: {bridge.last_error}")
        
    # Try to get entropy
    entropy = bridge.get_entropy()
    print(f"Entropy sample: {entropy}")
    
except Exception as e:
    print(f"Test Failed: {e}")
    import traceback
    traceback.print_exc()
