import sys
import os

# Add project root
sys.path.append(r"D:\Cosmos\Cosmos")

print("Testing direct import of quantum_bridge...")
try:
    from core.quantum_bridge import get_quantum_bridge
    print("SUCCESS: core.quantum_bridge imported.")
except MemoryError as e:
    print(f"CRITICAL: MemoryError during import: {e}")
except Exception as e:
    print(f"FAILURE: {type(e).__name__}: {e}")

print("\nTesting import with 'cosmos' prefix...")
sys.path.append(r"D:\Cosmos")
try:
    # Rename folder temporarily if needed? No, let's just try.
    # The local folder is 'Cosmos'
    from Cosmos.core.quantum_bridge import get_quantum_bridge
    print("SUCCESS: Cosmos.core.quantum_bridge imported.")
except Exception as e:
    print(f"FAILURE: {type(e).__name__}: {e}")
