
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath("c:/Users/corys/The-Cosmic-Davis-12D-Hebbian-Transformer--1/cosmos"))

try:
    print("Testing Multimodal Import...")
    from cosmos.core.multimodal import get_multimodal_system
    
    print("Initializing System...")
    system = get_multimodal_system()
    
    if system:
        print("✅ Multimodal System Initialized Successfully")
        print(f"Memory Size: {len(system.memory)}")
        print(f"Emotional State: {system.emotional_state}")
    else:
        print("❌ Failed to initialize system (returned None)")

except Exception as e:
    print(f"❌ Error during verification: {e}")
    import traceback
    traceback.print_exc()
