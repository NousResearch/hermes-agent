
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Mock necessary environment
os.environ["COSMOS_SKIP_TORCH"] = "1"

try:
    from Cosmos.web.cosmosynapse.engine.synaptic_field import SynapticField
    
    print("--- 12D Dimension Mapping Verification ---")
    field = SynapticField()
    field.initialize_12d_mapping()
    
    dimensions = field.get_12d_dimensions()
    print(f"Total Dimensions: {len(dimensions)}")
    
    for i, dim in enumerate(dimensions):
        print(f"Dimension {i+1}: {dim}")
    
    if len(dimensions) == 12:
        print("\nSUCCESS: 12D Mapping initialized correctly.")
    else:
        print(f"\nFAILURE: Expected 12 dimensions, found {len(dimensions)}.")

except Exception as e:
    print(f"Error during verification: {e}")
    import traceback
    traceback.print_exc()
