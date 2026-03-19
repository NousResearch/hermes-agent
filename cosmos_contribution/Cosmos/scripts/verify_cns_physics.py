import asyncio
import sys
import os
import numpy as np
import logging
import time

# Set up logging to see CNS output
logging.basicConfig(level=logging.INFO)

# Add project root parent to path so 'import Cosmos' works
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    print("Step 1: Importing SynapticField...")
    from Cosmos.web.cosmosynapse.engine.synaptic_field import SynapticField
    print("SUCCESS: SynapticField imported.")
    
    print("Step 2: Importing CosmosCNS...")
    from Cosmos.web.cosmosynapse.engine.cosmos_cns import CosmosCNS
    print("SUCCESS: CosmosCNS imported.")
except (ImportError, ModuleNotFoundError) as e:
    print(f"FAILURE: Import error: {e}")
    # Try alternate relative import
    try:
        from web.cosmosynapse.engine.cosmos_cns import CosmosCNS
        from web.cosmosynapse.engine.synaptic_field import SynapticField
        print("SUCCESS: Imported via relative path.")
    except Exception as e2:
        print(f"FAILURE: Alternate import failed: {e2}")
        sys.exit(1)
except MemoryError as e:
    print(f"CRITICAL: MemoryError during import: {e}")
    print("This often indicates a circular dependency or a corrupted pyc file.")
    sys.exit(1)
except Exception as e:
    print(f"FAILURE: Unexpected error: {type(e).__name__}: {e}")
    sys.exit(1)

async def test_cns_physics():
    print("\n--- Initializing Cosmos CNS ---")
    field = SynapticField()
    cns = CosmosCNS()
    cns.field = field
    
    # Manually trigger organ initialization
    cns.initialize_organs()
    
    # Check if organs loaded
    organs = ["quantum", "emeth", "plasticity", "awareness", "surgeon"]
    for organ in organs:
        status = "LOADED" if getattr(cns, organ) is not None else "MISSING"
        print(f"Organ {organ:12}: {status}")

    print("\n--- Running Physics Dry Run ---")
    # Set initial positions to non-zero values to ensure gravity calculation
    cns.swarm_positions = np.random.rand(11, 12).astype(np.float32) * 10.0
    cns.swarm_velocities = np.zeros((11, 12), dtype=np.float32)
    
    # Run a few ticks manually
    print("Calculating initial gravity forces...")
    # njit functions are jitted on first call
    forces = cns._calculate_synaptic_gravity(cns.swarm_positions, cns.swarm_mass)
    
    force_magnitude = np.linalg.norm(forces)
    print(f"Initial Swarm Force Magnitude: {force_magnitude:.4e}")
    
    if force_magnitude > 0:
        print("SUCCESS: Synaptic Gravity forces detected.")
    else:
        print("FAILURE: No gravitational forces detected.")

    # Test Chaos Coupling
    print("\nTesting Chaos Coupling...")
    field.dark_matter_state = {"x": 1.0, "y": 1.0, "z": 1.0, "w": 5.0, "q": 0.5}
    cns.last_tick_time = time.time() - 0.1 # Force a 100ms tick
    cns._apply_physics_tick()
    
    # Check if w influence reached velocities (12th dimension index 11)
    # The forces[:, 11] should have been boosted by w * 500.0
    v_12 = cns.swarm_velocities[:, 11]
    v_mag_12 = np.linalg.norm(v_12)
    print(f"12th Dimension (Chaos) Velocity Magnitude: {v_mag_12:.4e}")
    
    if v_mag_12 > 0:
        print("SUCCESS: Chaos coupling detected in 12th dimension.")
    else:
        print("FAILURE: No chaos coupling detected.")

if __name__ == "__main__":
    asyncio.run(test_cns_physics())
