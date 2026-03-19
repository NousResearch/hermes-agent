import sys
import os
import importlib

# The project structure: D:\Cosmos\Cosmos (root)
# Inside D:\Cosmos\Cosmos are the modules (web, core, etc.)
# server.py adds D:\Cosmos to path and aliases Cosmos to cosmos.

base_path = r"D:\Cosmos"
sys.path.insert(0, base_path)

# Alias Cosmos to cosmos
try:
    import Cosmos
    sys.modules['cosmos'] = Cosmos
    print("Aliased Cosmos to cosmos")
except ImportError:
    print("FAILED to alias Cosmos to cosmos")

# Add web to path for cosmosynapse
sys.path.append(os.path.join(base_path, "Cosmos", "web"))

modules_to_test = [
    "cosmosynapse.engine.cosmos_swarm_orchestrator",
    "cosmosynapse.engine.cosmos_cns",
    "cosmos.core.model_manager",
    "cosmos.core.llm_backend"
]

def test_import(module_name):
    print(f"\n--- Testing {module_name} ---")
    try:
        importlib.import_module(module_name)
        print(f"SUCCESS: {module_name}")
    except ImportError as e:
        print(f"FAILED: {module_name} - {e}")
        # Trace where it's coming from
        import traceback
        traceback.print_exc()

for m in modules_to_test:
    test_import(m)
