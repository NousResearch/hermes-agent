import sys
import os
import importlib

# Add the project roots to sys.path
sys.path.append(r"D:\Cosmos\Cosmos\web")
sys.path.append(r"D:\Cosmos")

modules_to_test = [
    "cosmosynapse.engine.cosmos_swarm_orchestrator",
    "cosmosynapse.engine.cosmos_cns",
    "cosmos.core.model_manager",
    "cosmos.core.llm_backend"
]

def test_import(module_name):
    print(f"Testing {module_name}...")
    try:
        importlib.import_module(module_name)
        print(f"SUCCESS: {module_name}")
    except ImportError as e:
        print(f"FAILED: {module_name} - {e}")
        # If it's the specific error, try to find where it's coming from
        if "cannot import name 'list' from 'typing'" in str(e):
             print(f"CRITICAL: Found it in {module_name} or its children")

for m in modules_to_test:
    test_import(m)
