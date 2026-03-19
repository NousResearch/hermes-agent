
import sys
import os
from pathlib import Path

# Add project roots to path
cosmos_root = os.path.abspath("c:/Users/corys/The-Cosmic-Davis-12D-Hebbian-Transformer--1/cosmos")
cosmos_root = os.path.abspath("c:/Users/corys/The-Cosmic-Davis-12D-Hebbian-Transformer--1/Cosmic Genesis A.Lmi Cybernetic Bio Resonance Core")

if cosmos_root not in sys.path:
    sys.path.append(cosmos_root)
if cosmos_root not in sys.path:
    sys.path.append(cosmos_root)

try:
    print("Testing Swarm Components Import...")
    from cosmos.core.swarm.deepseek_backbone import DeepSeekBackbone
    from cosmos.core.cognition.uncertainty_injector import UncertaintyInjector
    print("✅ cosmos Components Imported Successfully")
    
    print("Testing Cosmos Swarm Orchestrator Import...")
    from cosmosynapse.engine.cosmos_swarm_orchestrator import CosmosSwarmOrchestrator
    
    print("Initializing Orchestrator...")
    orchestrator = CosmosSwarmOrchestrator()
    
    print(f"DeepSeek Available: {orchestrator.deepseek is not None}")
    print(f"Uncertainty Injector Available: {orchestrator.uncertainty is not None}")
    
    if orchestrator.deepseek:
        print("✅ Hybrid Swarm: DeepSeek Backbone Active")
    else:
        print("⚠️ Hybrid Swarm: DeepSeek Backbone NOT Active")
        
    if orchestrator.uncertainty:
        print("✅ Hybrid Swarm: Uncertainty Injector Active")
    else:
        print("⚠️ Hybrid Swarm: Uncertainty Injector NOT Active")
        
    print("Verification Complete.")

except Exception as e:
    print(f"❌ Error during verification: {e}")
    import traceback
    traceback.print_exc()
