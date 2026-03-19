
import asyncio
import sys
import os
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def verify_system():
    print("=== cosmos SWARM VERIFICATION ===")
    
    # 1. Check Models
    print("\n[1] Checking AI Models (Ollama)...")
    try:
        import ollama
        resp = ollama.list()
        # Handle dict or object response
        models = resp.get('models', []) if isinstance(resp, dict) else getattr(resp, 'models', [])
        
        if not models:
            print(f"⚠️  Raw Ollama response: {resp}")
        
        found_deepseek = False
        found_phi = False
        
        print("Installed models:")
        for m in models:
            # Handle model object or dict
            name = m.get('name') if isinstance(m, dict) else getattr(m, 'name', str(m))
            print(f"  - {name}")
            if 'deepseek-r1:8b' in name: found_deepseek = True
            if 'phi3' in name: found_phi = True
            
        if found_deepseek: print("✅ DeepSeek R1 (Reasoning) - READY")
        else: print("❌ DeepSeek R1 - MISSING (Run 'ollama pull deepseek-r1:8b')")
        
        if found_phi: print("✅ Phi3 (Speed) - READY")
        else: print("⚠️ Phi3 - MISSING (Run 'ollama pull phi3')")
        
    except ImportError:
        print("⚠️ 'ollama' python package not installed.")
    except Exception as e:
        print(f"⚠️ Ollama check failed: {e}")

    # 2. Check Cosmos Architecture
    print("\n[2] Verifying Cosmos Swarm Architecture...")
    try:
        from textwrap import dedent
        
        # Add Cosmic Genesis path (Sibling of cosmos)
        # scripts/verify_swarm.py -> scripts -> cosmos -> Project Root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cosmos_root = os.path.join(project_root, "Cosmic Genesis A.Lmi Cybernetic Bio Resonance Core")
        
        print(f"  > Adding path: {cosmos_root}")
        sys.path.append(cosmos_root)
        
        from cosmosynapse.engine.cosmos_swarm_orchestrator import CosmosSwarmOrchestrator
        
        orch = CosmosSwarmOrchestrator()
        
        # Test 1: Bio-Signature Injection
        print("  > Testing Bio-Signature Empathy Injection...")
        user_physics = {
            'bio_signatures': {
                'emotion': 'GRIEF',
                'intensity': 0.95
            }
        }
        
        # We can't easily run the full async generate without a running backend,
        # but we can check the internal prompt construction logic by inspecting the method
        # or calling a private helper if available.
        # Alternatively, we just check if the method accepts the argument.
        
        import inspect
        sig = inspect.signature(orch._cosmos_synthesize)
        if 'user_physics' in sig.parameters:
            print("✅ Orchestrator accepts user_physics (Bio-Data confirmed)")
            
            # Simulate prompt construction (white-box test)
            prompt_context = orch._get_bio_context(user_physics)
            print(f"  > Generated Context:\n{dedent(prompt_context)}")
            
            if "GRIEF" in prompt_context and "MUST validate" in prompt_context:
                print("✅ Empathy Logic: CONFIRMED")
            else:
                print("❌ Empathy Logic: FAILED")
        else:
            print("❌ Orchestrator missing bio-data parameter!")

    except Exception as e:
        print(f"❌ Architecture check failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== VERIFICATION COMPLETE ===")

if __name__ == "__main__":
    asyncio.run(verify_system())
