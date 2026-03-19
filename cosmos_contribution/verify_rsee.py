
import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Mocking SwarmChatManager if needed
class MockSwarmManager:
    pass

async def verify_startup():
    print("--- RSEE Integration Verification ---")
    
    try:
        from Cosmos.core.evolution_loop import get_evolution_loop, start_evolution
        from Cosmos.core.evolution.recursive_self_evolution import get_rsee
        
        print("[1] Importing components: SUCCESS")
        
        loop = get_evolution_loop()
        rsee = get_rsee()
        
        print(f"[2] EvolutionLoop instance: {loop}")
        print(f"[3] RSEE instance: {rsee}")
        
        # Test start_evolution (background task)
        swarm_manager = MockSwarmManager()
        task = asyncio.create_task(start_evolution(swarm_manager))
        
        print("[4] Starting EvolutionLoop...")
        await asyncio.sleep(2) # Give it a second to ignite RSEE
        
        if loop.running:
            print("[5] EvolutionLoop running: YES")
        else:
            print("[5] EvolutionLoop running: NO (Check logs)")
            
        if rsee.running:
            print("[6] RSEE running: YES")
        else:
            print("[6] RSEE running: NO (Check logs)")
            
        if loop.running and rsee.running:
            print("\nVERIFICATION SUCCESSFUL: RSEE is correctly integrated and operational.")
            return True
        else:
            print("\nVERIFICATION FAILED: Components did not start correctly.")
            return False
            
    except Exception as e:
        print(f"\nERROR DURING VERIFICATION: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(verify_startup())
