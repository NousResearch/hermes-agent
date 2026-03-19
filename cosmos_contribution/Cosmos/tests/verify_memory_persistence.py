import asyncio
import os
import json
import shutil
from pathlib import Path
from datetime import datetime

# Setup paths
PROJECT_ROOT = Path("D:/Cosmos/Cosmos")
DATA_DIR = PROJECT_ROOT / "data"
TEST_DB_DIR = DATA_DIR / "dialogue_memory"
TEST_ORG_DIR = DATA_DIR / "organism_state"

# Import target modules
import sys
# Add PARENT of Cosmos to path so 'import Cosmos' works
sys.path.insert(0, str(PROJECT_ROOT.parent))

# ALIAS FIX (as done in server.py)
try:
    import Cosmos
    sys.modules['cosmos'] = Cosmos
    print("SUCCESS: Aliased 'Cosmos' to 'cosmos'")
except ImportError:
    print("WARNING: Could not import 'Cosmos' from parent dir")
    # Try importing from current dir if root is different
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        import core
        # If we can't find Cosmos, but we can find core, 
        # let's try to fake the cosmos package for the test
        class FakeCosmos:
            import core
        sys.modules['cosmos'] = FakeCosmos
    except ImportError:
         print("FAILURE: Could not find 'core' or 'Cosmos'")

# Fixed imports
from Cosmos.core.collective.dialogue_memory import get_dialogue_memory
from Cosmos.core.collective.deliberation import DeliberationResult, AgentTurn, DeliberationRound
from Cosmos.core.collective.orchestration import SwarmOrchestrator
from Cosmos.core.collective.organism import CollectiveOrganism

async def test_persistence():
    print("--- STARTING PERSISTENCE VERIFICATION ---")
    
    # 1. Test DialogueMemory durability
    memory = get_dialogue_memory()
    turn = AgentTurn(
        turn_id="test_turn", 
        timestamp=datetime.now(), 
        agent_id="test_bot", 
        content="Persistence Test", 
        round_type=DeliberationRound.PROPOSE
    )
    result = DeliberationResult(
        deliberation_id="test_123",
        prompt="Verification",
        participating_agents=["test_bot"],
        rounds={DeliberationRound.PROPOSE.value: [turn]},
        final_response="Persistence Test",
        winning_agent="test_bot",
        vote_breakdown={"test_bot": 1.0},
        total_duration_ms=0,
        consensus_reached=True
    )
    
    print("Step 1: Storing exchange...")
    await memory.store_exchange(result, session_type="swarm_chat")
    
    # Verify file exists
    ex_file = TEST_DB_DIR / "exchanges.json"
    if ex_file.exists():
        print(f"SUCCESS: DialogueMemory saved to {ex_file}")
    else:
        print("FAILURE: DialogueMemory file not found")
        return

    # 2. Test SwarmOrchestrator restoration
    print("Step 2: Testing Orchestrator restoration...")
    orchestrator = SwarmOrchestrator()
    # Wait for async history restoration
    await asyncio.sleep(0.5)
    history = orchestrator.conversation_history
    if any(h['content'] == "Persistence Test" for h in history):
        print("SUCCESS: SwarmOrchestrator restored history")
    else:
        print("FAILURE: SwarmOrchestrator history empty")

    # 3. Test CollectiveOrganism evolution persistence
    print("Step 3: Testing Organism evolution persistence...")
    org = CollectiveOrganism()
    initial_gen = org.generation
    print(f"Initial Generation: {initial_gen}")
    
    org.evolve()
    evolved_gen = org.generation
    print(f"Evolved Generation: {evolved_gen}")
    
    # Verify snapshot exists
    snap_file = TEST_ORG_DIR / "organism_state.json"
    if snap_file.exists():
        print(f"SUCCESS: Organism snapshot saved to {snap_file}")
    else:
        print("FAILURE: Organism snapshot not found")
        return

    # Restart Organism
    print("Step 4: Testing Organism state restoration...")
    org_restarted = CollectiveOrganism()
    if org_restarted.generation == evolved_gen:
        print(f"SUCCESS: Organism restored to Generation {org_restarted.generation}")
    else:
        print(f"FAILURE: Organism reset to Generation {org_restarted.generation}")

if __name__ == "__main__":
    asyncio.run(test_persistence())
