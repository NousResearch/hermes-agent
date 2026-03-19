
import sys
import os
import asyncio
from typing import Dict, Any

# Add project roots to path
cosmos_root = os.path.abspath("c:/Users/corys/The-Cosmic-Davis-12D-Hebbian-Transformer--1/cosmos")
cosmos_root = os.path.abspath("c:/Users/corys/The-Cosmic-Davis-12D-Hebbian-Transformer--1/Cosmic Genesis A.Lmi Cybernetic Bio Resonance Core")

if cosmos_root not in sys.path:
    sys.path.append(cosmos_root)
if cosmos_root not in sys.path:
    sys.path.append(cosmos_root)

# Colors for output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

async def verify_system():
    print(f"{Colors.HEADER}=== cosmos SYSTEM CAPABILITIES VERIFICATION ==={Colors.ENDC}\n")
    
    results = {}

    # 1. VERIFY MULTIMODAL SYSTEM
    print(f"{Colors.OKCYAN}1. Verifying Multimodal Ecosystem...{Colors.ENDC}")
    try:
        from cosmos.core.multimodal import get_multimodal_system, UnifiedMultimodalSystem
        system = get_multimodal_system()
        if system and isinstance(system, UnifiedMultimodalSystem):
            print(f"{Colors.OKGREEN}✓ UnifiedMultimodalSystem Initialized{Colors.ENDC}")
            # Check components
            
            # Create a dummy token to test fusion logic
            token, emotion, thought = system.process_multimodal_input(text="Hello world")
            print(f"{Colors.OKGREEN}✓ Bio-Fusion Engine Test: Thought Generated -> '{thought}'{Colors.ENDC}")
            print(f"{Colors.OKGREEN}✓ Emotional State: {emotion}{Colors.ENDC}")
            results['multimodal'] = True
        else:
            print(f"{Colors.FAIL}✗ Multimodal System Failed to Initialize{Colors.ENDC}")
            results['multimodal'] = False
    except Exception as e:
        print(f"{Colors.FAIL}✗ Multimodal Verification Error: {e}{Colors.ENDC}")
        results['multimodal'] = False

    # 2. VERIFY HYBRID SWARM
    print(f"\n{Colors.OKCYAN}2. Verifying Hybrid Swarm Intelligence...{Colors.ENDC}")
    try:
        from cosmosynapse.engine.cosmos_swarm_orchestrator import CosmosSwarmOrchestrator
        from cosmos.web.server import get_cosmos_swarm
        
        # Test lazy loader
        swarm = get_cosmos_swarm()
        if swarm:
            print(f"{Colors.OKGREEN}✓ Swarm Orchestrator Loaded via Server{Colors.ENDC}")
            
            # Check DeepSeek Backbone
            if swarm.deepseek:
                print(f"{Colors.OKGREEN}✓ DeepSeek Backbone: ACTIVE{Colors.ENDC}")
            else:
                print(f"{Colors.WARNING}⚠ DeepSeek Backbone: INACTIVE (Check imports){Colors.ENDC}")
                
            # Check Uncertainty Injector
            if swarm.uncertainty:
                print(f"{Colors.OKGREEN}✓ Uncertainty Injector: ACTIVE{Colors.ENDC}")
                # Test intuition
                hunch = swarm.uncertainty.evaluate_hunch(confidence=0.99, context_complexity=0.9)
                if hunch:
                    print(f"  - Simulated Hunch: \"{hunch}\"")
            else:
                print(f"{Colors.WARNING}⚠ Uncertainty Injector: INACTIVE{Colors.ENDC}")
            
            results['swarm'] = True
        else:
             print(f"{Colors.FAIL}✗ Swarm Orchestrator Failed to Load{Colors.ENDC}")
             results['swarm'] = False
    except Exception as e:
        print(f"{Colors.FAIL}✗ Swarm Verification Error: {e}{Colors.ENDC}")
        results['swarm'] = False

    # 3. VERIFY TOOL ROUTER REGISTRATION
    print(f"\n{Colors.OKCYAN}3. Verifying Tool Definitions...{Colors.ENDC}")
    try:
        from cosmos.integration.tool_router import ToolRouter
        router = ToolRouter()
        
        # Check specific new tools
        tool = router.get_tool("analyze_multimodal_content")
        if tool:
             print(f"{Colors.OKGREEN}✓ Tool 'analyze_multimodal_content' Registered{Colors.ENDC}")
             print(f"  - Category: {tool.category}")
             results['tools'] = True
        else:
             print(f"{Colors.FAIL}✗ Tool 'analyze_multimodal_content' NOT Found{Colors.ENDC}")
             results['tools'] = False
             
    except Exception as e:
        print(f"{Colors.FAIL}✗ Tool Verification Error: {e}{Colors.ENDC}")
        results['tools'] = False

    # SUMMARY
    print(f"\n{Colors.HEADER}=== VERIFICATION SUMMARY ==={Colors.ENDC}")
    all_passed = all(results.values())
    if all_passed:
        print(f"{Colors.OKGREEN}ALL SYSTEMS NOMINAL. READY FOR DEPLOYMENT.{Colors.ENDC}")
    else:
        print(f"{Colors.FAIL}SYSTEM CHECKS FAILED. REVIEW LOGS.{Colors.ENDC}")
        
if __name__ == "__main__":
    asyncio.run(verify_system())
