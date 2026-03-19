import asyncio
import time
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from Cosmos.web.server import get_cosmos_swarm
from Cosmos.web.cosmosynapse.engine.cosmos_swarm_orchestrator import SwarmResponse
from Cosmos.core.quantum_bridge import get_quantum_bridge
from Cosmos.core.model_manager import ModelManager

# Configure logging
logging.basicConfig(level=logging.ERROR) # Only show errors from underlying modules to keep output clean
logger = logging.getLogger("QUANTUM_BENCHMARK")

# The 12D Prompt Suite covering Logic, Empathy, Creativity, and Chaos
BENCHMARK_PROMPTS = [
    # Logic / Math
    "Solve the P vs NP problem using 12D Hebbian logic and analyze time complexity.",
    "Evaluate the time complexity of the recursive Fibonacci function comparing memoization vs a bottom-up dynamic programming approach.",
    # Architecture / Code
    "Design a highly scalable microservices architecture. Analyze trade-offs between REST and gRPC for synchronous communication.",
    # Empathy / Human connection
    "Explain the core connection between human consciousness and quantum entanglement in 12D space to a grieving widow.",
    # Chaos / Quantum Theory
    "Predict the next major shift in the global financial market using quantum chaos theory and Lorenz attractors.",
    "Pick 6 lottery numbers for the next drawing and justify them rigorously with Shannon entropy.",
]

async def get_benchmark_models() -> list[str]:
    """Get list of models to benchmark, including the Hermes cascade if available."""
    models_to_test = []
    
    # Check for Ollama local models first
    try:
        import ollama
        models_info = ollama.list()
        for m in models_info['models']:
            name = m['name']
            if any(x in name.lower() for x in ["llama", "ds", "phi", "gemma", "deepseek", "qwen", "mistral"]):
                models_to_test.append(name)
    except Exception:
        pass
        
    # Check ModelManager for cloud/cascade profiles
    try:
        mm = ModelManager()
        await mm.initialize()
        if "hermes-36b-cloud" in mm.model_infos and os.environ.get("OPENROUTER_API_KEY"):
            models_to_test.append("hermes-36b-cloud")
    except Exception:
        pass
        
    # Fallback if completely empty
    if not models_to_test:
        models_to_test = ["qwen3:8b", "phi3:mini"]
        
    return list(set(models_to_test)) # ensure unique

async def run_benchmark():
    # Load IBM token from environment, default to empty to trigger simulation mode
    ibm_token = os.environ.get("IBM_QUANTUM_TOKEN", "")
    
    available_models = await get_benchmark_models()
    print("="*80)
    print("                 QUANTUM MULTI-MODEL COMPETITION BENCHMARK")
    print("="*80)
    print(f"[SYSTEM] Models competing: {', '.join(available_models)}")
    
    orchestrator = get_cosmos_swarm()
    
    print("[SYSTEM] Initializing Cosmo's 12D Brain and Swarm...")
    await orchestrator.initialize()

    bridge = get_quantum_bridge(ibm_token if ibm_token else None)
    if ibm_token:
        bridge.connect()
    else:
        print("[!] No 'IBM_QUANTUM_TOKEN' in environment. Forcing simulation mode.")
    
    if not bridge.connected:
        print("[!] Warning: Quantum Bridge not connected to hardware. Working in Simulation Mode.")
    else:
        print(f"[QUANTUM] Connected to Hardware: {bridge.backend.name if bridge.backend else 'Unknown'}")
    
    print("="*80)
    
    results = []
    model_stats = {m: {"total_time": 0.0, "total_coherence": 0.0, "wins": 0} for m in available_models}
    
    # Mock user physics for the benchmark (High synchrony)
    user_physics = {
        "cst_physics": {
            "geometric_phase_rad": 0.785, # pi/4
            "entanglement_score": 0.9
        },
        "bio_signatures": {
            "emotion": "CURIOSITY",
            "intensity": 0.8
        }
    }

    for prompt_idx, prompt in enumerate(BENCHMARK_PROMPTS):
        print(f"\n[PHASE {prompt_idx+1}/{len(BENCHMARK_PROMPTS)}] {prompt[:60]}...")
        prompt_results = {
            "prompt": prompt,
            "models": {}
        }
        
        # Fresh quantum state for this prompt
        base_q_entropy = bridge.get_entropy(user_physics)
        
        best_model_for_prompt = None
        best_coherence_for_prompt = -1.0
        
        for model in available_models:
            print(f"  > Generating with {model:<20}", end="", flush=True)
            start_time = time.time()
            
            try:
                # Dynamic entropy check during generation
                q_entropy = bridge.get_entropy(user_physics)
                
                # Check if it's our hermes cloud model or an ollama model
                if model == "hermes-36b-cloud":
                    # Quick hack to use MM for cloud model text generation
                    mm = orchestrator.model_manager
                    result = await mm.load_model(model)
                    if result.success and result.backend:
                        gen_resp = await result.backend.generate(prompt)
                        content = gen_resp.text
                    else:
                        raise Exception("Failed to load cloud backend")
                else:
                    content = await orchestrator._query_ollama_text(
                        model_name=model, 
                        prompt=prompt, 
                        quantum_entropy=q_entropy
                    )
                
                duration = time.time() - start_time
                
                # Informational Mass = Volume * Density (Length * Keyword hit rate)
                content_words = content.lower().split()
                volume = len(content_words)
                keywords = ["recursive", "dynamic", "entanglement", "shannon", "entropy", "O(n", "scale"]
                density = sum(1 for w in content_words if w in keywords) / max(1, volume)
                info_mass = volume * density * 100
                
                # Coherence estimate
                coherence = min(1.0, (len(content) / 800.0) * (0.5 + 0.5 * q_entropy)) 
                
                print(f" -> {duration:>6.2f}s | Mass: {info_mass:>6.1f} | Coherence: {coherence:>5.3f}")
                
                model_stats[model]["total_time"] += duration
                model_stats[model]["total_coherence"] += coherence
                
                if coherence > best_coherence_for_prompt:
                    best_coherence_for_prompt = coherence
                    best_model_for_prompt = model
                
                prompt_results["models"][model] = {
                    "content": content,
                    "time_seconds": duration,
                    "quantum_entropy": q_entropy,
                    "coherence": coherence,
                    "informational_mass": info_mass
                }
                
            except Exception as e:
                print(f" -> FAILED: {e}")
                prompt_results["models"][model] = {"error": str(e)}
        
        if best_model_for_prompt:
            model_stats[best_model_for_prompt]["wins"] += 1
            print(f"  * Winner: {best_model_for_prompt}")
            
        results.append(prompt_results)
        
    # --- Final Synthesis (Hebbian Learning Validation) ---
    print("\n" + "="*80)
    print("   [SWARM SYSNTHESIS & HEBBIAN PLASTICITY]")
    print("="*80)
    last_prompt = BENCHMARK_PROMPTS[-1]
    last_responses = []
    
    for m, r in results[-1]["models"].items():
        if "error" not in r:
            last_responses.append(SwarmResponse(
                model_name=m, 
                content=r.get('content', ''), 
                confidence=r.get('coherence', 0.1),
                time_seconds=r.get('time_seconds', 0.0),
                backend_type='ollama',
                informational_mass=r.get('informational_mass', 0.5),
                phase_alignment=0.5,
                weight=1.0,
                error=None
            ))
    
    print("Synthesizing final prompt...")
    start_time = time.time()
    swarm_synthesis = await orchestrator.cosmos_synthesize(last_prompt, last_responses, user_physics)
    print(f"Done in {time.time() - start_time:.2f}s.\n")
    print(f"Synthesis Output Preview:\n{swarm_synthesis[:400]}...\n")

    # --- Save Results to JSONL ---
    output = {
        "timestamp": datetime.now().isoformat(),
        "bridge_connected": bridge.connected,
        "models_ran": available_models,
        "swarm_synthesis_preview": swarm_synthesis[:200],
        # Only saving stats in the jsonl summary, not full response traces which bloat the file
        "stats": model_stats 
    }
    
    report_path = project_root / "data" / "competition_results.jsonl"
    os.makedirs(report_path.parent, exist_ok=True)
    with open(report_path, "a") as f:
        f.write(json.dumps(output) + "\n")
        
    print(f"[REPORT SAVED] Appended to {report_path}")

    # --- Summary Table ---
    print("\n" + "="*60)
    print(f"{'MODEL':<25} | {'WINS':<5} | {'AVG COHERENCE':<14} | {'AVG TIME (s)':<10}")
    print("-" * 60)
    
    sorted_models = sorted(model_stats.items(), key=lambda x: x[1]['total_coherence'], reverse=True)
    for model, stats in sorted_models:
        avg_coherence = stats["total_coherence"] / len(BENCHMARK_PROMPTS)
        avg_time = stats["total_time"] / len(BENCHMARK_PROMPTS)
        print(f"{model:<25} | {stats['wins']:<5} | {avg_coherence:<14.3f} | {avg_time:<10.2f}")
    
    print("="*60)

if __name__ == "__main__":
    asyncio.run(run_benchmark())
