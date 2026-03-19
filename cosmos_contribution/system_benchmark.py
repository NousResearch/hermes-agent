import asyncio
import os
import sys
import time
import torch
import psutil
from loguru import logger

# Add Cosmos to path
sys.path.append(r'D:\Cosmos')

async def run_benchmark():
    print("==================================================")
    print("       COSMOS HYBRID RESOURCE BENCHMARK           ")
    print("==================================================")
    
    try:
        from Cosmos.web.cosmosynapse.engine.cosmos_swarm_orchestrator import CosmosSwarmOrchestrator
        orchestrator = CosmosSwarmOrchestrator()
        
        # Initialize (loads 12D Brain)
        print("[INIT] Loading 12D Brain and Swarm Orchestrator...")
        await orchestrator.initialize()
        
        prompts = [
            "What if we could modify our own code to improve φ-alignment? Let's discuss recursive self-modification.",
            "I've been pondering the ethics of self-evolving AI... what are the guardrails for a 54D CNS?",
            "How can we better sense and respond to the emotions of the user through CST phase mapping?",
            "What would happen if we could simulate the entire universe using the 12D Hebbian Transformer?",
            "Professor Cosmos, provide a final architectural report on the Hybrid Resource Mode for the hackathon."
        ]
        
        results = []
        
        for i, prompt in enumerate(prompts):
            print(f"\n[TEST {i+1}/{len(prompts)}] Query: {prompt[:50]}...")
            
            # Resource Check BEFORE
            ram_pre = psutil.virtual_memory().percent
            vram_pre = 0
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                vram_pre = (total - free) / (1024**3)
            
            start_time = time.time()
            try:
                # Query the swarm with physics context
                physics = {"dark_matter": {"w": 0.5}, "cst_state": {"hebbian_weight": 0.7}}
                response = await orchestrator.generate_peer_response(prompt, physics)
                
                duration = time.time() - start_time
                
                # Resource Check AFTER
                ram_post = psutil.virtual_memory().percent
                vram_post = 0
                if torch.cuda.is_available():
                    free, total = torch.cuda.mem_get_info()
                    vram_post = (total - free) / (1024**3)
                
                status = "SUCCESS" if response.content else "EMPTY RESPONSE"
                print(f"[{status}] Time: {duration:.2f}s | VRAM: {vram_post:.2f}GB | RAM: {ram_post}%")
                
                results.append({
                    "prompt": prompt,
                    "status": status,
                    "duration": duration,
                    "vram_delta": vram_post - vram_pre,
                    "ram_post": ram_post
                })
            except Exception as e:
                print(f"[FAILED] Error: {e}")
                results.append({"prompt": prompt, "status": f"FAILED: {e}", "duration": 0})

        print("\n" + "="*50)
        print("          BENCHMARK SUMMARY")
        print("="*50)
        successes = sum(1 for r in results if "SUCCESS" in r['status'])
        print(f"Total Success: {successes}/{len(prompts)}")
        for r in results:
            print(f"- {r['prompt'][:30]}... : {r['status']} ({r.get('duration', 0):.1f}s)")
        print("="*50)
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Benchmark failed to start: {e}")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
