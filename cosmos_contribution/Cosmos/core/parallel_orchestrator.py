"""
cosmos Parallel AI Orchestrator.

"Why use one brain when you can have a committee that actually agrees for once?"

This module handles fan-out/fan-in processing for LLM requests.
It allows running the same prompt across different models (Ollama, xAI, Gemini)
and fusing the results.
"""

from typing import List, Dict, Any, Optional, Callable
from loguru import logger
from Cosmos.core.nexus import nexus, Signal, SignalType

class ParallelAIOrchestrator:
    def __init__(self, backends: list[Callable]):
        self.backends = backends # list of LLM generate functions
        
    async def dispatch_parallel(self, prompt: str) -> list[str]:
        """Run the same prompt on all configured backends simultaneously."""
        logger.info(f"ParallelAI: Dispatching to {len(self.backends)} backends.")
        
        # Create tasks
        tasks = []
        for backend in self.backends:
            tasks.append(asyncio.create_task(self._safely_call_backend(backend, prompt)))
            
        results = await asyncio.gather(*tasks)
        return [r for r in results if r]

    async def _safely_call_backend(self, backend: Callable, prompt: str) -> str:
        try:
            if asyncio.iscoroutinefunction(backend):
                return await backend(prompt)
            else:
                return backend(prompt)
        except Exception as e:
            logger.error(f"ParallelAI: Backend failure: {e}")
            return ""

    async def fused_consensus(self, prompt: str) -> str:
        """Run in parallel and use a final step to fuse findings into a single best answer."""
        results = await self.dispatch_parallel(prompt)
        if not results:
            return "No responses obtained from parallel swarm."
            
        # Consensus step: Use the strongest model to fuse the results
        fusion_prompt = f"""I have multiple responses for the prompt: "{prompt}"
        
        Responses:
        {chr(10).join([f"Response {i+1}: {r}" for i, r in enumerate(results)])}
        
        Synthesize the best, most accurate information from these responses into one perfect answer.
        """
        
        # Use first backend as 'Synthesizer' for now
        return await self._safely_call_backend(self.backends[0], fusion_prompt)

# Global Instance Generator
def create_parallel_orchestrator(backends):
    return ParallelAIOrchestrator(backends)
