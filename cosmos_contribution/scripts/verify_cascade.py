"""
Automated test to verify the HermesCascadeBackend escalation logic.
It uses mock backends to ensure the cascade correctly routes simple prompts
to the local model and complex prompts to the cloud model.
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, patch

# Add parent directory to path to import Cosmos
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Cosmos.core.llm_backend import (
    HermesCascadeBackend,
    LLMBackend,
    GenerationResult,
    GenerationConfig,
)

class MockBackend(LLMBackend):
    def __init__(self, name: str, mock_confidence: float = 0.9):
        super().__init__("mock")
        self.name = name
        self.mock_confidence = mock_confidence
        self.call_count = 0

    @property
    def backend_type(self): return "mock"
    
    async def get_embedding(self, text: str): return [0.1]*10

    async def load(self): return True
    async def unload(self): return True
    
    async def generate(self, prompt: str, config=None) -> GenerationResult:
        from Cosmos.core.llm_backend import BackendType
        self.call_count += 1
        return GenerationResult(
            text=f"Response from {self.name}",
            tokens_generated=10,
            tokens_per_second=100.0,
            model_used=self.name,
            backend_used=BackendType.OLLAMA,
            confidence_score=self.mock_confidence,
            escalated=False,
            cascade_depth=0
        )
        
    async def generate_stream(self, prompt: str, config=None):
        pass # Not used in this basic test


async def main():
    print("Testing HermesCascadeBackend Escalation Logic...")
    
    local = MockBackend("Local-Qwen-8B", mock_confidence=0.9)
    cloud = MockBackend("Cloud-Hermes-36B")
    
    cascade = HermesCascadeBackend(
        local_backend=local,
        cloud_backend=cloud,
        escalation_threshold=0.65
    )
    
    # Test 1: Simple prompt -> should route to local
    print("\n[Test 1] Simple greeting...")
    simple_prompt = "Hello, how are you today?"
    result1 = await cascade.generate(simple_prompt)
    
    if result1.escalated:
        print("[FAILED] Simple prompt was incorrectly escalated.")
    else:
        print("[PASS] Simple prompt handled locally.")
        
    assert local.call_count == 1
    assert cloud.call_count == 0
    
    # Test 2: Complex prompt (keywords + length) -> should route to cloud
    print("\n[Test 2] Complex coding architect question...")
    complex_prompt = """
    We need to design a highly scalable microservices architecture.
    Please analyze the trade-offs between REST and gRPC for synchronous communication.
    
    ```python
    # Implement a resilient retry algorithm in Python (using exponential backoff and jitter)
    # that can handle transient network failures in a distributed system.
    def retry_with_backoff():
        pass
    ```
    
    Evaluate the time complexity of the recursive Fibonacci function comparing
    memoization vs a bottom-up dynamic programming approach.
    Synthesize these concepts into a cohesive design document.
    """
    cascade.escalation_threshold = 0.60
    result2 = await cascade.generate(complex_prompt)
    
    if result2.escalated:
        print(f"[PASS] Complex prompt correctly escalated to cloud! (complexity={cascade.estimate_hermes_complexity(complex_prompt):.2f})")
    else:
        print("[FAILED] Complex prompt was NOT escalated.")
        
    assert local.call_count == 1  # Should not have increased
    assert cloud.call_count == 1
    
    # Test 3: Simple prompt but local confidence is low -> should escalate
    print("\n[Test 3] Simple prompt but local model is unconfident...")
    low_conf_local = MockBackend("Local-Confused", mock_confidence=0.3)
    cascade_low_conf = HermesCascadeBackend(
        local_backend=low_conf_local,
        cloud_backend=cloud,
        escalation_threshold=0.65
    )
    
    result3 = await cascade_low_conf.generate("What is 1+1?")
    
    if result3.escalated:
        print("[PASS] Low local confidence correctly triggered fallback escalation!")
    else:
        print("[FAILED] System failed to escalate on low confidence.")
        
    assert low_conf_local.call_count == 1
    assert cloud.call_count == 2 # Cloud was called again
    
    # Print stats
    print("\n--- Cascade Stats ---")
    print(cascade.get_cascade_stats())

if __name__ == "__main__":
    asyncio.run(main())
