"""
cosmos Universal AI Gateway.

"If it thinks, I can link to it!"

This module allows cosmos to connect to ANY external AI/LLM API.
It normalizes inputs/outputs so the swarm can use Google AI Labs, Grok, OpenAI, etc.
transparently alongside local models.
"""

from typing import Dict, Any, AsyncIterator, Optional
from dataclasses import dataclass
from loguru import logger
import aiohttp
import json

from Cosmos.core.llm_backend import LLMBackend, GenerationConfig, GenerationResult, BackendType, StreamChunk

@dataclass
class APIEndpoint:
    name: str # e.g. "grok-1", "gemini-pro"
    base_url: str
    api_key: str
    provider_type: str # "openai_compatible", "anthropic", "google"

class UniversalAIBackend(LLMBackend):
    """
    A backend adapter that treats external APIs as just another swarm member.
    """
    def __init__(self, endpoint: APIEndpoint, config: Optional[GenerationConfig] = None):
        super().__init__(endpoint.name, config)
        self.endpoint = endpoint
        
    @property
    def backend_type(self) -> BackendType:
        # We assume most are roughly OpenAI compatible for now
        # Ideally add BackendType.REMOTE_API
        return BackendType.OLLAMA 

    async def load(self) -> bool:
        # Nothing to load locally
        self._is_loaded = True
        return True

    async def unload(self) -> bool:
        self._is_loaded = False
        return True

    async def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> GenerationResult:
        cfg = config or self.config
        
        async with aiohttp.ClientSession() as session:
            # Simple OpenAI-compatible implementation
            if self.endpoint.provider_type == "openai_compatible":
                headers = {"Authorization": f"Bearer {self.endpoint.api_key}"}
                payload = {
                    "model": self.endpoint.name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": cfg.temperature,
                    "max_tokens": cfg.max_tokens
                }
                
                async with session.post(f"{self.endpoint.base_url}/chat/completions", headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"API Error: {resp.status} {await resp.text()}")
                        
                    data = await resp.json()
                    text = data['choices'][0]['message']['content']
                    
                    return GenerationResult(
                        text=text,
                        tokens_generated=data.get('usage', {}).get('completion_tokens', 0),
                        tokens_per_second=0, # Hard to calc latency without start time
                        model_used=self.model_name,
                        backend_used=self.backend_type,
                        confidence_score=1.0 # Trust the API?
                    )
            
            # Google Logic (Gemini)
            elif self.endpoint.provider_type == "google":
                # Implementation for google-generativeai
                pass
                
        return GenerationResult(text="Error: Provider not supported", tokens_generated=0, tokens_per_second=0, model_used="", backend_used=self.backend_type)

    async def generate_stream(self, prompt: str, config: Optional[GenerationConfig] = None) -> AsyncIterator[StreamChunk]:
         # Logic for streaming
         yield StreamChunk(text="", token_index=0)

    async def get_embedding(self, text: str) -> list[float]:
        return []

# Factory to register known providers
def create_grok_backend(api_key: str) -> UniversalAIBackend:
    return UniversalAIBackend(APIEndpoint(
        name="grok-1",
        base_url="https://api.x.ai/v1",
        api_key=api_key,
        provider_type="openai_compatible"
    ))

def create_google_backend(api_key: str) -> UniversalAIBackend:
    return UniversalAIBackend(APIEndpoint(
        name="gemini-pro",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key=api_key,
        provider_type="google"
    ))
