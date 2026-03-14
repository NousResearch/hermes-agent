"""
Shared constants for the Hermes Agent infrastructure.

This module is designed to be 'import-safe' with zero external dependencies, 
ensuring it can be utilized across the entire agentic stack (gateway, tools, core) 
without the risk of circular inheritance or import deadlocks.
"""

# --- OpenRouter Configuration ---
# OpenRouter serves as a primary unified interface for accessing a wide array 
# of LLMs. These constants define the routing pathways for model discovery 
# and inference execution within the agent's reasoning loops.

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_URL = f"{OPENROUTER_BASE_URL}/models"
OPENROUTER_CHAT_URL = f"{OPENROUTER_BASE_URL}/chat/completions"

# --- Nous Infrastructure Configuration ---
# Internal Nous Research inference endpoints. These URLs point to the 
# optimized hosting environment for Hermes-series models, providing 
# high-performance, low-latency access to first-party weights.

NOUS_API_BASE_URL = "https://inference-api.nousresearch.com/v1"
NOUS_API_CHAT_URL = f"{NOUS_API_BASE_URL}/chat/completions"
