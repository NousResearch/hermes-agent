import os
import httpx
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

def invoke_model_router_bridge(messages: List[Dict[str, Any]], tier: str = "LocalBasic", role: str = "hermes", options: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    """
    Thin HTTP client pointing at bridge endpoint.
    Replaces direct Ollama calls under the gateway mode.
    Gated on HERMES_MODEL_ROUTER_BRIDGE_MODE env var ('direct' | 'parallel' | 'gateway')
    """
    mode = os.environ.get("HERMES_MODEL_ROUTER_BRIDGE_MODE", "direct").strip().lower()
    
    if mode == "direct":
        return None

    try:
        # Gateway APISIX resolves this to gateway service
        url = "http://localhost:9080/api/model-router/resolve"
        
        payload = {
            "tier": tier,
            "role": role,
            "messages": messages,
            "options": options or {}
        }
        
        response = httpx.post(url, json=payload, timeout=60.0)
        response.raise_for_status()
        data = response.json()
        
        if mode == "parallel":
            # For parallel run, the caller should compute divergence
            # We return it in a special wrapper
            return {"bridge_divergence_data": data}
            
        # Extract Langchain style or standard completion message
        return data.get("provider_response")
            
    except Exception as e:
        logger.error(f"[ModelRouter Client] invocation failed: {str(e)}")
        if mode == "gateway" or mode == "cutover":
            raise e
        return None
