"""llama-server KV-cache slot erasure for session reset."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def erase_all_slots(base_url: str) -> bool:
    """Erase KV-cache slots on llama-server to prevent context bleed.
    
    Probes GET /slots to detect llama-server capability, then erases all
    slots to ensure clean session state after /new or /reset command.
    
    Args:
        base_url: Base URL of the LLM server (e.g., http://localhost:8080)
        
    Returns:
        True if llama-server detected and erasure attempted, False otherwise
    """
    import httpx
    
    # Normalize base_url - remove trailing slash for consistent API calls
    base = base_url.rstrip("/")
    
    try:
        # Probe for llama-server capability via GET /slots
        # This endpoint is unique to llama-server and not present in other backends
        slots_response = httpx.get(f"{base}/slots", timeout=2.0)
        
        # If 404 or 501, this is not llama-server - skip erasure
        if slots_response.status_code in (404, 501):
            logger.debug("llama-server not detected at %s (status %d), skipping slot erasure", 
                        base_url, slots_response.status_code)
            return False
        
        # Parse slots response
        slots_data = slots_response.json()
        if not isinstance(slots_data, list):
            logger.debug("Unexpected /slots response format: %s", slots_data)
            return False
        
        # Erase ALL slots to prevent cross-slot context bleed
        # This is critical for --parallel N deployments where multiple slots exist
        for slot_info in slots_data:
            slot_id = slot_info.get("id")
            if slot_id is None:
                continue
            
            try:
                erase_response = httpx.post(
                    f"{base}/slots/{slot_id}?action=erase",
                    timeout=2.0
                )
                if erase_response.status_code != 200:
                    logger.debug("Failed to erase slot %d (status %d)", 
                                slot_id, erase_response.status_code)
            except httpx.HTTPError as e:
                logger.debug("Failed to erase slot %d: %s", slot_id, e)
        
        logger.debug("Successfully erased %d slots on llama-server", len(slots_data))
        return True
        
    except httpx.HTTPError as e:
        # Network error or timeout - assume not llama-server
        logger.debug("Failed to probe llama-server at %s: %s", base_url, e)
        return False
    except Exception as e:
        # Any other error - assume not llama-server
        logger.debug("Unexpected error probing llama-server: %s", e)
        return False
