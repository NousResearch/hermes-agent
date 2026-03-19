"""
Asynchronous MEV Detection Service integration for Cosmos AI.
Provides async interface to MEV detection API endpoints.
"""

import httpx
from typing import Optional, Dict, Any, Coroutine
from loguru import logger
from Cosmos.memory.memory_system import get_memory_system
from Cosmos.core.capability_registry import get_capability_registry
from Cosmos.core.collective.session_manager import get_session_manager
import asyncio
import json

class MEVService:
    def __init__(self, api_url: str = "https://api.mev-detection.example"):
        self.api_url = api_url
        self.client = httpx.AsyncClient(base_url=api_url)
        self.capabilities = get_capability_registry()
        self.memory = get_memory_system()
        self.session_manager = get_session_manager()

    async def detect_mev(
        self,
        transaction_data: Dict[str, Any],
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Asynchronously detect MEV opportunities in a transaction
        
        Args:
            transaction_data: Dictionary containing transaction details
            confidence_threshold: Minimum confidence for MEV detection
            
        Returns:
            Dictionary containing MEV detection results
            
        Raises:
            HTTPException: If API request fails
            ValueError: If processing fails
        """
        try:
            logger.info(f"Starting MEV detection for transaction {transaction_data}")
            
            # Validate input data
            if not isinstance(transaction_data, dict):
                raise ValueError("Transaction data must be a dictionary")
                
            # Use capability registry to check if service is available
            if not self.capabilities.has("mev_detection"):
                logger.warning("MEV Detection capability not available")
                return {"mev_detected": False, "confidence": 0.0, "details": "Capability not available"}
                
            # Check memory for similar transactions
            similar_transactions = self.memory.query(
                "SELECT * FROM transactions WHERE hash = ?", 
                [transaction_data.get("hash", "")]
            )
            
            if similar_transactions:
                logger.info(f"Similar transactions found: {len(similar_transactions)}")
                
            # Call MEV detection API
            async with self.client:
                response = await self.client.post(
                    "/api/v1/detect",
                    json={
                        "transaction": transaction_data,
                        "threshold": confidence_threshold
                    }
                )
                
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"MEV service returned error: {response.text}"
                )
                
            result = response.json()
            
            # Apply additional processing
            result["processing_time"] = asyncio.current_time() - self.start_time
            
            return result
            
        except Exception as e:
            logger.error(f"MEV detection failed: {str(e)}")
            raise

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def close(self):
        await self.client.aclose()

    def __del__(self):
        if hasattr(self, 'client') and not self.client.is_closed():
            asyncio.create_task(self.client.aclose())