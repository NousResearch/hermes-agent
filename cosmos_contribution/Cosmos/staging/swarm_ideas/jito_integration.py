"""
Module for integrating Cosmos with Jito's trading infrastructure.
Provides functions to fetch available assets and execute swaps.
"""

import asyncio
from typing import Dict, List, Optional
from loguru import logger
from aiohttp import ClientSession, ClientResponse
from Cosmos.memory.memory_system import get_memory_system
from Cosmos.core.capability_registry import get_capability_registry

class JitoIntegrationError(Exception):
    """Base exception class for Jito integration errors."""
    pass

class AuthenticationError(JitoIntegrationError):
    """Exception raised for authentication failures."""
    pass

class APIError(JitoIntegrationError):
    """Exception raised for API-related errors."""
    pass

class JitoIntegration:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.jito.com/v1"
        self.session = None
        self._capabilities = get_capability_registry().get("jito_integration", {})
        
    async def _initialize_session(self) -> None:
        """Initialize the aiohttp session with Jito API settings."""
        self.session = ClientSession(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
        
    async def close(self) -> None:
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None
            
    async def fetch_assets(self) -> List[Dict[str, str]]:
        """Fetch available assets from Jito platform.
        
        Returns:
            List[Dict[str, str]]: List of asset dictionaries with 'id' and 'name'
        
        Raises:
            AuthenticationError: If authentication fails
            APIError: If API returns an error status
        """
        try:
            if not self.session:
                await self._initialize_session()
                
            async with self.session.get("/assets") as response:
                if response.status == 401:
                    raise AuthenticationError("Invalid API key or credentials")
                elif response.status != 200:
                    raise APIError(f"API returned HTTP {response.status}")
                    
                data = await response.json()
                return data.get("assets", [])
                
        except aiohttp.ClientError as e:
            logger.error(f"Network error connecting to Jito API: {str(e)}")
            raise APIError(f"Network error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise
    
    async def execute_swap(self, from_asset_id: str, to_asset_id: str, amount: float) -> Dict[str, str]:
        """Execute a swap between two assets on Jito platform.
        
        Args:
            from_asset_id: ID of the source asset
            to_asset_id: ID of the target asset
            amount: Amount of source asset to swap
            
        Returns:
            Dict[str, str]: Swap transaction details
        
        Raises:
            AuthenticationError: If authentication fails
            APIError: If API returns an error status
            ValueError: If invalid asset IDs or amount
        """
        try:
            # Validate inputs
            if not from_asset_id or not to_asset_id:
                raise ValueError("Asset IDs must be non-empty")
            if amount <= 0:
                raise ValueError("Amount must be positive")
                
            if not self.session:
                await self._initialize_session()
                
            payload = {
                "from_asset": from_asset_id,
                "to_asset": to_asset_id,
                "amount": amount,
                "capabilities": self._capabilities
            }
            
            async with self.session.post("/swap", json=payload) as response:
                if response.status == 401:
                    raise AuthenticationError("Invalid API key or credentials")
                elif response.status != 200:
                    raise APIError(f"API returned HTTP {response.status}")
                    
                data = await response.json()
                return data
                
        except aiohttp.ClientError as e:
            logger.error(f"Network error executing swap: {str(e)}")
            raise APIError(f"Network error: {str(e)}")
        except ValueError as e:
            logger.error(f"Invalid parameters: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during swap: {str(e)}")
            raise

async def init_jito_integration(api_key: str) -> JitoIntegration:
    """Initialize and return a JitoIntegration instance."""
    return JitoIntegration(api_key)

async def fetch_jito_assets(api_key: str) -> List[Dict[str, str]]:
    """Factory function for fetching Jito assets."""
    jito = JitoIntegration(api_key)
    return await jito.fetch_assets()

async def swap_assets(api_key: str, from_asset: str, to_asset: str, amount: float) -> Dict[str, str]:
    """Factory function for executing Jito swaps."""
    jito = JitoIntegration(api_key)
    return await jito.execute_swap(from_asset, to_asset, amount)