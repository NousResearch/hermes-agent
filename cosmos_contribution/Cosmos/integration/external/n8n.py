"""
cosmos n8n Integration.

"I can wire anything into anything! I'm the Professor!"

This module allows cosmos to trigger and monitor n8n workflows.
"""

import asyncio
from typing import Dict, Any, List
from loguru import logger
import aiohttp

from .base import ExternalProvider, IntegrationConfig, ConnectionStatus
from Cosmos.core.nexus import nexus, Signal, SignalType

class N8nProvider(ExternalProvider):
    def __init__(self, api_key: str, base_url: str):
        super().__init__(IntegrationConfig(name="n8n", api_key=api_key))
        self.base_url = base_url.rstrip("/")
        
    async def connect(self) -> bool:
        if not self.config.api_key or not self.base_url:
            return False
            
        try:
            # Test connection by listing workflows
            async with aiohttp.ClientSession() as session:
                headers = {"X-N8N-API-KEY": self.config.api_key}
                async with session.get(f"{self.base_url}/api/v1/workflows", headers=headers) as resp:
                    if resp.status == 200:
                        logger.info("n8n: Connected")
                        self.status = ConnectionStatus.CONNECTED
                        return True
                    else:
                        logger.error(f"n8n connection failed: {resp.status}")
                        return False
        except Exception as e:
            logger.error(f"n8n connection error: {e}")
            self.status = ConnectionStatus.ERROR
            return False

    async def sync(self):
        # n8n doesn't inherently need sync unless we want to cache workflow IDs
        pass

    async def execute_action(self, action: str, params: Dict[str, Any]) -> Any:
        if self.status != ConnectionStatus.CONNECTED:
            raise ConnectionError("n8n not connected")

        async with aiohttp.ClientSession() as session:
            headers = {"X-N8N-API-KEY": self.config.api_key}
            
            if action == "trigger_workflow":
                workflow_id = params.get('workflow_id')
                data = params.get('data', {})
                
                # Activate via webhook or API execute
                # Assuming Webhook for simplicity or API exec if ID provided
                url = f"{self.base_url}/api/v1/workflows/{workflow_id}/execute"
                async with session.post(url, headers=headers, json=data) as resp:
                    return await resp.json()
            
            elif action == "list_workflows":
                async with session.get(f"{self.base_url}/api/v1/workflows", headers=headers) as resp:
                    return await resp.json()
            
            else:
                raise ValueError(f"Unknown action: {action}")
