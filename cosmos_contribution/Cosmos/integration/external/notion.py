"""
cosmos Notion Integration (Full Implementation).

"I keep my diagrams in a digital brain now."
"""

import asyncio
from typing import Dict, Any, List
from loguru import logger
from notion_client import Client, AsyncClient

from .base import ExternalProvider, IntegrationConfig, ConnectionStatus
from Cosmos.core.nexus import nexus, Signal, SignalType

class NotionProvider(ExternalProvider):
    def __init__(self, token: str):
        super().__init__(IntegrationConfig(name="notion", api_key=token))
        self.client = None
        
    async def connect(self) -> bool:
        if not self.config.api_key:
            return False
            
        try:
            self.client = AsyncClient(auth=self.config.api_key)
            # Test query
            await self.client.users.list()
            
            logger.info("Notion: Connected")
            self.status = ConnectionStatus.CONNECTED
            return True
        except Exception as e:
            logger.error(f"Notion connection failed: {e}")
            self.status = ConnectionStatus.ERROR
            return False

    async def sync(self):
        # Implementation depends on specific DB ID monitoring
        pass

    async def execute_action(self, action: str, params: Dict[str, Any]) -> Any:
        if self.status != ConnectionStatus.CONNECTED:
            raise ConnectionError("Notion not connected")

        if action == "create_page":
            parent_id = params.get('parent_id')
            title = params.get('title')
            
            page = await self.client.pages.create(
                parent={"database_id": parent_id} if "database" in parent_id else {"page_id": parent_id},
                properties={
                    "title": [
                        {"text": {"content": title}}
                    ]
                }
            )
            return {"id": page["id"], "url": page["url"]}
            
        elif action == "search":
            query = params.get('query')
            results = await self.client.search(query=query)
            return results.get('results', [])
            
        else:
            raise ValueError(f"Unknown action: {action}")
