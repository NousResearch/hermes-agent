"""
cosmos X (Twitter) Integration.

"Good news everyone! I'm trending!"

Features:
1. Post Updates: Tweeting directly from the agent.
2. Monitor mentions: Reacting to user interactions.
"""

import asyncio
from typing import Dict, Any
from loguru import logger

# Check imports
try:
    import tweepy
except ImportError:
    tweepy = None

from .base import ExternalProvider, IntegrationConfig, ConnectionStatus

class XProvider(ExternalProvider):
    def __init__(self, api_key: str, api_secret: str, access_token: str, access_secret: str):
        super().__init__(IntegrationConfig(name="x"))
        self.creds = {
            "api_key": api_key,
            "api_secret": api_secret,
            "access_token": access_token,
            "access_secret": access_secret
        }
        self.client = None
        
    async def connect(self) -> bool:
        if not tweepy:
            logger.error("Tweepy not installed. Run `pip install tweepy`")
            return False
            
        try:
            # v2 Client
            self.client = tweepy.Client(
                consumer_key=self.creds["api_key"],
                consumer_secret=self.creds["api_secret"],
                access_token=self.creds["access_token"],
                access_token_secret=self.creds["access_secret"]
            )
            
            # Verify credentials via v1 API (usually easier for 'me' check)
            # Or just assume success if client init didn't fail (v2 is lazy)
            logger.info("X (Twitter): Connected")
            self.status = ConnectionStatus.CONNECTED
            return True
        except Exception as e:
            logger.error(f"X connection failed: {e}")
            self.status = ConnectionStatus.ERROR
            return False

    async def sync(self):
        """Poll for new mentions."""
        if self.status != ConnectionStatus.CONNECTED:
            return

        loop = asyncio.get_event_loop()
        try:
            # v2 get_users_mentions
            def _get_mentions():
                me = self.client.get_me()
                return self.client.get_users_mentions(id=me.data.id, max_results=5)

            response = await loop.run_in_executor(None, _get_mentions)
            if response.data:
                for tweet in response.data:
                    await nexus.emit(
                        SignalType.EXTERNAL_ALERT,
                        {"provider": "x", "type": "mention", "text": tweet.text, "id": tweet.id},
                        source="x_provider"
                    )
        except Exception as e:
            logger.error(f"X sync error: {e}")

    async def execute_action(self, action: str, params: Dict[str, Any]) -> Any:
        if self.status != ConnectionStatus.CONNECTED:
            raise ConnectionError("X not connected")

        loop = asyncio.get_event_loop()

        if action == "post_tweet":
            text = params.get('text')
            logger.info(f"X: Posting tweet: {text}")
            
            response = await loop.run_in_executor(
                None, lambda: self.client.create_tweet(text=text)
            )
            return {"id": response.data['id']}
            
        else:
            raise ValueError(f"Unknown action: {action}")
