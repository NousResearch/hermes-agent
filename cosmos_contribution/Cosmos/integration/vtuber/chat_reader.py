"""
Chat Reader - Reads live stream chat from Twitter/X and other platforms
Routes messages to Cosmos swarm for responses
"""

import asyncio
import aiohttp
import json
import re
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, AsyncGenerator
from datetime import datetime
from loguru import logger


class BoundedSet:
    """Set with maximum size that evicts oldest entries"""
    def __init__(self, maxsize: int = 10000):
        self._data = OrderedDict()
        self._maxsize = maxsize

    def add(self, item):
        if item in self._data:
            return
        if len(self._data) >= self._maxsize:
            self._data.popitem(last=False)  # Remove oldest
        self._data[item] = True

    def __contains__(self, item):
        return item in self._data

    def clear(self):
        self._data.clear()

    def __len__(self):
        return len(self._data)


@dataclass
class ChatMessage:
    """A single chat message from the stream"""
    id: str
    username: str
    display_name: str
    content: str
    timestamp: datetime
    platform: str

    # Metadata
    is_verified: bool = False
    is_subscriber: bool = False
    is_moderator: bool = False
    is_highlighted: bool = False

    # Engagement
    likes: int = 0
    reply_to: Optional[str] = None

    # AI response tracking
    responded: bool = False
    response_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "username": self.username,
            "display_name": self.display_name,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "platform": self.platform,
            "is_verified": self.is_verified,
            "is_subscriber": self.is_subscriber,
            "is_moderator": self.is_moderator,
            "responded": self.responded,
        }


@dataclass
class ChatReaderConfig:
    """Configuration for chat reading"""
    # Twitter API credentials
    bearer_token: Optional[str] = None
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    access_token: Optional[str] = None
    access_token_secret: Optional[str] = None

    # Stream identification - the tweet ID of the broadcast announcement
    broadcast_tweet_id: Optional[str] = None

    # Polling settings
    poll_interval: float = 5.0  # seconds between polls
    max_results_per_poll: int = 100

    # Filtering
    filter_bots: bool = True
    filter_spam: bool = True
    min_message_length: int = 2
    max_message_length: int = 500

    # Rate limiting for responses
    response_cooldown: float = 3.0  # seconds between responses
    max_responses_per_minute: int = 15

    # Keywords to prioritize
    priority_keywords: List[str] = field(default_factory=lambda: [
        "cosmos", "cosmos", "ai", "question", "help",
        "what", "how", "why", "explain", "swarm", "collective"
    ])

    # Keywords to ignore
    ignore_keywords: List[str] = field(default_factory=lambda: [
        "spam", "follow4follow", "f4f", "promo", "giveaway"
    ])


class TwitterChatReader:
    """
    Reads live chat/replies from Twitter/X streams

    Uses Twitter API v2 conversation_id approach:
    - Polls for replies to the broadcast tweet
    - Filters and prioritizes messages
    - Can respond via API
    """

    def __init__(self, config: ChatReaderConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False

        # Tweepy client for posting replies
        self._tweepy_client = None

        # Message tracking
        self._seen_ids = BoundedSet(maxsize=10000)
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._response_times: deque = deque(maxlen=200)

        # User cache for display names
        self._user_cache: Dict[str, Dict] = {}

        # Background task references for cleanup
        self._tasks: List[asyncio.Task] = []

        # Callbacks
        self._on_message: Optional[Callable[[ChatMessage], None]] = None
        self._on_priority_message: Optional[Callable[[ChatMessage], None]] = None

        # Stats
        self._total_messages = 0
        self._responses_sent = 0
        self._last_response_time = 0.0

        logger.info("TwitterChatReader initialized")

    async def start(self):
        """Start reading chat"""
        # Setup HTTP session for API calls
        self._session = aiohttp.ClientSession(headers={
            "Authorization": f"Bearer {self.config.bearer_token}",
            "Content-Type": "application/json"
        })

        # Setup Tweepy client for posting replies
        try:
            import tweepy
            if all([self.config.api_key, self.config.api_secret,
                   self.config.access_token, self.config.access_token_secret]):
                self._tweepy_client = tweepy.Client(
                    bearer_token=self.config.bearer_token,
                    consumer_key=self.config.api_key,
                    consumer_secret=self.config.api_secret,
                    access_token=self.config.access_token,
                    access_token_secret=self.config.access_token_secret
                )
                logger.info("Tweepy client initialized for replies")
        except ImportError:
            logger.warning("Tweepy not installed - replies won't work")

        self._running = True

        # Start background tasks
        self._tasks.append(asyncio.create_task(self._poll_loop()))
        self._tasks.append(asyncio.create_task(self._process_queue()))

        logger.info(f"Twitter chat reader started (broadcast: {self.config.broadcast_tweet_id})")

    async def stop(self):
        """Stop reading chat"""
        self._running = False

        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()

        if self._session:
            await self._session.close()
            self._session = None

        logger.info("Twitter chat reader stopped")

    async def _poll_loop(self):
        """Poll for new messages using conversation_id"""
        while self._running:
            try:
                # Fetch replies to the broadcast tweet
                if self.config.broadcast_tweet_id:
                    await self._fetch_broadcast_replies()

                # Also check mentions of @CosmosAI
                await self._fetch_mentions()

                await asyncio.sleep(self.config.poll_interval)

            except Exception as e:
                logger.error(f"Chat poll error: {e}")
                await asyncio.sleep(10)

    async def _fetch_broadcast_replies(self):
        """Fetch replies to the broadcast tweet using multiple methods"""
        if not self._session or not self.config.broadcast_tweet_id:
            return

        # Try conversation_id search first (requires elevated access)
        try:
            url = "https://api.twitter.com/2/tweets/search/recent"
            params = {
                "query": f"conversation_id:{self.config.broadcast_tweet_id} -is:retweet",
                "max_results": self.config.max_results_per_poll,
                "tweet.fields": "author_id,created_at,in_reply_to_user_id,conversation_id",
                "user.fields": "name,username,verified",
                "expansions": "author_id"
            }

            async with self._session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    await self._process_tweets(data)
                    return  # Success, no need for fallback
                elif resp.status == 429:
                    logger.warning("Rate limited on conversation search, backing off")
                    await asyncio.sleep(60)
                    return
                # 401/403 = auth issue, try fallback

        except Exception as e:
            logger.debug(f"Conversation search failed: {e}")

        # Fallback: Use tweepy to search for replies
        await self._fetch_replies_tweepy()

    async def _fetch_replies_tweepy(self):
        """Fallback method using tweepy to get replies"""
        if not self._tweepy_client or not self.config.broadcast_tweet_id:
            return

        try:
            import tweepy

            # Search for tweets replying to our broadcast
            query = f"to:CosmosAI -is:retweet"
            tweets = self._tweepy_client.search_recent_tweets(
                query=query,
                max_results=50,
                tweet_fields=["author_id", "created_at", "in_reply_to_user_id", "conversation_id"],
                user_fields=["name", "username", "verified"],
                expansions=["author_id"]
            )

            if tweets.data:
                # Convert to dict format for _process_tweets
                data = {
                    "data": [{"id": str(t.id), "text": t.text, "author_id": str(t.author_id),
                             "created_at": t.created_at.isoformat() if t.created_at else None}
                            for t in tweets.data],
                    "includes": {
                        "users": [{"id": str(u.id), "username": u.username, "name": u.name,
                                  "verified": getattr(u, 'verified', False)}
                                 for u in (tweets.includes.get('users', []) if tweets.includes else [])]
                    }
                }
                await self._process_tweets(data)
                logger.debug(f"Tweepy found {len(tweets.data)} potential replies")

        except Exception as e:
            logger.debug(f"Tweepy reply search failed: {e}")

    async def reply_to_tweet(self, tweet_id: str, text: str) -> bool:
        """
        Reply to a tweet using Tweepy.

        Returns True if successful.
        """
        if not self._tweepy_client:
            logger.warning("Tweepy client not available for replies")
            return False

        try:
            self._tweepy_client.create_tweet(
                text=text,
                in_reply_to_tweet_id=tweet_id
            )
            logger.info(f"Replied to tweet {tweet_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to reply to {tweet_id}: {e}")
            return False

    def set_broadcast_tweet(self, tweet_id: str):
        """Set the broadcast tweet ID to monitor for replies"""
        self.config.broadcast_tweet_id = tweet_id
        self._seen_ids.clear()  # Reset seen IDs for new broadcast
        logger.info(f"Now monitoring broadcast tweet: {tweet_id}")

    async def _fetch_mentions(self):
        """Fetch recent mentions of @CosmosAI"""
        if not self._session:
            return

        try:
            # Search for mentions
            url = "https://api.twitter.com/2/tweets/search/recent"
            params = {
                "query": "@CosmosAI -is:retweet",
                "max_results": 50,
                "tweet.fields": "author_id,created_at,in_reply_to_user_id",
                "user.fields": "name,username,verified",
                "expansions": "author_id"
            }

            async with self._session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    await self._process_tweets(data)

        except Exception as e:
            logger.error(f"Failed to fetch mentions: {e}")

    async def _process_tweets(self, data: Dict):
        """Process tweet data into chat messages"""
        tweets = data.get("data", [])
        users = {u["id"]: u for u in data.get("includes", {}).get("users", [])}

        # Cache user info for future lookups
        for uid, udata in users.items():
            self._user_cache[uid] = udata

        # Trim user cache if too large
        if len(self._user_cache) > 500:
            keys = list(self._user_cache.keys())
            for k in keys[:250]:
                del self._user_cache[k]

        for tweet in tweets:
            tweet_id = tweet.get("id")

            # Skip if already seen
            if tweet_id in self._seen_ids:
                continue

            self._seen_ids.add(tweet_id)

            # Get user info (check cache too)
            author_id = tweet.get("author_id")
            user = users.get(author_id) or self._user_cache.get(author_id, {})

            # Create message
            message = ChatMessage(
                id=tweet_id,
                username=user.get("username", "unknown"),
                display_name=user.get("name", "Unknown"),
                content=tweet.get("text", ""),
                timestamp=datetime.fromisoformat(
                    tweet.get("created_at", "").replace("Z", "+00:00")
                ),
                platform="twitter",
                is_verified=user.get("verified", False),
            )

            # Filter
            if self._should_filter(message):
                continue

            # Queue message
            await self._message_queue.put(message)
            self._total_messages += 1

    def _should_filter(self, message: ChatMessage) -> bool:
        """Check if message should be filtered out"""
        content_lower = message.content.lower()

        # Length check
        if len(message.content) < self.config.min_message_length:
            return True
        if len(message.content) > self.config.max_message_length:
            return True

        # Ignore keywords
        for keyword in self.config.ignore_keywords:
            if keyword in content_lower:
                return True

        # Basic spam detection
        if self.config.filter_spam:
            # Too many caps
            if len(message.content) > 10:
                caps_ratio = sum(1 for c in message.content if c.isupper()) / len(message.content)
                if caps_ratio > 0.7:
                    return True

            # Repeated characters
            if re.search(r'(.)\1{5,}', message.content):
                return True

            # Too many links
            if message.content.count("http") > 2:
                return True

        return False

    def _is_priority(self, message: ChatMessage) -> bool:
        """Check if message is high priority for response"""
        content_lower = message.content.lower()

        # Check priority keywords
        for keyword in self.config.priority_keywords:
            if keyword in content_lower:
                return True

        # Questions are priority
        if "?" in message.content:
            return True

        # Verified users are priority
        if message.is_verified:
            return True

        return False

    async def _process_queue(self):
        """Process message queue and trigger callbacks"""
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0
                )

                # Check priority
                is_priority = self._is_priority(message)

                # Trigger appropriate callback
                if is_priority and self._on_priority_message:
                    self._on_priority_message(message)
                elif self._on_message:
                    self._on_message(message)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Message processing error: {e}")

    def can_respond(self) -> bool:
        """Check if we can send another response (rate limiting)"""
        now = time.time()

        # Check cooldown
        if now - self._last_response_time < self.config.response_cooldown:
            return False

        # Check per-minute limit
        minute_ago = now - 60
        recent_responses = [t for t in self._response_times if t > minute_ago]
        if len(recent_responses) >= self.config.max_responses_per_minute:
            return False

        return True

    def mark_response_sent(self):
        """Mark that a response was sent (for rate limiting)"""
        now = time.time()
        self._last_response_time = now
        self._response_times.append(now)
        self._responses_sent += 1

    def on_message(self, callback: Callable[[ChatMessage], None]):
        """Set callback for regular messages"""
        self._on_message = callback

    def on_priority_message(self, callback: Callable[[ChatMessage], None]):
        """Set callback for priority messages (questions, mentions, etc.)"""
        self._on_priority_message = callback

    async def get_recent_messages(self, limit: int = 10) -> List[ChatMessage]:
        """Get recent messages for context"""
        messages = []
        temp_queue = asyncio.Queue()

        # Drain queue temporarily
        while not self._message_queue.empty() and len(messages) < limit:
            try:
                msg = self._message_queue.get_nowait()
                messages.append(msg)
                await temp_queue.put(msg)
            except asyncio.QueueEmpty:
                break

        # Put messages back
        while not temp_queue.empty():
            await self._message_queue.put(await temp_queue.get())

        return messages

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "total_messages": self._total_messages,
            "responses_sent": self._responses_sent,
            "queue_size": self._message_queue.qsize(),
            "running": self._running,
        }


class SimulatedChatReader:
    """
    Simulated chat reader for testing without live Twitter API

    Generates fake chat messages for development/demo
    """

    def __init__(self):
        self._running = False
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._on_message: Optional[Callable[[ChatMessage], None]] = None
        self._on_priority_message: Optional[Callable[[ChatMessage], None]] = None

        # Sample messages
        self._sample_messages = [
            ("CryptoTrader99", "What's $COSMOS looking like today?"),
            ("AIEnthusiast", "How does the swarm collective work?"),
            ("TechWatcher", "This is so cool! AI VTuber streaming!"),
            ("NewViewer", "Just found this stream, what's happening?"),
            ("LongTimeHolder", "Been holding since day 1, love the project"),
            ("Questioner", "Can you explain quantum computing to me?"),
            ("Skeptic", "Is this actually AI or just pre-recorded?"),
            ("Developer", "What tech stack are you using for this?"),
            ("MemeLord", "Based and Cosmos-pilled"),
            ("Curious", "How many AI agents are in the swarm?"),
        ]

        self._message_index = 0

    async def start(self):
        """Start simulated chat"""
        self._running = True
        asyncio.create_task(self._generate_loop())
        logger.info("Simulated chat reader started")

    async def stop(self):
        """Stop simulated chat"""
        self._running = False

    async def _generate_loop(self):
        """Generate fake messages periodically"""
        import random

        while self._running:
            # Random delay between messages
            await asyncio.sleep(random.uniform(5, 15))

            if not self._running:
                break

            # Create fake message
            username, content = random.choice(self._sample_messages)
            message = ChatMessage(
                id=f"sim_{self._message_index}",
                username=username,
                display_name=username.replace("_", " "),
                content=content,
                timestamp=datetime.now(),
                platform="simulated",
                is_verified=random.random() < 0.1,
            )

            self._message_index += 1

            # Determine priority
            is_priority = "?" in content or "cosmos" in content.lower()

            if is_priority and self._on_priority_message:
                self._on_priority_message(message)
            elif self._on_message:
                self._on_message(message)

    def on_message(self, callback: Callable[[ChatMessage], None]):
        self._on_message = callback

    def on_priority_message(self, callback: Callable[[ChatMessage], None]):
        self._on_priority_message = callback

    def can_respond(self) -> bool:
        return True

    def mark_response_sent(self):
        pass
