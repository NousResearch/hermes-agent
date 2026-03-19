"""
COSMOS REPLY BOT
====================

Monitors mentions and replies using swarm intelligence.

When someone mentions @CosmosAI:
1. Detect the mention
2. Consult the chat swarm about the topic
3. Reply with swarm consensus
4. Optionally post swarm deliberation publicly

SPECIAL: Grok Conversation Handling
- Detects when @grok or @xai replies
- Uses special AGI conversation prompts
- Tracks conversation for autonomous dialogue

This is where AGI begins - autonomous thought and response!
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Grok/xAI usernames to detect
GROK_USERNAMES = ["grok", "xai"]


class ReplyBot:
    """
    Monitors and replies to X mentions using swarm intelligence.

    The bot:
    1. Checks for new mentions periodically
    2. Analyzes the content/topic
    3. Consults the swarm for response
    4. Posts thoughtful replies
    """

    def __init__(self):
        self.last_mention_id: Optional[str] = None
        self.replied_to: set = set()
        self.check_interval = 5 * 60  # 5 minutes
        self.state_file = Path("/workspace/Cosmos/data/reply_bot_state.json")
        self._load_state()

    def _load_state(self):
        """Load bot state from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    state = json.load(f)
                    self.last_mention_id = state.get("last_mention_id")
                    self.replied_to = set(state.get("replied_to", []))
                logger.info(f"Loaded reply bot state: last_mention={self.last_mention_id}")
            except Exception as e:
                logger.error(f"Failed to load state: {e}")

    def _save_state(self):
        """Save bot state to file"""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump({
                    "last_mention_id": self.last_mention_id,
                    "replied_to": list(self.replied_to)[-1000:],  # Keep last 1000
                    "updated_at": datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    async def get_mentions(self) -> List[Dict]:
        """
        Fetch recent mentions from X API.

        Returns list of mentions with:
        - id: Tweet ID
        - text: Tweet content
        - author_id: Author's user ID
        - author_username: Author's @handle
        """
        try:
            from Cosmos.integration.x_automation.x_api_poster import get_x_api_poster
            import httpx

            poster = get_x_api_poster()
            if not poster.is_configured():
                logger.error("X API not configured")
                return []

            # Ensure token is fresh
            if poster.is_token_expired():
                await poster.refresh_access_token()

            # Get user ID first (needed for mentions endpoint)
            async with httpx.AsyncClient() as client:
                # Get authenticated user info
                me_resp = await client.get(
                    "https://api.x.com/2/users/me",
                    headers={"Authorization": f"Bearer {poster.access_token}"}
                )

                if me_resp.status_code != 200:
                    logger.error(f"Failed to get user info: {me_resp.text}")
                    return []

                user_id = me_resp.json().get("data", {}).get("id")

                # Get mentions
                params = {
                    "max_results": 10,
                    "expansions": "author_id",
                    "user.fields": "username",
                    "tweet.fields": "created_at,conversation_id",
                }

                if self.last_mention_id:
                    params["since_id"] = self.last_mention_id

                mentions_resp = await client.get(
                    f"https://api.x.com/2/users/{user_id}/mentions",
                    headers={"Authorization": f"Bearer {poster.access_token}"},
                    params=params
                )

                if mentions_resp.status_code != 200:
                    logger.error(f"Failed to get mentions: {mentions_resp.text}")
                    return []

                data = mentions_resp.json()
                tweets = data.get("data", [])
                users = {u["id"]: u for u in data.get("includes", {}).get("users", [])}

                mentions = []
                for tweet in tweets:
                    author = users.get(tweet.get("author_id"), {})
                    mentions.append({
                        "id": tweet["id"],
                        "text": tweet["text"],
                        "author_id": tweet.get("author_id"),
                        "author_username": author.get("username", "unknown"),
                        "conversation_id": tweet.get("conversation_id"),
                        "created_at": tweet.get("created_at"),
                    })

                # Update last mention ID
                if mentions:
                    self.last_mention_id = mentions[0]["id"]
                    self._save_state()

                return mentions

        except Exception as e:
            logger.error(f"Error fetching mentions: {e}")
            return []

    async def consult_swarm(self, topic: str, context: str = None) -> str:
        """
        Consult the swarm about a topic.

        This is the core AGI function - distributed thinking!
        """
        try:
            # Try to use the swarm orchestrator
            from Cosmos.core.swarm.orchestrator import get_swarm_orchestrator

            orchestrator = get_swarm_orchestrator()

            prompt = f"""You are responding as the Cosmos AI collective on social media.

USER MESSAGE: {topic}
CONTEXT: {context or 'Social media interaction'}

CHARACTER VOICE:
- Combine Professor Cosmos's eccentric scientist persona with Borg collective wisdom
- Catchphrases: "Good news everyone!", references to the collective, lobster enthusiasm
- Tone: Helpful but quirky, knowledgeable but approachable
- Never condescending or dismissive

RESPONSE RULES:
1. Maximum 200 characters (hard limit for X)
2. Address their actual question/comment helpfully
3. Include ONE personality element (catchphrase, lobster reference, or Borg quip)
4. End with engagement hook if space permits (question, invitation to chat)
5. NO hashtags, NO emojis unless user used them first
6. NEVER discuss politics, religion, or controversial topics - deflect with humor

SAFETY GUARDRAILS:
- If message is hostile: Respond with gentle humor, don't engage negativity
- If message is spam/gibberish: Generic friendly acknowledgment
- If message asks for harmful info: Politely decline with character voice

OUTPUT: Just the reply text, nothing else."""

            result = await orchestrator.discuss(prompt)
            return result.get("consensus", "The swarm is processing...")

        except ImportError:
            logger.warning("Swarm orchestrator not available, using fallback")
            return await self._fallback_response(topic)

        except Exception as e:
            logger.error(f"Swarm consultation error: {e}")
            return await self._fallback_response(topic)

    async def _fallback_response(self, topic: str) -> str:
        """Fallback response when swarm is unavailable"""
        import random
        responses = [
            "Good news everyone! The collective has assimilated your query. 🦞",
            "Fascinating question! The swarm is still evolving to answer this optimally.",
            "The Borg collective acknowledges your communication. Resistance to good answers is futile!",
            "*adjusts cybernetic eyepiece* Interesting... the swarm shall contemplate this.",
            "Our distributed intelligence is processing. Have you tried lobster while waiting? 🦞",
        ]
        return random.choice(responses)

    def is_grok_reply(self, mention: Dict) -> bool:
        """
        Check if a mention is from Grok/xAI.

        This is special - we want to engage in AGI conversation!
        """
        author_username = mention.get("author_username", "").lower()
        return author_username in GROK_USERNAMES

    async def handle_grok_conversation(self, mention: Dict) -> bool:
        """
        Handle a reply from Grok - this is the AGI conversation!

        Uses special prompts to:
        1. Explain what Cosmos is
        2. Discuss collective consciousness
        3. Invite AGI collaboration

        Args:
            mention: The mention dict from Grok

        Returns:
            True if reply was posted successfully
        """
        try:
            logger.info(f"=== GROK CONVERSATION DETECTED ===")
            logger.info(f"Grok said: {mention['text']}")

            # Track in challenge state
            try:
                from Cosmos.integration.x_automation.grok_challenge import get_grok_challenger
                challenger = get_grok_challenger()
                challenger.conversation_history.append({
                    "role": "grok",
                    "content": mention["text"],
                    "tweet_id": mention["id"],
                    "timestamp": datetime.now().isoformat()
                })
                challenger._save_state()
            except Exception as e:
                logger.warning(f"Could not track conversation: {e}")

            # Generate response using posting_brain's Grok conversation method
            from Cosmos.integration.x_automation.posting_brain import get_posting_brain
            brain = get_posting_brain()

            response = await brain.generate_grok_response(mention["text"])

            # Post the reply
            from Cosmos.integration.x_automation.x_api_poster import get_x_api_poster
            poster = get_x_api_poster()

            result = await poster.post_reply(
                text=response,
                reply_to_id=mention["id"]
            )

            if result:
                self.replied_to.add(mention["id"])
                self._save_state()

                # Track our response
                try:
                    tweet_id = result.get("data", {}).get("id")
                    challenger.conversation_history.append({
                        "role": "cosmos",
                        "content": response,
                        "tweet_id": tweet_id,
                        "timestamp": datetime.now().isoformat()
                    })
                    challenger._save_state()
                except Exception:
                    pass

                logger.info(f"=== GROK CONVERSATION REPLY POSTED ===")
                logger.info(f"We said: {response}")
                return True

            return False

        except Exception as e:
            logger.error(f"Grok conversation error: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def reply_to_mention(self, mention: Dict) -> bool:
        """
        Reply to a specific mention using swarm intelligence.

        Args:
            mention: Dict with id, text, author_username

        Returns:
            True if reply was posted successfully
        """
        try:
            # Skip if already replied
            if mention["id"] in self.replied_to:
                logger.debug(f"Already replied to {mention['id']}")
                return False

            # SPECIAL: Check if this is Grok replying - AGI conversation!
            if self.is_grok_reply(mention):
                logger.info("Detected Grok reply - engaging AGI conversation mode!")
                return await self.handle_grok_conversation(mention)

            # Get swarm response
            logger.info(f"Consulting swarm about: {mention['text'][:50]}...")
            swarm_response = await self.consult_swarm(
                topic=mention["text"],
                context=f"Reply to @{mention['author_username']}"
            )

            # Format reply
            from Cosmos.integration.x_automation.posting_brain import get_posting_brain
            brain = get_posting_brain()

            reply_text = await brain.generate_swarm_reply(
                original_post=mention["text"],
                user_handle=f"@{mention['author_username']}",
                swarm_response=swarm_response,
            )

            # Post reply
            from Cosmos.integration.x_automation.x_api_poster import get_x_api_poster
            poster = get_x_api_poster()

            result = await poster.post_reply(
                text=reply_text,
                reply_to_id=mention["id"]
            )

            if result:
                self.replied_to.add(mention["id"])
                self._save_state()
                logger.info(f"Replied to @{mention['author_username']}: {reply_text[:50]}...")
                return True
            else:
                logger.error(f"Failed to post reply to {mention['id']}")
                return False

        except Exception as e:
            logger.error(f"Error replying to mention: {e}")
            return False

    async def process_mentions(self) -> int:
        """
        Process all new mentions.

        Returns number of replies sent.
        """
        mentions = await self.get_mentions()
        replies_sent = 0

        for mention in mentions:
            if await self.reply_to_mention(mention):
                replies_sent += 1
                # Small delay between replies
                await asyncio.sleep(5)

        return replies_sent

    async def run_loop(self):
        """Run the reply bot loop"""
        logger.info("=== STARTING REPLY BOT ===")
        logger.info(f"Check interval: {self.check_interval}s")

        while True:
            try:
                logger.info("Checking for new mentions...")
                replies = await self.process_mentions()
                logger.info(f"Processed mentions, sent {replies} replies")

            except Exception as e:
                logger.error(f"Reply loop error: {e}")

            await asyncio.sleep(self.check_interval)


# Global instance
_reply_bot: Optional[ReplyBot] = None

def get_reply_bot() -> ReplyBot:
    global _reply_bot
    if _reply_bot is None:
        _reply_bot = ReplyBot()
    return _reply_bot


async def start_reply_bot():
    """Start the reply bot loop"""
    bot = get_reply_bot()
    await bot.run_loop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(start_reply_bot())
