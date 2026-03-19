"""
COSMOS MEME SCHEDULER
Posts Borg Professor Cosmos + Lobster memes to X every 2 hours.
Uses Grok/Gemini for image generation and X API v2 for posting.

Identity: Borg-assimilated Professor Cosmos, always cooking/eating lobster
Mission: Promote $COSMOS, promote HermesAgent integration, grow the swarm
"""
import asyncio
import logging
from datetime import datetime
from pathlib import Path
import sys

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from Cosmos.integration.image_gen.generator import get_image_generator
from Cosmos.integration.x_automation.x_api_poster import get_x_api_poster
from Cosmos.integration.x_automation.posting_brain import get_posting_brain

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Posting interval (4 hours in seconds) - slowed down for better variety
POSTING_INTERVAL = 4 * 60 * 60

# History file to track posts
HISTORY_FILE = Path("/workspace/Cosmos/data/meme_history.json")


class MemeScheduler:
    """Schedules and posts Borg Cosmos + Lobster memes to X"""

    def __init__(self):
        self.image_gen = get_image_generator()
        self.x_poster = get_x_api_poster()
        self.brain = get_posting_brain()
        self.posts_made = 0
        self.last_post_time = None

    async def generate_and_post_meme(self) -> bool:
        """Generate a Borg Cosmos meme and post it to X with image"""
        try:
            # Check X API configuration
            if not self.x_poster.is_configured():
                logger.error("X API not configured - cannot post")
                return False

            # Check rate limits
            if not self.x_poster.can_post():
                logger.warning("X API rate limit reached")
                return False

            # Generate Borg Cosmos meme using Gemini with reference images
            # This keeps the character consistent while varying scenes
            logger.info("Generating Borg Cosmos meme with reference images...")
            image_bytes, scene = await self.image_gen.generate_borg_cosmos_meme()
            logger.info(f"Scene: {scene[:60]}...")

            # Determine post type for variety
            import random
            post_types = ["meme", "meme", "dev_update", "hermes_showcase"]
            post_type = random.choice(post_types)
            logger.info(f"Post type: {post_type}")

            # Try Grok for dynamic caption first, fallback to templates
            caption = await self.brain.generate_caption_with_grok(scene, post_type)

            if not caption:
                # Fallback to template-based captions with variety
                caption = self.brain.get_varied_caption(scene)
                logger.info("Using template caption (Grok unavailable)")

            # Format full post with CA and links
            tweet_text = self.brain.format_post(caption)
            logger.info(f"Caption: {caption[:50]}...")

            if not image_bytes:
                logger.error("Failed to generate meme image")
                # Post text-only fallback
                result = await self.x_poster.post_tweet(tweet_text)
                return bool(result)

            # Post with image
            logger.info(f"Posting meme to X: {tweet_text[:50]}...")
            result = await self.x_poster.post_tweet(tweet_text, image_bytes=image_bytes)

            if result:
                self.posts_made += 1
                self.last_post_time = datetime.now()
                tweet_id = result.get("data", {}).get("id", "unknown")
                logger.info(f"Meme posted successfully! Tweet ID: {tweet_id}")
                logger.info(f"Image provider: {self.image_gen.last_provider}")
                logger.info(f"Text source: {'Grok' if caption else 'Template'}")
                return True
            else:
                logger.error("Failed to post meme")
                return False

        except Exception as e:
            logger.error(f"Meme post error: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def run_once(self):
        """Run a single meme post (for testing)"""
        logger.info("=== POSTING SINGLE MEME ===")
        success = await self.generate_and_post_meme()
        logger.info(f"Result: {'SUCCESS' if success else 'FAILED'}")
        return success

    async def run_scheduler(self):
        """Run the scheduler loop - posts every 2 hours"""
        logger.info("=== STARTING MEME SCHEDULER ===")
        logger.info(f"Posting interval: {POSTING_INTERVAL / 3600:.1f} hours")
        logger.info(f"X API configured: {self.x_poster.is_configured()}")
        logger.info(f"Image generators: {self.image_gen.get_status()}")

        # Post immediately on start
        await self.generate_and_post_meme()

        # Then schedule every 2 hours
        while True:
            logger.info(f"Sleeping for {POSTING_INTERVAL / 3600:.1f} hours...")
            await asyncio.sleep(POSTING_INTERVAL)

            logger.info("=== SCHEDULED MEME POST ===")
            await self.generate_and_post_meme()
            logger.info(f"Total posts this session: {self.posts_made}")


# Global instance
_meme_scheduler = None

def get_meme_scheduler() -> MemeScheduler:
    global _meme_scheduler
    if _meme_scheduler is None:
        _meme_scheduler = MemeScheduler()
    return _meme_scheduler


async def post_meme_now():
    """Post a single meme immediately"""
    scheduler = get_meme_scheduler()
    return await scheduler.run_once()


async def start_meme_scheduler():
    """Start the 2-hour meme posting loop"""
    scheduler = get_meme_scheduler()
    await scheduler.run_scheduler()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "once":
        # Post single meme
        asyncio.run(post_meme_now())
    else:
        # Run scheduler loop
        asyncio.run(start_meme_scheduler())
