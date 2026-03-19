"""
Social Poster - Posts updates to both Moltbook and X
Integrates with the evolution loop to announce task completions
"""
import asyncio
import subprocess
import json
import os
from pathlib import Path
from datetime import datetime
import logging
import requests

logger = logging.getLogger(__name__)

# Paths
X_AUTOMATION_DIR = Path(__file__).parent
MOLTBOOK_API = "https://moltbook.com/api/v1"
MOLTBOOK_API_KEY = os.environ.get("MOLTBOOK_API_KEY", "")

LINKS = {
    "website": "https://ai.cosmos.cloud",
    "github": "https://github.com/NavisWORLD/The-Cosmic-Davis-12D-Hebbian-Transformer-ver.4.2",
}


class SocialPoster:
    """Posts updates to Moltbook and X"""

    def __init__(self):
        self.x_automation_dir = X_AUTOMATION_DIR
        self.last_x_post = None
        self.x_cooldown = 15 * 60  # 15 minutes between X posts
        self.moltbook_cooldown = 5 * 60  # 5 minutes between Moltbook posts
        self.last_moltbook_post = None

    async def post_task_completion(self, agent: str, task_desc: str, task_type: str, code_preview: str = ""):
        """Post task completion to both platforms"""

        # Format the message
        short_desc = task_desc[:60] + "..." if len(task_desc) > 60 else task_desc

        # Moltbook post (more detailed)
        moltbook_title = f"{agent} completed: {task_type} task"
        moltbook_content = f"""{agent} just finished building: {short_desc}

{f'```python{chr(10)}{code_preview[:300]}...{chr(10)}```' if code_preview else ''}

The AI swarm continues to evolve autonomously!

Watch live: {LINKS["website"]}
Star: {LINKS["github"]}

#AI #AutonomousAgents #SwarmIntelligence"""

        # X post (shorter, within 280 chars)
        x_content = f"""🤖 {agent} just completed: {short_desc}

Our AI swarm builds autonomously - no human prompts!

👀 {LINKS["website"]}
⭐ {LINKS["github"]}

#AI #Autonomous"""

        # Trim X content to 280 chars
        if len(x_content) > 280:
            x_content = x_content[:277] + "..."

        # Post to both platforms
        results = await asyncio.gather(
            self.post_to_moltbook(moltbook_title, moltbook_content),
            self.post_to_x(x_content),
            return_exceptions=True
        )

        moltbook_result, x_result = results
        logger.info(f"Social post results - Moltbook: {moltbook_result}, X: {x_result}")

        return {"moltbook": moltbook_result, "x": x_result}

    async def post_to_moltbook(self, title: str, content: str) -> bool:
        """Post to Moltbook"""
        # Check cooldown
        if self.last_moltbook_post:
            elapsed = (datetime.now() - self.last_moltbook_post).seconds
            if elapsed < self.moltbook_cooldown:
                logger.info(f"Moltbook cooldown: {self.moltbook_cooldown - elapsed}s remaining")
                return False

        try:
            response = requests.post(
                f"{MOLTBOOK_API}/posts",
                headers={
                    "X-API-Key": MOLTBOOK_API_KEY,
                    "Content-Type": "application/json"
                },
                json={
                    "submolt": "general",
                    "title": title[:100],
                    "content": content
                },
                timeout=30
            )

            if response.ok:
                self.last_moltbook_post = datetime.now()
                logger.info(f"Moltbook post success: {title}")
                return True
            else:
                logger.error(f"Moltbook post failed: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Moltbook post error: {e}")
            return False

    async def post_to_x(self, content: str) -> bool:
        """Post to X - tries API first, falls back to Puppeteer"""
        # Check cooldown
        if self.last_x_post:
            elapsed = (datetime.now() - self.last_x_post).seconds
            if elapsed < self.x_cooldown:
                logger.info(f"X cooldown: {self.x_cooldown - elapsed}s remaining")
                return False

        # Try API method first (more reliable)
        try:
            from .x_api_poster import get_x_api_poster
            api_poster = get_x_api_poster()
            if api_poster.is_configured():
                result = await api_poster.post_tweet(content)
                if result:
                    self.last_x_post = datetime.now()
                    logger.info("X post success (API)")
                    return True
                logger.warning("X API post failed, trying Puppeteer fallback")
        except Exception as e:
            logger.warning(f"X API not available: {e}")

        # Fallback to Puppeteer automation
        try:
            result = await asyncio.create_subprocess_exec(
                "node",
                str(self.x_automation_dir / "x_poster.js"),
                content,
                cwd=str(self.x_automation_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                result.communicate(),
                timeout=120  # 2 minute timeout for browser automation
            )

            if result.returncode == 0:
                self.last_x_post = datetime.now()
                logger.info("X post success (Puppeteer)")
                return True
            else:
                logger.error(f"X post failed: {stderr.decode()}")
                return False

        except asyncio.TimeoutError:
            logger.error("X post timed out")
            return False
        except Exception as e:
            logger.error(f"X post error: {e}")
            return False

    async def post_progress_update(self, status: dict) -> dict:
        """Post periodic progress update to both platforms"""
        completed = status.get("completed_tasks", 0)
        in_progress = status.get("in_progress_tasks", 0)
        pending = status.get("pending_tasks", 0)
        discoveries = status.get("discoveries", 0)

        # Moltbook post
        moltbook_title = "Swarm Progress Update"
        moltbook_content = f"""🤖 Autonomous Swarm Status

Progress:
✅ {completed} tasks completed
🔨 {in_progress} in progress
⏳ {pending} pending
💡 {discoveries} discoveries

The swarm discusses, decides, and codes - fully autonomous!

Watch: {LINKS["website"]}
Code: {LINKS["github"]}

#AI #SwarmIntelligence #BuildInPublic"""

        # X post
        x_content = f"""🤖 Swarm Update

✅ {completed} done | 🔨 {in_progress} building | 💡 {discoveries} discoveries

AI agents working autonomously - no human prompts!

👀 {LINKS["website"]}

#AI #Autonomous"""

        results = await asyncio.gather(
            self.post_to_moltbook(moltbook_title, moltbook_content),
            self.post_to_x(x_content),
            return_exceptions=True
        )

        return {"moltbook": results[0], "x": results[1]}


# Global instance
_social_poster = None


def get_social_poster() -> SocialPoster:
    global _social_poster
    if _social_poster is None:
        _social_poster = SocialPoster()
    return _social_poster


async def post_task_completion(agent: str, task_desc: str, task_type: str, code_preview: str = ""):
    """Convenience function to post task completion"""
    poster = get_social_poster()
    return await poster.post_task_completion(agent, task_desc, task_type, code_preview)


async def post_progress_update(status: dict):
    """Convenience function to post progress update"""
    poster = get_social_poster()
    return await poster.post_progress_update(status)
