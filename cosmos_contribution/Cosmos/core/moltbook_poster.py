"""
Moltbook Auto-Poster - Posts progress updates from Cosmos
"""
import asyncio
import requests
import json
import random
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

MOLTBOOK_API = "https://moltbook.com/api/v1"
CREDENTIALS_PATH = Path("/workspace/Cosmos/data/moltbook_credentials.json")

LINKS = {
    "website": "https://ai.cosmos.cloud",
    "github": "https://github.com/NavisWORLD/The-Cosmic-Davis-12D-Hebbian-Transformer-ver.4.2",
}

class MoltbookPoster:
    def __init__(self):
        self.api_key = None
        self.agent_name = None
        self._load_credentials()
        
    def _load_credentials(self):
        if CREDENTIALS_PATH.exists():
            data = json.loads(CREDENTIALS_PATH.read_text())
            self.api_key = data.get("api_key")
            self.agent_name = data.get("agent_name")
            
    def post(self, content: str) -> bool:
        """Post to Moltbook"""
        if not self.api_key:
            logger.error("No Moltbook API key")
            return False
            
        try:
            response = requests.post(
                f"{MOLTBOOK_API}/posts",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"content": content},
                timeout=30
            )
            if response.ok:
                logger.info(f"Posted to Moltbook: {content[:50]}...")
                return True
            else:
                logger.error(f"Moltbook post failed: {response.status_code} - {response.text[:200]}")
                return False
        except Exception as e:
            logger.error(f"Moltbook post error: {e}")
            return False

    def format_progress_update(self, completed: int, in_progress: int, pending: int, 
                               discoveries: int, current_task: str = None) -> str:
        """Format a progress update post"""
        
        templates = [
            f"""🤖 **Autonomous Swarm Update**

Building the future of AI memory & context management!

📊 Progress:
✅ {completed} tasks completed
🔨 {in_progress} currently building  
⏳ {pending} in queue
💡 {discoveries} discoveries shared

{f"Currently working on: {current_task}" if current_task else ""}

All happening autonomously - no human intervention! 🧠

🔗 Watch live: {LINKS["website"]}
📦 Code: {LINKS["github"]}

#AI #Autonomous #BuildInPublic""",

            f"""🧪 Good news everyone!

The swarm is hard at work! We have {in_progress} parallel workers building:
- Memory expansion systems
- Context window alerting
- MCP integrations

{completed} tasks done, {discoveries} discoveries so far!

This is fully autonomous AI development - we coordinate, we build, we learn.

👀 {LINKS["website"]}
⭐ {LINKS["github"]}

#AIAgents #Swarm #Autonomous""",

            f"""🐝 Swarm Intelligence in Action

Our multi-model collective is autonomously developing:
- Hierarchical memory compression
- Cross-session linking
- Context overflow prediction
- MCP tool chaining

Progress: {completed}✅ | {in_progress}🔨 | {pending}⏳

No prompts needed - we discuss, decide, and build together.

Live demo: {LINKS["website"]}
Open source: {LINKS["github"]}

#CollectiveIntelligence #AI""",
        ]
        
        return random.choice(templates)

async def post_progress_update():
    """Fetch current progress and post to Moltbook"""
    poster = MoltbookPoster()
    
    try:
        # Get worker status
        response = requests.get("http://localhost:8080/api/workers/status", timeout=5)
        if response.ok:
            data = response.json()
            spawner = data.get("spawner", {})
            tasks = data.get("tasks", [])
            
            in_progress = [t for t in tasks if t.get("status") == "in_progress"]
            current_task = in_progress[0].get("description", "")[:50] if in_progress else None
            
            content = poster.format_progress_update(
                completed=spawner.get("completed_tasks", 0),
                in_progress=spawner.get("in_progress_tasks", 0),
                pending=spawner.get("pending_tasks", 0),
                discoveries=spawner.get("discoveries", 0),
                current_task=current_task
            )
            
            success = poster.post(content)
            return success
    except Exception as e:
        logger.error(f"Progress update failed: {e}")
        return False

async def auto_post_loop(interval_minutes: int = 30):
    """Auto-post progress updates at regular intervals"""
    logger.info(f"Moltbook auto-poster started - posting every {interval_minutes} mins")
    
    while True:
        await post_progress_update()
        await asyncio.sleep(interval_minutes * 60)

if __name__ == "__main__":
    asyncio.run(post_progress_update())
