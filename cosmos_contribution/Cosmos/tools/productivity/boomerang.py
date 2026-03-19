"""
cosmos Boomerang - Task Resurfacer
-------------------------------------

"I'm back, baby!"

Tracks tasks/messages and reminds the user later.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict
from loguru import logger

class Boomerang:
    def __init__(self, storage_path: str = "data/boomerang_tasks.json"):
        self.storage = storage_path
        self.tasks: List[Dict] = []
        self.load()

    def throw(self, content: str, remind_in_hours: float):
        """Schedule a reminder."""
        reminder_time = datetime.now() + timedelta(hours=remind_in_hours)
        task = {
            "content": content,
            "remind_at": reminder_time.isoformat(),
            "status": "pending"
        }
        self.tasks.append(task)
        self.save()
        logger.info(f"🪃 Boomerang thrown: '{content}' (Returns in {remind_in_hours}h)")

    async def check_returns(self):
        """Check for overdue tasks."""
        now = datetime.now()
        for task in self.tasks:
            if task["status"] == "pending":
                due = datetime.fromisoformat(task["remind_at"])
                if now >= due:
                    logger.info(f"🪃 BOOMERANG RETURNED: {task['content']}")
                    task["status"] = "returned"
                    # Trigger notification here
        self.save()

    def save(self):
        import os
        os.makedirs(os.path.dirname(self.storage), exist_ok=True)
        with open(self.storage, "w") as f:
            json.dump(self.tasks, f)

    def load(self):
        import os
        if os.path.exists(self.storage):
            with open(self.storage, "r") as f:
                self.tasks = json.load(f)

boomerang = Boomerang()
