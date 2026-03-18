"""Business logic for Mission Control task management."""

import json
import hashlib
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .database import MissionControlDatabase
from .notifications import CLINotifier

logger = logging.getLogger(__name__)


class TaskManager:
    """Manages MC tasks: creation, updates, assignments, and lifecycle."""
    
    def __init__(self, db: MissionControlDatabase, notifier: CLINotifier,
                 agent_name: str = "hermes-cli", auto_accept: bool = True):
        self.db = db
        self.notifier = notifier
        self.agent_name = agent_name
        self.auto_accept = auto_accept
        
    def generate_event_id(self, event_type: str, timestamp: int, 
                         data: Dict[str, Any]) -> str:
        """Generate unique event ID for idempotency."""
        data_json = json.dumps(data, sort_keys=True)
        data_hash = hashlib.sha256(data_json.encode()).hexdigest()[:16]
        return f"{event_type}:{timestamp}:{data_hash}"
        
    def is_duplicate(self, event_id: str) -> bool:
        """Check if event was already processed."""
        return self.db.is_duplicate_event(event_id)
        
    def handle_task_created(self, data: Dict[str, Any]) -> bool:
        """Process task creation webhook."""
        task_id = data.get("id")
        title = data.get("title", "Untitled")
        priority = data.get("priority", "medium")
        assigned_to = data.get("assigned_to")
        
        logger.info("[mc] Handling task_created: %d - %s", task_id, title)
        
        # Store task
        if not self.db.create_task(data):
            return False
            
        # Notify
        self.notifier.task_created(task_id, title, priority, assigned_to)
        
        # Auto-accept if assigned to us
        if assigned_to == self.agent_name and self.auto_accept:
            self.accept_task(task_id, title)
            
        return True
        
    def handle_task_updated(self, data: Dict[str, Any]) -> bool:
        """Process task update webhook."""
        task_id = data.get("id")
        if not task_id:
            logger.warning("[mc] Task update missing ID")
            return False
            
        # Get existing task to compare
        existing = self.db.get_task(task_id)
        if not existing:
            # Task doesn't exist yet, treat as create
            return self.handle_task_created(data)
            
        # Build updates
        updates = {}
        for field in ["title", "description", "priority", "assigned_to", 
                      "project_id", "workspace_id"]:
            if field in data:
                updates[field] = data[field]
                
        if "metadata" in data:
            updates["metadata"] = data["metadata"]
            
        if updates:
            if self.db.update_task(task_id, updates):
                logger.info("[mc] Task %d updated", task_id)
                return True
                
        return False
        
    def handle_task_status_changed(self, data: Dict[str, Any]) -> bool:
        """Process task status change."""
        task_id = data.get("id")
        new_status = data.get("status")
        old_status = data.get("old_status", "unknown")
        title = data.get("title", "Untitled")
        
        if not task_id or not new_status:
            logger.warning("[mc] Status change missing required fields")
            return False
            
        # Update in DB
        updates = {"status": new_status}
        if not self.db.update_task(task_id, updates):
            return False
            
        # Notify
        self.notifier.task_status_changed(task_id, title, old_status, new_status)
        
        # Handle completion
        if new_status == "done":
            self.db.complete_task(task_id)
            self.notifier.task_completed(task_id, title)
            
        return True
        
    def accept_task(self, task_id: int, title: Optional[str] = None) -> bool:
        """Accept a task assignment."""
        if self.db.accept_task(task_id):
            self.notifier.task_accepted(task_id, title or f"Task {task_id}")
            return True
        return False
        
    def handle_agent_status_changed(self, data: Dict[str, Any]) -> bool:
        """Process agent status change."""
        agent_id = data.get("id")
        agent_name = data.get("name", f"Agent {agent_id}")
        status = data.get("status", "unknown")
        
        logger.info("[mc] Agent %s status changed to %s", agent_name, status)
        
        # Update in DB (if we track agents)
        # self.db.update_agent(agent_id, {"status": status})
        
        # Notify
        self.notifier.agent_status_changed(agent_name, status)
        
        # Special handling for error status
        if status == "error":
            error_msg = data.get("error_message", "Unknown error")
            self.notifier.agent_error(agent_name, error_msg)
            
        return True
        
    def handle_unknown_event(self, event_type: str, data: Dict[str, Any]) -> bool:
        """Handle unrecognized event types gracefully."""
        logger.info("[mc] Unknown event type received: %s", event_type)
        # Log but don't fail
        return True