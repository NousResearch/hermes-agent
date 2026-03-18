"""CLI notification handler for Mission Control events."""

import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class CLINotifier:
    """Sends notifications to CLI output for MC events."""
    
    # ANSI color codes
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    
    def __init__(self):
        self.enabled = True
        
    def _colorize(self, text: str, color: str) -> str:
        """Apply ANSI color to text."""
        return f"{color}{text}{self.RESET}"
        
    def _priority_color(self, priority: str) -> str:
        """Get color for priority level."""
        colors = {
            "critical": self.RED,
            "high": self.YELLOW,
            "medium": self.BLUE,
            "low": self.RESET,
        }
        return colors.get(priority.lower(), self.RESET)
        
    def send(self, message: str, level: str = "info"):
        """Send a notification message to CLI."""
        if not self.enabled:
            return
            
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = f"[{timestamp}] [MC]"
        
        if level == "error":
            formatted = self._colorize(f"{prefix} {message}", self.RED)
        elif level == "warning":
            formatted = self._colorize(f"{prefix} {message}", self.YELLOW)
        elif level == "success":
            formatted = self._colorize(f"{prefix} {message}", self.GREEN)
        else:
            formatted = f"{prefix} {message}"
            
        print(formatted)
        logger.info("[mc] Notification: %s", message)
        
    def task_created(self, task_id: int, title: str, priority: str, 
                     assigned_to: Optional[str]):
        """Notify about new task creation."""
        priority_colored = self._colorize(
            priority.upper(), 
            self._priority_color(priority)
        )
        
        if assigned_to:
            msg = f"📋 Task #{task_id} [{priority_colored}] assigned to {assigned_to}: {title}"
        else:
            msg = f"📋 Task #{task_id} [{priority_colored}] created: {title}"
            
        self.send(msg)
        
    def task_accepted(self, task_id: int, title: str):
        """Notify about task acceptance."""
        msg = f"✓ Auto-accepted task #{task_id}: {title}"
        self.send(msg, level="success")
        
    def task_completed(self, task_id: int, title: str):
        """Notify about task completion."""
        msg = f"✓ Task #{task_id} completed: {title}"
        self.send(msg, level="success")
        
    def task_status_changed(self, task_id: int, title: str, 
                           old_status: str, new_status: str):
        """Notify about status change."""
        msg = f"🔄 Task #{task_id} status: {old_status} → {new_status}: {title}"
        self.send(msg)
        
    def agent_status_changed(self, agent_name: str, status: str):
        """Notify about agent status change."""
        if status == "error":
            msg = f"⚠ Agent {agent_name} status: {status}"
            self.send(msg, level="error")
        else:
            msg = f"👤 Agent {agent_name} status: {status}"
            self.send(msg)
            
    def agent_error(self, agent_name: str, error: str):
        """Notify about agent error."""
        msg = f"🚨 Agent {agent_name} ERROR: {error}"
        self.send(msg, level="error")
        
    def webhook_received(self, event_type: str):
        """Debug notification for webhook receipt."""
        logger.debug("[mc] Webhook received: %s", event_type)