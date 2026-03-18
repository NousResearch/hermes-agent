"""
Integration module for parallel task execution in Hermes Gateway.

This module provides the integration points between the GatewayRunner
and the ParallelTaskManager, enabling concurrent task execution.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

# Import the parallel task manager
from gateway.parallel_task_manager import (
    ParallelTaskManager,
    ParallelTask,
    TaskStatus,
    get_task_manager,
)
from gateway.task_classifier import TaskType

logger = logging.getLogger(__name__)


class ParallelExecutionIntegration:
    """
    Integration layer for parallel task execution in the gateway.
    
    This class wraps the ParallelTaskManager and provides gateway-specific
    integration for running AI agent tasks in parallel.
    """
    
    def __init__(self, gateway_runner, max_concurrent: int = 3):
        """
        Initialize the parallel execution integration.
        
        Args:
            gateway_runner: The GatewayRunner instance
            max_concurrent: Maximum number of concurrent tasks
        """
        self.gateway = gateway_runner
        self._task_manager: Optional[ParallelTaskManager] = None
        self._max_concurrent = max_concurrent
        self._enabled = False
    
    async def initialize(self):
        """Initialize the task manager."""
        self._task_manager = await get_task_manager(self._max_concurrent)
        self._enabled = True
        logger.info(f"Parallel execution initialized (max_concurrent={self._max_concurrent})")
    
    async def shutdown(self):
        """Shutdown the task manager."""
        if self._task_manager:
            await self._task_manager.stop()
            self._enabled = False
            logger.info("Parallel execution shutdown complete")
    
    def is_enabled(self) -> bool:
        """Check if parallel execution is enabled."""
        return self._enabled and self._task_manager is not None
    
    async def submit_task(
        self,
        session_key: str,
        message: str,
        event: Any,  # MessageEvent
        conversation_history: Optional[List[Dict]] = None,
    ) -> Optional[ParallelTask]:
        """
        Submit a task for parallel execution.
        
        Args:
            session_key: Unique session identifier
            message: The user's message
            event: The MessageEvent
            conversation_history: Previous conversation history
            
        Returns:
            The submitted task, or None if parallel execution is disabled
        """
        if not self.is_enabled():
            return None
        
        # Create task runner that wraps the gateway's agent execution
        async def task_runner(task: ParallelTask) -> Dict[str, Any]:
            return await self._run_agent_task(task, event)
        
        task = await self._task_manager.submit_task(
            session_key=session_key,
            message=message,
            task_runner=task_runner,
            conversation_history=conversation_history,
        )
        
        # Set up status callback for progress updates
        task.status_callback = lambda t: self._on_task_status_change(t, event)
        
        return task
    
    async def _run_agent_task(
        self,
        task: ParallelTask,
        original_event: Any
    ) -> Dict[str, Any]:
        """
        Run the AI agent for a task.
        
        This wraps the gateway's _run_agent method to execute in parallel.
        """
        # Import here to avoid circular imports
        from gateway.delivery import MessageEvent
        
        # Create a copy of the event for this task
        event_data = {
            "source": original_event.source,
            "text": task.message,
            "message_type": original_event.message_type,
            "platform_message": original_event.platform_message,
        }
        
        # Build context
        context_prompt = self.gateway._build_context_prompt(original_event.source)
        
        # Get session
        session = self.gateway._get_or_create_session(original_event.source)
        session_id = session.session_id
        
        # Get conversation history
        history = self.gateway._get_conversation_history(session_id)
        
        # Run the agent
        try:
            result = await self.gateway._run_agent(
                message=task.message,
                context_prompt=context_prompt,
                history=history,
                source=original_event.source,
                session_id=session_id,
                session_key=task.session_key,
            )
            
            # Send the response
            response_text = result.get("final_response", "")
            if response_text:
                await self._send_task_response(task, original_event, response_text)
            
            return result
            
        except Exception as e:
            logger.error(f"Task {task.task_id[:8]} execution error: {e}")
            # Send error message
            await self._send_task_response(
                task, 
                original_event, 
                f"❌ Error processing your request: {str(e)}"
            )
            raise
    
    async def _send_task_response(
        self,
        task: ParallelTask,
        event: Any,
        response_text: str
    ):
        """Send a response for a completed task."""
        # Add task identifier to response
        if self._max_concurrent > 1:
            header = f"📋 Task completed (ID: {task.task_id[:8]})\n"
            if task.classification.task_type == TaskType.INDEPENDENT:
                header = f"⚡ Quick task completed\n"
            response_text = header + "─" * 30 + "\n" + response_text
        
        # Send via the appropriate adapter
        adapter = self.gateway.adapters.get(event.source.platform)
        if adapter:
            await adapter.send(event.source.chat_id, response_text)
    
    def _on_task_status_change(self, task: ParallelTask, event: Any):
        """Handle task status changes for progress updates."""
        # Could send progress updates to the user here
        pass
    
    def get_session_tasks(self, session_key: str) -> List[Dict[str, Any]]:
        """Get all tasks for a session as dictionaries."""
        if not self._task_manager:
            return []
        
        tasks = self._task_manager.get_session_tasks(session_key)
        return [task.to_dict() for task in tasks]
    
    def get_running_tasks(self, session_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get running tasks as dictionaries."""
        if not self._task_manager:
            return []
        
        tasks = self._task_manager.get_running_tasks(session_key)
        return [task.to_dict() for task in tasks]
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        if not self._task_manager:
            return False
        return await self._task_manager.cancel_task(task_id)
    
    async def cancel_session_tasks(self, session_key: str) -> int:
        """Cancel all tasks for a session."""
        if not self._task_manager:
            return 0
        return await self._task_manager.cancel_session_tasks(session_key)
    
    def should_use_parallel(self, message: str, session_key: str) -> bool:
        """
        Determine if a message should use parallel execution.
        
        This is a quick check before full classification.
        """
        if not self.is_enabled():
            return False
        
        # Check if there are already running tasks
        running = self._task_manager.get_running_tasks(session_key)
        if not running:
            return False  # No need for parallel if nothing is running
        
        # Quick keyword check for parallel intent
        parallel_keywords = [
            "meanwhile", "at the same time", "also", "additionally",
            "in parallel", "while you", "can you also"
        ]
        message_lower = message.lower()
        if any(kw in message_lower for kw in parallel_keywords):
            return True
        
        return True  # Default to trying parallel, let classifier decide
    
    def get_stats(self) -> Dict[str, Any]:
        """Get parallel execution statistics."""
        if not self._task_manager:
            return {"enabled": False}
        
        stats = self._task_manager.get_stats()
        stats["enabled"] = self._enabled
        return stats


# Global integration instance
_integration: Optional[ParallelExecutionIntegration] = None


async def initialize_parallel_execution(
    gateway_runner,
    max_concurrent: int = 3
) -> ParallelExecutionIntegration:
    """
    Initialize parallel execution for the gateway.
    
    Args:
        gateway_runner: The GatewayRunner instance
        max_concurrent: Maximum concurrent tasks
        
    Returns:
        The initialized integration
    """
    global _integration
    _integration = ParallelExecutionIntegration(gateway_runner, max_concurrent)
    await _integration.initialize()
    return _integration


def get_parallel_integration() -> Optional[ParallelExecutionIntegration]:
    """Get the global parallel integration instance."""
    return _integration
