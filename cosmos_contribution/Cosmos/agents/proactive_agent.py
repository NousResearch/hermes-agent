"""
cosmos Proactive Agent - Anticipatory Intelligence

Q2 2025 Feature: Proactive Intelligence
- Anticipatory Suggestions: Predict what the user might need next
- Task Automation: Learn and execute repetitive workflows
- Context Awareness: Adapt to time, mood, and project context
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any, Callable
import json

from loguru import logger

from Cosmos.agents.planner_agent import PlannerAgent
from Cosmos.memory.memory_system import MemorySystem


class ProactiveState(Enum):
    IDLE = "idle"
    OBSERVING = "observing"  # Watching user actions to learn
    ANALYZING = "analyzing"  # Analyzing context to find opportunities
    SUGGESTING = "suggesting" # Preparing a suggestion
    ACTING = "acting"        # Autonomously executing a task


@dataclass
class Suggestion:
    """A proactive suggestion for the user."""
    id: str
    title: str
    description: str
    confidence: float
    reasoning: str
    
    # Actionable payload
    action_type: str  # "run_task", "create_file", "search", "none"
    action_payload: dict = field(default_factory=dict)
    
    timestamp: datetime = field(default_factory=datetime.now)
    is_dismissed: bool = False
    is_accepted: bool = False


@dataclass
class ScheduledTask:
    """A task scheduled to run periodically or at a specific time."""
    id: str
    description: str
    schedule_type: str  # "interval" or "cron" or "one_off"
    schedule_value: Any # seconds for interval, cron string, or datetime
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    enabled: bool = True

class ProactiveAgent:
    """
    Agent responsible for background monitoring and proactive assistance.
    """

    def __init__(
        self,
        memory_system: MemorySystem,
        planner_agent: PlannerAgent,
        llm_fn: Optional[Callable] = None,
        check_interval_seconds: float = 60.0,
    ):
        self.memory = memory_system
        self.planner = planner_agent
        self.llm_fn = llm_fn
        self.check_interval = check_interval_seconds
        
        self.state = ProactiveState.IDLE
        self.suggestions: list[Suggestion] = []
        self.scheduled_tasks: list[ScheduledTask] = []
        
        self._is_running = False
        
        self._is_running = False
        self._task: Optional[asyncio.Task] = None
        
        logger.info("Proactive Agent initialized")

    async def start(self):
        """Start the background monitoring loop."""
        if self._is_running:
            return
            
        self._is_running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("Proactive Agent started")

    async def stop(self):
        """Stop the background loop."""
        self._is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Proactive Agent stopped")

    async def _monitor_loop(self):
        """Main background loop."""
        while self._is_running:
            try:
                self.state = ProactiveState.OBSERVING
                
                # 1. Get current context
                context = await self._analyze_context()
                
                # 2. Check for patterns/needs
                if context["activity_level"] > 0:
                    self.state = ProactiveState.ANALYZING
                    opportunity = await self._detect_opportunity(context)
                    
                    if opportunity:
                        self.state = ProactiveState.SUGGESTING
                        await self._generate_suggestion(opportunity)
                
                # 3. Process Scheduled Tasks
                await self._check_scheduled_tasks()

                self.state = ProactiveState.IDLE
                
            except Exception as e:
                logger.error(f"Error in proactive loop: {e}")
                self.state = ProactiveState.IDLE
            
            await asyncio.sleep(self.check_interval)

    async def _check_scheduled_tasks(self):
        """Check and execute scheduled tasks."""
        now = datetime.now()
        for task in self.scheduled_tasks:
            if not task.enabled:
                continue

            should_run = False
            if task.schedule_type == "interval":
                if not task.last_run:
                    should_run = True
                elif (now - task.last_run).total_seconds() >= task.schedule_value:
                    should_run = True
            elif task.schedule_type == "one_off":
                if task.next_run and now >= task.next_run and not task.last_run:
                    should_run = True

            if should_run:
                logger.info(f"Running scheduled task: {task.description}")
                self.state = ProactiveState.ACTING
                try:
                    # Execute the task via planner
                    result = await self._execute_scheduled_task(task)
                    task.last_run = now

                    # Disable one-off tasks after execution
                    if task.schedule_type == "one_off":
                        task.enabled = False

                    logger.info(f"Scheduled task completed: {task.id} - Success: {result.get('success', False)}")

                except Exception as e:
                    logger.error(f"Failed to run scheduled task {task.id}: {e}")
                    # Still mark as run to prevent infinite retries
                    task.last_run = now

                self.state = ProactiveState.IDLE

    async def _execute_scheduled_task(self, task: ScheduledTask) -> dict:
        """
        Execute a scheduled task using the planner or LLM.

        Args:
            task: The scheduled task to execute

        Returns:
            Result dictionary with success status and output
        """
        try:
            # If we have an LLM function, use it to determine the best action
            if self.llm_fn:
                prompt = f"""Execute the following scheduled task and provide a result:

Task: {task.description}
Task ID: {task.id}
Schedule Type: {task.schedule_type}

Determine what action to take and execute it. Return a JSON response with:
- success: true/false
- action_taken: description of what was done
- output: any relevant output or result
"""
                if asyncio.iscoroutinefunction(self.llm_fn):
                    response = await self.llm_fn(prompt)
                else:
                    response = self.llm_fn(prompt)

                # Parse JSON response
                try:
                    start = response.find('{')
                    end = response.rfind('}') + 1
                    if start >= 0 and end > start:
                        result = json.loads(response[start:end])
                        return result
                except json.JSONDecodeError as e:
                    logger.debug(f"Task response not JSON-formatted, treating as plain text: {e}")

                return {"success": True, "action_taken": "processed", "output": response}

            # Fallback: Use planner if available
            if self.planner and hasattr(self.planner, 'create_plan'):
                plan_id = await self.planner.create_plan(
                    goal=task.description,
                    context={"scheduled_task_id": task.id}
                )
                return {"success": True, "action_taken": "plan_created", "plan_id": plan_id}

            # No execution method available
            logger.warning(f"No execution method available for task {task.id}")
            return {"success": False, "error": "No execution method available"}

        except Exception as e:
            logger.error(f"Task execution error: {e}")
            return {"success": False, "error": str(e)}

    async def _analyze_context(self) -> dict:
        """
        Analyze current user context and system state.
        
        Novelty: ContextVector
        We create a multi-dimensional representation of context:
        [time_of_day, user_mood, focus_level, project_phase]
        """
        # 1. Time context
        now = datetime.now()
        is_work_hours = 9 <= now.hour <= 17
        
        # 2. Memory context (recent conversations)
        recent_context = await self.memory.get_conversation_context(max_turns=5)
        
        # 3. Planner context (active tasks)
        active_plan = None
        if self.planner.active_plan_id:
            active_plan = self.planner.get_plan_status(self.planner.active_plan_id)
            
        # 4. Activity level heuristic
        activity_level = 0.1
        if active_plan and active_plan['status'] == 'in_progress':
            activity_level += 0.5
        if len(recent_context) > 50:
            activity_level += 0.4
            
        # 5. Novel Context Attributes - Mood and Focus Analysis
        mood, focus_score = await self._analyze_sentiment(recent_context)
            
        return {
            "timestamp": now,
            "is_work_hours": is_work_hours,
            "recent_context": recent_context,
            "active_plan": active_plan,
            "activity_level": min(1.0, activity_level),
            "context_vector": {
                "mood": mood,
                "focus": focus_score,
                "phase": "execution" if active_plan else "planning"
            }
        }

    async def _detect_opportunity(self, context: dict) -> Optional[dict]:
        """Detect a potential opportunity to help."""
        if not self.llm_fn or context["activity_level"] < 0.3:
            return None

        prompt = f"""Analyze the user's context and suggest ONE proactive action if relevant.
        
Current Time: {context['timestamp']} (Work Hours: {context['is_work_hours']})
Active Plan: {json.dumps(context.get('active_plan', {}), default=str)}
Recent Conversation:
{context['recent_context']}

Identify if there is a clear, helpful action you can propose (e.g., creating a file, searching for info, automating a next step).
If NO clear action is needed, return {{}}.

If yes, return a JSON object with:
- title: Short title of suggestion
- description: Why this is helpful
- action_type: "run_task" | "plan_task" | "search"
- confidence: 0.0 to 1.0 (threshold 0.7)
"""
        try:
            if asyncio.iscoroutinefunction(self.llm_fn):
                response = await self.llm_fn(prompt)
            else:
                response = self.llm_fn(prompt)
                
            # Basic JSON extraction
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                if data and data.get("confidence", 0) > 0.7:
                    return data
                    
        except Exception as e:
            logger.error(f"Opportunity detection failed: {e}")
            
        return None

    async def _generate_suggestion(self, opportunity: dict):
        """Create and store a suggestion."""
        suggestion = Suggestion(
            id=f"sugg_{len(self.suggestions) + 1}",
            title=opportunity.get("title", "Suggestion"),
            description=opportunity.get("description", ""),
            confidence=opportunity.get("confidence", 0.0),
            reasoning="Context analysis",
            action_type=opportunity.get("action_type", "none"),
            action_payload=opportunity,
        )
        
        self.suggestions.append(suggestion)
        logger.info(f"Generated proactive suggestion: {suggestion.title}")

    async def _analyze_sentiment(self, context: str) -> tuple[str, float]:
        """
        Analyze sentiment and focus level from context.

        Args:
            context: Recent conversation context

        Returns:
            Tuple of (mood, focus_score)
        """
        context_lower = context.lower()

        # Use LLM for sentiment analysis if available
        if self.llm_fn and len(context) > 50:
            try:
                prompt = f"""Analyze the sentiment and focus level of this conversation context.

Context:
{context[:1000]}

Return a JSON object with:
- mood: one of "frustrated", "focused", "exploratory", "confused", "satisfied", "neutral"
- focus_score: 0.0 to 1.0 (how focused/intense the user seems)
- reasoning: brief explanation

Return ONLY the JSON object."""

                if asyncio.iscoroutinefunction(self.llm_fn):
                    response = await self.llm_fn(prompt)
                else:
                    response = self.llm_fn(prompt)

                # Parse JSON response
                start = response.find('{')
                end = response.rfind('}') + 1
                if start >= 0 and end > start:
                    data = json.loads(response[start:end])
                    return data.get("mood", "neutral"), data.get("focus_score", 0.5)

            except Exception as e:
                logger.debug(f"LLM sentiment analysis failed, using heuristics: {e}")

        # Fallback to heuristic analysis
        focus_score = 0.5
        mood = "neutral"

        # Frustration indicators
        frustration_words = ["error", "fail", "broken", "bug", "issue", "problem", "wrong", "crash", "exception"]
        if any(word in context_lower for word in frustration_words):
            mood = "frustrated"
            focus_score = 0.9

        # Confusion indicators
        confusion_words = ["confused", "don't understand", "what does", "how do", "why is", "unclear"]
        if any(word in context_lower for word in confusion_words):
            mood = "confused"
            focus_score = 0.7

        # Satisfaction indicators
        satisfaction_words = ["thanks", "perfect", "great", "awesome", "works", "solved", "fixed"]
        if any(word in context_lower for word in satisfaction_words):
            mood = "satisfied"
            focus_score = 0.3

        # Exploration indicators (long queries, many questions)
        if context_lower.count("?") > 3 or len(context) > 500:
            mood = "exploratory"
            focus_score = 0.6

        return mood, focus_score

    def add_scheduled_task(
        self,
        description: str,
        schedule_type: str = "interval",
        schedule_value: Any = 3600,
        task_id: Optional[str] = None,
    ) -> ScheduledTask:
        """
        Add a new scheduled task.

        Args:
            description: What the task should do
            schedule_type: "interval", "cron", or "one_off"
            schedule_value: Seconds for interval, cron string, or datetime for one_off
            task_id: Optional custom ID

        Returns:
            The created ScheduledTask
        """
        task = ScheduledTask(
            id=task_id or f"task_{len(self.scheduled_tasks) + 1}_{int(datetime.now().timestamp())}",
            description=description,
            schedule_type=schedule_type,
            schedule_value=schedule_value,
            next_run=schedule_value if schedule_type == "one_off" and isinstance(schedule_value, datetime) else None,
        )
        self.scheduled_tasks.append(task)
        logger.info(f"Added scheduled task: {task.id} - {description}")
        return task

    def remove_scheduled_task(self, task_id: str) -> bool:
        """Remove a scheduled task by ID."""
        for i, task in enumerate(self.scheduled_tasks):
            if task.id == task_id:
                self.scheduled_tasks.pop(i)
                logger.info(f"Removed scheduled task: {task_id}")
                return True
        return False

    def get_pending_suggestions(self) -> list[Suggestion]:
        """Get all non-dismissed suggestions."""
        return [s for s in self.suggestions if not s.is_dismissed]

    def dismiss_suggestion(self, suggestion_id: str) -> bool:
        """Dismiss a suggestion."""
        for s in self.suggestions:
            if s.id == suggestion_id:
                s.is_dismissed = True
                return True
        return False

    def accept_suggestion(self, suggestion_id: str) -> Optional[Suggestion]:
        """Accept a suggestion and return it for execution."""
        for s in self.suggestions:
            if s.id == suggestion_id:
                s.is_accepted = True
                return s
        return None

    def get_status(self) -> dict:
        """Get proactive agent status."""
        return {
            "state": self.state.value,
            "is_running": self._is_running,
            "suggestion_count": len(self.suggestions),
            "pending_suggestions": len(self.get_pending_suggestions()),
            "scheduled_tasks": len(self.scheduled_tasks),
            "active_scheduled_tasks": len([t for t in self.scheduled_tasks if t.enabled]),
        }
