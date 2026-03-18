"""
Task Classification System for Parallel Task Execution

This module provides intelligent classification of user tasks to determine
whether they can be executed in parallel with other running tasks.
"""

import enum
import re
from typing import Dict, List, Optional, Set
from dataclasses import dataclass


class TaskType(enum.Enum):
    """Classification of task independence level."""
    INDEPENDENT = "independent"  # Can run in parallel (image gen, search, calculation)
    DEPENDENT = "dependent"      # Depends on previous task results
    SEQUENTIAL = "sequential"    # Must run sequentially (code patches, file edits)
    BLOCKING = "blocking"        # Blocks everything (requires user confirmation)


@dataclass
class TaskClassification:
    """Result of task classification."""
    task_type: TaskType
    confidence: float  # 0.0 to 1.0
    reasoning: str
    suggested_toolsets: Set[str]
    estimated_duration: str  # "short", "medium", "long"


class TaskClassifier:
    """
    Classifies user messages to determine task type for parallel execution.
    
    Uses keyword patterns, intent analysis, and historical context to classify
    whether a task can run in parallel with others.
    """
    
    # Keywords that suggest INDEPENDENT tasks (can run in parallel)
    INDEPENDENT_KEYWORDS = {
        "search", "find", "look up", "google", "research",
        "generate", "create image", "draw", "make picture",
        "calculate", "compute", "math", "sum", "average",
        "analyze", "check", "verify", "test",
        "weather", "time", "date", "convert",
        "summarize", "translate", "explain",
    }
    
    # Keywords that suggest SEQUENTIAL tasks (file operations, code changes)
    SEQUENTIAL_KEYWORDS = {
        "edit", "modify", "change", "update", "patch", "fix",
        "create file", "write file", "delete", "remove", "move",
        "refactor", "restructure", "rename",
        "apply", "implement", "deploy",
        "install", "setup", "configure",
    }
    
    # Keywords that suggest DEPENDENT tasks (need previous context)
    DEPENDENT_KEYWORDS = {
        "based on", "using the", "from the previous",
        "now", "next", "then", "after that",
        "with those results", "from that",
        "update the", "modify the", "change the",
    }
    
    # Keywords that suggest BLOCKING tasks (require user input)
    BLOCKING_KEYWORDS = {
        "confirm", "approve", "verify with me",
        "should i", "what do you think",
        "is this okay", "does this look right",
    }
    
    # Tool patterns for quick classification
    TOOL_PATTERNS = {
        "image": TaskType.INDEPENDENT,  # Image generation
        "web_search": TaskType.INDEPENDENT,
        "browser": TaskType.INDEPENDENT,
        "terminal": TaskType.SEQUENTIAL,  # File operations
        "file": TaskType.SEQUENTIAL,
        "patch": TaskType.SEQUENTIAL,
        "execute_code": TaskType.INDEPENDENT,  # Code execution can be parallel
        "delegate_task": TaskType.INDEPENDENT,  # Subagents can run in parallel
    }
    
    def __init__(self):
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        self._independent_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(kw) for kw in self.INDEPENDENT_KEYWORDS) + r')\b',
            re.IGNORECASE
        )
        self._sequential_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(kw) for kw in self.SEQUENTIAL_KEYWORDS) + r')\b',
            re.IGNORECASE
        )
        self._dependent_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(kw) for kw in self.DEPENDENT_KEYWORDS) + r')\b',
            re.IGNORECASE
        )
        self._blocking_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(kw) for kw in self.BLOCKING_KEYWORDS) + r')\b',
            re.IGNORECASE
        )
    
    def classify(
        self,
        message: str,
        conversation_history: Optional[List[Dict]] = None,
        running_tasks: Optional[List[str]] = None
    ) -> TaskClassification:
        """
        Classify a user message to determine task type.
        
        Args:
            message: The user's message
            conversation_history: Previous conversation messages (optional)
            running_tasks: List of currently running task descriptions (optional)
            
        Returns:
            TaskClassification with type, confidence, and metadata
        """
        message_lower = message.lower()
        
        # Count keyword matches
        independent_matches = len(self._independent_pattern.findall(message_lower))
        sequential_matches = len(self._sequential_pattern.findall(message_lower))
        dependent_matches = len(self._dependent_pattern.findall(message_lower))
        blocking_matches = len(self._blocking_pattern.findall(message_lower))
        
        # Check for explicit parallel/background indicators
        parallel_intent = any(
            phrase in message_lower
            for phrase in ["in parallel", "at the same time", "while you", "background"]
        )
        
        # Determine task type
        if blocking_matches > 0:
            task_type = TaskType.BLOCKING
            confidence = min(0.5 + (blocking_matches * 0.15), 0.95)
            reasoning = f"Detected {blocking_matches} blocking keyword(s) indicating user confirmation needed"
            
        elif dependent_matches > 0 and running_tasks:
            task_type = TaskType.DEPENDENT
            confidence = min(0.5 + (dependent_matches * 0.15), 0.9)
            reasoning = f"Detected {dependent_matches} dependent keyword(s) suggesting reliance on previous results"
            
        elif sequential_matches > independent_matches:
            task_type = TaskType.SEQUENTIAL
            confidence = min(0.5 + (sequential_matches * 0.12), 0.9)
            reasoning = f"Detected {sequential_matches} sequential keyword(s) vs {independent_matches} independent"
            
        elif independent_matches > 0 or parallel_intent:
            task_type = TaskType.INDEPENDENT
            confidence = min(0.6 + (independent_matches * 0.1), 0.95)
            if parallel_intent:
                confidence = min(confidence + 0.1, 0.95)
                reasoning = f"Explicit parallel intent detected with {independent_matches} independent keyword(s)"
            else:
                reasoning = f"Detected {independent_matches} independent keyword(s), task appears parallelizable"
                
        else:
            # Default to sequential for unknown tasks (safer)
            task_type = TaskType.SEQUENTIAL
            confidence = 0.5
            reasoning = "No clear classification keywords found, defaulting to sequential for safety"
        
        # Determine suggested toolsets
        suggested_toolsets = self._suggest_toolsets(message_lower, task_type)
        
        # Estimate duration
        estimated_duration = self._estimate_duration(message_lower, task_type)
        
        return TaskClassification(
            task_type=task_type,
            confidence=confidence,
            reasoning=reasoning,
            suggested_toolsets=suggested_toolsets,
            estimated_duration=estimated_duration
        )
    
    def _suggest_toolsets(self, message_lower: str, task_type: TaskType) -> Set[str]:
        """Suggest appropriate toolsets based on message content."""
        toolsets = set()
        
        if any(kw in message_lower for kw in ["search", "google", "find", "look up"]):
            toolsets.add("web")
        
        if any(kw in message_lower for kw in ["image", "generate", "draw", "create picture"]):
            toolsets.add("image")
        
        if any(kw in message_lower for kw in ["code", "program", "script", "function"]):
            toolsets.add("terminal")
            toolsets.add("file")
        
        if any(kw in message_lower for kw in ["file", "edit", "modify", "patch"]):
            toolsets.add("file")
        
        if not toolsets:
            # Default toolsets based on task type
            if task_type == TaskType.INDEPENDENT:
                toolsets.add("web")
            else:
                toolsets.add("terminal")
                toolsets.add("file")
        
        return toolsets
    
    def _estimate_duration(self, message_lower: str, task_type: TaskType) -> str:
        """Estimate task duration based on complexity indicators."""
        # Long-running indicators
        long_indicators = [
            "large", "many", "multiple", "batch", "process all",
            "train", "fine-tune", "build", "compile",
            "download", "sync", "backup"
        ]
        
        # Short indicators
        short_indicators = [
            "quick", "simple", "check", "what is", "how to",
            "small", "one", "single", "brief"
        ]
        
        if any(ind in message_lower for ind in long_indicators):
            return "long"
        elif any(ind in message_lower for ind in short_indicators):
            return "short"
        elif task_type == TaskType.INDEPENDENT:
            return "short"  # Independent tasks are usually quick
        else:
            return "medium"
    
    def can_run_in_parallel(
        self,
        new_task: TaskClassification,
        running_tasks: List[TaskClassification]
    ) -> bool:
        """
        Determine if a new task can run in parallel with existing tasks.
        
        Args:
            new_task: Classification of the new task
            running_tasks: Classifications of currently running tasks
            
        Returns:
            True if the task can be executed in parallel
        """
        # Blocking tasks never run in parallel
        if new_task.task_type == TaskType.BLOCKING:
            return False
        
        # Sequential tasks don't run in parallel with other sequential tasks
        if new_task.task_type == TaskType.SEQUENTIAL:
            for task in running_tasks:
                if task.task_type == TaskType.SEQUENTIAL:
                    return False
        
        # Dependent tasks check if they depend on running tasks
        if new_task.task_type == TaskType.DEPENDENT:
            # For now, conservative: dependent tasks wait for all
            # In a more sophisticated version, we'd track task dependencies
            return len(running_tasks) == 0
        
        # Independent tasks can run in parallel
        if new_task.task_type == TaskType.INDEPENDENT:
            # Check for resource conflicts
            for task in running_tasks:
                # Two file operations might conflict
                if ("file" in new_task.suggested_toolsets and 
                    "file" in task.suggested_toolsets):
                    return False
            return True
        
        return False


# Global classifier instance
_classifier = None


def get_classifier() -> TaskClassifier:
    """Get the global task classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = TaskClassifier()
    return _classifier


def classify_task(
    message: str,
    conversation_history: Optional[List[Dict]] = None,
    running_tasks: Optional[List[str]] = None
) -> TaskClassification:
    """
    Convenience function to classify a task.
    
    Args:
        message: The user's message
        conversation_history: Previous conversation messages
        running_tasks: Currently running task descriptions
        
    Returns:
        TaskClassification result
    """
    return get_classifier().classify(message, conversation_history, running_tasks)
