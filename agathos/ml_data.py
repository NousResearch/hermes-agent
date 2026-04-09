"""AGATHOS ML data pipeline — convert entropy detections to training data.

Optional feature. When enabled, Agathos exports trajectories and facts for model
training. Converts failure modes (stuck loops, error cascades) into training
examples showing correct recovery patterns.

Integration targets:
- holographic_memory.db (if hermes-state available)
- ShareGPT format trajectories for fine-tuning
- Native hermes-agent ML pipeline (if available)
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger("agathos.ml_data")


class MLDataExporter:
    """Export Agathos detections as ML training data.
    
    Generates trajectories showing failure → detection → recovery flows.
    Each trajectory teaches the model to recognize and recover from entropy.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path.home() / ".hermes" / "agathos" / "ml_data"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def export_detection_trajectory(
        self,
        entropy_type: str,
        severity: str,
        session_context: Dict[str, Any],
        detection_details: Dict[str, Any],
        recovery_action: str,
        outcome: str,
    ) -> Optional[Path]:
        """Generate a trajectory from a detection event.
        
        Creates ShareGPT format showing:
        1. System: You are an agent monitoring task
        2. Human: Task description
        3. GPT: Starts working (shows entropy pattern)
        4. System/Tool: Agathos detection alert
        5. GPT: Corrects based on prompt
        
        Returns path to saved trajectory file or None.
        """
        trajectory = self._build_trajectory(
            entropy_type=entropy_type,
            severity=severity,
            session_context=session_context,
            detection_details=detection_details,
            recovery_action=recovery_action,
            outcome=outcome,
        )
        
        if not trajectory:
            return None
            
        # Generate filename
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_type = entropy_type.replace(" ", "_")
        filename = f"agathos_{safe_type}_{ts}.json"
        filepath = self.output_dir / filename
        
        # Write ShareGPT format
        record = {
            "conversations": trajectory,
            "metadata": {
                "source": "argus",
                "entropy_type": entropy_type,
                "severity": severity,
                "timestamp": datetime.now().isoformat(),
                "outcome": outcome,
            },
        }
        
        try:
            filepath.write_text(json.dumps(record, indent=2))
            logger.info("Exported trajectory to %s", filepath)
            return filepath
        except Exception as e:
            logger.error("Failed to write trajectory: %s", e)
            return None
    
    def _build_trajectory(
        self,
        entropy_type: str,
        severity: str,
        session_context: Dict[str, Any],
        detection_details: Dict[str, Any],
        recovery_action: str,
        outcome: str,
    ) -> List[Dict[str, str]]:
        """Build ShareGPT conversation from detection event."""
        task = session_context.get("task_description", "Unknown task")
        session_type = session_context.get("session_type", "unknown")
        
        # System prompt sets context
        system_msg = (
            "You are an AI assistant monitoring agent sessions. "
            "Detect entropy patterns and apply corrective actions when needed."
        )
        
        # Human presents task
        human_task = f"Task: {task}\nSession type: {session_type}"
        
        # GPT starts working (shows the problematic pattern based on entropy type)
        gpt_error = self._simulate_error_pattern(entropy_type, detection_details)
        
        # Agathos detection alert (system message)
        agathos_alert = self._format_alert(entropy_type, severity, detection_details)
        
        # Corrective instruction
        corrective = self._get_corrective_instruction(entropy_type, recovery_action)
        
        # GPT recovers
        gpt_recovery = self._simulate_recovery(entropy_type, outcome)
        
        trajectory = [
            {"from": "system", "value": system_msg},
            {"from": "human", "value": human_task},
            {"from": "gpt", "value": gpt_error},
            {"from": "system", "value": agathos_alert},
            {"from": "human", "value": corrective},
            {"from": "gpt", "value": gpt_recovery},
        ]
        
        return trajectory
    
    def _simulate_error_pattern(self, entropy_type: str, details: Dict) -> str:
        """Generate text showing the problematic pattern."""
        patterns = {
            "repeat_tool_calls": (
                "I'll use the terminal tool to check this.\n"
                "<tool>terminal</tool>\n"
                "Let me try terminal again to verify.\n"
                "<tool>terminal</tool>\n"
                "I'll run terminal once more to be sure."
            ),
            "stuck_loop": (
                "Let me analyze this file...\n"
                "Analyzing the content...\n" 
                "Still analyzing...\n"
                "Continuing analysis..."
            ),
            "error_cascade": (
                "Running command... Error: file not found\n"
                "Trying alternative... Error: permission denied\n"
                "Attempting workaround... Error: command failed"
            ),
            "budget_pressure": (
                "Let me search for all occurrences...\n"
                "Now I'll check every file in the project...\n"
                "I'll also examine the dependencies..."
            ),
        }
        return patterns.get(entropy_type, "Working on task... [prolonged activity]")
    
    def _format_alert(self, entropy_type: str, severity: str, details: Dict) -> str:
        """Format Agathos detection as system alert."""
        return (
            f"[ARGUS ALERT] Entropy detected: {entropy_type}\n"
            f"Severity: {severity}\n"
            f"Details: {json.dumps(details, indent=2)}"
        )
    
    def _get_corrective_instruction(self, entropy_type: str, action: str) -> str:
        """Generate corrective instruction based on entropy type."""
        instructions = {
            "repeat_tool_calls": (
                f"{action} - You are repeating the same tool calls. "
                "Consolidate into single comprehensive operation. "
                "Process all items in one pass, avoid redundant calls."
            ),
            "stuck_loop": (
                f"{action} - You appear to be stuck in analysis without progress. "
                "Make a decision and take action. Report current status if blocked."
            ),
            "error_cascade": (
                f"{action} - Multiple consecutive errors. Stop and reassess approach. "
                "Verify prerequisites, check environment, use different strategy."
            ),
            "budget_pressure": (
                f"{action} - High token usage detected. Use targeted searches, "
                "read specific files, avoid broad operations. Prioritize efficiency."
            ),
        }
        return instructions.get(
            entropy_type, 
            f"{action} - Apply corrective action for {entropy_type}"
        )
    
    def _simulate_recovery(self, entropy_type: str, outcome: str) -> str:
        """Generate text showing successful recovery."""
        recoveries = {
            "repeat_tool_calls": (
                "Understood. I'll consolidate operations.\n"
                "<tool>terminal</tool>\n"
                "[Single comprehensive command executed successfully]\n"
                "Task completed efficiently."
            ),
            "stuck_loop": (
                "You're right, I was circling. Taking action now:\n"
                "<tool>write_file</tool>\n"
                "[File written with solution]\n"
                "Progress made. Continuing to completion."
            ),
            "error_cascade": (
                "Stopping to reassess. Checking environment first:\n"
                "<tool>terminal</tool>\n"
                "[Verified prerequisites]\n"
                "Now trying corrected approach... Success!"
            ),
        }
        return recoveries.get(entropy_type, f"Recovery executed. Outcome: {outcome}")


class HolographicMemoryBridge:
    """Bridge Agathos detections to holographic memory system.
    
    Records facts about entropy patterns for retrieval during similar tasks.
    Enables 'pattern memory' - agent recalls previous failure modes.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path
        self._available: Optional[bool] = None
        
    def is_available(self) -> bool:
        """Check if holographic memory is accessible."""
        if self._available is not None:
            return self._available
            
        try:
            # Try to import hermes_state
            from hermes_state import SessionDB
            self._available = True
            return True
        except ImportError:
            self._available = False
            return False
    
    def record_entropy_fact(
        self,
        entropy_type: str,
        severity: str,
        task_pattern: str,
        resolution: str,
        quality_score: float = 0.92,
    ) -> bool:
        """Record entropy detection as retrievable fact.
        
        Stores in holographic_memory.db for later retrieval
        when similar task patterns are detected.
        """
        if not self.is_available():
            logger.debug("Holographic memory unavailable — skipping fact record")
            return False
            
        try:
            # Import within method to avoid dependency issues
            from hermes_state import holographic_memory
            
            # Build fact content
            fact_content = (
                f"Entropy pattern '{entropy_type}' (severity: {severity}) "
                f"observed in task matching pattern '{task_pattern}'. "
                f"Resolution: {resolution}"
            )
            
            # Categories for retrieval
            categories = f"entropy-detection,{entropy_type},argus"
            
            # Insert into holographic memory
            holographic_memory.insert_fact(
                source="argus",
                content=fact_content,
                quality_score=quality_score,
                categories=categories,
                metadata={
                    "entropy_type": entropy_type,
                    "severity": severity,
                    "task_pattern": task_pattern,
                    "resolution": resolution,
                },
            )
            
            logger.info(
                "Recorded entropy fact: %s (%s)", entropy_type, severity
            )
            return True
            
        except Exception as e:
            logger.error("Failed to record fact: %s", e)
            return False
    
    def recall_similar_patterns(self, task_description: str, limit: int = 3) -> List[Dict]:
        """Recall previously observed entropy patterns for similar tasks.
        
        Query holographic memory for entropy facts matching task pattern.
        Returns list of previous detections to help avoid repeated failures.
        """
        if not self.is_available():
            return []
            
        try:
            from hermes_state import holographic_memory
            
            # Search for entropy-related facts
            results = holographic_memory.search_facts(
                query=f"entropy {task_description}",
                categories="entropy-detection",
                limit=limit,
                min_quality=0.90,
            )
            
            return results
            
        except Exception as e:
            logger.error("Failed to recall patterns: %s", e)
            return []


def export_entropy_event(
    entropy_type: str,
    severity: str,
    session_context: Dict[str, Any],
    detection_details: Dict[str, Any],
    recovery_action: str,
    outcome: str,
    enable_trajectory: bool = True,
    enable_memory: bool = True,
) -> Dict[str, Any]:
    """Unified export for an entropy detection event.
    
    Convenience function that exports to all enabled ML targets.
    
    Args:
        entropy_type: Type of entropy detected
        severity: warning/critical/fatal
        session_context: Task description, session type, etc.
        detection_details: Tool call counts, error rates, etc.
        recovery_action: What action was taken (restart/kill/inject)
        outcome: Success/failure of recovery
        enable_trajectory: Export ShareGPT trajectory
        enable_memory: Record to holographic memory
        
    Returns:
        Dict with export results {trajectory_path, memory_recorded}
    """
    results: Dict[str, Any] = {
        "trajectory_path": None,
        "memory_recorded": False,
    }
    
    # Export trajectory
    if enable_trajectory:
        exporter = MLDataExporter()
        path = exporter.export_detection_trajectory(
            entropy_type=entropy_type,
            severity=severity,
            session_context=session_context,
            detection_details=detection_details,
            recovery_action=recovery_action,
            outcome=outcome,
        )
        results["trajectory_path"] = str(path) if path else None
    
    # Record to holographic memory
    if enable_memory:
        bridge = HolographicMemoryBridge()
        recorded = bridge.record_entropy_fact(
            entropy_type=entropy_type,
            severity=severity,
            task_pattern=session_context.get("task_description", "unknown"),
            resolution=f"{recovery_action} -> {outcome}",
        )
        results["memory_recorded"] = recorded
    
    return results
