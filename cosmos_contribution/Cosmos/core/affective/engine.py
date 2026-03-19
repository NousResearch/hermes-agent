from typing import Any, Dict, List, Optional
from datetime import datetime
from .models import AffectiveState, EmotionCategory, SystemAction, SystemPriority

logger = logging.getLogger(__name__)

class AffectiveEngine:
    """
    Emotion-to-Action Engine.
    Maps affective states to system priorities and actions.
    """
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.current_state = AffectiveState()
        self.action_history: list[SystemAction] = []
        
        # Default policy mapping
        # This could be loaded from a JSON/YAML file in the future
        self.policy_map = {
            EmotionCategory.FRUSTRATION: self._handle_frustration,
            EmotionCategory.EXHAUSTION: self._handle_exhaustion,
            EmotionCategory.FLOW: self._handle_flow,
            EmotionCategory.FOCUS: self._handle_focus,
            EmotionCategory.ANXIETY: self._handle_anxiety
        }

    def update_state(self, new_state: AffectiveState) -> list[SystemAction]:
        """
        Ingests a new affective state and returns a list of recommended system actions.
        """
        self.current_state = new_state
        logger.info(f"Affective State Updated: {new_state.primary_emotion.name} (V:{new_state.valence:.2f}, A:{new_state.arousal:.2f})")
        
        actions = self._evaluate_policy(new_state)
        self.action_history.extend(actions)
        return actions

    def _evaluate_policy(self, state: AffectiveState) -> list[SystemAction]:
        handler = self.policy_map.get(state.primary_emotion)
        if handler:
            return handler(state)
        return []

    def _handle_frustration(self, state: AffectiveState) -> list[SystemAction]:
        """
        User is frustrated. System should simplify, slow down, or ask for clarification.
        """
        actions = []
        if state.arousal > 0.7:
             actions.append(SystemAction(
                action_id="activate_emergency_brake",
                priority_delta=2.0,
                description="High frustration detected. Pausing non-essential background tasks.",
                parameters={"scope": "all_background"}
            ))
        
        actions.append(SystemAction(
            action_id="simplify_mode",
            priority_delta=1.0,
            description="Switching UI/Responses to concise mode.",
            parameters={"verbosity": "low"}
        ))
        return actions

    def _handle_exhaustion(self, state: AffectiveState) -> list[SystemAction]:
        """
        User is exhausted. System should summarize and defer non-critical items.
        """
        return [
            SystemAction(
                action_id="defer_low_priority",
                priority_delta=-1.0,
                description="Deferring notifications and low-priority tasks.",
                parameters={"threshold": "NORMAL"}
            ),
            SystemAction(
                action_id="summarization_boost",
                priority_delta=1.0,
                description="Auto-summarizing incoming long-form content.",
                parameters={"ratio": "0.2"}
            )
        ]

    def _handle_flow(self, state: AffectiveState) -> list[SystemAction]:
        """
        User is in flow state. DO NOT DISTURB.
        """
        return [
            SystemAction(
                action_id="enable_dnd",
                priority_delta=5.0,
                description="Enabling Do Not Disturb. Blocking all non-critical interruptions.",
                parameters={"mode": "strict"}
            ),
            SystemAction(
                action_id="suppress_suggestions",
                priority_delta=1.0,
                description="Suppressing proactive suggestions to maintain flow.",
                parameters={}
            )
        ]

    def _handle_focus(self, state: AffectiveState) -> list[SystemAction]:
         return [
            SystemAction(
                action_id="mute_notifications",
                priority_delta=2.0,
                description="Muting audible notifications.",
                parameters={"visual_only": "true"}
            )
        ]

    def _handle_anxiety(self, state: AffectiveState) -> list[SystemAction]:
        """
        User is anxious. Provide reassurance and break down tasks.
        """
        return [
            SystemAction(
                action_id="decomposition_assistant",
                priority_delta=1.5,
                description="Offering to break down current complex task into micro-steps.",
                parameters={"auto_decompose": "true"}
            ),
            SystemAction(
                action_id="soothing_ui",
                priority_delta=0.5,
                description="Adjusting UI colors to calmer palette if supported.",
                parameters={"theme": "calm"}
            )
        ]
