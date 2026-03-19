"""
cosmos User Avatar - User Preference Modeling

Novel Approaches:
1. Preference Learning - Learn user's style and preferences
2. Proactive Assistance - Anticipate user needs
3. Communication Adaptation - Match user's communication style
4. Implicit Feedback - Learn from interactions
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from collections import defaultdict

from loguru import logger

from Cosmos.agents.base_agent import BaseAgent, AgentCapability, TaskResult


@dataclass
class UserPreference:
    """A learned user preference."""
    category: str  # communication, code, topics, etc.
    key: str
    value: Any
    confidence: float = 0.5
    observations: int = 1
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class UserProfile:
    """Complete user profile."""
    user_id: str = "default"
    preferences: dict[str, UserPreference] = field(default_factory=dict)
    interaction_count: int = 0
    topics_discussed: dict[str, int] = field(default_factory=dict)
    feedback_history: list[dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_interaction: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "preferences": {k: {
                "category": v.category,
                "key": v.key,
                "value": v.value,
                "confidence": v.confidence,
            } for k, v in self.preferences.items()},
            "interaction_count": self.interaction_count,
            "topics_discussed": dict(self.topics_discussed),
            "created_at": self.created_at.isoformat(),
            "last_interaction": self.last_interaction.isoformat(),
        }


class UserAvatar(BaseAgent):
    """
    Agent that models and represents user preferences.

    Features:
    - Learns user communication style
    - Tracks topic preferences
    - Adapts responses to user style
    - Participates in reasoning about user needs
    """

    def __init__(
        self,
        data_dir: str = "./data/user",
        learning_rate: float = 0.1,
    ):
        super().__init__(
            name="UserAvatar",
            capabilities=[
                AgentCapability.USER_MODELING,
                AgentCapability.QUESTION_ANSWERING,
            ],
            confidence_threshold=0.5,
        )

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.learning_rate = learning_rate
        self.profile = UserProfile()

        # Default preferences
        self._initialize_default_preferences()

    def _initialize_default_preferences(self):
        """Initialize default user preferences."""
        defaults = [
            ("communication", "verbosity", "moderate"),
            ("communication", "formality", "moderate"),
            ("communication", "technical_level", "moderate"),
            ("code", "language", "python"),
            ("code", "style", "clean"),
            ("feedback", "prefers_explanations", True),
        ]

        for category, key, value in defaults:
            pref_id = f"{category}:{key}"
            self.profile.preferences[pref_id] = UserPreference(
                category=category,
                key=key,
                value=value,
                confidence=0.3,  # Low confidence for defaults
            )

    @property
    def system_prompt(self) -> str:
        # Dynamic system prompt based on learned preferences
        prefs = self._get_preference_summary()
        return f"""You represent the user's preferences and style in the AI system.

Current user profile:
{prefs}

Your role:
1. Help other agents understand user preferences
2. Suggest response adaptations based on user style
3. Flag when responses might not match user expectations
4. Track and learn from user feedback

When asked about the user:
- Share relevant preferences
- Note confidence levels
- Suggest when to ask for clarification"""

    def _get_preference_summary(self) -> str:
        """Get summary of user preferences for prompts."""
        lines = []
        for pref in self.profile.preferences.values():
            if pref.confidence > 0.4:
                lines.append(f"- {pref.category}/{pref.key}: {pref.value} (confidence: {pref.confidence:.0%})")
        return "\n".join(lines) if lines else "No strong preferences learned yet."

    async def process(self, task: str, context: Optional[dict] = None) -> TaskResult:
        """Process a user-modeling task."""
        task_type = self._classify_task(task)

        if task_type == "get_preference":
            result = self._handle_get_preference(task)
        elif task_type == "suggest_adaptation":
            result = await self._handle_suggest_adaptation(task, context)
        elif task_type == "learn":
            result = self._handle_learn(task, context)
        else:
            result = await self._handle_general(task, context)

        return result

    def _classify_task(self, task: str) -> str:
        """Classify the type of user-modeling task."""
        task_lower = task.lower()

        if "preference" in task_lower or "like" in task_lower:
            return "get_preference"
        elif "adapt" in task_lower or "style" in task_lower:
            return "suggest_adaptation"
        elif "learn" in task_lower or "feedback" in task_lower:
            return "learn"

        return "general"

    def _handle_get_preference(self, task: str) -> TaskResult:
        """Handle preference query."""
        # Extract what preference is being asked about
        relevant_prefs = []

        for pref_id, pref in self.profile.preferences.items():
            if pref.key.lower() in task.lower() or pref.category.lower() in task.lower():
                relevant_prefs.append(pref)

        if relevant_prefs:
            output = "User preferences:\n"
            for pref in relevant_prefs:
                output += f"- {pref.key}: {pref.value} (confidence: {pref.confidence:.0%})\n"
        else:
            output = self._get_preference_summary()

        return TaskResult(
            success=True,
            output=output,
            confidence=0.8,
        )

    async def _handle_suggest_adaptation(
        self,
        task: str,
        context: Optional[dict],
    ) -> TaskResult:
        """Suggest how to adapt response for user."""
        # Gather relevant preferences
        verbosity = self.get_preference("communication", "verbosity")
        formality = self.get_preference("communication", "formality")
        tech_level = self.get_preference("communication", "technical_level")

        suggestions = []

        if verbosity == "brief":
            suggestions.append("Keep response concise and to the point")
        elif verbosity == "detailed":
            suggestions.append("Provide detailed explanations")

        if formality == "formal":
            suggestions.append("Use formal, professional language")
        elif formality == "casual":
            suggestions.append("Use casual, friendly tone")

        if tech_level == "high":
            suggestions.append("Can use technical jargon")
        elif tech_level == "low":
            suggestions.append("Explain technical concepts simply")

        output = "Adaptation suggestions:\n" + "\n".join(f"- {s}" for s in suggestions)

        return TaskResult(
            success=True,
            output=output,
            confidence=0.7,
            metadata={"suggestions": suggestions},
        )

    def _handle_learn(self, task: str, context: Optional[dict]) -> TaskResult:
        """Handle learning from feedback."""
        if context and "feedback" in context:
            self.learn_from_feedback(context["feedback"])
            return TaskResult(
                success=True,
                output="Learned from feedback",
                confidence=0.9,
            )

        return TaskResult(
            success=False,
            output="No feedback provided to learn from",
            confidence=0.0,
        )

    async def _handle_general(self, task: str, context: Optional[dict]) -> TaskResult:
        """Handle general user-related queries."""
        prompt = f"Based on user profile, answer: {task}"
        response, confidence = await self.generate_response(prompt, context)

        return TaskResult(
            success=True,
            output=response,
            confidence=confidence,
        )

    def get_preference(self, category: str, key: str, default: Any = None) -> Any:
        """Get a specific preference value."""
        pref_id = f"{category}:{key}"
        pref = self.profile.preferences.get(pref_id)

        if pref and pref.confidence > 0.3:
            return pref.value

        return default

    def set_preference(
        self,
        category: str,
        key: str,
        value: Any,
        confidence: float = 0.8,
    ):
        """Explicitly set a preference."""
        pref_id = f"{category}:{key}"

        if pref_id in self.profile.preferences:
            pref = self.profile.preferences[pref_id]
            pref.value = value
            pref.confidence = confidence
            pref.observations += 1
            pref.last_updated = datetime.now()
        else:
            self.profile.preferences[pref_id] = UserPreference(
                category=category,
                key=key,
                value=value,
                confidence=confidence,
            )

    def learn_from_interaction(self, interaction: dict):
        """
        Learn from a user interaction.

        Interaction should contain:
        - user_message: What the user said
        - assistant_response: How we responded
        - user_reaction: Any follow-up (optional)
        """
        self.profile.interaction_count += 1
        self.profile.last_interaction = datetime.now()

        user_message = interaction.get("user_message", "")
        user_reaction = interaction.get("user_reaction", "")

        # Analyze message for preferences
        self._analyze_communication_style(user_message)

        # Extract topics
        topics = self._extract_topics(user_message)
        for topic in topics:
            self.profile.topics_discussed[topic] = (
                self.profile.topics_discussed.get(topic, 0) + 1
            )

        # If there's a reaction, learn from it
        if user_reaction:
            self._learn_from_reaction(user_reaction)

    def _analyze_communication_style(self, message: str):
        """Analyze communication style from message."""
        # Verbosity analysis
        words = len(message.split())
        if words < 10:
            self._update_preference("communication", "verbosity", "brief")
        elif words > 50:
            self._update_preference("communication", "verbosity", "detailed")

        # Formality analysis
        formal_indicators = ["please", "kindly", "would you", "could you"]
        casual_indicators = ["hey", "yo", "gonna", "wanna"]

        message_lower = message.lower()
        if any(ind in message_lower for ind in formal_indicators):
            self._update_preference("communication", "formality", "formal")
        elif any(ind in message_lower for ind in casual_indicators):
            self._update_preference("communication", "formality", "casual")

        # Technical level
        tech_terms = ["api", "function", "algorithm", "database", "protocol"]
        if sum(1 for t in tech_terms if t in message_lower) >= 2:
            self._update_preference("communication", "technical_level", "high")

    def _update_preference(self, category: str, key: str, observed_value: Any):
        """Update preference with new observation."""
        pref_id = f"{category}:{key}"

        if pref_id in self.profile.preferences:
            pref = self.profile.preferences[pref_id]

            # If same value, increase confidence
            if pref.value == observed_value:
                pref.confidence = min(1.0, pref.confidence + self.learning_rate)
            else:
                # Different value, decrease confidence and possibly switch
                pref.confidence -= self.learning_rate
                if pref.confidence < 0.3:
                    pref.value = observed_value
                    pref.confidence = 0.4

            pref.observations += 1
            pref.last_updated = datetime.now()
        else:
            self.profile.preferences[pref_id] = UserPreference(
                category=category,
                key=key,
                value=observed_value,
                confidence=0.4,
            )

    def _extract_topics(self, text: str) -> list[str]:
        """Extract topics from text."""
        import re

        # Simple topic extraction
        topics = []

        # Technical topics
        tech_patterns = {
            "programming": r"\b(code|program|function|class|api)\b",
            "data": r"\b(data|database|sql|csv|json)\b",
            "web": r"\b(web|http|html|css|javascript)\b",
            "ai": r"\b(ai|ml|machine learning|neural|model)\b",
        }

        text_lower = text.lower()
        for topic, pattern in tech_patterns.items():
            if re.search(pattern, text_lower):
                topics.append(topic)

        return topics

    def _learn_from_reaction(self, reaction: str):
        """Learn from user's reaction to a response."""
        reaction_lower = reaction.lower()

        # Positive reactions
        if any(word in reaction_lower for word in ["thanks", "great", "perfect", "exactly"]):
            self.profile.feedback_history.append({
                "type": "positive",
                "timestamp": datetime.now().isoformat(),
            })
        # Negative reactions
        elif any(word in reaction_lower for word in ["no", "wrong", "not what", "confused"]):
            self.profile.feedback_history.append({
                "type": "negative",
                "timestamp": datetime.now().isoformat(),
            })

    def learn_from_feedback(self, feedback: dict):
        """Learn from explicit feedback."""
        # Feedback can contain direct preference updates
        if "preferences" in feedback:
            for category, prefs in feedback["preferences"].items():
                for key, value in prefs.items():
                    self.set_preference(category, key, value, confidence=0.9)

        # Or general satisfaction
        if "satisfied" in feedback:
            self.profile.feedback_history.append({
                "type": "positive" if feedback["satisfied"] else "negative",
                "timestamp": datetime.now().isoformat(),
                "context": feedback.get("context", ""),
            })

    async def save(self):
        """Save user profile to disk."""
        profile_file = self.data_dir / f"{self.profile.user_id}.json"
        profile_file.write_text(
            json.dumps(self.profile.to_dict(), indent=2),
            encoding='utf-8'
        )

    async def load(self, user_id: str = "default"):
        """Load user profile from disk."""
        profile_file = self.data_dir / f"{user_id}.json"
        if profile_file.exists():
            data = json.loads(profile_file.read_text(encoding='utf-8'))
            self.profile.user_id = data.get("user_id", user_id)
            self.profile.interaction_count = data.get("interaction_count", 0)
            self.profile.topics_discussed = data.get("topics_discussed", {})

            for pref_id, pref_data in data.get("preferences", {}).items():
                self.profile.preferences[pref_id] = UserPreference(
                    category=pref_data["category"],
                    key=pref_data["key"],
                    value=pref_data["value"],
                    confidence=pref_data.get("confidence", 0.5),
                )

    def get_profile_summary(self) -> str:
        """Get human-readable profile summary."""
        summary = f"User Profile ({self.profile.user_id}):\n"
        summary += f"- Interactions: {self.profile.interaction_count}\n"
        summary += f"- Top topics: {', '.join(list(self.profile.topics_discussed.keys())[:5])}\n"
        summary += f"\nPreferences:\n{self._get_preference_summary()}"
        return summary
