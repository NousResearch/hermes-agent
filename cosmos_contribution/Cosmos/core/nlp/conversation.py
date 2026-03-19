"""
Conversation Handler for Multi-Turn NLP Interactions.

Maintains context across multiple exchanges.
"""

import logging
from typing import   Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque

from .intent_parser import Intent
from .command_router import CommandRouter, CommandResult

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    intent: Optional[Intent] = None
    result: Optional[CommandResult] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConversationContext:
    """Context maintained across conversation turns."""
    # Recent entities for reference resolution
    last_token: Optional[str] = None
    last_chain: Optional[str] = None
    last_amount: Optional[float] = None
    last_url: Optional[str] = None

    # Conversation state
    current_topic: Optional[str] = None
    pending_action: Optional[str] = None
    awaiting_confirmation: bool = False

    # User preferences learned
    preferences: dict = field(default_factory=dict)


class ConversationHandler:
    """
    Handle multi-turn conversations with context.

    Maintains state across exchanges for:
    - Reference resolution ("do it again", "that token")
    - Confirmation flows ("are you sure?")
    - Context-aware responses
    """

    def __init__(self, max_history: int = 20):
        self.router = CommandRouter()
        self.history: deque[ConversationTurn] = deque(maxlen=max_history)
        self.context = ConversationContext()

    async def process(self, user_input: str) -> str:
        """
        Process user input in conversation context.

        Args:
            user_input: User's message

        Returns:
            Response string
        """
        # Check for confirmation responses
        if self.context.awaiting_confirmation:
            return await self._handle_confirmation(user_input)

        # Resolve references to previous context
        resolved_input = self._resolve_references(user_input)

        # Execute command
        result = await self.router.execute(resolved_input)

        # Update context from result
        self._update_context(user_input, result)

        # Add to history
        self.history.append(ConversationTurn(
            role="user",
            content=user_input,
            result=result,
        ))

        self.history.append(ConversationTurn(
            role="assistant",
            content=result.response,
        ))

        return result.response

    def _resolve_references(self, text: str) -> str:
        """Resolve references like 'it', 'that', 'again'."""
        text_lower = text.lower()

        # "Do it again" / "repeat" / "again"
        if any(word in ["again", "repeat", "same"]):
            if self.history:
                # Find last user message
                for turn in reversed(self.history):
                    if turn.role == "user" and turn.content != text:
                        return turn.content

        # "That token" / "it" referring to last token
        if self.context.last_token:
            if "that token" in text_lower or "that coin" in text_lower:
                text = text.replace("that token", self.context.last_token)
                text = text.replace("that coin", self.context.last_token)

            if " it " in text_lower and self.context.current_topic == "crypto":
                # Careful replacement to avoid false positives
                pass

        # "Same amount" / "that amount"
        if self.context.last_amount and "same amount" in text_lower:
            text = text.replace("same amount", str(self.context.last_amount))

        return text

    def _update_context(self, user_input: str, result: CommandResult):
        """Update context from the latest interaction."""
        from .entity_extractor import EntityExtractor
        extractor = EntityExtractor()
        entities = extractor.extract(user_input)

        # Update token/chain context
        if entities.tokens:
            self.context.last_token = entities.tokens[0].upper()

        if entities.chains:
            self.context.last_chain = entities.chains[0]

        if entities.amounts:
            self.context.last_amount = entities.amounts[0]["value"]

        if entities.urls:
            self.context.last_url = entities.urls[0]

        # Update topic based on task type
        if result.task_type:
            from .task_classifier import TaskType
            if result.task_type in (TaskType.TRADE, TaskType.CRYPTO, TaskType.PREDICT):
                self.context.current_topic = "crypto"
            elif result.task_type in (TaskType.CODE, TaskType.FIX):
                self.context.current_topic = "development"
            else:
                self.context.current_topic = None

    async def _handle_confirmation(self, user_input: str) -> str:
        """Handle yes/no confirmation responses."""
        text_lower = user_input.lower().strip()

        affirmative = ["yes", "y", "yeah", "yep", "sure", "ok", "okay", "confirm", "do it"]
        negative = ["no", "n", "nope", "cancel", "stop", "abort", "nevermind"]

        self.context.awaiting_confirmation = False

        if any(word in affirmative):
            # Execute pending action
            if self.context.pending_action:
                result = await self.router.execute(self.context.pending_action)
                self.context.pending_action = None
                return result.response
            return "Confirmed, but I'm not sure what to do."

        elif any(word in negative):
            self.context.pending_action = None
            return "Cancelled."

        else:
            # Unclear response
            self.context.awaiting_confirmation = True
            return "I didn't understand. Please say yes or no."

    def request_confirmation(self, action: str, prompt: str) -> str:
        """
        Request user confirmation before an action.

        Args:
            action: The action to execute if confirmed
            prompt: The confirmation prompt to show

        Returns:
            Confirmation prompt
        """
        self.context.awaiting_confirmation = True
        self.context.pending_action = action
        return prompt

    def get_history_summary(self) -> str:
        """Get a summary of recent conversation history."""
        if not self.history:
            return "No conversation history."

        summary_parts = []
        for turn in list(self.history)[-6:]:  # Last 3 exchanges
            role = "You" if turn.role == "user" else "Farn"
            summary_parts.append(f"{role}: {turn.content[:100]}...")

        return "\n".join(summary_parts)

    def clear_history(self):
        """Clear conversation history."""
        self.history.clear()
        self.context = ConversationContext()
