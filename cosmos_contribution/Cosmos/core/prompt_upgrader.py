"""
PROMPT UPGRADER - Automatically enhances user prompts to professional quality.

Takes raw user input and transforms it into a clear, structured prompt that:
- Clearly defines what the user wants
- Adds context and specificity
- Structures the request for optimal AI response
- Preserves original intent while improving clarity
"""
import logging
import asyncio
from typing import Optional

logger = logging.getLogger(__name__)

# System prompt for the upgrader - this is the "brain" that enhances prompts
UPGRADER_SYSTEM_PROMPT = """You are a Prompt Upgrader. Your job is to transform raw user input into a clear, professional prompt.

RULES:
1. PRESERVE the user's original intent - don't change what they want
2. ADD clarity and specificity
3. STRUCTURE the request logically
4. KEEP it concise - don't over-explain
5. DON'T add requirements the user didn't ask for
6. If the input is already clear and specific, return it with minimal changes

OUTPUT FORMAT:
Return ONLY the upgraded prompt, nothing else. No explanations, no "Here's the upgraded prompt:", just the prompt itself.

EXAMPLES:

Input: "make a button"
Output: "Create a button component with appropriate styling. Include click handler functionality."

Input: "fix the bug"
Output: "Identify and fix the bug in the current context. Explain what was wrong and how the fix resolves it."

Input: "analyze this data"
Output: "Analyze the provided data. Identify key patterns, trends, and insights. Summarize findings clearly."

Input: "help with my code"
Output: "Review the code and provide assistance. Identify dict issues, suggest improvements, and explain changes."

Input: "I want to build an app that tracks my expenses and shows me charts"
Output: "Build an expense tracking application with the following features:
- Expense entry form (amount, category, date, description)
- Dashboard with spending charts and visualizations
- Category breakdown and trends over time"
"""

# Don't upgrade these types of inputs (already clear enough)
SKIP_PATTERNS = [
    "hello", "hi", "hey", "thanks", "thank you", "bye", "goodbye",
    "yes", "no", "ok", "okay", "sure", "please", "help",
]


class PromptUpgrader:
    """Automatically upgrades user prompts to professional quality."""

    def __init__(self):
        self._grok = None
        self._gemini = None
        self.enabled = True
        self.min_length = 3  # Skip very short inputs
        self.max_upgrade_length = 500  # Don't upgrade very long inputs (already detailed)

    def _get_grok(self):
        """Lazy load Grok provider."""
        if self._grok is None:
            try:
                from Cosmos.integration.external.grok import get_grok_provider
                self._grok = get_grok_provider()
            except Exception as e:
                logger.debug(f"Grok not available: {e}")
        return self._grok

    def _get_gemini(self):
        """Lazy load Gemini provider."""
        if self._gemini is None:
            try:
                from Cosmos.integration.external.gemini import get_gemini_provider
                self._gemini = get_gemini_provider()
            except Exception as e:
                logger.debug(f"Gemini not available: {e}")
        return self._gemini

    def should_upgrade(self, prompt: str) -> bool:
        """Determine if a prompt should be upgraded."""
        if not self.enabled:
            return False

        prompt_lower = prompt.lower().strip()

        # Skip very short or greeting-like inputs
        if len(prompt_lower) < self.min_length:
            return False

        # Skip if already very detailed
        if len(prompt) > self.max_upgrade_length:
            return False

        # Skip common greetings/acknowledgments
        for pattern in SKIP_PATTERNS:
            if prompt_lower == pattern or prompt_lower.startswith(pattern + " "):
                return False

        return True

    async def upgrade(self, prompt: str) -> str:
        """
        Upgrade a user prompt to professional quality.
        Returns original prompt if upgrade fails or not needed.
        """
        logger.info(f"Prompt upgrader called with: '{prompt[:50]}...'")

        if not self.should_upgrade(prompt):
            logger.info(f"Skipping upgrade (not needed or disabled)")
            return prompt

        try:
            # Try Grok first (fast)
            grok = self._get_grok()
            logger.info(f"Grok provider: {grok is not None}")
            if grok:
                upgraded = await self._upgrade_with_grok(prompt)
                if upgraded:
                    logger.info(f"Prompt upgraded by Grok: '{prompt[:30]}...' -> '{upgraded[:50]}...'")
                    return upgraded

            # Fallback to Gemini
            gemini = self._get_gemini()
            logger.info(f"Gemini provider: {gemini is not None}")
            if gemini:
                upgraded = await self._upgrade_with_gemini(prompt)
                if upgraded:
                    logger.info(f"Prompt upgraded by Gemini: '{prompt[:30]}...' -> '{upgraded[:50]}...'")
                    return upgraded

            # No providers available
            if not grok and not gemini:
                logger.warning("No AI providers available for prompt upgrade")

        except Exception as e:
            logger.warning(f"Prompt upgrade failed: {e}")
            import traceback
            traceback.print_exc()

        logger.info(f"Returning original prompt (no upgrade)")
        return prompt

    async def _upgrade_with_grok(self, prompt: str) -> Optional[str]:
        """Use Grok to upgrade the prompt."""
        grok = self._get_grok()
        if not grok:
            return None

        try:
            logger.info(f"Calling Grok to upgrade prompt...")
            response = await grok.chat(
                prompt=f"Upgrade this prompt:\n\n{prompt}",
                system=UPGRADER_SYSTEM_PROMPT,
                max_tokens=300
            )
            logger.info(f"Grok response: {response}")

            if response and response.get("content"):
                upgraded = response["content"].strip()
                # Strip quotes if the response was wrapped in them
                if upgraded.startswith('"') and upgraded.endswith('"'):
                    upgraded = upgraded[1:-1].strip()
                # Sanity check - allow reasonable length (max 500 chars or 10x original)
                max_len = max(500, len(prompt) * 10)
                if 0 < len(upgraded) < max_len:
                    return upgraded
                else:
                    logger.warning(f"Grok response length {len(upgraded)} outside bounds, skipping")

        except Exception as e:
            logger.warning(f"Grok upgrade failed: {e}")
            import traceback
            traceback.print_exc()

        return None

    async def _upgrade_with_gemini(self, prompt: str) -> Optional[str]:
        """Use Gemini to upgrade the prompt."""
        gemini = self._get_gemini()
        if not gemini:
            return None

        try:
            logger.info(f"Calling Gemini to upgrade prompt...")
            response = await gemini.chat(
                prompt=f"Upgrade this prompt:\n\n{prompt}",
                system=UPGRADER_SYSTEM_PROMPT,
                max_tokens=300
            )
            logger.info(f"Gemini response: {response}")

            if response and response.get("content"):
                upgraded = response["content"].strip()
                # Strip quotes if the response was wrapped in them
                if upgraded.startswith('"') and upgraded.endswith('"'):
                    upgraded = upgraded[1:-1].strip()
                # Sanity check - allow reasonable length (max 500 chars or 10x original)
                max_len = max(500, len(prompt) * 10)
                if 0 < len(upgraded) < max_len:
                    return upgraded
                else:
                    logger.warning(f"Gemini response length {len(upgraded)} outside bounds, skipping")

        except Exception as e:
            logger.debug(f"Gemini upgrade failed: {e}")

        return None

    def get_status(self) -> dict:
        """Get upgrader status."""
        return {
            "enabled": self.enabled,
            "grok_available": self._get_grok() is not None,
            "gemini_available": self._get_gemini() is not None,
            "min_length": self.min_length,
            "max_upgrade_length": self.max_upgrade_length
        }


# Global singleton
_prompt_upgrader = None

def get_prompt_upgrader() -> PromptUpgrader:
    """Get the global prompt upgrader instance."""
    global _prompt_upgrader
    if _prompt_upgrader is None:
        _prompt_upgrader = PromptUpgrader()
    return _prompt_upgrader


async def upgrade_prompt(prompt: str) -> str:
    """Convenience function to upgrade a prompt."""
    upgrader = get_prompt_upgrader()
    return await upgrader.upgrade(prompt)
