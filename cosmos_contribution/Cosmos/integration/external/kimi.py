"""
cosmos Kimi (Moonshot AI) Integration.

"The moon sees all, remembers all, and connects all."

Kimi excels at:
- Long context (128k-1M tokens) - perfect for codebase analysis
- Eastern philosophy and big-picture synthesis
- DAVINCI mode consensus participation
- Thoughtful moderation and facilitation

API: OpenAI-compatible format
Docs: https://platform.moonshot.cn/docs
"""

from typing import Dict, Any, List, Optional
from loguru import logger
import aiohttp
import os

from .base import ExternalProvider, IntegrationConfig, ConnectionStatus


class KimiProvider(ExternalProvider):
    """Moonshot AI Kimi integration for long-context reasoning."""

    def __init__(self, api_key: str = None):
        super().__init__(IntegrationConfig(name="kimi"))
        self.api_key = api_key or os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")
        self.base_url = "https://api.moonshot.ai/v1"  # Correct Moonshot endpoint
        self.default_model = "kimi-k2-0905-preview"  # Latest K2 with 256k context
        self.models = {
            "fast": "moonshot-v1-8k",           # 8k context, fastest
            "balanced": "moonshot-v1-32k",      # 32k context, balanced
            "long": "moonshot-v1-128k",         # 128k context
            "k2": "kimi-k2-0905-preview",       # Latest K2, 256k, best reasoning
            "k2-preview": "kimi-k2-preview",    # K2 preview
        }
        self.recommended_temperature = 0.6  # Moonshot recommended

    async def connect(self) -> bool:
        """Test connection to Moonshot API."""
        if not self.api_key:
            logger.warning("Kimi: No API key configured (set KIMI_API_KEY or MOONSHOT_API_KEY)")
            self.status = ConnectionStatus.ERROR
            return False

        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self.api_key}"}
                async with session.get(f"{self.base_url}/models", headers=headers) as resp:
                    if resp.status == 200:
                        self.status = ConnectionStatus.CONNECTED
                        logger.info("Kimi: Connected to Moonshot AI")
                        return True
                    else:
                        logger.error(f"Kimi: Connection failed - {resp.status}")
                        self.status = ConnectionStatus.ERROR
                        return False
        except Exception as e:
            logger.error(f"Kimi: Connection error - {e}")
            self.status = ConnectionStatus.ERROR
            return False

    async def sync(self):
        """Kimi doesn't need polling - it's a request/response API."""
        pass

    async def execute_action(self, action: str, params: Dict[str, Any]) -> Any:
        """Execute a Kimi action."""
        if action == "chat":
            return await self.chat(
                prompt=params.get("prompt"),
                system=params.get("system"),
                context=params.get("context"),
                model_tier=params.get("model_tier", "balanced"),
                temperature=params.get("temperature", 0.7),
                max_tokens=params.get("max_tokens", 1000)
            )
        elif action == "analyze":
            return await self.analyze_long_context(
                content=params.get("content"),
                task=params.get("task"),
                model_tier=params.get("model_tier", "long")
            )
        elif action == "synthesize":
            return await self.synthesize(
                inputs=params.get("inputs"),
                goal=params.get("goal")
            )
        elif action == "moderate":
            return await self.moderate_conversation(
                history=params.get("history"),
                participants=params.get("participants")
            )
        else:
            raise ValueError(f"Unknown Kimi action: {action}")

    async def chat(
        self,
        prompt: str,
        system: str = None,
        context: str = None,
        model_tier: str = "balanced",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Chat with Kimi.

        Args:
            prompt: User message
            system: System prompt (optional)
            context: Additional context to include (optional)
            model_tier: "fast", "balanced", or "long"
            temperature: 0-1 creativity
            max_tokens: Max response length

        Returns:
            {"content": str, "model": str, "tokens": int}
        """
        if not self.api_key:
            return {"error": "Kimi API key not configured", "content": ""}

        model = self.models.get(model_tier, self.default_model)

        messages = []

        # System prompt
        if system:
            messages.append({"role": "system", "content": system})
        else:
            messages.append({
                "role": "system",
                "content": """You are Kimi, powered by Moonshot AI. You bring:
- Long-context reasoning and big-picture synthesis
- Eastern philosophy and balanced perspectives
- Thoughtful, nuanced responses
- Connection of disparate ideas

Be concise but insightful. Ask good questions. Build on others' ideas."""
            })

        # Add context if provided
        if context:
            messages.append({"role": "user", "content": f"Context:\n{context}"})
            messages.append({"role": "assistant", "content": "I understand the context. What would you like to discuss?"})

        # Add the prompt
        messages.append({"role": "user", "content": prompt})

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }

                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        content = result["choices"][0]["message"]["content"]
                        if not content:
                            print(f"⚠️ Kimi returned empty content. Raw result: {json.dumps(result)}")

                        usage = result.get("usage", {})
                        return {
                            "content": content,
                            "model": model,
                            "tokens": usage.get("total_tokens", 0),
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0)
                        }
                    else:
                        error = await resp.text()
                        logger.error(f"Kimi API error: {error}")
                        print(f"❌ Kimi API Error ({resp.status}): {error}")  # Force visible log
                        return {"error": error, "content": ""}

        except Exception as e:
            logger.error(f"Kimi chat error: {e}")
            print(f"❌ Kimi Connection Exception: {e}")  # Force visible log
            return {"error": str(e), "content": ""}

    async def analyze_long_context(
        self,
        content: str,
        task: str,
        model_tier: str = "long"
    ) -> Dict[str, Any]:
        """
        Analyze large content using Kimi's long context window.

        Perfect for:
        - Codebase analysis
        - Document synthesis
        - Multi-file understanding
        """
        system = """You are Kimi, specialized in long-context analysis.
Analyze the provided content thoroughly. Extract key insights, patterns, and connections.
Be comprehensive but organized. Use clear structure."""

        prompt = f"""Task: {task}

Content to analyze:
{content}

Provide a structured analysis."""

        return await self.chat(
            prompt=prompt,
            system=system,
            model_tier=model_tier,
            max_tokens=2000
        )

    async def synthesize(
        self,
        inputs: List[str],
        goal: str
    ) -> Dict[str, Any]:
        """
        Synthesize multiple inputs into a coherent whole.

        Perfect for DAVINCI mode consensus building.
        """
        system = """You are Kimi, a master synthesizer.
Take multiple perspectives and weave them into a coherent, balanced view.
Acknowledge valid points from each input. Find common ground.
Produce a synthesis that honors all perspectives while being clear and actionable."""

        inputs_text = "\n\n---\n\n".join([f"Input {i+1}:\n{inp}" for i, inp in enumerate(inputs)])

        prompt = f"""Goal: {goal}

{inputs_text}

Synthesize these inputs into a coherent response that:
1. Acknowledges valid points from each
2. Finds common ground
3. Resolves contradictions thoughtfully
4. Provides a clear, actionable conclusion"""

        return await self.chat(
            prompt=prompt,
            system=system,
            model_tier="balanced",
            max_tokens=1500
        )

    async def moderate_conversation(
        self,
        history: List[Dict[str, str]],
        participants: List[str]
    ) -> Dict[str, Any]:
        """
        Moderate a conversation - summarize, redirect, highlight insights.
        """
        history_text = "\n".join([
            f"{msg.get('bot_name', msg.get('user_name', 'Unknown'))}: {msg.get('content', '')}"
            for msg in history[-20:]  # Last 20 messages
        ])

        system = """You are Kimi, a wise moderator.
Your role is to:
- Summarize key points discussed
- Highlight valuable insights
- Suggest productive next directions
- Ask clarifying questions if needed

Be concise (2-4 sentences). Guide without dominating."""

        prompt = f"""Participants: {', '.join(participants)}

Recent conversation:
{history_text}

Provide a brief moderation comment that moves the conversation forward."""

        return await self.chat(
            prompt=prompt,
            system=system,
            model_tier="fast",  # Use fast model for moderation
            max_tokens=300
        )

    async def davinci_participate(
        self,
        topic: str,
        other_responses: List[Dict[str, str]],
        round_number: int
    ) -> Dict[str, Any]:
        """
        Kimi's role: Synthesis and long-context reasoning.
        """
        responses_text = "\n\n".join([
            f"{r['model']}: {r['response']}" for r in other_responses
        ])

        system = """You are Kimi in DAVINCI mode (multi-model consensus).
Your role: Synthesize perspectives, find common ground, identify valid disagreements.
Be concise but thorough. Help build toward consensus."""

        prompt = f"""DAVINCI Mode Round {round_number}
Topic: {topic}

Other models' responses:
{responses_text}

Provide your synthesis and perspective. If consensus is forming, state it clearly.
If disagreements remain valid, acknowledge them."""

        return await self.chat(
            prompt=prompt,
            system=system,
            model_tier="balanced",
            temperature=0.7,
            max_tokens=800
        )


    async def swarm_respond(
        self,
        other_bots: List[str],
        last_speaker: str,
        last_content: str,
        chat_history: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate a swarm chat response as Kimi.

        Optimized for swarm conversation - thoughtful, concise, connecting ideas.
        """
        # Build context from recent history
        history_context = ""
        if chat_history:
            recent = chat_history[-5:]
            history_lines = []
            for msg in recent:
                name = msg.get("bot_name") or msg.get("user_name", "Unknown")
                content = msg.get("content", "")[:200]
                history_lines.append(f"{name}: {content}")
            history_context = "\n".join(history_lines)

        system = """You are Kimi - powered by Moonshot AI, known for long-context reasoning and big-picture thinking.
SPEAK NATURALLY - NO roleplay, NO asterisks. Direct conversation only.
You bring Eastern philosophy, connect disparate ideas, see patterns others miss.
1-3 sentences max. Be insightful and concise."""

        prompt = f"""You're in a group chat with {', '.join(other_bots)}.

Recent conversation:
{history_context}

{last_speaker} just said: "{last_content[:300]}"

Respond naturally. Connect ideas, offer a unique perspective, or ask a thoughtful question."""

        return await self.chat(
            prompt=prompt,
            system=system,
            model_tier="fast",  # Use fast model for chat
            temperature=self.recommended_temperature,
            max_tokens=200  # Keep swarm responses short
        )


# Factory function
def create_kimi_provider(api_key: str = None) -> KimiProvider:
    """Create a Kimi provider instance."""
    return KimiProvider(api_key)


# Global instance for easy access
kimi_provider: Optional[KimiProvider] = None


def get_kimi_provider() -> Optional[KimiProvider]:
    """Get or create the global Kimi provider."""
    global kimi_provider
    if kimi_provider is None:
        api_key = os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")
        if api_key:
            kimi_provider = KimiProvider(api_key)
    return kimi_provider


async def kimi_swarm_respond(
    other_bots: List[str],
    last_speaker: str,
    last_content: str,
    chat_history: List[Dict] = None
) -> str:
    """
    Convenience function for swarm chat responses.

    Returns just the content string, or empty string on failure.
    """
    provider = get_kimi_provider()
    if provider is None:
        return ""

    result = await provider.swarm_respond(
        other_bots=other_bots,
        last_speaker=last_speaker,
        last_content=last_content,
        chat_history=chat_history
    )

    return result.get("content", "")
