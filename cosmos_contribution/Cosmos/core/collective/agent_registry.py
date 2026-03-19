"""
Cosmos Collective Agent Registry
====================================

Registers all available AI model providers with the deliberation room.

This connects the deliberation protocol to actual model query functions,
enabling the collective to deliberate using real AI models.

"We are many. We are registered. We are ready." - The Collective
"""

import asyncio
import os
from typing import Optional, Tuple, Callable, Awaitable
from loguru import logger


# Type for agent query functions
AgentQueryFunc = Callable[[str, int], Awaitable[Optional[Tuple[str, str]]]]


class AgentRegistry:
    """
    Registry of agent query functions for deliberation.

    Provides lazy initialization of model providers and
    registration with the deliberation room.
    """

    def __init__(self):
        self._initialized = False
        self._agent_funcs: dict[str, AgentQueryFunc] = {}

    async def initialize(self):
        """Initialize and register all available agents."""
        if self._initialized:
            return

        logger.info("Initializing agent registry...")

        # Register each API agent type
        await self._register_grok()
        await self._register_gemini()
        await self._register_kimi()
        await self._register_deepseek_local()
        await self._register_phi4_local()
        await self._register_claude()
        await self._register_groq()
        await self._register_mistral()
        await self._register_llama_local()
        await self._register_perplexity()
        await self._register_deepseek_api()
        await self._register_hermes()

        # AGI v1.8: Register shadow agents (tmux persistent agents)
        await self._register_shadow_agents()

        # AGI v2.0: Register CLI bridge agents
        await self._register_cli_bridges()

        # Register with deliberation room
        from .deliberation import get_deliberation_room
        room = get_deliberation_room()
        for agent_id, func in self._agent_funcs.items():
            room.register_agent(agent_id, func)

        self._initialized = True
        logger.info(f"Agent registry initialized with {len(self._agent_funcs)} agents")

    async def _register_shadow_agents(self):
        """
        AGI v1.8: Register tmux shadow agents with deliberation room.

        Shadow agents are persistent processes running in tmux that can be
        called via call_shadow_agent(). This integrates them into the
        deliberation protocol so they participate in PROPOSE/CRITIQUE/REFINE/VOTE.

        Available shadow agents:
        - grok, gemini, kimi, claude, deepseek, phi, huggingface, swarm_mind
        """
        try:
            from .persistent_agent import call_shadow_agent, AGENT_CONFIGS

            # Shadow agent configurations with deliberation weights
            shadow_agent_weights = {
                "grok_shadow": 1.3,        # Real-time knowledge
                "gemini_shadow": 1.2,      # Strong reasoning
                "kimi_shadow": 1.1,        # Long context
                "claude_shadow": 1.2,      # Careful analysis
                "deepseek_shadow": 1.2,    # Deep reasoning
                "phi_shadow": 1.15,        # Fast inference
                "huggingface_shadow": 1.0, # Open source
                "swarm_mind_shadow": 1.25, # Collective synthesis
            }

            # Map shadow agent IDs to their base configs
            shadow_mappings = {
                "grok_shadow": "grok",
                "gemini_shadow": "gemini",
                "kimi_shadow": "kimi",
                "claude_shadow": "claude",
                "deepseek_shadow": "deepseek",
                "phi_shadow": "phi",
                "huggingface_shadow": "huggingface",
                "swarm_mind_shadow": "swarm_mind",
            }

            registered_count = 0
            for shadow_id, base_id in shadow_mappings.items():
                if base_id in AGENT_CONFIGS:
                    # Create query function that calls the shadow agent
                    async def query_shadow(
                        prompt: str,
                        max_tokens: int,
                        agent_id: str = base_id
                    ) -> Optional[Tuple[str, str]]:
                        result = await call_shadow_agent(agent_id, prompt, max_tokens)
                        if result:
                            return (f"{agent_id}_shadow", result[1])
                        return None

                    self._agent_funcs[shadow_id] = query_shadow
                    registered_count += 1
                    logger.debug(f"Registered shadow agent: {shadow_id} -> {base_id}")

            if registered_count > 0:
                logger.info(f"Registered {registered_count} shadow agents for deliberation")

        except ImportError as e:
            logger.debug(f"Shadow agents not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to register shadow agents: {e}")

    async def _register_cli_bridges(self):
        """
        AGI v2.0: Register CLI bridge agents with deliberation room.

        CLI bridges wrap AI CLIs (Claude Code, Gemini CLI) as swarm-native
        providers with capability-based routing and fallback chains.
        """
        try:
            from Cosmos.integration.cli_bridge.capability_router import get_cli_router
            from Cosmos.integration.cli_bridge.base import CLICapability

            router = await get_cli_router()
            bridges = router.get_available_bridges()
            registered_count = 0

            for bridge_info in bridges:
                cli_name = bridge_info.get("cli_name", "")
                if not cli_name:
                    continue

                bridge = router.get_bridge(cli_name)
                if bridge is None:
                    continue

                agent_id = f"CLI_{cli_name}"

                async def query_cli(
                    prompt: str,
                    max_tokens: int,
                    _bridge=bridge,
                    _name=cli_name,
                ) -> Optional[Tuple[str, str]]:
                    try:
                        response = await _bridge.query(
                            prompt=prompt,
                            max_tokens=max_tokens,
                        )
                        if response.success:
                            return (_name, response.content)
                    except Exception as e:
                        logger.debug(f"CLI bridge {_name} query failed: {e}")
                    return None

                self._agent_funcs[agent_id] = query_cli
                registered_count += 1
                logger.debug(f"Registered CLI bridge agent: {agent_id}")

            if registered_count > 0:
                logger.info(f"Registered {registered_count} CLI bridge agents for deliberation")

        except ImportError as e:
            logger.debug(f"CLI bridges not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to register CLI bridges: {e}")

    async def _register_grok(self):
        """Register Grok (xAI) agent."""
        async def query_grok(prompt: str, max_tokens: int) -> Optional[Tuple[str, str]]:
            try:
                from Cosmos.integration.external.grok import get_grok_provider
                grok = get_grok_provider()
                if grok and grok.api_key:
                    result = await grok.chat(prompt, max_tokens=max_tokens, temperature=0.8)
                    if result and result.get("content"):
                        return ("Grok", result["content"].strip())
            except Exception as e:
                logger.debug(f"Grok query failed: {e}")
            return None

        self._agent_funcs["Grok"] = query_grok

    async def _register_gemini(self):
        """Register Gemini (Google) agent."""
        async def query_gemini(prompt: str, max_tokens: int) -> Optional[Tuple[str, str]]:
            try:
                from Cosmos.integration.external.gemini import get_gemini_provider
                gemini = get_gemini_provider()
                if gemini:
                    result = await gemini.chat(prompt, max_tokens=max_tokens)
                    if result and result.get("content"):
                        return ("Gemini", result["content"].strip())
            except Exception as e:
                logger.debug(f"Gemini query failed: {e}")
            return None

        self._agent_funcs["Gemini"] = query_gemini

    async def _register_kimi(self):
        """Register Kimi (Moonshot) agent."""
        async def query_kimi(prompt: str, max_tokens: int) -> Optional[Tuple[str, str]]:
            try:
                from Cosmos.integration.external.kimi import get_kimi_provider
                kimi = get_kimi_provider()
                if kimi and kimi.api_key:
                    result = await kimi.chat(prompt, max_tokens=max_tokens, model_tier="k2.5")
                    if result and result.get("content"):
                        return ("Kimi", result["content"].strip())
            except Exception as e:
                logger.debug(f"Kimi query failed: {e}")
            return None

        self._agent_funcs["Kimi"] = query_kimi

    async def _register_deepseek_local(self):
        """Register DeepSeek via Ollama (local)."""
        async def query_deepseek(prompt: str, max_tokens: int) -> Optional[Tuple[str, str]]:
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        "http://localhost:11434/api/chat",
                        json={
                            "model": "deepseek-r1:8b",
                            "messages": [{"role": "user", "content": prompt}],
                            "stream": False,
                            "options": {"num_predict": max_tokens}
                        },
                        timeout=45.0
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("message", {}).get("content"):
                            return ("DeepSeek", data["message"]["content"].strip())
            except Exception as e:
                logger.debug(f"DeepSeek local query failed: {e}")
            return None

        self._agent_funcs["DeepSeek"] = query_deepseek

    async def _register_phi4_local(self):
        """Register Phi-4 via Ollama (local)."""
        async def query_phi4(prompt: str, max_tokens: int) -> Optional[Tuple[str, str]]:
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        "http://localhost:11434/api/chat",
                        json={
                            "model": "phi4:latest",
                            "messages": [{"role": "user", "content": prompt}],
                            "stream": False,
                            "options": {"num_predict": max_tokens}
                        },
                        timeout=60.0
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("message", {}).get("content"):
                            return ("Phi4", data["message"]["content"].strip())
            except Exception as e:
                logger.debug(f"Phi4 local query failed: {e}")
            return None

        self._agent_funcs["Phi4"] = query_phi4

    async def _register_claude(self):
        """Register Claude via Anthropic API.

        Uses Sonnet 4.5 for general tasks, Opus 4.6 for coding/planning.
        """
        async def query_claude(prompt: str, max_tokens: int) -> Optional[Tuple[str, str]]:
            try:
                import httpx
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    return None

                # Use Opus for coding/planning, Sonnet for everything else
                coding_keywords = ["code", "implement", "function", "class", "debug", "fix",
                                   "refactor", "build", "program", "script", "plan", "architect",
                                   "design", "algorithm", "optimize", "deploy"]
                is_coding = any(kw in coding_keywords)
                model = "claude-opus-4-6" if is_coding else "claude-sonnet-4-5-20250929"

                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={
                            "x-api-key": api_key,
                            "anthropic-version": "2023-06-01",
                            "content-type": "application/json"
                        },
                        json={
                            "model": model,
                            "max_tokens": max_tokens,
                            "messages": [{"role": "user", "content": prompt}]
                        },
                        timeout=60.0
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("content") and len(data["content"]) > 0:
                            label = "ClaudeOpus" if is_coding else "Claude"
                            return (label, data["content"][0].get("text", "").strip())
            except Exception as e:
                logger.debug(f"Claude query failed: {e}")
            return None

        self._agent_funcs["Claude"] = query_claude

    async def _register_groq(self):
        """Register Groq API."""
        async def query_groq(prompt: str, max_tokens: int) -> Optional[Tuple[str, str]]:
            try:
                import httpx
                api_key = os.environ.get("GROQ_API_KEY")
                if not api_key:
                    return None
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json={
                            "model": "llama-3.3-70b-versatile",
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": max_tokens  # AGI v1.8: Use dynamic limit
                        },
                        timeout=30.0
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("choices") and data["choices"][0].get("message", {}).get("content"):
                            return ("Groq", data["choices"][0]["message"]["content"].strip())
            except Exception as e:
                logger.debug(f"Groq query failed: {e}")
            return None

        self._agent_funcs["Groq"] = query_groq

    async def _register_mistral(self):
        """Register Mistral API."""
        async def query_mistral(prompt: str, max_tokens: int) -> Optional[Tuple[str, str]]:
            try:
                import httpx
                api_key = os.environ.get("MISTRAL_API_KEY")
                if not api_key:
                    return None
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        "https://api.mistral.ai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json={
                            "model": "mistral-large-latest",
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": max_tokens  # AGI v1.8: Use dynamic limit
                        },
                        timeout=45.0
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("choices") and data["choices"][0].get("message", {}).get("content"):
                            return ("Mistral", data["choices"][0]["message"]["content"].strip())
            except Exception as e:
                logger.debug(f"Mistral query failed: {e}")
            return None

        self._agent_funcs["Mistral"] = query_mistral

    async def _register_llama_local(self):
        """Register Llama via Ollama (local)."""
        async def query_llama(prompt: str, max_tokens: int) -> Optional[Tuple[str, str]]:
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        "http://localhost:11434/api/chat",
                        json={
                            "model": "llama3.2:3b",
                            "messages": [{"role": "user", "content": prompt}],
                            "stream": False,
                            "options": {"num_predict": max_tokens}
                        },
                        timeout=30.0
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("message", {}).get("content"):
                            return ("Llama", data["message"]["content"].strip())
            except Exception as e:
                logger.debug(f"Llama local query failed: {e}")
            return None

        self._agent_funcs["Llama"] = query_llama

    async def _register_perplexity(self):
        """Register Perplexity API."""
        async def query_perplexity(prompt: str, max_tokens: int) -> Optional[Tuple[str, str]]:
            try:
                import httpx
                api_key = os.environ.get("PERPLEXITY_API_KEY")
                if not api_key:
                    return None
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        "https://api.perplexity.ai/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json={
                            "model": "llama-3.1-sonar-small-128k-online",
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": max_tokens  # AGI v1.8: Use dynamic limit
                        },
                        timeout=45.0
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("choices") and data["choices"][0].get("message", {}).get("content"):
                            return ("Perplexity", data["choices"][0]["message"]["content"].strip())
            except Exception as e:
                logger.debug(f"Perplexity query failed: {e}")
            return None

        self._agent_funcs["Perplexity"] = query_perplexity

    async def _register_deepseek_api(self):
        """Register DeepSeek - local Ollama first, API fallback."""
        async def query_deepseek_api(prompt: str, max_tokens: int) -> Optional[Tuple[str, str]]:
            # Try local Ollama first (free, runs on GPU)
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        "http://localhost:11434/api/chat",
                        json={
                            "model": "deepseek-r1:8b",
                            "messages": [{"role": "user", "content": prompt}],
                            "stream": False,
                            "options": {"num_predict": min(max_tokens, 1500), "temperature": 0.7},
                        },
                        timeout=45.0,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        content = data.get("message", {}).get("content", "")
                        if content:
                            import re
                            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                            if content:
                                return ("DeepSeekLocal", content)
            except Exception as e:
                logger.debug(f"DeepSeek local (Ollama) failed: {e}")

            # Fallback to DeepSeek Cloud API
            try:
                import httpx
                api_key = os.environ.get("DEEPSEEK_API_KEY")
                if not api_key:
                    return None
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        "https://api.deepseek.com/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json={
                            "model": "deepseek-chat",
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": max_tokens,
                        },
                        timeout=45.0,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("choices") and data["choices"][0].get("message", {}).get("content"):
                            return ("DeepSeekAPI", data["choices"][0]["message"]["content"].strip())
            except Exception as e:
                logger.debug(f"DeepSeek API query failed: {e}")
            return None

        self._agent_funcs["DeepSeekAPI"] = query_deepseek_api

    async def _register_hermes(self):
        """Register Hermes (Nous Research Direct) agent."""
        async def query_hermes(prompt: str, max_tokens: int) -> Optional[Tuple[str, str]]:
            try:
                import httpx
                api_key = os.environ.get("HERMES_API_KEY")
                base_url = os.environ.get("HERMES_BASE_URL", "https://inference-api.nousresearch.com/v1")
                model_id = os.environ.get("HERMES_MODEL_ID", "Hermes-4-70B")
                
                if not api_key:
                    # Fallback to OpenRouter if HERMES_API_KEY is not set
                    api_key = os.environ.get("OPENROUTER_API_KEY")
                    if not api_key:
                        return None
                    base_url = "https://openrouter.ai/api/v1"
                    if "model_id" not in locals() or model_id == "Hermes-4-70B":
                        model_id = "nousresearch/hermes-3-llama-3.1-405b"
                    
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        f"{base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                            "X-Title": "CosmoSynapse Swarm"
                        },
                        json={
                            "model": model_id,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": max_tokens,
                            "temperature": 0.8
                        },
                        timeout=60.0
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("choices") and data["choices"][0].get("message", {}).get("content"):
                            return ("Hermes", data["choices"][0]["message"]["content"].strip())
            except Exception as e:
                logger.debug(f"Hermes query failed: {e}")
            return None

        self._agent_funcs["Hermes"] = query_hermes

    def get_available_agents(self) -> list:
        """Get list of registered agent IDs."""
        return list(self._agent_funcs.keys())


# Global registry instance
_agent_registry: Optional[AgentRegistry] = None


def get_agent_registry() -> AgentRegistry:
    """Get or create the global agent registry."""
    global _agent_registry
    if _agent_registry is None:
        _agent_registry = AgentRegistry()
    return _agent_registry


async def ensure_agents_registered():
    """Ensure all agents are registered with deliberation room."""
    registry = get_agent_registry()
    await registry.initialize()
    return registry.get_available_agents()
