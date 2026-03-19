"""
CLI Capability Router — Smart routing + fallback chains across CLI bridges.

"Route to the right brain. Free first, smart always." - The Collective

Features:
- Capability-based routing (web search → Gemini, code edit → Claude Code)
- Task signal detection from prompt keywords
- Fallback chains with auto-cascade on failure
- Cost optimization (free CLIs prioritized)
- Health-aware routing (skip unhealthy bridges)
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Set

from loguru import logger

from .base import CLIBridge, CLICapability, CLIResponse
from .rate_tracker import get_rate_tracker


# =============================================================================
# ROUTING DECISION
# =============================================================================

@dataclass
class RoutingDecision:
    """Result of routing logic."""
    primary: CLIBridge
    fallbacks: List[CLIBridge] = field(default_factory=list)
    reason: str = ""
    required_capabilities: Set[CLICapability] = field(default_factory=set)


# =============================================================================
# TASK SIGNAL DETECTION
# =============================================================================

# Keywords that suggest specific capabilities
_CAPABILITY_SIGNALS = {
    CLICapability.WEB_SEARCH: [
        "search", "latest", "current", "trending", "news",
        "today", "recent", "what is happening", "look up",
        "find out", "google", "web",
    ],
    CLICapability.CODE_EDIT: [
        "refactor", "fix", "edit", "modify", "change", "update code",
        "implement", "add function", "rewrite", "debug", "patch",
    ],
    CLICapability.LONG_CONTEXT: [
        "entire codebase", "all files", "whole project", "summarize all",
        "full document", "analyze everything", "complete",
    ],
    CLICapability.CODE_EXECUTION: [
        "run", "execute", "test", "benchmark", "profile",
    ],
    CLICapability.FILE_READ: [
        "read file", "show me", "what does", "look at",
    ],
}


def detect_required_capabilities(prompt: str) -> Set[CLICapability]:
    """Scan prompt for keywords that suggest required capabilities."""
    lower = prompt.lower()
    detected = set()

    for capability, keywords in _CAPABILITY_SIGNALS.items():
        if any(kw in lower for kw in keywords):
            detected.add(capability)

    return detected


# =============================================================================
# CAPABILITY ROUTER
# =============================================================================

class CLICapabilityRouter:
    """
    Smart router that selects the best CLI bridge for each query.

    Maintains a registry of available bridges, scores them by:
    - Capability match
    - Health status
    - Cost (free preferred)
    - Rate limit headroom
    And builds fallback chains from remaining CLIs.
    """

    def __init__(self):
        self._bridges: Dict[str, CLIBridge] = {}
        self._cost_order: List[str] = []  # Cheapest first
        self._initialized = False

    async def initialize(self):
        """
        Discover and health-check all available CLI bridges.

        Called once on startup. Bridges that fail health check
        are still registered but marked unavailable.
        """
        if self._initialized:
            return

        bridges_to_try = []

        # Try Claude Code
        try:
            from .claude_code_bridge import ClaudeCodeBridge
            bridges_to_try.append(ClaudeCodeBridge())
        except Exception as e:
            logger.debug(f"Claude Code bridge init failed: {e}")

        # Try Gemini CLI
        try:
            from .gemini_cli_bridge import GeminiCLIBridge
            bridges_to_try.append(GeminiCLIBridge())
        except Exception as e:
            logger.debug(f"Gemini CLI bridge init failed: {e}")

        # Health check each bridge
        for bridge in bridges_to_try:
            try:
                health = await bridge.check_health()
                self._bridges[bridge.cli_name] = bridge
                status = "available" if health.available else "unavailable"
                logger.info(f"CLI Bridge registered: {bridge.cli_name} ({status})")
            except Exception as e:
                logger.warning(f"CLI Bridge {bridge.cli_name} health check failed: {e}")
                # Still register — it may come back
                self._bridges[bridge.cli_name] = bridge

        # Cost ordering: Gemini (free) → Claude (subscription)
        self._cost_order = ["gemini_cli", "claude_code"]

        self._initialized = True
        logger.info(f"CLI Router initialized with {len(self._bridges)} bridges")

    def register_bridge(self, bridge: CLIBridge):
        """Register an additional CLI bridge."""
        self._bridges[bridge.cli_name] = bridge
        if bridge.cli_name not in self._cost_order:
            self._cost_order.append(bridge.cli_name)

    def route(
        self,
        prompt: str,
        required_capabilities: Optional[Set[CLICapability]] = None,
        prefer_free: bool = True,
        preferred_cli: Optional[str] = None,
    ) -> Optional[RoutingDecision]:
        """
        Decide which CLI bridge to use for a query.

        Args:
            prompt: The user's prompt (used for task signal detection)
            required_capabilities: Explicit capability requirements
            prefer_free: Prefer free-tier CLIs (default True)
            preferred_cli: Force a specific CLI if available

        Returns:
            RoutingDecision with primary bridge and fallback chain,
            or None if no bridge is available.
        """
        if not self._bridges:
            return None

        # Detect capabilities from prompt if not specified
        if required_capabilities is None:
            required_capabilities = detect_required_capabilities(prompt)

        tracker = get_rate_tracker()

        # If preferred CLI is specified and available, use it
        if preferred_cli and preferred_cli in self._bridges:
            bridge = self._bridges[preferred_cli]
            if bridge.health.available and tracker.can_request(preferred_cli):
                fallbacks = [
                    b for name, b in self._bridges.items()
                    if name != preferred_cli and b.health.available
                ]
                return RoutingDecision(
                    primary=bridge,
                    fallbacks=fallbacks,
                    reason=f"Preferred CLI: {preferred_cli}",
                    required_capabilities=required_capabilities,
                )

        # Score each bridge
        scored: List[tuple] = []  # (score, bridge)

        for name, bridge in self._bridges.items():
            if not bridge.health.available:
                continue
            if not tracker.can_request(name):
                continue

            score = 0.0
            bridge_caps = bridge.get_capabilities()

            # Capability match score (most important)
            if required_capabilities:
                matching = required_capabilities & bridge_caps
                if not matching:
                    continue  # Skip if no capability overlap
                score += len(matching) * 10
            else:
                score += len(bridge_caps)  # More capable = better default

            # Health score
            if bridge.health.consecutive_failures == 0:
                score += 5
            else:
                score -= bridge.health.consecutive_failures * 2

            # Cost score (free preferred)
            if prefer_free and name in self._cost_order:
                cost_rank = self._cost_order.index(name)
                score += max(0, 10 - cost_rank * 3)

            # Rate limit headroom
            stats = tracker.get_stats(name)
            if stats.get("daily_remaining") is not None:
                remaining_pct = stats["daily_remaining"] / max(stats.get("daily_limit", 1), 1)
                score += remaining_pct * 5

            scored.append((score, bridge))

        if not scored:
            return None

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        primary = scored[0][1]
        fallbacks = [b for _, b in scored[1:]]

        # Build reason
        reason_parts = []
        if required_capabilities:
            reason_parts.append(f"caps={[c.name for c in required_capabilities]}")
        reason_parts.append(f"score={scored[0][0]:.1f}")
        reason = f"Routed to {primary.cli_name}: {', '.join(reason_parts)}"

        return RoutingDecision(
            primary=primary,
            fallbacks=fallbacks,
            reason=reason,
            required_capabilities=required_capabilities,
        )

    async def query_with_fallback(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        session_id: Optional[str] = None,
        required_capabilities: Optional[Set[CLICapability]] = None,
        preferred_cli: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> CLIResponse:
        """
        Query with automatic fallback to next available CLI on failure.

        This is the primary entry point for the swarm to use CLI bridges.
        """
        if not self._initialized:
            await self.initialize()

        decision = self.route(
            prompt=prompt,
            required_capabilities=required_capabilities,
            preferred_cli=preferred_cli,
        )

        if not decision:
            return CLIResponse(
                error="No CLI bridges available",
                cli_name="router",
            )

        tracker = get_rate_tracker()

        # Try primary
        all_bridges = [decision.primary] + decision.fallbacks

        for bridge in all_bridges:
            if not tracker.can_request(bridge.cli_name):
                logger.debug(f"[router] Skipping {bridge.cli_name}: rate limited")
                continue

            logger.info(f"[router] Trying {bridge.cli_name}...")
            tracker.record_request(bridge.cli_name)

            response = await bridge.query(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                max_tokens=max_tokens,
                session_id=session_id,
                timeout=timeout,
            )

            if response.success:
                logger.info(
                    f"[router] {bridge.cli_name} succeeded "
                    f"({response.latency_ms:.0f}ms)"
                )
                return response

            # Record failure for rate tracking
            if "rate limit" in response.error.lower():
                tracker.record_rate_limit(bridge.cli_name)

            logger.warning(
                f"[router] {bridge.cli_name} failed: {response.error[:100]}, "
                f"trying next..."
            )

        return CLIResponse(
            error="All CLI bridges failed",
            cli_name="router",
        )

    async def query_streaming_with_fallback(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        preferred_cli: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream with fallback — tries primary, falls back on error."""
        if not self._initialized:
            await self.initialize()

        decision = self.route(prompt=prompt, preferred_cli=preferred_cli)
        if not decision:
            yield "[Error: No CLI bridges available]"
            return

        tracker = get_rate_tracker()
        all_bridges = [decision.primary] + decision.fallbacks

        for bridge in all_bridges:
            if not tracker.can_request(bridge.cli_name):
                continue

            tracker.record_request(bridge.cli_name)
            got_content = False

            try:
                async for delta in bridge.query_streaming(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=model,
                    max_tokens=max_tokens,
                ):
                    got_content = True
                    yield delta

                if got_content:
                    return  # Success
            except Exception as e:
                logger.warning(f"[router] Stream from {bridge.cli_name} failed: {e}")
                continue

        if not got_content:
            yield "[Error: All CLI bridges failed to stream]"

    def get_available_bridges(self) -> List[Dict]:
        """Get status of all registered bridges."""
        return [bridge.get_status() for bridge in self._bridges.values()]

    def get_bridge(self, cli_name: str) -> Optional[CLIBridge]:
        """Get a specific bridge by name."""
        return self._bridges.get(cli_name)


# =============================================================================
# SINGLETON
# =============================================================================

_router_instance: Optional[CLICapabilityRouter] = None


async def get_cli_router() -> CLICapabilityRouter:
    """Get or create the singleton CLI capability router."""
    global _router_instance
    if _router_instance is None:
        _router_instance = CLICapabilityRouter()
        await _router_instance.initialize()
    return _router_instance
