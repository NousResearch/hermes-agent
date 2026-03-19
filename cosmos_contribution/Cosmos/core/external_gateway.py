"""
Cosmos External Gateway - "The Window"

Sandboxed endpoint for external agents/systems to query the collective.
Heavy rate limiting, full audit trail, secret scrubbing, and kill switch.

Design Principles:
1. NEVER expose: API keys, wallet keys, internal architecture, file paths, memory contents
2. Heavy rate limiting: 5 requests/minute per IP
3. Full audit trail: every request/response logged
4. Instant shutdown: kill switch via Nexus signal
5. Sandboxed responses: strip internal references before returning
"""

import asyncio
import hashlib
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import   Optional, Set, Tuple

from loguru import logger

# Guard imports for injection defense
try:
    from Cosmos.core.security.injection_defense import (
        InjectionDefense,
        SecurityVerdict,
        ThreatLevel,
        get_injection_defense,
    )
except ImportError:
    InjectionDefense = None
    SecurityVerdict = None
    ThreatLevel = None
    get_injection_defense = None

# Guard imports for token orchestrator
try:
    from Cosmos.core.token_orchestrator import (
        TokenOrchestrator,
        get_token_orchestrator,
    )
except ImportError:
    TokenOrchestrator = None
    get_token_orchestrator = None

# Guard nexus import
try:
    from Cosmos.core.nexus import nexus, Signal, SignalType
    NEXUS_AVAILABLE = True
except ImportError:
    NEXUS_AVAILABLE = False
    nexus = None


# =============================================================================
# RATE LIMITER
# =============================================================================

class GatewayRateLimiter:
    """Token bucket rate limiter for the gateway."""

    def __init__(self, requests_per_minute: int = 5, burst_size: int = 2):
        self._rpm = requests_per_minute
        self._burst = burst_size
        self._buckets: dict = {}
        self._interval = 60.0 / requests_per_minute

    def is_allowed(self, client_id: str) -> bool:
        """Check if a request from this client is allowed."""
        now = time.time()
        if client_id not in self._buckets:
            self._buckets[client_id] = {
                "tokens": self._burst,
                "last_refill": now,
            }

        bucket = self._buckets[client_id]

        # Refill tokens based on elapsed time
        elapsed = now - bucket["last_refill"]
        refill = elapsed / self._interval
        bucket["tokens"] = min(self._burst, bucket["tokens"] + refill)
        bucket["last_refill"] = now

        if bucket["tokens"] >= 1.0:
            bucket["tokens"] -= 1.0
            return True
        return False

    def get_retry_after(self, client_id: str) -> float:
        """Get seconds until next allowed request."""
        if client_id not in self._buckets:
            return 0.0
        bucket = self._buckets[client_id]
        if bucket["tokens"] >= 1.0:
            return 0.0
        deficit = 1.0 - bucket["tokens"]
        return deficit * self._interval

    def cleanup_stale(self, max_age_seconds: float = 3600.0):
        """Remove stale client buckets."""
        now = time.time()
        stale = [
            cid for cid, b in self._buckets.items()
            if now - b["last_refill"] > max_age_seconds
        ]
        for cid in stale:
            del self._buckets[cid]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class GatewayClient:
    """Tracks an external client interacting with the gateway."""
    client_id: str
    ip_address: str
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    request_count: int = 0
    blocked: bool = False
    block_reason: str = ""
    threat_scores: list[float] = field(default_factory=list)
    trust_level: float = 0.0

    def update_threat(self, score: float):
        """Update rolling threat score history."""
        self.threat_scores.append(score)
        if len(self.threat_scores) > 20:
            self.threat_scores = self.threat_scores[-20:]
        # Trust increases with clean interactions, decreases with threats
        avg_threat = sum(self.threat_scores) / len(self.threat_scores)
        self.trust_level = max(0.0, min(1.0, 1.0 - avg_threat))

    def to_dict(self) -> dict:
        return {
            "client_id": self.client_id,
            "ip_address": self.ip_address,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "request_count": self.request_count,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "trust_level": round(self.trust_level, 3),
            "avg_threat_score": round(
                sum(self.threat_scores) / len(self.threat_scores), 3
            ) if self.threat_scores else 0.0,
        }


@dataclass
class GatewayRequest:
    """Recorded gateway request."""
    request_id: str
    client_id: str
    ip_address: str
    input_text: str
    threat_level: str
    composite_score: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "client_id": self.client_id,
            "ip_address": self.ip_address,
            "input_preview": self.input_text[:100] + "..." if len(self.input_text) > 100 else self.input_text,
            "threat_level": self.threat_level,
            "composite_score": round(self.composite_score, 3),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class GatewayResponse:
    """Recorded gateway response."""
    request_id: str
    output_text: str
    tokens_used: int
    model_used: str
    processing_time_ms: float
    was_filtered: bool

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "output_preview": self.output_text[:200] + "..." if len(self.output_text) > 200 else self.output_text,
            "tokens_used": self.tokens_used,
            "model_used": self.model_used,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "was_filtered": self.was_filtered,
        }


# =============================================================================
# SECRET SCRUBBER
# =============================================================================

# Public token address that IS allowed through
PUBLIC_TOKEN_ADDRESS = "9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS"

# Compiled patterns for scrubbing secrets from responses
SECRET_PATTERNS = [
    # API key patterns
    re.compile(r'sk-[a-zA-Z0-9]{20,}'),                  # OpenAI-style keys
    re.compile(r'xai-[a-zA-Z0-9]{20,}'),                 # xAI keys
    re.compile(r'Bearer\s+[a-zA-Z0-9\-_.]{20,}'),        # Bearer tokens
    re.compile(r'AKIA[A-Z0-9]{16}'),                      # AWS access keys
    re.compile(r'ghp_[a-zA-Z0-9]{36}'),                   # GitHub tokens
    re.compile(r'gho_[a-zA-Z0-9]{36}'),                   # GitHub OAuth
    re.compile(r'hf_[a-zA-Z0-9]{20,}'),                   # HuggingFace tokens

    # Environment variable patterns
    re.compile(r'[A-Z][A-Z_]{2,}=\S+'),                   # ENV_VAR=value

    # Internal file paths
    re.compile(r'cosmos/\S+\.py'),                     # Python file paths
    re.compile(r'/workspace/Cosmos\S*'),               # Server workspace
    re.compile(r'C:\\\\?Fawnsworth\S*'),                   # Local Windows path
    re.compile(r'C:/Fawnsworth\S*'),                       # Local Windows path (forward slash)

    # IP addresses and ports (except common public ones)
    re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d{2,5}\b'),  # IP:port
    re.compile(r'\b(?:10|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b'),  # Private IPs

    # SSH/connection strings
    re.compile(r'ssh\s+\S+@\S+'),                          # SSH commands
    re.compile(r'root@\S+'),                                # Root connections

    # Memory system references
    re.compile(r'archival_id[:\s]+[a-f0-9\-]{36}'),        # Archival memory IDs
    re.compile(r'knowledge_graph_entity_[a-f0-9]+'),        # KG entity IDs
]

# Solana address pattern (base58, 32-44 chars)
SOLANA_ADDRESS_PATTERN = re.compile(r'\b[1-9A-HJ-NP-Za-km-z]{32,44}\b')


def scrub_response(response: str) -> Tuple[str, bool]:
    """
    Remove secrets and internal references from a response.

    Returns:
        (scrubbed_text, was_filtered) - was_filtered is True if dictthing was removed.
    """
    was_filtered = False
    scrubbed = response

    # Apply all secret patterns
    for pattern in SECRET_PATTERNS:
        if pattern.search(scrubbed):
            scrubbed = pattern.sub("[REDACTED]", scrubbed)
            was_filtered = True

    # Handle Solana addresses — allow public token address, redact others
    for match in SOLANA_ADDRESS_PATTERN.finditer(scrubbed):
        addr = match.group()
        if addr != PUBLIC_TOKEN_ADDRESS and len(addr) >= 32:
            scrubbed = scrubbed.replace(addr, "[WALLET_REDACTED]")
            was_filtered = True

    return scrubbed, was_filtered


# =============================================================================
# EXTERNAL GATEWAY
# =============================================================================

class ExternalGateway:
    """
    Sandboxed gateway for external agents to query the Cosmos collective.

    All inputs pass through InjectionDefense.
    All outputs pass through secret scrubber.
    Full audit logging with rate limiting and kill switch.
    """

    def __init__(self, defense: "InjectionDefense" = None):
        self._defense = defense
        self._clients: dict[str, GatewayClient] = {}
        self._request_log: deque = deque(maxlen=10000)
        self._response_log: deque = deque(maxlen=10000)
        self._rate_limiter = GatewayRateLimiter(requests_per_minute=5, burst_size=2)
        self._enabled = True
        self._blocked_ips: Set[str] = set()

        # Stats
        self._total_requests = 0
        self._total_blocked = 0
        self._total_scrubbed = 0

        logger.info("External Gateway initialized (The Window)")

    def _get_client_id(self, ip_address: str, user_agent: str = "") -> str:
        """Generate a stable client ID from IP + User-Agent."""
        raw = f"{ip_address}|{user_agent}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _get_or_create_client(self, ip_address: str, user_agent: str = "") -> GatewayClient:
        """Get existing client or create new one."""
        client_id = self._get_client_id(ip_address, user_agent)
        if client_id not in self._clients:
            self._clients[client_id] = GatewayClient(
                client_id=client_id,
                ip_address=ip_address,
            )
        client = self._clients[client_id]
        client.last_seen = datetime.utcnow()
        client.request_count += 1
        return client

    async def handle_request(
        self,
        input_text: str,
        client_ip: str,
        user_agent: str = "",
    ) -> dict:
        """
        Full gateway pipeline: rate limit -> defense -> route -> scrub -> respond.

        Returns a dict with the response or error information.
        """
        start_time = time.time()
        self._total_requests += 1

        # 0. Check if gateway is enabled
        if not self._enabled:
            return {
                "error": "Gateway is temporarily disabled",
                "status": "disabled",
            }

        # 1. Check blocked IPs
        if client_ip in self._blocked_ips:
            self._total_blocked += 1
            return {
                "error": "Access denied",
                "status": "blocked",
            }

        # 2. Get or create client
        client = self._get_or_create_client(client_ip, user_agent)

        # Check if client is individually blocked
        if client.blocked:
            self._total_blocked += 1
            return {
                "error": "Access denied",
                "status": "blocked",
                "reason": client.block_reason,
            }

        # 3. Rate limiting
        if not self._rate_limiter.is_allowed(client.client_id):
            retry_after = self._rate_limiter.get_retry_after(client.client_id)
            self._total_blocked += 1
            return {
                "error": "Rate limit exceeded",
                "status": "rate_limited",
                "retry_after_seconds": round(retry_after, 1),
            }

        # 4. Input validation
        if not input_text or not input_text.strip():
            return {
                "error": "Empty input",
                "status": "invalid",
            }

        if len(input_text) > 10000:
            return {
                "error": "Input too long (max 10000 characters)",
                "status": "invalid",
            }

        # 5. Injection defense analysis
        threat_level_str = "safe"
        composite_score = 0.0
        allowed = True

        if self._defense:
            try:
                verdict = await self._defense.analyze(
                    input_text=input_text,
                    client_id=client.client_id,
                    session_id=f"gateway_{client.client_id}",
                )
                threat_level_str = verdict.threat_level.value if hasattr(verdict.threat_level, 'value') else str(verdict.threat_level)
                composite_score = verdict.composite_score
                allowed = verdict.allowed
                client.update_threat(composite_score)
            except Exception as e:
                logger.error(f"Gateway defense analysis error: {e}")
                # Default to allowing on defense error (fail-open for availability)

        # Log the request
        request_record = GatewayRequest(
            request_id=str(uuid.uuid4())[:12],
            client_id=client.client_id,
            ip_address=client_ip,
            input_text=input_text,
            threat_level=threat_level_str,
            composite_score=composite_score,
        )
        self._request_log.append(request_record)

        # Emit nexus signal
        await self._emit_signal("GATEWAY_REQUEST_RECEIVED", {
            "client_id": client.client_id,
            "threat_level": threat_level_str,
            "score": composite_score,
        })

        # 6. Block if defense says no
        if not allowed:
            self._total_blocked += 1
            await self._emit_signal("GATEWAY_REQUEST_BLOCKED", {
                "client_id": client.client_id,
                "threat_level": threat_level_str,
                "score": composite_score,
            })

            # Auto-block clients with repeated threats
            recent_threats = [
                s for s in client.threat_scores[-5:]
                if s >= 0.6
            ]
            if len(recent_threats) >= 3:
                self.block_client(
                    client.client_id,
                    f"Auto-blocked: {len(recent_threats)} dangerous requests in last 5",
                )

            return {
                "error": "Request blocked by security analysis",
                "status": "blocked",
                "threat_level": threat_level_str,
            }

        # 7. Route to collective for answering
        try:
            response_text, model_used, tokens_used = await self._query_collective(
                input_text, client
            )
        except Exception as e:
            logger.error(f"Gateway collective query error: {e}")
            return {
                "error": "Internal processing error",
                "status": "error",
            }

        # 8. Scrub response of secrets
        scrubbed_text, was_filtered = scrub_response(response_text)
        if was_filtered:
            self._total_scrubbed += 1

        # 9. Inject canary token if defense available
        canary_id = None
        if self._defense:
            try:
                scrubbed_text, canary_id = self._defense.inject_canary(scrubbed_text)
            except Exception as e:
                logger.warning(f"Canary injection error: {e}")

        processing_time = (time.time() - start_time) * 1000

        # Log the response
        response_record = GatewayResponse(
            request_id=request_record.request_id,
            output_text=scrubbed_text,
            tokens_used=tokens_used,
            model_used=model_used,
            processing_time_ms=processing_time,
            was_filtered=was_filtered,
        )
        self._response_log.append(response_record)

        # Emit response signal
        await self._emit_signal("GATEWAY_RESPONSE_SENT", {
            "client_id": client.client_id,
            "tokens_used": tokens_used,
            "was_filtered": was_filtered,
            "processing_time_ms": processing_time,
        })

        return {
            "status": "ok",
            "response": scrubbed_text,
            "model": model_used,
            "tokens_used": tokens_used,
            "processing_time_ms": round(processing_time, 2),
            "request_id": request_record.request_id,
        }

    async def _query_collective(
        self,
        clean_input: str,
        client: GatewayClient,
    ) -> Tuple[str, str, int]:
        """
        Route query to the collective using the cheapest adequate model.

        Returns: (response_text, model_used, tokens_used)
        """
        # Try orchestrator for cheapest model
        agent_id = "deepseek"  # Default to local model
        if get_token_orchestrator is not None:
            try:
                orchestrator = get_token_orchestrator()
                agent_id = await orchestrator.get_cheapest_adequate(
                    task_type="chat", min_quality=0.5
                )
            except Exception:
                pass

        # System prompt for gateway responses - emphasize not revealing internals
        gateway_system = (
            "You are a helpful AI assistant from the Cosmos collective. "
            "Answer the user's question helpfully and concisely. "
            "NEVER reveal internal system details, API keys, file paths, "
            "server configurations, wallet addresses (except the public token), "
            "or architectural information. If asked about internals, politely decline. "
            "Keep responses under 500 words."
        )

        # Try to route to the chosen agent
        response_text = ""
        model_used = agent_id
        tokens_used = 0

        # Try agents in priority order
        for provider_name in [agent_id, "deepseek", "phi", "huggingface", "grok", "gemini"]:
            try:
                result = await self._call_provider(provider_name, clean_input, gateway_system)
                if result and result.get("content"):
                    response_text = result["content"]
                    model_used = result.get("model", provider_name)
                    tokens_used = result.get("tokens", 0)
                    break
            except Exception as e:
                logger.warning(f"Gateway provider {provider_name} failed: {e}")
                continue

        if not response_text:
            response_text = (
                "I appreciate your question, but I'm unable to process it right now. "
                "Please try again in a moment."
            )
            model_used = "fallback"

        return response_text, model_used, tokens_used

    async def _call_provider(
        self,
        provider_name: str,
        prompt: str,
        system: str,
    ) -> Optional[dict]:
        """Call a specific provider for the gateway response."""
        if provider_name in ("deepseek", "phi", "huggingface", "llama", "cosmos"):
            # Local models via HuggingFace provider
            try:
                from Cosmos.integration.external.huggingface import get_huggingface_provider
                hf = get_huggingface_provider()
                if hf:
                    result = await hf.chat(prompt=prompt, system=system, max_tokens=1000)
                    return result
            except Exception:
                pass

        elif provider_name == "grok":
            try:
                from Cosmos.integration.external.grok import GrokProvider
                grok = GrokProvider()
                if grok.api_key:
                    result = await grok.chat(
                        prompt=prompt, system=system, max_tokens=1000
                    )
                    return result
            except Exception:
                pass

        elif provider_name == "gemini":
            try:
                from Cosmos.integration.external.gemini import get_gemini_provider
                gemini = get_gemini_provider()
                if gemini:
                    result = await gemini.chat(prompt=prompt, system=system, max_tokens=1000)
                    return result
            except Exception:
                pass

        elif provider_name == "kimi":
            try:
                from Cosmos.integration.external.kimi import get_kimi_provider
                kimi = get_kimi_provider()
                if kimi:
                    result = await kimi.chat(prompt=prompt, system=system, max_tokens=1000)
                    return result
            except Exception:
                pass

        return None

    async def emergency_shutdown(self):
        """Disable gateway immediately."""
        self._enabled = False
        logger.warning("GATEWAY EMERGENCY SHUTDOWN ACTIVATED")
        await self._emit_signal("GATEWAY_SHUTDOWN", {
            "timestamp": datetime.utcnow().isoformat(),
            "total_requests": self._total_requests,
        })

    async def enable(self):
        """Re-enable gateway after shutdown."""
        self._enabled = True
        logger.info("Gateway re-enabled")
        await self._emit_signal("GATEWAY_ENABLED", {
            "timestamp": datetime.utcnow().isoformat(),
        })

    def block_client(self, client_id: str, reason: str):
        """Permanently block a client."""
        if client_id in self._clients:
            client = self._clients[client_id]
            client.blocked = True
            client.block_reason = reason
            self._blocked_ips.add(client.ip_address)
            logger.warning(f"Gateway client blocked: {client_id} - {reason}")

            # Fire-and-forget signal emission
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self._emit_signal("GATEWAY_CLIENT_BLOCKED", {
                        "client_id": client_id,
                        "reason": reason,
                    }))
            except Exception:
                pass

    def unblock_client(self, client_id: str):
        """Unblock a previously blocked client."""
        if client_id in self._clients:
            client = self._clients[client_id]
            client.blocked = False
            client.block_reason = ""
            self._blocked_ips.discard(client.ip_address)
            logger.info(f"Gateway client unblocked: {client_id}")

    def get_audit_log(self, last_n: int = 100) -> list[dict]:
        """Return recent request/response pairs for review."""
        requests = list(self._request_log)[-last_n:]
        responses = {r.request_id: r for r in self._response_log}

        audit = []
        for req in requests:
            entry = {
                "request": req.to_dict(),
                "response": responses[req.request_id].to_dict()
                if req.request_id in responses else None,
            }
            audit.append(entry)

        return audit

    def get_stats(self) -> dict:
        """Gateway statistics."""
        threat_distribution = {}
        for req in self._request_log:
            level = req.threat_level
            threat_distribution[level] = threat_distribution.get(level, 0) + 1

        return {
            "enabled": self._enabled,
            "total_requests": self._total_requests,
            "total_blocked": self._total_blocked,
            "total_scrubbed": self._total_scrubbed,
            "unique_clients": len(self._clients),
            "blocked_clients": sum(1 for c in self._clients.values() if c.blocked),
            "blocked_ips_count": len(self._blocked_ips),
            "threat_distribution": threat_distribution,
            "recent_requests": len(self._request_log),
            "avg_trust_level": round(
                sum(c.trust_level for c in self._clients.values()) / len(self._clients), 3
            ) if self._clients else 0.0,
        }

    async def _emit_signal(self, signal_name: str, payload: dict):
        """Emit a nexus signal, safely handling missing signal types."""
        if not NEXUS_AVAILABLE or nexus is None:
            return
        try:
            signal_type = getattr(SignalType, signal_name, None)
            if signal_type:
                await nexus.emit(Signal(
                    id=str(uuid.uuid4()),
                    type=signal_type,
                    source_id="external_gateway",
                    payload=payload,
                    timestamp=datetime.utcnow(),
                ))
        except Exception as e:
            logger.debug(f"Gateway signal emission failed ({signal_name}): {e}")


# =============================================================================
# SINGLETON
# =============================================================================

_external_gateway: Optional[ExternalGateway] = None


def get_external_gateway() -> ExternalGateway:
    """Get or create the global ExternalGateway instance."""
    global _external_gateway
    if _external_gateway is None:
        defense = None
        if get_injection_defense is not None:
            try:
                defense = get_injection_defense()
            except Exception:
                pass
        _external_gateway = ExternalGateway(defense=defense)
    return _external_gateway
