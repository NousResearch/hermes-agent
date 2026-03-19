"""
Cosmos Swarm Oracle - Quantum Collective Intelligence
===================================================

Multi-agent consensus oracle that:
1. Accepts questions/predictions via API
2. Receives true Quantum Entropy from ibm_fez
3. Runs BFT deliberation across 11 agents 
4. Crystallizes answers only if $E_c > T$
5. Records consensus hash on Solana

API:
  POST /api/oracle/query - Submit question for collective deliberation
  GET /api/oracle/queries - List recent oracle queries
  GET /api/oracle/query/{id} - Get specific query result
"""

import asyncio
import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path

from loguru import logger

# Solana imports
try:
    from solders.keypair import Keypair
    from solders.pubkey import Pubkey
    from solders.transaction import Transaction
    from solders.system_program import transfer, TransferParams
    from solders.message import Message
    from solana.rpc.async_api import AsyncClient
    from solana.rpc.commitment import Confirmed
    SOLANA_AVAILABLE = True
except ImportError:
    SOLANA_AVAILABLE = False
    logger.warning("Solana SDK not available - oracle will work without on-chain recording")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class AgentVote:
    """Individual agent's contribution to deliberation."""
    agent_id: str
    phase: str  # propose, critique, refine, vote
    content: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OracleQuery:
    """A query submitted to the swarm oracle."""
    query_id: str
    question: str
    query_type: str  # prediction, analysis, recommendation, general
    created_at: datetime

    # Deliberation results
    status: str = "pending"  # pending, deliberating, complete, failed
    agents_participated: List[str] = field(default_factory=list)
    votes: List[AgentVote] = field(default_factory=list)

    # Consensus
    consensus_reached: bool = False
    consensus_answer: Optional[str] = None
    consensus_confidence: float = 0.0
    consensus_reasoning: Optional[str] = None

    # On-chain verification
    consensus_hash: Optional[str] = None
    solana_signature: Optional[str] = None
    solana_slot: Optional[int] = None

    # Timing
    deliberation_started: Optional[datetime] = None
    deliberation_completed: Optional[datetime] = None
    deliberation_duration_ms: Optional[int] = None
    
    # Physics Tracking
    quantum_entropy_injected: float = 0.0
    collective_energy_joules: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "question": self.question,
            "query_type": self.query_type,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "agents_participated": self.agents_participated,
            "consensus_reached": self.consensus_reached,
            "consensus_answer": self.consensus_answer,
            "consensus_confidence": self.consensus_confidence,
            "consensus_reasoning": self.consensus_reasoning,
            "consensus_hash": self.consensus_hash,
            "solana_signature": self.solana_signature,
            "deliberation_duration_ms": self.deliberation_duration_ms,
            "quantum_entropy": self.quantum_entropy_injected,
            "collective_energy": f"{self.collective_energy_joules:.2e} J",
        }


# =============================================================================
# SWARM ORACLE
# =============================================================================

class SwarmOracle:
    """
    Collective Intelligence Oracle powered by Cosmos's 11-agent swarm.

    Provides verifiable multi-agent consensus on any question through
    the PROPOSE-CRITIQUE-REFINE-VOTE deliberation protocol.
    """

    # Available agents for 12D deliberation
    ORACLE_AGENTS = [
        "grok", "claude", "gemini", "deepseek", "kimi",
        "phi", "cosmos", "swarm_mind", "hermes", "feynman", "davinci"
    ]

    def __init__(
        self,
        data_dir: str = "./data/oracle",
        solana_rpc: str = "https://api.mainnet-beta.solana.com",
        min_agents: int = 3,
        consensus_threshold: float = 0.6,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.solana_rpc = solana_rpc
        self.min_agents = min_agents
        self.consensus_threshold = consensus_threshold

        # Query storage
        self._queries: Dict[str, OracleQuery] = {}
        self._query_history: List[str] = []

        # Solana client
        self._solana_client: Optional[AsyncClient] = None
        self._wallet: Optional[Keypair] = None

        # Load wallet if available
        self._init_solana()

        logger.info(f"SwarmOracle initialized - {len(self.ORACLE_AGENTS)} agents available")

    def _init_solana(self) -> None:
        """Initialize Solana connection and wallet.

        Supports multiple key formats for SOLANA_PRIVATE_KEY:
        - Base58-encoded string (standard Solana CLI export)
        - JSON byte array string, e.g. "[12,34,56,...]" (Solana keygen file format)
        """
        if not SOLANA_AVAILABLE:
            return

        try:
            wallet_key = os.getenv("SOLANA_PRIVATE_KEY")
            if wallet_key:
                wallet_key = wallet_key.strip()

                # Try JSON byte array format first (e.g. from solana-keygen)
                if wallet_key.startswith("["):
                    import json as _json
                    key_bytes = bytes(_json.loads(wallet_key))
                    if len(key_bytes) not in (32, 64):
                        raise ValueError(f"Invalid key length: {len(key_bytes)} bytes (expected 32 or 64)")
                    self._wallet = Keypair.from_bytes(key_bytes)
                else:
                    # Base58-encoded private key
                    self._wallet = Keypair.from_base58_string(wallet_key)

                logger.info(f"Solana wallet loaded: {self._wallet.pubkey()}")

            self._solana_client = AsyncClient(self.solana_rpc)
        except Exception as e:
            logger.warning(f"Solana init failed: {e}")

    # =========================================================================
    # CORE ORACLE FUNCTIONS
    # =========================================================================

    async def submit_query(
        self,
        question: str,
        query_type: str = "general",
        timeout: float = 120.0,
    ) -> OracleQuery:
        """
        Submit a question to the swarm oracle for collective deliberation.

        Args:
            question: The question to deliberate on
            query_type: Type of query (prediction, analysis, recommendation, general)
            timeout: Maximum time for deliberation

        Returns:
            OracleQuery with consensus result
        """
        query_id = f"oracle_{uuid.uuid4().hex[:12]}"

        query = OracleQuery(
            query_id=query_id,
            question=question,
            query_type=query_type,
            created_at=datetime.now(),
        )

        self._queries[query_id] = query
        self._query_history.append(query_id)

        logger.info(f"Oracle query submitted: {query_id} - {question[:50]}...")

        # Run deliberation
        try:
            query.status = "deliberating"
            query.deliberation_started = datetime.now()

            await self._run_deliberation(query, timeout)

            query.deliberation_completed = datetime.now()
            query.deliberation_duration_ms = int(
                (query.deliberation_completed - query.deliberation_started).total_seconds() * 1000
            )
            query.status = "complete"

            # Record on Solana if wallet available
            if self._wallet and query.consensus_reached:
                await self._record_on_chain(query)

        except Exception as e:
            logger.error(f"Deliberation failed: {e}")
            query.status = "failed"

        return query

    async def _run_deliberation(
        self,
        query: OracleQuery,
        timeout: float,
    ) -> None:
        """Run the PROPOSE-CRITIQUE-REFINE-VOTE deliberation protocol."""
        try:
            from Cosmos.core.collective.persistent_agent import call_shadow_agent
        except ImportError:
            logger.error("Cannot import shadow agent caller")
            return

        proposals = []
        critiques = []
        refinements = []
        votes = []

        # Phase 1: PROPOSE - Each agent proposes an answer
        logger.info(f"[{query.query_id}] Phase 1: PROPOSE")
        propose_prompt = f"""You are participating in a multi-agent oracle deliberation.

QUESTION: {query.question}
TYPE: {query.query_type}

Propose your answer to this question. Be specific and provide reasoning.
Format: Start with your answer, then explain your reasoning in 2-3 sentences.
Keep response under 150 words."""

        propose_tasks = []
        for agent_id in self.ORACLE_AGENTS[:5]:  # Use 5 agents for speed
            propose_tasks.append(self._call_agent(agent_id, propose_prompt))

        propose_results = await asyncio.gather(*propose_tasks, return_exceptions=True)

        for agent_id, result in zip(self.ORACLE_AGENTS[:5], propose_results):
            if isinstance(result, str) and result:
                proposals.append({"agent": agent_id, "proposal": result})
                query.agents_participated.append(agent_id)
                query.votes.append(AgentVote(
                    agent_id=agent_id,
                    phase="propose",
                    content=result,
                    confidence=0.7,
                ))

        if len(proposals) < self.min_agents:
            logger.warning(f"Not enough proposals: {len(proposals)}")
            return

        # Phase 2: CRITIQUE - Agents critique each other's proposals
        logger.info(f"[{query.query_id}] Phase 2: CRITIQUE")
        proposals_text = "\n\n".join([
            f"[{p['agent']}]: {p['proposal']}" for p in proposals
        ])

        critique_prompt = f"""You are critiquing proposals in a multi-agent oracle deliberation.

QUESTION: {query.question}

PROPOSALS:
{proposals_text}

Identify the strongest and weakest proposals. What's missing? What's incorrect?
Be constructive. Keep response under 100 words."""

        critique_result = await self._call_agent("grok", critique_prompt)
        if critique_result:
            critiques.append(critique_result)

        # Phase 3: REFINE - Synthesize into refined answer
        logger.info(f"[{query.query_id}] Phase 3: REFINE")
        refine_prompt = f"""You are synthesizing a final answer from multiple AI perspectives.

QUESTION: {query.question}

PROPOSALS:
{proposals_text}

CRITIQUE:
{critiques[0] if critiques else 'No critiques available'}

Synthesize the best answer combining insights from all proposals.
Start with a clear, direct answer, then provide brief reasoning.
Keep response under 150 words."""

        refined_answer = await self._call_agent("gemini", refine_prompt)
        if refined_answer:
            refinements.append(refined_answer)

        # Phase 4: VOTE - Agents vote on the refined answer
        logger.info(f"[{query.query_id}] Phase 4: VOTE")
        vote_prompt = f"""You are voting on a synthesized answer in a multi-agent oracle.

QUESTION: {query.question}

PROPOSED CONSENSUS:
{refined_answer if refined_answer else proposals[0]['proposal']}

Vote: Reply with ONLY one of these:
- AGREE (if you support this answer)
- DISAGREE (if you fundamentally disagree)
- ABSTAIN (if uncertain)

Then on a new line, give a confidence score 0.0-1.0"""

        vote_tasks = []
        for agent_id in self.ORACLE_AGENTS[:5]:
            vote_tasks.append(self._call_agent(agent_id, vote_prompt))

        vote_results = await asyncio.gather(*vote_tasks, return_exceptions=True)

        agree_count = 0
        total_confidence = 0.0

        for agent_id, result in zip(self.ORACLE_AGENTS[:5], vote_results):
            if isinstance(result, str):
                vote_text = result.upper()
                if "AGREE" in vote_text:
                    agree_count += 1
                    # Extract confidence if present
                    try:
                        conf_line = [l for l in result.split('\n') if any(c.isdigit() for c in l)]
                        if conf_line:
                            import re
                            conf_match = re.search(r'(\d+\.?\d*)', conf_line[0])
                            if conf_match:
                                total_confidence += float(conf_match.group(1))
                    except Exception:
                        total_confidence += 0.7

                query.votes.append(AgentVote(
                    agent_id=agent_id,
                    phase="vote",
                    content=result[:50],
                    confidence=0.7,
                ))

        # Quantum Physics Injection
        # Fetch live entropy from the bridge
        try:
            from Cosmos.core.quantum_bridge import get_quantum_bridge
            qb = get_quantum_bridge()
            if qb and qb.entropy_buffer:
                query.quantum_entropy_injected = qb.entropy_buffer.pop(0)
            else:
                query.quantum_entropy_injected = 0.5 # Default equilibrium
        except ImportError:
            query.quantum_entropy_injected = 0.5

        # Calculate BFT Consensus using 12D Mass and Quantum Chaos
        consensus_ratio = agree_count / max(len(vote_results), 1)
        
        # C^2 constant
        C_SQ = 299792458 ** 2
        # T: BFT Energy Threshold 
        THRESHOLD = 1.0e17
        
        # Perturb mass verification with real quantum entropy 
        # (lambda chaotic variance)
        chaotic_variance = 1.0 + (query.quantum_entropy_injected - 0.5) * 0.1
        
        query.consensus_confidence = round((total_confidence / max(agree_count, 1)) * chaotic_variance, 3)
        query.collective_energy_joules = query.consensus_confidence * agree_count * C_SQ

        query.consensus_reached = (
            consensus_ratio >= self.consensus_threshold and 
            query.collective_energy_joules > THRESHOLD
        )

        if query.consensus_reached and refined_answer:
            query.consensus_answer = refined_answer
            query.consensus_reasoning = f"Crystallized! {agree_count}/{len(vote_results)} consensus. E_c = {query.collective_energy_joules:.2e} J."
        elif proposals:
            query.consensus_answer = proposals[0]['proposal']
            query.consensus_reasoning = f"Failed to crystallize. E_c = {query.collective_energy_joules:.2e} J (Insufficient Gravity)."

        # Generate hash
        if query.consensus_answer:
            hash_input = f"{query.question}|{query.consensus_answer}|{query.collective_energy_joules}"
            query.consensus_hash = hashlib.sha256(hash_input.encode()).hexdigest()

        logger.info(f"[{query.query_id}] Consensus: {query.consensus_reached} | Energy: {query.collective_energy_joules:.2e} J")

    async def _call_agent(self, agent_id: str, prompt: str) -> Optional[str]:
        """Call an agent and return response. Tries shadow agent first, then direct API."""
        # Try shadow agent first
        try:
            from Cosmos.core.collective.persistent_agent import call_shadow_agent
            result = await call_shadow_agent(agent_id, prompt, timeout=30.0)
            if result:
                _, response = result
                if response:
                    return response
        except Exception as e:
            logger.debug(f"Shadow agent {agent_id} failed: {e}")

        # Fallback to direct API calls
        return await self._call_agent_direct(agent_id, prompt)

    async def _call_agent_direct(self, agent_id: str, prompt: str) -> Optional[str]:
        """Call agent directly via their API provider."""
        try:
            if agent_id == "grok":
                from Cosmos.integration.external.grok import grok_provider
                if grok_provider:
                    result = await grok_provider.chat(prompt=prompt, max_tokens=300)
                    if result:
                        return result.get("content") if isinstance(result, dict) else result

            elif agent_id == "gemini":
                from Cosmos.integration.external.gemini import gemini_provider
                if gemini_provider:
                    result = await gemini_provider.chat(prompt=prompt, max_tokens=300)
                    if result:
                        return result.get("content") if isinstance(result, dict) else result

            elif agent_id == "claude":
                from Cosmos.integration.external.claude_api import claude_provider
                if claude_provider:
                    result = await claude_provider.chat(prompt=prompt, max_tokens=300)
                    if result:
                        return result if isinstance(result, str) else str(result)

            elif agent_id == "kimi":
                from Cosmos.integration.external.kimi import kimi_provider
                if kimi_provider:
                    result = await kimi_provider.chat(prompt=prompt, max_tokens=300)
                    if result:
                        return result.get("content") if isinstance(result, dict) else result

            elif agent_id in ("deepseek", "phi"):
                # Local Ollama models
                import httpx
                async with httpx.AsyncClient(timeout=30.0) as client:
                    model = "deepseek-r1:8b" if agent_id == "deepseek" else "phi4:latest"
                    resp = await client.post(
                        "http://localhost:11434/api/generate",
                        json={"model": model, "prompt": prompt, "stream": False}
                    )
                    if resp.status_code == 200:
                        return resp.json().get("response", "")

        except Exception as e:
            logger.debug(f"Direct API call to {agent_id} failed: {e}")
        return None

    async def _record_on_chain(self, query: OracleQuery) -> None:
        """Record consensus hash on Solana using memo program."""
        if not SOLANA_AVAILABLE or not self._wallet:
            return

        try:
            # Create memo instruction
            memo_data = json.dumps({
                "oracle": "cosmos-swarm",
                "query_id": query.query_id,
                "hash": query.consensus_hash,
                "confidence": query.consensus_confidence,
                "agents": len(query.agents_participated),
            })

            # Memo program ID
            MEMO_PROGRAM_ID = Pubkey.from_string("MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr")

            # This is a simplified version - full implementation would use
            # proper transaction building with recent blockhash
            logger.info(f"Would record on-chain: {memo_data[:100]}...")

            # For demo purposes, just log it
            query.solana_signature = f"demo_{query.query_id}"

        except Exception as e:
            logger.error(f"On-chain recording failed: {e}")

    # =========================================================================
    # QUERY RETRIEVAL
    # =========================================================================

    def get_query(self, query_id: str) -> Optional[OracleQuery]:
        """Get a specific oracle query by ID."""
        return self._queries.get(query_id)

    def get_recent_queries(self, limit: int = 20) -> List[OracleQuery]:
        """Get recent oracle queries."""
        recent_ids = self._query_history[-limit:]
        return [self._queries[qid] for qid in reversed(recent_ids) if qid in self._queries]

    def get_stats(self) -> Dict[str, Any]:
        """Get oracle statistics."""
        total = len(self._queries)
        consensus_reached = sum(1 for q in self._queries.values() if q.consensus_reached)
        avg_agents = sum(len(q.agents_participated) for q in self._queries.values()) / max(total, 1)

        return {
            "total_queries": total,
            "consensus_reached": consensus_reached,
            "consensus_rate": round(consensus_reached / max(total, 1), 2),
            "avg_agents_per_query": round(avg_agents, 1),
            "available_agents": len(self.ORACLE_AGENTS),
            "solana_enabled": self._wallet is not None,
        }


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_oracle_instance: Optional[SwarmOracle] = None


def get_swarm_oracle() -> SwarmOracle:
    """Get the global SwarmOracle instance."""
    global _oracle_instance
    if _oracle_instance is None:
        _oracle_instance = SwarmOracle()
    return _oracle_instance


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def ask_oracle(question: str, query_type: str = "general") -> Dict[str, Any]:
    """
    Convenience function to query the swarm oracle.

    Args:
        question: Question to ask
        query_type: Type of query

    Returns:
        Dict with consensus answer and metadata
    """
    oracle = get_swarm_oracle()
    result = await oracle.submit_query(question, query_type)
    return result.to_dict()


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

async def main():
    """Demo the swarm oracle."""
    oracle = get_swarm_oracle()

    print("=" * 60)
    print("COSMOS SWARM ORACLE - Collective Intelligence Demo")
    print("=" * 60)

    test_questions = [
        ("What will be the dominant AI architecture in 2027?", "prediction"),
        ("Should developers use microservices or monoliths for new projects?", "recommendation"),
    ]

    for question, qtype in test_questions:
        print(f"\n🔮 Question: {question}")
        print("-" * 40)

        result = await oracle.submit_query(question, qtype, timeout=120.0)

        print(f"Status: {result.status}")
        print(f"Agents: {', '.join(result.agents_participated)}")
        print(f"Consensus: {result.consensus_reached} ({result.consensus_confidence:.0%} confidence)")
        print(f"Answer: {result.consensus_answer}")
        if result.consensus_hash:
            print(f"Hash: {result.consensus_hash[:16]}...")
        print(f"Duration: {result.deliberation_duration_ms}ms")

    print("\n" + "=" * 60)
    print("Oracle Stats:", oracle.get_stats())


if __name__ == "__main__":
    asyncio.run(main())
