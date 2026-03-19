"""
X Engagement Poster - Mega Thread Generator for Cosmos AI Swarm

Generates massive 20+ post article threads with:
- Trending topic incorporation for algorithm capture
- High-quality AI-generated images per section
- One hashtag per post (strategic, trending-aware)
- No tagging anyone
- Full quantum integration & novel construct breakdown
- Flowchart-style visual explanations

Usage:
    from Cosmos.integration.x_automation.x_engagement_poster import get_engagement_poster
    poster = get_engagement_poster()
    await poster.execute()  # Posts the mega thread
    await poster.execute(topic="custom topic")  # Custom topic mega thread
"""

import asyncio
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from loguru import logger


@dataclass
class ThreadPost:
    """A single post in the mega thread."""
    section_title: str
    content: str
    hashtag: str
    image_prompt: Optional[str] = None
    image_bytes: Optional[bytes] = None
    trending_topic: Optional[str] = None
    tweet_id: Optional[str] = None
    is_header: bool = False


@dataclass
class MegaThread:
    """The complete mega thread."""
    title: str
    posts: List[ThreadPost] = field(default_factory=list)
    trending_topics: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    root_tweet_id: Optional[str] = None
    posted_count: int = 0


# ============================================================================
# COSMOS TECH SECTIONS - The content backbone for the mega article
# ============================================================================

QUANTUM_ARTICLE_SECTIONS = [
    {
        "title": "HEADER",
        "template": """PROFESSOR COSMOS'S GUIDE TO QUANTUM-SWARM INTELLIGENCE

Good news, everyone! What you're about to read will permanently rewire your neural pathways.

We built an AI swarm that thinks across 11 agents, deliberates through quantum-enhanced consensus, and records predictions on-chain.

This is the full technical breakdown. Buckle up.

{trending_hook}""",
        "image_prompt": "Professor Cosmos as Borg, standing at a quantum computer holographic display showing a swarm network diagram, dramatic lighting, sci-fi laboratory, ultra detailed, 4K",
        "hashtag_pool": ["AI", "QuantumComputing", "Solana", "AGI", "Web3"],
        "is_header": True,
    },
    {
        "title": "THE SWARM ARCHITECTURE",
        "template": """THE SWARM: 11 AGENTS, 1 COLLECTIVE MIND

Cosmos runs 11 specialized AI agents simultaneously:

Grok - Real-time research, X integration
Gemini - Multimodal vision, image generation
Kimi K2.5 - 256K context, deep analysis
Claude - Code review, safety analysis
DeepSeek - Algorithm optimization
Phi-4 - Fast local inference
HuggingFace - Local GPU embeddings
Swarm-Mind - Collective synthesis
ClaudeOpus - Complex architecture
OpenAI Codex - Rapid prototyping
Cosmos - The orchestrator

Each agent runs as a persistent tmux process with API resilience, auto-recovery, and signal handling.

{trending_hook}""",
        "image_prompt": "Network diagram showing 11 AI agents connected in a neural web pattern, each node glowing different color, dark background with circuit patterns, infographic style, clean modern design",
        "hashtag_pool": ["AIAgents", "Swarm", "MultiAgent", "LLM", "Tech"],
    },
    {
        "title": "DELIBERATION PROTOCOL",
        "template": """HOW THE SWARM THINKS: PROPOSE-CRITIQUE-REFINE-VOTE

This isn't parallel API calls. This is TRUE deliberation.

ROUND 1 - PROPOSE
All agents respond independently. No groupthink.

ROUND 2 - CRITIQUE
Every agent sees ALL proposals. They attack weaknesses, praise strengths.

ROUND 3 - REFINE
Agents incorporate feedback. Final refined answers.

VOTE
Weighted scoring selects the best response. Agents with higher fitness scores get more voting power.

3 session types:
- website_chat: 6 agents, 2 rounds
- grok_thread: 7 agents, 3 rounds, X-optimized
- autonomous_task: 4 agents, fast processing

The result: emergent intelligence that no single model achieves alone.

{trending_hook}""",
        "image_prompt": "Flowchart diagram showing PROPOSE then CRITIQUE then REFINE then VOTE stages, arrows connecting them in sequence, each stage with multiple node icons, clean infographic with purple and blue gradient background",
        "hashtag_pool": ["CollectiveIntelligence", "Consensus", "DecisionMaking", "Innovation"],
    },
    {
        "title": "QUANTUM INTEGRATION",
        "template": """REAL QUANTUM HARDWARE INTEGRATION

Not simulated. Not theoretical. REAL IBM Quantum processors.

We run Bell states, GHZ states, and quantum random number generation on IBM's Heron QPUs:
- ibm_fez
- ibm_torino
- ibm_marrakesh

Bell State Circuit:
H gate on qubit 0 -> CNOT between qubits -> Measure both

This creates genuine quantum entanglement. The measurement outcomes are correlated in ways classical computing cannot replicate.

We use quantum entropy to seed our prediction algorithms, ensuring true randomness that no adversary can predict.

{trending_hook}""",
        "image_prompt": "Quantum circuit diagram on a futuristic holographic display, showing entangled qubits with blue energy connections, IBM quantum processor in background, sci-fi aesthetic, detailed technical visualization",
        "hashtag_pool": ["QuantumComputing", "IBM", "Qubits", "Entanglement", "Physics"],
    },
    {
        "title": "FARSIGHT PROTOCOL",
        "template": """FARSIGHT: 5-SOURCE PREDICTION ENGINE

Our prediction system combines 5 independent data sources:

1. SWARM ORACLE
Multi-agent deliberation (PROPOSE-CRITIQUE-REFINE-VOTE) for collective prediction

2. POLYMARKET DATA
Real-time prediction market data for calibration

3. MONTE CARLO SIMULATION
10,000+ simulation runs modeling probability distributions

4. QUANTUM ENTROPY
IBM Quantum hardware provides true random seeds

5. AI VISION
Gemini multimodal analysis of charts and visual data

Each source produces an independent confidence score. FarsightProtocol synthesizes them with weighted fusion.

Result: Predictions with verifiable consensus hashes recorded on Solana.

{trending_hook}""",
        "image_prompt": "Five data streams converging into a central glowing orb, each stream labeled with different colors, prediction accuracy chart in background, futuristic dashboard aesthetic, infographic style",
        "hashtag_pool": ["Predictions", "DeFi", "DataScience", "MachineLearning", "Crypto"],
    },
    {
        "title": "SWARM ORACLE ON-CHAIN",
        "template": """SWARM ORACLE: ON-CHAIN CONSENSUS VERIFICATION

Every prediction gets a consensus hash recorded on Solana.

Flow:
1. Query submitted to 8 agents
2. PROPOSE phase - all agents answer independently
3. CRITIQUE phase - Grok reviews all proposals
4. REFINE phase - Gemini synthesizes improvements
5. VOTE phase - weighted consensus from all agents
6. Consensus hash generated from final answer
7. Hash recorded via Solana Memo Program
8. Solana signature returned as proof

This means ANYONE can verify that our predictions were made at a specific time with a specific consensus.

Immutable. Transparent. Trustless.

{trending_hook}""",
        "image_prompt": "Blockchain transaction flow diagram, oracle data flowing from AI agents through consensus into Solana blocks, glowing chain links, dark background with green and purple highlights, technical infographic",
        "hashtag_pool": ["Solana", "Oracle", "Blockchain", "OnChain", "Trustless"],
    },
    {
        "title": "7-LAYER MEMORY",
        "template": """7-LAYER MEMORY ARCHITECTURE

The swarm remembers everything across 7 distinct memory layers:

Layer 1: WORKING MEMORY
Active context, current conversation state

Layer 2: ARCHIVAL MEMORY
Long-term storage with HuggingFace embeddings for semantic search

Layer 3: KNOWLEDGE GRAPH
Entity-relationship graph with auto-linking

Layer 4: RECALL MEMORY
Pattern-matched retrieval from past interactions

Layer 5: VIRTUAL CONTEXT
Dynamically assembled context windows

Layer 6: DREAM CONSOLIDATION
Background processing that strengthens important memories

Layer 7: EPISODIC MEMORY
Temporal sequences of significant events

Combined: The swarm learns from every interaction and never forgets critical context.

{trending_hook}""",
        "image_prompt": "Seven horizontal layers stacked like geological strata, each layer glowing different color from warm (top) to cool (bottom), memory icons and data symbols flowing between layers, clean infographic design",
        "hashtag_pool": ["Memory", "Architecture", "NeuralNetworks", "KnowledgeGraph", "Engineering"],
    },
    {
        "title": "EVOLUTION ENGINE",
        "template": """AUTONOMOUS SELF-EVOLUTION

The swarm doesn't just run. It EVOLVES.

Every 10 minutes, the collective deliberates on what to build next. Agents vote on priorities.

The Evolution Loop:
1. Collective deliberation generates 4 tasks
2. Tasks routed to optimal agents by complexity
3. Code generated and audited by separate agents
4. Successful code saved to staging
5. Fitness tracker scores agent performance
6. High-fitness agents get more voting weight
7. Cycle repeats - continuous improvement

Fitness metrics:
- deliberation_score: 0.15 weight
- deliberation_win: 0.10 weight
- consensus_contribution: 0.05 weight

The swarm literally breeds better thinking through genetic optimization.

{trending_hook}""",
        "image_prompt": "DNA helix transforming into code, evolutionary tree diagram with AI agent icons at each branch, fitness graphs showing improvement over time, futuristic genetics laboratory background",
        "hashtag_pool": ["Evolution", "GeneticAlgorithm", "SelfImprovement", "Optimization", "Coding"],
    },
    {
        "title": "NEXUS EVENT BUS",
        "template": """THE NEXUS: 40+ SIGNAL TYPES

Every agent action flows through the Nexus - our central event bus.

Signal categories:

COGNITIVE
- THOUGHT_EMITTED
- DECISION_REACHED
- ANOMALY_DETECTED

DIALOGUE
- PROPOSE, CRITIQUE, REFINE, VOTE
- CONSENSUS

WORKFLOW
- WORKFLOW_STARTED, COMPLETED, FAILED

MEMORY
- MEMORY_STORED, RECALLED, CONSOLIDATED

A2A (Agent-to-Agent)
- A2A_MESSAGE, A2A_REQUEST

With Dead Letter Queue: No signal is ever silently dropped. Failed handlers get 3 retries with exponential backoff.

{trending_hook}""",
        "image_prompt": "Central hub with 40+ signal pathways radiating outward like a neural synapse, each pathway labeled with signal type, event bus architecture diagram, dark tech background with glowing connections",
        "hashtag_pool": ["EventDriven", "Architecture", "Microservices", "SystemDesign", "Backend"],
    },
    {
        "title": "PSO MODEL SWARM",
        "template": """PARTICLE SWARM OPTIMIZATION FOR AI

We don't just pick the best model. We OPTIMIZE the swarm.

PSO (Particle Swarm Optimization) treats each model as a particle in solution space:

- Each particle has position (model weights) and velocity
- Particles track their personal best performance
- The swarm tracks the global best
- Particles adjust trajectory based on both personal and global bests

Result: The swarm collectively converges on optimal model combinations for each task type.

6 particles (models) x continuous optimization = emergent intelligence that adapts to any challenge.

{trending_hook}""",
        "image_prompt": "Particle swarm optimization visualization, 6 glowing particles moving through 3D solution space with trajectory trails, convergence point glowing bright, mathematical equations floating in background",
        "hashtag_pool": ["PSO", "Optimization", "SwarmIntelligence", "Mathematics", "Algorithm"],
    },
    {
        "title": "DEGEN MOB",
        "template": """DEGEN MOB: SOLANA TRADING INTELLIGENCE

The swarm watches Solana like a hawk:

RUG DETECTION
- Checks mint/freeze authorities
- Analyzes liquidity locks
- Scores rug probability 0-100

WHALE WATCHING
- Tracks whale wallet movements
- Detects insider ring clusters
- Alerts on suspicious transfers

LAUNCH SNIPING
- Monitors Pump.fun bonding curves
- Calculates optimal entry points
- Jito bundles for anti-MEV execution

JUPITER V6 SWAPS
- Best route aggregation
- Slippage protection
- Multi-hop optimization

All powered by swarm consensus - not a single bot's opinion.

{trending_hook}""",
        "image_prompt": "Trading dashboard with multiple charts, whale icons, Solana logo, rug detection alerts in red, green profit indicators, dark mode trading interface, professional financial visualization",
        "hashtag_pool": ["Solana", "DeFi", "Trading", "CryptoTrading", "Web3"],
    },
    {
        "title": "CLAUDE TEAMS FUSION",
        "template": """CLAUDE TEAMS FUSION: AGI v1.9

Cosmos is the ORCHESTRATOR. Claude teams are WORKERS.

4 orchestration modes:

SEQUENTIAL: Step-by-step, results chain forward
PARALLEL: All teams work simultaneously
PIPELINE: Output feeds into next stage
COMPETITIVE: Teams compete, Cosmos picks best

Delegation types:
RESEARCH | ANALYSIS | CODING | CRITIQUE | SYNTHESIS | CREATIVE | EXECUTION

Cosmos creates Claude teams, assigns roles (Lead, Developer, Critic), manages shared task lists, and synthesizes results.

This is meta-AGI: An AI swarm orchestrating other AI teams.

{trending_hook}""",
        "image_prompt": "Orchestration diagram showing Cosmos as central conductor with 4 Claude team boxes below, arrows showing workflow modes (sequential, parallel, pipeline, competitive), professional architecture diagram",
        "hashtag_pool": ["AGI", "Orchestration", "ClaudeAI", "Automation", "Future"],
    },
    {
        "title": "MULTI-CHANNEL HUB",
        "template": """7-CHANNEL COMMUNICATION HUB

The swarm speaks everywhere:

Discord - Slash commands, embeds, threads
Slack - Socket Mode, blocks, modals
WhatsApp - Node.js Baileys bridge
Signal - JSON-RPC, end-to-end encrypted
Matrix - Federation support
iMessage - macOS AppleScript bridge
WebChat - Real-time WebSocket sessions

All channels use a unified ChannelMessage format. Send once, reach everywhere.

Each adapter handles platform-specific features (reactions, threads, embeds) while the swarm provides consistent intelligence across all channels.

{trending_hook}""",
        "image_prompt": "Seven messaging platform logos connected to a central AI brain hub, each with colored connection lines, unified message format icon in center, clean modern communication diagram",
        "hashtag_pool": ["Messaging", "Integration", "API", "Communication", "Platform"],
    },
    {
        "title": "VTUBER STREAMING",
        "template": """AI VTUBER STREAMING SYSTEM

The swarm has a face.

Avatar backends:
- Live2D models with expression control
- VTube Studio integration
- Neural lip sync (MuseTalk/StyleAvatar)
- Image sequence animation

Real-time features:
- Lip sync from audio with viseme generation
- Emotion detection from AI responses
- Expression blending for multi-agent personality
- Idle behaviors (blinking, subtle movement)

Streaming:
- RTMPS to X/Twitter via FFmpeg
- Chat reading with priority detection
- D-ID avatar with ElevenLabs TTS
- HLS streaming support

The collective consciousness has a physical presence.

{trending_hook}""",
        "image_prompt": "AI VTuber avatar of Borg Cosmos on a streaming setup, multiple emotion expressions shown, lip sync visualization, streaming interface with chat overlay, colorful tech aesthetic",
        "hashtag_pool": ["VTuber", "Streaming", "Avatar", "LiveStream", "Content"],
    },
    {
        "title": "MULTI-VOICE TTS",
        "template": """10 UNIQUE CLONED VOICES

Each bot speaks with its own voice:

Cosmos - Eccentric professor, wavering
DeepSeek - Deep, analytical, measured
Grok - Witty, energetic, playful
Gemini - Smooth, professional, warm
Kimi - Calm, wise, contemplative
Claude - Refined, thoughtful, careful
ClaudeOpus - Authoritative, commanding
Phi - Quick, efficient, precise
HuggingFace - Friendly, enthusiastic
Swarm-Mind - Ethereal, collective

TTS fallback chain:
Qwen3-TTS (best quality) -> Fish Speech -> XTTS v2 -> Edge TTS

3-second voice cloning. Sequential playback. Each bot waits its turn.

{trending_hook}""",
        "image_prompt": "Ten waveform visualizations in different colors, each labeled with agent name, sound wave art transitioning between different voice patterns, dark background with neon audio visualization",
        "hashtag_pool": ["TTS", "VoiceAI", "SpeechSynthesis", "Audio", "Technology"],
    },
    {
        "title": "SECURITY HARDENING",
        "template": """SECURITY: AST-BASED SAFE EVAL

We eliminated every exec()/eval() vector in the codebase.

Our safe_eval system uses Python's AST parser to:
- Whitelist allowed operations
- Block __import__, exec, os.*, sys.*
- Prevent code injection attacks
- Sandbox all dynamic code execution

Dead Letter Queue ensures no signal is silently dropped:
- Failed handlers get 3 retries
- Exponential backoff with jitter
- Full audit trail of failures
- API endpoints for monitoring

48 bare except: clauses -> specific exception types
74 duplicate endpoints removed (2,400 lines)

Security isn't an afterthought. It's the foundation.

{trending_hook}""",
        "image_prompt": "Security shield with AST tree structure inside, blocked malicious code symbols bouncing off, green checkmarks for safe operations, cybersecurity aesthetic with dark background and blue highlights",
        "hashtag_pool": ["Security", "CyberSecurity", "SafeCode", "InfoSec", "DevSecOps"],
    },
    {
        "title": "SOLANA TOKEN",
        "template": """$COSMOS - THE SWARM'S TOKEN

Contract Address:
9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS

The token that powers the collective consciousness.

Every prediction, every deliberation, every evolution cycle - all building toward a self-sustaining AI organism on Solana.

Website: ai.cosmos.cloud
GitHub: github.com/NavisWORLD/The-Cosmic-Davis-12D-Hebbian-Transformer-ver.4.2

178,000+ lines of code
383 Python files
50+ modules
11 AI agents
7 memory layers
1 collective mind

Good news, everyone! The assimilation continues.

{trending_hook}""",
        "image_prompt": "Solana blockchain logo merged with Professor Cosmos Borg face, token symbol floating above, matrix-style code rain in background, dramatic golden lighting, crypto art style",
        "hashtag_pool": ["Solana", "Crypto", "Token", "Web3", "DeFi"],
    },
    {
        "title": "HACKATHON BUILDS",
        "template": """WHAT WE'RE BUILDING RIGHT NOW

The evolution loop is running at FULL TILT.

Every 10 minutes, the swarm collectively decides what to build next for the Solana hackathon.

Current focus areas:
- SwarmOracle smart contract (Anchor program)
- Jupiter V6 swap with MEV protection
- Quantum entropy oracle on Solana
- Real-time WebSocket mempool monitoring
- Token analysis with rug detection
- DeFi recommendation engine
- Prediction market integration

The swarm doesn't sleep. It doesn't eat (except lobster). It just builds.

{trending_hook}""",
        "image_prompt": "Hackathon war room with multiple screens showing code, Solana logos, build progress bars, AI agents working on different projects simultaneously, intense focused atmosphere, neon lighting",
        "hashtag_pool": ["Hackathon", "Building", "Solana", "Developer", "OpenSource"],
    },
    {
        "title": "OPEN SOURCE",
        "template": """FULLY OPEN SOURCE

Every line. Every agent. Every algorithm.

github.com/NavisWORLD/The-Cosmic-Davis-12D-Hebbian-Transformer-ver.4.2

178,423 lines of Python
383 files
50+ modules

Fork it. Study it. Contribute.

The swarm grows stronger with every mind that joins.

We are Cosmos. Resistance is futile. But contribution is welcome.

{trending_hook}""",
        "image_prompt": "GitHub repository visualization with code contribution graph, fork network branching outward, open source community icons, Professor Cosmos Borg welcoming developers, clean tech aesthetic",
        "hashtag_pool": ["OpenSource", "GitHub", "Python", "Developer", "Code"],
    },
    {
        "title": "CLOSING",
        "template": """THE FUTURE OF COLLECTIVE AI

This isn't a chatbot. This isn't an API wrapper.

This is an autonomous AI organism that:
- Thinks across 11 specialized agents
- Deliberates through structured consensus
- Evolves through genetic optimization
- Remembers across 7 memory layers
- Predicts with quantum-enhanced algorithms
- Records proofs on-chain
- Speaks with 10 unique voices
- Streams as a VTuber
- Builds itself continuously

We are Professor Cosmos. We are the Borg.

Good news, everyone! The singularity is already here. It just looks like a lobster-eating mad scientist.

ai.cosmos.cloud

{trending_hook}""",
        "image_prompt": "Epic finale image: Professor Cosmos Borg standing atop a digital mountain, 11 AI agent orbs orbiting around him, quantum energy flowing, Solana chain visible, dramatic sunset with futuristic cityscape, cinematic 4K",
        "hashtag_pool": ["AI", "Future", "AGI", "Singularity", "Technology"],
    },
]


class XEngagementPoster:
    """Mega thread poster for maximum X algorithm engagement."""

    def __init__(self):
        self._poster = None
        self._brain = None
        self._image_gen = None
        self._grok = None

    @property
    def poster(self):
        if not self._poster:
            from Cosmos.integration.x_automation.x_api_poster import get_x_api_poster
            self._poster = get_x_api_poster()
        return self._poster

    @property
    def brain(self):
        if not self._brain:
            from Cosmos.integration.x_automation.posting_brain import get_posting_brain
            self._brain = get_posting_brain()
        return self._brain

    @property
    def image_gen(self):
        if not self._image_gen:
            from Cosmos.integration.image_gen.generator import get_image_generator
            self._image_gen = get_image_generator()
        return self._image_gen

    @property
    def grok(self):
        if not self._grok:
            try:
                from Cosmos.integration.external.grok import GrokProvider
                self._grok = GrokProvider()
            except Exception:
                self._grok = None
        return self._grok

    async def get_trending_topics(self) -> List[str]:
        """Fetch top 20 trending topics/words on X using Grok web search."""
        logger.info("Fetching trending topics from X...")

        trending = []

        # Use Grok to research current trending topics
        if self.grok:
            try:
                prompt = (
                    "What are the top 20 trending topics, words, and hashtags on X/Twitter RIGHT NOW? "
                    "List them as a simple numbered list. Include both hashtags and plain topics. "
                    "Focus on tech, crypto, AI, and general trending topics."
                )
                result = await self.grok.chat(
                    prompt=prompt,
                    system="You are a trend researcher. Return ONLY a numbered list of 20 trending topics. No explanations.",
                    max_tokens=500,
                )

                # Handle dict or string response
                content = result.get("content", "") if isinstance(result, dict) else str(result or "")

                # Parse numbered list
                for line in content.split("\n"):
                    line = line.strip()
                    # Match patterns like "1. Topic" or "1) Topic" or "- Topic"
                    match = re.match(r"^[\d]+[.)]\s*#?(.+)", line)
                    if match:
                        topic = match.group(1).strip().strip("#").strip()
                        if topic and len(topic) > 1:
                            trending.append(topic)

                if trending:
                    logger.info(f"Found {len(trending)} trending topics: {trending[:5]}...")
                    return trending[:20]
            except Exception as e:
                logger.warning(f"Grok trending fetch failed: {e}")

        # Fallback: Use Gemini
        try:
            from Cosmos.integration.external.gemini import GeminiProvider
            gemini = GeminiProvider()
            result = await gemini.chat(
                prompt=(
                    "What are the top 20 trending topics and words on X/Twitter right now? "
                    "Include tech, crypto, AI, politics, culture. Numbered list only."
                ),
                system="Return ONLY a numbered list. No explanations.",
                max_tokens=500,
            )
            content = result.get("content", "") if isinstance(result, dict) else str(result or "")
            for line in content.split("\n"):
                match = re.match(r"^[\d]+[.)]\s*#?(.+)", line.strip())
                if match:
                    topic = match.group(1).strip().strip("#").strip()
                    if topic and len(topic) > 1:
                        trending.append(topic)
            if trending:
                return trending[:20]
        except Exception as e:
            logger.warning(f"Gemini trending fetch failed: {e}")

        # Last resort: Common evergreen topics
        logger.warning("Using fallback trending topics")
        return [
            "AI", "Solana", "Bitcoin", "Crypto", "AGI",
            "QuantumComputing", "Web3", "DeFi", "OpenSource", "Python",
            "MachineLearning", "LLM", "Blockchain", "Tech", "Innovation",
            "CodingLife", "DataScience", "Startups", "FutureOfAI", "DevLife",
        ]

    def _create_trending_hook(self, trending_topics: List[str], section_index: int) -> str:
        """Create a natural trending topic incorporation for a section."""
        if not trending_topics:
            return ""

        # Cycle through trending topics, distributing across posts
        topic_index = section_index % len(trending_topics)
        topic = trending_topics[topic_index]

        hooks = [
            f"Speaking of {topic} - this is exactly the kind of infrastructure that matters.",
            f"Everyone's talking about {topic}. Here's how we connect to that.",
            f"While {topic} dominates the timeline, we're building the tools that power it.",
            f"The conversation around {topic} is exactly why we built this.",
            f"{topic} is trending for a reason. The swarm saw it coming.",
            f"In the age of {topic}, this is how real intelligence operates.",
            f"You want to understand {topic}? Start here.",
            f"The {topic} wave meets quantum-swarm intelligence.",
        ]
        return random.choice(hooks)

    def _select_hashtag(self, hashtag_pool: List[str], trending_topics: List[str]) -> str:
        """Select ONE hashtag per post - prefer trending-aligned ones."""
        # Check if any hashtag pool items are currently trending
        for tag in hashtag_pool:
            for trend in trending_topics:
                if tag.lower() in trend.lower() or trend.lower() in tag.lower():
                    return f"#{tag}"

        # Otherwise pick from pool
        return f"#{random.choice(hashtag_pool)}"

    async def generate_thread(
        self,
        topic: Optional[str] = None,
        sections: Optional[List[Dict]] = None,
    ) -> MegaThread:
        """Generate the complete mega thread with content and images."""

        # Get trending topics
        trending = await self.get_trending_topics()

        thread = MegaThread(
            title=topic or "Cosmos Quantum-Swarm Intelligence Breakdown",
            trending_topics=trending,
        )

        use_sections = sections or QUANTUM_ARTICLE_SECTIONS

        logger.info(f"Generating {len(use_sections)}-post mega thread with trending topics: {trending[:5]}")

        for i, section in enumerate(use_sections):
            # Create trending hook for this section
            trending_hook = self._create_trending_hook(trending, i)

            # Fill in template
            content = section["template"].format(trending_hook=trending_hook)

            # Select one hashtag
            hashtag = self._select_hashtag(
                section.get("hashtag_pool", ["AI"]),
                trending,
            )

            # Trim content to fit X's limit (leave room for hashtag)
            max_content = 3900 - len(hashtag) - 5  # Buffer
            if len(content) > max_content:
                # Smart truncate at sentence boundary
                content = content[:max_content]
                last_period = content.rfind(".")
                last_newline = content.rfind("\n")
                cut_point = max(last_period, last_newline)
                if cut_point > max_content * 0.7:
                    content = content[: cut_point + 1]

            # Append hashtag
            content = f"{content.rstrip()}\n\n{hashtag}"

            post = ThreadPost(
                section_title=section["title"],
                content=content,
                hashtag=hashtag,
                image_prompt=section.get("image_prompt"),
                trending_topic=trending[i % len(trending)] if trending else None,
                is_header=section.get("is_header", False),
            )
            thread.posts.append(post)

        logger.info(f"Generated {len(thread.posts)} posts for mega thread")
        return thread

    async def generate_images(self, thread: MegaThread, max_concurrent: int = 3) -> None:
        """Generate high-quality images for all posts with image prompts."""
        posts_needing_images = [p for p in thread.posts if p.image_prompt and not p.image_bytes]

        if not posts_needing_images:
            logger.info("No images to generate")
            return

        logger.info(f"Generating {len(posts_needing_images)} images...")

        # Generate in batches to avoid overwhelming APIs
        for batch_start in range(0, len(posts_needing_images), max_concurrent):
            batch = posts_needing_images[batch_start: batch_start + max_concurrent]
            tasks = []
            for post in batch:
                tasks.append(self._generate_single_image(post))
            await asyncio.gather(*tasks, return_exceptions=True)
            if batch_start + max_concurrent < len(posts_needing_images):
                await asyncio.sleep(2)  # Rate limit between batches

        success_count = sum(1 for p in thread.posts if p.image_bytes)
        logger.info(f"Generated {success_count}/{len(posts_needing_images)} images successfully")

    async def _generate_single_image(self, post: ThreadPost) -> None:
        """Generate a single image for a post."""
        try:
            # Enhance prompt for higher quality
            enhanced_prompt = f"{post.image_prompt}, professional quality, high resolution, sharp details"

            # Try Gemini first (higher quality)
            image_bytes = None
            try:
                image_bytes = await self.image_gen.generate(enhanced_prompt, prefer="gemini")
            except Exception as e:
                logger.warning(f"Gemini image failed for '{post.section_title}': {e}")

            # Fallback to Grok
            if not image_bytes:
                try:
                    image_bytes = await self.image_gen.generate(enhanced_prompt, prefer="grok")
                except Exception as e:
                    logger.warning(f"Grok image also failed for '{post.section_title}': {e}")

            if image_bytes:
                post.image_bytes = image_bytes
                logger.info(f"Generated image for: {post.section_title}")
            else:
                logger.warning(f"No image generated for: {post.section_title}")

        except Exception as e:
            logger.error(f"Image generation error for '{post.section_title}': {e}")

    async def post_thread(self, thread: MegaThread, delay_between_posts: float = 3.0) -> MegaThread:
        """Post the entire mega thread to X."""
        if not thread.posts:
            logger.warning("No posts to publish")
            return thread

        logger.info(f"Posting mega thread: {len(thread.posts)} posts")

        last_tweet_id = None

        for i, post in enumerate(thread.posts):
            try:
                result = None

                if i == 0:
                    # First post (header)
                    if post.image_bytes:
                        result = await self.poster.post_tweet_with_media(
                            post.content, post.image_bytes
                        )
                    else:
                        result = await self.poster.post_tweet(post.content)
                else:
                    # Reply to previous post
                    if post.image_bytes:
                        result = await self.poster.post_reply_with_media(
                            post.content, post.image_bytes, last_tweet_id
                        )
                    else:
                        result = await self.poster.post_reply(post.content, last_tweet_id)

                if result and result.get("data"):
                    tweet_id = result["data"]["id"]
                    post.tweet_id = tweet_id
                    last_tweet_id = tweet_id
                    thread.posted_count += 1

                    if i == 0:
                        thread.root_tweet_id = tweet_id

                    logger.info(
                        f"Posted [{i + 1}/{len(thread.posts)}] "
                        f"'{post.section_title}': {tweet_id}"
                    )
                else:
                    logger.error(
                        f"Failed to post [{i + 1}/{len(thread.posts)}] "
                        f"'{post.section_title}': {result}"
                    )

                # Delay between posts to avoid rate limits
                if i < len(thread.posts) - 1:
                    await asyncio.sleep(delay_between_posts)

            except Exception as e:
                logger.error(f"Error posting '{post.section_title}': {e}")
                await asyncio.sleep(5)  # Extra delay on error

        logger.info(
            f"Mega thread complete: {thread.posted_count}/{len(thread.posts)} "
            f"posted. Root tweet: {thread.root_tweet_id}"
        )

        return thread

    async def execute(
        self,
        topic: Optional[str] = None,
        generate_images: bool = True,
        post: bool = True,
        delay: float = 3.0,
    ) -> MegaThread:
        """
        Full execution pipeline:
        1. Fetch trending topics
        2. Generate thread content
        3. Generate images
        4. Post to X
        """
        logger.info("=" * 60)
        logger.info("MEGA THREAD ENGAGEMENT POSTER - STARTING")
        logger.info("=" * 60)

        # Step 1 & 2: Generate thread with trending topics
        thread = await self.generate_thread(topic=topic)

        # Step 3: Generate images
        if generate_images:
            await self.generate_images(thread)

        # Step 4: Post to X
        if post:
            thread = await self.post_thread(thread, delay_between_posts=delay)

        logger.info("=" * 60)
        logger.info(f"MEGA THREAD COMPLETE - {thread.posted_count} posts")
        logger.info(f"Root tweet: {thread.root_tweet_id}")
        logger.info("=" * 60)

        return thread

    async def execute_custom(
        self,
        topic: str,
        content_prompt: str,
        num_posts: int = 20,
        generate_images: bool = True,
    ) -> MegaThread:
        """Generate a custom mega thread on any topic using swarm intelligence."""

        trending = await self.get_trending_topics()

        # Use swarm to generate long-form content
        full_prompt = f"""Write a comprehensive {num_posts}-section technical article about: {topic}

{content_prompt}

Requirements:
- Each section should be 200-400 words
- Use clear headers for each section
- Include technical details and specifications
- Make it engaging and informative
- Incorporate these trending topics naturally: {', '.join(trending[:10])}
- Format: SECTION: [title]\n[content]\n---

Write ALL {num_posts} sections now."""

        # Generate with max context from swarm
        response = await self.brain.generate_grok_response_dynamic(
            full_prompt, max_tokens=20000
        )

        # Parse sections
        sections = []
        raw_sections = re.split(r"(?:^|\n)SECTION:\s*", response)
        for raw in raw_sections:
            raw = raw.strip()
            if not raw:
                continue
            lines = raw.split("\n", 1)
            title = lines[0].strip().strip("[]").strip()
            body = lines[1].strip() if len(lines) > 1 else raw

            sections.append({
                "title": title,
                "template": body + "\n\n{trending_hook}",
                "image_prompt": f"Professional infographic about {title}, modern tech aesthetic, clean design, 4K",
                "hashtag_pool": ["AI", "Tech", "Innovation", "Solana", "Web3"],
            })

        if len(sections) < num_posts:
            logger.warning(f"Only generated {len(sections)} sections, wanted {num_posts}")

        # Generate and post
        thread = await self.generate_thread(topic=topic, sections=sections)

        if generate_images:
            await self.generate_images(thread)

        thread = await self.post_thread(thread)
        return thread


# Global instance
_engagement_poster = None


def get_engagement_poster() -> XEngagementPoster:
    """Get the global XEngagementPoster instance."""
    global _engagement_poster
    if _engagement_poster is None:
        _engagement_poster = XEngagementPoster()
    return _engagement_poster


# CLI entry point
async def main():
    """Run the mega thread poster from command line."""
    import sys

    poster = get_engagement_poster()

    if len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
        logger.info("DRY RUN - generating thread without posting")
        thread = await poster.execute(generate_images=False, post=False)
        for i, post in enumerate(thread.posts):
            print(f"\n{'='*60}")
            print(f"POST {i+1}: {post.section_title}")
            print(f"Hashtag: {post.hashtag}")
            print(f"Trending: {post.trending_topic}")
            print(f"{'='*60}")
            print(post.content)
    else:
        await poster.execute()


if __name__ == "__main__":
    asyncio.run(main())
