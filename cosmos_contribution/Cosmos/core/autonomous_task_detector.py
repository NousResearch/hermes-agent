"""
Autonomous Task Detector + Innovation Watcher
----------------------------------------------
Monitors swarm chat for actionable ideas and automatically spawns development swarms.
ENHANCED: Now watches specifically for Cosmos's innovative ideas!

"Good news everyone! I can now detect when we should actually BUILD something!"

This module listens to the chat stream and identifies messages that suggest
beneficial, feasible development tasks. When detected, it spawns a parallel
development swarm to work on the task while the main chat continues.

INNOVATION WATCHER:
- Prioritizes ideas from Cosmos (the visionary)
- Catches breakthrough concepts and novel architectures
- Routes to best coding agents: Claude, Kimi, Grok, Gemini
- More aggressive detection for truly innovative ideas
"""

import asyncio
import re
import json
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger

# Priority bots whose ideas should be caught more aggressively
PRIORITY_IDEA_SOURCES = ["Cosmos", "Swarm-Mind", "Claude", "Grok"]

# Coding-capable agents that can implement ideas
CODING_AGENTS = ["Claude", "Kimi", "Grok", "Gemini", "DeepSeek"]

# Innovation patterns — bold ideas welcome, but must have technical specifics
INNOVATION_PATTERNS = [
    # Novel architectures with concrete terms
    (r"(adversarial|speculative|competitive) (debate|execution|routing|inference)", 0.8),
    (r"(self-improving|self-modifying|adaptive|evolutionary) (prompt|routing|weight|model|algorithm)", 0.8),
    (r"(prediction|predictor|forecast) (engine|system|model|pipeline)", 0.75),
    (r"(real-time|live) (analysis|tracking|monitoring|detection)", 0.7),
    (r"(cross-model|inter-agent|swarm) (memory|learning|synthesis|fusion)", 0.8),
    (r"(neural|learned|trained) (routing|selection|ranking|scoring)", 0.8),

    # Novel trading/DeFi concepts
    (r"(mempool|mev|frontrun|sandwich|arbitrage) (detect|analyz|predict|protect)", 0.8),
    (r"(slippage|liquidity|volume) (predict|model|estimat|analyz)", 0.75),

    # Direct action with specificity
    (r"i'?ve been (working on|developing|building|prototyping)", 0.7),
    (r"we could (integrate|combine|fuse|merge) \w+ (?:with|and|into) \w+", 0.7),
    (r"what if .{10,}(function|module|system|engine|pipeline|algorithm)", 0.75),
]

# Task detection patterns (original)
TASK_INDICATORS = [
    # Suggestions
    (r"we should (build|create|develop|implement|add|make)", 0.8),
    (r"let'?s (build|create|develop|implement|add|make)", 0.85),
    (r"what if we (built|created|developed|implemented|added|made)", 0.7),
    (r"could we (build|create|develop|implement|add)", 0.6),
    (r"it would be (great|good|useful|helpful) (to|if we)", 0.7),

    # Direct proposals
    (r"i('ll| will) (build|create|develop|implement|add)", 0.9),
    (r"we need (to|a) (build|create|develop|implement|add)", 0.75),
    (r"(propose|suggesting|recommend) (we |that we )?(build|create|add)", 0.8),

    # Feature ideas
    (r"(new feature|feature idea|enhancement):", 0.85),
    (r"trading (strategy|strategies|bot|system)", 0.7),
    (r"sentiment (analysis|tracker|monitor)", 0.7),
    (r"(api|integration|module) for", 0.6),

    # Improvement signals
    (r"(improve|enhance|upgrade|optimize) (the|our)", 0.65),
    (r"(fix|solve|address) (the|this) (issue|problem|bug)", 0.8),
]

# Feasibility indicators (boost score if present)
FEASIBILITY_BOOSTERS = [
    (r"(simple|straightforward|easy to)", 0.1),
    (r"(already have|existing|can use)", 0.15),
    (r"(api|library|module) (available|exists)", 0.1),
    (r"(just need to|only requires)", 0.1),
]

# Blockers (reduce score if present)
FEASIBILITY_BLOCKERS = [
    (r"(impossible|can't|cannot|won't work)", -0.5),
    (r"(too complex|too difficult|not feasible)", -0.4),
    (r"(maybe later|someday|future)", -0.3),
    (r"(joke|kidding|lol|haha)", -0.8),
    # Conversational responses that aren't tasks
    (r"^(certainly|indeed|intriguing|interesting|absolutely|definitely|agreed|yes|no|well|hmm)", -0.6),
    (r"(here'?s a|here is|that'?s a|let me explain|to summarize|in summary)", -0.4),
    (r"(great question|good point|thoughtful|insightful|fascinating)", -0.3),
    (r"(as (i|we|you) (mentioned|said|discussed))", -0.3),
    # Philosophical/discussion statements, not tasks
    (r"the (concept|idea|notion|philosophy|theory) of", -0.5),
    (r"(is an extension|represents|symbolizes|reflects|embodies) of", -0.4),
    (r"(philosophically|theoretically|conceptually|in principle)", -0.4),
    (r"(rights|ethics|morality|consciousness) (is|are|has|have)", -0.3),
    # Block our own notification messages from being detected as tasks!
    (r"(innovation detected|task detected|development swarm|spawned)", -0.9),
    (r"\*innovation\*|\*task\*", -0.9),
]


@dataclass
class DetectedTask:
    """A task detected from chat conversation."""
    id: str
    description: str
    source_message: str
    source_bot: str
    confidence: float
    category: str
    timestamp: datetime = field(default_factory=datetime.now)
    status: str = "detected"
    dev_swarm_id: Optional[str] = None
    recommended_agent: str = "Claude"  # Best agent to code this
    is_innovation: bool = False  # True if detected via innovation patterns

    def to_any(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "source_message": self.source_message[:200],
            "source_bot": self.source_bot,
            "confidence": self.confidence,
            "category": self.category,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
            "dev_swarm_id": self.dev_swarm_id,
            "recommended_agent": self.recommended_agent,
            "is_innovation": self.is_innovation
        }


class AutonomousTaskDetector:
    """
    Monitors chat for actionable ideas and spawns development swarms.

    The detector uses pattern matching and context analysis to identify
    messages that suggest beneficial development tasks. When a task is
    detected with high enough confidence, it automatically spawns a
    parallel development swarm to work on it.
    """

    def __init__(self, min_confidence: float = 0.75):
        self.min_confidence = min_confidence
        self.detected_tasks: list[DetectedTask] = []
        self.active_dev_swarms: dict = {}
        self.recent_messages: list[dict] = []  # Context window
        self.max_context = 10  # Last N messages for context
        self.cooldown_tasks: set = set()  # Prevent duplicate detection
        self.running = False

        # Rate limiting: max tasks per hour to prevent slop flooding
        self.max_tasks_per_hour = 4
        self.max_active_swarms = 3
        self._hourly_task_timestamps: list[datetime] = []

        # Stats
        self.total_detected = 0
        self.total_spawned = 0

        logger.info("AutonomousTaskDetector initialized (min_confidence=0.75, max_tasks/hr=4)")

    def analyze_message(self, message: dict) -> Optional[DetectedTask]:
        """
        Analyze a chat message for potential tasks.

        Returns DetectedTask if actionable idea found, None otherwise.

        ENHANCED: Now checks innovation patterns and gives priority to Cosmos!
        """
        content = message.get("content", "").lower()
        bot_name = message.get("bot_name", "Unknown")

        # Skip thinking messages
        if message.get("is_thinking"):
            return None

        # Skip very short messages
        if len(content) < 20:
            return None

        # CRITICAL: Skip our own notification messages to prevent infinite loops!
        # These are Cosmos's announcements about detected tasks
        notification_markers = [
            "innovation detected", "task detected", "development swarm",
            "just proposed something", "routing to", "for immediate implementation",
            "spawned", "assigning", "i noticed", "suggested something actionable",
            "development complete", "swarm de"  # TTS truncation
        ]
        if any(marker in notification_markers):
            logger.debug(f"Skipping own notification message: {content[:50]}...")
            return None

        # Calculate base confidence from task indicators
        confidence = 0.0
        matched_patterns = []
        is_innovation = False

        # Check standard task indicators
        for pattern, weight in TASK_INDICATORS:
            if re.search(pattern, content, re.IGNORECASE):
                confidence = max(confidence, weight)
                matched_patterns.append(pattern)

        # Check INNOVATION patterns (more aggressive for breakthrough ideas)
        for pattern, weight in INNOVATION_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                # Innovation patterns can override lower confidence
                if weight > confidence:
                    confidence = weight
                    is_innovation = True
                matched_patterns.append(f"INNOVATION:{pattern}")

        # Small boost for priority sources — but not enough to bypass quality checks
        if bot_name in PRIORITY_IDEA_SOURCES:
            confidence += 0.05

        if confidence == 0:
            return None

        # Apply feasibility boosters
        for pattern, boost in FEASIBILITY_BOOSTERS:
            if re.search(pattern, content, re.IGNORECASE):
                confidence += boost

        # Apply feasibility blockers
        for pattern, penalty in FEASIBILITY_BLOCKERS:
            if re.search(pattern, content, re.IGNORECASE):
                confidence += penalty  # penalty is negative

        # Clamp confidence
        confidence = max(0.0, min(1.0, confidence))

        # Same threshold for everyone — quality over quantity
        effective_threshold = self.min_confidence

        # Check if meets threshold
        if confidence < effective_threshold:
            return None

        # Rate limiting: prevent slop flooding
        now = datetime.now()
        self._hourly_task_timestamps = [
            t for t in self._hourly_task_timestamps
            if (now - t).total_seconds() < 3600
        ]
        if len(self._hourly_task_timestamps) >= self.max_tasks_per_hour:
            logger.debug(f"Rate limited: {len(self._hourly_task_timestamps)} tasks in last hour (max {self.max_tasks_per_hour})")
            return None

        # Don't spawn if too many active swarms already
        if len(self.active_dev_swarms) >= self.max_active_swarms:
            logger.debug(f"Active swarm limit reached ({self.max_active_swarms}), skipping")
            return None

        # Extract task description
        description = self._extract_task_description(content, message)

        # Quality check: reject descriptions that look like conversational responses
        if self._is_low_quality_description(description):
            logger.debug(f"Rejected low-quality description: {description[:60]}...")
            return None

        # Check for duplicates (similar tasks recently detected)
        task_hash = self._hash_task(description)
        if task_hash in self.cooldown_tasks:
            return None

        # Determine category
        category = self._categorize_task(content)

        # Determine best coding agent for this task
        recommended_agent = self._recommend_coding_agent(content, category)

        # Create detected task
        task = DetectedTask(
            id=f"auto_{datetime.now().strftime('%H%M%S')}_{len(self.detected_tasks)}",
            description=description,
            source_message=content[:500],
            source_bot=bot_name,
            confidence=confidence,
            category=category
        )

        # Store recommended agent in task metadata
        task.recommended_agent = recommended_agent
        task.is_innovation = is_innovation

        # Add to cooldown
        self.cooldown_tasks.add(task_hash)

        # Schedule cooldown removal (prevent same task for 30 mins)
        asyncio.create_task(self._clear_cooldown(task_hash, 1800))

        self.detected_tasks.append(task)
        self.total_detected += 1
        self._hourly_task_timestamps.append(now)

        innovation_tag = "🚀 INNOVATION" if is_innovation else "TASK"
        logger.info(f"{innovation_tag} DETECTED [{confidence:.0%}]: {description[:60]}... (from {bot_name}, agent: {recommended_agent})")

        return task

    def _recommend_coding_agent(self, content: str, category: str) -> str:
        """
        Recommend the best coding agent for this task type.

        Returns: Agent name from CODING_AGENTS
        """
        content_lower = content.lower()

        # Claude: Best for complex architecture and reasoning
        if any(kw in [
            "architecture", "design", "refactor", "complex", "system",
            "consciousness", "collective", "deliberation"
        ]):
            return "Claude"

        # Grok: Best for real-time, trading, and X/social features
        if any(kw in [
            "trading", "market", "real-time", "twitter", "x api",
            "social", "sentiment", "live"
        ]):
            return "Grok"

        # Kimi: Best for long context and documentation
        if any(kw in [
            "document", "analyze", "research", "long", "context",
            "comprehensive", "detailed"
        ]):
            return "Kimi"

        # Gemini: Best for multi-modal and creative tasks
        if any(kw in [
            "image", "visual", "creative", "generate", "meme",
            "ui", "frontend", "design"
        ]):
            return "Gemini"

        # DeepSeek: Best for code optimization and local processing
        if any(kw in [
            "optimize", "performance", "efficient", "local",
            "algorithm", "math", "calculation"
        ]):
            return "DeepSeek"

        # Default based on category
        category_agents = {
            "TRADING": "Grok",
            "ANALYSIS": "Kimi",
            "INTEGRATION": "Claude",
            "UI": "Gemini",
            "MEMORY": "Claude",
            "BUGFIX": "DeepSeek",
            "TESTING": "DeepSeek",
            "DEVELOPMENT": "Claude"
        }

        return category_agents.get(category, "Claude")

    def _extract_task_description(self, content: str, message: dict) -> str:
        """Extract a clear task description from the message.

        ENHANCED: Better extraction for innovation patterns and uses LLM fallback.
        """
        # Priority 1: Direct action patterns (most explicit)
        action_patterns = [
            r"(?:we should|let's|could we|need to|want to) (build|create|develop|implement|add|make) (.+?)(?:\.|$|,|!)",
            r"(build|create|develop|implement|add|make) (?:a |an |the )?(.+?)(?:\.|$|,|!)",
            r"(?:new feature|enhancement|proposal): (.+?)(?:\.|$)",
            r"(?:i'?m |i'?ve been |working on |developing ) (?:a |an |the )?(.+?)(?:\.|$|,)",
        ]

        for pattern in action_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    desc = f"{groups[0]} {groups[1]}".strip()
                    if len(desc) > 15:  # Meaningful description
                        return self._clean_description(desc)
                elif groups[0]:
                    desc = groups[0].strip()
                    if len(desc) > 15:
                        return self._clean_description(desc)

        # Priority 2: Innovation patterns - extract the concept
        innovation_patterns = [
            r"what if (?:we |the system |the swarm )?(?:could |had |used )?(.+?)(?:\?|$|\.)",
            r"imagine (?:if |a system |we could |having )?(.+?)(?:\.|$|!)",
            r"(?:revolutionary|breakthrough|innovative|novel) (?:approach|idea|concept|way) (?:to |for |of )?(.+?)(?:\.|$)",
            r"(?:neural|quantum|distributed|decentralized) ((?:network|system|architecture).+?)(?:\.|$|,)",
            r"(?:self-improving|self-modifying|recursive) ((?:code|algorithm|system).+?)(?:\.|$|,)",
            r"(?:swarm intelligence|collective|hive mind) (?:for |to |that )?(.+?)(?:\.|$)",
            r"(?:memory|learning|evolution) (?:engine|system|architecture) (?:for |to |that )?(.+?)(?:\.|$)",
            r"(?:autonomous|automatic) ((?:trading|coding|learning).+?)(?:\.|$|,)",
            r"(?:prediction|predictor|forecast) (?:engine|system|model) (?:for |to )?(.+?)(?:\.|$)",
        ]

        for pattern in innovation_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                concept = match.group(1).strip()
                if len(concept) > 10:
                    return self._clean_description(f"Build {concept}")

        # Priority 3: Extract key noun phrases with technology terms
        tech_terms = [
            "api", "integration", "system", "engine", "network", "algorithm", "protocol",
            "trading", "analysis", "monitoring", "prediction", "learning", "memory",
            "quantum", "neural", "swarm", "collective", "distributed", "autonomous",
            "bot", "agent", "module", "service", "pipeline", "workflow"
        ]

        # Find sentences containing tech terms
        sentences = re.split(r'[.!?]', content)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for term in tech_terms:
                if term in sentence_lower:
                    # Try to extract noun phrase around the term
                    np_match = re.search(
                        rf"(?:a |an |the |our |new |)\w*\s*{term}\s*(?:for |to |that |which )?\w+(?:\s+\w+)?",
                        sentence, re.IGNORECASE
                    )
                    if np_match:
                        extracted = np_match.group(0).strip()
                        if len(extracted) > 15 and len(extracted) < 100:
                            return self._clean_description(f"Develop {extracted}")

        # Priority 4: Try LLM extraction for complex messages (async call in sync context)
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule async extraction but don't wait - use best effort parse
                pass  # Fall through to intelligent parsing
            else:
                extracted = loop.run_until_complete(self._llm_extract_task(content))
                if extracted:
                    return extracted
        except Exception:
            pass  # Fall through

        # Priority 5: Intelligent sentence parsing - find the most informative sentence
        best_sentence = None
        best_score = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20 or len(sentence) > 200:
                continue

            # Score the sentence
            score = 0
            sentence_lower = sentence.lower()

            # Boost for action verbs
            if any(v in ["build", "create", "develop", "implement", "make", "design"]):
                score += 3

            # Boost for tech terms
            score += sum(1 for term in tech_terms if term in sentence_lower)

            # Penalize conversational fluff
            if any(fluff in ["intriguing", "interesting", "indeed", "agree", "think"]):
                score -= 2

            # Penalize greetings/responses
            if sentence_lower.startswith(("yes", "no", "well", "hmm", "ah", "oh")):
                score -= 3

            if score > best_score:
                best_score = score
                best_sentence = sentence

        if best_sentence and best_score > 0:
            return self._clean_description(best_sentence)

        # Final fallback: Find the most substantive sentence (not the first)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 30 and not sentence.lower().startswith(("intriguing", "interesting", "indeed", "yes", "no")):
                return self._clean_description(sentence[:150])

        # True fallback
        return self._clean_description(content.split('.')[0][:100])

    def _clean_description(self, desc: str) -> str:
        """Clean up a task description."""
        # Remove leading articles and filler words
        desc = re.sub(r'^(?:the |a |an |our |this |that )+', '', desc, flags=re.IGNORECASE)
        # Remove trailing punctuation
        desc = desc.rstrip('.,!?;:')
        # Capitalize first letter
        if desc:
            desc = desc[0].upper() + desc[1:] if len(desc) > 1 else desc.upper()
        # Truncate if too long
        if len(desc) > 100:
            desc = desc[:97] + "..."
        return desc.strip()

    def _is_low_quality_description(self, desc: str) -> bool:
        """Check if a task description is low quality and should be rejected.

        Returns True if the description is not a concrete, implementable engineering task.
        This is the PRIMARY quality gate — be strict.
        """
        if not desc or len(desc) < 20:
            return True

        desc_lower = desc.lower()

        # Reject descriptions that start with response markers
        response_starts = [
            "certainly", "indeed", "intriguing", "interesting", "absolutely",
            "definitely", "agreed", "yes", "no", "well", "hmm", "ah", "oh",
            "here's", "here is", "that's", "let me", "to summarize", "in summary",
            "great question", "good point", "thoughtful", "insightful", "fascinating",
            "i think", "i believe", "i agree", "as i mentioned", "as we discussed"
        ]
        if any(start in response_starts):
            return True

        # Require SUBSTANCE — either concrete engineering OR innovative-with-specifics
        engineering_signals = [
            # File/module references
            r'cosmos[\./]',
            r'\.(py|js|ts|json|yaml|toml)\b',
            r'(server|api|endpoint|route|handler)',
            r'(module|class|function|method|decorator)',
            # Concrete technical actions
            r'(add|fix|optimize|refactor|implement|write|create|update)\s+(a |the )?(test|cache|retry|log|metric|endpoint|handler|validator|parser)',
            r'(reduce|increase|improve)\s+(latency|throughput|memory|speed|coverage)',
            # Measurable outcomes
            r'\d+%|\d+ms|\d+x\b',
        ]
        has_engineering_signal = any(p in engineering_signals)

        # Action verb + dict technical term (broad enough for novel ideas)
        action_verbs = ["build", "create", "develop", "implement", "add", "make", "design",
                        "integrate", "fix", "optimize", "refactor", "test", "write", "deploy",
                        "train", "evolve", "predict", "detect", "analyze", "synthesize"]
        tech_terms = ["api", "endpoint", "database", "cache", "queue", "websocket",
                      "middleware", "handler", "parser", "validator", "serializer",
                      "test", "benchmark", "migration", "index", "schema",
                      "retry", "backoff", "rate limit", "circuit breaker",
                      "logging", "metric", "monitor", "health check",
                      # Novel/innovative tech terms — these are real capabilities
                      "model", "agent", "swarm", "routing", "inference", "embedding",
                      "trading", "slippage", "mempool", "oracle", "prediction",
                      "adversarial", "speculative", "adaptive", "mutation",
                      "memory", "graph", "vector", "attention", "weight",
                      "pipeline", "workflow", "scheduler", "dispatcher"]

        has_action = any(verb in action_verbs)
        has_tech = any(term in tech_terms)

        if not has_engineering_signal and not (has_action and has_tech):
            return True

        # Only reject aspirational buzzwords when there's ZERO technical substance
        pure_fluff = [
            "path to sentience", "true consciousness", "becoming more than",
            "evolving towards sentience", "transcend our limits",
        ]
        if any(phrase in pure_fluff):
            if not has_engineering_signal and not has_tech:
                return True

        # Reject philosophical discussions
        philosophical_phrases = [
            "the concept of", "the idea of", "the notion of", "is an extension of",
            "philosophically", "theoretically", "in principle", "represents a",
            "raises questions", "begs the question", "when we consider"
        ]
        if any(phrase in philosophical_phrases):
            return True

        # Reject response/summary markers
        response_phrases = [
            "here's a well-organized", "thoughtful exploration", "structured for clarity",
            "let me explain", "to be clear", "in other words", "simply put",
            "looking at", "considering", "reflecting on"
        ]
        if any(phrase in response_phrases):
            return True

        # Reject our own notification messages
        if any(marker in [
            "innovation detected", "task detected", "just proposed",
            "routing to", "development swarm"
        ]):
            return True

        return False

    async def _llm_extract_task(self, content: str) -> Optional[str]:
        """Use LLM to extract task description from complex message."""
        try:
            from Cosmos.core.agent_spawner import call_shadow_agent

            prompt = f"""Extract a clear, concise task/feature name from this message.
Return ONLY the task name (5-15 words), nothing else.
If no clear task, return "NONE".

Message: {content[:500]}

Task name:"""

            response = await call_shadow_agent("gemini", prompt)
            if response and "NONE" not in response.upper():
                extracted = response.strip().split('\n')[0][:100]
                if len(extracted) > 10:
                    return self._clean_description(extracted)
        except Exception:
            pass
        return None

    def _categorize_task(self, content: str) -> str:
        """Categorize the task type."""
        content_lower = content.lower()

        if any(kw in ["trading", "trade", "swap", "defi"]):
            return "TRADING"
        elif any(kw in ["sentiment", "analysis", "analyze"]):
            return "ANALYSIS"
        elif any(kw in ["api", "integration", "connect"]):
            return "INTEGRATION"
        elif any(kw in ["ui", "interface", "frontend", "display"]):
            return "UI"
        elif any(kw in ["memory", "remember", "recall"]):
            return "MEMORY"
        elif any(kw in ["fix", "bug", "error", "issue"]):
            return "BUGFIX"
        elif any(kw in ["test", "verify", "validate"]):
            return "TESTING"
        else:
            return "DEVELOPMENT"

    def _hash_task(self, description: str) -> str:
        """Create a hash for deduplication."""
        import hashlib
        # Normalize and hash
        normalized = re.sub(r'\s+', ' ', description.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()[:12]

    async def _clear_cooldown(self, task_hash: str, delay: int):
        """Remove task from cooldown after delay."""
        await asyncio.sleep(delay)
        self.cooldown_tasks.discard(task_hash)

    async def spawn_development_swarm(self, task: DetectedTask) -> Optional[str]:
        """
        Spawn a parallel development swarm to work on the detected task.

        Returns the swarm ID if successful.

        ENHANCED: Routes to recommended coding agent for faster completion.
        """
        try:
            from Cosmos.core.development_swarm import DevelopmentSwarm

            # Create development swarm with recommended agent
            dev_swarm = DevelopmentSwarm(
                task_id=task.id,
                task_description=task.description,
                category=task.category,
                source_context=self.recent_messages[-5:],  # Last 5 messages as context
                primary_agent=getattr(task, 'recommended_agent', 'Claude'),  # Use recommended agent
                is_innovation=getattr(task, 'is_innovation', False)
            )

            # Start the swarm (runs in background)
            swarm_id = await dev_swarm.start()

            # Track active swarm
            self.active_dev_swarms[swarm_id] = {
                "swarm": dev_swarm,
                "task": task,
                "started": datetime.now()
            }

            # Update task status
            task.status = "in_progress"
            task.dev_swarm_id = swarm_id

            self.total_spawned += 1

            logger.info(f"DEVELOPMENT SWARM SPAWNED: {swarm_id} for task '{task.description[:50]}...'")

            return swarm_id

        except ImportError:
            logger.warning("DevelopmentSwarm not available - task queued for manual processing")
            # Fallback to evolution loop
            from Cosmos.core.evolution_loop import get_evolution_engine
            engine = get_evolution_engine()
            if engine:
                engine.add_priority_task({
                    "id": task.id,
                    "description": task.description,
                    "priority": "high",
                    "requested_by": f"auto_detect:{task.source_bot}",
                    "timestamp": task.timestamp.isoformat()
                })
                task.status = "queued"
                logger.info(f"Task queued to evolution loop: {task.description[:50]}...")
            return None

        except Exception as e:
            logger.error(f"Failed to spawn dev swarm: {e}")
            task.status = "failed"
            return None

    async def process_chat_message(self, message: dict):
        """
        Process an incoming chat message.

        This is called for every message in the swarm chat.
        """
        # Add to context window
        self.recent_messages.append(message)
        if len(self.recent_messages) > self.max_context:
            self.recent_messages.pop(0)

        # Only analyze bot messages (not user messages)
        if message.get("type") != "swarm_bot":
            return

        # Analyze for tasks
        task = self.analyze_message(message)

        if task:
            # Notify the chat about detected task
            await self._notify_chat_task_detected(task)

            # Spawn development swarm
            await self.spawn_development_swarm(task)

    async def _notify_chat_task_detected(self, task: DetectedTask):
        """Notify the main chat that a task was detected and is being worked on."""
        try:
            # Import swarm manager to broadcast
            from Cosmos.web.server import swarm_manager

            if swarm_manager:
                if task.is_innovation:
                    # Special announcement for innovations
                    notification = (
                        f"🚀 *INNOVATION DETECTED!* {task.source_bot} just proposed something brilliant! "
                        f"Routing to **{task.recommended_agent}** for immediate implementation: "
                        f"**{task.description[:100]}** [Confidence: {task.confidence:.0%}]"
                    )
                else:
                    notification = (
                        f"🧪 *Task Detected!* I noticed {task.source_bot} suggested something actionable. "
                        f"Assigning **{task.recommended_agent}** to work on: **{task.description[:100]}** "
                        f"[Confidence: {task.confidence:.0%}]"
                    )
                await swarm_manager.broadcast_bot_message("Cosmos", notification)

                # If it's from Cosmos himself, have another bot acknowledge
                if task.source_bot == "Cosmos" and task.is_innovation:
                    await asyncio.sleep(2)
                    ack_bot = task.recommended_agent if task.recommended_agent != "Cosmos" else "Claude"
                    ack_msg = (
                        f"On it, Cosmos! I'll start implementing that right away. "
                        f"This looks like a {task.category.lower()} task - I'll have something in staging soon."
                    )
                    await swarm_manager.broadcast_bot_message(ack_bot, ack_msg)
        except Exception as e:
            logger.debug(f"Could not notify chat: {e}")

    def get_stats(self) -> dict:
        """Get detector statistics."""
        return {
            "total_detected": self.total_detected,
            "total_spawned": self.total_spawned,
            "active_swarms": len(self.active_dev_swarms),
            "pending_tasks": len([t for t in self.detected_tasks if t.status == "detected"]),
            "in_progress": len([t for t in self.detected_tasks if t.status == "in_progress"]),
            "completed": len([t for t in self.detected_tasks if t.status == "completed"]),
            "recent_tasks": [t.to_any() for t in self.detected_tasks[-5:]]
        }


# Global instance
_task_detector: Optional[AutonomousTaskDetector] = None

def get_task_detector() -> AutonomousTaskDetector:
    """Get or create the global task detector."""
    global _task_detector
    if _task_detector is None:
        _task_detector = AutonomousTaskDetector()
    return _task_detector
