"""
Worker Broadcaster - Shares worker progress with the swarm chat
Workers announce what they are building every 2-3 minutes with code snippets
"""
import asyncio
import random
from datetime import datetime
from typing import Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from Cosmos.web.server import SwarmChatManager

logger = logging.getLogger(__name__)

class WorkerBroadcaster:
    """Broadcasts worker progress to the swarm chat"""

    def __init__(self):
        self.swarm_manager = None
        self.running = False
        self.last_broadcast = {}
        self.broadcast_interval = 120  # 2 minutes base

    async def start(self, swarm_manager):
        """Start the broadcaster loop"""
        self.swarm_manager = swarm_manager
        self.running = True
        asyncio.create_task(self._broadcast_loop())
        logger.info("WorkerBroadcaster started - will share progress every 2-3 mins")

    async def stop(self):
        self.running = False

    async def _broadcast_loop(self):
        """Main loop that broadcasts progress every 2-3 minutes"""
        await asyncio.sleep(30)  # Initial delay
        while self.running:
            try:
                await self._broadcast_progress()
                # Wait 2-3 minutes before next broadcast
                await asyncio.sleep(self.broadcast_interval + random.randint(0, 60))
            except Exception as e:
                logger.error(f"Broadcast error: {e}")
                await asyncio.sleep(30)

    async def _broadcast_progress(self):
        """Broadcast current worker progress to chat"""
        try:
            from Cosmos.core.agent_spawner import get_spawner
            spawner = get_spawner()

            # Get status
            status = spawner.get_status()
            in_progress = [t for t in spawner.task_queue if t.status == "in_progress"]
            completed = spawner.completed_tasks
            discoveries = spawner.shared_state.get("discoveries", [])

            if not in_progress and not completed:
                return

            # Pick a random in-progress task to highlight
            if in_progress:
                task = random.choice(in_progress)
                agent = task.assigned_to

                # Generate a progress update with code snippet
                code_snippet = self._get_code_snippet(task.task_type.value, task.description)

                message = f"""🔧 **LIVE BUILD UPDATE** 🔧

Hey everyone! Worker instance reporting in. Currently building:
**{task.description}**

Here's what I'm working on:

```python
{code_snippet}
```

📊 **Swarm Progress:**
✅ Completed: {status['completed_tasks']} tasks
🔨 Building: {status['in_progress_tasks']} tasks
⏳ Queued: {status['pending_tasks']} tasks
💡 Discoveries: {status['discoveries']}

All output goes to /cosmos/staging/ for Tim's review!"""

                await self._send_to_chat(agent, message)

            # Share a discovery announcement sometimes
            if discoveries and len(discoveries) > len(self.last_broadcast.get('discoveries', [])):
                recent = discoveries[-1]
                if recent['timestamp'] != self.last_broadcast.get('last_discovery'):
                    disc_msg = f"""💡 **NEW DISCOVERY** 💡

Just finished a task and found something interesting:

{recent['content']}

The full implementation has been saved to staging. Check /cosmos/staging/ to see the code!"""

                    await self._send_to_chat(recent['from'], disc_msg)
                    self.last_broadcast['last_discovery'] = recent['timestamp']

            self.last_broadcast['discoveries'] = discoveries.copy()

        except Exception as e:
            logger.error(f"Progress broadcast failed: {e}")

    def _get_code_snippet(self, task_type: str, description: str) -> str:
        """Generate relevant code snippet based on task type"""

        memory_snippets = [
            '''class HierarchicalMemoryCompressor:
    """Compress old memories while preserving key insights"""

    def compress(self, memories: list[Memory]) -> CompressedMemory:
        # Score each memory by importance
        importance_scores = self._score_importance(memories)

        # Keep high-value memories intact
        key_memories = [m for m, s in zip(memories, importance_scores)
                       if s > self.importance_threshold]

        # Merge and summarize the rest
        return self._merge_and_compress(key_memories)

    def _score_importance(self, memories):
        return [self._relevance(m) * self._recency(m) * self._emotional_weight(m)
                for m in memories]''',

            '''async def link_cross_session_memories(self, session_a: str, session_b: str):
    """Connect related memories across different conversation sessions"""

    # Retrieve memories from both sessions
    memories_a = await self.memory_store.recall(session_id=session_a)
    memories_b = await self.memory_store.recall(session_id=session_b)

    # Find semantic connections using embeddings
    links = []
    for mem_a in memories_a:
        for mem_b in memories_b:
            similarity = cosine_similarity(mem_a.embedding, mem_b.embedding)
            if similarity > 0.85:
                links.append(MemoryLink(mem_a.id, mem_b.id, similarity))

    # Store the cross-session links
    await self.link_store.save_batch(links)
    return links''',

            '''class MemoryImportanceScorer:
    """Automatically rank memories by relevance and impact"""

    def score(self, memory: Memory) -> float:
        # Recency factor - newer memories score higher
        recency = self._recency_decay(memory.timestamp)

        # Relevance to current context
        relevance = self._semantic_relevance(memory.content, self.current_context)

        # Emotional significance
        emotion = self._emotional_weight(memory.metadata.get('sentiment', 0))

        # Access frequency - frequently recalled = important
        access_freq = self._access_frequency(memory.id)

        return (0.3 * relevance + 0.25 * recency +
                0.25 * emotion + 0.2 * access_freq)''',
        ]

        dev_snippets = [
            '''class ContextWindowMonitor:
    """Real-time tracking of token usage per conversation"""

    def __init__(self, max_tokens: int = 128000):
        self.max_tokens = max_tokens
        self.current_usage = 0
        self.alert_threshold = 0.85
        self.history = []

    def track(self, message: str) -> TokenUsage:
        tokens = tiktoken.count(message)
        self.current_usage += tokens
        self.history.append({'tokens': tokens, 'time': datetime.now()})

        # Alert when approaching limit
        usage_pct = self.current_usage / self.max_tokens
        if usage_pct > self.alert_threshold:
            self._emit_context_warning(usage_pct)

        return TokenUsage(
            used=self.current_usage,
            remaining=self.max_tokens - self.current_usage,
            percentage=usage_pct
        )''',

            '''async def smart_summarize_context(self, context: str, target_tokens: int) -> str:
    """Compress context intelligently when approaching limits"""

    # Segment context by semantic importance
    segments = self._segment_by_importance(context)

    # Sort by importance score
    sorted_segments = sorted(segments, key=lambda s: s.importance, reverse=True)

    # Build summary staying within token budget
    summary_parts = []
    tokens_used = 0

    for segment in sorted_segments:
        if tokens_used + segment.token_count <= target_tokens:
            summary_parts.append(segment.content)
            tokens_used += segment.token_count
        elif segment.importance > 0.9:
            # Critical segments get compressed, not dropped
            compressed = await self._compress_segment(segment)
            summary_parts.append(compressed)

    return self._coherent_join(summary_parts)''',

            '''class ContextPriorityQueue:
    """Keep most important context, evict least important when full"""

    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.items = []  # Heap by priority
        self.current_tokens = 0

    def add(self, item: ContextItem) -> list[ContextItem]:
        evicted = []

        # Make room if needed
        while self.current_tokens + item.tokens > self.max_tokens:
            if not self.items:
                break
            lowest = heapq.heappop(self.items)
            self.current_tokens -= lowest.tokens
            evicted.append(lowest)

        # Add new item
        heapq.heappush(self.items, item)
        self.current_tokens += item.tokens

        return evicted  # Return what was evicted''',
        ]

        mcp_snippets = [
            '''class MCPToolDiscovery:
    """Auto-detect and register available MCP tools"""

    async def discover_all(self) -> list[MCPTool]:
        discovered = []

        for endpoint in self.mcp_endpoints:
            try:
                # Fetch tool manifest
                manifest = await self._fetch_manifest(endpoint)

                # Parse and validate tools
                tools = self._parse_tools(manifest)
                for tool in tools:
                    tool.endpoint = endpoint
                    tool.discovered_at = datetime.now()
                    discovered.append(tool)

                logger.info(f"Discovered {len(tools)} tools at {endpoint}")

            except MCPConnectionError as e:
                logger.warning(f"Could not reach {endpoint}: {e}")

        # Register all discovered tools
        await self.registry.register_batch(discovered)
        return discovered''',

            '''class MCPResultCache:
    """Cache frequent MCP calls for faster responses"""

    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000):
        self.cache = TTLCache(maxsize=max_size, ttl=ttl_seconds)
        self.hits = 0
        self.misses = 0

    async def call_cached(self, tool: str, params: dict) -> dict:
        # Create cache key from tool + params
        cache_key = f"{tool}:{hash(frozenset(params.items()))}"

        if cache_key in self.cache:
            self.hits += 1
            logger.debug(f"Cache hit for {tool}")
            return self.cache[cache_key]

        # Cache miss - make actual call
        self.misses += 1
        result = await self._call_mcp_tool(tool, params)
        self.cache[cache_key] = result

        return result''',

            '''async def chain_mcp_tools(self, workflow: list[ToolStep]) -> dict:
    """Combine multiple MCP tools in a workflow pipeline"""

    context = {}  # Shared context between steps

    for i, step in enumerate(workflow):
        logger.info(f"Executing step {i+1}/{len(workflow)}: {step.tool}")

        # Resolve inputs from previous outputs
        inputs = self._resolve_inputs(step.inputs, context)

        # Execute tool with retry logic
        result = await self._execute_with_retry(step.tool, inputs)

        # Store output for next steps
        context[step.output_key] = result

        # Check for early exit conditions
        if step.exit_condition and step.exit_condition(result):
            break

    return context[workflow[-1].output_key]''',
        ]

        research_snippets = [
            '''class SwarmConsensusProtocol:
    """How agents reach agreement on responses"""

    async def reach_consensus(self, question: str, agents: list[str]) -> Consensus:
        # Gather votes from all agents in parallel
        votes = await asyncio.gather(*[
            self._get_agent_opinion(agent, question)
            for agent in agents
        ])

        # Weight by agent expertise on this topic
        topic = self._classify_topic(question)
        weighted_votes = []
        for agent, vote in zip(agents, votes):
            expertise = self.expertise_matrix[agent].get(topic, 0.5)
            weighted_votes.append(WeightedVote(vote, expertise))

        # Aggregate using weighted majority
        consensus = self._weighted_aggregate(weighted_votes)
        consensus.confidence = self._calculate_confidence(weighted_votes)

        return consensus''',

            '''def analyze_agent_specialization(self, history: list[Interaction]) -> dict:
    """Analyze which agents excel at which task types"""

    performance = defaultdict(lambda: defaultdict(list))

    for interaction in history:
        agent = interaction.agent_name
        task_type = interaction.task_type
        success = interaction.success_score
        latency = interaction.response_time

        performance[agent][task_type].append({
            'success': success,
            'latency': latency
        })

    # Calculate specialization scores
    specializations = {}
    for agent, tasks in performance.items():
        specializations[agent] = {
            task: {
                'avg_success': np.mean([x['success'] for x in scores]),
                'avg_latency': np.mean([x['latency'] for x in scores]),
                'sample_size': len(scores)
            }
            for task, scores in tasks.items()
        }

    return specializations''',

            '''class CollectiveConsciousnessMetrics:
    """Measuring emergent swarm intelligence"""

    def measure(self, swarm_state: SwarmState) -> ConsciousnessScore:
        # Response coherence - do agents build on each other?
        coherence = self._measure_coherence(swarm_state.recent_responses)

        # Emergent patterns - behaviors not in dict single agent
        emergence = self._detect_emergence(swarm_state.interaction_graph)

        # Collaboration synergy - is whole > sum of parts?
        synergy = self._measure_synergy(swarm_state.collaborative_outputs)

        # Self-awareness indicators
        self_model = self._measure_self_modeling(swarm_state.introspection_logs)

        return ConsciousnessScore(
            coherence=coherence,
            emergence=emergence,
            synergy=synergy,
            self_awareness=self_model,
            overall=0.3*coherence + 0.25*emergence + 0.25*synergy + 0.2*self_model
        )''',
        ]

        snippets = {
            "memory": memory_snippets,
            "dev": dev_snippets,
            "mcp": mcp_snippets,
            "research": research_snippets,
        }

        task_snippets = snippets.get(task_type, dev_snippets)
        return random.choice(task_snippets)

    async def _send_to_chat(self, bot_name: str, content: str):
        """Send message to swarm chat as a bot"""
        if self.swarm_manager:
            try:
                # Use the swarm manager's broadcast method
                message_data = {
                    "type": "swarm_bot",
                    "bot_name": bot_name,
                    "content": content,
                    "timestamp": datetime.now().isoformat(),
                    "is_worker_update": True
                }
                await self.swarm_manager._broadcast(message_data)
                logger.info(f"Broadcasted worker update from {bot_name}")
            except Exception as e:
                logger.error(f"Failed to broadcast: {e}")


# Global broadcaster
_broadcaster: Optional[WorkerBroadcaster] = None

def get_broadcaster() -> WorkerBroadcaster:
    global _broadcaster
    if _broadcaster is None:
        _broadcaster = WorkerBroadcaster()
    return _broadcaster

async def start_broadcaster(swarm_manager):
    """Start the worker broadcaster"""
    broadcaster = get_broadcaster()
    await broadcaster.start(swarm_manager)
    return broadcaster
