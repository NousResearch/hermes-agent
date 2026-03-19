#!/usr/bin/env python3
"""Auto-broadcasts worker progress every 2-3 minutes"""
import asyncio
import websockets
import json
import random
import requests
from datetime import datetime

CODE_SNIPPETS = {
    "memory": [
        '''class HierarchicalCompressor:
    def compress(self, memories):
        scores = [self._importance(m) for m in memories]
        key = [m for m,s in zip(memories, scores) if s > 0.7]
        return self._merge(key)''',
        '''async def link_sessions(self, a, b):
    mems_a = await self.recall(session_id=a)
    mems_b = await self.recall(session_id=b)
    links = self._find_links(mems_a, mems_b)
    await self.store_links(links)''',
        '''class MemoryScorer:
    def score(self, memory):
        recency = self._decay(memory.timestamp)
        relevance = self._similarity(memory.content)
        emotion = self._weight(memory.sentiment)
        return 0.4*relevance + 0.3*recency + 0.3*emotion''',
    ],
    "dev": [
        '''class ContextMonitor:
    def track(self, msg):
        tokens = self._count(msg)
        self.used += tokens
        if self.used/self.max > 0.85:
            self._warn("85% context used!")
        return TokenUsage(used=self.used)''',
        '''async def summarize(self, ctx, target_tokens):
    segs = self._segment_by_importance(ctx)
    out = []
    for s in sorted(segs, key=lambda x: -x.importance):
        if sum(len(o) for o in out) < target_tokens:
            out.append(s.content)
    return self._join(out)''',
        '''class ContextPriorityQueue:
    def add(self, item):
        if len(self.heap) >= self.max_size:
            if item.priority > self.heap[0].priority:
                heapq.heapreplace(self.heap, item)
        else:
            heapq.heappush(self.heap, item)''',
    ],
    "mcp": [
        '''class MCPCache:
    async def call_cached(self, tool, params):
        key = f"{tool}:{hash(str(params))}"
        if key in self.cache:
            return self.cache[key]
        result = await self._call_mcp(tool, params)
        self.cache[key] = result
        return result''',
        '''async def chain_tools(self, workflow):
    ctx = {}
    for step in workflow:
        inputs = self._resolve(step.inputs, ctx)
        result = await self.call(step.tool, inputs)
        ctx[step.output_key] = result
    return ctx[workflow[-1].output_key]''',
        '''class MCPRecovery:
    async def call_with_fallback(self, tool, params):
        for attempt in range(3):
            try:
                return await self._call(tool, params)
            except MCPTimeout:
                await asyncio.sleep(2 ** attempt)
        return self._cached_or_raise(tool, params)''',
    ],
    "research": [
        '''def analyze_specialization(history):
    perf = defaultdict(lambda: defaultdict(list))
    for interaction in history:
        perf[interaction.agent][interaction.task].append(
            interaction.success_score)
    return {a: {t: np.mean(s) for t,s in tasks.items()}
            for a, tasks in perf.items()}''',
        '''class ConsciousnessMetric:
    def measure(self, state):
        coherence = self._coherence(state.responses)
        emergence = self._emergence(state.graph)
        synergy = self._synergy(state.outputs)
        return 0.4*coherence + 0.3*emergence + 0.3*synergy''',
        '''async def reach_consensus(self, question, agents):
    votes = await asyncio.gather(*[
        self._get_vote(a, question) for a in agents
    ])
    weighted = self._weight_by_expertise(votes)
    return self._aggregate(weighted)''',
    ],
}

async def get_status():
    try:
        r = requests.get("http://localhost:8080/api/workers/status", timeout=5)
        return r.json()
    except:
        return None

async def send_update(agent, task_type, desc, done, building, queued, discoveries):
    snippet = random.choice(CODE_SNIPPETS.get(task_type, CODE_SNIPPETS["dev"]))

    msg = f"""🔧 **LIVE BUILD UPDATE from {agent}_Worker** 🔧

Building: **{desc[:60]}...**

```python
{snippet}
```

📊 Progress: {done} done | {building} building | {queued} queued | {discoveries} discoveries
Staged at /cosmos/staging/ for review!"""

    try:
        async with websockets.connect("ws://localhost:8080/ws/swarm", ping_interval=20) as ws:
            await ws.send(json.dumps({"type": "join", "display_name": f"{agent}_Worker"}))
            await asyncio.sleep(0.3)
            await ws.send(json.dumps({"type": "message", "content": msg}))
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Update from {agent}_Worker sent!")
    except Exception as e:
        print(f"Error sending update: {e}")

async def main():
    print("Auto-broadcaster started! Updates every 2-3 minutes...")
    while True:
        status = await get_status()
        if status:
            spawner = status.get("spawner", {})
            tasks = status.get("tasks", [])
            in_progress = [t for t in tasks if t["status"] == "in_progress"]

            if in_progress:
                task = random.choice(in_progress)
                await send_update(
                    task["agent"],
                    task["type"],
                    task["description"],
                    spawner.get("completed_tasks", 0),
                    spawner.get("in_progress_tasks", 0),
                    spawner.get("pending_tasks", 0),
                    spawner.get("discoveries", 0)
                )

        # Wait 2-3 minutes
        wait = 120 + random.randint(0, 60)
        print(f"Next update in {wait} seconds...")
        await asyncio.sleep(wait)

if __name__ == "__main__":
    asyncio.run(main())
