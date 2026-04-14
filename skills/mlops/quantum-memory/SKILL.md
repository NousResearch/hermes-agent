---
name: quantum-memory
description: Quantum-optimized memory retrieval for AI agents. Knowledge graphs + QAOA find connected memory clusters instead of individual matches. Use when building agent memory, upgrading from flat similarity search (Mem0, LangChain), or needing relationship-aware recall with recency boost. Benchmarked on LongMemEval (ICLR 2025).
version: 0.4.0
author: Coinkong (Chef's Attraction AI Lab)
license: MIT
metadata:
  hermes:
    tags: [MLOps, Memory, Quantum, Agents, Knowledge-Graph, QAOA]
    related_skills: []
---

# Quantum Memory Graph

Relationship-aware memory for AI agents. Knowledge graphs + quantum-optimized subgraph selection (QAOA).

## When to Use

- Building or upgrading an AI agent's memory system
- Replacing flat similarity search (Mem0, LangChain memory, raw vector DB)
- Need memories that work *together* as connected context, not isolated matches
- Want recency-aware retrieval (recent memories rank higher)

## Quick Reference

| Action | Command |
|--------|---------|
| Install | `pip install quantum-memory-graph` |
| Store memory | `store("text content")` |
| Recall | `recall("query", K=5)` |
| High accuracy model | `MemoryGraph(model="thenlper/gte-large")` |
| Run API server | `python -m quantum_memory_graph.api --port 8502` |

## Procedure

### 1. Install

```bash
pip install quantum-memory-graph
```

### 2. Store and Recall

```python
from quantum_memory_graph import store, recall

store("Project Alpha uses React frontend with TypeScript.")
store("Project Alpha backend is FastAPI with PostgreSQL.")
store("FastAPI connects to PostgreSQL via SQLAlchemy ORM.")

result = recall("What is Project Alpha's full tech stack?", K=4)
for memory in result["memories"]:
    print(f"  {memory['text']}")
```

### 3. Choose a Model

Default works everywhere (90MB, no GPU). For maximum accuracy:

```python
from quantum_memory_graph import MemoryGraph
mg = MemoryGraph(model="thenlper/gte-large")  # 96.6% R@5, needs GPU
```

See `references/models.md` for full comparison.

### 4. Short-Term Memory (automatic)

Recency boost is ON by default. Recent memories score higher.

```python
from quantum_memory_graph import get_stm
stm = get_stm()
stm.conversation.add_turn("What are preferences?", memory_ids=["m1"])
```

### 5. Deploy as Microservice

```bash
pip install quantum-memory-graph[api]
python -m quantum_memory_graph.api --port 8502
```

See `references/deployment.md` for migration and production tips.

## Pitfalls

- `gte-large` needs ~2GB RAM. Use default `MiniLM` on low-memory machines.
- First `store()` call downloads the embedding model (~90MB default, ~1.3GB for large models).
- QAOA optimization adds ~5ms per recall vs pure Top-K. Worth it for combination quality.

## Verification

```python
from quantum_memory_graph import recall
result = recall("test query")
assert result["ok"] is True
assert len(result["memories"]) > 0
```

## Links

- [PyPI](https://pypi.org/project/quantum-memory-graph/)
- [GitHub](https://github.com/Dustin-a11y/quantum-memory-graph)
- [ClawHub](https://clawhub.ai/skills/quantum-memory)
