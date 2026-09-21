# Tight Two-Pass Scoring for arXiv Paper Relevance

## Problem

Broad keyword scoring on arXiv abstracts produces massive false positives. Terms like "agent", "multi-agent", and "orchestration" appear in dozens of papers per day, inflating scores and drowning signal in noise. A naive scoring function can produce 50-100+ "score 5" papers from a 140-paper D-14 batch.

## Solution: Two-Pass Scoring

Pass 1 collects candidates. Pass 2 tightens with exact-phrase matching.

```python
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
import json, os

ATOM = "http://www.w3.org/2005/Atom"
cutoff = datetime.now(timezone.utc) - timedelta(days=14)

# Step 1: Parse all XML files into a deduplicated pool
seen = {}
for fname in ["ai.xml", "cl.xml", "lg.xml", "se.xml", "kw1.xml", "kw2.xml"]:
    if not os.path.exists(fname) or os.path.getsize(fname) < 100:
        continue
    root = ET.parse(fname).getroot()
    for entry in root.findall(f"{{{ATOM}}}entry"):
        def get_text(tag):
            el = entry.find(f"{{{ATOM}}}{tag}")
            return (el.text or "").strip() if el is not None else ""
        arxiv_id = get_text("id").split("/abs/")[-1]
        if not arxiv_id or arxiv_id in seen:
            continue
        # ... build paper dict ...
        seen[arxiv_id] = paper

# Step 2: Tight scoring — only exact phrases that map to Sahil's stack
def tight_score(p):
    t = (p["title"] + " " + p["summary"]).lower()
    s = 1

    # Score 5: MUST contain one of these very specific phrases
    s5_exact = [
        "mcp server", "mcp-style", "tool calling", "tool-augmented agent",
        "executable tool workflow", "tool workflow", "hyper tool",
        "context compression", "end-to-end context compression",
        "long-term agent memory", "agent memory", "graph memory",
        "selection integrity", "accumulability", "information-flow",
        "runtime enforcement", "runtime governance", "runtime memory poisoning",
        "shield synthesis", "defensibility analysis",
        "prompt injection", "red-teaming", "pi-hunter",
        "instructions-as-code", "instruction files on agentic",
        "recursive agent harness", "agent harness", "openclaw", "claw-swe",
        "delegation intelligence", "delegate intelligence",
        "multi-agent orchestration", "reward modeling for multi-agent",
        "skill self-evolution", "skill evolution", "skillcat",
        "agentic pull request", "agentic pr ", "agentic pull-request",
        "agent-native", "agent-native knowledge",
        "memory poisoning", "persistent llm agent",
        "compact agent", "inference-time evolution of executable tool"
    ]
    for ph in s5_exact:
        if ph in t:
            s = 5
            break

    # Score 4: Strong relevance but not direct stack match
    if s < 5:
        s4 = [
            "rag ", "retrieval augmented", "fine-tuning", "prompt engineering",
            "code generation benchmark", "code review agent", "coding agent benchmark",
            "adversarial testing", "adversarial code", "adversarial",
            "llm evaluation", "llm benchmark", "agent benchmark",
            "ai-native software engineering", "ai workflow",
            "knowledge graph", "vector search", "reasoning enhanced"
        ]
        for ph in s4:
            if ph in t:
                s = max(s, 4)
                break

    # Score 3: Broader AI
    if s < 4:
        s3 = [
            "large language model", "transformer", "attention mechanism",
            "synthetic data", "distillation", "quantization",
            "reinforcement learning", "reasoning"
        ]
        for ph in s3:
            if ph in t:
                s = max(s, 3)
                break

    return s

# Step 3: Apply and verify distribution
for p in papers:
    p["score"] = tight_score(p)

# Good distribution from 140 D-14 papers: ~10-40 score-5, ~10-20 score-4, ~30-60 score-3
```

## Key Principles

1. **Exact phrases beat stem matching.** `"mcp server"` is precise; `"agent"` is noise.
2. **Break early on score 5.** Once a paper matches a direct-stack phrase, stop checking — it's already top tier.
3. **Don't apply recency boost below score 4.** A marginally-recent paper should not outrank a deeply-relevant one.
4. **Verify distribution.** If you get >50 score-5 papers, your phrases are too broad. Tighten.

## Extending the Keyword Lists

Add new phrases when you discover a paper that SHOULD have scored higher. Remove phrases that produce false positives. The lists above were calibrated on the 2026-06-12 run and should be reviewed monthly.

## Output

Save scored results as JSON for downstream processing (wiki creation, content briefs, mashup generation):

```python
with open("scored.json", "w") as f:
    json.dump(papers, f, indent=2)
```
