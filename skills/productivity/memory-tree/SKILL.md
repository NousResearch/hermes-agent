---
name: memory-tree
description: Build hierarchical memory trees from session knowledge, compress into ≤3K-token markdown chunks scored by relevance, and optionally sync to Obsidian vault. Use this after long or complex sessions, when context is >50% full, when the user says "summarize this", "save to memory", "build knowledge tree", "organize what we learned", or "sync to Obsidian". Also use proactively after sessions with 10+ tool calls or 5+ distinct topics.
---

# Memory Tree — Hierarchical Knowledge Compressor

Inspired by openhuman's Memory Tree architecture: take flat session knowledge → organize hierarchically → compress to ≤3K chunks → scored → synced.

## Three-Layer Architecture

```
Layer 3: Root (≤3K tokens) — top-level knowledge map
  ├── Layer 2: Topics (≤1K tokens each) — grouped by domain
  │   ├── Topic A: "Project X architecture"
  │   ├── Topic B: "Tool Y quirks"
  │   └── Topic C: "User preferences"
  └── Layer 1: Raw facts (≤500 tokens each) — atomic observations
      ├── Fact 1: "Project uses pytest with xdist"
      ├── Fact 2: "API endpoint changed from v1 to v2"
      └── Fact 3: "User prefers dark theme"
```

## When to Build a Memory Tree

Build after ANY of:
- Session >30 messages or >10 tool calls
- User says "summarize", "remember all this", "save progress"
- Context usage >50%
- 5+ distinct topics discussed
- Before ending a long session

## How to Build

### Step 1: Extract raw facts from session
Scan the entire conversation. For each distinct observation:
- Tool quirks discovered
- User preferences stated/corrected
- Architecture decisions made
- Bug patterns identified
- Workflow/process steps learned

Write as atomic declarative facts (not imperatives).

### Step 2: Group into topics
Cluster facts by domain (e.g., "project setup", "API behavior", "user style").

### Step 3: Score relevance
For each fact, assign 1-5 score:
- 5: User explicitly said "remember this" or corrected you
- 4: Direct user preference or critical environment fact
- 3: Useful workflow pattern or tool behavior
- 2: Context that might matter later
- 1: Interesting but not critical

### Step 4: Compress to markdown
```markdown
---
title: "Session Summary — [date]"
topics: [setup, api, preferences]
totalFacts: 12
---

## 🔴 Critical (Score 5)
- Fact 1
- Fact 2

## 🟡 Important (Score 3-4)
### Project Setup
- Fact 3
- Fact 4

### API Behavior
- Fact 5

## 🟢 Nice to Know (Score 1-2)
- Fact 6
```

### Step 5: Save to memory + optionally Obsidian

**To Hermes memory:**
```
memory(action='add', target='memory', content='[session-summary: topic] key facts')
```

**To Obsidian vault** (if configured):
```bash
# Write tree to Obsidian
python3 ~/.hermes/skills/memory-tree/scripts/sync_to_obsidian.py \
  --vault "/mnt/d/Syncthing/LLMWiki-Vault" \
  --output "Hermes Sessions/[date]-session-summary.md" \
  --content "$MARKDOWN"
```

## The sync script

`scripts/sync_to_obsidian.py` handles:
- Writing markdown to Obsidian vault
- Creating backlinks from existing notes
- Updating a master index `Hermes Sessions/INDEX.md`

```bash
python3 ~/.hermes/skills/memory-tree/scripts/sync_to_obsidian.py \
  --vault <path> --output <relative-path> --content <markdown>
```

## Anti-Patterns
- Don't save task progress (PR numbers, commit SHAs) — these stale in 7 days
- Don't save "completed X" — use session_search for that
- Don't use imperative phrasing ("Always do X") — write declarative ("User prefers X")
- Don't build tree for <5 facts — overhead > benefit
- Skip if user hasn't said anything new since last tree
