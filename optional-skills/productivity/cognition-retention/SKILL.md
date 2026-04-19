---
name: cognition-retention
description: >
  Track what you learn, predict what you'll forget, and get proactive review nudges.
  Connects Hermes to a Cognition server for Weibull forgetting curves calibrated to your
  personal knowledge graph. Every conversation becomes a tracked learning event.
version: 1.0.0
author: Keshav Saxena (Cognition)
license: MIT
platforms: [macos, linux, windows]
requires_toolsets: [terminal]
required_environment_variables:
  - name: COGNITION_API_KEY
    prompt: Cognition API key (cog_me_...)
    help: Get a free key at https://cognitionus.com/clo/developers/claude-code
    required_for: full functionality
  - name: COGNITION_URL
    prompt: Cognition server URL
    help: Defaults to https://www.cognitionus.com/api/integrations/claude-code/mcp
    required_for: self-hosted only
metadata:
  hermes:
    tags: [Productivity, Learning, Memory, Retention, Spaced-Repetition, Cognition, Forgetting-Curves]
    related_skills: [memento-flashcards, note-taking, research-paper-writing]
---

# Cognition Retention

Track what you learn across Hermes sessions, predict what you'll forget, and get proactive review nudges — powered by [Cognition](https://cognitionus.com)'s operator-theoretic retention model.

Unlike a notebook or flashcard system, Cognition doesn't just *store* what you learned — it models *how fast you're forgetting it* using Weibull forgetting curves calibrated to your personal review history, and tells you exactly when to revisit.

## When to Use

- **Session start** → `cognition_briefing.py` to see what's decaying and what teammates recently learned
- **After completing a task** → `cognition_log.py` to record the concept you just practiced
- **Between deep-work blocks** → `cognition_review.py` to spend 10 minutes on what's slipping
- **Weekly check-in** → `cognition_brain.py` for a full retention dashboard
- **Bulk import** → `cognition_batch.py` to ingest a vault, notebook, or reading list

## Quick Reference

| Action | Command |
|--------|---------|
| Session briefing | `python scripts/cognition_briefing.py` |
| What to review | `python scripts/cognition_review.py` |
| Log a concept | `python scripts/cognition_log.py "react-hooks" --topic "React" --weight active` |
| Batch import | `python scripts/cognition_batch.py concepts.json` |
| Brain dashboard | `python scripts/cognition_brain.py` |

## Setup

```bash
# 1. Get a free API key at https://cognitionus.com/clo/developers/claude-code
# 2. Set the key (Hermes stores it securely)
hermes config set cognition_api_key cog_me_YOUR_KEY_HERE
# 3. (Optional) Self-hosted server
hermes config set cognition_url https://your-server.com/api/integrations/claude-code/mcp
```

No pip install needed — all scripts use Python stdlib only.

## Procedure

### 1. Start a session

```bash
python scripts/cognition_briefing.py
```

```
🧠 Brain — 78% retention (Slipping)
────────────────────────────────────────
📉 Weakest concepts:
  • docker-networking  [DevOps]  42%
  • react-suspense     [React]   55%
📚 Due for review:
  • kubernetes-hpa     [DevOps]
👥 Teammate signal:
  • graphql-federation [API Design]
```

### 2. Work on something — then log it

```bash
python scripts/cognition_log.py "docker-networking" \
  --topic "DevOps" --score 0.85 --weight active \
  --excerpt "Fixed bridge network DNS resolution between containers"
# ✓ UPDATED: docker-networking — retention 85%, next review 2026-04-26
```

### 3. Check what to review next

```bash
python scripts/cognition_review.py --count 5
```

```
📚 Review now (ranked by urgency):
─────────────────────────────────────────────
  1. 🔴 react-suspense         [React]       55%  OVERDUE
  2. 🟡 kubernetes-hpa         [DevOps]      62%
  3. 🟢 python-decorators      [Python]      71%
```

### 4. Batch import

```json
[
  {"concept": "react-hooks", "topic": "React", "score": 0.8, "practice_weight": "active"},
  {"concept": "docker-compose", "topic": "DevOps", "score": 0.7}
]
```

```bash
python scripts/cognition_batch.py concepts.json
# ✓ Batch complete: 2 ingested, 2 new, 0 errors
```

## How It Works

Cognition models memory using the Weibull forgetting curve: `R(t) = exp(-(t/S)^β)` where `S` (stability) is derived from the **spectral gap** of your personal knowledge graph's Laplacian. Well-connected concepts decay slower; isolated concepts decay faster. Every learning event updates the model.

## Practice Weights

| Weight | Multiplier | When to use |
|---|---|---|
| `active` | 1.0× | Wrote code, debugged it, explained it |
| `passive` | 0.5× | Read docs, watched a tutorial |
| `reference` | 0.2× | Glanced at an index, bulk-imported |

## Pitfalls

- **Don't over-log.** One concept per distinct topic. "react-hooks" not "react-useState" + "react-useEffect" separately.
- **Score honestly.** 0.9 = could teach it. 0.7 = understood. 0.5 = struggled. Inflated scores delay helpful reviews.
- **Use `active` weight only for real practice.** Reading docs is `passive`. `active` gives 2× the stability gain.

## Verification

```bash
# 1. Log a test concept
python scripts/cognition_log.py "hermes-test" --topic "Testing" --score 0.9 --weight active
# Expected: ✓ NEW: hermes-test — retention 90%, next review <date>

# 2. Fetch your briefing
python scripts/cognition_briefing.py
# Expected: shows overall retention % and the concept you just logged

# 3. Full brain report
python scripts/cognition_brain.py
# Expected: shows "Testing" topic with 1 concept at ~90%
```

## Privacy

- Stores concept IDs, labels, topics, and short excerpts (≤4000 chars) only
- Never sends full file contents, message bodies, or credentials
- All data scoped to your API key and org
- Self-hosting supported — point `COGNITION_URL` to your own server
