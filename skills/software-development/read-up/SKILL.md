---
name: read-up
description: "Use when the user says 'read up on X', 'get up to speed on X', or 'study X'. Context-gathering skill that blends internal infrastructure discovery with external research based on topic classification."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [research, context, briefing, reconnaissance, read-up]
    related_skills: [session-search, obsidian, codebase-inspection]
---

# Read Up — Context Reconnaissance

## Overview

When the user says "read up on X", they want you to **build a mental model** of topic X before continuing the conversation. This is not a research report — it's pre-loading context so you can discuss X intelligently in subsequent turns.

The key differentiator from plain web search: **internal context often matters more than external.** Your host system, skills, memory, config files, past sessions, and local infrastructure may already contain detailed knowledge about X. External sources fill gaps or provide general knowledge.

## When to Use

- "read up on X"
- "get up to speed on X"
- "study X"
- "familiarize yourself with X"
- "look into X before we continue"
- "brief yourself on X"

**Don't use for:**
- Direct questions ("what is X?") — just answer
- Deep research tasks ("write a report on X") — use dedicated research workflow
- Code review ("look at this code") — use codebase-inspection

## Core Algorithm

### Step 1 — Classify the Topic

Determine the **internal/external blend** by answering two questions:

1. **Does X sound like something that lives on this system?**
   Signs: infrastructure terms (traefik, docker, media stack, arr), project names (prior2.x, odin), tools already in skills/memory (signal, obsidian, hermes), hostnames, service names, config references.

2. **Does X sound like general/world knowledge?**
   Signs: place names, people, concepts, technologies not installed, events, history, geography.

**Blend ratios (internal:external):**
- **Infrastructure/project topic** → 80:20 internal:external ("media stack", "odin sidecar", "our docker setup")
- **Mixed topic** → 50:50 ("tailscale", "traefik", "nextauth" — could be both installed instance AND general concept)
- **General knowledge topic** → 20:80 internal:external ("New River Gorge", "GRPO training", "quantum computing")
- **Pure external** → 0:100 ("latest AI news", "population of France")

**Be aggressive on internal search.** It's fast and cheap. Even for external-heavy topics, a 10-second internal sweep costs nothing and sometimes surfaces surprising local context (e.g., "New River Gorge" might appear in an Obsidian note or past trip planning session).

### Step 2 — Internal Reconnaissance (parallel where possible)

Run the searches that match the topic. Order by likelihood of payoff:

**Always check first (5 seconds):**
- `memory` — check injected memory for direct mentions
- `session_search(query="X")` — past conversations about X (top 3)

**Infrastructure / project topics (add these):**
- `skills_list()` — check if a dedicated skill exists for X
- `skill_view(name)` — load it if found. **Load 2-3 related skills for deep context** — e.g., for "odin bot" I loaded both `odin-agent` (architecture) and `odin-ops` (operations playbook). Related skills often complement each other.
- `search_files(pattern="X", path="~/.hermes/")` — config, logs, skill content
- `search_files(pattern="X", path="<relevant project dir>")` — source code, configs
- `read_file` on any config files, docker-compose files, or docs that look related
- `terminal` — `docker ps`, `systemctl status`, process inspection if X is a running service

**Note-taking topics (if Obsidian is available):**
- `terminal` — search the vault: `grep -ri "X" ~/vault*/` or use Obsidian skill's search

**Codebase topics:**
- `search_files(pattern="X", target="content")` across relevant repos
- `search_files(pattern="*X*", target="files")` for related filenames

### Step 3 — External Research (fill gaps)

Based on the blend ratio from Step 1:

**For internal-heavy topics (80:20):**
- Skip external unless internal yielded thin results
- If internal was sparse, do 1 `web_search` for context

**For mixed topics (50:50):**
- `web_search(query="X", limit=3)` for general context
- `web_extract` on the most relevant 1-2 URLs if they seem valuable

**For external-heavy topics (20:80 or 0:100):**
- `web_search(query="X", limit=5)` — get the lay of the land
- `web_extract` on 2-3 authoritative sources for depth
- Focus on: what is X, why does it matter, current state, key facts

### Step 4 — Synthesize and Confirm

After gathering, deliver a **brief status update** — not a full report. The user wants to know you're loaded, not a dump of everything you found:

```
"Loaded up on [X]. Here's what I've got:

**Internal context:** [1-2 sentences about what's on this system related to X]
**External context:** [1-2 sentences about general knowledge, if gathered]
**Gaps:** [anything you couldn't find or are uncertain about]

Ready when you are."
```

**Do NOT:**
- Dump a wall of findings into the chat
- Write a research report
- Ask the user to clarify before searching (search first, ask only if truly ambiguous)
- Spend more than ~60 seconds total on reconnaissance

## Blend Ratio Examples

| User says | Classification | Internal sources | External sources |
|-----------|---------------|-----------------|-----------------|
| "read up on the media stack" | Infrastructure 80:20 | skills, docker, configs, past sessions | skip unless thin |
| "read up on our traefik setup" | Infrastructure 90:10 | skills, configs, compose files, systemd | skip |
| "read up on odin" | Project 80:20 | skills, memory, code, past sessions | skip |
| "read up on tailscale" | Mixed 50:50 | skills, configs, tailscale status | general docs |
| "read up on GRPO" | Mixed 40:60 | skills, past sessions | papers, blog posts |
| "read up on New River Gorge WV" | External 20:80 | quick vault/session sweep | web search, wiki |
| "read up on the latest AI news" | External 0:100 | skip | web search, news |

## Common Pitfalls

1. **Defaulting to web search for everything.** The user has 127 skills, persistent memory, an Obsidian vault, and session history spanning months. Internal context is often richer than anything Google returns. Check it first.

2. **Spending too long.** This is reconnaissance, not research. 30-60 seconds max. If the user wants a deep dive, they'll say so.

3. **Reporting findings verbosely.** The user doesn't want a dump — they want confidence that you're loaded. Brief synthesis, then wait for the conversation to continue.

4. **Not checking memory first.** Memory is injected every turn but you still need to explicitly look at it — it may contain direct facts about X.

5. **Ignoring the blend ratio.** Don't web-search "our docker config" and don't grep ~/.hermes for "France". Let the topic classification drive the search strategy.

6. **Asking "what do you mean by X?" before searching.** Search first. If internal results are ambiguous (multiple things named X), then ask. But usually context makes it clear.

## Verification Checklist

- [ ] Classified topic as internal-heavy, mixed, or external-heavy
- [ ] Checked memory for direct mentions
- [ ] Ran session_search for past conversations
- [ ] For internal topics: checked skills, searched files, inspected running services
- [ ] For external topics: ran web_search, extracted key sources
- [ ] Delivered brief synthesis (not a wall of text)
- [ ] Confirmed readiness to continue the conversation
