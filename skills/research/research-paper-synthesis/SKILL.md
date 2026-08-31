---
name: research-paper-synthesis
description: "Daily arXiv paper synthesis pipeline — fetches D-14 to D+0 papers from arXiv + Semantic Scholar + HuggingFace + Papers With Code, cross-references against KenseiAgent repos, GitRadar wiki, and product portfolio, produces research digest entries, LLM wiki pages, GitRadar mashup ideas, content creation briefs, and brain fact promotion into ~/brain/ with provenance enforcement (M6)."
version: 2.2
author: KENSEI
metadata:
  hermes:
    tags: [research, arxiv, synthesis, cross-reference, wiki, content, cron, semantic-scholar, huggingface, papers-with-code, brain, provenance]
    category: research
    related_skills: [arxiv, research-digest, llm-wiki, remii-triage, ceecee-platform-content, brand-voices]
    created_for_profile: kensei
adoption_status: permanent
---

# Research Paper Synthesis v2 — Multi-Source Cross-Reference Pipeline

Fetches recent papers (D-14 to D+0) from **4 sources** and cross-references them
against Sahil's entire ecosystem. The four sources provide complementary signals:
arXiv (raw discovery), Semantic Scholar (quality/citation filtering), HuggingFace
Daily Papers (community curation), and Papers With Code (implementation signal).

## When This Skill Activates

- Daily cron at 06:30 UK (after research-digest at 05:30)
- Ad-hoc: "scan recent papers for things I can use"
- "Cross-reference this paper against our repos"

## Pipeline Overview

```
arXiv ────────→ ┐
SemScholar ────→ ├─→ Deduplicate → Score → Filter → Synthesize → Cross-Reference → Output
HF Daily ──────→ │   (by arXiv ID)    (v2)     (≥4)       (top 5)          │
PapersWCode ───→ ┘                                                        │
                              ┌───────────────────────────────────────────┼───────────────────────┬──────────────┐
                              ▼                                           ▼                       ▼               ▼
                        Research Digest                            LLM Wiki + GitRadar       Content Briefs    Brain Facts
                        (1-3 top papers)                          (durable knowledge)      (X/LI/Blog angles)  (~/brain/)
                                                                   + raw/papers/                               (v2.2)
```

## Phase 0 — Source Overview

Four sources, each providing a different signal dimension:

| Source | Signal | Cost | Rate Limit | Priority |
|---|---|---|---|---|
| **arXiv** | Raw paper discovery (zero-day) | Free | ~1 req/3s | Primary — the input stream |
| **Semantic Scholar** | Citation counts, influence, recommendations | Free | 1 req/s (100/s with key) | Quality filter — applied to top 30 arXiv hits |
| **HuggingFace Daily Papers** | Community curation (what's trending) | Free | No strict limit | Validation — cross-ref against arXiv hits |
| **Papers With Code** | Implementation links, GitHub stars | Free | No strict limit | Implementation signal — checked for score 4-5 papers |

## Phase 1 — Fetch (multi-source)

### 1A: arXiv (primary discovery)

Search arXiv for papers in these domains, sorted by submissionDate descending,
max 30 results per query:

**Category searches:**
```
cat:cs.AI       — artificial intelligence
cat:cs.CL       — NLP, language models
cat:cs.LG       — machine learning
cat:cs.SE       — software engineering (coding agents, tools)
cat:cs.HC       — human-computer interaction (UX for AI)
```

**Keyword searches:**
```
all:"coding agent" | all:"LLM agent" | all:"MCP server" | all:"tool calling"
all:"context window" | all:"prompt engineering" | all:"agent memory"
all:"Hermes Agent" | all:"Claude Code" | all:"Codex CLI"
all:"AI workflow" | all:"agent orchestration"
all:"local LLM" | all:"model serving" | all:"fine-tuning"
all:"AI product" | all:"SaaS AI" | all:"PropTech AI"
all:"football AI" | all:"coaching AI" — lower priority, CoachOS-relevant
```

**Date filter:** Only papers with submissionDate within the last 14 days.
Parse the `<published>` field from the Atom XML.

**Execution:** Use the `arxiv` skill's API commands. Fetch in batches (one
per category + 2 keyword combos) using `web_extract` for speed.

### 1B: Semantic Scholar (quality enrichment)

For the top 30 papers from arXiv (deduplicated by ID), fetch citation and
influence metrics:

```bash
# Per paper (batch by arXiv ID)
curl -s "https://api.semanticscholar.org/graph/v1/paper/arXiv:ID?fields=title,citationCount,influentialCitationCount,publicationVenue,year"
```

**Key fields to extract:**
- `citationCount` — raw citations. 0 = weak signal. 50+ = strong.
- `influentialCitationCount` — citations from highly-cited papers. Better indicator of real impact.
- `publicationVenue` — if published at a top venue (NeurIPS, ICLR, ICML, ACL), boost quality.

**Also fetch:** Paper recommendations for top-scored papers to discover
related work not caught by keyword search:

```bash
curl -s -X POST "https://api.semanticscholar.org/recommendations/v1/papers/" \
  -H "Content-Type: application/json" \
  -d '{"positivePaperIds": ["arXiv:2606.XXXXX"], "negativePaperIds": []}' | python3 -m json.tool
```

Add any recommended papers with score ≥0.7 relevance to the candidate pool.

### 1C: HuggingFace Daily Papers (curation cross-reference)

Fetch the community-curated daily selection:

```bash
curl -s "https://huggingface.co/api/daily_papers?limit=30" | python3 -m json.tool
```

This returns the current day's curated papers with title, paper ID, upvotes,
and discussion link. The response structure uses `paper.id` which is the raw
arXiv ID (e.g., `"2606.00467"`) — NOT a URL. However, older API versions may
return a full URL, so handle both formats.

**Parsing the arXiv ID from HF Daily:**

```python
paper_id = p.get('paper', {}).get('id', '')
arxiv_id = ''
# Format 1 (current): raw arXiv ID like "2606.00467"
if paper_id and not paper_id.startswith('http'):
    arxiv_id = paper_id.split('v')[0] if 'v' in paper_id else paper_id
# Format 2 (legacy): URL like "https://arxiv.org/abs/2606.00467"
elif 'arxiv.org/abs/' in paper_id:
    arxiv_id = paper_id.split('/abs/')[-1].split('v')[0]
elif 'arxiv.org/pdf/' in paper_id:
    arxiv_id = paper_id.split('/pdf/')[-1].split('.pdf')[0].split('v')[0]
```

For each HF Daily paper:

1. Extract the arXiv ID using the dual-format parser above
2. Cross-reference against our arXiv candidate pool
3. If a paper appears in BOTH arXiv hits AND HF Daily → set `hf_featured: true` (quality weight table handles the bump)
4. If a paper appears on HF Daily but NOT in arXiv hits → add to candidate pool with a `source: hf-daily` tag

### 1D: Papers With Code (implementation signal)

For score 4-5 papers, check if there's a linked implementation:

```bash
# Search by arXiv ID
curl -s "https://paperswithcode.com/api/v1/papers/?arxiv_id=2606.XXXXX" | python3 -m json.tool
```

**Key fields to extract:**
- `repository_url` — link to GitHub/implementation
- `stars` — GitHub stars if available (via GitHub API or web_extract)
- `framework` — PyTorch, JAX, TensorFlow, etc.

**Implementation signal scoring:**
- No repo found → implementation multiplier = 1.0
- Repo found, <50 stars → multiplier = 1.2
- Repo found, 50-500 stars → multiplier = 1.3
- Repo found, 500+ stars → multiplier = 1.5
- PyPI/npm package published → multiplier = 1.5 (bonus: directly usable)

## Phase 2 — Score & Triage (v2 multi-dimensional)

The v2 scoring adds quality and implementation dimensions on top of raw relevance.

### Step 1: Base Relevance (1-5)

Same as v1 — how well does this paper's topic align with Sahil's domains?

- 5: Directly applicable to KenseiAgent, a public repo, or a product
- 4: Relevant to agent architecture, tool design, or AI infra we use
- 3: General AI/ML insight that could inform strategy or content
- 2: Interesting but not directly actionable
- 1: Not relevant (drop)

### Step 2: Quality Weight (0.3 — 1.2)

Derived from Semantic Scholar + HuggingFace signals. Tighter than v2.0 to prevent
vapourware from inflating — a paper must PROVE quality, not just match keywords.

| Citation Count | HF Daily Featured? | Published at Top Venue? | Weight |
|---|---|---|---|
| 0-2 | No | No | 0.5 |
| 3-10 | No | No | 0.6 |
| 11-50 | No | No | 0.7 |
| 0-10 | Yes | No | 0.8 |
| 11-50 | Yes | No | 0.9 |
| 50+ | No | No | 0.9 |
| 50+ | Yes | No | 1.0 |
| 100+ | Yes | No | 1.1 |
| Any | Yes | Yes (NeurIPS, ICLR, ICML, ACL) | 1.2 |
| 50+ | No | Yes (top venue) | 1.1 |

**v2.1.1 change:** 0-2 citation floor raised from 0.3→0.5, 3-10 floor raised from 0.5→0.6.
Also added a **recency boost**: +0.1 weight for papers published in the last 3 days
(only applied if base relevance score ≥ 4, preventing marginal papers from inflating).
This prevents brand-new pre-prints from being immediately penalised to Skip tier.

### Step 3: Implementation Signal (1.0 — 1.3)

Derived from Papers With Code + GitHub. Lower cap than v2.0 — a great repo
shouldn't single-handedly carry a weak paper to top tier.

| Implementation Found? | Stars | Published Package? | Multiplier |
|---|---|---|---|
| No | — | — | 1.0 |
| Yes | <50 | No | 1.1 |
| Yes | 50-500 | No | 1.2 |
| Yes | 500+ | No | 1.25 |
| Yes | Any | PyPI/npm | 1.3 |

### Step 4: Final Score

```
Final Score = Base Relevance × Quality Weight × Implementation Multiplier
(no cap — score is real, not clamped to 5.0)

Examples:
  Paper about agent memory, 30 citations, HF Daily, 200★ repo
  Old: 4 × 0.9 × 1.2 = 4.3 → Ask First
  New (v2.1.1): 4 × 0.9 × 1.2 = 4.3 → Ask First (unchanged — already fair)

  Paper about context compression directly usable by Toolaria, 5 citations, no repo
  Old: 5 × 0.5 × 1.0 = 2.5 → File
  New (v2.1.1): 5 × 0.6 × 1.0 = 3.0 → Ask First (raised from File)

  Strong paper: Tool-calling agent paper, 60 citations, HF Daily, top venue, 800★ repo
  Old: 5 × 1.1 × 1.25 = 6.9 → Write Now
  New (v2.1.1): 5 × 1.1 × 1.25 = 6.9 → Write Now (unchanged)

  Brand-new pre-print: agent memory paper, 0 citations, published yesterday, no repo
  Old: 4 × 0.3 × 1.0 = 1.2 → Skip
  New (v2.1.1): 4 × 0.5 × 1.0 = 2.0 → File (raised from Skip, recency boost not applied since base=4 but 0.5 floor already covers it)
```

**Score thresholds:**

| Final Score | Action |
|---|---|
| ≥5.5 | **Write Now** — full treatment, all 4 output streams |
| 3.0 — 5.4 | **Ask First** — wiki page, content brief, surfaced for review |
| 1.5 — 2.9 | **File** — mention in existing wiki page if relevant |
| <1.5 | **Skip** — drop quietly |

**Important:** The quality weight can demote a high-relevance paper. A paper
scoring 5 on relevance but with 0 citations and no code drops to 5 × 0.3 × 1.0 = 1.5
— from "Write Now" (v1) to "Skip" (v2.1.0). This filters out vapourware and
pre-prints with no track record. The tighter scoring is intentional: only papers
that PROVE quality through citations, curation, or implementations reach top tiers.

### Relevance heuristics (unchanged from v1)
- References "Hermes", "MCP", "agent tool", "context compression" → direct KenseiAgent relevance
- About "coding agent benchmarks" or "tool-use evaluation" → Octacon-relevant
- About "memory systems" or "knowledge graphs" → Mnemosyne/GBrain-relevant
- About "prompt optimization" or "context engineering" → prompt-optimizer/Toolaria-relevant
- About "multi-agent coordination" or "orchestration" → workforce governance-relevant
- About "React Native AI", "voice AI", "kitchen AI" → product-relevant
- About "football analytics AI" or "coaching AI" → CoachOS-relevant (lower priority)

## Phase 3 — Synthesize

For each score ≥3.5 paper:

1. **Read the full paper** (web_extract the PDF if available; fallback to abstract)
2. **Extract the core insight** in 1-2 sentences
3. **Identify the "so what"** — what would change if this idea were adopted?
4. **Note any code/repos/tools** mentioned that could be added to GitRadar
5. **Include quality metadata** in output: citations, venue, HF featured, repo stars

## Phase 4 — Cross-Reference (the multiplier)

For each score ≥3.5 paper, cross-reference against:

### A) Sahil's Public Repos
Run `gh repo list Sahil-SS9 --visibility public --limit 20` to get current list.
Match paper topics against:

| Repo | Match If Paper Mentions |
|------|------------------------|
| **Toolaria-Protocol** | context compression, tool output rescue, blob storage, LLM context |
| **hermes-memlock** | context integrity, instruction drift, prompt re-assertion |
| **hermes-multichannel-prompt-optimizer** | prompt engineering, LLM optimization, quality scoring |
| **hermes-simplify-swarm** | multi-agent code review, parallel agents, code quality |
| **hermaguard** | adversarial testing, code security, multi-agent review |
| **MrHermagi-tutorbot** | AI education, curriculum generation, tutoring agents |
| **mnemosyne** | memory systems, knowledge graphs, vector search |
| **GitRadar-Self-Improvement** | automated discovery, threshold tuning, repo ranking |
| **hermes-Custom-CLI-Themes** | CLI UX, terminal design (narrow — rare match) |
| **hermes-agent** (fork) | Hermes-specific improvements, gateway, tool system |

### B) KenseiAgent Internals
Cross-reference against active plugin/skill/tool areas:

| Area | Match If Paper Mentions |
|------|------------------------|
| Toolaria plugin | tool output rescue, blob store, context rescue |
| context-guard plugin | standing instruction preservation, context drift |
| prompt-optimizer plugin | prompt rewriting, LLM optimization |
| Mnemosyne memory | memory provider, vector search, knowledge graphs |
| Multi-bot gateways | multi-agent dispatch, gateway routing |
| Kanban system | task orchestration, agent workflow |
| Delegate system (delegate_task) | subagent spawning, parallel execution |
| Workforce governance | agent org structure, lead/worker patterns |

Match by reading `~/.hermes/plugins/` directory listing and checking paper abstracts
against known component responsibilities.

### C) GitRadar Wiki (222+ repos)
Read `~/docs/wiki/repos/index.md` for the full classified repo list. For each score ≥3.5 paper:

- Scan the index for repos whose description or classification aligns with the paper
- If a paper mentions a tool/technique/pattern, check if the wiki has it → if not, flag as "GitRadar candidate"
- If a paper's insight could be combined with a repo from the wiki → flag as "Mashup Idea"
- **New v2:** If the paper has a Papers With Code repo, cross-reference THAT repo against the wiki too

**Mashup format (expanded with idea breakdown + Remii action):**

Each mashup entry must include four parts:

**Part 1 — The Idea (what to build)**
A bold claim of what the combined paper + repo produces. Be specific — name the
output artifact, not just a concept.

**Part 2 — How It Works (architecture)**
2-4 sentences explaining the integration mechanism. What does the paper contribute?
What does the repo provide? How do they connect? Include concrete technical details
(API hooks, data formats, integration points) so the idea is falsifiable.

**Part 3 — Why It Matters (impact)**
1-2 sentences on what this unlocks for KENSEI or Sahil's stack. Context savings,
capability improvement, time saved, competitive advantage.

**Part 4 — Remii's Recommendation (validation + action)**
This is NOT a thoughtless output — it requires Remii to apply reasoning and
validation before writing. For each mashup, Remii must:

1. **Validate plausibility** — Is the paper's technique mature enough for a real
   implementation? (pre-print with no code = lower confidence). Does the GitRadar
   repo actually work as claimed? (check stars, last commit, open issues).
2. **Estimate effort** — Small (<1 day prototype), Medium (1-3 days), Large (1-2 weeks)
3. **Assign risk** — Low (low-hanging fruit), Medium (novel integration), High (unproven technique)
4. **Recommend next action** — One of:
   - **Read & Learn** — The idea is directional but doesn't warrant current build time.
     File to wiki as a concept reference. No kanban task.
   - **Investigate** — The idea is promising but needs a spike to validate feasibility.
     Create a `research` board kanban task with `priority: 3` and `--triage`.
   - **Prototype** — The idea is well-understood and directly applicable. Create a
     `research` board kanban task with `priority: 2` and a clear development brief.
   - **Escalate** — The idea affects a specialist lead's domain (Octacon, Wesker, etc.).
     Create a kanban task assigned to the relevant lead.
   - **Defer** — The idea is sound but conflicts with current priorities. Log to
     paper-mashups.md as a reference and revisit in 30 days.

**Mashup output example:**

```
### Mashup: HyperTool + hermes-edu-skills = Composite Skill Bundler

**Idea:** A Hermes plugin that composes MCP tool calls into atomic skill units with
local intermediate value passing, reducing the model's visible step count from N
sequential calls down to 1 composite call.

**How it works:** HyperTool introduces "composite tool code blocks" — deterministic
sub-workflows that the model invokes as a single call rather than step-by-step. The
plugin intercepts skill invocations in Hermes, groups adjacent MCP calls by the
same tool server, wraps them in a HyperTool-style composite block, executes the
full chain in one round, and returns all results batched. This collapses N tool
calls into 1 model-visible step. hermes-edu-skills provides the skill template
pattern — each composite block maps to a skill template (.md with frontmatter).

**Why it matters:** Reduces context bloat by up to 80% when 5+ skills are needed
for a single task (e.g., CoachOS session planning). The model spends fewer turns
managing data flow and more turns on actual reasoning.

**Remii's Recommendation:**
- Validation: HyperTool paper is pre-print (no code yet). hermes-edu-skills is
  mature (120★, active repo). The plugin concept is sound but needs a spike to
  verify the Hermes plugin system supports composite-block interception.
- Effort: Medium (1-3 days for prototype)
- Risk: Medium (HyperTool is unproven outside the paper)
- Action: Investigate — create research board task for a 1-day spike on Hermes
  plugin composite-block interception. If the spike passes, promote to Prototype.
```

See the `references/mashup-format-examples.md` file for more examples.

### D) Product Portfolio
| Product | Match If Paper Mentions |
|---------|------------------------|
| **Plenishd** | voice AI, kitchen/pantry AI, React Native AI |
| **CoachOS** | coaching AI, football/sports AI, training analytics |
| **MatchdayMaestro** | football prediction, live data AI, gamification |
| **Kick-tionary** | educational AI, sports education |
| **Player Portfolio Builder** | CV generation, portfolio AI |

### E) Content Angles
For each score ≥3.5 paper, generate content angles:

**Twitter/X:**
- "Hot take" on the paper's insight (1-2 tweets)
- "Building in public" angle if it relates to a repo we maintain
- "Prediction" angle — how this changes the next 6 months
- **New v2:** "Citation flex" — if paper has 100+ citations, that's a credibility hook

**LinkedIn:**
- Deeper analysis (300-500 word post)
- "What I learned from [paper]" format
- "How we're applying this at [project]" if applicable
- **New v2:** "Code walkthrough" — if paper has a public repo, that's a LI post hook

**Blog:**
- Full article (800-1500 words) if paper scores ≥5.0
- Title: "What [Paper Name] Means for [Domain]"
- Include code examples if paper provides any
- **New v2:** Add "Implementation" section if Paper With Code repo exists — show the code

For content generation, load `brand-voices` + `ceecee-platform-content` skills.
Output content briefs to `~/.hermes/content-briefs/` directory.

## Phase 5 — Output

### Stream 1: Research Digest Entry
For score ≥5.0 papers (1-3 per run), produce entries with quality metadata:

```
📄 arXiv Signal — DD/MM/YY
N papers · cross-ref hits: R repos, M mashups, C content angles
Sources: arXiv + Semantic Scholar + HuggingFace + Papers With Code

1. [Paper Title] (score 5.0/5.0 · 30 citations · HF Featured · 200★ repo)
   Core insight: [1 sentence]
   KenseiAgent impact: [which component/repo this affects]
   Code: [github.com/... if available]
   Remii's Take: [opinionated 1-liner]
   Link: https://arxiv.org/abs/ID
```

### Stream 2: LLM Wiki Pages
For score ≥3.5 papers, create or update wiki pages in `~/docs/wiki/concepts/`:

1. Create `~/docs/wiki/raw/papers/{arxiv-id}.md` — immutable source with frontmatter (include citation count, venue, source list)
2. Create/update concept page(s) — follow `llm-wiki` conventions
3. Update `~/docs/wiki/index.md` with new page entries
4. Append to `~/docs/wiki/log.md`
5. **New v2:** Tag wiki pages with `quality: high|medium|low` based on citation count + venue

For score 2.0-3.4 papers: optionally add a one-line reference to an existing concept page.
Don't create new pages.

### Stream 3: GitRadar Mashup Ideas
For cross-reference matches between papers and GitRadar repos (score ≥3.5 papers only):

1. Write to `~/docs/wiki/_meta/paper-mashups.md` (append, dated) using the **expanded
   mashup format** (Idea + How It Works + Why It Matters + Remii's Recommendation)
2. For each mashup, **Remii MUST apply validation before writing**:
   - Check the paper's arXiv ID for code links (Semantic Scholar "externalIds" field)
   - Check the GitRadar repo's star count, last commit date, open issues
   - If the paper has no code AND the repo is dormant, flag as "High Risk" or skip entirely
3. After validation, file kanban tasks per Remii's Recommendation:
   - **Investigate** → `research` board, `priority: 3`, `--triage`
   - **Prototype** → `research` board, `priority: 2`, clear dev brief in body
   - **Escalate** → assigned to relevant lead's board
   - **Read & Learn** or **Defer** → no task, just log to mashups file
4. Tag any created tasks with `paper-mashup` (not `paper-synthesis` — the synthesis cron
   already tagged itself; this tag is for downstream consumers)
5. **New v2:** If the paper has a Papers With Code repo, add that repo to the mashup entry

### Stream 4: Content Briefs
For each content angle identified:

1. Write brief to `~/.hermes/content-briefs/{date}-{paper-id}.md`
2. Brief includes: paper summary, 3 angles (X, LI, Blog), target voice, suggested schedule
3. CeeCee's content-review cron picks these up automatically
4. Notify `#content` Discord channel with summary of new briefs
5. **New v2:** Include quality signals in brief (citations, stars) for credibility hooks

### Stream 5: Brain Fact Promotion (NEW v2.2)

For scored concepts that cross-link Sahil's product stack or KenseiAgent internals,
promote a durable one-line fact into `~/brain/`. This is the upward flow from
research synthesis into the structured GBrain memory tier (D7 fix).

**Activation rule:** Promote only if Phase 4 cross-reference produced a match
against Sahil's product portfolio (Plenishd, CoachOS, MatchdayMaestro, Kick-tionary,
Player Portfolio Builder) OR a KenseiAgent internal component (Toolaria, Mnemosyne,
Kanban, gateway system, delegate system, Hermes-Agent fork, Hermes plugin system).
Do NOT promote generic AI/ML papers with no stack link.

**Target page mapping:**
- KenseiAgent/Hermes internal match → `~/brain/conventions/infrastructure.md`
- GitRadar wiki match → `~/brain/projects/gitradar.md`
- Job hunt / career match → `~/brain/projects/job-hunt.md`
- Content engine / marketing match → `~/brain/projects/content-pipeline.md`
- Product portfolio match (Plenishd, CoachOS, MatchdayMaestro, Kick-tionary,
  Player Portfolio Builder) → `~/brain/apps/portfolio.md`

**Append format:**
```
- YYYY-MM-DD: Research synthesis: <one-line insight> ([[wiki/concepts/<concept-slug>]])
```

The [[wiki/concepts/...]] link creates a cross-tier reference to the LLM wiki
concept page, enabling follow-through from ~/brain to the full synthesis.

**Workflow per concept:**
1. Read the target brain page content (via write_file/read_file or terminal cat)
2. Check the existing content for mechanical duplication (exact text match on
   the `<one-line insight>` part — if found anywhere, skip)
3. If no duplicate, append the formatted bullet line before any section marker
   (`## See Also`, `---`, `## Changelog`) or at the end of the file
4. Write the updated content back (preserve-first — never overwrite existing content)
5. Do NOT create new brain pages. Only append to existing ones.

**Kanban integration:** If a scored concept cross-links a product stack component
that has no brain page yet, file a kanban task:
`hermes kanban create "Brain page needed: <product>" --triage --assignee wesker --priority 3`
This tracks the gap so someone creates the page later.

### Provenance (M6) — Raw source retention lint

Before finalising output for any concept-level run (score ≥3.5 paper), run a
provenance check using the `scripts/provenance-lint.py` script:

1. Run `python3 scripts/provenance-lint.py` — if it exits with code 1, inspect
   the flagged concepts and fix or explain each orphan before delivering output.
2. If a flagged concept is a legitimate non-paper concept (operational note,
   educational reference, article summary) that was correctly excluded by the
   script's arXiv-detection heuristic, add a `sources:` field anyway to
   document its origin and suppress the flag.
3. Flag any unresolved orphaned concepts in the Discord output.
4. If the run has no prompt-level insight about new concepts, skip this check.

This prevents concept drift — every durable wiki insight must trace back to a
verifiable source.

## Execution Rules

### Deduplication
- Deduplicate across ALL FOUR sources using arXiv ID as the canonical key
- Before creating wiki pages, check `~/docs/wiki/index.md` — don't create duplicate pages
- Before filing kanban tasks, check existing boards for related tasks
- If a paper was already processed earlier this week, reference the existing work instead of re-creating

### Cost Management
- Don't fetch full PDFs by default — abstracts are sufficient for scoring
- Only fetch PDFs for score ≥3.5 papers (max 5 per run)
- Semantic Scholar: batch by arXiv ID (up to 30 in one id_list pipe)
- HuggingFace Daily Papers: single API call per run
- Papers With Code: only for score ≥3.5 papers (max 5 lookups)
- Total estimated cost: ~$0.12-0.22/run (up from v1's $0.08-0.15)

### Source Priority
1. arXiv is the primary input stream — all other sources ENRICH, not replace
2. Semantic Scholar lookups are mandatory for the top 30 arXiv hits (quality filtering)
3. HuggingFace Daily Papers is a cross-reference — don't re-fetch if already in arXiv set
4. Papers With Code is opportunistic — only for score ≥3.5 papers

### Fallbacks
- If Semantic Scholar API is slow/unavailable OR rate-limited (HTTP 429): use citation-free scoring (quality weight defaults to 0.7). The batch endpoint is particularly susceptible — if it returns 0 records due to 429, treat it the same as unavailable.
- If HuggingFace API fails: skip curation cross-ref, note in output
- If Papers With Code returns nothing: that's fine, implementation multiplier = 1.0
- If arXiv API is slow (>15s): use cached results from previous run + only fetch new since then
- If PDF extraction fails: work with abstract only — mark wiki entries with `confidence: medium`

### Error Handling
- Withdrawn/retracted papers: skip with a note
- Papers with empty/missing abstracts: skip
- Papers appearing in multiple sources with conflicting metadata: prefer Semantic Scholar metadata
- If 0 papers pass the score ≥2.0 threshold: deliver [SILENT]

## Output Contract

**Primary delivery:** Discord `#research-ops` with compact summary + quality metadata.
**Wiki writes:** Direct to `~/docs/wiki/` following `llm-wiki` conventions.
**Content briefs:** `~/.hermes/content-briefs/` directory.
**Kanban tasks:** `research` board for mashup/build ideas.
**Brain facts:** `~/brain/` pages via file write — see Stream 5.

If the run finds nothing actionable (0 papers scoring ≥2.0): deliver [SILENT].

## Integration Points

- **research-digest (05:30 UK):** This cron runs AFTER the digest. Complementary — digest covers news; this covers paper deep-dives.
- **GitHub Radar (daily):** Papers that mention new repos → file as GitRadar discovery tasks. Papers With Code repos → cross-ref against radar.
- **CeeCee Content Review (05:30):** Content briefs from this run are picked up by CeeCee's existing review pipeline.
- **Denji Governance:** Significant wiki changes (10+ pages) trigger a governance log entry.

## Verification

- [ ] At least 1 paper scored ≥3.5 per run (or [SILENT])
- [ ] Every wiki page created has valid frontmatter with quality metadata
- [ ] Provenance lint passed (run `python3 scripts/provenance-lint.py` — 0 orphaned concepts)
- [ ] Every cross-reference match cites specific file/repo/wiki page
- [ ] Content briefs include paper ID + date + target voice + quality signals
- [ ] Stack-relevant concepts promoted to ~/brain with [[wiki/concepts/...]] links (v2.2)
- [ ] No duplicate brain facts (mechanical dedup before append)
- [ ] Speculative mashup ideas did NOT auto-file kanban tasks
- [ ] All 4 sources were attempted (even if some returned empty)
- [ ] No duplicate wiki pages or kanban tasks created
- [ ] All external links verified (no 404s)

## Pitfalls

- **arXiv API caps at 500 results.** Use `sortBy=submittedDate&sortOrder=descending` — filter first 50 by date.
- **Atom XML namespace parsing requires expanded URIs.** The tag is `{{http://www.w3.org/2005/Atom}}entry`, not `a:entry`. See `references/arXiv-namespace-parsing.md` for the full parsing recipe.
- **Scoring over-match is the #1 pitfall.** Initial keyword-based scoring on broad terms ("agent", "multi-agent", "orchestration") will produce 50-100+ score-5 papers from a 30-result batch. Always use a **two-pass scoring approach**: pass 1 = broad keyword detection, pass 2 = tighten with exact phrases and manual review of score-5 candidates. A good tight-scoring script should produce 10-40 score-5 papers from 140+ unique D-14 results.
- **Write Python scripts to disk, not inline heredocs.** On setups where `python3 << 'PYEOF'` triggers pending approval, write the script to a `.py` file with `write_file`, then execute with `terminal`.
- **Recency boost can over-inflate scores.** Adding +1 to ALL papers published in the last 3 days pushes marginally-relevant papers to score 5. Only apply recency boost to papers that already scored ≥4 on content relevance.
- **Semantic Scholar rate limit: 1 req/s for individual lookups, but the batch endpoint (`/paper/batch`) has tighter undocumented limits.** On 2026-06-13, 3 batch calls (10 IDs each) with 1.1s sleep between them triggered a 429 response. The batch endpoint appears to count each ID in the batch toward the rate limit, or has a lower per-second cap than individual lookups. **Recommendation:** either (a) use an API key for 100 req/s, (b) stagger batch calls with 3s+ between them, or (c) accept that brand-new pre-prints won't have Semantic Scholar data yet and rely on the fallback weight (0.7). The fallback is often the pragmatic choice for the daily cron — brand-new papers rarely have citations.
- **HuggingFace Daily Papers are TODAY ONLY.** The API returns the current day's curated papers. There's no date-range query. Check the `publishedDate` field — if it's stale (>24h), skip.
- **Papers With Code coverage is incomplete.** Many papers don't have linked implementations. That's normal — implementation multiplier = 1.0 in that case.
- **Paper dates are UTC.** Convert to UK time for the D-14 window.
- **Don't over-create wiki pages.** Only score ≥3.5 papers get their own pages. Score 2.0-3.4 get a mention in an existing concept page.
- **Content briefs pile up.** CeeCee's review cron skims the directory. Flag backlog if >5 unprocessed briefs.
- **The GitRadar wiki index is auto-generated.** Don't write to `~/docs/wiki/repos/index.md` directly — use `~/docs/wiki/_meta/paper-mashups.md`.
- **Mashup ideas are speculative.** Flag as "investigate" not "build now." Don't auto-file kanban tasks.
- **Brand voices are required for content briefs.** Load them before drafting angles.
- **Quality weight can demote papers.** A paper scoring 5 on relevance with 0 citations drops to 2.5. This is intentional — it prevents recommending vapourware. Trust the methodology.
- **Source metadata conflicts.** If arXiv says one thing and Semantic Scholar says another, prefer Semantic Scholar (it has editorial review).
- **Don't double-count sources.** A paper that appears on both arXiv and HF Daily is ONE paper, not two. Deduplicate by arXiv ID.
- **Script implementation reference:** See `references/script-implementation-notes.md` for the working fetch script pattern, HF Daily dual-format parsing, Semantic Scholar rate-limit workaround, Firecrawl credit fallback strategy, and arXiv HTTPS + URL encoding fix (all validated 2026-06-13 through 2026-06-14).
- **Pre-processing script URL encoding:** The `research_paper_preprocess.py` script MUST use `https://` (not `http://`) for arXiv API calls, and MUST URL-encode query strings with `urllib.parse.quote()`. Unencoded keyword queries like `all:"coding agent"` will crash with `http.client.InvalidURL`. This was fixed 2026-06-14 — if the script is ever edited or regenerated, verify both `https://` and `quote()` are present.
- **Pre-processing script integration:** See `references/pre-processing-script-integration.md` for the Phase 3 Python pre-processor that offloads API fetching + dedup + scoring from the LLM. Integrated 14/06/26.
- **Provenance lint script:** The `scripts/provenance-lint.py` script determines paper-derivation by arXiv ID pattern match. Some non-paper concepts (operational notes from vulnerability reports, platform documentation) may be flagged as candidates — add a `sources:` field to document their origin and suppress the flag rather than ignoring the linter.
- **Brain page write access:** The cron agent writes to `~/brain/*.md` using file/toolset write operations. If `gbrain_put` becomes available as a tool call in the cron's toolset, prefer it over file writes for the additional path-traversal guard it provides. Until then, direct file writes are safe because the brain is markdown-native as of Phase 2.5.
