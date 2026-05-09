# Proactive Communication Loop

> *"I'd love to see some sort of proactive loop where every night Hermes takes everything
> from the day, synthesizes it, and tries to start finding ways to act or message proactively.
> I'd love to have Hermes message me occasionally on its own. That can't be easy though..."*
>
> — @charlesmcdowell, May 8 2026 · 2.2K views
>
> Teknium replied: *"This is a good idea 🤔"*

---

## Quick Start

These three steps describe what you need once the gateway scheduler and delivery path are wired (same rollout pattern as other Hermes subsystems):

```bash
# 1) Opt in via Hermes config (defaults keep the loop off)
echo 'proactive_communication.enabled: true' >> ~/.hermes/config.yaml

# 2) Point BartokGraph at a workspace that contains a built graph (see below)
#    Default workspace is home — graph path: ~/.bartokgraph/graph.json

# 3) Trigger or schedule the synthesis pass (gateway hookup — follow-up PR)
#    Engine API: ProactiveCommunicationLoop.run_synthesis(session_id)
```

Without a `graph.json` file, the loop still runs in **recency-only** mode (conversation history only).

---

## What This Is

The **Proactive Communication Loop** gives Hermes a synthesis-and-initiative pass that runs
on a configurable schedule (by default: nightly). After synthesizing the day, Hermes decides
**on its own** whether it has something worth saying — and if so, sends the user a message
**without being asked**.

This is the difference between a tool and a partner. A tool waits. A partner notices.

---

## The Two Modes

### Mode 1: Recency-only synthesis

Hermes reviews the last N hours of conversation history and asks:
*"Did I finish something worth reporting? Is there an unresolved thread that now has an answer?
Did the user ask me to let them know about something?"*

This works well for daily wrap-ups and completed task notifications.

### Mode 2: BartokGraph-augmented synthesis (the powerful one)

When a local model and BartokGraph are available, Hermes goes further. BartokGraph is a
**knowledge graph builder** that maps concepts, projects, people, and ideas from the user's
files and conversation history into a weighted graph with typed edges
(`TEACHES`, `BUILDS_ON`, `CONTRADICTS`, `MENTIONS`, etc.).

With BartokGraph, the synthesis pass can answer:

> *"Does anything from today connect to something the user worked on 3 weeks ago
> that they may have forgotten? Are there cross-domain connections they can't see
> because they can't hold months of context in their head?"*

**Three new message types only BartokGraph enables:**

| Type | Example message |
|------|----------------|
| **Temporal bridge** | "Hey — you worked on this exact problem 3 weeks ago. The approach you used then applies here." |
| **Cross-domain connection** | "Your regime detection work and your soil carbon research share the same underlying structure — both are looking for state transitions in noisy signals." |
| **Person-knowledge bridge** | "Alice mentioned the Kenya soil project last week. You're working on bioavailability today. These connect to Guruji's objective #6 directly." |

Without BartokGraph: Hermes sees a *transcript snippet*.  
With BartokGraph: Hermes sees a *web of weighted, time-stamped knowledge* — and can ask
whether today's work activates a dormant thread.

This is what makes users say **"how did it know that?"**

---

## BartokGraph

BartokGraph is an open-source knowledge graph builder created by the same author.
It is included with this PR as an **optional bundled plugin** (`plugins/bartokgraph/`).

### What BartokGraph does

1. **Scans** a folder (your Hermes workspace, notes, research files) and extracts concepts,
   entities, and relationships.
2. **Builds** a weighted graph where nodes are concepts and edges are typed relationships.
3. **Detects communities** (clusters of related concepts) using topology-based analysis —
   **no embeddings or API calls required**.
4. **Runs locally** using Ollama (default model: `qwen3:8b`) — **zero API cost**.
5. **Supports** any OpenAI-compatible endpoint as an alternative.

### How BartokGraph builds the graph (design contract)

The proactive loop’s **adapter** reads a pre-built `graph.json` under the configured workspace.
The **full graph builder** (scanner, weighting, edge extraction) ships as the broader BartokGraph
tooling; this PR wires Hermes to the graph **file format** and traversal.

When the builder runs against your workspace, it typically:

**Sources scanned**

| Source | Role |
|--------|------|
| `SOUL.md` and similar identity / preference docs | High signal “who the user is” |
| Daily memory / journal-style captures | Medium signal “what happened recently” |
| Project notes (`README`, specs, research markdown) | Structured project context |
| Code files | Lower-weight lexical hooks into implementation |

**Default node weights (design targets for the scanner)**

| Source kind | Weight |
|-------------|--------|
| SOUL.md–class identity docs | 50 |
| Daily memory entries | 20 |
| Project notes / markdown docs | 15 |
| Code files | 1 |

**Typed edges the builder detects** (examples)

| Edge | Meaning |
|------|---------|
| `TEACHES` | One concept explains or introduces another |
| `BUILDS_ON` | Extension or continuation of prior work |
| `CONTRADICTS` | Tension or disagreement between ideas |
| `MENTIONS` | Lightweight co-occurrence / reference |
| `IS_ABOUT` | Topical linkage |
| `IMPLEMENTS` | Code realizing a concept |

The adapter consumes **nodes** with `content`, `weight`, `last_seen_ts`, and optional `node_type`,
and uses word overlap plus dormancy (not touched in the last 24h) to propose connections.

### Runtime graph location

Hermes loads:

`{workspace}/.bartokgraph/graph.json`

with `workspace` from `proactive_communication.bartokgraph.workspace` (default `"~"`, expanded).
If the file is missing, malformed, or uses an unsupported schema, traversal is skipped and the
loop stays on recency-only synthesis — **no crash, no user-visible error**.

### Local model detection (for graph tooling)

`BartokGraphAdapter` and `_resolve_local_model_provider()` probe hosts in order:

1. `BARTOKGRAPH_API_BASE` + `BARTOKGRAPH_API_KEY` + `BARTOKGRAPH_LLM_MODEL` (explicit API)
2. Ollama — `OLLAMA_URL` (default `http://localhost:11434`) — `GET /api/tags`, **2s timeout**
3. LM Studio — `http://localhost:1234/v1/models`, **2s timeout**
4. Other common ports — `http://127.0.0.1:{8080,8000,5000}/v1/models`, **2s timeout** each
5. **`topology_only`** — overlap-based traversal still works without an LLM

### Standalone `hermes bartokgraph` CLI

The plugin package documents future commands such as `hermes bartokgraph build <path>`.
End-to-end CLI registration is **not** part of this PR; use an external BartokGraph build or place
a valid `graph.json` at the path above until the CLI is wired.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│              PROACTIVE COMMUNICATION LOOP                      │
│                                                               │
│  Trigger: cron schedule (default: 10pm nightly)               │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  SYNTHESIS PASS                                          │  │
│  │                                                          │  │
│  │  1. Load recent history (last 16h)                       │  │
│  │  2. [Optional] BartokGraph traversal:                    │  │
│  │     - Find today's active topics                         │  │
│  │     - Query graph for dormant related nodes              │  │
│  │     - Detect cross-temporal connections                  │  │
│  │  3. Build synthesis prompt (with or without graph ctx)   │  │
│  │  4. Judge call (cheap/fast local model):                 │  │
│  │     - Score novelty (0-1) + relevance (0-1)              │  │
│  │     - Apply threshold (conservative=0.75 default)        │  │
│  │  5. If above threshold: compose natural message           │  │
│  └─────────────────────────────────────────────────────────┘  │
│                         │                                     │
│                   should_send?                                │
│                    ↓        ↓                                 │
│                  YES        NO                                │
│                   │          │                                │
│            Send message    Silence (prefer silence)           │
│            via configured  Log reasoning                      │
│            channels        to audit trail                     │
└───────────────────────────────────────────────────────────────┘
```

### New files (this PR)

```
hermes_cli/
├── proactive_communication_loop.py   ← Core engine
└── bartokgraph_adapter.py            ← BartokGraph → Hermes bridge

plugins/bartokgraph/
└── __init__.py                       ← Plugin metadata / exports

docs/features/
└── proactive-communication-loop.md   ← This document

tests/
├── test_proactive_communication_loop.py
├── test_proactive_smoke.py
└── fixtures/bartokgraph_graph.json   ← Synthetic graph for adapter tests
```

---

## Configuration reference

| Config key | Type | Default | Used by engine |
|------------|------|---------|----------------|
| `proactive_communication.enabled` | bool | `false` | Gateway / installer (loop is opt-in) |
| `proactive_communication.schedule` | string | `"0 22 * * *"` | Gateway cron (not read inside `ProactiveCommunicationLoop`) |
| `proactive_communication.threshold` | string | `conservative` | Yes — `conservative` (0.75), `balanced` (0.55), `eager` (0.35), or `@register_threshold` name |
| `proactive_communication.max_per_day` | int | `1` | Yes — hard cap via `SessionDB.get_proactive_sent` |
| `proactive_communication.bartokgraph.enabled` | bool | `true` | Yes — if false, adapter not loaded |
| `proactive_communication.bartokgraph.workspace` | string | `"~"` | Yes — expanded path; graph at `{workspace}/.bartokgraph/graph.json` |
| `proactive_communication.bartokgraph.local_model` | string | `qwen3:8b` | Documented for operators; graph **building** uses env `BARTOKGRAPH_LLM_MODEL` |
| `proactive_communication.bartokgraph.rebuild_interval_days` | int | `7` | Reserved for scheduled rebuilds (not read by adapter in this PR) |

**Environment variables (BartokGraph tooling / detection)**

| Variable | Purpose |
|----------|---------|
| `BARTOKGRAPH_API_BASE` | OpenAI-compatible API base URL |
| `BARTOKGRAPH_API_KEY` | API key when using hosted inference |
| `BARTOKGRAPH_LLM_MODEL` | Model id (default `qwen3:8b`) |
| `OLLAMA_URL` | Override Ollama base URL (default `http://localhost:11434`) |

---

## Privacy and Safety

- **Opt-in by default**: `proactive_communication.enabled = false`
- **Rate limiting**: hard cap of `max_per_day` messages (default: 1)
- **Audit log**: every synthesis pass recorded with reasoning, whether sent or not
- **Kill switch**: `hermes proactive off` immediately stops all future proactive messages (when CLI exists)
- **BartokGraph privacy**: redacts personal identifiers (phone numbers, email, VIP IDs) before graph storage (builder responsibility)
- **Local first**: BartokGraph runs entirely on-device with local models — no data leaves the machine
- **No graph = graceful degradation**: if BartokGraph is not installed or has no data, falls back to recency-only synthesis. The loop never fails.

---

## Troubleshooting

### “BartokGraph not finding connections”

- Confirm `graph.json` exists at `{workspace}/.bartokgraph/graph.json` after expanding `~`.
- Nodes must have `last_seen_ts` **older than ~24 hours** to count as “dormant” versus today’s topics.
- Overlap uses simple word overlap with a minimum strength **0.35** — sparse or very short nodes may never match.
- If JSON is invalid or the schema omits `nodes`, the adapter returns no graph context (recency-only).

### “Local model not detected”

- Expected when nothing listens on the probed URLs; the adapter falls back to **`topology_only`** (still usable).
- Check Ollama: `curl -sS --max-time 2 "$OLLAMA_URL/api/tags"` (default port 11434).
- Check LM Studio: `curl -sS --max-time 2 http://localhost:1234/v1/models`.
- All probes use **2 second** timeouts so startup cannot hang indefinitely.

### “Messages not sending”

- Combined score must clear the threshold: `0.6 * novelty + 0.4 * relevance` (each clamped to `[0,1]`).
- The JSON field `should_send` can veto delivery even when scores are high.
- Empty history, daily cap, or parse failures yield **no send** by design.

---

## Message Examples

**Without BartokGraph (recency-only):**
> "Hey — I finished scanning those logs you asked about earlier. Found something:
> errors appear every 4 hours at exactly :15 past. That's almost certainly a cron job.
> Want me to find which one?"

**With BartokGraph (temporal bridge):**
> "Connecting something — you worked on funding rate arbitrage today, and 3 weeks ago
> you designed the HMM regime detector. They're solving the same problem from different
> angles: both are trying to detect which of two stable states the market is in.
> The regime detector could gate the funding arb bot."

**With BartokGraph (cross-domain):**
> "Your soil carbon work and your trading bot regime detection share an interesting structure.
> Both are looking for state transitions in noisy time-series signals.
> The HMM you built for BTC markets could potentially be adapted for soil health monitoring."

---

## What's Left for Follow-up PRs

This PR is the full architecture, engine, BartokGraph plugin, tests, and documentation.
The remaining work to wire it into a running gateway deployment:

1. Gateway cron scheduling hookup (triggers `run_synthesis` at configured time)
2. Per-provider LLM call implementation (wires into session's configured model)
3. Delivery path integration (uses `callbacks.py` notify to send via configured channels)
4. `hermes bartokgraph` CLI registration for build/query/report commands

The scaffolding pattern (ship the engine cleanly, wire it in a follow-up) is how GoalManager
was landed. It keeps this diff reviewable while establishing the complete design.
