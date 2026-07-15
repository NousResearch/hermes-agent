# JARVIS — merged agent plan for HERMES//HUB

This reconciles the ambitious 9-layer "Jarvis Agent" build plan with what the
dashboard already ships. The guiding constraint is the app's ethos: a
**zero-dependency Python stdlib server** and a **zero-build vanilla-JS
frontend**. Most of the spec's layers already exist here in a leaner form; the
merge keeps every good idea and drops the heavyweight machinery (FastAPI,
uvicorn, APScheduler, Docker sandbox, socket.io, pgvector) that the stdlib
server, SSE, the automations daemon thread and SQLite already cover.

## Layer map — spec → what exists here

| Spec layer | Status in HERMES//HUB | Notes |
|---|---|---|
| A. Model core (Claude API) | **Done** | `assistant.py`, optional `anthropic` SDK; local rule-based fallback when absent |
| B. Agent runtime / tool loop | **Done** | Custom loop: server relays Claude turns, client executes `tool_use` (`agent.js` + `actions.js`); SSE streaming |
| C. Tool registry | **Done** | `DASHBOARD_TOOLS` (client-executed) + `SERVER_TOOLS` (research/memory/automation), versioned in one place |
| D. Orchestrator / server | **Done (stdlib)** | `server.py` — REST + SSE, session-less (history lives client-side + synced), permission boundary at the API |
| E. Memory store | **Partial** | `data/memory.md` (facts) + `hub.db` (synced state) + `telemetry.jsonl` (routing + tool outcomes, Phase 3 **done**). Vector recall still optional |
| F. Permission / safety gate | **Done** | auto/confirm/blocked tiers + approval card in the Agent widget (`assistant.TOOL_TIERS`) |
| G. Self-evolution loop | **Deferred** | Highest risk. Not until F + telemetry are solid; keep human-approval on every structural change |
| H. Dashboard integration | **Done** | Chat panel + streaming + approval inbox + System status widget |
| I. Model router / advisor | **Done (routing)** | Cost-aware tiering in `router.py`; live advisor escalation is Phase 5 |

## Fixed safety boundaries (never self-editable)

- The permission classification (Layer F) lives in server code/config only.
  The agent cannot change its own tiers.
- No tool for moving money, entering credentials, or irreversible deletes
  without an explicit confirm.
- Every autonomous action (automations, future reflection) is logged.
- A kill switch freezes all autonomous behavior (automations + any future
  self-evolution) from the dashboard. (Phase 4 below.)

## Build phases (merged, re-sequenced for this stack)

**Phase 1 — Cost-aware model router (Layer I).** `router.py`: classify each
request into a tier and pick the cheapest viable model; escalate to the deep
tier only when the task is flagged hard/ambiguous/security/self-modifying.
Log every decision for later tuning. Deep-tier calls rate-capped. Tiers are
env-overridable; a single `HERMES_HUB_MODEL` pins all tiers (back-compat).

- Tier FAST — Haiku 4.5: summaries, classification, formatting
- Tier CORE — Sonnet 5: default chat and tool loops
- Tier DEEP — Opus 4.8: hard reasoning, briefings-as-reflection, escalations

**Phase 2 — Permission gate + approval inbox (Layers F + H).** Classify each
`tool_use` auto/confirm/blocked. `confirm`-tier tools (add_app, open_url to
external, future write/shell) surface an approval card in the Agent widget and
block until the user clicks. `blocked` refuses with a reason. Classification
is server-owned and mirrored to the client for pre-flight UX only.

**Phase 3 — Telemetry + status widget (Layers E + H). ✅ DONE.** Bounded
routing + tool-call telemetry (`data/telemetry.jsonl`, `telemetry.py`); the
System widget shows active engine/tier, tier line-up, deep-tier budget, the
permission split, and a feed of recent tool calls (name · tier · outcome).

**Phase 4 — Kill switch + hardening (Phase 6 in spec).** One dashboard toggle
freezes automations and any autonomous run; checked before every tick. Audit
log surfaced read-only.

**Phase 5 — Advisor escalation, live (rest of Layer I).** When the core model
self-reports low confidence or fails a subtask twice, call the deep tier as a
*scoped advisor* (guidance only, no tools, no final output), inject its answer,
and resume at the core tier. Needs Phase 3 telemetry to tune thresholds.

**Phase 6 — Bounded self-evolution (Layer G).** Nightly reflection over
telemetry proposes prompt tweaks / new-tool specs / memory pruning. Safe
changes auto-apply; structural changes queue in the approval inbox; every
applied change is git-committed for rollback. Explicitly last, explicitly
gated. Do not start until Phases 1–5 are stable with logged data.

## Deliberately rejected (keeps the ethos intact)

- FastAPI/uvicorn/Express — the stdlib `http.server` already serves REST + SSE.
- APScheduler/node-cron — the automations daemon thread is the scheduler.
- Docker sandbox for `run_shell` — no shell tool ships; if one is ever added it
  gets `blocked` tier by default, never `auto`.
- pgvector/FAISS — memory volume doesn't justify a vector index yet; revisit
  only if recall quality demands it.
