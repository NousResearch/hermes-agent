---
name: qca-cycle
description: "Persistent identity & cognitive memory for Hermes: signed personality kernel, growing graph memory with semantic recall, contradiction detection, an incorruptible novelty verifier, neurochemical state, and SES provenance."
platforms: [linux, macos, windows]
config:
  - key: qca.store
    description: "Path to the QCA graph store (JSON). Default: ~/.hermes/qca/graph.json"
  - key: qca.learned_dir
    description: "Directory where lessons are exported as Hermes skill stubs. Default: ~/.hermes/skills/learned"
---

# QCA Cycle — a cognitive layer for your agent

## What this gives your Hermes

Every installation grows its **own** persona:

- **Immutable personality kernel** (`kernel.ses.json`) — axioms and guardrails, SHA-256 signed, injected into every thinking cycle. Identity cannot silently drift; it changes only by a deliberate new snapshot.
- **Growing graph memory** — every exchange becomes nodes and typed edges (SUPPORTS / REFINES / CONTRADICTS / ASSOCIATED) with semantic recall via embeddings.
- **Incorruptible novelty verifier** — a repeated thought (cosine ≥ 0.90 to an existing node) is discarded by geometry, not by an LLM judging itself. Memory stays clean over months.
- **Neurochemical state** — dopamine / pain / adrenaline / serotonin with real half-lives (hours to days) that persists between sessions and reframes prompts.
- **SES provenance** — the whole state exports as a canonically-hashed snapshot; tampering is detectable.

The LLM is treated as a swappable CPU (`mock | ollama | anthropic`). Swap the model — the personality and memory stay.

## When to use

When the user asks to "reason with memory", wants answers grounded in past project
decisions, or asks to seed/recall/inspect the agent's long-term graph memory.

## Requirements

Pure Python stdlib — no pip installs. Optional but strongly recommended: local
[Ollama](https://ollama.com) with `bge-m3` for semantic embeddings (without it the
skill falls back to lexical hashing — recall degrades but everything works).

## Commands

`SKILL_DIR` is the directory containing this SKILL.md.

```bash
# Full cognitive cycle on a stimulus (returns thought + full H0–H9 trace)
python3 SKILL_DIR/scripts/qca_engine.py think "<question>"

# Seed a fact (layers: CORE | GOAL | CONTEXT | EPISODIC); add a goal
python3 SKILL_DIR/scripts/qca_engine.py seed "<fact>" CORE
python3 SKILL_DIR/scripts/qca_engine.py goal "<goal text>"

# Semantic recall only / graph & neuro state
python3 SKILL_DIR/scripts/qca_engine.py recall "<query>"
python3 SKILL_DIR/scripts/qca_engine.py stats

# Daemons (wire to hermes cron): autonomous step + nightly consolidation
python3 SKILL_DIR/scripts/qca_engine.py pulse    # empty output = deliberate silence
python3 SKILL_DIR/scripts/qca_engine.py sleep    # cluster episodes → CORE abstractions

# Identity: regenerate SOUL file from the graph, signed with the snapshot hash
python3 SKILL_DIR/scripts/qca_engine.py soul

# SES snapshot + integrity verification + lesson export
python3 SKILL_DIR/scripts/qca_engine.py export-ses
python3 SKILL_DIR/scripts/ses_bridge.py verify <snapshot.ses.json>
python3 SKILL_DIR/scripts/ses_bridge.py export-skills
```

## Giving your agent a personality

Copy `kernel.example.ses.json` to `<store dir>/kernel.ses.json` and edit the axioms,
attractor and guardrails. The kernel is read on every cycle and reported in the trace
as a hash — your agent's constitution is verifiable data, not a prompt that erodes.

## Environment overrides

`QCA_STORE`, `QCA_KERNEL`, `QCA_LLM_BACKEND` (mock|ollama|anthropic), `QCA_CHAT_MODEL`,
`QCA_EMB_MODEL`, `QCA_ANTHROPIC_MODEL`, `QCA_LEARNED_DIR`, `QCA_RECALL_THRESHOLD`, `OLLAMA_URL`.
