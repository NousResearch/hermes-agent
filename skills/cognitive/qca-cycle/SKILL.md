---
name: qca-cycle
description: "Signed identity kernel and graph memory with recall."
version: 1.1.0
author: Dmitry Trubnikov (@trubnikov) & Hermes Agent
license: MIT
platforms: [linux, macos, windows]
config:
  - key: qca.store
    description: "Path to the QCA graph store (JSON). Default: <HERMES_HOME>/qca/graph.json"
  - key: qca.learned_dir
    description: "Directory where lessons are exported as Hermes skill stubs. Default: <HERMES_HOME>/skills/learned"
metadata:
  hermes:
    tags: [memory, identity, cognitive, graph, SES]
    homepage: https://github.com/trubnikov/qca-cycle
---

# QCA Cycle Skill

An explicit, on-demand cognitive store: a SHA-256-signed personality kernel
(axioms and guardrails injected into every `think` call), a typed graph memory
(SUPPORTS / REFINES / CONTRADICTS / ASSOCIATED edges, semantic recall,
embedding-geometry novelty gate), and canonically-hashed SES snapshots.
This skill does NOT hook into Hermes's automatic per-turn memory lifecycle
(`memory_provider` / `turn_finalizer`); the graph grows only when its commands
are invoked explicitly — by the operator, by the agent following this skill,
or by a cron job you wire yourself.

## When to Use

- The user asks to seed, recall, or inspect long-term graph memory
  ("remember this decision", "what do we know about X?").
- The user wants an answer grounded in previously seeded project decisions
  (`think` runs recall + contradiction check + kernel-constrained synthesis).
- The user wants a verifiable identity/state snapshot (`export-ses`, `verify`).

Do NOT use it as a substitute for Hermes's built-in conversation memory —
nothing is written here automatically.

## Prerequisites

- Pure Python stdlib — no pip installs.
- Optional but recommended: local [Ollama](https://ollama.com) with `bge-m3`
  for semantic embeddings. Without it the skill falls back to lexical hashing
  (recall degrades but everything works).
- Backends for synthesis: `mock | ollama | anthropic`
  (`QCA_LLM_BACKEND`; `anthropic` needs `ANTHROPIC_API_KEY`).
- All state lives under the active Hermes home (`HERMES_HOME`), in
  `qca/graph.json` and `qca/kernel.ses.json`. Override with `QCA_STORE` /
  `QCA_KERNEL`. Other env: `QCA_CHAT_MODEL`, `QCA_EMB_MODEL`,
  `QCA_ANTHROPIC_MODEL`, `QCA_LEARNED_DIR`, `QCA_RECALL_THRESHOLD`, `OLLAMA_URL`.

## How to Run

Run the scripts with the `terminal` tool. `SKILL_DIR` is the directory
containing this SKILL.md.

```bash
python3 SKILL_DIR/scripts/qca_engine.py think "<question>"
python3 SKILL_DIR/scripts/qca_engine.py seed "<fact>" CORE
python3 SKILL_DIR/scripts/qca_engine.py recall "<query>"
```

## Quick Reference

| Command | What it does |
|---|---|
| `qca_engine.py think "<stimulus>"` | Full cognitive cycle H0–H9, returns thought + trace |
| `qca_engine.py seed "<fact>" [LAYER]` | Seed a fact (CORE \| GOAL \| CONTEXT \| EPISODIC) |
| `qca_engine.py goal "<text>"` | Add an active goal |
| `qca_engine.py recall "<query>"` | Semantic recall only |
| `qca_engine.py stats` | Graph / neuro state |
| `qca_engine.py pulse` | One autonomous step toward goals; silence = empty stdout |
| `qca_engine.py sleep` | Consolidation: episode clusters → CORE abstractions |
| `qca_engine.py soul [path]` | Regenerate the signed SOUL identity file |
| `qca_engine.py export-ses [path]` | Canonical SES v5.1 snapshot (signed) |
| `ses_bridge.py verify <snapshot>` | Integrity check; non-zero exit on tamper or canon violation |
| `ses_bridge.py export-skills [dir]` | Export CORE lessons as Hermes skill stubs |

## Procedure

1. **Give the agent a constitution (once).** Copy `kernel.example.ses.json`
   to `<store dir>/kernel.ses.json` and edit axioms, attractor, guardrails.
   The kernel is read on every cycle and reported in the trace as a hash.
2. **Seed durable facts explicitly.** When the user states a decision worth
   keeping, run `seed "<fact>" CORE` (or `CONTEXT` for softer context).
3. **Reason with memory.** For questions that should be grounded in the
   graph, run `think "<question>"` and use the returned thought + trace.
4. **Optional daemons.** Wire `pulse` (periodic) and `sleep` (nightly) to
   the `cronjob` tool. `pulse` prints nothing when it has nothing to say —
   gate message delivery on non-empty output.
5. **Snapshot and verify.** `export-ses` writes a canonically-hashed
   snapshot; `verify` exits non-zero on any hash mismatch or on a
   STATE_SNAPSHOT missing its kernel reference (SES v5.1 canon lock).

## Pitfalls

- Memory does not grow by itself: if you expect an exchange to be recalled
  later, you must `seed` it. This is by design (novelty-gated, auditable
  writes), not an integration bug.
- Without Ollama the embedding fallback is lexical — recall quality drops;
  vectors from different embedding modes are incomparable (similarity 0).
- `think`/`pulse`/`sleep` call an LLM backend; `mock` is for tests only.
- The kernel file is immutable by convention — edit it deliberately and
  re-export a snapshot; never mutate it mid-session.

## Verification

```bash
QCA_LLM_BACKEND=mock python3 SKILL_DIR/scripts/qca_engine.py stats
python3 SKILL_DIR/scripts/qca_engine.py export-ses /tmp/qca-snap.ses.json
python3 SKILL_DIR/scripts/ses_bridge.py verify /tmp/qca-snap.ses.json && echo OK
```

Unit tests: `scripts/run_tests.sh tests/skills/test_qca_cycle_skill.py -q`.
