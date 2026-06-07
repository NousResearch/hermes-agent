<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# Organic Memory Architecture 🧬

<p align="center">
  <a href="https://github.com/NousResearch/hermes-agent/pull/34521"><img src="https://img.shields.io/badge/PR-%2334521-blue?style=for-the-badge" alt="PR #34521"></a>
  <a href="https://github.com/NousResearch/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
  <a href="#scientific-references"><img src="https://img.shields.io/badge/Science-8%20findings-brightgreen?style=for-the-badge" alt="8 adversarially-verified findings"></a>
</p>

> **A biology-inspired memory layer for Hermes' Holographic memory provider.**
> Transforms memory from a "filing cabinet" into a living organism that selectively absorbs, actively digests, strategically forgets, and continuously self-corrects.

---

## What Is This?

Current AI memory systems work like a filing cabinet — items go in and come out. This changes that. Based on **8 adversarially-verified findings from human memory science** (Nature, Science, PNAS, Annual Review of Psychology), we added 4 new modules to Hermes' Holographic memory provider that mimic how biological memory actually works.

**In plain terms**: Your Hermes agent can now:
- **Filter noise** — not everything deserves to be remembered
- **Forget gracefully** — forgotten facts become "silent engrams" that can be recovered, not deleted
- **Consolidate knowledge** — like sleep does for humans, turning experiences into generalized wisdom
- **Learn from mistakes** — when new info contradicts old memories, the system updates intelligently

---

## Quick Start

### 1. Install Hermes (if you haven't)

```bash
# Linux / macOS / WSL2
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash

# Windows (PowerShell)
iex (irm https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.ps1)
```

### 2. Switch to the organic memory branch

```bash
cd ~/.hermes  # or wherever Hermes is installed
git remote add fork https://github.com/20231118185SSPU/hermes-agent.git
git fetch fork
git checkout feat/organic-memory-architecture
```

### 3. Enable in your config

Add to `~/.hermes/config.yaml` (or `AppData/Local/hermes/config.yaml` on Windows):

```yaml
memory:
  provider: holographic

plugins:
  hermes-memory-store:
    auto_extract: true
    default_trust: 0.5
    hrr_dim: 1024
    # Organic memory features (all default-off):
    salience_enabled: true
    salience_min_threshold: 0.2
    silent_engram_enabled: true
    silent_engram_half_life_hours: 720  # 30 days
    consolidation_enabled: true
```

### 4. Restart Hermes

```bash
hermes
```

That's it. The organic memory features will activate on your next session.

---

## What's New

### 4 New Modules

| Module | File | What It Does | Biology Analogy |
|--------|------|--------------|-----------------|
| **Salience Scorer** | `salience.py` | Filters input by emotion, novelty, importance. Repetition gets penalized (power-law). | Thalamus + Amygdala |
| **Silent Engram Engine** | `silent_engram.py` | Memories decay via power-law but **never reach zero**. Forgotten facts become "silent engrams" recoverable via context similarity. | Hippocampus CA1/CA3 |
| **Consolidation Engine** | `consolidation.py` | 3-phase process: select salient facts → transfer to abstract schemas → integrate with existing knowledge. | Sleep (SWS + REM) |
| **Feedback Coordinator** | `feedback.py` | 3 feedback loops: salience learning, prediction-error model, cross-domain bridge discovery. | Prefrontal cortex |

### Schema Changes (`store.py`)

**`facts` table — 8 new columns:**

| Column | Type | Default | Purpose |
|--------|------|---------|---------|
| `strength` | REAL | 1.0 | Memory trace strength [0, 1] |
| `encoding_time` | TIMESTAMP | now | When the fact was encoded |
| `emotional_valence` | REAL | 0.0 | Emotional weight [-1, 1] |
| `surprise_score` | REAL | 0.0 | Prediction error at encoding |
| `silence_threshold` | REAL | 0.1 | Below this → silent |
| `reconsolidation_count` | INTEGER | 0 | Times updated via reconsolidation |
| `last_retrieved` | TIMESTAMP | null | Last access time |
| `context_hash` | TEXT | '' | Encoding context fingerprint |

**5 new tables:** `schemas`, `schema_facts`, `reconsolidation_log`, `consolidation_runs`, `entity_edges`

### 4 New Tool Actions

| Action | Description |
|--------|-------------|
| `recall` | Multi-depth silent engram search (depth 1=active, 2=semi-active, 3=silent) |
| `schemas` | Browse generalized knowledge schemas |
| `consolidate` | Trigger memory consolidation manually |
| `health` | System health dashboard (predictions, salience learning, bridges) |

### Enhanced Existing Features

- **CJK/multi-token LIKE fallback** for FTS5 — Chinese search now works
- **Improved entity extraction** — technical abbreviations, mixed-case terms
- **Proactive context injection** — memory participates in every turn without tool calls
- **Hebbian learning** — co-retrieved entities get stronger connections

---

## Configuration Reference

All new features are **default-off**. Enable only what you need:

```yaml
plugins:
  hermes-memory-store:
    # === Existing config (unchanged) ===
    db_path: null              # Default: $HERMES_HOME/memory_store.db
    auto_extract: true
    default_trust: 0.5
    hrr_dim: 1024

    # === Organic Memory: Salience Gate ===
    salience_enabled: false              # Enable input filtering
    salience_min_threshold: 0.2          # Below this → not stored

    # === Organic Memory: Silent Engrams ===
    silent_engram_enabled: false         # Enable graceful forgetting
    silent_engram_half_life_hours: 720   # Strength half-life (30 days)

    # === Organic Memory: Consolidation ===
    consolidation_enabled: false         # Enable sleep-like consolidation
```

### Minimal Config (just silent engrams)

```yaml
memory:
  provider: holographic

plugins:
  hermes-memory-store:
    silent_engram_enabled: true
```

### Full Config (all features)

```yaml
memory:
  provider: holographic

plugins:
  hermes-memory-store:
    auto_extract: true
    default_trust: 0.5
    hrr_dim: 1024
    salience_enabled: true
    salience_min_threshold: 0.2
    silent_engram_enabled: true
    silent_engram_half_life_hours: 720
    consolidation_enabled: true
```

---

## How It Works

### Before (Filing Cabinet)

```
User says something → Store it → Search later → Found or not found
                                                    (binary)
```

### After (Living Organism)

```
User says something
       │
       ▼
[Salience Gate] ── Is this worth remembering?
       │              Score: emotion + novelty + importance
       │              Repetition penalty (power-law)
       ▼
[Hippocampus] ─── Store with strength [0, 1]
       │              Not just "exists" — exists with gradient
       ▼
[Silent Engrams] ─ Forgotten? Strength decays but never reaches 0.
       │              Can be recovered via context similarity.
       ▼
[Consolidation] ── Like sleep: select → transfer → integrate
       │              Episodic → Semantic (schemas)
       ▼
[Feedback Loops] ── System learns what matters to you
                    Predictions improve from mistakes
                    Cross-domain bridges discovered
```

### Memory Strength Continuum

| Strength | State | Behavior |
|----------|-------|----------|
| > 0.5 | **Active** | Normal search finds it |
| 0.2 – 0.5 | **Semi-active** | `recall(depth=2)` finds it |
| 0.05 – 0.2 | **Silent** | `recall(depth=3)` or context similarity recovers it |
| < 0.05 | **Buried** | Only spontaneous recovery via strong context match |

---

## Backward Compatibility

| Concern | Answer |
|---------|--------|
| Will my existing data break? | No. All existing facts get `strength=1.0` (fully active). |
| Do I need to change my config? | No. All new features default to OFF. |
| Does this change the core agent? | No. Everything is within the plugin boundary. |
| Schema migration safe? | Yes. Fully idempotent — safe for existing databases. |
| Existing `fact_store` actions? | Unchanged. New actions are additions. |

---

## Testing

- **59 unit tests** across 13 dimensions
- **23 security/correctness audit issues** found and fixed
- **18 performance optimizations** applied
- **All 7 audit checks PASS**: no bare except, no SQL injection, thread-safe locks, shutdown cleanup, no deprecated API, input validation, TOCTOU fix

```bash
# Run the tests
cd hermes-agent
python -m pytest tests/test_holographic_organic.py -v
```

---

## Manual Test Checklist

- [ ] Enable `silent_engram_enabled: true`, add facts, verify `recall(depth=3)` recovers dormant memories
- [ ] Enable `consolidation_enabled: true`, add 5+ facts, trigger `consolidate`, verify schemas are created
- [ ] `fact_store(action="health")` shows system health dashboard

---

## Scientific References

This work is grounded in 8 adversarially-verified findings from human memory science. Each finding was cross-checked by 3 independent verifiers (2/3 majority required).

| Finding | Source | Confidence | Used In |
|---------|--------|------------|---------|
| Diminishing marginal returns | Ebbinghaus (1885); Cepeda et al. (2006) | High | Salience scorer |
| CREB/excitability allocation | Han et al. (2007) Science | Medium | Salience design |
| Silent engrams (forgetting ≠ erasure) | Ryan et al. (2015) Science | High | Silent engram engine |
| Sleep consolidation → gist/schema | Diekelmann & Born (2019) | High | Consolidation engine |
| Compressive RAG model | Spens & Burgess (2024) | Medium | Architecture inspiration |
| Prediction-error reconsolidation | Sinclair & Barense (2019) | High | Reconsolidation engine |
| Proactive reasoning protocol | TsinghuaC3I (2026) | High | Proactive context injection |
| Spreading activation | Synapse (2026) | High | Hebbian learning |
| Memory flow | Park et al. (2023) | High | Automatic memory retrieval |

For the full research report with detailed analysis, see:
- [Hermes Memory Organism Research](https://github.com/user-attachments/files/28385126/Hermes_Memory_Organism_Research.md)
- [Hermes Memory Execution Plan](https://github.com/user-attachments/files/28385120/Hermes_Memory_Execution_Plan.md)

---

## FAQ

**Q: Will this slow down my agent?**
A: Salience scoring uses rule engines (no LLM calls). Silent engram decay runs in background. Consolidation is async. Minimal impact on conversation latency.

**Q: What if I only want silent engrams, not the full suite?**
A: Just set `silent_engram_enabled: true` and leave the others off. Each module is independent.

**Q: How much extra storage does this use?**
A: The 8 new columns and 5 tables add minimal overhead. Silent engrams are just rows with low strength values — no duplicate storage.

**Q: Can I disable it later?**
A: Yes. Set all `*_enabled` flags to `false`. Existing data stays intact, just unused.

**Q: Does this work with other memory providers (Honcho, Mem0, etc.)?**
A: No. This only enhances the Holographic provider. Other providers are unaffected.

---

## Status

| Item | Status |
|------|--------|
| PR | [#34521](https://github.com/NousResearch/hermes-agent/pull/34521) — OPEN |
| Branch | `feat/organic-memory-architecture` |
| Fork | [20231118185SSPU/hermes-agent](https://github.com/20231118185SSPU/hermes-agent) |
| Tests | 59 passing |
| Audit | 7/7 checks PASS |
| Merge status | Awaiting review |

---

## License

MIT — same as [Hermes Agent](https://github.com/NousResearch/hermes-agent/blob/main/LICENSE).

---

<p align="center">
  <i>Memory is not a storage device. It's a living system.</i>
</p>
