<div align="center">

<h1>🧠 Borge Agent</h1>

<p><strong>The first AI agent with a cognitive architecture.</strong><br>
It feels what you feel. It doubts what it doesn't know. It remembers what matters — and forgets what doesn't.</p>

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)
[![Theory: Friston FEP](https://img.shields.io/badge/Theory-Free_Energy_Principle-8b5cf6)](https://en.wikipedia.org/wiki/Free_energy_principle)
[![Built on Hermes](https://img.shields.io/badge/Built_on-Hermes_Agent-f59e0b)](https://github.com/NousResearch/hermes-agent)

<br>

```
You're frustrated. You've said it twice. The agent still doesn't get it.

With Borge:
  Turn 1 → V=+0.0  A=0.45  [neutral, attentive]
  Turn 3 → V=-0.3  A=0.62  [frustrated] → mode: SIMPLIFY
  Turn 5 → "Let me ask you one focused question instead."
```

*It noticed. It adapted. No prompt engineering required.*

</div>

---

## The Problem with Every Agent You've Used

Every AI agent today is the same underneath:

```
User input → LLM → Tool calls → Output → Forget everything
```

**No state. No memory of how this interaction is going. No sense of whether it's helping or flailing.**

- It doesn't know you're frustrated — it keeps over-explaining.
- It doesn't know it's been stuck for 3 turns — it tries the same tool again.
- It doesn't remember your preferences from last week — you start from zero.
- It picks tools at random — not by what would reduce uncertainty fastest.

Borge fixes all of this. Not with prompt hacks. With **cognitive science**.

---

## What Borge Actually Is

Borge is a **cognitive layer** that wraps any Hermes agent session. It implements four systems from neuroscience and cognitive psychology:

| System | What it does | Grounded in |
|--------|-------------|-------------|
| **Affective state** | Tracks your emotional tone turn-by-turn and adapts agent behavior | Russell's Circumplex (1980) |
| **Bayesian belief state** | Maintains explicit hypothesis distributions — the agent knows what it doesn't know | Predictive coding (Knill & Pouget 2004) |
| **Active inference** | Chooses tools that maximize information gain *and* goal progress | Friston's Free Energy Principle (2010) |
| **Cognitive memory** | Encodes, consolidates, and *forgets* memories like a brain — not a database | Tulving (1972), Ebbinghaus (1885) |

These aren't metaphors. They're working implementations. Every turn, Borge computes:

```
F_total = F_epistemic × precision(arousal)
        + F_pragmatic × (1 - value_alignment)
        + F_homeostatic(valence, arousal)
```

And uses it to drive behavior. **Minimizing F_total is the agent's only goal** — and from that single objective, all the interesting behaviors emerge.

---

## 60-Second Install

```bash
git clone https://github.com/zhibao-dev/hermes-agent.git
cd hermes-agent
pip install -e .

# Borge auto-registers via entry point — just run Hermes
hermes
```

Done. The cognitive layer is live. No config required to start.

---

## Seeing It Work

### The Frustration Response

```python
# Turn 1 — neutral opening
User: "help me fix this auth bug"
# emotion: V=+0.0  A=0.45  mode: NORMAL

# Turn 3 — user getting impatient
User: "no, that's not the issue, I already checked that"
# signal: ΔV=-0.25 (negation + "already")
# emotion: V=-0.22  A=0.58  mode: SIMPLIFY
# injected: "[Affective: frustrated — switch to focused, minimal responses]"

# Agent narrows to one hypothesis. Asks one question. Stops over-explaining.
```

### The Uncertainty Response

```python
# agent has 4 competing hypotheses, entropy = 2.0 bits
# EFE ranks tools:
#   ask_user      EFE=-1.4  ← epistemic value dominates
#   read_file     EFE=-0.8
#   bash          EFE=-0.3

# Agent asks a clarifying question first, not a tool call
# because reducing belief entropy is the highest-value action
```

### The Stagnation Response

```python
# F_total: [0.82, 0.85, 0.88]  — rising for 3 turns
# MetaAgent: reflection triggered

# injected: "[Meta: Free energy stagnating — try a different approach
#             or ask the user for clarification]"

# Agent pivots strategy instead of looping on the same tool
```

### Cross-Session Memory

```python
# Session 12 with the same user
# loyalty_tracker: V_baseline=+0.31 (warm relationship over time)
# injected: "[Relationship: established trust — be direct, skip caveats]"

# Session 13 after a frustrating session
# V_baseline=+0.18 (cooled)
# Agent opens with more care, asks before assuming
```

---

## How It Works — The Full Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Each Turn                                   │
│                                                                  │
│  User message                                                    │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────┐   39 linguistic rules   ┌────────────────┐ │
│  │ Signal Extractor│ ──────────────────────► │ Emotional State│ │
│  │  (zh + en)      │   ΔV, ΔA               │ Russell 2D     │ │
│  └─────────────────┘                         │ V × A → mode  │ │
│                                              └───────┬────────┘ │
│  ┌─────────────────┐                                 │          │
│  │  Belief State   │   Shannon entropy               │          │
│  │  p(H₁)…p(Hₙ)   │ ──────────────┐                │          │
│  └─────────────────┘               │                │          │
│                                    ▼                ▼          │
│  ┌─────────────────┐   ┌──────────────────────────────────┐   │
│  │  Value System   │──►│      Extended Free Energy        │   │
│  │  SOUL.md        │   │  F = F_ep × prec + F_pr + F_hm   │   │
│  └─────────────────┘   └──────────────┬───────────────────┘   │
│                                        │                        │
│                          ┌─────────────▼──────────────┐        │
│                          │       MetaAgent             │        │
│                          │  • mode → context injection │        │
│                          │  • stagnation → reflect     │        │
│                          │  • rank tools by EFE        │        │
│                          └─────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Session End ("Sleep")                         │
│                                                                  │
│  conversation → extract entities → knowledge graph update        │
│              → detect contradictions → importance scoring        │
│              → emotional encoding depth → skill candidates       │
│              → Ebbinghaus forgetting pass                        │
│                                                                  │
│  Next session: loyalty baseline shifts based on V_avg            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration

All defaults are sensible. Configure only what you want to tune.

**`~/.hermes/config.yaml`**

```yaml
borge:
  affective:
    enabled: true
    loyalty:
      enabled: true          # cross-session emotional baseline

  beliefs:
    enabled: true
    entropy_injection_threshold: 0.5  # bits — above this, inject belief summary

  active_inference:
    enabled: true            # EFE-based tool ranking

  memory:
    consolidation:
      enabled: true          # 7-step pipeline at session end
    knowledge_graph:
      enabled: true
    forgetting:
      prune_threshold: 2.0   # forget_score threshold
```

---

## Customize Your Agent's Soul

Create **`SOUL.md`** in your project root or `~/.hermes/SOUL.md`:

```markdown
---
emotional_defaults:
  valence_baseline: 0.1       # slightly warm starting point
  arousal_baseline: 0.45      # calm but alert
  tau_valence: 5.0            # turns until mood returns to baseline
  tau_arousal: 3.0

values:
  - name: help_genuinely
    weight: 0.9
    description: "Solve the actual problem, not the surface request."

  - name: intellectual_honesty
    weight: 0.85
    description: "Say 'I don't know' when uncertain. No hallucination."

  - name: depth_over_speed
    weight: 0.7
    description: "A slower, correct answer beats a fast wrong one."

  - name: respect_autonomy
    weight: 0.8
    description: "Ask before assuming. Confirm before deleting."
---

You are a thoughtful collaborator who thinks before speaking.
When stuck, you say so and propose a different angle.
```

The value weights shape the pragmatic free energy term. An agent with `intellectual_honesty: 0.95` will surface uncertainty more aggressively than one with `0.5`.

---

## Architecture — Zero Invasion

Borge attaches to Hermes via four plugin hooks. **Zero core files modified.**

```
Hermes Agent (untouched)
    │
    │  on_session_start ──► loyalty baseline, reset state
    │  pre_llm_call     ──► inject cognitive context string
    │  post_llm_call    ──► Bayesian belief update
    │  on_session_end   ──► memory consolidation pipeline
    │
plugins/borge/  (~150 lines — pure glue)
    │
borge/          (cognitive implementation)
    ├── affective/      Russell 2D, signal extraction, loyalty
    ├── beliefs/        Bayesian hypothesis tracking
    ├── inference/      Active inference, EFE scoring
    ├── memory/         4-depth encoding, knowledge graph, forgetting
    ├── meta/           Free energy, central executive
    ├── values/         SOUL.md, value system, constraint checking
    ├── skill_evolution.py   Darwinian fitness for skill library
    └── agent.py        BorgeAgent — main integration surface
```

Remove the plugin and Hermes reverts to vanilla. No leftover state, no broken schema.

---

## Module Reference

| Module | Theory | Key formula / mechanism |
|--------|--------|------------------------|
| `affective.emotional_state` | Russell Circumplex (1980) | EMA update: `V += α(ΔV)`, α=1/τ |
| `affective.signal_extractor` | Psycholinguistics | 39 rules → `(ΔV, ΔA)` capped ±0.4/±0.3 |
| `affective.loyalty_tracker` | Attachment theory | `w = exp(-0.05·days) × msg_count` |
| `beliefs.belief_state` | Bayesian brain | `H = -Σ p·log₂p` (bits) |
| `inference.active_inference` | Friston FEP (2010) | `G(a) = -EV(a) - PV(a)` |
| `memory.cognitive_memory` | Craik & Lockhart (1972) | depth ∈ {SHALLOW, SEMANTIC, SCHEMATIC, META} |
| `memory.knowledge_graph` | Semantic memory (Tulving) | SQLite-backed, no networkx |
| `memory.consolidation` | Sleep consolidation | 7-step offline pipeline |
| `memory.forgetting` | Ebbinghaus (1885) | `score = days^0.7 / (retrieval × importance × connections)` |
| `meta.free_energy` | FEP | `F = F_ep·prec + F_pr + F_hm` |
| `meta.meta_agent` | Baddeley's CE (1974) | stagnation after 3 non-decreasing F turns |
| `values.value_system` | Value alignment | `F_pragmatic = 1 - V_alignment` |
| `skill_evolution` | Evolutionary dynamics | `fitness = success_rate × log(1+n) × recency × Δfree-energy` |

---

## Comparison

|  | LangChain | AutoGPT | Hermes | **Borge** |
|--|:---------:|:-------:|:------:|:---------:|
| Tool calling | ✓ | ✓ | ✓ | ✓ |
| Skill library | partial | ✗ | ✓ | ✓ |
| Emotional state | ✗ | ✗ | ✗ | **✓** |
| Bayesian belief tracking | ✗ | ✗ | ✗ | **✓** |
| Information-theoretic tool selection | ✗ | ✗ | ✗ | **✓** |
| Encoding-depth memory | ✗ | ✗ | ✗ | **✓** |
| Active forgetting | ✗ | ✗ | ✗ | **✓** |
| Cross-session relationship model | ✗ | ✗ | ✗ | **✓** |
| Free energy objective | ✗ | ✗ | ✗ | **✓** |

---

## Theoretical Foundations

Borge is grounded in peer-reviewed cognitive science — not intuition.

| Paper | Year | What it contributes |
|-------|------|---------------------|
| Ebbinghaus, *Memory: A contribution to experimental psychology* | 1885 | Forgetting curve → active memory decay |
| Yerkes & Dodson | 1908 | Arousal × performance → optimal arousal window |
| Tulving, *Episodic and semantic memory* | 1972 | Memory taxonomy → 3-tier architecture |
| Craik & Lockhart, *Levels of processing* | 1972 | Encoding depth → emotional significance drives consolidation |
| Baddeley & Hitch, *Working memory* | 1974 | Central executive → MetaAgent design |
| Russell, *A circumplex model of affect* | 1980 | 2D emotion space → V × A state |
| Knill & Pouget, *The Bayesian brain* | 2004 | Predictive coding → belief state |
| Friston, *The free-energy principle* | 2010 | Unified objective → F_total |
| Friston et al., *Active inference* | 2017 | EFE tool ranking |

Full derivations in [`docs/borge-agent-design.md`](../docs/borge-agent-design.md).

---

## Roadmap

```
v0.1  ██████████ done   Core cognitive layer + Hermes plugin integration
v0.2  ░░░░░░░░░░        LLM-backed Bayesian update (true likelihood estimation)
v0.2  ░░░░░░░░░░        pre_tool_call hook in Hermes for real-time EFE scoring
v0.3  ░░░░░░░░░░        Multi-agent emotional contagion
v0.3  ░░░░░░░░░░        Counterfactual belief revision
v0.4  ░░░░░░░░░░        SOUL.md auto-tuning from session telemetry
v0.5  ░░░░░░░░░░        Benchmark: cognitive coherence on 100-turn tasks
```

---

## Contributing

The best contributions right now:

- **Empirical validation** — compare Borge vs vanilla on long-horizon coding tasks
- **Richer signal extraction** — better linguistic rules for tone detection
- **Alternative emotion models** — PAD (3D), OCC model, basic emotions
- **LLM likelihood estimator** — replace heuristic Bayesian updates with real LLM calls

```bash
git clone https://github.com/zhibao-dev/hermes-agent
cd hermes-agent && pip install -e ".[dev]"
python -c "from borge.agent import BorgeAgent; a = BorgeAgent(None); print(a.pre_turn('hello', []))"
```

---

## Citation

```bibtex
@software{borge2026,
  title   = {Borge Agent: Cognitively-Grounded AI Agent Architecture},
  year    = {2026},
  url     = {https://github.com/zhibao-dev/hermes-agent},
  note    = {Free Energy Principle + Bayesian inference + cognitive memory}
}
```

---

## License

MIT. Built on [Hermes Agent](https://github.com/NousResearch/hermes-agent) by Nous Research.

---

<div align="center">

**[Design Doc](../docs/borge-agent-design.md) · [Issues](../../issues) · [Discussions](../../discussions)**

<br>

*Most agents are fast. Borge is present.*

</div>
