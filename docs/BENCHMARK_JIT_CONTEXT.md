# Hermes JIT Context Engine — Empirical Benchmark & Architecture Report

**Author:** Wojciech Wiesner (<wojciech@theones.io>)  
**Repository:** [wojciechwiesner/jit-context](https://github.com/wojciechwiesner/jit-context)  
**CERN Zenodo DOI:** [10.5281/zenodo.22649542](https://doi.org/10.5281/zenodo.22649542)  
**Target:** Hermes Agent Native Context Engine (`context.engine: jit`)

---

## 1. Executive Summary

In long-running autonomous software engineering missions, conventional context management strategies exhibit severe degradation:
1. **Unbounded Context Inflation (Vanilla):** Repeated diffs, terminal outputs, and system tools cause wire payloads to grow linearly to 80k–120k tokens. Small models (7B–9B) hit context limits on Turn 1, while frontier models incur quadratic latency and massive token burn.
2. **Lossy Compression / Summarization (Legacy Compressors):** Calling an auxiliary LLM to summarize previous turns induces hallucinations, forgets precise file paths/line numbers, and loses cryptographic/compliance invariants.

**Hermes JIT Context OS** solves this via a deterministic, multi-tier context engine:
- **L0 Session Overlay:** High-concurrency SQLite WAL storing full lossless execution trace.
- **L1 Scope Hysteresis & AST Working Set:** Dynamic lean context capsule (<1.5k tokens) maintaining active architectural contracts, exact file symbols, and direct user intents.
- **L2 Native Wire Pruning:** Actively selecting and pruning wire messages sent to LLM APIs to keep the active window predictably bounded (3,000–5,000 tokens).

---

## 2. Empirical Benchmark: Wynajmujemy.xyz Backend Implementation

All tests were executed under identical conditions using Hermes Agent on macOS (Apple Silicon), implementing a full production backend module from scratch based on a comprehensive 35KB product specification:
- `models.py`: Pydantic v2 data models, strict types, Grosze PLN rates, SHA-256 draft hashes.
- `compliance.py`: 5D source permission engine, OLX source blocking, DNS SSRF guardrails, Art. 398 PKE anti-cold-outreach.
- `server.py`: FastAPI REST server with draft lifecycles, optimistic concurrency locking, and risk triage.
- `test_wynajmujemy.py`: Full pytest verification suite.

### Comparative Results Matrix

| Metric | 1. Vanilla (No JIT) | 2. Standard JIT (Hook) | 3. JIT Hermes OS (Native) | Local Qwen 3.8 9B (with JIT) |
|---|---|---|---|---|
| **Model** | `gemini-3.8-flash` | `gemini-3.8-flash` | `gemini-3.8-flash` | `qwen3.8:jit` (Metal Q4) |
| **DoD Verification** | **PASS** | **PASS** | **PASS** | **PASS** |
| **Pytest Tests Passed** | 20 / 20 (100%) | 24 / 24 (100%) | **58 / 58 (100%)** | 25 / 25 (100%) |
| **Wall Clock Time** | 527.1s (8.8 min) | **327.9s (5.5 min)** | 386.7s (6.4 min) | 455.0s (7.6 min) |
| **Total Agent Turns** | 85 turns | **37 turns** (-56.5%) | 42 turns (-50.6%) | **14 turns** |
| **Uncached Input Tokens** | 446,500 | 310,461 (-30.5%) | **237,742 (-46.8%)** | N/A (Local Metal) |
| **Total Tokens (Wire + Cache)** | 7,176,703 | **3,113,605** (-56.6%) | 3,469,158 (-51.7%) | N/A (Local Metal) |
| **Estimated Cost (USD)** | $0.9560 | $0.5546 (-42.0%) | **$0.5221 (-45.4%)** | **$0.0000** |

---

## 3. Visual Performance Charts

### Total Wire Token Consumption (Lower is Better)
```
Vanilla (No JIT)        [████████████████████████████████████████] 7,176,703 tokens ($0.956)
Standard JIT (Hook)     [█████████████████                       ] 3,113,605 tokens ($0.555)
JIT Hermes OS (Native)  [█████████████████                       ] 3,469,158 tokens ($0.522) [58 tests!]
```

### Uncached Input Tokens (Immediate Wire Footprint)
```
Vanilla (No JIT)        [████████████████████████████████████████] 446,500 tokens
Standard JIT (Hook)     [██████████████████████████              ] 310,461 tokens (-30.5%)
JIT Hermes OS (Native)  [█████████████████████                   ] 237,742 tokens (-46.8%)
```

### Turn Count / Latency Efficiency
```
Vanilla (No JIT)        [████████████████████████████████████████] 85 turns (527s)
Standard JIT (Hook)     [█████████████████                       ] 37 turns (328s)
JIT Hermes OS (Native)  [███████████████████                     ] 42 turns (387s)
Qwen 3.8 9B (Local JIT) [███████                                 ] 14 turns (455s)
```

---

## 4. The Local Model Paradigm Shift

In standard Hermes Agent, the base system prompt plus full tool schemas spans **35,000 to 43,600 tokens**.
- For local models (e.g. Qwen 2.5 Coder 7B, Qwen 3.8 9B) running on consumer hardware via Ollama / llama.cpp with 16k or 32k context windows:
  - **Without JIT:** The model crashes on Turn 1 with `Context length exceeded` or forces an uncontrolled fallback to cloud providers.
  - **With JIT Hermes OS:** Context is actively curated below 5k tokens, enabling 100% autonomous local task completion on Apple Silicon / consumer GPUs with **zero cloud spend and zero data leakage**.

---

## 5. How to Enable in Hermes Agent

In `~/.hermes/config.yaml`:
```yaml
context:
  engine: jit
```
Or via environment variable:
```bash
export HERMES_CONTEXT_ENGINE=jit
```
