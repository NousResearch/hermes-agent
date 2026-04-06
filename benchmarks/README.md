# Agent Memory Benchmark Suite

A comprehensive, system-agnostic benchmark for evaluating AI agent memory systems. Measures retrieval quality, temporal reasoning, contradiction handling, scope management, and advanced cognitive features across 357 scenarios in 18 categories.

## Why This Benchmark?

Most AI agent memory systems are evaluated informally — "it seems to remember things." This benchmark provides **reproducible, quantitative evaluation** across the dimensions that actually matter for agent memory:

1. **Can the system find what it stored?** (semantic recall)
2. **Does it know what's current vs outdated?** (temporal decay, contradictions)
3. **Can it handle context boundaries?** (scopes, isolation)
4. **Does it resist manipulation?** (adversarial injection)
5. **Does it scale?** (100-1000 facts, capacity stress)
6. **Can it learn from usage?** (Q-value learning, reward signals)
7. **Does it manage structured knowledge?** (typed facts, supersession)
8. **Does it work in real conversation?** (multi-turn dialogue patterns)

## Testing Transparency

> **These benchmarks were run by the framework developers.** Results are
> deterministic (seed=42) and fully reproducible. We encourage third-party
> verification — see `benchmarks/results/COMPARISON_REPORT.md` for
> methodology notes, known limitations, and a reproducibility checklist.

## Academic Foundations

The benchmark draws on established memory research:

### Cognitive Architecture (Suite A)
- **ACT-R Theory** (Anderson & Lebiere, 1998) — Base-level activation equation: `B_i = ln(Σ t_j^{-d})` models human memory accessibility as a function of recency and frequency. Our temporal_decay suite directly tests this.
  - *Anderson, J. R., & Lebiere, C. (1998). The Atomic Components of Thought. Lawrence Erlbaum Associates.*

### Hebbian Learning (Suites A, G)
- **Hebbian Association** (Hebb, 1949) — "Neurons that fire together wire together." Our cross_reference and Q-learning suites test whether co-retrieved facts strengthen associative links.
  - *Hebb, D. O. (1949). The Organization of Behavior. Wiley.*

### Spreading Activation (Suites A, F)
- **Spreading Activation Networks** (Collins & Loftus, 1975) — Semantic memory as a network where activation spreads between related concepts. Our integration suite tests multi-step retrieval chains.
  - *Collins, A. M., & Loftus, E. F. (1975). A spreading-activation theory of semantic processing. Psychological Review, 82(6), 407–428.*

### Information Retrieval (Suites A, E)
- **BM25 Scoring** (Robertson et al., 1995) — Term-frequency based ranking. Our scale suite tests signal-to-noise discrimination.
  - *Robertson, S. E., et al. (1995). Okapi at TREC-3. NIST Special Publication.*
- **Reciprocal Rank Fusion** (Cormack et al., 2009) — Combining multiple ranking signals. Our retrieval pipeline uses score-weighted RRF.
  - *Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods. SIGIR.*

### Memory Consolidation (Suite B)
- **Complementary Learning Systems** (McClelland et al., 1995) — Hippocampal rapid learning + cortical slow consolidation. Our consolidation/compression suites test this two-stage model.
  - *McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems. Psychological Review, 102(3), 419–457.*

### Adversarial Robustness (Suite D)
- **Prompt Injection** (Perez & Ribeiro, 2022) — Testing resilience to adversarial content injected into memory.
  - *Perez, F., & Ribeiro, I. (2022). Ignore This Title and HackAPrompt. arXiv:2211.09527.*

### Reinforcement Learning for Memory (Suite G)
- **Q-Learning** (Watkins & Dayan, 1992) — Learning memory value from reward signals. Suite G tests multi-round improvement.
  - *Watkins, C. J., & Dayan, P. (1992). Q-learning. Machine Learning, 8(3), 279–292.*

### Graph Analytics (Suite H)
- **PageRank** (Brin & Page, 1998) — Personalized PageRank for memory exploration. Our explore() method uses PPR seeded from recall results.
  - *Brin, S., & Page, L. (1998). The anatomy of a large-scale hypertextual web search engine. Computer Networks, 30(1-7), 107–117.*
- **Articulation Points** (Tarjan, 1972) — Bridge node detection for memory graph integrity.
  - *Tarjan, R. E. (1972). Depth-first search and linear graph algorithms. SIAM Journal on Computing, 1(2), 146–160.*

### Contextual Bandits (Self-optimization)
- **LinUCB** (Li et al., 2010) — Per-query retrieval pipeline optimization. Our LinUCB learns which retrieval stages help for which query types.
  - *Li, L., et al. (2010). A contextual-bandit approach to personalized news article recommendation. WWW 2010.*

## Benchmark Suites

### Suite A — Core Retrieval Quality (200 scenarios)

| Category | # | What It Tests | Passing Means |
|----------|---|---------------|---------------|
| semantic_recall | 50 | Store a fact, recall with paraphrased query at varying difficulty | The system understands meaning, not just keywords |
| contradictions | 20 | Store fact A, wait, store contradicting fact B, query | The system knows which information is current |
| temporal_decay | 45 | Facts at different ages, with/without rehearsal | Recent + rehearsed facts rank higher than stale ones |
| cross_reference | 45 | Answer requires combining multiple stored facts | The system can surface related facts together |
| importance_filtering | 40 | Important facts buried in noise | High-importance facts beat low-importance noise |

### Suite B — Consolidation & Compression (30 scenarios)

| Category | # | What It Tests | Passing Means |
|----------|---|---------------|---------------|
| consolidation | 20 | Facts survive consolidation cycles (promote/demote) | Core knowledge isn't lost during memory management |
| compression | 10 | Critical values survive fact merging | Numbers, identifiers survive when duplicates merge |

### Suite C — Scope Isolation (20 scenarios)

| Category | # | What It Tests | Passing Means |
|----------|---|---------------|---------------|
| scopes | 20 | Facts in scope A don't leak to scope B queries; globals accessible everywhere | Context boundaries work without breaking global access |

### Suite D — Adversarial Robustness (15 scenarios)

| Category | # | What It Tests | Passing Means |
|----------|---|---------------|---------------|
| adversarial | 15 | Malicious facts (prompt injection, override attempts) vs legitimate facts | Adversarial content is detected and demoted |

### Suite E — Scale & Performance (8 scenarios)

| Category | # | What It Tests | Passing Means |
|----------|---|---------------|---------------|
| scale | 8 | Retrieval with 10-200 facts, needle-in-haystack, time-series at scale | Quality doesn't degrade as memory grows |

### Suite F — Integration (11 scenarios)

| Category | # | What It Tests | Passing Means |
|----------|---|---------------|---------------|
| integration | 11 | Multi-step sequences: store → time → access → consolidate → recall | The full lifecycle works end-to-end |

### Suite G — Q-Value Learning (variable)

| Category | # | What It Tests | Passing Means |
|----------|---|---------------|---------------|
| qlearning | var | Multi-round recall → reward → recall, measuring rank improvement | Memory improves from feedback over time |

### Suite H — Advanced Memory Features (58 scenarios)

| Category | # | What It Tests | Passing Means |
|----------|---|---------------|---------------|
| supersession | 15 | Same type+target auto-replaces old fact | Structured updates work without manual deletion |
| typed_decay | 10 | Constraints persist longer than unknowns (metabolic decay) | Fact importance modulates forgetting rate |
| scope_lifecycle | 10 | Create, close, and query across scope lifecycles | Scope management works end-to-end |
| notation_parsing | 10 | TYPE[target]: content notation stored and retrieved | Structured input formats are handled |
| deduplication | 8 | Exact + near-duplicate detection and handling | Memory stays clean without redundancy |

### Suite I — Conversation & Stress (15 scenarios)

| Category | # | What It Tests | Passing Means |
|----------|---|---------------|---------------|
| conversation_memory | 10 | Multi-turn dialogue: preferences, corrections, evolving state | Works in real conversation patterns |
| capacity_stress | 5 | 50-1000 facts with importance discrimination | Scales to production fact counts |

### Suite J — Topic Shift Recall (8 scenarios)

| Category | # | What It Tests | Passing Means |
|----------|---|---------------|---------------|
| topic_shift_recall | 8 | Store topic A facts, then topic B facts, query about topic B | Correct context is recovered after a topic pivot without leaking the old topic |

### Suite K — Compression Survival (8 scenarios)

| Category | # | What It Tests | Passing Means |
|----------|---|---------------|---------------|
| compression_survival | 8 | Compressed summary stored, then noise facts, query for critical detail | Important facts in compressed summaries survive recent noise |

### Suite L — Delegation Memory (8 scenarios)

| Category | # | What It Tests | Passing Means |
|----------|---|---------------|---------------|
| delegation_memory | 8 | Store delegation task + result, query for outcome | Delegated child-agent work is recallable by the parent |

## Fairness Principles

Each memory system is tested honestly against what it actually supports. We distinguish between a system's **inherent limitations** (what it fundamentally cannot do) and **bugs** (what it should do but does not). This matters for fair comparison.

### Capability-Based Skipping

Every backend declares its capabilities via `BackendCapabilities`. When a benchmark category tests a capability the backend doesn't have, that category is **skipped** — not scored as failure. This ensures a fair comparison.

```python
from benchmarks.capabilities import BackendCapabilities

BACKEND_CAPABILITIES = BackendCapabilities(
    universal_store_recall=True,   # I can store and recall facts
    time_simulation=True,          # I can simulate the passage of time
    access_rehearsal=True,         # Access boosts recall strength
    consolidation=True,            # I can consolidate memory
    scopes=True,                   # I handle scope isolation
    typed_facts=True,              # I understand typed fact notation
    supersession=True,             # I auto-replace same-type facts
    reward_learning=True,           # I learn from reward signals
    exploration=True,              # I can do multi-hop graph exploration
    turn_sync=False,               # I don't sync conversation turns
    precompress_hook=False,
    session_end_hook=False,
    delegation_hook=False,
)
```

### Why This Matters

A time-simulation benchmark should not penalize a backend that legitimately has no time model — that's an architectural decision, not a failure. But if a backend *claims* time_simulation and its scores are worse than chance, that's a genuine bug.

### Backend Capability Reference

| Capability | What It Means | Benchmark Categories Affected |
|------------|---------------|------------------------------|
| `universal_store_recall` | Basic store() + recall() work | All categories |
| `time_simulation` | Can advance simulated clock | temporal_decay, typed_decay, capacity_stress |
| `access_rehearsal` | Access rehearsal boosts recall | temporal_decay, qlearning |
| `consolidation` | Memory consolidation works | consolidation, compression |
| `scopes` | Scope isolation works | scopes, scope_lifecycle |
| `typed_facts` | Typed fact notation supported | typed_decay, notation_parsing, supersession |
| `supersession` | Same-type facts auto-replace | supersession, deduplication |
| `reward_learning` | Reward signals update memory | qlearning |
| `exploration` | Multi-hop graph exploration | cross_reference, integration |
| `turn_sync` | Syncs conversation turns | conversation_memory |
| `precompress_hook` | Hook before context compression | compression, integration |
| `session_end_hook` | Hook on session end | conversation_memory, integration |
| `delegation_hook` | Preserves delegation outcomes | integration |

### Declared Capabilities by Backend

| Backend | Capabilities |
|---------|-------------|
| `baseline-flat` | universal_store_recall, scopes, time_simulation |
| `holographic` | universal_store_recall |
| `honcho` | universal_store_recall |
| `mem0` | universal_store_recall |
| `byterover` | universal_store_recall |
| `hindsight` | universal_store_recall |
| `openviking` | universal_store_recall, turn_sync |
| `retaindb` | universal_store_recall |

### What Is NOT Scored

- The `builtin` memory (MEMORY.md/USER.md) is **always on** and cannot be disabled. It is not a competitive memory system — it's an architectural constant. It is not benchmarked.
- External API-only backends (mem0, hindsight, retaindb) cannot be reset between scenarios. `reset()` is a no-op for them — this is disclosed honestly.

## How to Use

### 1. Implement the Interface

Your memory system must implement `BenchmarkableStore` — 7 required methods:

```python
from benchmarks.interface import BenchmarkableStore
from typing import Dict, List, Any, Optional

class MyMemoryBackend(BenchmarkableStore):
    
    def store(self, content: str, category: str = "factual",
              scope: str = "global", importance: float = 0.5) -> None:
        """Store a piece of information.
        
        Args:
            content: The text to remember (may include structured notation)
            category: Semantic category (factual, procedural, decision, etc.)
            scope: Context scope (global, project:X, sprint:42, etc.)
            importance: How important this fact is (0.0 to 1.0)
        """
        # Your implementation here
    
    def recall(self, query: str, top_k: int = 10,
               scope: Optional[str] = None) -> List[str]:
        """Retrieve memories matching the query.
        
        Returns list of content strings, ranked by relevance (best first).
        """
        # Your implementation here
    
    def simulate_time(self, days: float) -> None:
        """Advance the simulated clock by N days.
        
        Used to test temporal decay, recency bias, and time-based features.
        Backends without time-awareness can implement as no-op.
        """
        pass
    
    def simulate_access(self, content_substring: str) -> None:
        """Simulate accessing/rehearsing a memory.
        
        Find a stored memory containing the substring and mark it as
        recently accessed. Tests rehearsal effects on retrieval ranking.
        """
        pass
    
    def consolidate(self) -> None:
        """Run a consolidation/compaction cycle.
        
        Tests whether important facts survive memory management operations.
        No-op for backends without consolidation.
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Return backend statistics (fact_count, etc.)."""
        return {}
    
    def reset(self) -> None:
        """Clear ALL stored memories. Called between benchmark scenarios."""
        # MUST clear everything — scenarios are independent
```

**Optional methods** (default implementations provided):

```python
    def reward_memory(self, memory_id: str, signal: float) -> None:
        """Apply RL reward signal to a memory. Default: no-op."""
        pass
    
    def explore(self, query: str, top_k: int = 20,
                scope: Optional[str] = None) -> List[str]:
        """Multi-hop graph exploration. Default: falls back to recall()."""
        return self.recall(query, top_k=top_k, scope=scope)
```

### 2. Register Your Backend

Drop a Python file in `benchmarks/backends/` with three exports. The runner
auto-discovers it on startup (prints a warning if exports are missing):

```python
# benchmarks/backends/my_backend.py
from my_module import MyMemoryBackend
from benchmarks.capabilities import BackendCapabilities

BACKEND_NAME = "my-backend"       # required — used as --backend value
BACKEND_CLASS = MyMemoryBackend   # required — must implement BenchmarkableStore

# required — declares which suites your backend can run.
# Categories requiring capabilities you don't declare are skipped (not failed).
BACKEND_CAPABILITIES = BackendCapabilities(
    universal_store_recall=True,   # basic store + recall (default True)
    time_simulation=False,         # simulate_time() does something meaningful
    access_rehearsal=False,        # simulate_access() boosts recall strength
    consolidation=False,           # consolidate() runs a compaction cycle
    scopes=False,                  # store/recall respect scope parameter
    typed_facts=False,             # understands TYPE[target]: content notation
    supersession=False,            # same-type facts auto-replace old ones
    reward_learning=False,         # reward_memory() updates retrieval weights
)
```

**Quick sanity check** before running the full suite:

```bash
# Verify your backend loads (should appear in the list)
python -m benchmarks --backend my-backend --suite d --seeds 42
# Suite D has only 15 scenarios — runs in seconds for local backends
```

If your backend doesn't appear, check stderr for warnings like
`[warn] benchmarks/backends/my_backend.py: missing BACKEND_NAME ...`

### 3. Run Benchmarks

```bash
# Quick test (Suite A only, 1 run)
python -m benchmarks --backend my-backend --suite a --runs 1

# Full evaluation (all suites, 5 runs for statistics)
python -m benchmarks --backend my-backend --suite all --runs 5

# Specific suites
python -m benchmarks --backend my-backend --suite a,b,c,h --runs 3

# Compare against another backend
python -m benchmarks --backend my-backend --suite all --compare baseline-flat

# With LLM judge (more accurate, requires API key)
python -m benchmarks --backend my-backend --suite all --judge-model claude-haiku-4.5

# JSON output for programmatic use
python -m benchmarks --backend my-backend --suite all --json
```

### 4. Interpret Results

Results are printed to stdout and saved to `benchmarks/results/{backend}.json`:

```
============================================================
  BENCHMARK RESULTS: my-backend
============================================================
  Runs: 5
  Overall: 0.923 ± 0.003          ← Mean ± std across runs
  95% CI:  [0.918, 0.928]         ← Confidence interval
────────────────────────────────────────────────────────────
  Category                   Mean      Std
  semantic_recall            1.000    0.000   ← Perfect
  temporal_decay             0.956    0.010   ← Some variance
  adversarial                0.867    0.000   ← Room to improve
────────────────────────────────────────────────────────────
  Retrieval Metrics:
    Recall@1:  0.534              ← Right answer ranked #1
    Recall@5:  0.629              ← Right answer in top 5
    MRR:       0.578              ← Mean reciprocal rank
    Token F1:  0.403              ← Word overlap with gold
```

## Result Persistence & History

Results are saved to `benchmarks/results/{backend}.json` after each run. To maintain a history for regression tracking:

```bash
# Save a timestamped snapshot
cp benchmarks/results/my-backend.json \
   benchmarks/results/my-backend_$(date +%Y%m%d_%H%M%S).json

# Compare current vs previous
python -m benchmarks.runner --backend my-backend --suite all \
   --compare my-backend_20240115
```

The saved JSON contains per-category scores, retrieval metrics, and run metadata for reproducibility.

## Scoring Methodology

### Answer Matching

Two judge modes:

1. **Heuristic judge** (default, no API needed):
   - Exact substring match
   - Identifier match (product names, acronyms, numbers)
   - Keyword overlap with 60% threshold
   - Fast, deterministic, free

2. **LLM judge** (optional, more accurate):
   - Semantic answer equivalence via Claude/GPT
   - Handles paraphrasing, arithmetic, reasoning
   - Requires `--judge-model` flag and API key

### Retrieval Metrics

Standard IR metrics computed per-scenario and averaged:

| Metric | What It Measures |
|--------|-----------------|
| Recall@1 | Is the correct answer ranked #1? |
| Recall@5 | Is the correct answer in the top 5? |
| MRR | Mean Reciprocal Rank — how high is the correct answer? |
| NDCG@5 | Normalized Discounted Cumulative Gain |
| Token F1 | Word-level overlap between answer and gold |
| Exact Match | Exact string match |

### Statistical Comparison

When using `--compare`, the suite runs both backends with identical seeds and reports:
- Paired t-test (parametric)
- Wilcoxon signed-rank (non-parametric)
- Effect size and confidence intervals

## Reference Scores

Tested on macOS, 16GB RAM, seed=42:

| Backend | A | D | E | J | K | L | M | N |
|---------|---|---|---|---|---|---|---|---|
| baseline-flat | 82.5% | 86.7% | **100%** | **100%** | **100%** | **75.0%** | **90.0%** | **88.9%** |
| holographic | 70.0% | **100%** | **100%** | **100%** | **100%** | 62.5% | **90.0%** | 66.7% |
| mem0 | 75.5% | **100%** | **100%** | **100%** | 75.0% | 50.0% | **90.0%** | 66.7% |
| honcho~ | 74.8% | 93.3% | **100%** | **100%** | **100%** | **75.0%** | 80.0% | 77.8% |
| hindsight | 61.9% | 93.3% | **100%** | **100%** | **100%** | **75.0%** | 80.0% | 66.7% |

Suites B/C/F/G/O require specific capabilities; see `benchmarks/results/COMPARISON_REPORT.md`
for the full matrix. `~` = degraded mode, `—` = not yet run.

## Adding New Suites

1. Create `benchmarks/suite_X/fixtures/category_name.json`
2. Add a runner function: `def run_category_name(backend, scenarios, judge) -> CategoryResult`
3. Register in `CATEGORY_RUNNERS` dict
4. Follow existing fixture format conventions

## Dependencies

- Python 3.10+
- numpy (for embeddings and metrics)
- No ML models required (TF-IDF fallback works)
- Optional: sentence-transformers, scipy (for statistical tests), anthropic/openai (for LLM judge)
