# Codebase navigability eval

Measures what a codebase costs an **LLM agent** to work in, as opposed to what it costs the CPU.
Built for the Sep 2026 decomposition (PR #102117) and kept so future refactors are held to the same
numbers. Everything here is offline and deterministic; no model calls.

## The three questions it answers

1. **How much code has to be read to see one definition?** (`bench.py`)
   Workload = every `from <first-party module> import <Name>` in `tests/`, resolved through re-exports
   to the module that *defines* the name. That is ~19k real "locate X" tasks nobody hand-picked.
   Per task: tokens of the defining file, tokens of the symbol itself, overhead = file − symbol,
   whether the file fits a 32k / 128k context, how many 2,000-line `read_file` windows it spans,
   how many unrelated top-level siblings sit in the same file, and the symbol's cyclomatic complexity.
   Tokens are real (`tiktoken` `o200k_base`); falls back to bytes/4 if tiktoken is missing and says so.

2. **What does a careful agent actually pay to look a symbol up?** (`lookup_sim.py`)
   Simulates the policy a good model follows with our tools: `grep -n` for the definition (1 call),
   `read_file` a 200-line window around the hit (1 call), page forward in 2,000-line windows only while
   the definition is still running. Charges tool calls and returned tokens. Same random sample of symbols
   that exist on BOTH trees, so it is a paired comparison.

3. **Shape and runtime.** (`static_metrics.py`, `runtime_bench.py`)
   LOC split (code/comment/docstring), file and function size distributions, elif-chain lengths,
   nesting depth, radon CC/MI, import graph (edges, fan-in/out, Tarjan SCC cycles); fresh-interpreter
   import time / module count / RSS for the entry points, CLI end-to-end, in-process hot paths, pytest
   collection, bytecode footprint. `runtime_bench.py` pins `PYTHONPATH` to the tree and asserts no
   module resolved from another checkout (an editable install will silently cross-contaminate otherwise).

## Usage

```bash
# deps: tiktoken + radon (bench venv or the project venv)
uv pip install tiktoken radon

# 1 + 2: pass two checkouts (git worktree add is the easy way to get the baseline)
git worktree add /tmp/base origin/main
python evals/codebase_navigability/bench.py /tmp/base base --out out/
python evals/codebase_navigability/bench.py .        head --out out/
python evals/codebase_navigability/lookup_sim.py /tmp/base . --sample 4000 --out out/

# 3
NAV_OUT=out/ python evals/codebase_navigability/static_metrics.py /tmp/base base
NAV_OUT=out/ python evals/codebase_navigability/static_metrics.py .        head
NAV_OUT=out/ python evals/codebase_navigability/runtime_bench.py  /tmp/base base 9
NAV_OUT=out/ python evals/codebase_navigability/runtime_bench.py  .        head 9
```

`bench.py` and `static_metrics.py` take ~2 min each on a 1M-line tree; `lookup_sim.py` ~10 min for
4,000 symbols (it tokenizes every window it "reads"); `runtime_bench.py` ~4 min per tree at 9 reps.

## Reading the results honestly

- `bench.py` measures the **naive** cost (read the whole defining file). It is the number that drops
  ~3× when god files are split, and the one that decides whether a file fits a context window at all.
- `lookup_sim.py` measures the **skilled** cost. It barely moves with a file split, because grep + a
  200-line window already dodges file size. What moves it is (a) the definition itself getting shorter
  and (b) token *density* of the surrounding code. Stripping comments makes each line denser, so a fixed
  200-line window costs more tokens after a comment-stripping refactor even though the code is smaller.
  Both effects are real and pull in opposite directions; report both numbers, not the flattering one.
- Import time goes **up** with a split in Python (per-module overhead), and the import graph's largest
  cycle typically grows (intra-file coupling becomes inter-module edges). Neither is hidden by these tools.

Results for PR #102117 are in the PR description; raw JSON for that run lives in the PR thread.
