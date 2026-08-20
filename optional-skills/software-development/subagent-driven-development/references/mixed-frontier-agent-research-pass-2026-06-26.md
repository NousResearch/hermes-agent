# Mixed Claude/Codex Research-Implementation Pass (2026-06-26)

Use this reference when executing a broad implementation plan across a large Rust workspace where independent phases can be implemented by different frontier coding agents, but the controller must own verification and receipts.

## Session pattern that worked

Context: `/path/to/workspace` was already heavily dirty. The active plan excluded HuggingFace and CUDA. The user explicitly wanted Claude and Codex used where available.

Controller flow:

1. Establish scope and tree safety first.
   - Write a starting-tree receipt before feature work.
   - Record that the repo is dirty; do not pretend full-tree closure.
   - Run a baseline on the first target crate before implementation.

2. Convert the plan into independent, file-scoped agent briefs.
   - Each brief listed exact allowed files and forbidden files.
   - Each brief had required public API, behavior, tests, and cargo commands.
   - Each brief said: no commit, no HuggingFace, no CUDA, no performance claims.

3. Run mixed agents in parallel.
   - Claude Code handled algorithmic/new-module tasks:
     - `fib-quant/src/rope.rs`
     - `semantic-memory/src/hubness.rs`
     - `fib-quant/src/lattice.rs`
     - `llm-output-parser/src/cypher.rs`
     - perspective/reinstatement building blocks
   - Codex handled fixture/receipt/model tasks:
     - `quant-eval/src/rag.rs`
     - `bitemporal-runtime` graph edge model
     - diagnostic/formal-check receipt types

4. Treat agent summaries as self-reports only.
   - Re-read important files in the controller.
   - Re-run every cargo command independently in the controller.
   - Run a scoped security scan over new Rust files.
   - Save semantic-memory facts only after controller verification.

5. Keep receipts live, not afterthoughts.
   - Write baseline and control docs at the start.
   - Write/update a final receipt after all phases.
   - Include explicit claim boundaries and explicit not-done items.

## Brief template

```text
# Task: <one phase>

Repo: <absolute path>
Crate: <crate>

Context:
- Heavy dirty tree. Do not touch unrelated files.
- No HuggingFace, no CUDA, no public paper-performance claims.
- Additive experimental API only.

Files you MAY touch:
- <exact path>
- <exact path>

Files you MUST NOT touch:
- Cargo.toml unless absolutely necessary
- <unrelated crates>

Required API:
<exact structs/functions/enums>

Behavior:
<deterministic rules and edge cases>

Tests required:
- <test 1>
- <test 2>

Run:
```bash
cd <repo>
cargo fmt -p <crate>
cargo test -p <crate> <filter> -- --nocapture
cargo check -p <crate> --all-targets
```

Return summary with exact files changed and command results. Do not commit.
```

## Controller verification checklist

For each agent result:

```bash
cargo test -p <crate> <filter> -- --nocapture
cargo check -p <crate> --all-targets
```

Then run a scoped scan, not a hand-wavy "looks safe":

```bash
python3 - <<'PY'
from pathlib import Path
import re
files = [
    'crate/src/new_file.rs',
]
patterns = [
    re.compile(r'(api_key|secret|password|token|passwd)\s*=\s*[\"\'][^\"\']{6,}[\"\']', re.I),
    re.compile(r'os\.system\(|subprocess.*shell=True|\beval\(|\bexec\(|pickle\.loads?\(', re.I),
    re.compile(r'execute\(f\"|\.format\(.*SELECT|\.format\(.*INSERT', re.I),
]
found=[]
for f in files:
    p=Path(f)
    if not p.exists():
        continue
    for i,line in enumerate(p.read_text(errors='ignore').splitlines(),1):
        for pat in patterns:
            if pat.search(line):
                found.append((f,i,line))
if found:
    for f,i,line in found:
        print(f'{f}:{i}:{line}')
    raise SystemExit(1)
print('security scan: no matches')
PY
```

## Claim-boundary discipline

When translating research into code, never let a prototype imply the paper's measured result.

Examples from this pass:

- RoPE block-energy allocator: safe claim = "experimental CPU-side infrastructure"; blocked claim = "Block-GTQ-equivalent quality."
- Local RAG fixture metrics: safe claim = "local fixture harness"; blocked claim = "TREC reproduction."
- Hubness scoring: safe claim = "building block"; blocked claim = "recall improved by X%."
- Lattice prototype: safe claim = "A2-shaped pair prototype"; blocked claim = "full hexagonal nearest-lattice quantization."
- Cypher parser: safe claim = "parser/extractor only"; blocked claim = "safe query execution."

## Semantic-memory capture pattern

After controller verification, save one durable fact per shipped building block in the project namespace. Include:

- date
- exact file/API
- verification command and result
- claim boundary
- evidence refs to file path and agent log

Do not save task progress before verification. Do not save "phase done" without receipts.

## Pitfalls

- Do not let parallel agents all edit broad shared files unless unavoidable. `lib.rs` exports are okay, but the controller should inspect the final diff.
- Do not trust `git diff --stat` for untracked files; it only reports tracked diffs. Use `git status --short -- <paths>` and `wc -l` for new files.
- A shell security scan with complicated quote regex can fail due quoting; use a small Python script instead.
- If the tree starts dirty, say so in the receipt and keep claims scoped. Do not run broad rollback commands.
