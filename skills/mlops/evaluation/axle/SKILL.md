---
name: axle
description: >-
  AXLE (Axiom Lean Engine) — CLI and Python client for Lean 4 proof manipulation
  via the Axiom Math cloud API. Use when verifying proofs, checking Lean code,
  extracting/merging/renaming theorems, repairing broken proofs, disproving
  statements, or normalizing Lean files. 14 tools for Lean 4 + Mathlib workflows.
version: 1.0.0
author: Orchestra Research
license: MIT
dependencies: [axiom-axle]
metadata:
  hermes:
    tags: [Lean 4, Theorem Proving, Mathlib, Formal Verification, Proof Assistant, AXLE, Axiom Math, CLI, Python API]
---

# AXLE - Lean 4 Proof Manipulation

AXLE provides 14 tools for validating, analyzing, and transforming Lean source code — all running as Lean metaprograms on the Axiom Math cloud infrastructure.

## When to use AXLE

**Use AXLE when:**
- Verifying a candidate proof against a formal statement
- Checking if Lean 4 code compiles (remote, no local Lean install needed)
- Extracting individual theorems from a multi-theorem file with dependency metadata
- Merging multiple Lean files with deduplication and conflict resolution
- Repairing broken proofs after a Mathlib version bump
- Simplifying proofs by removing unused tactics
- Creating problem sets by stripping proofs to sorry
- Attempting to disprove false statements
- Normalizing Lean file formatting before merge operations

**Use alternatives instead:**
- **Local Lean/Lake**: When you need a full local development environment or IDE integration
- **lean4checker/Comparator/SafeVerify**: When verifying untrusted/adversarial code (AXLE's verify-proof trusts the Lean environment and can be exploited via metaprogramming)
- **Pantograph**: When you need machine-to-machine proof interaction beyond AXLE's tool set

## Quick start

**Install**: `axiom-axle` from PyPI (requires Python 3.11+).

Set your API key (optional, increases concurrent request limit from 10 to 20):
```bash
export AXLE_API_KEY=your_key_from_https://axle.axiommath.ai/app/console
```

**Check if code compiles:**
```bash
echo 'theorem foo : 1 + 1 = 2 := by norm_num' \
  | axle check --environment lean-4.28.0 --ignore-imports -
```

**Verify a proof:**
```bash
echo 'theorem foo : 1 + 1 = 2 := by sorry' > statement.lean
echo 'theorem foo : 1 + 1 = 2 := by norm_num' > proof.lean
axle verify-proof --environment lean-4.28.0 --ignore-imports statement.lean proof.lean
```

**Python async client:**
```python
from axle import AxleClient

async with AxleClient() as client:
    result = await client.check(
        content="theorem foo : 1 + 1 = 2 := by norm_num",
        environment="lean-4.28.0",
        ignore_imports=True,
    )
    print(result.okay)  # True
```

## Common workflows

### Workflow 1: Verify proof attempts

Validate that a candidate proof correctly proves a formal statement.

```text
Proof Verification:
- [ ] Step 1: Write formal statement (sorried)
- [ ] Step 2: Write or generate candidate proof
- [ ] Step 3: Run verify-proof
- [ ] Step 4: Check results
```

**Step 1–2: Prepare files**

```bash
# Statement file (with sorry placeholder)
cat > statement.lean << 'EOF'
theorem add_comm_nat : ∀ (a b : Nat), a + b = b + a := by sorry
EOF

# Candidate proof
cat > proof.lean << 'EOF'
theorem add_comm_nat : ∀ (a b : Nat), a + b = b + a := by
  intros a b; omega
EOF
```

**Step 3: Verify**

```bash
# Both args are FILE PATHS (not inline strings!)
axle verify-proof --environment lean-4.28.0 --ignore-imports \
  statement.lean proof.lean

# Or: statement from file, proof from stdin
cat proof.lean | axle verify-proof --environment lean-4.28.0 \
  --ignore-imports statement.lean -
```

**Step 4: Interpret results**

Output includes `okay` (bool), `failed_declarations` (list), and `tool_messages.errors` with specific failure reasons like "Missing required declaration", "Theorem does not match expected signature", or "Declaration uses 'sorry'".

### Workflow 2: Repair proofs after Mathlib update

Fix proofs that broke due to a Lean/Mathlib version change.

```text
Proof Repair:
- [ ] Step 1: Normalize the file
- [ ] Step 2: Run repair-proofs
- [ ] Step 3: Verify repairs
```

**Step 1: Normalize**

```bash
axle normalize --environment lean-4.28.0 --ignore-imports \
  broken.lean -o normalized.lean
```

**Step 2: Repair**

```bash
# Default terminal tactic is grind; customize with --terminal-tactics
axle repair-proofs --environment lean-4.28.0 --ignore-imports \
  normalized.lean -o repaired.lean --strict
```

Three repair strategies run by default:
- `remove_extraneous_tactics` — remove tactics after proof is complete
- `apply_terminal_tactics` — replace sorry with grind (or custom tactics)
- `replace_unsafe_tactics` — replace native_decide with "decide +kernel"

**Step 3: Verify**

```bash
axle check --environment lean-4.28.0 --ignore-imports repaired.lean --strict
```

### Workflow 3: Merge multiple proof files

Combine separate Lean files with intelligent deduplication.

```text
File Merge:
- [ ] Step 1: Normalize each file
- [ ] Step 2: Merge
- [ ] Step 3: Review output
```

**Step 1: Normalize**

```bash
for f in *.lean; do
  axle normalize --environment lean-4.28.0 --ignore-imports "$f" -o "norm_$f"
done
```

**Step 2: Merge**

```bash
axle merge --environment lean-4.28.0 --ignore-imports \
  norm_*.lean -o merged.lean
```

The merge algorithm handles: topological ordering, auto-renaming conflicts (A → A_1), deduplication of identical types (even across different names via type_hash), preference for error-free/sorry-free versions, and preserving failed attempts as comments. Use `--include-alts-as-comments` to keep all versions.

## CLI reference

All commands accept Lean code via **file path** or **stdin** (`-`). Positional args are always file paths, never inline strings.

**Global flags** (must go before subcommand): `--json`, `--url URL`, `--version`

**Common flags**: `--environment` (required), `--ignore-imports`, `--timeout-seconds N`, `--strict` (exit code 3 on failure), `-o FILE`

**Exit codes**: 0=success, 1=failure, 2=file exists, 3=strict validation failed, 130=interrupted

### Validation tools

```bash
# Check if code compiles (also works as REPL: #check, #eval output in lean_messages.infos)
axle check --environment lean-4.28.0 --ignore-imports FILE_OR_-
# Options: --mathlib-linter --strict

# Verify proof against formal statement (both args are files)
axle verify-proof --environment lean-4.28.0 --ignore-imports STATEMENT.lean PROOF.lean
# Options: --permitted-sorries name1,name2  --no-use-def-eq  --mathlib-linter  --strict

# Attempt to disprove theorems (prove the negation via Plausible)
axle disprove --environment lean-4.28.0 --ignore-imports FILE_OR_-
# Options: --names X  --indices X  --terminal-tactics tac1,tac2
```

### Transformation tools

```bash
# Strip proofs, replace with sorry (create problem sets)
axle theorem2sorry --environment lean-4.28.0 --ignore-imports FILE_OR_-
# Options: --names X  --indices 0,-1  -o FILE

# Convert theorem ↔ lemma keywords
axle theorem2lemma --environment lean-4.28.0 --ignore-imports --target lemma FILE_OR_-
# Options: --names X  --indices X  --target theorem|lemma

# Rename declarations (updates all references)
axle rename --environment lean-4.28.0 --ignore-imports \
  --declarations '{"old_name": "new_name"}' FILE_OR_-
# Also: --declarations old=new,foo=bar  or  --declarations-file mapping.json

# Normalize formatting (run before merge)
axle normalize --environment lean-4.28.0 --ignore-imports FILE_OR_- -o FILE
# Options: --normalizations remove_sections,expand_decl_names  --no-failsafe
# Defaults: remove_sections, remove_duplicates, split_open_in_commands
# Available: + expand_decl_names, normalize_module_comments, normalize_doc_comments
```

### Analysis tools

```bash
# Extract individual theorems with full dependency metadata
axle extract-theorems --environment lean-4.28.0 --ignore-imports FILE_OR_- -d output/ -f
# Each theorem: signature, type, type_hash, tactic_counts, proof_length, 6 dependency lists

# Simplify proofs (remove unused tactics/haves, rename unused vars)
axle simplify-theorems --environment lean-4.28.0 --ignore-imports FILE_OR_-
# Options: --simplifications remove_unused_tactics,remove_unused_haves,rename_unused_vars

# Repair broken proofs
axle repair-proofs --environment lean-4.28.0 --ignore-imports FILE_OR_- --strict
# Options: --repairs remove_extraneous_tactics,apply_terminal_tactics,replace_unsafe_tactics
#          --terminal-tactics grind,omega,simp
```

### Extraction tools

```bash
# Extract have statements to standalone lemmas
axle have2lemma --environment lean-4.28.0 --ignore-imports FILE_OR_-
# Options: --include-have-body  --no-include-whole-context  --reconstruct-callsite
#          --verbosity 0|1|2  --names X  --indices X

# Replace have bodies with sorry
axle have2sorry --environment lean-4.28.0 --ignore-imports FILE_OR_-

# Extract sorry/error positions to standalone lemmas
axle sorry2lemma --environment lean-4.28.0 --ignore-imports FILE_OR_-
# Options: --no-extract-sorries  --no-extract-errors  --reconstruct-callsite

# Merge multiple files
axle merge --environment lean-4.28.0 --ignore-imports f1.lean f2.lean -o merged.lean
# Options: --include-alts-as-comments  --no-use-def-eq
```

### Utility

```bash
# List available Lean environments
axle environments
# Available: lean-4.21.0 through lean-4.28.0 (8 versions, all include Mathlib)
```

## Python API

See [references/python-api.md](references/python-api.md) for full method signatures and response types.

```python
from axle import AxleClient

async with AxleClient(
    api_key=None,              # fallback: AXLE_API_KEY env var
    url=None,                  # fallback: AXLE_API_URL (default: https://axle.axiommath.ai)
    max_concurrency=None,      # fallback: AXLE_MAX_CONCURRENCY (default: 20)
    base_timeout_seconds=None, # fallback: AXLE_TIMEOUT_SECONDS (default: 1800, retry window)
) as client:
    # Validation
    r = await client.check(content=code, environment="lean-4.28.0", ignore_imports=True)
    r = await client.verify_proof(formal_statement=stmt, content=proof, environment="lean-4.28.0")
    r = await client.disprove(content=code, environment="lean-4.28.0")

    # Transformation
    r = await client.theorem2sorry(content=code, environment="lean-4.28.0")
    r = await client.theorem2lemma(content=code, environment="lean-4.28.0", target="lemma")
    r = await client.rename(content=code, declarations={"old": "new"}, environment="lean-4.28.0")
    r = await client.normalize(content=code, environment="lean-4.28.0")

    # Analysis & repair
    r = await client.extract_theorems(content=code, environment="lean-4.28.0")
    r = await client.simplify_theorems(content=code, environment="lean-4.28.0")
    r = await client.repair_proofs(content=code, environment="lean-4.28.0")
    r = await client.merge(documents=[code1, code2], environment="lean-4.28.0")

    # Extraction
    r = await client.have2lemma(content=code, environment="lean-4.28.0")
    r = await client.have2sorry(content=code, environment="lean-4.28.0")
    r = await client.sorry2lemma(content=code, environment="lean-4.28.0")

    # Utility
    envs = await client.environments()
    status = client.check_status()  # synchronous health check
```

**Helper functions** (exported from `axle`):
```python
from axle import remove_comments, inline_lean_messages

# Strip comments from Lean code (handles nesting, strings, module docs)
clean = remove_comments(code, include_module_docs=False, include_docstrings=False)

# Inline compiler messages as comments at source locations
annotated = inline_lean_messages(code, messages=result.lean_messages.errors)
```

## Common issues

**Issue: "Imports mismatch: expected '[Mathlib]', got '[]'"**

Add `--ignore-imports` to all commands when your code doesn't start with `import Mathlib`:
```bash
axle check --environment lean-4.28.0 --ignore-imports FILE
```

**Issue: CLI positional argument treated as file path**

All positional args are file paths. Write content to a temp file or pipe via stdin:
```bash
# Wrong: axle check --environment lean-4.28.0 'theorem foo ...'
# Right:
echo 'theorem foo : True := trivial' | axle check --environment lean-4.28.0 --ignore-imports -
```

**Issue: verify-proof says "Missing required declaration" but code is correct**

Both args to verify-proof must be file paths. The first arg cannot be stdin:
```bash
# Wrong: echo '...' | axle verify-proof --environment lean-4.28.0 - -
# Right: use a file for statement, stdin for proof
axle verify-proof --environment lean-4.28.0 --ignore-imports statement.lean -
```

**Issue: okay=false even though code compiles in Lean**

AXLE is stricter than the Lean compiler. Check `tool_messages.errors` for reasons: native_decide usage, sorry in proofs, type mismatch, banned `open private`, or non-standard axioms.

**Issue: "All executors failed after N attempts"**

Usually an OOM condition. Simplify input, break into smaller files, or reduce timeout:
```bash
axle check --environment lean-4.28.0 --ignore-imports --timeout-seconds 300 FILE
```

**Issue: --json flag not recognized**

`--json` is a global flag that must go before the subcommand:
```bash
# Wrong: axle check --json --environment ...
# Right: axle --json check --environment ...
```

**Issue: merge produces conflicting non-declaration commands**

`set_option`, `variable`, and `maxHeartbeats` from different files may conflict. Always normalize before merging:
```bash
axle normalize --environment lean-4.28.0 --ignore-imports file.lean -o norm_file.lean
```

## Advanced topics

**verify-proof security**: AXLE trusts the Lean environment. A creative adversary can exploit Lean metaprogramming to make invalid proofs appear valid. For untrusted code, use lean4checker, Comparator, or SafeVerify instead. See [references/security.md](references/security.md).

**Extract-theorems metadata**: Each extracted theorem includes 20+ fields — type_hash for dedup, 6 dependency lists (local/external × type/value/syntactic), tactic_counts, proof_length, and standalone compilation status. See [references/extract-theorems-metadata.md](references/extract-theorems-metadata.md).

**HTTP API**: Direct REST calls to `POST /api/v1/<tool_name>` with JSON body. Environments at `GET /v1/environments`, health at `GET /v1/status`. Auth via `Authorization: Bearer <key>` header. See [references/http-api.md](references/http-api.md).

**Error handling**: Python exceptions: `AxleIsUnavailable` (503, auto-retried), `AxleRateLimitedError` (429, auto-retried), `AxleInvalidArgument` (400), `AxleForbiddenError` (403), `AxleInternalError` (500), `AxleRuntimeError`. Base: `AxleApiError`. Root: `AxleError` (import from `axle.exceptions`). See [references/python-api.md](references/python-api.md).

## Resources

- **Docs**: https://axle.axiommath.ai/v1/docs/
- **GitHub**: https://github.com/AxiomMath/axiom-lean-engine
- **Console (API keys)**: https://axle.axiommath.ai/app/console
- **Web UI**: https://axle.axiommath.ai/
- **MCP server**: axiom-axle-mcp (separate package)
- **Lean 4**: https://lean-lang.org/
- **Mathlib**: https://leanprover-community.github.io/mathlib4_docs/
