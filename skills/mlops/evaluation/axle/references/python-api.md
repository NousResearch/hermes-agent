# AXLE Python API Reference

## Client

```python
from axle import AxleClient

client = AxleClient(
    api_key=None,              # fallback: AXLE_API_KEY
    url=None,                  # fallback: AXLE_API_URL (default: https://axle.axiommath.ai)
    max_concurrency=None,      # fallback: AXLE_MAX_CONCURRENCY (default: 20)
    base_timeout_seconds=None, # fallback: AXLE_TIMEOUT_SECONDS (default: 1800)
)
```

Always use as async context manager: `async with AxleClient() as client:`

The `base_timeout_seconds` is the retry window for 503/429 errors, NOT the per-request timeout. Per-request timeout is controlled by `timeout_seconds` on each method call (default 120, max 300 for non-admin).

Session internals: keepalive_timeout=120s, DNS cache TTL=300s, trust_env=True (respects HTTP_PROXY).

## Method signatures

### Validation

```python
await client.check(
    content: str,                    # Lean source code
    environment: str,                # e.g., "lean-4.28.0"
    mathlib_linter: bool = False,    # enable Mathlib standard linters
    ignore_imports: bool = False,    # auto-replace imports with environment defaults
    timeout_seconds: float = 120,    # per-request timeout (max 300)
) -> CheckResponse
# Returns: okay, content, lean_messages, tool_messages, failed_declarations, timings, info

await client.verify_proof(
    formal_statement: str,           # sorried theorem to verify against
    content: str,                    # candidate proof
    environment: str,
    permitted_sorries: list[str] | None = None,  # names allowed to have sorry
    mathlib_linter: bool = False,
    use_def_eq: bool = True,         # kernel reduction for type comparison (False = faster)
    ignore_imports: bool = False,
    timeout_seconds: float = 120,
) -> VerifyProofResponse
# Returns: okay, content, failed_declarations, lean_messages, tool_messages, timings, info
# Timings: {total_ms, formal_statement_ms, declarations_ms, candidate_ms}

await client.disprove(
    content: str,
    environment: str,
    names: list[str] | None = None,
    indices: list[int] | None = None,
    terminal_tactics: list[str] | None = None,  # default: ["grind"]
    ignore_imports: bool = False,
    timeout_seconds: float = 120,
) -> DisproveResponse
# Returns: content, lean_messages, tool_messages, results, disproved_theorems, timings, info
```

### Transformation

```python
await client.theorem2sorry(
    content: str,
    environment: str,
    names: list[str] | None = None,
    indices: list[int] | None = None,   # supports negative (-1 = last)
    ignore_imports: bool = False,
    timeout_seconds: float = 120,
) -> Theorem2SorryResponse

await client.theorem2lemma(
    content: str,
    environment: str,
    names: list[str] | None = None,
    indices: list[int] | None = None,
    target: str = "lemma",               # "lemma" or "theorem"
    ignore_imports: bool = False,
    timeout_seconds: float = 120,
) -> Theorem2LemmaResponse

await client.rename(
    content: str,
    declarations: dict[str, str],        # {"old_name": "new_name"}
    environment: str,
    ignore_imports: bool = False,
    timeout_seconds: float = 120,
) -> RenameResponse

await client.normalize(
    content: str,
    environment: str,
    normalizations: list[str] | None = None,  # see below
    failsafe: bool = True,              # return original if normalization breaks code
    ignore_imports: bool = False,
    timeout_seconds: float = 120,
) -> NormalizeResponse
# normalize_stats shows which passes made changes
```

Available normalizations: `remove_sections` (default), `remove_duplicates` (default), `split_open_in_commands` (default), `expand_decl_names`, `normalize_module_comments`, `normalize_doc_comments`

### Analysis & repair

```python
await client.extract_theorems(
    content: str,
    environment: str,
    ignore_imports: bool = False,
    timeout_seconds: float = 120,
) -> ExtractTheoremsResponse
# Returns: content, documents (dict[str, Document]), lean_messages, tool_messages, timings, info
# No okay field. No names/indices filtering — extracts ALL theorems.

await client.simplify_theorems(
    content: str,
    environment: str,
    names: list[str] | None = None,
    indices: list[int] | None = None,
    simplifications: list[str] | None = None,  # default: all
    ignore_imports: bool = False,
    timeout_seconds: float = 120,
) -> SimplifyTheoremsResponse
# simplification_stats shows counts per strategy

await client.repair_proofs(
    content: str,
    environment: str,
    names: list[str] | None = None,
    indices: list[int] | None = None,
    repairs: list[str] | None = None,            # default: all
    terminal_tactics: list[str] | None = None,   # default: ["grind"]
    ignore_imports: bool = False,
    timeout_seconds: float = 120,
) -> RepairProofsResponse
# Returns: okay, content, repair_stats, lean_messages, tool_messages, timings, info

await client.merge(
    documents: list[str],               # list of Lean code strings
    environment: str,
    use_def_eq: bool = True,
    include_alts_as_comments: bool = False,
    ignore_imports: bool = False,
    timeout_seconds: float = 120,
) -> MergeResponse
# No okay field.
```

Available simplifications: `remove_unused_tactics`, `remove_unused_haves`, `rename_unused_vars`

Available repairs: `remove_extraneous_tactics`, `apply_terminal_tactics`, `replace_unsafe_tactics`

### Extraction

```python
await client.have2lemma(
    content: str,
    environment: str,
    names: list[str] | None = None,
    indices: list[int] | None = None,
    include_have_body: bool = False,       # include proof (may introduce errors)
    include_whole_context: bool = True,    # False = minimize context (may miss deps)
    reconstruct_callsite: bool = False,    # replace have with lemma call (fragile)
    verbosity: float = 0,                 # 0=default, 1=robust, 2=extra robust (pp.explicit)
    ignore_imports: bool = False,
    timeout_seconds: float = 120,
) -> Have2LemmaResponse
# Returns: content, lemma_names, lean_messages, tool_messages, timings, info

await client.have2sorry(
    content: str,
    environment: str,
    names: list[str] | None = None,
    indices: list[int] | None = None,
    ignore_imports: bool = False,
    timeout_seconds: float = 120,
) -> Have2SorryResponse

await client.sorry2lemma(
    content: str,
    environment: str,
    names: list[str] | None = None,
    indices: list[int] | None = None,
    extract_sorries: bool = True,
    extract_errors: bool = True,
    include_whole_context: bool = True,
    reconstruct_callsite: bool = False,
    verbosity: float = 0,
    ignore_imports: bool = False,
    timeout_seconds: float = 120,
) -> Sorry2LemmaResponse
# Returns: content, lemma_names, lean_messages, tool_messages, timings, info
```

### Utility

```python
await client.environments(timeout_seconds=None) -> list[dict]
# Each: {name, lean_toolchain, repo_url, revision, subdir, imports, description}

client.check_status(timeout_seconds=60) -> dict
# SYNCHRONOUS (uses requests, not aiohttp). Health check at GET /v1/status.

await client.run_one(method: str, request: dict) -> dict
# Low-level: call any tool by name, returns raw dict (not typed response)
```

## Response types

All importable from `axle`:

```python
from axle import (
    CheckResponse, VerifyProofResponse, ExtractTheoremsResponse,
    RenameResponse, MergeResponse, Theorem2LemmaResponse, Theorem2SorryResponse,
    SimplifyTheoremsResponse, RepairProofsResponse, Have2LemmaResponse,
    Have2SorryResponse, Sorry2LemmaResponse, DisproveResponse, NormalizeResponse,
    Document,  # per-theorem metadata from extract_theorems
)
# Messages type (NOT in __all__):
from axle.types import Messages  # .errors, .warnings, .infos (all list[str])
```

Every response has an `info` field (always present):
```json
{
  "request_id": "uuid-v4",
  "environment": "lean-4.28.0",
  "total_request_time_ms": 47,
  "queue_time_ms": 4,
  "execution_time_ms": 42,
  "cached_response": false
}
```

Identical requests return `cached_response: true` with faster times.

## Error handling

```python
from axle.exceptions import AxleError       # Root base (NOT in __all__)
from axle import (
    AxleApiError,          # Base for API errors (.status_code, may be None)
    AxleIsUnavailable,     # 503 — auto-retried (.url, .details attributes)
    AxleRateLimitedError,  # 429 — auto-retried with exponential backoff
    AxleInvalidArgument,   # 400
    AxleForbiddenError,    # 403
    AxleNotFoundError,     # 404
    AxleConflictError,     # 409
    AxleInternalError,     # 500
    AxleRuntimeError,      # operation failure (timeout, OOM)
)
```

Hierarchy:
- `AxleError` → `AxleIsUnavailable` (retryable, NOT subclass of AxleApiError)
- `AxleError` → `AxleApiError` → all other exceptions

Auto-retried with exponential backoff (1-15s jitter): 503, 429, connection errors.
NOT retried: 400, 403, 404, 409, 500.

## Helper functions

```python
from axle import remove_comments, inline_lean_messages

# Remove comments from Lean code
clean = remove_comments(
    code,
    include_module_docs=False,   # keep /-! ... -/ if True
    include_docstrings=False,    # keep /-- ... -/ if True
)
# Handles: line comments (--), block comments (/- -/), nested blocks,
# module docs (/-! -/), docstrings (/-- -/), string literal awareness

# Inline compiler messages as comments at source locations
annotated = inline_lean_messages(
    code,
    messages=["file.lean:3:0: error: type mismatch"],
    prefix="/- ",
    suffix=" -/",
)
# Parses file:line:col format, inserts after corresponding lines
# Line 0 messages prepended, beyond-end messages appended
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `AXLE_API_KEY` | none | API key (get at axle.axiommath.ai/app/console) |
| `AXLE_API_URL` | `https://axle.axiommath.ai` | API server URL |
| `AXLE_TIMEOUT_SECONDS` | 1800 | Retry window for 503/429 (NOT per-request timeout) |
| `AXLE_MAX_CONCURRENCY` | 20 | Client-side concurrency limit |
| `AXLE_REQUEST_SOURCE` | `sdk` | Sets X-Request-Source header |
