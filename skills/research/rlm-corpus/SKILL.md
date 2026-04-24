---
name: rlm-corpus
description: >
  Answer questions over a local document corpus that is too large for the root
  model's context. Uses the Recursive Language Models pattern (arXiv:2512.24601):
  the corpus is loaded as a Python variable inside a live Jupyter kernel, and the
  root LM writes code to explore/filter/chunk it, optionally dispatching sub-LLM
  calls via `llm_query()` on individual chunks. Supports PDF/Markdown/LaTeX/plain
  text. Never stuffs the full corpus into the root model's context.
version: 0.1.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [Research, RLM, Corpus, PDF, Papers, Jupyter, Recursive-Reasoning]
    related_skills: [arxiv, ocr-and-documents, jupyter-live-kernel]
prerequisites:
  commands: [python3]
  packages: [jupyter_client, ipykernel, anthropic, pymupdf]
  env_vars: [ANTHROPIC_API_KEY]
---

# RLM-Corpus

Recursive-LM agent over a local document corpus. The root LM drives a Python
REPL against the corpus-as-variable; sub-LM calls inside the REPL handle any
chunk that needs semantic work. Built for physics paper corpora (SVT/HHG) but
generalises to any document collection.

Reference: Zhang, Kraska, Khattab. "Recursive Language Models." arXiv:2512.24601.

## Files

| File | Purpose |
|------|---------|
| `ingestion.py` | Directory / URLs / crawls → JSON cache (hash-incremental) |
| `web_fetch.py` | URL-to-markdown backends: Cloudflare Browser Rendering, Jina Reader |
| `corpus_loader.py` | JSON cache → in-memory dict |
| `rlm_engine.py` | Jupyter kernel + agent loop |
| `prompts.py` | System prompt templates |
| `llm_clients.py` | Anthropic / OpenAI / OMLX client factory |
| `config.py` | `RLMConfig` dataclass (env-var aware) |
| `skill.py` | Top-level CLI entry point |

## Two-phase workflow

### 1. Ingest (one-time / incremental)

```bash
python ingestion.py ingest \
  --source ~/physics/svt-corpus/ \
  --cache  ~/.hermes/rlm-cache/svt-corpus/ \
  --backend auto \
  --workers 4
```

- Supported file types: `.pdf`, `.md`, `.markdown`, `.tex`, `.txt`
- Backends: `pymupdf` (default, light), `marker` (heavy, better for math),
  `pypdf` (last-resort fallback). `--backend auto` tries them in order.
- Idempotent — files whose SHA-256 hash hasn't changed are skipped.
- Per-file failures are logged to `_ingest_errors.json`; a bad file does not
  abort the batch.
- `--dry-run` lists what would be processed.

Per-doc output JSON schema: `file_path`, `file_hash`, `ingested_at`, `metadata`
(title / authors / year / doi / source_type / source_url), `full_text`,
`sections` (heading, level, text, char offsets), `references`, `stats`.

#### Ingesting websites

Single pages (or a list of URLs) via the Cloudflare Browser Rendering
`/markdown` endpoint:

```bash
export CLOUDFLARE_ACCOUNT_ID=...   # or RLM_CF_ACCOUNT_ID
export CLOUDFLARE_API_TOKEN=...    # needs "Browser Rendering - Edit"

python ingestion.py ingest-urls \
  --cache ~/.hermes/rlm-cache/my-corpus/ \
  --url   https://example.com/paper-a \
  --url   https://example.com/paper-b

# Or from a newline-separated file (# comments allowed)
python ingestion.py ingest-urls --cache ~/.hermes/rlm-cache/my-corpus/ \
  --urls-file reading-list.txt
```

Whole-site crawls via Cloudflare's `/crawl` endpoint (async job, pagination
and status polling handled internally):

```bash
python ingestion.py ingest-crawl \
  --cache ~/.hermes/rlm-cache/docs-site/ \
  --start-url https://docs.example.com/ \
  --max-depth 3 \
  --limit 200 \
  --include "/reference/*" \
  --exclude "*.pdf"
```

URL-sourced docs share the cache with file-sourced ones and appear in the
same `corpus` dict at query time. `metadata.source_type` is `"url"` and
`metadata.source_url` holds the original URL. The cache filename is
`sha1(url).json`; content re-ingestion is hash-gated.

Fallback without Cloudflare creds: set `RLM_WEB_FETCHER=jina` or pass
`--fetcher jina` to route single-page fetches through the unauthenticated
`https://r.jina.ai/<url>` reader. (Crawl is Cloudflare-only.)

### 2. Query

```bash
python skill.py query \
  --corpus ~/.hermes/rlm-cache/svt-corpus/ \
  --query "Compare how Volovik and Liberati handle emergent Lorentz invariance."
```

Or ingest-on-demand from a source directory:

```bash
python skill.py query --corpus ~/physics/svt-corpus/ --auto-ingest \
  --query "..."
```

Output is the root LM's `FINAL(...)` answer with a resolved References block
appended (citations written as `[filename.pdf]` in the answer are matched
against corpus metadata).

## REPL environment given to the root LM

Inside the kernel, the root model has:

| Name | Type | Purpose |
|------|------|---------|
| `corpus` | `dict[filename -> doc]` | The full corpus |
| `list_papers()` | `-> list[dict]` | One-line-per-paper summary |
| `search(pattern, regex=False, case_sensitive=False, ...)` | `-> list[dict]` | Match with context |
| `get_section(filename, heading)` | `-> str` | Fetch a specific section |
| `get_paper(filename)` | `-> dict` | Full document record |
| `llm_query(prompt, max_chars=500_000)` | `-> str` | Sub-LLM call on any text |
| `fetch_url(url, timeout=60)` | `-> str` | Fetch a URL as markdown (Cloudflare/Jina) |
| `ingest_url(url, timeout=60)` | `-> str` | Fetch + add to `corpus` in-memory (non-persistent) |
| standard library | — | `re`, `json`, `collections`, etc. |

`fetch_url` and `ingest_url` are advertised in the system prompt only when
Cloudflare or Jina credentials are present. `ingest_url` is live-only — to
persist a URL across sessions, run `ingest-urls` / `ingest-crawl` via the CLI.

## Protocol

The root LM is instructed to respond in one of two ways each turn:

1. A single ```repl fenced code block — executed in the kernel, stdout/stderr
   truncated to `max_repl_output_chars` and fed back.
2. `FINAL(answer text with [filename] citations)` — terminates the loop.
   `FINAL_VAR(varname)` pulls the answer out of a kernel variable.

## Configuration

Env vars (see `config.py`):

| Var | Default | Purpose |
|-----|---------|---------|
| `RLM_ROOT_MODEL` | `claude-opus-4-7` | Root LLM |
| `RLM_SUB_MODEL` | `claude-haiku-4-5` | Sub-LLM for `llm_query()` |
| `RLM_SUB_ENDPOINT` | `anthropic` | Also `openai`, `omlx` |
| `RLM_SUB_BASE_URL` | — | For OMLX / local OpenAI-compatible |
| `RLM_MAX_ITERATIONS` | `20` | Agent loop cap |
| `RLM_REPL_OUTPUT_CHARS` | `4000` | Per-turn output truncation |
| `RLM_SUB_LLM_CHARS` | `500000` | Prompt cap fed to `llm_query()` |
| `RLM_KERNEL_TIMEOUT` | `120` | Per-exec timeout (s) |
| `RLM_TEMPERATURE` | `0.3` | Root LM temperature |
| `RLM_ENABLE_SUB_CALLS` | `true` | Turn off for ablation / cost control |
| `RLM_CACHE_DIR` | `~/.hermes/rlm-cache` | Default cache root |
| `RLM_WEB_FETCHER` | `cloudflare` | Also `jina` (unauthenticated fallback) |
| `RLM_CF_ACCOUNT_ID` / `CLOUDFLARE_ACCOUNT_ID` | — | Cloudflare account |
| `RLM_CF_API_TOKEN` / `CLOUDFLARE_API_TOKEN` | — | Token with Browser Rendering - Edit |

## Dependencies

Required (for a live session):

```
jupyter_client >= 8.0
ipykernel >= 6.0
anthropic   >= 0.40        # root + sub via Anthropic
pymupdf    >= 1.25         # lightweight PDF backend
```

Optional:

```
marker-pdf                 # heavy but excellent for math-heavy PDFs
pypdf                      # last-resort PDF fallback
openai                     # only if RLM_SUB_ENDPOINT=openai or omlx
requests   >= 2.33         # only if using web fetching (ingestion.py already
                           #   needs this indirectly; already a hermes core dep)
```

## Testing

```bash
cd skills/research/rlm-corpus
python -m pytest tests/ -o addopts=""
```

Unit tests cover ingestion, loader, prompt builders, protocol parsing, and
citation formatting. An integration suite in `tests/test_engine_integration.py`
exercises the full kernel+loop against a stub LLM; it self-skips if
`jupyter_client` / `ipykernel` aren't available.

## Security notes

- The kernel executes arbitrary Python written by the root LM. Fine for local
  single-user use; **do not** expose this as a network service without sandboxing.
- Sub-LLM calls happen *inside* the kernel — API keys in the parent environment
  are inherited. Network egress from inside the kernel is unrestricted. If the
  corpus is sensitive and you want sub-calls routed to a local model, set
  `RLM_SUB_ENDPOINT=omlx` and point `RLM_SUB_BASE_URL` at your local endpoint.
- File system access in the REPL is unrestricted.

## Stretch goals (not v1)

- Multi-corpus queries with namespace awareness
- Kernel state persisted across queries
- Interactive mid-trajectory steering
- HTML trajectory visualizer
- Scorecard mode (user-defined rubric)
- Draft-vs-literature mode
