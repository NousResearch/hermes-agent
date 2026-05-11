# ADR-001: MemSearch Plugin — Critical Production Bugs & Fixes

**Status:** Accepted  
**Date:** 2025-05-11  
**Author:** Victor Pham  
**Context:** MemSearch Memory Provider Plugin for Hermes Agent (`plugins/memory/memsearch/`)

---

## Overview

During production deployment of the MemSearch memory provider, five critical bugs were discovered that caused silent failures, empty databases, or incorrect behavior. This ADR documents each bug, its root cause, the fix, and the invariant that must be maintained to prevent recurrence.

**Rule of thumb:** If you are modifying `plugins/memory/memsearch/__init__.py`, read this ADR first.

---

## Decision 1: Tilde (`~`) in `milvus_uri` Must Be Expanded to Real Home

### Context

Hermes Agent runs the gateway inside a sandboxed profile directory (`~/.hermes/profiles/<name>/home/`). When the gateway process starts, `$HOME` is rewritten to this sandbox path. The MemSearch plugin reads `milvus_uri` from `memsearch_config.json`, which users naturally write as `~/.memsearch/milvus.db`.

**Bug:** `~` resolves to `~/.hermes/profiles/<name>/home/.memsearch/milvus.db` inside the gateway, creating a new empty database. The CLI (running outside the gateway) writes to the real home. Result: `stats` shows 0 chunks, search returns nothing, and the user has two separate databases.

### Decision

Implement `_real_home()` and `_expand_paths()` helpers in `__init__.py`:

- `_real_home()` detects if `$HOME` contains `/profiles/` and falls back to `pwd.getpwuid().pw_dir`
- `_expand_paths()` replaces `~` with the real home for all path config keys (`milvus_uri`, `index_paths`)
- Called during `initialize()` before any Milvus connection is made

```python
def _real_home() -> str:
    import pwd
    for candidate in (os.environ.get("SUDO_HOME"), os.environ.get("HOME")):
        if candidate and "/profiles/" not in candidate:
            return candidate
    try:
        return pwd.getpwuid(os.getuid()).pw_dir
    except Exception:
        return os.path.expanduser("~")

def _expand_paths(cfg: dict) -> dict:
    real_home = _real_home()
    for key in ("milvus_uri", "index_paths"):
        val = cfg.get(key)
        if isinstance(val, str) and "~" in val:
            cfg[key] = val.replace("~", real_home, 1)
    return cfg
```

### Consequences

- **All `milvus_uri` values must use absolute paths in production** (`/var/home/<user>/.memsearch/milvus.db`)
- The `~` fallback works for CLI usage but is unreliable inside the gateway
- `memsearch_config.json` AND `~/.memsearch/config.toml` must both use absolute paths
- Future changes to config loading must call `_expand_paths()` before using any path

---

## Decision 2: `memsearch stats` Does NOT Support `--json-output`

### Context

The original implementation assumed the `memsearch` CLI had a `--json-output` flag for the `stats` subcommand, consistent with `search` and `expand`.

**Bug:** `memsearch stats --json-output` fails with "Error: no such option: --json-output". The subprocess raised an exception, `system_prompt_block()` caught it silently, and returned a generic fallback message. The chunk count was never displayed to the user or LLM.

### Decision

Parse the **text output** of `memsearch stats` using regex:

```python
result = subprocess.run(
    ["memsearch", "stats", "--collection", collection],
    capture_output=True, text=True, timeout=10,
)
m = re.search(r"Total indexed chunks:\s*(\d+)", (result.stdout or "") + (result.stderr or ""))
count = int(m.group(1)) if m else 0
```

### Consequences

- `stats` output format is a **text contract** — if the CLI changes its output format, the regex will break
- No JSON parsing for stats; future maintainers must not add `--json-output` to stats calls
- The regex `"Total indexed chunks:"` is intentionally specific; generic patterns risk false matches

---

## Decision 3: Switching Embedding Providers Requires Collection Reset

### Context

MemSearch creates a Milvus collection with a fixed dimension matching the embedding model. The default OpenAI `text-embedding-3-small` produces 1536-dim vectors. Google `gemini-embedding-001` produces 768-dim vectors.

**Bug:** After switching from OpenAI to Google (or any provider with different dimensions), all search and index operations fail with a Milvus dimension mismatch error. The collection was created with the old dimension and cannot accept new vectors.

### Decision

**Document (do not automate)** that switching embedding providers requires:

1. Back up the old database:
   ```bash
   cp ~/.memsearch/milvus.db ~/.memsearch/milvus.db.bak.<old-provider>-<dim>dim
   ```
2. Drop and recreate the collection:
   ```bash
   memsearch reset --yes
   ```
3. Re-index all content with the new provider:
   ```bash
   memsearch index ~/docs/ --collection hermes_memory --provider google
   ```

The plugin does **not** auto-detect dimension changes or auto-reset collections. This is intentional — data loss must be an explicit user action.

### Consequences

- `config.yaml` / `memsearch_config.json` should include a comment warning about dimension changes
- Test environments must reset the DB when switching providers in CI
- The `hermes memsearch reset` CLI command is the canonical way to handle this

---

## Decision 4: `is_available()` Must Accept Both `GOOGLE_API_KEY` and `GEMINI_API_KEY`

### Context

Google's `google-genai` library accepts both `GOOGLE_API_KEY` and `GEMINI_API_KEY` environment variables. Users who set only `GEMINI_API_KEY` (the newer, more specific name) expect the plugin to work.

**Bug:** The original `is_available()` only checked `GOOGLE_API_KEY`. Users with only `GEMINI_API_KEY` saw `available ✗` even though their key was valid and the provider would work at runtime.

### Decision

Check both environment variables for the Google provider:

```python
elif provider == "google":
    return bool(
        os.environ.get("GOOGLE_API_KEY", "")
        or os.environ.get("GEMINI_API_KEY", "")
    )
```

### Consequences

- Both keys are treated as equivalent for availability checks
- If a user sets **both** keys to different values, the first non-empty one wins (undefined which provider actually uses which — but `google-genai` has its own precedence)
- Future providers with multiple accepted env vars should follow the same pattern

---

## Decision 5: Gemini Free Tier Rate Limit (100 req/min)

### Context

Google Gemini embedding API free tier allows 100 requests per minute. Bulk indexing a large directory (e.g., 195 Markdown files) exceeds this limit.

**Bug:** `memsearch index` starts failing with rate-limit errors after the first 100 chunks. The CLI exits with a non-zero code, but the error is buried in stderr and logged only as a warning.

### Decision

**No code change in the plugin** — the `memsearch` CLI already implements:

- Exponential backoff retry on 429 errors
- Content-hash deduplication (re-indexing unchanged files is a no-op)
- Batch processing with configurable rate limits

The plugin trusts the CLI to handle rate limiting. If bulk indexing fails, the user should:

1. Wait a minute and re-run the same command — dedup prevents double-indexing
2. Upgrade to a paid tier for higher RPM
3. Index in smaller batches (`find ~/docs -name "*.md" | head -50 | xargs memsearch index ...`)

### Consequences

- Plugin logs show `logger.warning("MemSearch index failed...")` for rate-limit errors — this is expected and recoverable
- Do **not** add client-side rate limiting in the plugin — duplication with CLI logic
- For CI/testing, use `local` or `onnx` provider to avoid API limits

---

## Summary Table

| # | Bug | Fix | Invariant |
|---|-----|-----|-----------|
| 1 | `~` resolves to sandboxed HOME | `_real_home()` + `_expand_paths()` | Always use absolute paths for `milvus_uri` |
| 2 | `stats --json-output` unsupported | Parse text with regex | Never pass `--json-output` to `stats` |
| 3 | Dimension mismatch on provider switch | Document manual reset | Switch provider → backup, reset, re-index |
| 4 | `GEMINI_API_KEY` rejected | Check both env vars | Both `GOOGLE_` and `GEMINI_` keys are valid |
| 5 | Gemini 100 req/min rate limit | Trust CLI retry/backoff | Re-run command; dedup handles duplicates |

---

## Decision 6: Dual-Write Config (JSON + TOML) Retained — Unification Deferred

### Context

The plugin currently writes config to two places:
1. **`$HERMES_HOME/memsearch_config.json`** — Hermes plugin config (read at init)
2. **`~/.memsearch/config.toml`** — memsearch CLI native config (read by CLI commands)

A proposed refactor (Task 8) would make JSON the single source of truth and stop writing TOML.

### Decision

**Defer config unification.** Keep the existing dual-write pattern in `save_config()`:

```python
def save_config(self, values, hermes_home):
    # Write JSON (Hermes plugin config)
    (Path(hermes_home) / "memsearch_config.json").write_text(...)
    # Write TOML (memsearch CLI config)
    for key, val in values.items():
        subprocess.run(["memsearch", "config", "set", ...])
```

**Reasons for deferral:**

1. **No performance gain** — Config I/O is <0.1% of plugin runtime. The bottleneck is embedding API calls and Milvus search, not config writes.
2. **Breaking change risk** — Users who run `memsearch config set` CLI commands directly would find their changes ignored by the plugin. Existing workflows break silently.
3. **Migration burden** — Would require a migration script to merge existing TOML configs into JSON, plus documentation updates.
4. **Works correctly today** — Dual-write has no known bugs. The CLI and plugin stay in sync.

### Consequences

- `save_config()` remains ~30 LOC with dual-write logic
- If memsearch CLI ever drops TOML config, this decision should be revisited
- Future cleanup should include a deprecation period (log warnings) before removing TOML sync

---

## Summary Table

| # | Bug | Fix | Invariant |
|---|-----|-----|-----------|
| 1 | `~` resolves to sandboxed HOME | `_real_home()` + `_expand_paths()` | Always use absolute paths for `milvus_uri` |
| 2 | `stats --json-output` unsupported | Parse text with regex | Never pass `--json-output` to `stats` |
| 3 | Dimension mismatch on provider switch | Document manual reset | Switch provider → backup, reset, re-index |
| 4 | `GEMINI_API_KEY` rejected | Check both env vars | Both `GOOGLE_` and `GEMINI_` keys are valid |
| 5 | Gemini 100 req/min rate limit | Trust CLI retry/backoff | Re-run command; dedup handles duplicates |
| 6 | Dual-write config complexity | **Deferred** — no fix needed | Keep JSON+TOML sync until CLI drops TOML |

---

## Related

- Plugin source: `plugins/memory/memsearch/__init__.py`
- Installation guide: `plugins/memory/memsearch/README.md`
- Plan: `~/.hermes/profiles/dtpham/plans/memsearch-plugin-export.md`
- memsearch CLI docs: https://github.com/memsearch/memsearch
