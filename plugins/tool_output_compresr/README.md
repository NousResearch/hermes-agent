# tool_output_compresr

Per-turn **tool-output compression** for Hermes, powered by [Compresr](https://compresr.ai) (YC W26).

Large tool outputs — a 5k-line `read_file`, a verbose `grep`, a noisy
`execute_code`/test run — are the real driver of context-window bloat. This
plugin compresses them **as they arrive** (on the `transform_tool_result` hook),
*before* they ever enter the context, using Compresr's query-specific
tool-output API with the **tool's own intent** as the query.

It is a **lossy summary with a recoverable source**. The API's compressed text is
passed through **unchanged** (its own `[N tokens removed]` markers included), and
a footer points the agent at the full verbatim original:

```
…compressed tool output from the API, verbatim…

[compresr:recover] Tool output compressed 8120→2310 tokens (~72% saved). The full
verbatim original is cached at /root/.hermes/cache/compresr/tool-output/a3f9c1 —
if you need exact details that were summarized away, recover them with
read_file("…") or search_files.
```

The original is cached verbatim under `~/.hermes/cache/compresr/tool-output/<id>`
on the host (PR #6 *cache-authority*). When the active backend can prove an
agent-visible path, the footer points at that mounted or synced location instead;
if the path cannot be guaranteed, the plugin **fails open** and leaves the
original output untouched — never a pointer the agent can't read.

## How it works — one path

```
1. Below the threshold (min_tokens)? → return None, original output unchanged.
2. Above it → call the Compresr tool-output API (toc_latte_v2, coarse) with the
   tool's intent as the query.
3. Did it meaningfully shorten the output? If not → return None (unchanged).
4. Otherwise: cache the verbatim original (cache-authority) and return
   <API compressed text, unchanged> + a recovery footer.
```

It **fails open**: an API error — or an output that isn't meaningfully shorter —
returns `None` (original output unchanged), with a short cooldown so a failing
endpoint isn't hammered.

Recovery is deliberately simple: the agent `read_file`s (or `search_files`) the
cached original named in the footer. There is no line-level anchoring to trust —
the whole original is one `read_file` away.

## Data sent to Compresr

When enabled, this plugin sends large tool outputs to the configured Compresr
API endpoint before those outputs enter the conversation context. Those outputs
can include file contents, command output, logs, and other data returned by
Hermes tools. Do not enable per-turn tool-output compression unless that
third-party processing is acceptable for your deployment.

## Relationship to `context_engine/compresr`

This is the **per-turn** complement to the **compaction-time**
[`context_engine/compresr`](../context_engine/compresr) engine. They compose:
this plugin is a pre-filter that shrinks each tool output once, so compaction has
far less residual to summarize. Enable either or both.

## Activation

Enable the plugin — run `hermes setup` and choose **Compresr**, or add
`tool_output_compresr` to `plugins.enabled` — and provide a key:

```yaml
# ~/.hermes/config.yaml
compresr:
  tool_output_enabled: true   # master switch — on by default
```
```bash
# ~/.hermes/.env
COMPRESR_API_KEY=cmp_...
```

Once the plugin is loaded and a key is present, tool-output compression is **on
by default** (`tool_output_enabled` defaults to `true`). Set it to `false` to
keep the plugin loaded but leave tool-output compression off. The plugin stays
fully inert until it is listed in `plugins.enabled`, so nothing is sent to
Compresr until you opt in.

## Configuration

| env / `compresr:` key | default | meaning |
|---|---|---|
| `COMPRESR_API_KEY` | — | **required** `cmp_…` key, read from `.env` only |
| `COMPRESR_BASE_URL` / `base_url` | `https://api.compresr.ai/api` | API base |
| `COMPRESR_TOOL_OUTPUT_ENABLED` / `tool_output_enabled` | `true` | master switch |
| `COMPRESR_TOOL_OUTPUT_MODEL` / `tool_output_model` | `toc_latte_v2` | model (tool-output endpoint only accepts `toc_*` models) |
| `COMPRESR_TOOL_OUTPUT_MIN_TOKENS` / `tool_output_min_tokens` | `1500` | skip smaller outputs |
| `COMPRESR_TOOL_OUTPUT_TIMEOUT` / `tool_output_timeout` | `30` | request timeout (s) |
| `COMPRESR_TOOL_OUTPUT_MAX_CACHE_MB` / `tool_output_max_cache_mb` | `256` | best-effort Hermes cache cap; `0` disables pruning |
| `COMPRESR_TOOL_OUTPUT_TARGET_RATIO` / `tool_output_target_ratio` | `2.0` | Nx compression target; `0` lets the API decide |

`base_url` must be `https://` (an `http://` value other than localhost is
rejected and the default is used) so a stray config change can't send the API
key or tool output in cleartext.

The cache cap is a disk-usage guard, not a retention guarantee. Old cached
originals can be pruned after enough later compressed outputs are written into
the profile's shared Hermes cache, so a very old resumed session may contain a
recovery reference whose backing cache file has since been evicted. Set
`tool_output_max_cache_mb: 0` if preserving historical recovery references
across long-lived resumed sessions matters more than bounding profile cache
growth.

## Tests

`tests/test_tool_output_compresr.py` — the footer/pointer contract, hook gating
(small output, already-compressed, JSON-envelope folding) and fail-open (API
error, cache-write failure, not-shorter), cache-authority path translation
(local/docker/ssh-sync/singularity), size-based retention, and config.
