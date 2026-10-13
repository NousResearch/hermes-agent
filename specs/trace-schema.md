# Trace Schema v1 — Per-Run Structured Trace Specification

**Schema version:** 1.0.0
**Status:** ratified (initial)
**Canonical machine-readable schema:** `specs/trace-v1.schema.json` (same directory as this document)
**Audience:** implementers of the trace instrumentation (`agent/`), the trace writer, the redaction hook, and dataset-assembly tooling.

This document is the canonical specification for Hermes Agent's per-run structured trace. Every dispatched run — Kanban-backed or not, successful or failed — must emit exactly one trace record conforming to this schema. The schema is designed so that records are directly reconstructable into SFT examples (target: >=500 clean traces on a single profile) and DPO preference pairs (target: >=1,000 pairs), which requires that `messages` and `tool_calls` are joinable per `run_id` with content preserved.

---

## 1. Format and storage

- **Encoding:** newline-delimited JSON (JSONL). Exactly one record per line, serialized with `json.dumps(..., separators=(",", ":"))` style compaction is legal; pretty-printing is not, because append atomicity relies on a single line per record.
- **One record per run.** Never zero, never two. The run lifecycle is responsible for the exactly-once guarantee (try/finally plus signal handlers on every terminal path).
- **Trace root:** `~/.hermes/sessions/traces/`.
- **Rotation:** date-based file naming, `traces-YYYY-MM-DD.jsonl` (UTC date of record write, not run start). A record is appended to the file for the date on which the run *ends*. A secondary size cap of 512 MB per file applies: when exceeded mid-day, a segment suffix is appended (`traces-YYYY-MM-DD.002.jsonl`, monotonically increasing, zero-padded 3 digits). Segment files are created, never rewritten.
- **Append semantics:** writers must open with `O_APPEND` (POSIX) and end each append with `fsync` before returning. Writers must refuse to open a target that is not a regular file, and must never open with truncation flags. A torn final line on read (from `kill -9` mid-append) must be skipped with a warning, never crash the reader. Prior lines are guaranteed intact.
- **Directory layout note:** the trace root is deliberately outside any Hermes profile directory. Traces aggregate across profiles; `profile_slug` is a record field, not a path component. This keeps retention policy uniform.

## 2. Record fields

All fields below are required keys unless marked nullable. "Nullable" means the key is always present and may carry JSON `null`. "Required" means the key is always present with a non-null value. Times are integer Unix epoch seconds (not ISO strings) to match Kanban event timestamps elsewhere in the system; readers needing ISO can convert losslessly.

### Identity

| Field | Type | Nullability | Description |
|---|---|---|---|
| `schema_version` | string | required | Trace schema version that produced this record, e.g. `"1.0.0"`. Readers must reject records with an unrecognized major version. |
| `run_id` | integer | required | Dispatcher-assigned run identifier. Correlates to the `run_id` column of the Kanban `runs` table when the run is Kanban-backed, and to a unique integer otherwise. |
| `session_id` | string | required | Hermes session identifier (state.db `sessions` key). Present for every run, including non-messaging runs, because every run executes inside a session context. |
| `profile_slug` | string | required | The worker profile that executed the run, e.g. `"viiy-coder"`. This is also the dataset-assembly partition key for the >=500 clean traces per profile target. |
| `board_slug` | string or null | nullable | Kanban board slug that dispatched the run. `null` for runs not dispatched from a board (interactive CLI, cron, gateway message handling). |
| `task_id` | string or null | nullable | Kanban task identifier (e.g. `"t_c3944be6"`), or `null` when the run is not Kanban-backed. Nullable on purpose: interactive and cron runs are valid training signal and must not be inventing task ids. |
| `trace_id` | string | required | UUID4 generated at trace emission. Guards against duplicate rows after at-least-once retries: dedup key is `(run_id, trace_id)` with first-wins semantics. |

### Model and harness

| Field | Type | Nullability | Description |
|---|---|---|---|
| `model` | string | required | Model name exactly as passed to the provider (e.g. `"kimi-k3"`, `"claude-opus-5"`). |
| `provider` | string | required | Provider slug (e.g. `"ollama-cloud"`, `"anthropic"`, `"openrouter"`). |
| `harness_version` | object | required | Reproducibility fingerprint: `{prompt_builder_sha, config_sha256}`. See below. |
| `harness_version.prompt_builder_sha` | string | required | Git commit SHA (full 40 hex chars) of `agent/prompt_builder.py` at run start, resolved from the running repo's `git rev-parse HEAD:agent/prompt_builder.py` or the packaged build manifest. |
| `harness_version.config_sha256` | string | required | SHA-256 hex digest of the effective run configuration (profile config.yaml after overlay resolution, canonical JSON serialization with sorted keys). Any config change that alters prompt assembly is captured here without storing the config itself (which may contain secrets). |

### Conversation

| Field | Type | Nullability | Description |
|---|---|---|---|
| `messages` | array of message objects | required | Full conversation in order: system, user, assistant, tool messages interleaved as built by `prompt_builder.py`. Content preserved (post-redaction only for the marker cases in S4). Minimum 1 (the system prompt). |
| `messages[].role` | string enum | required | One of `system`, `user`, `assistant`, `tool`. |
| `messages[].content` | string or null | nullable | Text content. `null` on assistant messages that carried only tool calls (provider wire format). String otherwise; empty string permitted but flagged by dataset assembly. |
| `messages[].tool_call_id` | string or null | nullable | Present on `role: "tool"` messages, linking the tool result to the originating assistant tool call (`tool_calls[].id`). `null` on all other roles. |
| `messages[].name` | string or null | nullable | Tool name on `role: "tool"` messages (mirrors `tool_calls[].name`); `null` otherwise. |
| `messages[].index` | integer | required | Zero-based position of this message in the conversation. Included explicitly so truncated/ordered reconstruction needs no positional inference. |

### Tool calls

| Field | Type | Nullability | Description |
|---|---|---|---|
| `tool_calls` | array of tool-call objects | required | One entry per tool invocation executed during the run, in execution order. Empty array valid (pure-text runs). |
| `tool_calls[].id` | string | required | Provider-assigned tool call id (e.g. `"call_abc123"`) when the provider assigns one; otherwise a harness-generated UUID. Joins `messages[].tool_call_id`. |
| `tool_calls[].name` | string | required | Tool name exactly as dispatched (e.g. `"read_file"`, `"kanban_complete"`). **Always-safe: never redacted.** |
| `tool_calls[].arguments` | object (JSON) or redaction-marker object | required | Parsed argument object as passed to the tool, post-redaction. If the entire argument payload was redacted, replaced by the redaction marker (S4). |
| `tool_calls[].result` | string or number or boolean or object or array or null or redaction-marker | nullable | Raw tool return value, post-redaction, preserving JSON type. `null` when the tool produced no result (crashed before returning) — distinguish from the string `"null"`. |
| `tool_calls[].exit_code` | integer or null | nullable | Exit code for tools that have one (`terminal`, `execute_code`); `null` for tools without the notion (pure Python tools). **Always-safe: never redacted.** |
| `tool_calls[].duration_ms` | integer | required | Wall-clock duration of the tool call in milliseconds, measured by the harness around dispatch. **Always-safe: never redacted.** |
| `tool_calls[].is_terminal` | boolean | required | `true` on the tool call that ended the run (the call identified by `terminal_call`), `false` otherwise. Makes terminal-status reconstruction a filter, not a name comparison. |

### Outcome

| Field | Type | Nullability | Description |
|---|---|---|---|
| `outcome` | string enum | required | One of `completed`, `blocked`, `crashed`, `timeout`, `review_requested`. Maps the dispatcher's terminal classification. |
| `terminal_call` | string or null | nullable | Name of the kanban tool that ended the run: one of `kanban_complete`, `kanban_block`, `kanban_request_review`, `kanban_request_changes`. `null` when the run ended without a kanban terminal call (crashed, timeout, non-Kanban run). |
| `exit_code` | integer or null | nullable | Process exit code of the worker. `0` on clean completion, non-zero on crashed/timeout where known, `null` when the harness cannot observe it (e.g. out-of-process kill by the OS). |
| `error_class` | string or null | nullable | Sanitized failure classification when `outcome` is `crashed` or `timeout` (e.g. `"KeyboardInterrupt"`, `"SIGTERM"`, `"ProviderTimeout"`). Never a stack trace, never message content that could carry PII. `null` for non-failure outcomes. |
| `killswitch` | string or null | nullable | Name of the killswitch that fired, if any (e.g. `"HERMES_DISABLE_FILE_STATE_GUARD"`, billing killswitch), or the structured value `{"name": "...", "env_var": "..."}`. `null` when no killswitch engaged. The field carries the switch identity, never the underlying secret or state that caused it to fire. |
| `safety_tier_hit` | string or null | nullable | Highest approval gate engaged during the run: one of `"green"`, `"yellow"`, `"red"`, or `null` when none. Red-tier hits always surface here even when approved, because they are a review signal for dataset inclusion. |

### Accounting

| Field | Type | Nullability | Description |
|---|---|---|---|
| `token_counts` | object | required | Aggregate token accounting across ALL provider calls in the run. |
| `token_counts.input` | integer | required | Sum of input (prompt) tokens. May be 0 for runs that crashed before any provider call. |
| `token_counts.output` | integer | required | Sum of output (completion) tokens. |
| `token_counts.cache_read` | integer | required | Sum of prompt-cache read tokens (0 when provider does not report). |
| `token_counts.cache_write` | integer | required | Sum of prompt-cache write tokens (0 when provider does not report). |
| `token_counts.total` | integer | required | `input + output + cache_read + cache_write`. Redundancy is deliberate: dataset assembly must not re-derive. |
| `cost_usd` | number (float) or null | nullable | Total provider cost in USD, aggregated across all calls. `null` when the provider does not expose pricing (self-hosted, unpriced models) — distinguish from zero cost. |
| `token_counts_complete` | boolean | required | `false` if any provider call's usage was unobservable (e.g. crashed mid-call with no usage object). Dataset assembly may filter on `token_counts_complete == true` for cost analyses. |

### Timing

| Field | Type | Nullability | Description |
|---|---|---|---|
| `started_at` | integer (epoch seconds) | required | Run start, when the dispatcher handed the task to the worker process. |
| `ended_at` | integer (epoch seconds) | required | Run end, when the terminal state was assigned (before trace write). |
| `wall_clock_seconds` | integer | required | `ended_at - started_at`. Redundancy is deliberate: downstream joins on integer seconds should not need to subtract. |

## 3. Field table summary (machine-grepable)

| # | Field | Type | Nullable | Always-safe (no redaction) |
|---|---|---|---|---|
| 1 | `schema_version` | string | no | yes |
| 2 | `run_id` | integer | no | yes |
| 3 | `session_id` | string | no | yes |
| 4 | `profile_slug` | string | no | yes |
| 5 | `board_slug` | string | yes | yes |
| 6 | `task_id` | string | yes | yes |
| 7 | `trace_id` | string | no | yes |
| 8 | `model` | string | no | yes |
| 9 | `provider` | string | no | yes |
| 10 | `harness_version` | object | no | yes |
| 11 | `messages` | array | no | NO — content is redactable |
| 12 | `messages[].role` | string | no | yes |
| 13 | `messages[].content` | string | yes | NO — redactable |
| 14 | `messages[].tool_call_id` | string | yes | yes |
| 15 | `messages[].name` | string | yes | yes |
| 16 | `messages[].index` | integer | no | yes |
| 17 | `tool_calls` | array | no | structural — see element fields |
| 18 | `tool_calls[].id` | string | no | yes |
| 19 | `tool_calls[].name` | string | no | yes |
| 20 | `tool_calls[].arguments` | object | no | NO — redactable |
| 21 | `tool_calls[].result` | any | yes | NO — redactable |
| 22 | `tool_calls[].exit_code` | integer | yes | yes |
| 23 | `tool_calls[].duration_ms` | integer | no | yes |
| 24 | `tool_calls[].is_terminal` | boolean | no | yes |
| 25 | `outcome` | string enum | no | yes |
| 26 | `terminal_call` | string | yes | yes |
| 27 | `exit_code` | integer | yes | yes |
| 28 | `error_class` | string | yes | yes |
| 29 | `killswitch` | string or object | yes | yes |
| 30 | `safety_tier_hit` | string | yes | yes |
| 31 | `token_counts` | object | no | yes |
| 32 | `cost_usd` | number | yes | yes |
| 33 | `token_counts_complete` | boolean | no | yes |
| 34 | `started_at` | integer | no | yes |
| 35 | `ended_at` | integer | no | yes |
| 36 | `wall_clock_seconds` | integer | no | yes |

## 4. Redaction hook interface

Redaction runs **before persistence**, on every tool_call `arguments` and `result`, and on every `messages[].content`, at trace-assembly time. The hook interface is the contract between the instrumentation (parent task: lifecycle capture) and the redactor module (parent task: credential/PII redaction); the two implementations are decoupled and either can be swapped independently.

### 4.1 Interface

```
redact(value: Any, *, context: RedactionContext) -> Any
```

- Input: any JSON-typed value (object, array, string, number, boolean, null) plus a `RedactionContext` carrying the field path (e.g. `"tool_calls[3].arguments.command"`) and the field's always-safe flag.
- Output: a JSON-typed value of the same shape, with any detected credential/PII spans replaced by the redaction marker (4.3). Numbers, booleans, and nulls always pass through unchanged. Always-safe fields pass through unchanged without detector evaluation.
- Failure mode: a redactor exception must never crash a run. On redactor failure, the affected value is replaced by `{"redacted": true, "reason": "redactor_error", "sha256": null}` and the incident is logged. Fail-closed, not fail-open.

### 4.2 Policy: allow-list first, deny-by-default detectors

1. **Always-safe allow-list.** The 25 fields marked "always-safe" in the S3 table and the fixed key names of message/tool-call structure (`role`, `tool_call_id`, `name`, `index`, `id`, `exit_code`, `duration_ms`, `is_terminal`) bypass detectors entirely. This guarantees structural joins survive redaction.
2. **Detector list.** Everything else — string content inside `messages[].content`, `tool_calls[].arguments`, `tool_calls[].result` — is scanned by detector patterns loaded from config (default config path: `~/.hermes/trace_redaction.yaml`; packaged default shipped alongside this schema), not hardcoded in code. The initial detector set:
   - `bearer_token`: case-insensitive `bearer [A-Za-z0-9\-._~+/]+=*`
   - `api_key_prefixed`: `sk-[A-Za-z0-9]{16,}`, `AKIA[0-9A-Z]{16}`, `ghp_[A-Za-z0-9]{36}`, `github_pat_[A-Za-z0-9_]{22,}`, `xox[baprs]-[A-Za-z0-9\-]{10,}`, `sk-ant-[A-Za-z0-9\-]{20,}`
   - `basic_auth`: `Authorization: Basic [A-Za-z0-9+/=]{8,}`
   - `private_key_block`: `-----BEGIN [A-Z ]*PRIVATE KEY-----` through `-----END`
   - `password_field`: JSON keys named `password`, `passwd`, `secret`, `token`, `api_key`, `apikey`, `authorization` (case-insensitive) — the whole value is redacted regardless of content
   - `email`: RFC5322-simplified email pattern
   - `phone_e164`: `\+?[0-9][0-9 \-()]{7,}[0-9]` with >=8 digits after strip
   - `card_number`: Luhn-valid 13–19 digit sequences
   - `jwt`: `eyJ[A-Za-z0-9_\-]+\.eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+`
3. **Detector taxonomy maps to `reason`:** `bearer_token`, `api_key_prefixed`, `basic_auth`, `private_key_block`, `password_field`, `jwt` → `"credential"`; `email`, `phone_e164`, `card_number` → `"pii"`. New detectors must declare which of the two existing reasons they map to (no new reason enum without a minor version bump).

### 4.3 Redaction marker (structure-preserving)

A redacted span is replaced by an object of exactly this shape:

```json
{"redacted": true, "reason": "credential", "sha256": "<64-hex-chars>"}
```

- `sha256` is the SHA-256 hex digest of the redacted span **as it appeared in the value**, unsalted. This preserves dedup ability downstream: two traces hitting the same leaked token produce the same marker hash and can be identified as sharing a secret without ever revealing it.
- When a redacted span is the *entire* string value of a scalar field (e.g. a whole `password` field), the field's value becomes the marker object itself (a type change: string → object). This is the one deliberate type exception in the schema; dataset assembly must handle it.
- When a redacted span is *inside* a larger string (e.g. a line in a shell command), the span is replaced with a compact string encoding `«REDACTED:credential:sha256:<first-12-hex>…»` keeping surrounding text intact, with the full marker recoverable from `redaction_log` on the record root if needed.
- `reason: "redactor_error"` (with `sha256: null`) is reserved for the fail-closed path in 4.1 and must be treated as `credential` by retention rules (4.4).

### 4.4 Credential-bearing tool_calls — the hard rule

Any tool call whose `arguments` or `result` produced **one or more `reason: "credential"` markers** is a *credential-bearing tool call* for retention purposes (S5). Such records:

- may be retained locally only for the local retention horizon (90 days),
- are **excluded from dataset assembly exports** by default (require explicit per-export opt-in), and
- are **excluded from archive** unless the archive process has its own redaction gate.

This rule is keyed on the **presence of the marker**, not on the detector firing during export — detection runs once, at capture time, and the marker is the durable record.

## 5. Retention policy

| Tier | Scope | Horizon | Policy |
|---|---|---|---|
| **Local hot** | `~/.hermes/sessions/traces/traces-*.jsonl` on the originating host | 90 days | Append-only. Auto-pruned by the writer process on startup when files' date segments are older than the horizon. Pruning deletes whole segment files, never edits lines. |
| **Local warm (archive)** | `~/.hermes/sessions/traces-archive/` | up to 12 months | On day 91, eligible segments are moved (not copied) to the archive tree. Eligibility: the segment must have zero credential-bearing records (see 4.4); segments containing any credential marker are **deleted, not archived**. PII-marked records (`reason: "pii"`) are archivable — the marker already removed the PII — but the dataset exporter honors them as lower-trust for general SFT. |
| **External / dataset export** | wherever dataset assembly writes | per dataset contract | Exported records are a **copy**, not a move. Export applies its own redaction re-pass as a safety net before leaving the host. Exported datasets carry their own retention documented in the dataset contract (Data Contracts, SOUL 1.3). |

**PII handling rule:** PII is redacted at capture (S4), before the record ever touches disk outside process memory. The retention tiers above govern the *redacted record*, not the original content — original PII never exists in any trace file. If a redactor miss is discovered post-hoc, the remediation is (a) add the detector, (b) re-scan the hot store and rewrite offending records *in place* with the marker, (c) rotate any exported datasets containing the leak. Rewriting records post-hoc is the single permitted exception to append-only semantics and must be logged in the record's `remediation_log` (array, default empty/absent).

**Credential rule summary:** credential-bearing tool_calls are hot-store-only (90 days, then deleted) and excluded from archive and default dataset export (S4.4).

**Legal/user request handling:** on a user deletion request, `session_id`-scoped deletion is supported: locate all records with that `session_id` in hot and archive tiers and remove them; the writer tolerates post-hoc line-removal compactions for this specific purpose, logged in the segment's sidecar `.audit` file.

## 6. Downstream design contract — SFT and DPO reconstruction

Dataset assembly imposes these requirements, which this schema satisfies:

1. **Joinability.** Every `tool_calls[]` entry joins to its result message via `id == messages[].tool_call_id`. Every `role: "tool"` message joins back via the same key. `messages[].index` provides unambiguous ordering.
2. **Completeness.** `token_counts_complete` flags partial-accounting runs so cost analyses don't silently undercount. `outcome` enumerates all five terminal classes so failure runs are filterable, not silently missing.
3. **SFT example shape.** An SFT example is `{"messages": [...], "_meta": {run_id, profile_slug, harness_version, outcome}}` — a direct projection of the trace with no re-assembly. Eligible: `outcome == "completed"`, assistant content non-empty, no credential markers present.
4. **DPO pair shape.** A preference pair joins two traces sharing `(task_id, board_slug)` with different outcomes or quality signals. `terminal_call` identifies the self-reported terminal reason; `safety_tier_hit` and `error_class` are pair-quality signals. DPO assembly requires both messages and tool outcomes preserved — which is exactly what the schema mandates.
5. **MoE/replay friendliness.** `harness_version` permits exact prompt reconstruction for replay comparisons (prompt_builder SHA + config SHA).

## 7. Validation

A JSON Schema (Draft 2020-12) encoding this specification ships alongside this document at `specs/trace-v1.schema.json`. It is normative: where this prose and the JSON Schema disagree, the JSON Schema wins, and the disagreement must be resolved by a schema version bump, not by editing prose.

Validators (e.g. the acceptance test for the writer task) must call `jsonschema.Draft202012Validator` against that file. The schema marks redaction-marker shapes explicitly so downstream code can distinguish "redacted span" from "unredacted object" by key presence.

## 8. Changelog

### 1.0.0 — 2026-09-29

- Initial specification.
- Fields: 36 top-level and nested fields as enumerated in S3.
- Redaction: allow-list + deny-by-default detector model introduced; structure-preserving marker with unsalted SHA-256 for dedup; credential-bearing tool_call retention rule defined (hot-only, no archive, no default export).
- Retention: 90-day hot / 12-month warm-archive / exported-dataset-per-contract, with in-place remediation rewrite permitted for post-hoc redactor misses (logged).
- Storage: `~/.hermes/sessions/traces/`, date-rotated JSONL, append-only with `O_APPEND` + `fsync`, 512 MB segment cap.
