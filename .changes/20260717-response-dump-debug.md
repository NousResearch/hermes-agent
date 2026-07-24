# ChangeRequest: Capture model responses next to request dumps

- **Date:** 2026-07-17
- **Author:** kuehnberger
- **Branch:** `feat/response-dump-debug`
- **Target:** `NousResearch/hermes-agent` (`main`)

## Problem

Today Hermes can dump the *request* it sends to an inference provider
(`request_dump_<sid>_<ts>.json`, written by `agent_runtime_helpers.dump_api_request_debug`
when `HERMES_DUMP_REQUESTS=1`). It captures **no record of the response** — neither
on success nor on error.

Consequences:

- When a model answers in a way that is surprising, wrong, or quirky, there is no
  post-mortem artifact to inspect. The only way to see "what the model actually
  returned" is to re-run the exact turn — which is impossible for interactive
  sessions or time-sensitive failures.
- Tool-call argument parsing, `finish_reason` handling, reasoning extraction, and
  stream-vs-non-stream shaping all happen *around* the raw response. With no
  response capture we cannot debug whether a bug lives in the provider, the SDK,
  or our consumer code.
- `state.db` stores only the *parsed* transcript (roles, trimmed text, tool-call
  names, token counts). It deliberately does **not** keep the raw completion body,
  HTTP status, or provider headers, so it cannot serve as a response log.

In short: request-side debugging exists, response-side debugging does not.

## Need / desired outcome

Add an opt-in, file-based **response dump** that mirrors the existing request dump:

- Same trigger gate (`HERMES_DUMP_REQUESTS=1`), so there is a single "debug dumps on" switch.
- Same destination folder (`agent.logs_dir`), same filename shape
  (`response_dump_<sid>_<ts>.json`), same atomic write + secret-redaction path.
- Captures the raw completion body, HTTP status / headers, `finish_reason`,
  usage, and the model/provider that produced it.
- Captures **both** the success path and the error paths (non-retryable client
  error, terminal failure) — because the most valuable debugging is precisely the
  failure cases where the model/proxy returned something unexpected.

This gives a complete request→response pair on disk for every debugged call,
enabling offline inspection of model behaviour and answers.

## Proposed contract

New helper (sibling of `dump_api_request_debug`):

```python
def dump_api_response_debug(
    agent,
    *,
    response: Optional[Any] = None,      # raw SDK completion / stream final object
    status: Optional[int] = None,        # HTTP status if available
    headers: Optional[Dict[str, str]] = None,
    reason: str,                          # "success" | "non_retryable_client_error" | "terminal_failure"
    error: Optional[Exception] = None,
) -> Optional[Path]:
    ...
```

Behaviour:

1. Build `response_dump_<safe_sid>_<ts>.json` in `agent.logs_dir`.
2. Serialize a structured payload: `timestamp`, `session_id`, `reason`,
   `model`/`provider`/`base_url`, `status`, `headers` (redacted), and a
   `response` object holding the normalized completion (choices / content,
   `finish_reason`, `usage`, `id`, `created`).
3. Run the serialized payload through the same `redact_sensitive_text(..., force=True)`
   scrubber already used by the request dumper, then `atomic_json_write`.
4. `agent._vprint(...)` a breadcrumb line (mirrors request dumper UX).

Wiring (in `conversation_loop.py`):

- **Preflight** (line ~1327, inside `if env_var_enabled("HERMES_DUMP_REQUESTS")`)
  → also call response dump once a response object exists.
- **Non-retryable client error** (line ~3878) → call response dump with the
  error's captured response body/status.
- **Terminal failure** (line ~4194) → call response dump with the final error
  context.
- **Success boundary** → capture the returned completion into a response dump so
  successful turns are also recorded.

No new config flag is introduced: response dumps ride the existing
`HERMES_DUMP_REQUESTS` gate. (A separate `HERMES_DUMP_RESPONSES` toggle can be
added later if desired, but is out of scope here to keep the diff minimal and the
debug switch coherent.)

## Files touched

- `agent/agent_runtime_helpers.py` — add `dump_api_response_debug` + `__all__` entry + docstring header.
- `agent/conversation_loop.py` — 3–4 call sites.
- `.changes/20260717-response-dump-debug.md` — this CR.

## Non-goals / explicitly declined

- **Not** writing responses into `state.db`. The DB stores parsed transcript only
  by design; adding raw bodies there contradicts that contract and bloats the DB.
- **Not** changing the request-dump format.
- **Not** adding a new env toggle in this change (kept to one debug switch).
- **Not** capturing streaming token fragments — only the final/normalized
  completion object, to stay consistent with how the parsed transcript is produced.

## Verification

1. `python -m py_compile agent/agent_runtime_helpers.py agent/conversation_loop.py`
2. Run the agent with `HERMES_DUMP_REQUESTS=1` against a live provider (OmniRoute).
3. Confirm both `request_dump_*.json` and `response_dump_*.json` appear in
   `~/.hermes/sessions/` for the same session id.
4. Force a non-retryable error (e.g. bad model name) and confirm a
   `response_dump_*` with `reason=non_retryable_client_error` + captured status.
5. Confirm redaction: no cleartext API key / embedded secret survives in the
   dumped file (same scrubber as request dumps).
