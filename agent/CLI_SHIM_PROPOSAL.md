# cli-shim Provider — RFC / Proposal

Status: DRAFT. Looking for upstream guidance before promoting to a non-draft PR.

This document accompanies the `feat/cli-shim-provider` branch. It proposes a new
hermes model-provider, `cli-shim`, that satisfies inference requests by shelling
out to locally-installed CLI tools (`claude`, `codex`, `gemini`) that the user
has already authenticated with their OAuth subscription, instead of via REST
APIs with an API key.

Source under review:
- `agent/cli_shim_client.py` (~530 LOC) — the OpenAI-shaped facade
- `plugins/model-providers/cli-shim/` — provider profile + manifest
- Hooks in `run_agent.py`, `agent/auxiliary_client.py`, `hermes_cli/auth.py`

---

## 1. Motivation

- **Subscription users have no API keys.** Customers on Claude Max ($100–$200/mo),
  ChatGPT Pro ($200/mo), and Gemini Advanced already pay for capacity, but the
  vendor-issued credential is bound to the desktop/CLI app via OAuth, not to a
  programmatic `sk-...` key. Today they cannot use hermes without paying a
  second time for API access.
- **Fleet scenarios hit 429s on API plans.** A 27-agent hermes fleet (kanban
  workers, gateway sub-agents, cron jobs) overruns Anthropic / OpenAI per-minute
  rate limits within seconds. The OAuth-backed CLIs route through a different
  quota bucket (interactive Pro plan), so spilling fleet traffic there is a real
  release valve.
- **Every existing hermes provider requires an API key.** The `providers/`
  registry, the setup wizard, and `resolve_provider_client()` all assume an
  `env_vars=("ANTHROPIC_API_KEY", ...)` style credential. There is no first-class
  affordance for a provider whose auth lives in `~/.codex/auth.json` or
  `~/.claude/.credentials.json`.
- **Local-first is on-trend.** Developers running claude/codex/gemini CLIs
  already trust them; reusing those sessions is the path of least surprise and
  zero extra secret management.
- **Graceful degradation for the budget chain.** Wiring `cli-shim` at the bottom
  of `fallback_model` chains lets paid-API hermes fleets fail over to the
  user's interactive subscription rather than crashing on quota exhaustion.

## 2. Architecture

### 2.1 Dispatch table

The single function `_dispatch_for_model(model)` in `cli_shim_client.py` maps a
hermes-side model alias to a concrete CLI invocation plus a "mode" flag that
selects the run path:

    claude-sonnet-cli / claude-sonnet / sonnet-cli  -> claude --print --model sonnet  (mode=print)
    claude-opus-cli   / claude-opus   / opus-cli    -> claude --print --model opus    (mode=print)
    codex-gpt5-cli    / codex-gpt5.5-cli / codex    -> codex exec --skip-git-repo-check
                                                              --model gpt-5.5         (mode=codex_exec)
    gemini-cli        / gemini-acp    / gemini      -> gemini --acp                   (mode=acp)
    <anything else>                                 -> claude --print --model <model> (mode=print, fallback)

### 2.2 Subprocess invocation per CLI

- **`mode=print`** — `_run_print()`. Pure single-shot `subprocess.Popen(..., stdin=PIPE,
  stdout=PIPE, stderr=PIPE)`, prompt written on stdin, stdout consumed
  whole on completion, timeout enforced via `proc.communicate(timeout=...)`.
- **`mode=codex_exec`** — `_run_codex_exec()`. Same as above, but appends
  `--output-last-message <tmpfile>`; the codex `exec` subcommand writes only
  the final assistant turn there, sidestepping the noisy `user\n...\ncodex\n...`
  scaffold that pollutes stdout. If the tempfile is empty (no assistant turn),
  falls back to `_scrub_codex_stdout()` which regex-strips the scaffold.
- **`mode=acp`** — Delegates to the existing `CopilotACPClient._run_prompt()`
  with `command="gemini" args=["--acp"]`. Gemini's ACP mode supports streamed
  tool-use, so the full ACP loop is reused verbatim.

### 2.3 Concurrency caps

A 27-agent fleet that all decide to invoke `cli-shim` on the same turn would
spawn 27 simultaneous `claude` subprocesses (~500 MB resident each) and OOM the
host. To prevent that, every dispatch path passes through `_ConcurrencyGate`,
which acquires two `threading.BoundedSemaphore` slots in order:

1. A **global** semaphore — `HERMES_CLI_SHIM_GLOBAL_MAX` (default `6`).
2. A **per-CLI** semaphore — `HERMES_CLI_SHIM_CLAUDE_MAX` (3),
   `HERMES_CLI_SHIM_CODEX_MAX` (4), `HERMES_CLI_SHIM_GEMINI_MAX` (3).

If either acquire times out (`HERMES_CLI_SHIM_QUEUE_TIMEOUT`, default 120 s)
the gate raises `RuntimeError`, the call fails fast, and the agent's fallback
chain can try the next provider.

### 2.4 Streaming-chunk synthesis

Hermes' agent loop calls `client.chat.completions.create(..., stream=True)` and
iterates the result with `for chunk in stream:`. None of the underlying CLIs
emit per-token streams over their CLI surface, so `_CliShimChatCompletions.create()`:

1. Pops `stream` out of `kwargs`,
2. Runs the full single-shot subprocess and builds an OpenAI-shaped response,
3. If `stream=True`, hands the response to `_wrap_response_as_stream()` which
   yields three (or more) `SimpleNamespace` chunks:
   - chunk 1: `delta.role="assistant"`, full content as a single delta,
   - chunks 2..n: one chunk per extracted tool_call, each carrying full args,
   - chunk N+1: `delta` empty, `finish_reason="stop"|"tool_calls"`, `usage` attached.

Each chunk shape matches `openai.types.chat.ChatCompletionChunk` closely enough
for hermes' stream consumer.

### 2.5 Output extraction

After the subprocess returns, response text is fed to
`_extract_tool_calls_from_text()` (re-exported from `agent.copilot_acp_client`).
That helper parses fenced tool_call JSON blocks the CLI emitted and returns
`(tool_calls, cleaned_text)`. The cleaned text becomes
`assistant_message.content`; `finish_reason` flips to `"tool_calls"` when any
were found.

### 2.6 Diagram

```
  hermes run_agent.AIAgent
        │  (chat.completions.create stream=True)
        ▼
  CliShimClient ──── _dispatch_for_model(model) ──┐
        │                                         │
        │                            ┌────────────┼────────────┐
        │                            │            │            │
        │                       claude-*       codex-*       gemini-*
        │                            │            │            │
        ▼                            ▼            ▼            ▼
  _ConcurrencyGate(global=6)   _run_print  _run_codex_exec  CopilotACPClient
        +per-CLI(3/4/3)            │            │             (ACP loop)
        │                          │            │             │
        ▼                          ▼            ▼             ▼
        ──────── subprocess.Popen(claude|codex|gemini --acp) ──
                                   │
                                   ▼
                       stdout / --output-last-message
                                   │
                                   ▼
                  _extract_tool_calls_from_text()
                                   │
                                   ▼
        ──── OpenAI-shaped response (SimpleNamespace) ─────
                                   │
                                   ▼  (when stream=True)
                _wrap_response_as_stream() → chunk1 / tcN / final
```

## 3. Configuration

End-user wiring in `~/.hermes/config.yaml`:

```yaml
provider: cli-shim
model: codex-gpt5.5-cli       # or claude-sonnet-cli / claude-opus-cli / gemini-cli
```

No `~/.hermes/.env` entries are required — the underlying CLIs read their own
OAuth caches (`~/.claude/`, `~/.codex/auth.json`, `~/.gemini/`).

For fallback chaining, put it last in the budget chain:

```yaml
provider: anthropic
model: claude-sonnet-4
fallback_model:
  - provider: openai
    model: gpt-5
  - provider: cli-shim
    model: codex-gpt5.5-cli
```

Concurrency tuning (optional):

```bash
export HERMES_CLI_SHIM_GLOBAL_MAX=6        # total simultaneous subprocesses
export HERMES_CLI_SHIM_CLAUDE_MAX=3
export HERMES_CLI_SHIM_CODEX_MAX=4
export HERMES_CLI_SHIM_GEMINI_MAX=3
export HERMES_CLI_SHIM_QUEUE_TIMEOUT=120   # seconds before bail-out
```

## 4. Interface (Provider protocol surface)

`CliShimClient` mimics the subset of the `openai.OpenAI` client that
`AIAgent.run_conversation()` actually touches:

- `client.chat.completions.create(model=..., messages=..., tools=..., tool_choice=..., stream=..., timeout=...)`
  Returns either:
    - `SimpleNamespace(choices=[SimpleNamespace(message=..., finish_reason=...)], usage=..., model=...)` (when `stream=False`)
    - a generator yielding chunk `SimpleNamespace`s (when `stream=True`)
- `client.close()` — terminates any active subprocess and flips `is_closed`.
- Attributes: `api_key`, `base_url`, `is_closed`.

`message` shape on the non-stream return:
```
SimpleNamespace(
    content=<cleaned_text>,
    tool_calls=[...],          # OpenAI shape: id/type=function/function.name+arguments
    reasoning=<text or None>,
    reasoning_content=<text or None>,
    reasoning_details=None,
)
```
`usage` has `prompt_tokens=0, completion_tokens=0, total_tokens=0,
prompt_tokens_details.cached_tokens=0` (the CLIs don't expose token accounting).

Provider profile (`plugins/model-providers/cli-shim/__init__.py`):
`ProviderProfile(name="cli-shim", aliases=("local-cli-shim","cli"),
api_mode="chat_completions", env_vars=(), base_url="cli://shim",
auth_type="external_process")`. `fetch_models()` returns the four canonical
aliases.

Routing hooks: `agent/auxiliary_client.py::resolve_provider_client()` and
`run_agent.py::AIAgent._build_openai_client()` both detect `provider=="cli-shim"`
or `base_url=="cli://shim"` and instantiate `CliShimClient`.

## 5. Failure modes

| Condition | Surface |
|---|---|
| CLI binary not on `PATH` | `_run_print` / `_run_codex_exec` catch `FileNotFoundError` and raise `RuntimeError("Could not start CLI '<name>'. Install it or check PATH.")` |
| Subprocess timeout | `subprocess.TimeoutExpired` → `proc.kill()` → re-raised as `TimeoutError("CLI '<name>' timed out after <s>s")` |
| Non-zero exit | `RuntimeError("CLI '<name>' exited <rc>: <stderr tail (last 800 chars)>")` |
| OAuth expired | The CLI itself prints its login URL on stderr; we surface it via the same non-zero-exit `RuntimeError` so the agent log captures the URL |
| Upstream rate-limit (CLI prints 429-style banner and exits non-zero) | Re-raised as `RuntimeError`; hermes' fallback chain sees a normal exception and tries the next provider |
| Global semaphore exhausted | `RuntimeError("cli-shim global concurrency cap reached (...) queue timeout 120s exceeded")` after `HERMES_CLI_SHIM_QUEUE_TIMEOUT` |
| Per-CLI semaphore exhausted | Same pattern, "per-CLI cap reached for '<cli>'" |
| Codex writes no tempfile (no assistant turn produced) | `_scrub_codex_stdout()` fallback extracts whatever it can from the noisy stdout |
| Client `close()` while subprocess in flight | `terminate()` then `wait(2s)` then `kill()`; guarded by `_active_process_lock` |

## 6. Operational notes

- **Per-CLI concurrency caps** track empirical RSS: claude ~500 MB, codex ~400 MB,
  gemini ~350 MB. Defaults (3/4/3 with global 6) leave headroom on a 16 GB box.
- **`codex --output-last-message`** is essential. Without it, codex `exec`
  stdout is a multi-line scaffold (`user\n<prompt>\ncodex\n<resp>\ntokens used\n<n>`),
  and `_extract_tool_calls_from_text` mis-fires on the scaffold text. Writing
  the final message to a tempfile and reading it back yields clean output.
- **gpt-5.5 pinning.** Commit `63f2814fe` pins `codex exec` to `--model gpt-5.5`.
  The ChatGPT $200 Pro plan grants unrestricted gpt-5.5 access; without the
  explicit `--model` flag, codex inherits whatever the OAuth account's interactive
  default happens to be (often a smaller model), so hermes traffic silently
  downgrades. Pinning makes the inference predictable.
- **Gemini ACP mode.** Unlike claude/codex, gemini's CLI supports the Agent
  Communication Protocol over stdio. We piggy-back on the existing
  `CopilotACPClient._run_prompt` rather than reimplementing the JSON-RPC loop —
  tool-use, streaming, and cancellation all come for free.
- **Streaming is synthesized, not native.** A single delta chunk carries the
  whole assistant turn. This is acceptable for hermes' iterate-and-then-handle
  loop but would not satisfy a real UI that expects per-token deltas.

## 7. Open questions for upstream

1. **OAuth detection — hermes-core feature?** Today each CLI's OAuth cache is
   discovered implicitly (we just exec the binary and let it find its own
   credentials). Should hermes-core have a first-class affordance to detect
   "is the user logged into claude/codex/gemini" so the setup wizard can show
   it as an auth option alongside API keys? That would also let the gateway UI
   surface "Login expired — re-run `claude login`" diagnostics rather than
   buried-in-stderr `RuntimeError`s.
2. **Streaming synthesis acceptable?** This PR synthesizes streaming chunks
   from a single-shot CLI invocation. Is that acceptable as a permanent design
   choice, or do you want a manifest flag like `supports_streaming: false` so
   callers can branch and avoid the fake-stream code path entirely?
3. **No-API-key manifest convention.** `plugins/model-providers/cli-shim/__init__.py`
   sets `env_vars=()` and `auth_type="external_process"`. The setup wizard's
   API-key prompt is suppressed for these, but the convention isn't
   documented anywhere. Worth a `requires_api_key: false` (or
   `auth: oauth_external`) manifest field that the wizard, doctor, and config
   validator can all read?
4. **Provider-level fallback semantics.** When `cli-shim` raises (semaphore
   exhausted, CLI not installed), should hermes-core auto-skip to the next
   `fallback_model` entry, or is that already the contract? The current code
   relies on the existing exception propagation path — confirmation appreciated.
5. **Per-CLI concurrency policy.** The caps are env-var-tuned today. Would
   upstream prefer a config-yaml schema (e.g. `cli_shim.concurrency.claude: 3`)
   so the values can be checkpointed alongside the rest of the profile?
6. **Token-accounting reporting.** `usage` is zeroed because the CLIs don't
   expose counts. Should we estimate via the local tokenizer for budget
   bookkeeping, or is `0` acceptable since the cost is borne by the
   subscription, not the API budget?

---

## Appendix: commits on this branch

    11df32884  feat(cli-shim): local CLI provider for OAuth-backed claude/codex/gemini fallback
    3e5d0ec90  fix(cli-shim): concurrency caps + codex --output-last-message + ChatGPT-account model fix
    04e7d1cd5  fix(cli-shim): synthesize streaming chunks when stream=True
    63f2814fe  feat(cli-shim): pin codex to gpt-5.5 (ChatGPT Pro plan unrestricted)
