# Claude OAuth DirectSDK

Experimental bundled Hermes provider: `claude-oauth-directsdk`, displayed as **Claude OAuth DirectSDK**. It uses the unmodified official Claude Code executable as a request-scoped model client. Hermes retains its normal agent loop and tool executor. Despite the name, this implementation speaks native stream-json directly and does not require the Python Agent SDK package.

## Status

**Single-request admission is implemented and live-qualified.** Native 2.1.263 can attempt extra generations despite `--max-turns 1` and disabled HTTP retries. A request-scoped loopback relay now forwards only the first Messages request and rejects subsequent attempts locally. Hermes receives the first completed upstream response, its actual usage and native stop reason; blocked native recovery does not turn a completed response into an exception. Truncated/incomplete streams and upstream HTTP errors remain failures.

The integrated subscription-backed review completed seven Hermes calls with exactly seven upstream requests, twelve Hermes tool executions and twelve durable tool results. Context grew from 138,498 to 165,697 tokens; follow-up cache reads averaged 97.13%. Native accounting reported $0.8793728 in list-price equivalent, not a verified subscription charge. An earlier attempt began at 187K but exceeded the current native route's 200K bound after tools; that incomplete attempt is not counted as a successful review.

A real subscription-backed Hermes task built and tested a CSV auditor. Separate live qualifications exercised streaming tool rounds, restart/resume, host-side denial, authentic steering, cancellation during generation, and a real CLI subagent completion. This is a review build, not a full-parity or production-readiness claim. See the remaining limitations below.

Requires Python 3.10+, POSIX, and a separately installed official Claude Code CLI. Native **2.1.263** is qualified. Replay acknowledgments and extra-body behavior are version-sensitive interfaces, not a public arbitrary-history SDK guarantee. This review PR includes both the provider and its generic host support; no external plugin installation is needed.

## Login and select

```sh
claude auth login
hermes --provider claude-oauth-directsdk -m sonnet
```

Authentication belongs to the official CLI. The plugin never opens, copies, refreshes, or prints its credential files. No Hermes API key is required or sent by the plugin. The normal Hermes client path rejects inherited API-key, custom Anthropic endpoint, and cloud-backend overrides before spawning; the error names conflicting environment variables without printing their values. Remove those overrides from the launching environment when selecting OAuth. There is no silent HTTP/API-key fallback in this client.

Subscription entitlement and extra-usage settings still belong to the account and native service. Disable extra usage in the account if you do not want overage billing. A native list-price cost estimate is not proof of a subscription charge.

For a separately CLI-managed auth directory:

```sh
CLAUDE_CONFIG_DIR=/path/to/official-cli-config claude auth login
export CLAUDE_OAUTH_DIRECTSDK_CONFIG_DIR=/path/to/official-cli-config
```

An inherited `CLAUDE_CONFIG_DIR` also works. To select an executable outside PATH, set `CLAUDE_OAUTH_DIRECTSDK_COMMAND` to its absolute path. There is no unrestricted public CLI-flags setting; isolation and denial flags are plugin-owned. The low-level Python `Client(env=...)` injection is available for explicitly controlled local fixtures and does not apply the inherited-environment guard. It is not the normal Hermes provider path or an OAuth certification mechanism.

Persistent configuration:

```yaml
model:
  provider: claude-oauth-directsdk
  default: sonnet
```

Auxiliary/fallback routing remains owned by Hermes. Configure those routes explicitly if they must also use the subscription; this provider does not silently change other selected providers.

## Ownership and replay

Each `chat.completions.create` starts a fresh process in a private temporary directory. Native tools, skills and setting sources are disabled. MCP advertises only the current Hermes tool inventory, has inert callbacks, and is denied execution by native `dontAsk`. Full descriptions and schemas are supplied through tools plus validated generation fields in `CLAUDE_CODE_EXTRA_BODY`, applied from a private native settings file; the system prompt uses a private file too. This avoids the OS per-argument/environment-string limit. Authentication and identity fields are never replaced.

Canonical history is replayed in order. Historical user frames use `shouldQuery:false`, each with a zero-turn acknowledgment; the final user/tool-result frame queries. There is no parked native session or native approval wait, and the adapter adds no synthetic continue prompt. The local admission relay prevents native recovery from issuing another upstream request. The native token-budget reminder is disabled because Hermes owns budgets and replay reconstructs that reminder across the cache boundary. Other native annotations remain present, so the wire prompt is not byte-identical Hermes-only context.

The relay binds an ephemeral loopback port with a random per-request route. Native authorization headers pass through memory directly to the upstream; headers are not logged or persisted. The upstream request body and native identity headers are preserved, while HTTP transfer encoding is normalized. The relay captures streamed text, signed thinking, tool arguments, usage and stop reason before native recovery can replace them. Cancellation shuts down the active upstream connection and the native process; request teardown removes the listener. No external relay service or bundled vendor executable is required.

### Long-context caching qualification

A real Sonnet 5 Hermes review task reproduced poor cache reuse: its first request had 187,049 input tokens; the next tool round read only 6,989 of 190,840 input tokens from cache (3.66%), rewriting 183,849 tokens into the one-hour cache. Identical-request testing at 179K tokens had passed, but did not exercise this replay boundary.

Disabling the native reminder restored stable prefixes without adding cache markers or changing cache TTL. A completed seven-call review used ten real Hermes tool executions, ran six offline tests, and wrote its review artifacts. Context grew from 187K to 212K; follow-up requests averaged 97.99% cache reads, or 85.45% including the cold start. Native list-price accounting totalled $1.1030034, not a verified subscription charge. These are Sonnet 5 results: Fable 5.1 required usage credits on the qualification account and was not exercised with paid credits. Model entitlement and allowance consumption remain native-account dependent.

Text streams incrementally. A complete tool batch is published only after assistant completion, `message_stop`, final usage and native exit. Hermes then applies its own hooks, approvals, tools and persistence. Tool names map through `mcp__hermes__`; original names must be unique ASCII alphanumeric/underscore/hyphen identifiers of at most 50 characters.

`--max-turns 1` is a logical native step; the relay supplies the HTTP admission boundary. `error_max_turns` is accepted with a complete tool batch, usage and exit code 1. Native `num_turns` may be 2 at that boundary. A completed first response also survives a locally denied recovery attempt or native refusal rendering; Hermes receives the actual refusal, not the CLI's synthetic error text. Other failures remain failures.

A versioned `reasoning_details` envelope retains ordered native assistant messages and signed thinking. Unchanged projections preserve native blocks, including harmless surrounding-whitespace normalization. Transformed assistant text/tool projections replay canonical text and tool-use blocks instead of stale signed thinking; foreign provider reasoning carriers are ignored. Edited-assistant replay passed against the real service. Native autocompaction is disabled so Hermes retains compaction ownership; this does not establish parity for every history transformation or cross-model signed replay.

Mid-turn `/steer` uses Hermes' standard delivery: a standalone typed user row appended after the newest tool result. This provider adds no steering-specific transport. Natural change-of-plan steering passed live in the real loop; transport fidelity cannot guarantee model obedience.

## Lifecycle and request support

Outside an event loop, `create` is synchronous; inside an event loop, it returns an offloaded coroutine. Streams also support `async for`. One client should belong to one independently cancellable Hermes owner.

`cancel()` signals owned POSIX process groups without closing another thread's active descriptors. `close()` prevents new calls and finalizes idle/unstarted streams; active consumers unwind after cancellation. Early stream exit requires `close()` / `aclose()`. Live interruption stopped generation and the observed native PID exited.

Supported translation includes text, base64/native images and documents, canonical tools/results, output-token limits, stop sequences, reasoning enable/disable and effort, and JSON-schema response-format projection. Unsupported native sampling fields are omitted rather than forwarding deprecated `temperature` from auxiliary callers. Reasoning effort is clamped to native-supported levels, including Hermes minimal/ultra inputs. Native thinking deltas surface as `reasoning_content`. Model/service restrictions still apply.

Unknown parameters fail explicitly. Unsupported surfaces include assistant prefill, strict function mode, forced tool choice, `parallel_tool_calls=False`, `n>1`, JSON-object-only mode, arbitrary headers/body fields, remote image downloads, non-POSIX cleanup, and cross-model signed-history parity. The read-idle timeout defaults to 180 seconds, resets on native output, and accepts Hermes' finite HTTPX read-timeout shape. Large prompts remain subject to native/OS limits.

## Model metadata and accounting

The picker exposes these explicit native routes:

| Model | Native selection | Context |
| --- | --- | --- |
| Sonnet 5 | `claude-sonnet-5[1m]` | 1,000,000 |
| Haiku 4.5 | `claude-haiku-4-5-20251001` | 200,000 |
| Opus 5 | `claude-opus-5[1m]` | 1,000,000 |
| Opus 4.8 | `claude-opus-4-8[1m]` | 1,000,000 |
| Fable 5.1 | `claude-fable-5-1[1m]` | 1,000,000 |

Short names `sonnet`, `haiku`, `opus` and `fable` resolve to the corresponding pinned routes above. Known 1M model IDs also receive the native `[1m]` suffix automatically; Haiku does not. Unknown model IDs pass through unchanged with no invented context metadata. An explicit Hermes `model.context_length` still overrides the host's window, including a smaller compaction budget.

The local relay sets `ANTHROPIC_BASE_URL`, which makes Claude Code apply its gateway defaults. Its documented Sonnet 5 gateway default is 200K unless `[1m]` is selected; this was the cause of the earlier downgrade, not evidence of a general subscription limit. Both native argv and Hermes metadata now select the same window. See [Claude Code model configuration](https://code.claude.com/docs/en/model-config#sonnet-5-context-window).

Native initialization can enumerate the current picker without a Messages request, but its returned list did not include Opus 4.8. The plugin therefore keeps the five requested version-pinned entries rather than silently dropping the older model or promoting every unknown future model to 1M. It does not spawn a model or query an HTTP `/models` endpoint when opening Hermes' picker. Existing installations may need **Refresh models** to discard the previous cached list.

The follow-up probes used native **2.1.258**. The Sonnet 5 1M route accepted **902,783 actual input tokens** and returned the requested response, with native `contextWindow: 1000000` and exactly one upstream request. Native list-price equivalent was **$3.611238**. A subsequent short `sonnet` request verified automatic routing to `claude-sonnet-5[1m]`; a real `haiku` request resolved to Haiku 4.5 with a 200K window. Opus 5 and Opus 4.8 were then smoke-tested through the integrated relay: each resolved to its `[1m]` route, reported a 1,000,000-token native window, returned the requested reply in exactly one upstream request, and cost $0.001485 list-equivalent each. Fable 5.1 was rejected by the native client on the qualification account: `Fable 5.1 requires usage credits`. That is Anthropic's documented plan rule, not a routing defect: on Pro (and Team standard seats) Fable 5 / 5.1 bill to pay-as-you-go usage credits from the first request, while Max plans include them up to 50% of weekly limits ([Claude Fable models on your plan](https://support.claude.com/en/articles/15424964-claude-fable-models-on-your-plan)). The entry stays in the picker because every paid plan can reach it; the provider surfaces the native error verbatim and never falls back to another model. Native discovery also labels Opus 1M as drawing usage credits on Pro. Plan-aware picker badges from the native `initialize` handshake (which reports `subscriptionType` without an upstream request) are a follow-up.

### Hermes compaction qualification

A real multi-turn run with a **200K Hermes window** and `compression.threshold: 0.75` automatically compacted after a turn with 154,038 input tokens, at an approximately 168K next-request preflight estimate. The real summarizer completed, SQLite archived fifteen original rows and retained a summary, and the next ordinary request succeeded at 144,568 input tokens. Two additional model calls executed a real `read_file` and returned both the remembered project codename and the exact file value; that tool result was verified in SQLite. This final-routing run used eleven upstream requests total and **$1.221155** native list-price equivalent. Native autocompaction remained disabled: the summary and durable commit were owned by Hermes. The earlier native-200K control also passed; no compressor implementation change was needed.

Token usage retains native uncached/cache-read/cache-write/output components. Completed responses also retain native `total_cost_usd` and `modelUsage`. When every reported model has `costBasis: list` and the total is finite and nonnegative, Hermes records that exact native amount as **estimated API list-price equivalent**, not an actual subscription invoice or extra-usage charge. It is never marked free/included or replaced with guessed alias prices. Missing or invalid final accounting remains unknown; interrupted requests must not be interpreted as free or zero-token service work. Hermes' iteration and runtime budgets remain host-owned; this provider does not add an account-level overage cap.

## Verification

```sh
scripts/run_tests.sh tests/providers/
python evals/directsdk_admission.py /path/to/claude
python evals/directsdk_cache_wire.py /path/to/claude
```

Transport invariant tests cover signed replay and harmless normalization, transformed projections, final tool batches/usage, async use, lazy failure, invalid parameters, conflicting auth, and active/paused/unstarted stream cleanup. The admission regression fails on the previous implementation (two upstream requests) and passes with one request, preserving first-response usage including zero values. Its cancellation control verifies upstream socket closure. A real-native ten-case loopback qualification covers normal text, tools, output/context limits, thinking-only recovery, refusal, HTTP errors, disconnects and cancellation, with one upstream request per call. Its responses are synthetic protocol fixtures, not paid-model evidence.

Real bundled-provider discovery is covered separately against a temporary `HERMES_HOME`, including constructing the bundled client without spawning native or accessing auth. A fresh subscription-backed AIAgent loop completed two API calls with host `read_file` execution and SQLite persistence; separate service requests accepted edited-assistant replay. Native loopback qualification also accepted 182K of tool schemas plus a 176K system prompt through file-backed settings. Loopback responses remain fixtures, not paid-model evidence.

The subscription-backed task, CLI delegation, streaming, resume, denial, steering and interruption receipts are separate private artifacts. No auth data or trajectories are committed to this repository. Remaining qualification includes broader history transformations, cross-model signed replay, native versions/platforms, and adversarial steering reliability. Subscription invoice/overage reconciliation is not available from native list-price accounting.
