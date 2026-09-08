# Claude OAuth DirectSDK

Experimental bundled Hermes provider: `claude-oauth-directsdk`, displayed as **Claude OAuth DirectSDK**. It uses the unmodified official Claude Code executable as a request-scoped model client. Hermes retains its normal agent loop and tool executor. Despite the name, this implementation speaks native stream-json directly and does not require the Python Agent SDK package.

## Status

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

Canonical history is replayed in order. Historical user frames use `shouldQuery:false`, each with a zero-turn acknowledgment; the final user/tool-result frame queries. There is no parked native session, synthetic continue prompt, or native approval wait. Native date/budget reminders and cache annotations remain present, so the wire prompt is not byte-identical Hermes-only context.

Text streams incrementally. A complete tool batch is published only after assistant completion, `message_stop`, final usage and native exit. Hermes then applies its own hooks, approvals, tools and persistence. Tool names map through `mcp__hermes__`; original names must be unique ASCII alphanumeric/underscore/hyphen identifiers of at most 50 characters.

`--max-turns 1` is a logical native step, not a guarantee of one HTTP request under native retries. `error_max_turns` is accepted only with a complete tool batch, usage and exit code 1. Native `num_turns` may be 2 at that boundary. Other failures remain failures.

A versioned `reasoning_details` envelope retains ordered native assistant messages and signed thinking. Unchanged projections preserve native blocks, including harmless surrounding-whitespace normalization. Transformed assistant text/tool projections replay canonical text and tool-use blocks instead of stale signed thinking; foreign provider reasoning carriers are ignored. Edited-assistant replay passed against the real service. Native autocompaction is disabled so Hermes retains compaction ownership; this does not establish parity for every history transformation or cross-model signed replay.

This provider opts into delivering actual queued steering as a canonical user message after the tool batch. It does not parse tool text to manufacture user authority. Other providers retain their existing steering behavior. Natural change-of-plan steering passed in the real loop; an exact synthetic acknowledgment instruction was still rejected even with correct user-role delivery. Transport fidelity cannot guarantee model obedience.

## Lifecycle and request support

Outside an event loop, `create` is synchronous; inside an event loop, it returns an offloaded coroutine. Streams also support `async for`. One client should belong to one independently cancellable Hermes owner.

`cancel()` signals owned POSIX process groups without closing another thread's active descriptors. `close()` prevents new calls and finalizes idle/unstarted streams; active consumers unwind after cancellation. Early stream exit requires `close()` / `aclose()`. Live interruption stopped generation and the observed native PID exited.

Supported translation includes text, base64/native images and documents, canonical tools/results, output-token limits, stop sequences, reasoning enable/disable and effort, and JSON-schema response-format projection. Unsupported native sampling fields are omitted rather than forwarding deprecated `temperature` from auxiliary callers. Reasoning effort is clamped to native-supported levels, including Hermes minimal/ultra inputs. Native thinking deltas surface as `reasoning_content`. Model/service restrictions still apply.

Unknown parameters fail explicitly. Unsupported surfaces include assistant prefill, strict function mode, forced tool choice, `parallel_tool_calls=False`, `n>1`, JSON-object-only mode, arbitrary headers/body fields, remote image downloads, non-POSIX cleanup, and cross-model signed-history parity. The read-idle timeout defaults to 180 seconds, resets on native output, and accepts Hermes' finite HTTPX read-timeout shape. Large prompts remain subject to native/OS limits.

## Model metadata and accounting

The qualified Claude Code **2.1.263** baked first-party catalog maps `sonnet` to `claude-sonnet-5` (1,000,000-token catalog window), `opus` to `claude-opus-5` (1,000,000), and `haiku` to `claude-haiku-4-5` (200,000). These are source-derived defaults, not a live account catalog: native alias overrides, remote catalog updates and entitlements can change the effective route. The provider does not query an HTTP `/models` endpoint or run an HTTP health check.

The plugin declares a conservative **200,000-token preflight bound** for these aliases and canonical IDs. This is not a claim that Sonnet 5 has a 200K catalog window: native context resolution can clamp a 1M model to 200K when the corresponding entitlement is unavailable. Only raise `model.context_length` after qualifying the actual selected native route and allowance; requalify after CLI/alias changes. Unknown model IDs remain undeclared rather than receiving invented metadata.

Token usage retains native uncached/cache-read/cache-write/output components. Completed responses also retain native `total_cost_usd` and `modelUsage`. When every reported model has `costBasis: list` and the total is finite and nonnegative, Hermes records that exact native amount as **estimated API list-price equivalent**, not an actual subscription invoice or extra-usage charge. It is never marked free/included or replaced with guessed alias prices. Missing or invalid final accounting remains unknown; interrupted requests must not be interpreted as free or zero-token service work. Hermes' iteration and runtime budgets remain host-owned; this provider does not add an account-level overage cap.

## Verification

```sh
scripts/run_tests.sh tests/providers/
```

Transport invariant tests cover signed replay and harmless normalization, transformed projections, final tool batches/usage, async use, lazy failure, invalid parameters, conflicting auth, and active/paused/unstarted stream cleanup. A separate real-native loopback qualification passed parallel tools, full long descriptions/schemas, signed ordering, host-only results, exact usage, incremental streaming and native exit checks. Its responses are synthetic protocol fixtures, not paid-model evidence.

Real bundled-provider discovery is covered separately against a temporary `HERMES_HOME`, including constructing the bundled client without spawning native or accessing auth. A fresh subscription-backed AIAgent loop completed two API calls with host `read_file` execution and SQLite persistence; separate service requests accepted edited-assistant replay. Native loopback qualification also accepted 182K of tool schemas plus a 176K system prompt through file-backed settings. Loopback responses remain fixtures, not paid-model evidence.

The subscription-backed task, CLI delegation, streaming, resume, denial, steering and interruption receipts are separate private artifacts. No auth data or trajectories are committed to this repository. Remaining qualification includes broader history transformations, cross-model signed replay, native versions/platforms, and adversarial steering reliability. Subscription invoice/overage reconciliation is not available from native list-price accounting.
