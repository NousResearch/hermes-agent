# Freemaxxing

An **opt-in Hermes model-provider plugin**. It exposes one authenticated local
OpenAI-compatible endpoint and selects eligible free routes behind the stable
`provider: freemaxxing`, `model: freemaxxing` identity. All implementation lives
in this directory. No core loader, transport, session, dependency, or global
configuration patch is required.

## Activation

Before launching Hermes, explicitly set `FREEMAXXING_ENABLED=1`. Leave it unset
(or `0`) to keep the plugin inactive. Inactive discovery registers metadata only;
it opens no socket and reads no upstream credential or catalog. Select the
Freemaxxing provider/model in Hermes after activation. Restart after changing
enrollment or listener settings; the listener capability is immutable.

OpenCode Free is keyless. An existing `OPENROUTER_API_KEY` adds OpenRouter Free.
No real bearer is sent to OpenCode. The local endpoint always requires its own
random process-local bearer; it binds only to `127.0.0.1` on an ephemeral port.
The existing provider resolver receives the actual endpoint and local bearer.

For additional accounts, set `FREEMAXXING_FREE_TIER_PROVIDERS` to a comma-separated
subset of `nous-portal,groq,gemini,mistral` **only after disabling paid use/overage
at those accounts**. This is an explicit operator assertion, not automatic
billing verification. A credential alone does not enroll an allowance account.

Nous uses the existing Hermes OAuth resolver (or `NOUS_API_KEY`) and admits only
models with fresh, unambiguous zero input/output pricing in the authenticated
catalog. The old DeepSeek Flash allowlist is removed. Groq uses `GROQ_API_KEY`,
Gemini uses `GEMINI_API_KEY` or `GOOGLE_API_KEY`, and Mistral uses `MISTRAL_API_KEY`.
Missing, expired, contradictory, or nonzero price evidence cannot authorize a
Nous route. Catalog evidence is bound to the exact endpoint and credential;
authentication refresh cannot reuse another credential's grant.

An optional local fallback accepts `FREEMAXXING_LOCAL_BASE_URL`, for example
`http://127.0.0.1:11434/v1`. Only numeric loopback is accepted, not remote hosts or
`localhost` DNS aliases. `FREEMAXXING_LOCAL_MODELS` can seed comma-separated local
model IDs; otherwise the local catalog is discovered. Hardware and electricity
are not free, but the local route incurs no hosted token charge.

## Spending boundary

OpenRouter routes require `:free` or `openrouter/free`, reject contradictory
pricing, and receive a zero `provider.max_price` cap at dispatch. Explicit model
IDs undergo the same admission check as automatic selection. Caller-supplied
paid extensions, upstream fallback lists, and server-executed tools are refused.
Only client-executed function tools are accepted; the plugin never executes them.

OpenCode discovery admits supported Chat Completions free SKUs, not arbitrary
`-free` model names. Responses/Messages-only models are excluded rather than sent
to the wrong endpoint. Authenticated providers never redirect credentials.

A cached catalog is not a provider-side spending lock. For Nous and allowance
providers, the configured free-only account boundary remains essential; this
plugin cannot inspect or change an external billing configuration. Promotions,
quotas, and provider availability can change. It never intentionally substitutes
a paid model. Global Hermes fallbacks and non-model tools configured outside
Freemaxxing remain outside this plugin's spending boundary.

## Recovery and latency

Catalog refresh runs in bounded, single-flight background workers. Safe known
OpenCode/OpenRouter IDs remain usable during catalog outages. Empty catalogs are
cached too. Refresh errors do not poison a healthy inference route. Connections
are pooled, scheduling is local (no extra LLM routing call), and successful
session affinity is retained in a bounded in-memory cache.

Model failures and account failures are distinct. A missing/broken model can
fall through to another model, while an account-wide 429 respects its complete
`Retry-After` rather than retrying every model under the same exhausted quota.
Authentication refresh is serialized against the actual failed credential.
Request concurrency, attempts, catalogs, response bytes, and stream events are
bounded. Optional `FREEMAXXING_REQUEST_TIMEOUT` (default 90 seconds),
`FREEMAXXING_MAX_ATTEMPTS` (12), and `FREEMAXXING_CONCURRENCY` (8) tune the recovery
budget. Transport idle timeout is 25 seconds and connection timeout is 5 seconds;
the recovery deadline is checked between I/O operations, with an in-flight read
bounded by its already-assigned timeout.

**Streaming uses atomic replay.** The router buffers and validates one complete
upstream stream before committing downstream HTTP 200. Keepalives, malformed
JSON, incomplete tool arguments, and missing terminal completion do not commit a
response. An interruption anywhere upstream can therefore fail over without
splicing generations or leaking partial tool calls. Exactly one complete winning
stream is replayed. This deliberately increases time to first visible token; it
is not a zero-latency claim or live token-by-token passthrough.

When no eligible free route remains, the endpoint returns a typed retryable 503
with `Retry-After`, never a fabricated assistant completion or `[DONE]`. The
plugin does not delete sessions, alter their transcripts, execute/replay tools,
or own durable turn recovery. It does not promise that all providers remain
available, that an exhausted turn completes, or that a process crash is recovered
without the host's existing session facilities. Multiplex-profile runtimes are
refused before upstream credential resolution.

Authenticated `GET /healthz` exposes bounded route health; public
`GET /v1/healthz` contains only a liveness marker. `GET /v1/models` is metadata-only.
Error responses contain no upstream credentials or raw provider error text.

## Free MoA with the existing Hermes engine

`examples/free-moa.yaml` supplies a disabled-by-default named preset for the
existing Hermes MoA runtime. It does not introduce another agent loop. All three
advisors and the acting aggregator use Freemaxxing, so their requests traverse
the same free-route boundary. The qualified selectors
`opencode-free::freemaxxing`, `openrouter::freemaxxing`, and
`nous-portal::freemaxxing` keep advisor provider pools distinct while discovering
models dynamically. They are provider diversity, not a guarantee of distinct
underlying model families. Exact `provider-name::model-id` pins also work.

Fan-out occurs once per user turn, not every tool iteration; advisor output and
time are bounded, and failed advisors are disclosed by the existing host policy.
The plugin does not automatically install or activate the preset, guarantee an
ensemble quality improvement, or reserve future provider quota for its aggregator.

## Verification

From a complete repository checkout:

```sh
scripts/run_tests.sh tests/plugins/model_providers/test_freemaxxing_proxy.py tests/plugins/model_providers/test_freemaxxing_registration.py -q
```

The loopback suite drives the real HTTP handler, pool, transport, and stream
validator with disposable upstream HTTP servers; no external model inference or
account mutation is performed. Registration fixtures isolate the host API; the
native-loader tests additionally exercise real Hermes provider discovery and
credential resolution in fresh subprocesses when the full checkout is present.
The native tests explicitly skip in a standalone plugin-only test workspace.

## Provider references and interlocks

Provider contracts checked on 2026-09-05; discovery does not assume they remain
unchanged indefinitely:

- OpenCode free model endpoints: https://opencode.ai/docs/zen/
- OpenRouter price limits: https://openrouter.ai/docs/guides/routing/provider-selection
- Nous model pricing: https://portal.nousresearch.com/models
- Groq free limits: https://console.groq.com/docs/rate-limits
- Gemini pricing: https://ai.google.dev/gemini-api/docs/pricing
- Mistral account billing: https://docs.mistral.ai/admin/billing-usage/usage-limits

Canonical implementation: https://github.com/NousResearch/hermes-agent/pull/85631
Adjacent, not duplicated: OpenCode model hygiene #101448 and fallback API-mode
preservation #102229/#102148. Their core files are not changed here. OpenCode
Chat Completions qualification remains plugin-local; unsupported protocols do not
become a new core router or a second implementation of those fixes.
