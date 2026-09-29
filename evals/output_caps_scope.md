# Output-cap removal: scope and local evidence

Run from a checkout (Python dependencies installed):

```sh
HERMES_HOME=$(mktemp -d) FULL_OUTPUT_CAP_SURFACES=1 \
  VERIFY_OUTPUT_CAP_REMOVAL=1 PYTHONPATH=$PWD \
  .venv/bin/python evals/output_caps_local_capture.py
```

The HTTP fixture returns `LOCAL_CAPTURE_ONLY`, never vendor inference. The same
harness runs against the unchanged baseline with `VERIFY_OUTPUT_CAP_REMOVAL`
unset, using that checkout's working directory and `PYTHONPATH`. Capture bodies,
not the fake response length, are the oracle. The fixture exercises installed
OpenAI, Anthropic and boto3 SDKs; the latter is a local Converse capture, **not an
AWS call**. The PTY transcript is saved under the disposable `HERMES_HOME`.

## Boundary

Removed: `HERMES_MAX_TOKENS`, `model.max_tokens`, dedicated named/custom provider
`max_output_tokens` fields, model metadata output overrides, batch CLI cap,
MoA preset/reference/slot caps, and auxiliary compression user caps. Dedicated
caps are no longer lifted through gateway/API/CLI, child, review or curator
runtime resolution. Stale configuration is ignored; no global request-field
scrubber is introduced.

Preserved:

- **Protocol fields and internal task arguments:** native Anthropic Messages
  requires `max_tokens`; native Anthropic on Bedrock is distinct from Converse.
  Converse `inferenceConfig.maxTokens` is optional. Provider defaults need not
  equal a model's maximum.
- **Provider implementation constraints:** existing NVIDIA, Qwen OAuth, Meta,
  Kimi, Opencode and native Gemini profile/adapter budgets are unchanged. This
  change does not establish fresh vendor acceptance or billing for those paths.
- **Task APIs:** `AIAgent(max_tokens=...)`, auxiliary `call_llm` budgets and
  `llm.oneshot` remain callable by application tasks, not user model settings.
  Desktop `requestOneShot` serves commit-message and other short generation
  tasks; the project-idea action also uses the stateless RPC. There is no output
  limit input in that UI. The optional Desktop maxTokens wrapper currently has
  no explicit maxTokens-valued caller; its task RPC contract is retained.
- **MCP security policy:** `SamplingHandler.max_tokens_cap` bounds a particular
  server's sampling requests, alongside rate/timeout/tool-round limits. Its
  Nix, migration, example and documentation settings belong to that separate
  trust boundary and remain supported.
- **Generic protocol passthrough:** `request_overrides`/`extra_body` remain
  arbitrary request-shaping mechanisms. They are not dedicated output-cap
  controls; stripping arbitrary native fields would break other protocols.
- Session `--max-tokens` selection filters, observed usage, context limits,
  provider-error recovery and tool-output size budgets are unrelated.

## Observed before / after

| Local production surface | Baseline cap | Fixed cap |
| --- | --- | --- |
| Main agent conversation | 17 | omitted |
| Child build and completed lifecycle | 17 | omitted |
| Full MoA: two advisors + aggregator | 23 / 23 / omitted | all omitted |
| Compression `call_llm` | 47 | omitted |
| CLI one-shot through PTY | 13 | omitted |
| Converse through boto3 to loopback | 4096 | omitted |
| Native Anthropic SDK | 16384 | 16384 |
| Explicit internal task budget | 43 | 43 |

The initial capture also covers real gateway/API/custom-provider resolvers and
the registered custom provider profile, completed curator and review forks, and
a review fork preserving an explicit internal budget of 43. Scheduling itself
is not exercised. Separate native Electron A/B opened Model settings against
isolated real backends and toggled the fixture MoA preset: both saved `enabled:
false`; baseline retained preset caps 29/31 while fixed removed them. Desktop
and dashboard TypeScript checks and the production Desktop build also passed.
No local fixture proves remote provider behavior or billing.
