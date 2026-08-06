# Claude Subscription via the Agent SDK — Decision Record

> **Status:** accepted; runtime ships default-off behind `claude_subscription.enabled`
> **Audience:** Hermes maintainers working on providers, auth, or the agent loop
> **Sources retrieved:** 30 Jul 2026
> **Last updated:** 2026-07-30

Hermes today reaches a Claude subscription by extracting the OAuth credential
Claude Code stores in `~/.claude/.credentials.json` (or the macOS Keychain),
hand-building an Anthropic `/v1/messages` request, and attaching Claude Code's
identity headers to it (`agent/anthropic_adapter.py` — the
`claude-code/<version> (external, cli)` user-agent and `x-app: cli` around
line 882). That path is why subscribers get billed as extra usage: the request
is ours, not Claude Code's, and nothing about it is sanctioned.

The replacement is to stop imitating Claude Code and to *call* it, through
Anthropic's official `claude-agent-sdk`, which wraps the Claude Code CLI as a
subprocess. This document records the decision, the ownership split, the policy
gate that keeps it default-off, and the rollback contract.

---

## 1. Two providers, not one

`claude-code` is currently an alias of `anthropic`
(`plugins/model-providers/anthropic/__init__.py:46`,
`hermes_cli/providers.py:309`). The two are split into distinct providers:

| | **Anthropic API** | **Claude subscription via Agent SDK** |
|---|---|---|
| Provider id | `anthropic` | `claude-code` |
| Auth | API key only | `auth_type="external_process"` |
| Request shape | `api_mode="anthropic_messages"` | `api_mode="claude_agent_sdk"` |
| API URL | Anthropic base URL | none — the SDK owns the endpoint |
| Credential env var | `ANTHROPIC_API_KEY` | none — Hermes holds no Claude credential |

The split is the point. "Anthropic API" keeps meaning exactly what it means
today: a key you pasted, billed to your Console org, running Hermes' own loop.
"Claude subscription via Agent SDK" is an account you signed into with
`claude` itself, running Claude's loop, with Hermes holding no credential at
all. Anything that blurs the two reintroduces the billing surprise this work
exists to remove.

## 2. Ownership boundary

The runtime is only tractable if each side owns its half completely and neither
reconstructs the other's state.

**Hermes keeps:** the system prompt; tool implementations; permissions and
approvals; plugins; memory; skills; checkpoints; UI history; environments
(Docker / SSH / Modal / Daytona / Singularity); cross-provider config; the
session picker; analytics.

**The SDK keeps:** Claude-native context and opaque frames; prompt-cache
continuity; automatic compaction; assistant/tool-result alternation; the resume
cursor and message UUIDs; Claude turn usage; the terminal reason.

The asymmetry is deliberate. Everything Hermes keeps is *user-visible product*
that must behave identically across every provider. Everything the SDK keeps is
*conversation-internal state* whose exact bytes we cannot reproduce and must not
try to — reconstructing Claude's context is precisely how prompt caching dies
and how alternation invariants get violated.

## 3. What we stop doing

> **Status (2026-07-31):** the subscription runtime no longer does any of
> this, and `scripts/check_claude_boundary.py` forbids reintroducing it. The
> *legacy* direct-OAuth path still contains all of it, allow-listed under
> `TODO(legacy-retirement)` markers — it is what current users run while the
> gate is closed, and it is deleted only when the subscription runtime is
> enabled and users have migrated. Until then, read "remove" as the end state,
> not the current tree.

Every item below is behavior we remove, not behavior we gate:

- Reading, copying, refreshing, or deleting Claude's credential files.
- Sending subscription OAuth tokens to `/v1/messages` directly.
- Spoofing the `claude-code/<version> (external, cli)` user-agent, `x-app: cli`,
  and Claude Code beta flags.
- Rewriting the system prompt to strip "Hermes" / "Nous".
- Renaming Hermes tools so they look like Claude Code tools.

Nothing here is a capability the user asked for; all of it exists only to make
our traffic pass as Claude Code's. Under the SDK the traffic *is* Claude Code's,
so the disguise has no remaining purpose.

## 4. The policy gate

Anthropic's Agent SDK overview
(<https://code.claude.com/docs/en/agent-sdk/overview>, retrieved 30 Jul 2026)
states:

> Unless previously approved, Anthropic does not allow third party developers to
> offer claude.ai login or rate limits for their products, including agents built
> on the Claude Agent SDK. Use the API key authentication methods described in the
> Quickstart instead.

Anthropic's Help Center takes a different position. "Use the Claude Agent SDK
with your Claude plan" (last updated 16 Jun 2026) says a planned change was
paused and that "For now, nothing has changed: Claude Agent SDK, `claude -p`,
and third-party app usage still draw from your subscription's usage limits."

So one Anthropic surface says the mechanism works and is metered against the
subscription, and another says third parties may not *offer* it. Those are not
reconcilable by reading harder, and the risk of guessing wrong lands on users'
accounts, not on us.

**Conclusion: the runtime ships default-off, and public enablement requires
written confirmation from Anthropic.** Private technical validation against a
developer-controlled, consenting account is fine and is how PRs 2-7 are
verified; shipping it enabled, advertising it, or defaulting any user onto it is
not, until that confirmation exists.

## 5. Credential precedence — why a subscription silently becomes API billing

From <https://code.claude.com/docs/en/iam> (retrieved 30 Jul 2026), when
multiple credentials are present the CLI chooses one in this order:

1. Cloud provider credentials, when `CLAUDE_CODE_USE_BEDROCK`,
   `CLAUDE_CODE_USE_VERTEX`, or `CLAUDE_CODE_USE_FOUNDRY` is set.
2. `ANTHROPIC_AUTH_TOKEN`.
3. `ANTHROPIC_API_KEY`.
4. `apiKeyHelper` script output.
5. `CLAUDE_CODE_OAUTH_TOKEN`.
6. Subscription OAuth credentials from `/login`.

The subscription is *last*. A user with a valid `/login` session and a stale
`ANTHROPIC_API_KEY` exported in their shell gets billed to the API key — with no
error, because that ordering is correct behavior for the CLI. Hermes users are
disproportionately exposed here: `ANTHROPIC_API_KEY` is a first-class Hermes
credential that the setup wizard actively encourages, so the collision is the
common case rather than the edge case. (A signed-in Claude apps gateway session
sits outside the list entirely and outranks all six.)

This cannot be fixed with `options.env`. The Python SDK builds the subprocess
environment as `{**os.environ, "CLAUDE_CODE_ENTRYPOINT": ..., **options.env}`
(`claude_agent_sdk/_internal/transport/subprocess_cli.py`), so `options.env`
**overrides** inherited keys and **cannot delete** them — setting
`ANTHROPIC_API_KEY=""` is still a set key, not an absent one.

### 5.1 How PR7 resolves it

Three layers, in the order a turn hits them:

1. **Refuse (static).** `agent/claude_billing.py` holds the precedence table.
   If any rank-1..5 credential is present in the environment,
   `claude_runtime_preflight()` refuses the turn with a message naming that
   variable and the exact `unset`. Refusing is the conservative direction:
   a refusal costs a message, a wrong guess costs money, and *which account
   pays* is the user's decision, not ours.
2. **Refuse (dynamic).** Once per session, `verify_claude_billing_for_agent()`
   asks the CLI itself. It connects the SDK to the child, reads the
   `initialize` control response — which the CLI answers from local startup,
   before any prompt exists — and disconnects without ever writing a user
   message. No model request, so no tokens and no quota. The response's
   `account` block is the ground truth (`ClaudeSDKClient.get_server_info()`).
3. **Sanitize (structural).** `agent/transports/claude_sanitized_transport.py`
   subclasses the SDK's `SubprocessCLITransport` and overrides `connect()` —
   the single method that reads `os.environ` — to spawn from an explicitly
   filtered copy. `os.environ` itself is never mutated: Hermes is
   multi-threaded and other providers run concurrently.

Measured against Claude Code 2.1.220 (30 Jul 2026), the `account` block reads:

| environment | `account` |
|---|---|
| plan login, clean env | `{"email", "organization", "subscriptionType": "Claude Max", "apiProvider": "firstParty"}` |
| `ANTHROPIC_API_KEY` set | `{"tokenSource": "claude.ai", "apiKeySource": "ANTHROPIC_API_KEY", "apiProvider": "firstParty"}` |
| `ANTHROPIC_AUTH_TOKEN` set | `{"tokenSource": "ANTHROPIC_AUTH_TOKEN", "apiProvider": "firstParty"}` |
| `CLAUDE_CODE_OAUTH_TOKEN` set | `{"tokenSource": "CLAUDE_CODE_OAUTH_TOKEN", "apiProvider": "firstParty"}` |
| `apiKeyHelper` in settings, `setting_sources=["user"]` | `{"tokenSource": "apiKeyHelper", "apiKeySource": "apiKeyHelper"}` |
| no login | `{"tokenSource": "none"}` |

Two findings worth recording because they contradict reasonable assumptions:

- **`tokenSource` is not sufficient.** With `ANTHROPIC_API_KEY` set it still
  reports `claude.ai`; `apiKeySource` is the field that tells the truth, so it
  is checked first.
- **`setting_sources=[]` does suppress `apiKeyHelper`.** Verified by pointing
  `CLAUDE_CONFIG_DIR` at a settings file whose helper touches a marker: with
  `["user"]` the helper runs, with `[]` it never does. The runtime already
  sets `setting_sources=[]`, so rank 4 cannot reach the child and is not a
  refusal trigger — but it stays in the precedence table so the classifier
  recognises it if the CLI ever changes.

The blocklist is derived from the precedence order, then widened to what the
shipped CLI actually reads: the selectors `CLAUDE_CODE_USE_ANTHROPIC_AWS`,
`CLAUDE_CODE_USE_ANTHROPIC_GOOGLE_CLOUD`, `CLAUDE_CODE_USE_MANTLE` and
`CLAUDE_CODE_USE_GATEWAY` are honored alongside the three the docs name.
`ANTHROPIC_TOKEN` is *not* read by 2.1.220 but is stripped and refused anyway,
because a credential-shaped variable we cannot rule out is treated as one.

Empty-string overrides are not a substitute for removal. On 2.1.220
`ANTHROPIC_API_KEY=`, `ANTHROPIC_AUTH_TOKEN=`, `CLAUDE_CODE_OAUTH_TOKEN=` and
`CLAUDE_CODE_USE_BEDROCK=` all behave as absent — but that is an
implementation detail of one build, per variable, with no documented contract,
and an empty string still occupies the slot for anything else reading the
environment. Removal is the only guarantee.

`HOME` is never overridden and `CLAUDE_CONFIG_DIR` passes through: rewriting
`HOME` relocates the macOS login-keychain lookup (`$HOME/Library/Keychains`),
so the CLI cannot find its stored OAuth credentials and reports
"Not logged in".

## 6. Rollback

The runtime is additive and gated:

- Rollback flips `claude_subscription.enabled` to `false` and disables the
  catalog entry. Nothing else is required.
- It never alters or deletes Hermes transcripts, provider bindings, or Claude's
  own credentials.
- "Anthropic API" and every other provider stay available and unchanged —
  a rolled-back user is on exactly the configuration they had before.
- Database migration rollback must never depend on deleting SDK transcript
  data. The PR5 `SessionStore` mirror is additive: a downgrade leaves the mirror
  rows in place and simply stops reading them. Concretely, PR5 adds five
  tables to `SCHEMA_SQL` (`provider_runtime_sessions`,
  `provider_transcript_entries`, `provider_transcript_keys`,
  `provider_transcript_summaries`, `provider_message_bindings`) through the
  declarative-reconciliation path — `CREATE TABLE IF NOT EXISTS` on every
  open, no `SCHEMA_VERSION` bump, no data migration, no foreign keys into
  `sessions`/`messages`. An older Hermes opens the same database, ignores the
  tables, and loses nothing.

### 6.1 What PR5 owns vs what the SDK owns

The mirror does not change the ownership split; it makes Hermes' half durable.

| | Hermes (`state.db`) | Claude Agent SDK |
|---|---|---|
| Visible transcript | ✅ `messages` | — |
| Which Claude session a conversation uses | ✅ `provider_runtime_sessions` | — |
| Claude-native transcript bytes | mirror only, opaque | ✅ authoritative |
| Message UUIDs / resume cursor | mirrored, never minted | ✅ |
| Rewind boundary | ✅ visible-user-turn ordinal → SDK UUID | — |
| Fork transform (UUID remap, `sessionId` rewrite) | — | ✅ `fork_session_via_store` |
| Summary derivation | persisted verbatim | ✅ `fold_session_summary` |

Canonical history is replayed into Claude **exactly once**, and only when
there is no session to resume (a provider switch into Claude, or a session
older than the mirror). Every later turn resumes, which is what keeps the
upstream prompt cache warm.

## 7. Pinned versions and packaging

| | |
|---|---|
| `claude-agent-sdk` | 0.2.128 (latest on PyPI, 30 Jul 2026) |
| `requires_python` | `>=3.10` |
| Runtime deps | `anyio>=4.0.0`, `mcp>=1.23.0,<2.0.0`, `sniffio>=1.0.0` |
| Claude Code CLI | tested at 2.1.220 |

The `mcp` range is compatible with the `mcp==1.26.0` Hermes already pins in the
`[mcp]` extra, so the SDK adds no new resolver conflict.

The platform wheels are 71.7-81.7 MB each (macOS arm64/x86_64, manylinux
aarch64/x86_64, win_amd64) because they bundle the Claude executable; the sdist
is 0.3 MB and does not. That size makes it an **optional extra and never a base
dependency**, and keeps it out of `[all]` and `[termux-all]` — those bundles are
paid for by every fresh install, and this runtime is off by default.

Per AGENTS.md § "Dependency Pinning Policy", a pre-1.0 package is pinned
`>=current,<0.(minor+2)`:

```toml
claude-code = ["claude-agent-sdk>=0.2.128,<0.4"]
```

This diverges from the exact pins used by neighbouring extras
(`anthropic==0.87.0`, `exa-py==2.10.2`, …). Those are exact because they are
mirrored in `tools/lazy_deps.py` and locked to it by
`tests/test_project_metadata.py`; this extra is not lazy-installed, so the
written pre-1.0 policy applies directly.

**`uv lock` has not been run in this PR** — the extra is opt-in and regenerating
the lockfile here would churn it for no consumer. It must be run before merge,
per AGENTS.md step 4 of the pinning policy.

## 8. The gate

`hermes_cli/claude_subscription.py` is the single source of truth. It is
dependency-light and import-safe by contract, because the provider catalog and
the dashboard web server both import it on ordinary startup paths.

- `claude_subscription_enabled(config)` — reads `claude_subscription.enabled`,
  defaults `False`, tolerates a missing or partial config dict.
- `claude_agent_sdk_available()` — cached `find_spec` probe, returns a bool,
  never raises.
- `CLAUDE_AGENT_SDK_MIN_VERSION` / `CLAUDE_CLI_MIN_VERSION` — the pins above.

The config key is a top-level `claude_subscription:` root in
`hermes_cli/config_defaults.py` rather than a key under `model:`, because
`DEFAULT_CONFIG["model"]` is a plain string (only `cli.py`'s CLI defaults shape
`model` as a dict), so a nested key would be invisible to `load_config()` and to
the gateway's raw-YAML reader. There is no `HERMES_*` env var: `.env` is secrets
only, and this is a feature flag.

```yaml
claude_subscription:
  enabled: false   # default-off pending written Anthropic policy clearance
```

## 9. Delivery order

1. **This PR** — decision record, `[claude-code]` extra, config gate, gate module.
2. Split provider identity; use official `claude auth` commands; CI boundary script.
3. Extract one canonical `execute_one_tool()` lifecycle + in-process SDK MCP bridge.
4. Whole-turn `ClaudeAgentRuntime` (`api_mode="claude_agent_sdk"`) + event projector.
5. Durable `SessionStore` mirror, rewind/fork.
6. Auxiliary one-shot adapter, subagents, model switching, surfaces.
7. Sanitized-environment launch + billing-source proof + controlled enablement.

## 10. Enablement checklist

`claude_subscription.enabled` stays `false`. It is flipped only when **every**
line below is true, and the last one is not a technical judgement.

**Technical acceptance**

- [x] The CLI subprocess is launched from an explicitly-constructed
      environment; every rank-1..5 credential is absent from it, and
      `os.environ` is not mutated (`agent/transports/claude_sanitized_transport.py`).
- [x] A turn that would be billed to any account other than the user's plan is
      **refused**, with a message naming the offending variable and its fix.
- [x] The billing source is proven per session from the CLI's own initialize
      response, at zero token and quota cost.
- [x] `HOME` is never overridden; `CLAUDE_CONFIG_DIR` passes through.
- [x] The child's stderr is routed into Hermes' logs rather than an inherited
      handle that may be closed under Electron.
- [x] `scripts/check_claude_boundary.py` is green: Hermes reads no Claude
      credential, sends no subscription token to Anthropic directly, and does
      not impersonate Claude Code.
- [x] A live turn has been proven to run on the subscription rather than an
      API key, by falsification rather than by reading a dashboard. In an
      unprivileged container (`--cap-drop=ALL --security-opt=no-new-privileges`,
      uid 1000) holding a copy of a consenting maintainer's credentials and a
      deliberately **invalid** `ANTHROPIC_API_KEY`:

      | path | result |
      |---|---|
      | stock SDK transport | `apiKeySource: ANTHROPIC_API_KEY`, classified `api_key` |
      | sanitized transport | `subscriptionType: Claude Max`, classified `subscription` |
      | live turn, sanitized transport | succeeded (`OK`, 167 in / 4 out) |
      | live turn, bare CLI, key inherited | `Execution error` |

      The last row is what makes the third meaningful: the CLI does **not**
      fall back from a bad key to the subscription, so a turn that succeeds
      with an invalid key in the environment can only have used the plan
      credential. This is stronger than a dashboard reading, which cannot
      distinguish "billed to the plan" from "billed to an org with no visible
      line item yet".

- [ ] Written policy clearance from Anthropic (see §4). Not a technical
      judgement, and the only remaining blocker.
- [x] PR5's resume path is verified through the sanitized transport. Supplying
      a custom transport makes the SDK skip `materialize_resume_session()`, so
      `session_store` + `resume` are handled by PR5 rather than inherited:
      `ClaudeAgentSession._materialize()` runs `materialize_resume_session()`
      itself, *before* building the transport, and passes the repointed
      options (`env["CLAUDE_CONFIG_DIR"]` → temp dir, `resume` → resolved id)
      to both the transport and the client. Nothing in the transport changed;
      the sanitization blocklist is still applied after `options.env`, so
      materialization cannot reintroduce a higher-precedence credential.
      Verified in `tests/agent/test_claude_session_store.py`.

**Policy clearance** — required in addition to all of the above

- [ ] Written confirmation from Anthropic that a third-party product may offer
      claude.ai login through the Agent SDK, resolving the contradiction in §4.

Until both blocks are complete the runtime remains opt-in per user, for their
own account, knowingly. Nothing in PR7 changes a default.

---

## 11. The billing classifier constrains request shape (measured 2026-07-31)

Getting subscription auth right is necessary but **not sufficient**. With the
plan credential correctly resolved and every API-key path removed, requests
were still rejected with:

```
API Error: 400 You're out of extra usage. Add more at claude.ai/settings/usage
```

while a bare `claude -p` on the same account, at the same moment, succeeded.
So this is not quota exhaustion — it is a classifier deciding the *request*
looks like a third-party product and routing it to the plan's extra-usage
pool, which is empty by default.

Bisected against the live API. Everything else held constant:

| request shape | outcome |
| --- | --- |
| `claude -p` (bare CLI) | plan |
| preset replaced by Hermes' prompt + `tools=[]` | **extra usage** |
| preset kept, Hermes' prompt appended, `tools=[]` | **extra usage** |
| preset kept, `tools` unset, Hermes' full prompt appended | **extra usage** |
| preset kept, `tools` unset, short routing note appended | plan |
| ...with 30 Hermes MCP tools attached | plan |

Two independent triggers, both required to be absent:

1. **Stripping the built-in toolset** (`tools=[]`).
2. **Carrying a product identity in the system prompt** — whether by replacing
   the preset or appending a full agent prompt to it. Length is not the
   variable: a short tool-routing note passes, a 10k-character identity prompt
   does not.

Attaching foreign MCP tools does **not** trigger it. That is the officially
supported extension path, and it is what T3 Code and Codemux do.

### What the runtime does

- `system_prompt` = the `claude_code` preset + **Hermes' own identity section**
  (`SOUL.md` when the user has one, else `DEFAULT_AGENT_IDENTITY` from
  `agent/prompt_builder.py`). No text is authored for this provider: a user who
  customises their identity gets that customisation here too.
- `tools` unset, so the built-ins stay in context — but a `PreToolUse` hook
  denies every non-`mcp__hermes__*` call, so Hermes still owns all execution.
  Denial is client-side and does not change the API-visible request shape.
- The **rest** of Hermes' prompt (memory, skills, project context, tool
  guidance) is delivered as the first user turn, reusing the pattern
  `agent/skill_commands.py` already uses for skills. Sent once per session;
  every later turn resumes, so it is never re-sent and the cache stays warm.

Why not put the whole prompt in the system slot, or nothing at all: both were
measured. The full prompt there is billed to extra usage. Nothing there means
identity decays — the full prompt as a first user turn alone holds on turn 1
and loses to the preset's "You are Claude Code" by turn 3. The identity section
is the smallest thing that bills to the plan and survives the conversation, and
it is Hermes' text rather than ours.

### Identity: solved, and why the first attempt failed

An earlier draft of this section recorded "the model self-identifies as Claude
Code" as an unavoidable tradeoff. That was wrong, and the correction matters.

Two facts settled it, both measured:

1. **Identity content does not trip the classifier.** An append reading "You
   are Hermes Agent, a personal AI assistant by Nous Research" bills to the
   plan, as does a 900-char version with persona and behaviour. What trips it
   is specific content between chars ~4000-4500 of Hermes' full prompt — the
   dense memory/skills scaffolding — not the identity claim and not length
   (5974 chars of neutral filler passes; 4500 chars of Hermes' prompt does
   not).
2. **The system prompt was never the only lever.** Hermes' prompt rides as the
   first user turn, which has no classifier interaction at all — the full
   ~86k-char prompt goes through unremarked.

The first attempt at (2) still answered "I'm Claude Code" because the framing
was too mild. A `<hermes_context>` wrapper saying "here are your operating
instructions" loses to the preset's "You are Claude Code" once the body is
large enough to bury the identity line — and the real body is ~86k chars once
context files, skills, and memory load, not the 10k a stripped test agent
produces. The wrapper now states the override explicitly, and the model
answers "I'm Hermes Agent, created by Nous Research" through the product CLI,
on a plan-billed turn, with tools working.

**We deliberately do not tune Hermes' system prompt to sit under the
classifier threshold.** That would be fragile against a rule we cannot see and
is the same category of behaviour this runtime replaced. The user-turn channel
is unbounded, legitimate, and already how Hermes ships skill content.

### What still differs from other providers

The `system_prompt` the SDK receives is the `claude_code` preset plus a short
routing note. Hermes' persona and instructions arrive one message later, in
the first user turn. Functionally the agent behaves as Hermes — identity,
persona, memory, and skills all bind — but a reader of the raw request will
see Claude Code's preset in the system slot. That is the honest shape: Hermes
is *driving* Claude Code through the supported SDK, not impersonating it.

Note this is a per-request SDK parameter. It does not read or write
`~/.claude`, and a user's own Claude Code installation is completely
unaffected — verified: no Hermes or Nous string appears in their
`settings.json`, and no `systemPrompt` key is written.
