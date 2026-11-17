# Multiplex hardening for fork features — design

**Date:** 2026-11-16
**Status:** Approved 2026-11-16 (scope extended after approval: 4d + acceptance section)
**Part of:** the product-education teammate (a second Slack bot on its own
Hermes profile). This doc covers only step 1 of 3. Step 2 is a draft-only
Zendesk tool; step 3 is the teammate's profile, persona, skills and jobs. Both
get their own design docs.

## Why

We want the education teammate to run in the **same gateway process** as the
current bot. Upstream calls this `gateway.multiplex_profiles: true`. The
current bot stays the `default` profile. The teammate is a secondary profile
(working name `educator`) with its own Slack app, `.env`, `config.yaml`,
`SOUL.md`, skills, memory and cron. We chose one gateway over two processes so
that later teammates are cheap to add: one service, one restart, one deploy.

Upstream's multiplex isolation is sound. `_profile_runtime_scope`
([gateway/run.py](../../../gateway/run.py) ~2487) installs two context-local
scopes that propagate into the agent worker thread through `copy_context()`:

- a home override, so `get_hermes_home()` returns the profile's home;
- a secret scope, so `agent.secret_scope.get_secret()` returns that profile's
  `.env` values.

What the scope does **not** change is `os.environ`. That still holds the
default profile's `.env`, plus first-writer-wins YAML→env bridge writes from
whichever profile loaded first. Fork code that reads `os.getenv`,
`self.adapters[...]`, or a cache keyed without the profile path silently gets
the **default** profile's values. No test today combines multiplex with any
fork feature.

## Goals

- Under multiplex, every fork feature acts as the profile that owns the turn
  or cron job: its credentials, its Slack bot identity, its RBAC roles, its
  Slack behaviour flags.
- A secondary-profile miss **fails closed**. A missing value reads as unset or
  denied, never as the default profile's value. This matches upstream's
  `get_secret` / `_platform_gate_env` contract (AGENTS.md "Profiles", rule 7).
- Single-profile deployments are **byte-for-byte unchanged**. That is the VM
  today, until multiplex is switched on.
- One test file proves the isolation end to end for each fork feature.

## Non-goals

- Creating the `educator` profile, its persona, or its tools. That is step 3.
- Fixing upstream-owned platforms other than Slack.
- Per-profile MCP server processes. The teammate is not planned to use MCP.
  If it later needs MCP, that needs per-profile MCP processes first. Fix 4c keeps the default
  profile's MCP servers away from it in the meantime.

## Anti-goals

These would make the work a failure even if every listed fix lands:

- The teammate acts in Jira, Notion, Drive or Slack with the **current bot's**
  credentials or bot identity.
- A setting in one profile's `config.yaml` changes the other bot's behaviour.
- A regression for the single-profile VM path: RBAC, cron ceiling, quiet
  channels, or the bot-authored-threads opt-in behave differently with
  multiplex off.

## Approaches considered

1. **Two gateway processes, one per profile.** Each process has its own
   `os.environ`, so no code changes are needed. Rejected by the user: one
   gateway is the desired end state, and later teammates multiply the
   processes.
2. **Harden fork code for multiplex, then run one gateway (chosen).** About
   3–4 days of mechanical fixes plus tests. It makes the fork honour upstream's
   isolation contract, so every future profile inherits correct scoping.
3. **Write the default's leaked values into each profile so the leak is
   harmless.** Rejected. It hides the bug rather than fixing it. Credentials
   would still be shared, which breaks least privilege per bot.

## The fixes

The 2026-11-16 audit found each of these. File:line references are as of
`37be6cfae2`.

### 1. RBAC config cache keyed without a path

- **Bug.** [gateway/tool_access.py](../../../gateway/tool_access.py)
  `_load_config_cached` (~459) uses one global slot keyed on
  `(st_mtime_ns, st_size)` of `get_hermes_home()/config.yaml`.
  `hermes profile create --clone` copies with `shutil.copy2`, which preserves
  mtime, so the clone's identical file makes one profile use the other's
  `user_roles` / `roles` / `channel_roles`. All three enforcement points and
  the cron ceiling go through this cache.
- **Fix.** Key the cache by `(resolved path, mtime_ns, size)`. A per-path dict
  is fine.

### 2. Slack behaviour flags that live only in `os.environ`

- **Bug.** `_apply_yaml_config` in
  [plugins/platforms/slack/adapter.py](../../../plugins/platforms/slack/adapter.py)
  (~9950–10040) bridges these YAML keys to env under a `not os.getenv` guard,
  so the first profile to load wins and the value leaks both ways:
  `wake_in_bot_authored_threads`, `strict_mention`, `thread_require_mention`,
  `ignore_other_user_mentions`, `allow_bots`, `ignored_channels`,
  `disable_dms`, `require_mention_channels`, and the reaction keys.
- **Bug.** The keys that *are* bridged into the adapter's `extra` in
  [gateway/config.py](../../../gateway/config.py) (~1717:
  `quiet_channels`, `free_response_channels`, `allow_bots_channels`,
  `require_mention`) fall back to `os.getenv` when the profile's `extra` lacks
  them, which again returns the default profile's value.
- **Bug.** The cron wake-zone guard (adapter ~3090) reads
  `wake_in_bot_authored_threads` through this same env path.
- **Fix.** Bridge every Slack key the adapter reads into `extra` in
  `gateway/config.py`.
- **Fix.** Route each adapter read through one helper with this order:
  1. the adapter's own `extra`;
  2. the scoped env via `_platform_gate_env` semantics;
  3. the default value.

  With multiplex off, step 2 still reads `os.environ`, preserving today's
  behaviour.
- **Fix.** The adapter's interactive-auth read of `SLACK_ALLOW_ALL_USERS`
  (~7573) uses the same scoped read.

### 3. Gateway code that picks the default Slack adapter

- **Bug.** The relevance gate passes `self.adapters.get(event.source.platform)`
  (run.py ~19532), so thread context is fetched as the **default** bot.
- **Bug.** `_apply_channel_model_override` (run.py ~28186) reads
  `self.adapters.get(Platform.SLACK)` `extra` first, so the default's
  `channel_models` win for any shared channel id.
- **Fix.** Both use `self._adapter_for_source(source)`
  ([gateway/authz_mixin.py](../../../gateway/authz_mixin.py) ~149). Grep for
  other fork-added `self.adapters.get(Platform.SLACK)` /
  `self.adapters[...]` sites on the inbound path and fix any found the same
  way.

### 4. Fork tools reading credentials from `os.environ`

- **Bug.** These tools read credentials with `os.getenv`:
  - [tools/jira_api_tool.py](../../../tools/jira_api_tool.py): `JIRA_*`, both
    the read and write tools;
  - [tools/notion_api_tool.py](../../../tools/notion_api_tool.py): its `ntn`
    subprocess inherits `dict(os.environ)`;
  - [tools/webflow_asset_tool.py](../../../tools/webflow_asset_tool.py):
    `WEBFLOW_*`, including the `WEBFLOW_SITE_TOKENS` indirection, whose named
    vars must also resolve through the scope;
  - [tools/slack_react_tool.py](../../../tools/slack_react_tool.py)
    `_resolve_slack_token` fallback (~76–86).
- **Bug.** The google_drive_sa plugin
  ([plugins/google_drive_sa/client.py](../../../plugins/google_drive_sa/client.py)
  ~95) caches credentials process-wide from env.
- **Fix.** Read every credential through `get_secret()`.
- **Fix.** Build the `ntn` subprocess env from the scope. Take `os.environ`
  minus profile secrets, then overlay the scoped values.
  `tools/env_passthrough.py` may already do this; reuse it if so.
- **Fix.** Key the Drive credential cache by profile home.
- **Fix.** The "self-heal `~/.hermes/.env`" step in these tools must not count
  as a credential source under multiplex. It loads into `os.environ`, which is
  the default's environment.
- **Fix.** The `slack_react` / `slack_post_thread` fallback must not post as
  another bot. Under multiplex, a miss returns "no token". It must never reach
  `os.getenv("SLACK_BOT_TOKEN")`.

### 4b. Upstream Slack/gateway reads on the secondary bot's path

**Scope decision (designer's call).** The anti-goal is "one profile's config
changes the other bot". That holds whoever wrote the line, so these upstream
reads are in scope too:

- **Slash-command admins.** `slash_access.policy_for_source(self.config, …)`
  (run.py ~22878, `slash_commands.py`) uses the default's `allow_admin_from`,
  which the fork keeps in sync with the RBAC `admin` role. **Fix:** resolve
  the policy from the source's profile config.
- **Ignored channels.** `_is_slack_ignored_channel(self.config, …)` (run.py
  ~1692) reads the default's config and falls back to `os.getenv`. **Fix:**
  use the source's profile config plus the scoped read.
- **Remaining adapter env reads.** About 20 more `SLACK_*` env reads remain
  in the adapter, for example `SLACK_ALLOWED_CHANNELS`,
  `SLACK_MENTION_PATTERNS` and `SLACK_HOME_CHANNEL_PROMPT`. **Fix:** route
  them through the same helper as fix 2.

Keep these upstream-file changes small and self-contained. They are good
candidates to send upstream later, which shrinks our merge surface. With these
added (plus 4d), the estimate is **about 4 days**.

### 4c. Default-profile MCP tools must not reach secondary profiles

MCP server processes run once per gateway under the **default** profile's
credentials (Webflow, Stripe, GitHub). A secondary-profile user whose role
grants `*` (admin), or any `mcp-*` toolset, would act with the current bot's
credentials.

**Fix (designer's call, revised in PR review 2026-11-17):** under multiplex,
a secondary profile's resolved toolsets exclude **every** MCP toolset, with no
exemption. The first version let a profile keep a server its own
`config.yaml` declares, but MCP servers live in one process-global registry
keyed by server name (`tools/mcp_tool.py`) and discovery is idempotent by
name, so a same-named server dispatched to the default's process and
credentials. Apply this in `_resolve_enabled_toolsets_for_source`, in cron's
toolset resolution, and in the dispatch backstop, failing closed. Per-profile
MCP processes stay a non-goal; until they exist a secondary profile has no
MCP.

### 4d. File tools must not cross profile homes

**Bug.** Both profiles run as one OS uid. `agent/file_safety.py` already
blocks `.env` files and the session and memory stores under any Hermes home.
A profile's `config.yaml`, `SOUL.md`, `skills/`, `memories/MEMORY.md`, cron
`jobs.json` and caches are still readable, and writable through `file` /
`patch`, by the other profile's agent.

**Fix:** under multiplex, `read_file` / `search_files` / `write_file` /
`patch` deny any path inside another **served** profile's home. For the
default profile, that home is `~/.hermes` minus `profiles/`. This uses the
same matcher-first pattern as `is_protected_data_path`, and the denial goes
to `data_access_audit`. `terminal` remains admin-only and is not a boundary.
That is unchanged and already documented.

### 4e. Per-profile self-modification guard

Added 2026-11-16 from the teammate design's review. A profile can opt in
with `agent_write_guard: true` in its own config.yaml. The current bot does
not opt in, so its behaviour is unchanged. For an opted-in profile, the
agent's file tools and `skill_manage` deny writes to:

- its own `config.yaml`, `SOUL.md`, `cron/`, `scripts/` and `.env*`;
- any `skills.external_dirs`;
- any path the profile's config lists under `agent_write_guard.extra_paths`.

This is keyed on the **owning profile's** home, never the process-wide
`HERMES_WRITE_SAFE_ROOT`, which would also restrict the other bot. Denials
go to `data_access_audit`.

### 4f. Cron script env is the owning profile's

A cron job's pre-run `script` subprocess must receive the owning profile's
scoped environment: its `.env` over the global environment, never the
default profile's credentials. Add an assertion to the cron-per-profile
test.

### 4g. Cron's per-run `.env` reload must not rewrite `os.environ`

Added 2026-11-16 during implementation (approved by the designer). The
"already correct" verdict on cron's secret handling was wrong.

- **Bug.** `cron/scheduler.py` `run_job` reloads the job's profile `.env`
  into the process environment on every run:
  `load_hermes_dotenv(hermes_home=_get_hermes_home())` on the agent path
  (~6128: `override=True`, and it also clears known keys the `.env` lacks)
  and on the `no_agent` path (~5677). Under multiplex every secondary-profile
  run rewrote the shared `os.environ` with that profile's credentials, so the
  default bot's unscoped reads and its subprocesses (MCP, terminal) picked up
  the educator's values. That breaks the Credentials row in the
  educator → default direction.
- **Fix.** Skip both reloads when `is_multiplex_active()`. Both paths already
  run inside `set_secret_scope(build_profile_secret_scope(home))`, installed
  by `_run_one_job_body` (the only caller of `run_job`) from the same `.env`
  and secret sources. The script subprocess env comes from that scope (4f).
  With multiplex off both reloads run exactly as before.
- **Same class, second site.** `run_agent.py` loads a `.env` into
  `os.environ` at import, from `get_hermes_home()`. The gateway imports it
  lazily, so the first import can happen inside a secondary profile's turn or
  cron run. Under multiplex it now loads the process (default) home's
  `.env`, which the gateway already loaded at startup.
- **Tests.** The cron-per-profile test asserts both directions: after an
  educator run, `os.environ` and the default's unscoped reads are unchanged;
  after a default run, nothing of the default reaches an educator script or
  turn. A multiplex-off regression asserts both reloads still run.

### 4h. Fixes from PR review (2026-11-17)

- Slack `allow_bots` for the message gate is resolved through the source's
  adapter under multiplex (the profile's `config.yaml`, not only its `.env`).
- A failed scoped config load gives slash commands a deny-all policy.
- Cron Bot Chat delivery (`_deliver_to_bot_chat`) runs the child as the
  owning profile (`-p <profile>`, `scoped_subprocess_env()`).
- `tools/checkpoint_manager.py` and `tools/process_registry.py` resolve their
  home per call (module-level capture sweep).
- `search_files` hits are checked per hit in every output mode (they already
  were, via `_search_result_read_block_error` → `get_read_block_error`; tests
  for both directions and all modes now pin it).

### Known limitation (not fixed here)

`terminal.cwd` is bridged once into the process-wide `TERMINAL_CWD` from the
default profile's config (gateway/run.py ~2618). Under multiplex, a
secondary profile's `terminal.cwd` is ignored on gateway turns. Profiles
without `terminal` work around it with cron `workdir` and absolute paths.
Fixing it is upstream-sized work and is deferred.

### Behaviour rules settled during review (designer's calls)

- **Precedence with multiplex off is unchanged.** For a key the adapter used
  to read only from env, `.env` still beats `config.yaml` exactly as today.
  The `not os.getenv` bridge guard means the env value wins. The shared
  helper is therefore mode-aware:
  - **Multiplex off:** today's order per key.
  - **Multiplex on:** adapter `extra` → the profile's `config.yaml`
    `slack:` block → scoped env → default. Once each profile owns its config,
    `config.yaml` winning is the natural rule (confirmed in PR review).

  The regression tests assert today's precedence with multiplex off,
  including a key set in both places with different values. Because turning
  multiplex on can therefore silently flip the **default** bot wherever its
  `.env` and `config.yaml` disagree on an env-only Slack key, multiplex
  startup logs a WARNING per profile naming every such key (names only,
  never values).
- **Fail closed under multiplex.** A guard that cannot evaluate denies:
  the cross-profile file guard denies the path, the MCP guard treats an
  unresolvable context as secondary, `scoped_subprocess_env` strips every
  non-global variable when the default's secrets cannot be enumerated, and a
  failed scoped config load yields a deny-all slash policy (never "no
  policy"). Inert with multiplex off.
- **An unscoped secret read under multiplex** raises `UnscopedSecretError`
  inside `get_secret`. The fork tools catch it, log a WARNING naming the tool
  and the key, and return their normal "not configured" error. `check_fn`
  probes return False. Nothing reads `os.environ` instead. Tests cover both a
  scoped miss and an unscoped read.

### 5. Tests

- Add `tests/gateway/test_multiplex_fork_features.py`. Model it on the
  existing `tests/gateway/test_multiplex_*.py` fixtures. Use two profiles with
  **different** values for every item above, plus a cloned config with the
  same mtime and size for fix 1.
- Assert, inside each profile's `_profile_runtime_scope`:
  - the RBAC grant;
  - each Slack flag;
  - the adapter chosen by the relevance gate and the channel-model lookup;
  - each tool's resolved credential;
  - Drive credential separation;
  - no fallthrough to `os.environ` on a scoped miss.
- Assert the same calls with multiplex **off** return today's values. This is
  the regression guard.
- Add the new invariants to `tests/test_fork_feature_inventory.py`.

### 6. Docs

- Add a short "Multiplex" note to CLAUDE.md. Fork code reading credentials or
  platform settings must use `get_secret` / `_platform_gate_env` /
  `_adapter_for_source`, never `os.getenv` or `self.adapters[...]`.
- Update the CLAUDE.md descriptions of these tools ("self-heal `.env`") to
  match.

## Acceptance: two fully separate Slack bots in one gateway

The user's bar: two Slack bots in one gateway, with complete separation of
processing, permissions and everything else. "Done" means all of these hold
for **both directions** (default → educator and educator → default):

| Axis | Must be true |
|---|---|
| Identity | Each bot receives only its own app's events and replies, reacts, and posts in threads as itself. |
| Permissions | RBAC roles, `channel_roles`, slash-command admins and the cron ceiling come from the owning profile's config only. |
| Slack behaviour | Every Slack flag (mentions, quiet channels, free response, allow-bots, ignored channels, wake rules, channel models, channel prompts) comes from the owning profile only. |
| Credentials | Jira, Notion, Webflow, Drive, Slack and Zendesk (step 2) credentials resolve from the owning profile's `.env` only; a miss reads as "not configured". |
| Tools | Each bot sees only its own `platform_toolsets` / role grants. The other profile's MCP servers are never offered (4c). |
| State | Sessions, memory, `SOUL.md`, skills, cron jobs, the ownership registry, the audit log and busy-queue modes are per profile. Neither agent can read the other's files through the file tools (4d). |
| Cost | `hermes costs` attributes each profile's spend to its own store. |

Proven by an **end-to-end test** that starts a multiplexed gateway with two
Slack adapters (mocked Slack clients, distinct tokens) and runs one inbound
message per bot through the real `_handle_message` path, asserting every axis
above. A second test runs one cron job per profile. After rollout, a manual
check on the VM repeats the identity, permissions and credentials rows with
the real apps.

**What one process cannot separate (accepted):**

- CPU and memory are shared.
- A crash or restart takes down both bots.
- A deploy restarts both.
- Anyone with `terminal` (super admins only) can reach both profiles' files.

If any of these becomes unacceptable, the fallback is two gateway processes
on the same profiles, with no profile changes needed.

## Already correct (verified, no change)

- Cron: per-profile scheduling, delivery through the owning profile's adapter,
  and the RBAC ceiling's registry lookup. The ceiling also depends on fix 1.
  (The secret scope around each run is installed, but the per-run `.env`
  reload leaked through `os.environ`. That is fixed as 4g.)
- The automation-ownership registry and the `data_access_audit` paths, which
  use `get_hermes_home()`.
- `_is_quiet_channel`, the relevance-gate config, and `channel_prompts`, which
  use scoped config loads or the adapter's own `extra`.
- Upstream multiplex: secondary adapter start, credential-collision refusal,
  and secondary authz.

## Rollout

1. Merge the fixes with multiplex still **off** on the VM. Behaviour is
   unchanged and the regression tests guard it.
2. Step 3 creates the `educator` profile and its Slack app.
3. Turn on `gateway.multiplex_profiles: true` in the default profile's
   `config.yaml` with `educator` on the profile allowlist, then restart the
   gateway.
4. Verify on the VM:
   - each bot answers as itself;
   - `hermes -p educator` shows its own RBAC roles;
   - an educator Jira call uses the educator's account. Check the Jira audit
     log or the API user.

## Also check: module-level home captures

`cron/notepad.py` sets `NOTEPAD_FILE = get_hermes_home()… ` at **import**
time. Under multiplex, that freezes the default profile's path for every
profile. That is upstream code, but the teammate's cron jobs will run there.
Grep cron/, agent/ and tools/ for module-level `get_hermes_home()` captures
that a secondary profile's cron or turn would hit. Make each one resolve per
call, and add the notepad to the cron-per-profile test.

## Left to the builder

- The exact helper names and where the shared scoped-read helper lives. It
  could be `_platform_gate_env` imported into the adapter, or a small Slack
  wrapper.
- Whether the Drive cache becomes a per-profile dict or is rebuilt per call.
- Any further `os.getenv` credential reads in fork-only code that the audit
  missed. Grep for `os.getenv` / `os.environ` in fork-added files.
