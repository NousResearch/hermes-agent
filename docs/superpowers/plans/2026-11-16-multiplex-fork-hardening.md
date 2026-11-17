# Multiplex fork hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Under `gateway.multiplex_profiles: true`, every fork feature acts as the profile that owns the turn or cron job (its credentials, its Slack bot, its RBAC, its Slack flags), while single-profile behaviour stays byte-for-byte unchanged.

**Architecture:** Every fix is gated on `agent.secret_scope.is_multiplex_active()`. With the flag off, each touched code path runs exactly the code it runs today (the legacy branch is kept verbatim, not re-derived). With the flag on, reads go to the adapter's own `extra`, then the profile's secret scope, then the default — never `os.environ` for profile values. New fork logic lives in new fork-owned modules so upstream files only gain small call sites.

**Tech Stack:** Python 3.11, pytest via `scripts/run_tests.sh`, contextvars (`set_hermes_home_override`, `set_secret_scope`).

**Spec:** `docs/superpowers/specs/2026-11-16-multiplex-fork-hardening-design.md`

## Global Constraints

- Multiplex off → today's code path runs unchanged; regression tests assert today's values, including `.env` beating `config.yaml` for a key set in both.
- Multiplex on → precedence is adapter `extra` → scoped env → default; a scoped miss never falls through to `os.environ`.
- `UnscopedSecretError` in a fork tool → WARNING naming tool + key, then the tool's normal "not configured" error; `check_fn` returns False.
- The `.env` self-heal (`load_hermes_dotenv`) never runs under multiplex (it writes the default's `os.environ`).
- Tests: `scripts/run_tests.sh <path>` only; never without a path; one package at a time.
- Out of scope: `terminal.cwd` (known limitation).

## Review Focus

1. A Slack key set in **both** `.env` and `config.yaml` with multiplex off — `.env` must still win (Task 3 test `test_off_env_beats_yaml_for_env_only_keys`).
2. The default bot's **unscoped** adapter path under multiplex (adapter built outside a scope) must read its own env, and a secondary's YAML must not have leaked into `os.environ` (Task 3 test `test_secondary_yaml_does_not_reach_os_environ`).
3. A cloned profile (`shutil.copy2`, same mtime+size, different bytes) must not share RBAC (Task 1).
4. A secondary profile with **no** credential for a tool must get "not configured", not the default's value (Task 2 tests per tool, Drive must not fall back to ADC).
5. A secondary profile whose role is `*` must not be offered or be able to call the default's MCP tools, including through the server-name alias toolset (Task 5).

---

### Task 1: RBAC config cache keyed by path (fix 1)

**Files:** Modify `gateway/tool_access.py` (`_config_cache`, `_load_config_cached`). Test: `tests/gateway/test_tool_access_cache_per_profile.py`.

**Interfaces:** Produces `_load_config_cached()` (unchanged signature) returning the config of `get_hermes_home()/config.yaml`, cached per `(resolved path) → (mtime_ns, size, config)`.

- [ ] Test `test_cloned_profile_same_mtime_size_isolated`: two homes, config files of equal size with `user_roles` `{U1: admin}` vs `{U2: admin}`, `os.utime` both to the same ns; under each `set_hermes_home_override`, `policy_for_platform("slack").grant_for("U1", None)` is `{"*"}`-containing only in home A. Also `test_cache_still_reused_within_one_home` (load called once for two reads).
- [ ] Run, expect FAIL (second home returns first's config).
- [ ] Implement: `_config_cache: Dict[str, Tuple[fp, config]]` keyed by `str(cfg_file.resolve())`; keep the lock.
- [ ] Run `scripts/run_tests.sh tests/gateway/test_tool_access_cache_per_profile.py tests/gateway/test_tool_access.py` → PASS. Commit.

### Task 2: Fork tool credentials through the scope (fix 4)

**Files:** Create `agent/fork_secrets.py`. Modify `tools/jira_api_tool.py`, `tools/notion_api_tool.py`, `tools/webflow_asset_tool.py`, `tools/slack_react_tool.py` (`_resolve_slack_token`, used by `slack_post_thread`), `plugins/google_drive_sa/client.py`. Test: `tests/tools/test_multiplex_fork_tool_credentials.py`.

**Interfaces (produces):**
- `read_secret(name: str, *, tool: str, default: str = "") -> str` — multiplex off: `os.getenv(name, default).strip()` (exactly today). On: `get_secret`, stripped; `UnscopedSecretError` → `logger.warning("[%s] %s read with no profile scope under multiplex; treating as not configured", tool, name)` → `default`.
- `heal_dotenv(*, hermes_home=None) -> bool` — runs `load_hermes_dotenv` only when multiplex is off; returns whether it ran.
- `scoped_subprocess_env() -> dict[str, str]` — off: `dict(os.environ)`. On: `os.environ` minus every non-global key of `build_profile_secret_scope(get_process_hermes_home())`, then overlay `current_secret_scope()` (empty overlay when unscoped).

Steps:
- [ ] Tests, each with two scopes A/B holding different values and `os.environ` holding a third "default" value, multiplex on: jira `_jira_creds()` returns the scope's triple; scoped miss returns empties (not env); unscoped read returns empties and logs WARNING naming `jira_api` and `JIRA_API_TOKEN`; `_check_jira_api()` False when unscoped. Notion `_ntn_env()["NOTION_API_TOKEN"]` is the scope's key (bridged from `NOTION_API_KEY`), contains no default-only secret, and the dotenv loader is never called. Webflow `_webflow_token(site)` resolves `WEBFLOW_SITE_TOKENS` and the named var from the scope. Slack `_resolve_slack_token()` returns the scope's `SLACK_BOT_TOKEN` and `""` on a scoped miss (with `load_gateway_config` patched to a config without a Slack token). Drive: `_get_credentials()` called under home A then home B with different `GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON` returns distinct objects built from each scope; a scoped miss raises the "not configured" error and never calls `google.auth.default`. Multiplex-off regression: each tool returns the `os.environ` value and the self-heal still runs when env is empty.
- [ ] Run → FAIL.
- [ ] Implement `agent/fork_secrets.py`; switch each tool's `os.getenv` credential reads to `read_secret`, `load_hermes_dotenv` self-heals to `heal_dotenv`, notion's env base to `scoped_subprocess_env()` under multiplex (legacy branch untouched). Drive: under multiplex cache credentials in `dict[hermes_home_key(), creds]` and per-thread services keyed `(home_key, api, version)`; SA path from `read_secret`; no SA → `RuntimeError("Google Drive service account not configured for this profile")`.
- [ ] Run file + `tests/tools/test_jira_api_tool.py tests/tools/test_notion_api_tool.py tests/tools/test_webflow_asset_tool.py tests/tools/test_slack_react_tool.py tests/plugins/` (existing names as found) → PASS. Commit.

### Task 3: Slack adapter settings (fix 2, 4b adapter reads, cron wake guard)

**Files:** Modify `plugins/platforms/slack/adapter.py`. Test: `tests/gateway/test_slack_multiplex_settings.py`.

**Interfaces (produces):**
- Module constant `PROFILE_SETTINGS_KEY = "profile_settings"`.
- `_apply_yaml_config` returns `{PROFILE_SETTINGS_KEY: {k: v for every slack key it handles + quiet_channels, mention_patterns}}` (merged into `extra` by the existing dispatch in `gateway/config.py`), and skips every `os.environ` write when `is_multiplex_active()`.
- `SlackAdapter._slack_env(name: str, default=None) -> Optional[str]` — off: `os.getenv(name, default)`. On: the adapter's construction-time scope (`self._profile_secret_scope`, captured in `__init__` when multiplex is on) else `current_secret_scope()`; scope present → `scope.get(name, default)`; no scope → `os.getenv` (the unscoped default-profile adapter).
- `SlackAdapter._slack_setting(key: str, env_name: str, default=None)` — `extra[key]` if not None; then (multiplex on only) `extra[PROFILE_SETTINGS_KEY][key]`; then `_slack_env(env_name, default)`.
- Every `os.getenv("SLACK_…")` in the adapter goes through these (`_slack_require_mention`, strict/thread/other-user mention, wake-in-bot-authored, quiet, free-response, disable_dms, allowed/ignored/require-mention channels, mention patterns, allow_bots(+channels), reactions, reaction triggers/target, dedup TTL, interactive-auth `SLACK_ALLOW_ALL_USERS` and its `_env` helper — whose post-miss `os.getenv` fallback is removed under multiplex).

Steps:
- [ ] Tests: two adapters built under scopes A/B (multiplex on) with different `profile_settings` and scope values; `os.environ` set to a third value for every `SLACK_*` key; assert every reader above returns its own profile's value, a scoped miss returns the reader's default, and the cron-delivery wake-zone registration in `send()` follows each adapter's own `wake_in_bot_authored_threads`. `test_secondary_yaml_does_not_reach_os_environ`: `_apply_yaml_config` under multiplex writes nothing to env. Regression with multiplex off: `_apply_yaml_config` writes env exactly as today; `test_off_env_beats_yaml_for_env_only_keys` (`SLACK_STRICT_MENTION=false` in env, `strict_mention: true` in YAML → False); readers return today's values.
- [ ] Run → FAIL. Implement. Run file + `tests/gateway/test_slack*.py` → PASS. Commit.

### Task 4: Gateway runner reads per source profile (fix 3, 4b runner sites)

**Files:** Modify `gateway/run.py`, `gateway/slash_commands.py`, `gateway/authz_mixin.py`. Test: `tests/gateway/test_multiplex_runner_profile_reads.py`.

**Interfaces (produces):** `GatewayRunner._gateway_config_for_source(source) -> Any` — off: `self.config`. On: profile config loaded via `tool_access._load_config_cached()` inside `_profile_runtime_scope(self._resolve_profile_home_for_source(source))`; load failure → `None` (callers already treat `None` as "no policy").

Steps:
- [ ] Tests (runner via `object.__new__` with `adapters` / `_profile_adapters` of two fake Slack adapters with distinct `config.extra`): relevance gate receives the source's adapter; `_apply_channel_model_override` uses the educator adapter's `channel_models`; the RBAC "not authorized" DM goes out the source's adapter; `_check_slash_access` and `/whoami`-style policy use the source profile's `allow_admin_from`; `_is_slack_ignored_channel` uses the source profile's list; home-channel prompt flag per profile; `_slack_allow_bots_channels` env fallback reads the scope. Multiplex-off regression for each: `self.config` / `self.adapters` used exactly as today.
- [ ] Run → FAIL. Implement: `self._adapter_for_source(source)` at run.py ~19532, ~28186, ~18188; `_gateway_config_for_source` at the five `policy_for_source(self.config, …)` sites and both `_is_slack_ignored_channel` sites; `_slack_ignored_channels_from_gateway_config` and `_home_channel_prompt_enabled` env fallback via `_platform_gate_env`; authz_mixin fallback via `_platform_gate_env`.
- [ ] Run file + `tests/gateway/test_slash*.py tests/gateway/test_relevance*.py tests/gateway/test_channel_model*.py` → PASS. Commit.

### Task 5: Default-profile MCP tools never reach a secondary profile (4c)

**Files:** Create `gateway/multiplex_guard.py`. Modify `gateway/run.py` (`_resolve_enabled_toolsets_for_source`), `model_tools.py` (backstop beside `denial_for_current_tool`). Test: `tests/gateway/test_multiplex_mcp_guard.py`.

**Interfaces (produces):**
- `is_secondary_profile_context() -> bool` — multiplex on and `hermes_home_key()` ≠ `hermes_home_key(get_process_hermes_home())`.
- `mcp_server_for_toolset(toolset: str) -> Optional[str]` — `mcp-<x>` → `x`; a registry alias whose target is `mcp-<x>` → `x`; else None.
- `filter_foreign_mcp_toolsets(toolsets: list[str]) -> list[str]` — secondary context only: drop MCP toolsets whose server is not a key of the profile's own `config.yaml` `mcp_servers`.
- `foreign_mcp_denial(tool_name: str) -> Optional[str]` — secondary context: tool whose toolset is MCP and not declared → denial string; any internal error with an `mcp__` tool name → denial (fail closed).

Steps:
- [ ] Tests: secondary context drops `mcp-webflow` and alias `webflow`, keeps `mcp-own` declared in its config, keeps non-MCP; default context and multiplex-off return input unchanged; `foreign_mcp_denial("mcp__webflow__x")` denies in secondary, None in default/off; `_resolve_enabled_toolsets_for_source` applies it after RBAC.
- [ ] Run → FAIL. Implement. Run file + `tests/gateway/test_tool_access.py` → PASS. Commit.

### Task 6: File tools never cross served profile homes (4d) + opt-in write guard (4e)

**Files:** Create `agent/multiplex_file_guard.py`, `agent/profile_write_guard.py`. Modify `agent/file_safety.py` (`get_read_block_error`, `_classify_write_denial`, `get_write_denied_error`), `tools/skill_manager_tool.py` (`skill_manage`), `gateway/run.py` (register served homes where `served_profiles` is recorded). Tests: `tests/agent/test_multiplex_file_guard.py`, `tests/agent/test_profile_write_guard.py`.

**Interfaces (produces):**
- `set_served_profile_homes(homes: Iterable[str | Path]) -> None`; `cross_profile_denial(path: str) -> Optional[str]` — multiplex on only; denies a path inside another served home; the default home excludes its `profiles/` subtree; audits `record_access(tool="file", action="cross_profile_denied", target=path)`.
- `write_guard_denial(path: str) -> Optional[str]` — reads the **active** home's `config.yaml` `agent_write_guard` (`true`, or a mapping with `enabled` default true and `extra_paths`); denies `config.yaml`, `SOUL.md`, `cron/**`, `scripts/**`, `.env*` in that home, every `skills.external_dirs` entry, and `extra_paths` (relative → under home); audits `action="write_guard_denied"`. Independent of multiplex; off unless opted in.
- `_classify_write_denial` returns new categories `"cross_profile"` / `"write_guard"`; `get_write_denied_error` messages for both.

Steps:
- [ ] Tests: default↔educator both directions for read and write of `config.yaml`, `SOUL.md`, `skills/x/SKILL.md`, `memories/MEMORY.md`, `cron/jobs.json`; own-home paths allowed; non-served profile not denied; multiplex off → no denial; audit line written. Write guard: opted-in profile denied for each listed target, allowed for `skills/` and a workspace file; not opted in → allowed; skill_manage patch of a skill in an external dir refused.
- [ ] Run → FAIL. Implement. Run files + `tests/agent/test_file_safety*.py tests/tools/test_file_tools*.py tests/tools/test_skill_manager*.py` → PASS. Commit.

### Task 7: Cron runs as its profile (4f + module-level home sweep)

**Files:** Modify `cron/scheduler.py` (`_run_job_script` env base → `scoped_subprocess_env()`), `cron/notepad.py`, `cron/suggestions.py`, `agent/auxiliary_client.py` (`_AUTH_JSON_PATH`). Test: `tests/cron/test_multiplex_cron_per_profile.py`.

**Interfaces:** `cron.notepad._notepad_file() -> Path` and `cron.suggestions._cron_dir() -> Path`: the patched module constant if a test changed it, else `get_hermes_home()` per call (the `cron/jobs.py` pattern). Same for `_auth_json_path()`.

Sweep decisions: fix notepad, suggestions, auxiliary `auth.json` (credential). Leave `tools/process_registry.py` and `tools/checkpoint_manager.py` (terminal/checkpoint state, not reachable without `terminal`; listed as follow-ups), gateway-level constants (process-level by design), skills modules (already resolve per call).

Steps:
- [ ] Test: two profiles, multiplex on, a `no_agent` script job per profile run via `run_job` under `set_hermes_home_override` + `use_cron_store` + scope; each script prints `JIRA_API_TOKEN` and writes the notepad; assert own token, never the default's, and each notepad row lands in its own home. Multiplex-off regression: script env equals today's `build_subprocess_env()`.
- [ ] Run → FAIL. Implement. Run `scripts/run_tests.sh tests/cron/` → PASS (modulo pre-existing `test_monitor_kind.py` drift_skip). Commit.

### Task 8: End-to-end two Slack bots in one gateway (Acceptance)

**Files:** Create `tests/gateway/test_multiplex_fork_features.py`.

- [ ] One runner, multiplex on, two profile homes with distinct `.env` (SLACK/JIRA/NOTION/WEBFLOW/Drive values), `config.yaml` (user_roles, channel_roles, allow_admin_from, quiet/free-response/ignored channels, channel_models, platform_toolsets, mcp_servers), two fake Slack adapters with distinct tokens registered in `adapters` / `_profile_adapters`. Push one inbound message per bot through `_handle_message` with `_run_agent` stubbed to capture, inside the run, the RBAC grant, enabled toolsets, each tool credential, the file guard verdict for the other home, the session DB / memory / audit paths, and the replying adapter. Assert every acceptance row for both directions. Same flow with multiplex off returns today's values.
- [ ] Run → PASS (all earlier tasks in place). Commit.

### Task 9: Inventory, docs, verification, PR

- [ ] Add `WIRING` rows to `tests/test_fork_feature_inventory.py` for each new call site (cache key, `read_secret` in each tool, `_slack_setting`, `_gateway_config_for_source`, `filter_foreign_mcp_toolsets`, `foreign_mcp_denial`, `cross_profile_denial`, `write_guard_denial`, `scoped_subprocess_env` in cron).
- [ ] CLAUDE.md: "Multiplex" note + update the tools' "self-heal `.env`" wording.
- [ ] Run touched packages one at a time: `tests/gateway/`, `tests/tools/`, `tests/agent/`, `tests/cron/ tests/plugins/`, `tests/test_fork_feature_inventory.py`; diff failures against a clean tree.
- [ ] Push branch, open PR against `main`.
