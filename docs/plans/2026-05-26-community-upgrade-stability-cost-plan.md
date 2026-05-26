# Community-Driven Stability, Cost, and Extensibility Upgrade Plan

> **Draft PR target:** documentation/planning only. This plan turns recent
> community pain points into a reviewable PR stack. It does not implement the
> code changes by itself.

**Goal:** Upgrade Hermes from a broad feature surface into a more reliable,
lower-cost, auditable agent runtime. The plan prioritizes user-visible
stability, subprocess credential isolation, prompt/tool token reduction,
registry-first plugin behavior, and rollbackable self-improvement.

**Primary community signals:**

- Token overhead and context bloat: #4379, #6839.
- Gateway run lifecycle races and silent delivery failures: #31884, #10849.
- TUI/dashboard resource leaks: #32283, #32377.
- Plugin availability gates bypassing provider registries: #31873.
- Subprocess credential isolation gaps: #32314.
- Setup, Docker/VPS, memory drift, and self-improvement rollback friction from
  Reddit/community reports and recurring issue labels.

**Architecture direction:** Add deeper runtime modules with small interfaces:

- `RunLifecycle`: owns gateway turn states, generation tokens, final response
  classification, delivery result, and user-visible fallback behavior.
- `CredentialExposurePolicy`: owns env/file credential filtering for terminal,
  execute_code, MCP servers, subprocesses, and delegated agents.
- `ToolSchemaLoader`: owns eager/lazy schema exposure and platform-aware
  defaults.
- `ProviderAvailability`: registry-first availability, capability checks, and
  healthcheck behavior for plugin-backed providers.
- `MutationReceipt`: audit and rollback record for skills, memory, cron, and
  config changes.

**Open-source references to borrow from, not necessarily depend on:**

- OpenHands sandbox terminology and runtime separation:
  <https://docs.openhands.dev/openhands/usage/runtimes/overview>
- LangGraph durable execution, interrupt/resume, and checkpoint ideas:
  <https://docs.langchain.com/oss/python/langgraph/durable-execution>
- Temporal durable execution and activity retry model:
  <https://docs.temporal.io/>
- Aider repository-map approach for indexed context and on-demand expansion:
  <https://aider.chat/docs/repomap.html>
- LiteLLM routing, fallback, budget, and spend-tracking concepts:
  <https://docs.litellm.ai/>
- Open WebUI Functions/Pipelines split between lightweight in-process plugins
  and heavier out-of-process extension points:
  <https://docs.openwebui.com/features/extensibility/>

---

## Proposed PR Stack

### PR 1: `docs(plans): add community upgrade plan`

**Scope:** This file only.

**Purpose:** Give reviewers a shared roadmap before code lands.

**Validation:** Markdown review only.

### PR 2: `security(runtime): centralize subprocess credential exposure policy`

**Problem:** Provider-specific secrets can leak into terminal, execute_code,
MCP server, and delegated subprocess environments when they are not expressed
as `api_key_env_vars`, especially SDK-style providers such as Bedrock.

**Files likely involved:**

- `tools/environments/local.py`
- `tools/code_execution_tool.py`
- `tools/environments/docker.py`
- `tools/environments/base.py`
- `providers/base.py`
- `agent/bedrock_adapter.py`
- `tests/tools/`
- `tests/agent/`

**Implementation sketch:**

- Introduce `CredentialExposurePolicy` with one public interface:
  `sanitize_env(env, context) -> dict`.
- Include explicit SDK provider secrets such as `AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_BEARER_TOKEN_BEDROCK`,
  `AWS_PROFILE`, `AWS_ROLE_ARN`, and `AWS_WEB_IDENTITY_TOKEN_FILE`.
- Preserve explicit `terminal.env_passthrough` and skill-declared passthrough,
  but log when a passthrough overrides a normally blocked credential.
- Use the same policy in local terminal, Docker terminal, execute_code, and
  subprocess-backed helpers.

**Tests:**

- AWS credential vars are absent from terminal subprocess env.
- AWS credential vars are absent from execute_code child env.
- Explicit passthrough still works and is auditable.
- Non-secret OS essentials still pass on Windows.

### PR 3: `fix(gateway): make run lifecycle finalization explicit`

**Problem:** Gateway turns can complete with zero-length responses after
interrupts or generation invalidation, leaving users with no delivered reply
while internal work may continue.

**Files likely involved:**

- `gateway/run.py`
- `gateway/platforms/base.py`
- `tests/gateway/test_session_race_guard.py`
- `tests/gateway/test_interrupt_key_match.py`
- new `tests/gateway/test_run_lifecycle.py`

**Implementation sketch:**

- Add a `RunLifecycleResult` structure with:
  `status`, `final_response`, `generation`, `api_calls`, `tool_calls`,
  `delivery_required`, `undeliverable_reason`.
- Treat empty response for normal user text as `undeliverable`, not success.
- After `/stop`, ensure the next user message begins a clean generation.
- Log a structured lifecycle event for accepted, interrupted, completed,
  undeliverable, and delivered states.

**Tests:**

- `/stop` followed immediately by a normal Telegram-style message cannot
  produce a silent zero-length completion.
- Stale generation cannot clear newer run state.
- Empty response from a failed run sends a user-visible fallback.

### PR 4: `fix(tui): reap PTY slash workers and cache native library extraction`

**Problem:** Dashboard/TUI sessions can leak slash-worker subprocesses after
websocket disconnects, and native library extraction may fill `/tmp`.

**Files likely involved:**

- `hermes_cli/pty_bridge.py`
- `tui_gateway/server.py`
- `tui_gateway/entry.py`
- `ui-tui/`
- `tests/test_tui_gateway_server.py`
- new PTY lifecycle tests where feasible

**Implementation sketch:**

- Start TUI child and slash worker in a process group where supported.
- On websocket disconnect, close PTY, terminate process group, then kill after
  a bounded grace period.
- Add a startup cleanup/reaper for orphaned `tui_gateway.slash_worker`
  processes owned by the current Hermes dashboard process group.
- Cache native `.so` extraction under a deterministic Hermes cache path, or
  clean previous identical temp copies on startup.

**Tests:**

- Repeated PTY open/close does not leave slash worker processes.
- Close path is idempotent.
- Cleanup handles already-exited processes.

### PR 5: `perf(toolsets): add platform-aware messaging defaults`

**Problem:** Messaging platforms inherit nearly all core tools, including heavy
or unusable browser/computer/kanban/TTS surfaces. This increases fixed prompt
cost on every turn.

**Files likely involved:**

- `toolsets.py`
- `hermes_cli/config.py`
- `website/docs/reference/toolsets-reference.md`
- `tests/agent/test_memory_provider.py`
- new `tests/test_toolsets_platform_defaults.py`

**Implementation sketch:**

- Add `_HERMES_MESSAGING_TOOLS` derived from core tools minus heavy tools that
  are not generally useful in chat platforms.
- Keep existing full access available via explicit config override.
- Document migration and override examples.

**Tests:**

- Telegram/WhatsApp/Slack/Signal default to slim messaging tools.
- User `platform_toolsets` override still restores full or custom tools.
- Webhook safe toolset remains constrained.

### PR 6: `perf(skills): support lazy skills prompt mode`

**Problem:** The skills catalog is injected into every system prompt even when
the turn does not need a skill.

**Files likely involved:**

- `agent/prompt_builder.py`
- `agent/system_prompt.py`
- `hermes_cli/config.py`
- `tools/skills_tool.py`
- `tests/run_agent/test_run_agent.py`
- `tests/agent/test_skill_commands_reload.py`

**Implementation sketch:**

- Add config:

```yaml
skills:
  prompt_mode: eager  # eager | lazy | off
```

- `eager`: current behavior.
- `lazy`: short instruction plus `skills_list`/`skill_view`, no full catalog.
- `off`: no skills prompt, tools remain callable if enabled.
- Keep `/reload-skills` cache-preserving behavior.

**Tests:**

- `lazy` mode excludes full catalog but keeps skill-use guidance.
- `eager` remains byte-compatible enough for existing assertions.
- `off` does not remove skill tools from valid tool names.

### PR 7: `fix(web): make web provider availability registry-first`

**Problem:** Plugin-registered web providers can be configured but hidden from
tool schemas because availability checks use hardcoded backend lists.

**Files likely involved:**

- `tools/web_tools.py`
- `agent/web_search_registry.py`
- `hermes_cli/tools_config.py`
- `tests/tools/test_web_tools_config.py`
- `tests/plugins/web/test_web_search_provider_plugins.py`

**Implementation sketch:**

- Make `_get_backend`, `_is_backend_available`, and `check_web_api_key`
  consult `agent.web_search_registry` first.
- Fall back to legacy built-ins only when no registry provider matches.
- Log configured-but-unregistered and configured-but-unavailable cases.

**Tests:**

- Fake plugin provider with custom env var makes `check_web_api_key()` true.
- Configured unavailable provider yields a clear unavailable result.
- Legacy built-ins keep existing behavior.

### PR 8: `feat(self-improvement): add mutation receipts and rollback snapshots`

**Problem:** Auto-improving skills/memory/config changes are powerful but users
cannot easily audit or roll them back when a change breaks a working setup.

**Files likely involved:**

- `tools/skills_tool.py`
- `agent/curator.py`
- `agent/background_review.py`
- `agent/memory_manager.py`
- `tools/cronjob_tools.py`
- `hermes_cli/backup.py`
- `tests/agent/test_curator_backup.py`
- new `tests/test_mutation_receipts.py`

**Implementation sketch:**

- Add `MutationReceipt` records under `$HERMES_HOME/receipts/`.
- Record mutation type, files touched, old/new hash, session id, reason, and
  rollback command.
- Snapshot before `skill_manage`, curator writes, cron writes, and memory
  compactions that mutate durable files.
- Add `hermes receipts list/show/rollback`.

**Tests:**

- Skill patch creates a receipt and rollback restores previous content.
- Failed mutation does not leave a committed receipt.
- Receipts are profile-aware through `get_hermes_home()`.

### PR 9: `feat(doctor): add full runtime smoke matrix and update rollback`

**Problem:** Users spend time discovering broken provider/gateway/dashboard
state manually after installs, restarts, and updates.

**Files likely involved:**

- `hermes_cli/doctor.py`
- `hermes_cli/update.py` or current update path
- `hermes_cli/web_server.py`
- `website/docs/getting-started/updating.md`
- `website/docs/user-guide/docker.md`
- `tests/hermes_cli/`

**Implementation sketch:**

- Add `hermes doctor --full` checks for provider, terminal, execute_code,
  web backend, memory provider, gateway adapters, dashboard websocket, cron,
  HERMES_HOME permissions, and disk pressure.
- Add pre-update snapshot manifest and `hermes rollback update`.
- Add dashboard read-only health panel data endpoint.

**Tests:**

- Doctor reports degraded subsystems without crashing.
- Update snapshot is created before mutation.
- Rollback restores config/state paths in a temp Hermes home.

---

## Draft PR Body For PR 1

## What does this PR do?

Adds a community-driven upgrade plan covering the highest-signal reliability,
cost, extensibility, and operator-experience pain points currently visible in
Hermes Agent community reports and open issues.

The plan converts those findings into a staged PR stack so maintainers can
review the intended architecture before implementation starts. It focuses on:

- subprocess credential isolation;
- gateway run lifecycle finalization;
- TUI/dashboard process cleanup;
- platform-aware toolset defaults;
- lazy skills/tool schema loading;
- registry-first provider availability;
- mutation receipts and rollback for self-improvement;
- fuller install/update/runtime diagnostics.

## Related Issue

Refs #4379, #6839, #31873, #31884, #32314, #32283, #32377, #10849.

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Security fix
- [x] Documentation update
- [ ] Tests (adding or improving test coverage)
- [ ] Refactor (no behavior change)
- [ ] New skill (bundled or hub)

## Changes Made

- Added `docs/plans/2026-05-26-community-upgrade-stability-cost-plan.md`.
- Captured the proposed implementation PR stack, likely files, test strategy,
  and open-source references.

## How to Test

1. Read the plan and verify the issue references match current community pain
   points.
2. Confirm the proposed PR stack is reviewable and can be split into
   independently mergeable implementation PRs.
3. Confirm the plan does not change runtime behavior.

## Checklist

### Code

- [x] My PR contains only changes related to this fix/feature.
- [ ] I've run `pytest tests/ -q` and all tests pass.
- [x] No runtime code changed.

### Documentation & Housekeeping

- [x] I've updated relevant documentation.
- [x] Config changes are only proposed, not implemented.
- [x] Architecture/workflow impact is documented in the plan.
- [x] Cross-platform impact is called out where relevant.

## Screenshots / Logs

N/A. Documentation-only planning PR.
