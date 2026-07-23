# HQ Hermes Pre-Update Audit Report

Generated: 2026-07-05T04:38:46 local time
Repo: `C:/Users/82109/AppData/Local/hermes/hermes-agent`
Mode: read-only audit; no reset, merge, rebase, stash apply, delete, or update was performed.

## Executive summary

- **Update status:** `hermes update --check` reports already up to date.
- **Git divergence:** current `main` is **ahead 74 / behind 0** vs `origin/main`.
- **Current HEAD:** `8b97f5e51` / `8b97f5e51aff74daaaa97cd11e467b621a1ca119`
- **origin/main:** `7203898ce` / `7203898ce47c9ab90e64866d6cff0e6e9ad8d1cc`
- **Tracked worktree:** clean except branch divergence; no staged/tracked dirty files in `git status`.
- **Untracked:** `apps/desktop/release-staging/builder-debug.yml` is present and should be reviewed before cleanup.
- **Local carried stack:** `138 files changed, 11025 insertions(+), 605 deletions(-)`
- **Stashes:** 9 visible stash entries.
- **No-merged local branches:** 35 entries in the first listed page.
- **Worktrees:** 24 worktrees recorded.

## Current git state

```text
## main...origin/main [ahead 74]
?? apps/desktop/release-staging/
```

```text
branch=main
head=8b97f5e51aff74daaaa97cd11e467b621a1ca119
upstream=origin/main
upstream_full=7203898ce47c9ab90e64866d6cff0e6e9ad8d1cc
origin_main=7203898ce47c9ab90e64866d6cff0e6e9ad8d1cc
ahead_behind_HEAD_origin_main=74	0
```

## Hermes version/update check

```text
Hermes Agent v0.18.0 (2026.7.1) · upstream 7203898c · local 8b97f5e5 (+74 carried commits)
Project: C:\Users\82109\AppData\Local\hermes\hermes-agent
Python: 3.11.15
OpenAI SDK: 2.24.0
Up to date

→ Fetching from upstream...
→ Fetching from origin...
✓ Already up to date.
```

## Clarify preservation spot-check

The earlier missing clickable choice was checked because it could indicate a dropped branch/commit. Current evidence says the core clarify stack is **present in HEAD** and the miss was an agent response-protocol failure, not a code-loss symptom.

```text
IN_HEAD b3123618c feat(clarify): restore native multi-select prompts
IN_HEAD 13c8445f4 fix(clarify): harden constrained-choice semantics
IN_HEAD 45e737a75 fix(clarify): keep multi-select choice UX visible
IN_HEAD e78c7f85b fix(clarify): enforce multi-select response bounds
IN_HEAD fa522dae9 fix(clarify): reject empty required multi-select replies
IN_HEAD 2aa38fba5 fix(clarify): cover task report scope choices
IN_HEAD d6c19f820 fix(desktop): restore clarify keyboard shortcuts
```

Additional focused tests already run in this session:

```text
tests/tools/test_clarify_tool.py: 44 passed
Desktop clarify focused tests: 3 files passed, 22 tests passed
```

## Local carried diff scope

```text
138 files changed, 11025 insertions(+), 605 deletions(-)
```

Directory-level touched file counts:

```text
apps: 53
tests: 35
references: 15
tools: 10
hermes_cli: 5
scripts: 4
agent: 3
gateway: 3
tui_gateway: 3
plugins: 2
website: 2
cron: 1
reports: 1
skills: 1
```

Top changed files (`origin/main..HEAD`):

```text
M	agent/agent_runtime_helpers.py
A	agent/eval_gate.py
M	agent/tool_executor.py
M	apps/desktop/electron/backend-ready.test.cjs
M	apps/desktop/electron/main.cjs
M	apps/desktop/electron/preload.cjs
M	apps/desktop/electron/update-relaunch.cjs
M	apps/desktop/electron/update-relaunch.test.cjs
M	apps/desktop/src/app/chat/composer/attachments.test.tsx
M	apps/desktop/src/app/chat/sidebar/index.tsx
M	apps/desktop/src/app/command-center/index.tsx
M	apps/desktop/src/app/command-palette/index.tsx
M	apps/desktop/src/app/gateway/hooks/use-gateway-boot.ts
M	apps/desktop/src/app/messaging/index.test.tsx
A	apps/desktop/src/app/session/hooks/use-message-stream/gateway-event.test.tsx
M	apps/desktop/src/app/session/hooks/use-message-stream/gateway-event.ts
M	apps/desktop/src/app/session/hooks/use-preview-routing.test.tsx
M	apps/desktop/src/app/session/hooks/use-prompt-actions/index.test.tsx
M	apps/desktop/src/app/session/hooks/use-prompt-actions/submit.ts
M	apps/desktop/src/app/session/hooks/use-prompt-actions/utils.ts
M	apps/desktop/src/app/settings/index.tsx
M	apps/desktop/src/app/settings/model-settings.test.tsx
M	apps/desktop/src/app/settings/types.ts
M	apps/desktop/src/app/shell/hooks/use-statusbar-items.tsx
A	apps/desktop/src/components/assistant-ui/clarify-tool.test.tsx
M	apps/desktop/src/components/assistant-ui/clarify-tool.tsx
M	apps/desktop/src/components/assistant-ui/tool/fallback-model.test.ts
M	apps/desktop/src/components/boot-failure-overlay.tsx
M	apps/desktop/src/components/gateway-connecting-overlay.test.tsx
A	apps/desktop/src/components/haptics-provider.test.tsx
M	apps/desktop/src/components/haptics-provider.tsx
M	apps/desktop/src/components/pane-shell/pane-shell.test.tsx
M	apps/desktop/src/global.d.ts
M	apps/desktop/src/i18n/en.ts
M	apps/desktop/src/i18n/ja.ts
M	apps/desktop/src/i18n/types.ts
M	apps/desktop/src/i18n/zh-hant.ts
M	apps/desktop/src/i18n/zh.ts
M	apps/desktop/src/lib/chat-messages.test.ts
M	apps/desktop/src/lib/chat-messages.ts
A	apps/desktop/src/lib/haptics.test.ts
M	apps/desktop/src/lib/haptics.ts
M	apps/desktop/src/lib/icons.ts
A	apps/desktop/src/lib/skill-mode-prefix.test.ts
A	apps/desktop/src/lib/skill-mode-prefix.ts
M	apps/desktop/src/store/boot.ts
M	apps/desktop/src/store/clarify.ts
A	apps/desktop/src/store/layout.test.ts
M	apps/desktop/src/store/layout.ts
M	apps/desktop/src/store/panes.test.ts
M	apps/desktop/src/store/session.test.ts
M	apps/desktop/src/store/session.ts
M	apps/desktop/src/styles.css
A	apps/desktop/src/test/setup.ts
M	apps/desktop/src/vite-env.d.ts
M	apps/desktop/vite.config.ts
M	cron/scheduler.py
M	gateway/platforms/base.py
M	gateway/platforms/whatsapp_cloud.py
M	gateway/run.py
M	hermes_cli/_subprocess_compat.py
M	hermes_cli/config.py
M	hermes_cli/plugins.py
M	hermes_cli/tools_config.py
M	hermes_cli/web_server.py
M	plugins/platforms/discord/adapter.py
M	plugins/platforms/telegram/adapter.py
A	references/hq-eval-gate-build-packet-20260620.md
A	references/hq-harness-awesome-patterns-mapping.md
A	references/hq-harness-dashboard-adapter-build-packet.md
A	references/hq-health-dashboard-harness-integration-build-packet.md
A	references/hq-intention-tool-graph-prototype.md
A	references/hq-interactive-coder-policy.md
A	references/hq-ledgeragent-execution-state-mapping.md
A	references/hq-lifelong-eval-fixture-plan.md
A	references/hq-memory-quality-gate-review-prep.md
A	references/hq-memory-trust-gate-integration-plan.md
A	references/hq-memory-trust-gate-tam-pattern.md
A	references/hq-runtime-budget-policy-v0.md
A	references/hq-self-improvement-replay-protocol.md
A	references/hq-stamina-bench-fixture-plan.md
A	references/hq-trace-code-state-space-metrics.md
A	reports/hq_memory_trust_gate_audit_latest.md
A	scripts/hq_coder_depth_classifier.py
A	scripts/hq_harness_dashboard_adapter.py
A	scripts/hq_harness_validator.py
A	scripts/hq_health_dashboard.py
A	skills/devops/hermes-agent-cli/SKILL.md
M	tests/agent/test_credential_pool_oauth_writethrough.py
A	tests/agent/test_eval_gate.py
M	tests/cli/test_cli_approval_ui.py
A	tests/cron/test_eval_gate_pre_dispatch.py
A	tests/cron/test_scheduler_silent_delivery.py
M	tests/gateway/test_config_env_bridge_authority.py
M	tests/gateway/test_discord_clarify_buttons.py
M	tests/gateway/test_platform_base.py
M	tests/gateway/test_pre_gateway_dispatch.py
M	tests/gateway/test_restart_drain.py
M	tests/gateway/test_telegram_clarify_buttons.py
M	tests/gateway/test_telegram_network_reconnect.py
M	tests/gateway/test_whatsapp_cloud.py
M	tests/hermes_cli/test_config.py
M	tests/hermes_cli/test_debug.py
M	tests/hermes_cli/test_tools_config.py
M	tests/run_agent/test_run_agent.py
M	tests/run_agent/test_tool_call_incremental_persistence.py
A	tests/scripts/test_hq_harness_dashboard_adapter.py
A	tests/scripts/test_hq_harness_validator.py
A	tests/scripts/test_hq_health_dashboard_harness_integration.py
M	tests/test_profile_isolation_runtime.py
M	tests/test_tui_gateway_server.py
M	tests/test_tui_gateway_ws.py
M	tests/tools/test_base_environment.py
A	tests/tools/test_browser_input_execution.py
M	tests/tools/test_clarify_gateway.py
M	tests/tools/test_clarify_tool.py
A	tests/tools/test_computer_use_capabilities.py
A	tests/tools/test_desktop_control_routing.py
A	tests/tools/test_local_cdp_browser_backend.py
M	tests/tools/test_skill_manager_tool.py
A	tests/tools/test_windows_uia_readonly.py
M	tests/tui_gateway/test_inline_rpc_gil_starvation.py
A	tests/tui_gateway/test_loop_monitor.py
M	tools/clarify_gateway.py
M	tools/clarify_tool.py
A	tools/computer_use/browser_input.py
A	tools/computer_use/capabilities.py
A	tools/computer_use/proposals.py
A	tools/computer_use/routing.py
A	tools/computer_use/windows_uia_readonly.py
M	tools/environments/local.py
M	tools/kanban_tools.py
M	tools/process_registry.py
A	tui_gateway/loop_monitor.py
M	tui_gateway/server.py
M	tui_gateway/ws.py
A	website/docs/developer-guide/desktop-control-routing.md
M	website/sidebars.ts
```

## Untracked files

```text
apps/desktop/release-staging/builder-debug.yml
```

### `apps/desktop/release-staging/builder-debug.yml`

present; 17 lines, 828 bytes; first lines:

```yaml
x64:
  firstOrDefaultFilePatterns:
    - '!**/node_modules/**'
    - '!build{,/**/*}'
    - '!release-staging{,/**/*}'
    - dist/**
    - assets/**
    - electron/**
    - public/**
    - package.json
    - '!**/*.{iml,hprof,orig,pyc,pyo,rbc,swp,csproj,sln,suo,xproj,cc,d.ts,mk,a,o,obj,forge-meta,pdb}'
    - '!**/._*'
    - '!**/electron-builder.{yaml,yml,json,json5,toml,ts}'
    - '!**/{.git,.hg,.svn,CVS,RCS,SCCS,__pycache__,.DS_Store,thumbs.db,.gitignore,.gitkeep,.gitattributes,.npmignore,.idea,.vs,.flowconfig,.jshintrc,.eslintrc,.circleci,.yarn-integrity,.yarn-metadata.json,yarn-error.log,yarn.lock,package-lock.json,npm-debug.log,pnpm-lock.yaml,bun.lock,bun.lockb,appveyor.yml,.travis.yml,circle.yml,.nyc_output,.husky,.github,electron-builder.env}'
    - '!.yarn{,/**/*}'
    - '!.editorconfig'
    - '!.yarnrc.yml'
```

Recommendation: classify this as `defer` until its purpose is known. It is outside tracked source and may be a local desktop release/build diagnostic artifact. Do not delete without approval.

## Stash inventory

```text
stash@{2026-07-03 02:45:56 +0900}: On main: hq-pre-active-main-apply-dirty-20260703-024555
stash@{2026-06-29 04:37:01 +0900}: On main: hermes-update-autostash-20260628-193701
stash@{2026-06-28 22:21:28 +0900}: On main: hq-pre-pins-dirty-maincjs-20260628-222128
stash@{2026-06-28 07:27:22 +0900}: On main: hq-pre-recovery-dirty-maincjs-package-lock-report-20260628-072722
stash@{2026-06-27 07:07:08 +0900}: On main: hermes-update-autostash-20260626-220708
stash@{2026-06-27 01:14:09 +0900}: On main: hermes-update-autostash-20260626-161409
stash@{2026-06-24 10:20:50 +0900}: On main: hermes-update-autostash-20260624-012050
stash@{2026-06-23 07:58:56 +0900}: On hq/performance-auto-skill-compact-20260623: hermes-update-autostash-20260622-225856
stash@{2026-06-15 19:36:34 +0900}: On main: hermes-update-autostash-20260615-103634
```

Recommendation: `defer` all stashes until inspected path-by-path. Several names are update/autostash/recovery related and may preserve local HQ changes. Do not apply wholesale.

## No-merged branch inventory, first 120

```text
backup/main-before-clarify-multiselect-20260629-033040 2026-06-29 03:23:27 +0900 87dfee314 fix(desktop): explain prompt submit timeouts
computer-use-browser-uia-v0-20260623-0415 2026-06-23 15:24:15 +0900 c4373913d feat(computer-use): add safe local CDP browser backend
feat/clarify-multi-select 2026-06-25 01:50:24 +0900 341d8e236 feat(clarify): support native multi-select prompts
fix/hq-gateway-windows-restart-20260626-093823 2026-06-26 09:39:24 +0900 14e4e37c4 fix(gateway): harden Windows restart path
hq-harness-validator-v0 2026-06-23 02:36:40 +0900 52aec00b2 feat(hq): add eval gate quality harness
hq-main-sync-20260629-055845 2026-06-29 07:48:43 +0900 3f8d58de6 fix(clarify): harden constrained-choice semantics
hq/adopt-audit-candidates-20260704-032635 2026-07-04 05:21:29 +0900 b9a483b02 fix(desktop): adopt audited clarify, auto-skill, and verification helpers
hq/clarify-multiselect-profile-routing-final-20260627-185047 2026-06-27 19:03:19 +0900 afbf21d13 fix: route clarify responses by owner profile
hq/clarify-multiselect-v4-20260626-052357 2026-06-26 05:24:11 +0900 bb68eb0dd feat(clarify): restore native multi-select flow
hq/clarify-nonblocking-polish-20260627-191809 2026-06-27 19:55:15 +0900 6e48743a4 test: polish clarify multi-select i18n coverage
hq/clarify-select-ui-polish-20260626-065024 2026-06-26 06:54:17 +0900 dd7571175 fix(desktop): align clarify multi-select action UI
hq/current-main-with-clarify-20260629-0458 2026-06-29 03:35:37 +0900 4e4e94367 feat(clarify): restore native multi-select prompts
hq/desktop-c1c2-restage-20260702-014521 2026-07-02 02:00:48 +0900 1b57a8432 style(desktop): satisfy restaged prompt submit lint
hq/desktop-c1c2-staging-20260701-214241 2026-07-01 23:16:09 +0900 d4186dfae fix(clarify): reject mixed constrained responses
hq/desktop-clarify-stream-bounds-20260701 2026-07-01 03:43:03 +0900 5c4e995bd fix(desktop): bound live terminal output events
hq/fork-main-total-20260704-023904 2026-07-04 03:00:35 +0900 b023a5bda merge: apply active local main stack to total candidate
hq/hermes-native-verification-ledger-20260629 2026-06-29 02:48:29 +0900 98bdf7b7b feat: add verification and ULW evidence helpers
heads/hq/integration-ready-20260629-120408 2026-06-29 11:25:02 +0900 0f38d976d style(desktop): satisfy clarify lint spacing
hq/integration-ready-post-rebase-20260629-120801 2026-06-29 12:05:54 +0900 5b1ed6cd4 style(desktop): sort terminal imports after rebase
hq/latest-main-20260629-085629 2026-06-29 09:05:51 +0900 11fb67d4a fix(clarify): harden constrained-choice semantics
hq/local-reintegrate-20260629-111738 2026-06-29 12:05:54 +0900 5b1ed6cd4 style(desktop): sort terminal imports after rebase
hq/local-total-ux-ui-20260701 2026-07-01 05:05:24 +0900 c458e48ee test(desktop): reconcile clarify integration expectations
hq/merge-choice-pins-numbering-20260626-001436 2026-06-26 00:19:29 +0900 f54111a1f feat(desktop): sort numbered sessions consistently
hq/open-work-completion-20260622 2026-06-23 00:04:03 +0900 603175758 feat(desktop): show live clarify selection status
hq/performance-auto-skill-compact-20260623 2026-06-23 00:04:03 +0900 603175758 feat(desktop): show live clarify selection status
hq/pre-desktop-integrate-main-20260629-075945 2026-06-29 03:35:37 +0900 4e4e94367 feat(clarify): restore native multi-select prompts
heads/hq/pre-latest-main-apply-20260629-0920 2026-06-29 08:05:48 +0900 8cde1e99b fix(clarify): harden constrained-choice semantics
hq/pre-local-main-sync-20260630-000147 2026-06-30 00:01:41 +0900 97db1f5c3 fix(desktop): distinguish lost gateway and bound tool result events
hq/pre-update-20260629-105143 2026-06-29 10:37:14 +0900 6a20a5d88 fix(clarify): reject empty required multi-select replies
hq/preserve-untracked-experiments-20260626-062751 2026-06-26 06:28:42 +0900 67725756f chore(hq): preserve local desktop experiments
hq/protect-old-c1c2-staging-20260702-014521 2026-07-01 23:16:09 +0900 d4186dfae fix(clarify): reject mixed constrained responses
hq/protected-main-20260629-0516 2026-06-29 03:35:37 +0900 4e4e94367 feat(clarify): restore native multi-select prompts
hq/recovery-stack-pr-20260628-230443 2026-06-28 22:29:44 +0900 bd5e975e5 feat(desktop): add file-backed session pins bridge
hq/update-staging-origin-main-20260704-164509 2026-07-04 17:59:22 +0900 af60fd065 chore(hq): stage active main on latest origin/main
pr/adopt-audit-candidates-origin-main-20260704-0529 2026-07-04 06:15:59 +0900 47378bc9d fix(desktop): adopt audited clarify, auto-skill, and verification helpers
```

Recommended classification approach:

| Class | Meaning for this repo | Current candidates |
|---|---|---|
| `adopt` | already on active main or should be kept exactly | active carried stack on `main` until proven otherwise |
| `integrate-better` | valuable HQ/Desktop/Gateway feature but should be refactored/upstream-aligned before publish | clarify/Desktop/HQ harness/computer-use branches with overlapping newer implementations |
| `bugfix` | fixes a real current regression | Windows gateway restart, delayed clarify, prompt timeout fixes if current tests show regression |
| `delete-by-policy` | stale experiment/no longer useful/security/UX clutter | none recommended without deeper branch-specific review |
| `defer` | preserve branch/tag/stash and review later | old worktrees/stashes and untracked release-staging artifact |

## Worktrees

```text
worktree C:/Users/82109/AppData/Local/hermes/hermes-agent
HEAD 8b97f5e51aff74daaaa97cd11e467b621a1ca119
branch refs/heads/main

worktree C:/Users/82109/AppData/Local/hermes/worktrees/adopt-audit-candidates-20260704-032635
HEAD b9a483b021a8f1c1488d9b8492aaab735bf71cf1
branch refs/heads/hq/adopt-audit-candidates-20260704-032635

worktree C:/Users/82109/AppData/Local/hermes/worktrees/adopt-audit-origin-pr-20260704-0529
HEAD 47378bc9d528263f762db1fa214b2d5e27a1df2f
branch refs/heads/pr/adopt-audit-candidates-origin-main-20260704-0529

worktree C:/Users/82109/AppData/Local/hermes/worktrees/computer-use-browser-uia-v0-20260623-0415
HEAD c4373913dd3a6e4dcbb2cf6a9b3ce667c4555d88
branch refs/heads/computer-use-browser-uia-v0-20260623-0415

worktree C:/Users/82109/AppData/Local/hermes/worktrees/hq-desktop-c1c2-restage-20260702-014521
HEAD 1b57a843259a851f31f0d4050d8400f343a66d19
branch refs/heads/hq/desktop-c1c2-restage-20260702-014521

worktree C:/Users/82109/AppData/Local/hermes/worktrees/hq-desktop-c1c2-restage-20260702-051612
HEAD 13eaee13cb9ee7d9184036a3a8527a2fc5ade878
branch refs/heads/hq/desktop-c1c2-restage-20260702-051612

worktree C:/Users/82109/AppData/Local/hermes/worktrees/hq-desktop-c1c2-staging-20260701-214241
HEAD d4186dfae2e8b165d627ab643ba082d15350a9f2
branch refs/heads/hq/desktop-c1c2-staging-20260701-214241

worktree C:/Users/82109/AppData/Local/hermes/worktrees/hq-desktop-clarify-stream-bounds-20260701
HEAD 5c4e995bdc8cedfde8b26694b19d5005f68b212f
branch refs/heads/hq/desktop-clarify-stream-bounds-20260701

worktree C:/Users/82109/AppData/Local/hermes/worktrees/hq-fork-main-total-20260701-053639
HEAD b4d1beb4e1290f0957c957a03286dea7715e6795
branch refs/heads/hq/fork-main-total-20260701-053639

worktree C:/Users/82109/AppData/Local/hermes/worktrees/hq-fork-main-total-20260702-144936
HEAD 52206678ebdd98ed19a3ba79af68b11336e872e6
branch refs/heads/hq/fork-main-total-20260702-144936

worktree C:/Users/82109/AppData/Local/hermes/worktrees/hq-fork-main-total-20260704-023904
HEAD b023a5bda0bd45a7500da6f8fb5b438584a363e0
branch refs/heads/hq/fork-main-total-20260704-023904

worktree C:/Users/82109/AppData/Local/hermes/worktrees/hq-gateway-windows-restart-20260626-093823
HEAD 14e4e37c47ac0e98b9ce4bc4bfdbacd4cf638e1b
branch refs/heads/fix/hq-gateway-windows-restart-20260626-093823

worktree C:/Users/82109/AppData/Local/hermes/worktrees/hq-haptics-reapply-20260703-0850
HEAD ed4123792c135558e7be2e486505bc569faa2a74
branch refs/heads/hq/haptics-reapply-20260703-0850

worktree C:/Users/82109/AppData/Local/hermes/worktrees/hq-haptics-reapply-latest-20260703-1010
HEAD 89acc196067c3a4a8987a8f0d01ed4e08d7daa2d
branch refs/heads/hq/haptics-reapply-latest-20260703-1010

worktree C:/Users/82109/AppData/Local/hermes/worktrees/hq-harness-validator-v0
HEAD 52aec00b245b7651a58372bc48c383510662bec3
branch refs/heads/hq-harness-validator-v0

worktree C:/Users/82109/AppData/Local/hermes/worktrees/hq-hermes-scope-c-20260629
HEAD 98bdf7b7b542c6ee3dfed0a7229da1d13efdfe1f
branch refs/heads/hq/hermes-native-verification-ledger-20260629

worktree C:/Users/82109/AppData/Local/hermes/worktrees/hq-latest-main-20260629-085629
HEAD 11fb67d4abae82f6585064139589d7e4022604b6
branch refs/heads/hq/latest-main-20260629-085629

worktree C:/Users/82109/AppData/Local/hermes/worktrees/hq-local-total-ux-ui-20260701
HEAD c458e48eeb2fc222f75ec0bb3aa897d2b151ca1c
branch refs/heads/hq/local-total-ux-ui-20260701

worktree C:/Users/82109/AppData/Local/hermes/worktrees/hq-main-sync-20260629-055845
HEAD 3f8d58de6f482a3dd0d09f6d67c0ee920c023247
branch refs/heads/hq-main-sync-20260629-055845

worktree C:/Users/82109/AppData/Local/hermes/worktrees/hq-origin-main-drift7-20260705-013118
HEAD 8b97f5e51aff74daaaa97cd11e467b621a1ca119
branch refs/heads/hq/origin-main-drift7-20260705-013118

worktree C:/Users/82109/AppData/Local/hermes/worktrees/hq-origin-main-security-drift-20260704-224926
HEAD 55e9e47eb126e66f4a5501a84d25a4e8a10f8940
branch refs/heads/hq/origin-main-security-drift-20260704-224926

worktree C:/Users/82109/AppData/Local/hermes/worktrees/hq-update-staging-20260704-164509
HEAD af60fd0651dedcc71b9f63cac8eeae4a12af37de
branch refs/heads/hq/update-staging-origin-main-20260704-164509

worktree C:/Users/82109/AppData/Local/hermes/worktrees/hq-update-staging-20260704-181613
HEAD d6c19f820a7f4905018c97dcafc923aabdfaca49
branch refs/heads/hq/update-staging-origin-main-20260704-181613

worktree C:/Users/82109/AppData/Local/hermes/worktrees/hq-upstream-verify-20260628-222651
HEAD bd5e975e54d9d7ce87d02f2b4bb5c62b342e73f9
branch refs/heads/hq/recovery-stack-pr-20260628-230443
```

Risk note: many linked worktrees still point to local HQ branches. Before pruning any branch or worktree, verify it is either merged into active `main`, superseded by a newer branch, or intentionally archived.

## Local commits not in origin/main, first 120

```text
8b97f5e51 (HEAD -> main, tag: backup-pre-runtime2-apply-20260705-013852, hq/origin-main-drift7-20260705-013118, backup/pre-runtime2-apply-20260705-013852) merge: integrate latest origin/main drift
55e9e47eb (tag: backup-pre-drift12-source-ff-20260705-013437, hq/origin-main-security-drift-20260704-224926, backup/pre-drift12-source-ff-20260705-013437) merge: integrate origin/main security drift
d6c19f820 (tag: backup-pre-active-source-ff-20260704-231628, fork/hq/update-staging-origin-main-20260704-181613, hq/update-staging-origin-main-20260704-181613, backup/pre-active-source-ff-20260704-231628) fix(desktop): restore clarify keyboard shortcuts
dc065158f chore(hq): stage active main update integration
8bcbc5179 (tag: hq/pre-update-active-main-20260704-163817, tag: hq/pre-total-active-main-20260704-023904, tag: hq-pre-active-main-apply-20260704-d6c19f820-pre-active-apply, fork/perf/tabler-direct-icons, perf/tabler-direct-icons, hq/protect-active-main-before-update-20260704-163817, hq/pre-active-main-apply-20260704-d6c19f820-pre-active-apply) perf(desktop): import Tabler icons directly
d00f0f4e9 chore: clean up desktop build warnings
10b092d82 test: sync desktop renderer expectations
545277df4 feat: gate desktop haptics debug telemetry
2875dad0c fix: harden desktop ws loop stall diagnostics
52206678e (tag: hq/pre-active-main-apply-candidate-20260703-024555, hq/fork-main-total-20260702-144936) merge: refresh total candidate before active apply
533cca9ab merge: record active main stack as integrated into total candidate
e9db66991 (tag: hq/pre-active-main-history-merge-20260703-023952) merge: refresh total candidate onto current upstream main
ea9c92334 test(config): expect platform-native Hermes home
6b950d625 merge: bring total candidate onto latest upstream main
4607d1758 (tag: hq/pre-total-merge-candidate-20260702-214843) fix(desktop): expire stale clarify requests
2d43187f7 style(desktop): sort project store imports
51150b9c7 fix(clarify): resolve latest-origin Desktop integration
9cbdedeff fix(clarify): honor configured Desktop clarify timeout
ff795c13b fix(clarify): cover task report scope choices
8ba27ee18 style(desktop): satisfy restaged prompt submit lint
6f3029374 fix(clarify): reject mixed constrained responses
e8368cfc7 fix(clarify): enforce constrained selection responses
aaf04e0f5 fix(desktop): drop unused clarify import
61505325e fix(desktop): tolerate delayed clarify responses
76f87f8a9 merge: bring fork main candidate onto latest upstream main
2aa38fba5 (tag: hq/pre-update-main-20260702-144936, tag: hq/pre-total-merge-active-main-20260702-214843, tag: hq/pre-active-main-apply-20260703-024555) fix(clarify): cover task report scope choices
13eaee13c (hq/desktop-c1c2-restage-20260702-051612) style(desktop): satisfy restaged prompt submit lint
35aa9f395 fix(clarify): reject mixed constrained responses
d4ba30a27 fix(clarify): enforce constrained selection responses
edfa58b50 fix(desktop): drop unused clarify import
c11ada21e style(desktop): satisfy clarify lint spacing
fa522dae9 fix(clarify): reject empty required multi-select replies
e78c7f85b fix(clarify): enforce multi-select response bounds
45e737a75 fix(clarify): keep multi-select choice UX visible
2e7d55643 fix(desktop): tolerate slow backend readiness during startup
13c8445f4 fix(clarify): harden constrained-choice semantics
b3123618c feat(clarify): restore native multi-select prompts
3051d4f0d fix(desktop): explain prompt submit timeouts
6527445b1 fix(desktop): bound live terminal output events
307924c72 fix(desktop): distinguish lost gateway and bound tool result events
dba26fbac fix(desktop): tolerate delayed clarify responses
b4d1beb4e (tag: hq/pre-update-fork-main-20260702-144936, tag: hq/pre-total-merge-fork-main-20260702-214843, tag: hq/pre-total-fork-main-20260704-023904, tag: hq/pre-active-main-apply-fork-20260703-024555, tag: hq/fork-main-c1c2-source-20260701-214241, fork/main, fork/HEAD, hq/fork-main-total-20260701-053639) test: make new runtime guard tests Windows portable
74bb8c18f merge: refresh fork main total candidate with latest upstream main
46193b8b8 docs: fix desktop control routing MDX links
0e90dea3b test: normalize profile isolation path assertions on Windows
8ebe8c492 merge: refresh fork main total candidate with latest upstream main
7be1b00e8 test: keep Windows env for gateway config bridge subprocess
9a17d6b24 merge: refresh fork main total candidate with latest upstream main
7987663d0 fix: stabilize fork main total verification on Windows
2e42f2504 (tag: hq/fork-main-total-candidate-20260701_064446) merge: refresh fork main total candidate with latest upstream main
a12326a22 (tag: hq/fork-main-total-candidate-20260701_064244) feat(hq): add eval gate quality harness
efa2f10ae feat: add HQ harness evidence dashboard integration
4202eaf95 chore: normalize fork main total whitespace
0590ae4fe feat(computer-use): add safe local CDP browser backend
9f85bc2b6 feat(computer-use): add Windows UIA read-only enumeration
6e5145ef1 feat(computer-use): add browser input executor
2d76f95ea feat(computer-use): add safe input proposals
6a1f376ec feat: add safe desktop control routing status surface
65fda9573 chore: checkpoint HQ Hermes workspace updates
d5e24331d style(desktop): satisfy clarify lint spacing
cb0410001 fix(clarify): reject empty required multi-select replies
f2f675a46 fix(clarify): enforce multi-select response bounds
3f17156b7 fix(clarify): keep multi-select choice UX visible
33f588e87 fix(desktop): tolerate slow backend readiness during startup
336352acd fix(clarify): harden constrained-choice semantics
b7c7d78e7 feat(clarify): restore native multi-select prompts
2bd044507 fix(desktop): explain prompt submit timeouts
723cdf2ab feat(desktop): sort numbered sessions consistently
c1710d52e fix(gateway): harden Windows restart path
a73b6e648 feat(desktop): add file-backed session pins bridge
d7e347c1f merge: bring fork main candidate onto latest upstream main
18e8f4465 (tag: hq-pre-fork-main-total-20260701-053639) fix(desktop): bound live terminal output events
1492c1718 fix(desktop): distinguish lost gateway and bound tool result events
cf5cf315a fix(desktop): tolerate delayed clarify responses
```

## Incoming commits from origin/main

```text
none
```

## Recommended next action

1. **Do not run `hermes update` right now** unless a fresh `update --check` later shows behind > 0. Current check says already up to date.
2. **Preserve the active state** with a pre-update tag if any mutation is about to happen:

   ```bash
   cd "C:/Users/82109/AppData/Local/hermes/hermes-agent"
   git tag "hq/pre-hermes-update-$(date +%Y%m%d-%H%M%S)" HEAD
   ```

3. **Inspect the untracked release-staging file** and decide keep/delete/defer.
4. **Perform branch/stash triage in batches** rather than broad cleanup:
   - batch A: active carried `main` stack vs `origin/main`
   - batch B: clarify/Desktop UX branches
   - batch C: Gateway/cron/runtime safety branches
   - batch D: old stashes/autostashes
5. If a real upstream update appears later, use the existing active-main-preservation flow: tag first, check dirty state, close/stop Desktop only if the updater requires it, then run update and immediately re-run `hermes update --check`.

## Verification of this report

This report is Markdown-only and does not claim full repo test coverage. It is based on read-only git/Hermes commands plus focused clarify tests from the current session.
