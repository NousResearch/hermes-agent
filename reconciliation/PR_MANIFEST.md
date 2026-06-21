# PR MANIFEST — live snapshot (independent verification)

Generated: 2026-06-21T13:09:23.300664Z
Total open PRs on fork arminanton: 39
All state=OPEN: True
Draft: 31  Ready-for-review: 8

| PR | state | head SHA | branch |
|----|-------|----------|--------|
| #48024 | review | `c5346ea34f8e` | feat/expose-reasoning-api-server |
| #48057 | review | `a4589b1650a2` | feat/drop-empty-name-tools |
| #48065 | review | `ea7e64f4606f` | fix/context-engine-tool-schema-doublewrap |
| #48069 | review | `4a1fbe9e1651` | fix/mcp-keepalive-inflight-race |
| #48101 | review | `af2016d1ef93` | feat/per-model-system-prompt-prelude |
| #49184 | review | `d684ef80833d` | feat/copilot-claude-v1messages-routing |
| #49449 | review | `cd27defe2bea` | feat/copilot-codex-true-limits-pr |
| #49644 | review | `9ea0cab9a06c` | feat/reasoning-max-effort |
| #49915 | draft | `e33e4f781fd8` | feat/tui-ctrlc-interrupt-vscode |
| #49916 | draft | `1da02f7484cd` | fix/tui-yolo-badge-approval-mode |
| #49917 | draft | `6bc37d8f4162` | fix/tui-notify-autodispatch-gate |
| #50021 | draft | `80ce952014d0` | feat/tool-timing-sidecar |
| #50022 | draft | `6f6ab5839962` | feat/model-router-proxy-tool |
| #50031 | draft | `1bdae0a130a0` | feat/copilot-auto-mode-router |
| #50032 | draft | `122087da90fb` | feat/source-accelerator |
| #50033 | draft | `d1cee7c83b29` | feat/gemini-cli-user-agent |
| #50038 | draft | `cf14b57314c2` | feat/codex-cli-identity |
| #50039 | draft | `5b26c1fb24f3` | feat/agy-cli-provider |
| #50040 | draft | `25b495a06a72` | feat/delegate-task-persona |
| #50041 | draft | `f3288ad358d8` | fix/doctor-optional-integrations |
| #50042 | draft | `a54e706a7fb6` | feat/file-tools-read-guards |
| #50045 | draft | `5a12f857a41b` | feat/skills-hub-tool-guard |
| #50046 | draft | `0d226f1eea73` | feat/stable-tag-update-check |
| #50047 | draft | `b3695d09af5a` | fix/gateway-liveness-and-root-guard |
| #50048 | draft | `f41f87b317cf` | feat/send-plain-text-directive |
| #50049 | draft | `e4e34884d055` | fix/subdir-hints-expanduser-guard |
| #50053 | draft | `d0ffb661c312` | feat/context-engine-grounding-hooks |
| #50054 | draft | `352b1318456d` | feat/plugin-register-command-override |
| #50055 | draft | `35da2881600a` | fix/copilot-assistant-prefill-trailing-user |
| #50056 | draft | `c219bded8eff` | feat/sqlite-driver-selection |
| #50064 | draft | `d8e353f3359c` | feat/copilot-cli-identity-claude-vision |
| #50066 | draft | `37a76f469d0e` | tests/bedrock-region-sessions-pagination |
| #50068 | draft | `123efe8bef36` | feat/tui-autopilot-yolo-status-badges |
| #50073 | draft | `a12d4aebd984` | feat/compression-oversized-message-offload |
| #50078 | draft | `94ef105b0fb9` | tests/orphan-test-catchup-and-discord-max |
| #50080 | draft | `e5e4a8b1367f` | tests/context-engine-unwrap-and-compression-runtime |
| #50086 | draft | `c7e2620c904e` | fix/web-server-profiles-sessions-dedupe |
| #50111 | draft | `89566e64a52d` | deferred/residual-lines-on-v0.17.0 |
| #50146 | draft | `b674df63082b` | feat/gateway-media-image-classification |

## CI status note

Fork-draft PRs do not trigger the repo's gated CI (Tests / Lint / etc. run on
`ready_for_review` + maintainer approval), so 0 check-runs appear on draft heads.
As a CI proxy, #50046's changed file was verified locally:
- `ruff check hermes_cli/config.py` → All checks passed
- `pytest tests/hermes_cli/test_stable_update.py test_update_check.py test_banner_git_state.py` → 24 passed
- `ast.parse(config.py)` → OK

When a maintainer marks these ready-for-review, the repo's full CI (Tests, Lint
ruff+ty, Regression, Supply Chain Audit) runs against each.

## Deferred categories — out-of-scope, tracked in this PR (#50111)

Per the campaign's documented rulings, these categories are intentionally NOT part
of the public "all ./src changes in separate PRs" set and are tracked here:
- `private-overlay/` + `private-overlay-phaseh/` — v2026.6.5 phase-h update-merge machinery, NOT contributable
- `private-feature-mixed/` — agy-cli/cmx/auto-router/review-path tokens, would LEAK if public
- `copilot-limits/` — account-specific caps, keep-deferred/generalize-later
- `cmx/` — single CMX-implementation PR, never piecemeal (travels with that PR when opened)

These are listed file-by-file with rule ids in the category README tables on this branch.
