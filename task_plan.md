# Task Plan

## Goal
Finish QQ NapCat group reply gating so group messages can wake Hermes when relevant without requiring repeated `@` mentions after a recent bot reply, while still preserving silent drop behavior for irrelevant follow-ups.

## Phases
- [completed] Inspect current QQ adapter, config bridge, and tests.
- [completed] Implement config and adapter changes.
- [completed] Run targeted verification and fix failures.
- [completed] Sync to remote host, update config, restart, and verify service behavior.

## Constraints
- Do not revert unrelated dirty worktree changes.
- Use `apply_patch` for manual file edits.
- Remote Python commands must activate `venv`.
- Remote verification is the reliable test environment.

## Notes
- Existing admin-only dangerous-operation controls are already deployed remotely.
- QQ bot name should be `马噶`.
- Dangerous operations must remain admin-only for QQ `179033731`.
- Final runtime configuration was revised after live endpoint validation:
  - Primary text model: `glm-5.1 @ https://wududu.edu.kg/v1`
  - Fallback chain: `api.888933 gpt-5.4` → `pay.kxaug gpt-5.4`
  - Vision remains on `pay.kxaug gpt-5.4` until a second endpoint is proven on real image traffic.
