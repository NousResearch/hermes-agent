# BO-008 — Kanban-native runtime gap evidence

Authority: Kanban `BO-008` (`t_842fcced`)
Repo: `chriskim12/hermes-agent`
Branch: `yuuka/bo-008-kanban-autopilot-selection`

## Routing verdict

- Verdict: Hermes direct
- Reason: deterministic `/autopilot` dry-run/admission and Kanban-native CLI surfaces can be verified locally with focused tests; no executor dispatch or gateway lifecycle mutation is required.
- Boundary: review=Chris; push/PR=allowed; merge=separate approval for BO-008 PR; gateway restart/reload=forbidden; secrets/env/BWS mutation=forbidden.

## Runtime state observed before this slice

`gateway/autopilot.py` already contains a Kanban-native dry-run selection path:

- `repo_policy.kanban.native_candidate_selection` / `repo_policy.kanban_candidate_selection` opt-in flag.
- Approved Kanban tenant allowlist checks.
- Work-state active lock fail-closed before queue selection.
- Kanban ready task selection before Linear fallback.
- Optional `kanban.fallback_to_linear` gate; without it, no eligible Kanban task blocks instead of silently falling back to Linear.
- Read-only dry-run semantics: no work_state write, no executor spawn, no Linear mutation, no Kanban task claim.

This means the high-value runtime gap was narrower than originally estimated: the selection path existed on merged `main`, but the native card creation CLI still had a wrapper ergonomics bug.

## Implemented in this slice

`hermes kanban native-create` now accepts both:

- `--profile`
- `--worker-profile`

The alias exists because the top-level `hermes` wrapper can consume global `--profile` before the `kanban native-create` parser receives it. `--worker-profile` is an unambiguous subcommand-level profile argument for Kanban-native admission.

## Verification

Focused checks run in the BO-008 worktree:

```text
python -m py_compile hermes_cli/kanban.py gateway/autopilot.py tests/test_kanban_native_admission.py tests/gateway/test_autopilot_command.py
pytest -q tests/test_kanban_native_admission.py
# 15 passed in 3.17s
pytest -q tests/gateway/test_autopilot_command.py -k 'kanban_native_candidate_selection or kanban_payload_dry_run'
# 4 passed in 3.10s
git diff --check
```

Additional manual note: invoking the installed `hermes` executable from the live environment still reflects the currently installed/live main code until the BO-008 patch is merged and the executable/runtime is refreshed. No gateway restart/reload was performed for this slice.
