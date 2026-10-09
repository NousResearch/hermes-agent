# Phase 2 Follow-up Repair — Systemd Authority Ordering

**Date:** 2026-10-04  
**Repository:** `hermes-agent`  
**PR:** NousResearch/hermes-agent#125028 — Phase 2, Gateway Control and Topology  
**Implementation branch:** `split/phase2-gateway`  
**Current head:** `a91182e7053ee063a8f937b3784fd48601e6fa1a`  
**Phase 1 behavioral parent:** `db6d2da391a132bddc24018a83ad7c79ba660881`  
**Preceding repair spec:** `docs/refactor/phase2-review-fixes-2026-10-04.md`

## 1. Objective

Close the one remaining Phase 2 review regression without reopening the completed Phase 2 repair work.

The previous Phase 2 review repairs are **not** being reworked:

1. systemd manager `Environment=` framing is repaired and covered;
2. systemd environment writer/reader round-trip semantics are repaired and covered;
3. setup-driven legacy-unit preflight is restored and covered.

This follow-up exists because repair (3) exposed one additional ordering bug in the **direct CLI system-service install path**.

The repair must restore the Phase 1 fail-fast authority invariant:

```
system-scope install requested
    -> establish root authority
    -> only then perform any legacy-unit cleanup
    -> install the new system service
```

A caller that lacks authority for the requested system-scope install must not mutate an existing service configuration before the command is rejected.

---

## 2. Current defect

### 2.1 Affected path

Current direct CLI flow in:

`hermes_cli/gateway.py::_install_systemd_from_cli(...)`

is:

```python
_remove_legacy_systemd_units_before_install(
    non_interactive=non_interactive
)
_systemd_lifecycle.install(
    force=force,
    system=system,
    run_as_user=run_as_user,
    enable_on_startup=start_on_login,
)
```

The lifecycle owner correctly performs:

```python
if system:
    systemd_identity.require_root("install")
```

but that check occurs **after** the CLI-owned legacy preflight.

### 2.2 Failure case

For:

```
non-root caller
+ hermes gateway install --system
+ legacy user-scope hermes.service present
```

the current ordering permits:

```
legacy preflight
    -> remove_legacy_hermes_units(interactive=False)
    -> gateway.systemd_legacy.remove_units()
    -> stop/disable/unlink the legacy user unit

then

systemd_lifecycle.install(system=True)
    -> systemd_identity.require_root("install")
    -> SystemScopeRequiresRootError
```

The requested system install fails, but the pre-existing user service may already have been removed.

This is a real behavior regression relative to the Phase 1 parent.

### 2.3 Why the previous repair tests missed it

`tests/hermes_cli/test_phase2_systemd_review_regressions.py` currently covers the setup-wizard user/system legacy-preflight matrix.

The setup wizard's system branch already checks root before running its legacy preflight, so those tests pass.

The missing cell is:

```
direct CLI
+ system=True
+ non-root
+ legacy unit present
```

The direct CLI path therefore needs its own authority-ordering invariant.

---

## 3. Scope

### 3.1 In scope

Repair only the ordering of the direct systemd CLI install path so that system-scope authority is established before any legacy-unit mutation.

Add regression coverage for:

- non-root direct `--system` install;
- root direct `--system` install as a positive control;
- direct user-scope install as a positive control;
- preservation of preflight-before-install ordering for authorized calls.

### 3.2 Explicit non-goals

Do **not**:

- redesign systemd installation;
- move root-authority ownership back into `hermes_cli`;
- remove the defensive root check already present in `gateway.systemd_lifecycle.install()`;
- change setup-wizard scope selection;
- change legacy-unit discovery or removal behavior;
- change interactive/non-interactive consent semantics;
- change user-scope install behavior;
- change launchd or Windows paths;
- revisit the already repaired systemd environment parsing/codec work;
- introduce a compatibility shim;
- broaden this into a general Phase 2 cleanup.

The current placement of `_remove_legacy_systemd_units_before_install()` in `hermes_cli.gateway` is an architectural follow-up from the earlier spec, but relocating it is **not required to close this correctness regression**. Avoid coupling that cleanup to this repair unless necessary for correctness.

---

## 4. Ownership invariant

Authority belongs to the canonical Gateway identity owner:

`gateway/systemd_identity.py::require_root(action)`

The CLI owns interaction/policy about legacy services.

The lifecycle owner owns installation mechanics and retains its own authority check as defense in depth.

The intended direct-CLI sequence is:

```
hermes_cli.gateway._install_systemd_from_cli
    |
    |-- if system:
    |      gateway.systemd_identity.require_root("install")
    |
    |-- resolve start-now/start-on-login policy
    |
    |-- CLI legacy-unit preflight
    |
    |-- gateway.systemd_lifecycle.install(...)
    |      |
    |      '-- defensive require_root("install") remains
    |
    '-- optional gateway.systemd_lifecycle.start(...)
```

The caller must use the canonical authority function. Do not duplicate the authority rule with a new direct `os.geteuid()` check in the CLI.

---

## 5. Required implementation

### R2.4.1 — Gate direct system installs before legacy cleanup

In:

`hermes_cli/gateway.py::_install_systemd_from_cli(...)`

establish system-scope authority before:

```python
_remove_legacy_systemd_units_before_install(...)
```

Preferred implementation:

```python
if system:
    _systemd_identity.require_root("install")
```

Place the authority check early enough that no service mutation can occur before it.

The exact placement may remain after harmless local flag calculation/prompt policy if preserving Phase 1 interaction ordering is important, but it **must be before the legacy preflight** and every other mutating systemd operation.

Do not remove this existing lifecycle check:

`gateway/systemd_lifecycle.py::install()`

```python
if system:
    systemd_identity.require_root("install")
```

That owner-level check protects all non-CLI callers and is not redundant architectural debt.

### R2.4.2 — Do not solve this by suppressing cleanup

The repair is **not**:

- skipping legacy cleanup for all system installs;
- filtering user-scope legacy units out of system installs;
- catching `SystemScopeRequiresRootError` after cleanup;
- attempting to restore deleted units after failure.

Authority must precede mutation.

---

## 6. Regression tests

Extend:

`tests/hermes_cli/test_phase2_systemd_review_regressions.py`

Keep the tests behavioral and small.

### 6.1 Required negative regression

Add one test for:

```
direct CLI
system=True
non-root
legacy units present
```

The test must prove:

1. the canonical root authority gate is reached;
2. the call raises `gateway.systemd_identity.SystemScopeRequiresRootError`;
3. legacy preflight/removal is **never** invoked;
4. lifecycle install is **never** invoked;
5. lifecycle start is **never** invoked.

Preferred fixture shape:

- call `gateway_cli._install_systemd_from_cli(..., system=True, ...)`;
- provide explicit `start_now` / `start_on_login` values so the test does not depend on TTY prompting;
- monkeypatch the canonical `systemd_identity.os.geteuid` to a non-root UID, using `raising=False` for Windows collection safety;
- make the legacy-preflight seam fail the test immediately if reached;
- make lifecycle install/start fail immediately if reached.

The test must be red on `a91182e705` and green after the repair.

### 6.2 Required positive-control matrix

Add one parameterized test covering authorized direct installs:

| Path | `system` | authority | expected |
| --- | ---: | --- | --- |
| direct CLI | false | user scope | preflight -> install |
| direct CLI | true | root | authority -> preflight -> install |

For both rows:

- legacy preflight must occur;
- preflight must occur before lifecycle install;
- lifecycle install receives the correct `system` value;
- no new difference in start-now/start-on-login policy is introduced.

For the root/system row, use the canonical authority path rather than bypassing it entirely.

If an existing direct-CLI regression already proves one positive row by the time implementation begins, extend it rather than duplicating coverage.

### 6.3 Existing setup tests remain unchanged

The existing parameterized:

`test_setup_preserves_legacy_service_preflight`

continues to prove setup-wizard behavior.

Do not rewrite it to cover the direct CLI path; these are distinct orchestration entry points and the defect exists precisely because only one of them had its own authority gate.

---

## 7. Verification sequence

### Step 0 — baseline

Before editing:

```
git status
git rev-parse HEAD
```

Required baseline:

```
a91182e7053ee063a8f937b3784fd48601e6fa1a
```

The current Phase 2 worktree should be clean.

### Step 1 — reproduce the missing invariant

Add the negative regression first and run only that test against the current implementation.

It must fail because the legacy preflight is reached before root rejection.

Do not treat a test that passes on the current head as sufficient proof of this defect.

### Step 2 — implement the authority gate

Add the canonical `_systemd_identity.require_root("install")` gate before legacy cleanup in the direct CLI systemd install path.

Do not change unrelated install logic.

### Step 3 — focused regression file

Run:

```bash
scripts/run_tests.sh tests/hermes_cli/test_phase2_systemd_review_regressions.py
```

Acceptance:

- new negative authority-ordering regression passes;
- new positive direct-CLI controls pass;
- existing manager-framing tests pass;
- existing codec tests pass;
- existing setup legacy-preflight tests pass.

### Step 4 — install/lifecycle regression slice

Run the existing install/service coverage that exercises the surrounding boundary:

```bash
scripts/run_tests.sh tests/hermes_cli/test_gateway_service.py
scripts/run_tests.sh tests/hermes_cli/test_gateway.py
scripts/run_tests.sh tests/hermes_cli/test_ensure_gateway_service.py
```

If a named file does not exist on the current branch, substitute the current file containing the same service-install contract and record the substitution.

### Step 5 — ownership/boundary gate

Run:

```bash
scripts/run_tests.sh tests/gateway/test_migration_cli_boundary.py
```

The repair must not introduce a reverse dependency from `gateway/` into CLI policy.

### Step 6 — static checks

Run:

```bash
python -m ruff check hermes_cli/gateway.py tests/hermes_cli/test_phase2_systemd_review_regressions.py
git diff --check
```

### Step 7 — exact-head evidence

After the final repair commit:

- record the exact SHA;
- push the Phase 2 branch;
- reply to the unresolved review thread with the behavioral explanation and test receipts;
- request re-review against that exact SHA;
- resolve the thread only after the implementation and regression are present on the PR head.

Hosted CI remains a separate verification gate. Do not represent zero-job / `action_required` workflows as passing CI.

---

## 8. Acceptance gates

### Functional

- [ ] Non-root direct `hermes gateway install --system` fails before legacy-unit preflight.
- [ ] A rejected system install cannot stop, disable, unlink, or otherwise mutate a legacy user service.
- [ ] A rejected system install does not call lifecycle install.
- [ ] A rejected system install does not call lifecycle start.
- [ ] Root direct system installs still perform the legacy preflight.
- [ ] Authorized system installs still perform preflight before lifecycle install.
- [ ] User-scope direct installs still perform the legacy preflight.
- [ ] Existing interactive/non-interactive legacy consent semantics are unchanged.
- [ ] Setup-wizard user/system preflight behavior remains green.

### Architectural

- [ ] Root authority remains owned by `gateway.systemd_identity.require_root()`.
- [ ] CLI code calls the canonical owner rather than duplicating the EUID rule.
- [ ] `gateway.systemd_lifecycle.install()` retains its defensive authority gate.
- [ ] Legacy warning/prompt policy remains CLI-owned.
- [ ] No compatibility shim is added.
- [ ] No launchd or Windows behavior changes.
- [ ] No unrelated Phase 2 ownership surfaces are modified.

### Verification

- [ ] The new negative regression fails on `a91182e705`.
- [ ] Focused Phase 2 systemd review regression file passes on repaired head.
- [ ] Surrounding gateway install/service tests pass.
- [ ] Phase 2 migration/ownership boundary test passes.
- [ ] Ruff passes on changed Python files.
- [ ] `git diff --check` passes.
- [ ] Exact repair SHA is posted to PR #125028.
- [ ] The open reviewer thread at `hermes_cli/gateway.py:2652` is answered with exact-head evidence.

---

## 9. Commit shape

This is one defect with one invariant. Prefer one commit:

```
fix(gateway): gate system install before legacy cleanup
```

Expected production delta should be very small: one authority gate at the direct CLI boundary, plus focused tests.

Do not fold unrelated refactoring or helper relocation into this commit.

---

## 10. Completion definition

This repair is complete when the direct CLI path has the same destructive-action ordering guarantee as the Phase 1 parent:

```
unauthorized system install
    -> reject
    -> zero service mutation
```

and authorized paths retain:

```
authorized install
    -> legacy preflight
    -> lifecycle install
    -> optional start
```

At that point the newly reported source-level Phase 2 regression is closed. Any remaining inability to merge because hosted workflows do not execute is a **verification infrastructure gate**, not additional Phase 2 implementation work.
