# Phase 2 Review Fixes — Systemd Behavior Preservation

**Date:** 2026-10-04  
**Repository:** `hermes-agent`  
**PR:** NousResearch/hermes-agent#125028 — Phase 2, Gateway Control and Topology  
**Implementation branch:** `split/phase2-gateway`  
**Reviewed baseline:** `4eb21abf1393528ae1ec94552e4f8985a3286fdf`  
**Phase 1 parent:** `db6d2da391a132bddc24018a83ad7c79ba660881`

## 1. Objective

Close the three remaining P2 correctness regressions reported against Phase 2 without changing the Phase 2 ownership model.

Phase 2 is an **ownership extraction**. These repairs must restore the behavior of the Phase 1 parent at the new canonical ownership boundaries; they must not add compatibility shims, reintroduce service implementation into `hermes_cli.gateway`, or redesign service setup.

The three defects are independent:

1. systemd manager environment framing is parsed incorrectly when `HERMES_HOME` is the first assignment;
2. the extracted systemd unit environment reader no longer exactly reverses the unit writer's escaping;
3. the setup wizard no longer performs the legacy-systemd-unit preflight/removal that the parent performed before installation.

All three are required for Phase 2 review closure.

## 2. Baseline and dirty-worktree rule

The authoritative behavioral comparison is:

```
db6d2da391a132bddc24018a83ad7c79ba660881  Phase 1 parent
        ↓
4eb21abf1393528ae1ec94552e4f8985a3286fdf  reviewed Phase 2 head
```

At the time this specification was written, the Phase 2 worktree contained uncommitted edits in the same repair surfaces. Those edits are **not acceptance evidence** and are not part of this specification's baseline.

An implementer must:

- preserve unrelated concurrent work;
- compare behavior against committed HEAD `4eb21abf` and the Phase 1 parent, not against whatever happens to be dirty in the worktree;
- never mark a repair complete solely because an uncommitted candidate implementation exists;
- prove each repaired invariant with tests that fail on `4eb21abf` and pass on the repaired head.

## 3. Scope and ownership constraints

### 3.1 Gateway-owned behavior

These remain canonical Gateway owners:

- `gateway/systemd_runtime.py`
  - manager interaction;
  - `systemctl` output interpretation;
  - adoption of the installed service's runtime home.
- `gateway/service_identity.py`
  - systemd unit identity reads;
  - decoding stored `Environment=` values.
- `gateway/systemd_unit_render.py`
  - encoding generated `Environment=` lines.
- `gateway/systemd_lifecycle.py`
  - install/start/stop/uninstall mechanics;
  - optional execution of an already-decided `remove_legacy=True` policy.

Do not route these responsibilities back through `hermes_cli.gateway`.

### 3.2 CLI-owned interaction/policy

The decision to warn, prompt, and interpret the user's answer about legacy services remains CLI/application policy.

Use the existing topical orchestration module:

- `hermes_cli/gateway_setup_service.py`

for shared legacy-install preflight behavior used by both:

- the explicit `hermes gateway install` route; and
- `install_linux_gateway_from_setup()` / setup-wizard service installation.

Do **not** add a new permanent helper to the already-large `hermes_cli.gateway` facade unless no existing sibling can own the policy. The facade may late-import and call the sibling.

### 3.3 Explicit non-goals

Do not:

- change systemd service naming;
- change user-vs-system scope selection;
- change root requirements;
- change linger behavior;
- change service generation beyond restoring codec identity;
- move prompting into `gateway/systemd_lifecycle.py`;
- make `systemd_lifecycle.install()` prompt;
- restore a retired CLI implementation path or compatibility alias;
- broaden parsing into a general systemd parser;
- change non-systemd backends;
- fix unrelated inherited test failures.

---

# 4. Repair R2.1 — systemd manager Environment framing

## 4.1 Defect

Owner:

`gateway/systemd_runtime.py::sync_home_from_unit(system)`

The extracted implementation requests:

```
systemctl show <service> --no-pager --property Environment
```

without `--value`.

A successful result can therefore be framed as:

```
Environment=HERMES_HOME=/srv/hermes
```

or:

```
Environment=PATH=/usr/bin HERMES_HOME=/srv/hermes
```

The reviewed Phase 2 code splits the complete response into tokens and only accepts tokens beginning with `HERMES_HOME=`.

That means:

- `Environment=HERMES_HOME=/srv/hermes` → **missed**;
- `Environment=PATH=/usr/bin HERMES_HOME=/srv/hermes` → accidentally works.

The selected service's identity therefore depends on assignment order.

## 4.2 Required behavior

`sync_home_from_unit(system=True)` must:

1. prefer the explicit `HERMES_HOME` already stored in the unit file through
   `service_identity.hermes_home_pinned_by_unit()`;
2. only query the manager when the unit itself does not provide the home;
3. remove the outer systemctl property framing before interpreting assignments;
4. find `HERMES_HOME` regardless of whether it is the first or a later assignment;
5. replace the caller's current `HERMES_HOME` only when a non-empty selected-service home is found;
6. retain the existing no-op behavior for `system=False`.

The fix should be narrow. It does not need a complete shell/systemd environment parser because the parent contract being restored is specifically the manager's `Environment` property output.

## 4.3 Preferred implementation shape

Keep manager-output normalization local to `gateway.systemd_runtime`.

Conceptually:

```
manager_value = stdout.strip()
if manager_value begins with the requested property wrapper:
    manager_value = remove exactly that outer wrapper

for assignment in manager_value.split():
    if assignment begins HERMES_HOME=:
        adopt its value
```

Do not special-case token order.

If the implementation chooses to request `--value` instead, it must prove equivalent behavior against both framed and value-only fixtures, and it must not introduce an unnecessary behavioral difference from the parent. A narrow framing strip is the lower-risk repair.

## 4.4 Invariant tests

Add no more than two behavioral test functions for this defect.

Minimum parameterized coverage:

| Manager stdout | Expected |
| --- | --- |
| `Environment=HERMES_HOME=/srv/hermes\n` | adopts `/srv/hermes` |
| `Environment=PATH=/usr/bin HERMES_HOME=/srv/hermes\n` | adopts `/srv/hermes` |

Also retain one control in the same test:

- when the unit file already pins `HERMES_HOME`, the manager value does not override it.

The test must exercise `gateway.systemd_runtime.sync_home_from_unit()`, not a new parsing helper in isolation.

---

# 5. Repair R2.2 — systemd environment codec identity

## 5.1 Defect

Writer:

`gateway/systemd_unit_render.py::systemd_env_line(name, value)`

Reader:

`gateway/service_identity.py::unit_environment_value(unit_path, name)`

The writer escapes:

- `\` → `\\`;
- `"` → `\"`;
- `%` → `%%`.

The moved reader changed its escape-match literals so ordinary writer output containing backslashes or quotes is no longer decoded to the original value.

This violates the essential ownership contract:

```
unit_environment_value(systemd_env_line(name, value), name) == value
```

for every value the writer supports.

This is observable in `LD_LIBRARY_PATH`: when the invoking shell does not supply one, `ld_library_path_line()` reads the installed unit and writes that value back during regeneration. A lossy reader therefore changes an otherwise-preserved unit on the next generation.

## 5.2 Required behavior

For values generated by `systemd_env_line()`, the reader must be its exact inverse for the supported escaping surface.

At minimum preserve:

- plain values;
- percent signs;
- literal backslashes;
- literal double quotes;
- mixtures of those characters.

The contract applies equally to:

- `HERMES_HOME`;
- `LD_LIBRARY_PATH`.

Do not add an independent second codec. The writer and reader remain a pair with one explicit round-trip contract.

## 5.3 Required decoding order

The implementation must restore the parent semantics, not simply make the current tests green.

The quoted `Environment=` body must decode the writer's escape forms so that:

```
encode(value) -> stored bytes -> decode(...) -> value
```

holds.

Pay attention to replacement order: quote escapes and doubled backslashes overlap. Use the exact parent behavior or another implementation proven equivalent by the round-trip matrix.

Percent decoding remains required because systemd unit generation doubles `%`.

## 5.4 Invariant tests

Use one parameterized codec test rather than separate snapshot tests for every string.

Required values should include at least:

```
"/plain/path"
"/opt/pct%dir/lib"
r"/opt/a\b/lib"
'/opt/cu"da/lib'
r'/opt/a\b/"quoted"/pct%dir'
```

Run the same values through both keys:

- `HERMES_HOME`;
- `LD_LIBRARY_PATH`.

The test should:

1. write `systemd_env_line(name, value)` to a real temporary unit file;
2. call `service_identity.unit_environment_value(unit, name)`;
3. assert exact equality with the original value.

A second behavioral test must pin regeneration:

1. generate/install a unit with an `LD_LIBRARY_PATH` containing at least a backslash or quote;
2. remove `LD_LIBRARY_PATH` from the caller environment;
3. regenerate through the production fallback;
4. assert the preserved logical value is unchanged and repeated regeneration is stable.

Prefer semantic equality / idempotence over freezing an entire unit-file snapshot.

---

# 6. Repair R2.3 — restore legacy-unit preflight in setup

## 6.1 Defect

Parent behavior performed a legacy-systemd-service preflight before installing the new gateway service.

Phase 2 correctly moved installation mechanics into:

`gateway/systemd_lifecycle.py::install(..., remove_legacy=False)`

and the explicit CLI install route retained the interactive legacy warning/removal decision.

However:

`hermes_cli.gateway::install_linux_gateway_from_setup()`

now calls the lifecycle owner without first performing that policy decision.

For both user and system scope, a setup-driven install can therefore:

1. leave a recognized legacy `hermes.service` in place;
2. install/enable the new gateway unit;
3. report setup success;
4. potentially start the new service while the old supervisor still exists.

The direct CLI route is a passing control; the setup route lost the preflight during extraction.

## 6.2 Ownership rule

Separate **policy** from **mechanism**:

- `hermes_cli/gateway_setup_service.py` owns the user-facing legacy preflight:
  - detect legacy units through the canonical Gateway owner;
  - render warning;
  - ask for consent when interactive;
  - interpret explicit decline;
  - request removal when policy says remove.
- `gateway/systemd_lifecycle.install()` owns actual install mechanics and may accept an already-decided `remove_legacy` flag.

Do not make the lifecycle owner interactive.

Do not duplicate the preflight separately in setup and direct CLI routes.

## 6.3 Shared preflight contract

Introduce or deepen one CLI-owned helper in `gateway_setup_service.py` with a small interface, for example conceptually:

```
preflight_legacy_systemd_install(*, non_interactive: bool) -> bool | None
```

The exact name/return type may differ; behavior matters.

Required behavior:

### No legacy unit

- no warning;
- no prompt;
- no removal;
- installation proceeds.

### Legacy unit + interactive affirmative consent

- render the existing legacy warning;
- prompt using the existing wording/default;
- remove legacy units through the existing canonical path;
- installation proceeds only after cleanup call completes.

### Legacy unit + interactive explicit decline

Preserve parent behavior:

- render warning;
- prompt;
- do not remove legacy units;
- installation still proceeds.

The repair must not silently reinterpret "decline removal" as "cancel installation" unless the Phase 1 parent did so.

### Legacy unit + explicit non-interactive CLI install

Preserve the existing explicit CLI behavior:

- no prompt;
- removal is performed according to the current non-interactive install policy;
- installation proceeds.

The setup wizard itself is interactive; it should use the same helper with interactive policy rather than bypassing the check.

## 6.4 Call sites

The shared preflight must cover:

1. explicit `hermes gateway install` systemd path;
2. setup wizard → user systemd install;
3. setup wizard → system systemd install.

The preflight must occur **before** `systemd_lifecycle.install()`.

Do not move it after unit write/enable.

## 6.5 Invariant tests

Use parameterization to avoid a sprawling fixture matrix.

At minimum prove:

| Entry path | Scope | Legacy | Consent | Remove called | New install |
| --- | --- | ---: | ---: | ---: | ---: |
| setup | user | yes | yes | yes | yes |
| setup | system | yes | yes | yes | yes |
| setup | user | yes | no | no | yes |
| setup | system | yes | no | no | yes |
| setup | either | no | n/a | no | yes |
| direct CLI | either | yes | noninteractive | yes | yes |

Also retain the setup-scope skip control:

- if the user chooses no service scope, neither removal nor installation occurs.

The assertions must observe the canonical removal/install calls, not merely whether a prompt function was invoked.

---

# 7. Test placement

Prefer one focused file for the three review regressions:

`tests/hermes_cli/test_phase2_systemd_review_regressions.py`

if that file already exists in concurrent work, reconcile rather than overwrite it.

However, ownership-specific tests may remain in existing files when that gives a clearer interface test:

- systemd codec/regeneration coverage may extend
  `tests/hermes_cli/test_gateway_service.py`;
- setup orchestration coverage may extend the existing setup/service tests.

The repository rule is **1–2 invariant test functions per fix**, not one test for every example. Use parameterization.

Do not add change-detector tests that merely assert a helper exists or an import points at a certain module. The test surface is behavior.

---

# 8. Execution sequence

## Step 0 — establish the repair baseline

Before editing:

1. record `git status`;
2. preserve or separately identify concurrent uncommitted work;
3. confirm `HEAD == 4eb21abf...` or explicitly document a newer review baseline;
4. run the focused reproduction tests against the reviewed head if the worktree can be cleanly reconstructed.

Do not reset or discard concurrent edits without ownership confirmation.

## Step 1 — R2.1 manager framing

Repair `sync_home_from_unit()`.

Run only its focused test file first.

Acceptance:

- first-position and later-position `HERMES_HOME` both adopt the same selected home;
- unit-file pin still wins;
- no new CLI dependency from `gateway/`.

## Step 2 — R2.2 codec identity

Repair `unit_environment_value()` as the inverse of `systemd_env_line()`.

Acceptance:

- full parameterized round-trip passes for both key names;
- `LD_LIBRARY_PATH` fallback/regeneration is stable;
- existing service-generation tests remain green.

## Step 3 — R2.3 shared legacy preflight

Put shared interaction policy in `hermes_cli/gateway_setup_service.py`.

Update the direct install and setup call sites to use that one policy path.

Acceptance:

- both setup scopes perform the preflight;
- affirmative consent removes;
- explicit decline preserves the legacy unit but preserves parent installation behavior;
- direct noninteractive install keeps its current behavior;
- lifecycle owner remains noninteractive.

## Step 4 — focused regression suite

Use the repository runner, **not bare pytest**:

```bash
scripts/run_tests.sh tests/hermes_cli/test_phase2_systemd_review_regressions.py
scripts/run_tests.sh tests/hermes_cli/test_gateway_service.py
scripts/run_tests.sh tests/hermes_cli/test_gateway.py
scripts/run_tests.sh tests/hermes_cli/test_ensure_gateway_service.py
```

If tests are placed elsewhere, substitute the actual files while retaining coverage of all four areas above.

## Step 5 — Phase 2 ownership gates

Run the Phase 2 structural/boundary coverage, including:

```bash
scripts/run_tests.sh tests/gateway/test_migration_cli_boundary.py
```

and any existing Phase 2 ownership/compatibility gate files touched by the branch.

Then:

```bash
python -m ruff check <changed-python-files>
git diff --check
```

The repair must not create a reverse dependency from `gateway/` to `hermes_cli.gateway*`.

## Step 6 — broader Phase 2 regression slice

Re-run the previously reported Phase 2 focused sweep that covers:

- gateway lifecycle/service behavior;
- service identity and refresh;
- setup/service installation;
- Desktop ticket/bootstrap bridge;
- profile ownership compatibility.

Inherited failures must be compared against the unchanged Phase 1 parent. Do not claim a Phase 2 regression merely because both heads fail the same test.

## Step 7 — exact-head review evidence

Once the repair commit is final:

1. record the exact SHA;
2. post the focused test receipts;
3. request re-review specifically against that SHA;
4. do not claim hosted CI green until the hosted jobs actually execute.

If hosted workflows still return `action_required` / zero jobs, state that as an open verification gate rather than a test failure.

---

# 9. Acceptance gates

Phase 2 review fixes are complete only when all of the following are true.

## Functional

- [ ] Manager `Environment=` property framing cannot hide first-position `HERMES_HOME`.
- [ ] Manager assignment order does not affect selected-home adoption.
- [ ] Unit-file `HERMES_HOME` remains authoritative over manager fallback.
- [ ] `unit_environment_value(systemd_env_line(...))` round-trips supported values exactly.
- [ ] Backslashes, quotes, and percent signs are all covered.
- [ ] Installed `LD_LIBRARY_PATH` survives regeneration when the caller lacks the variable.
- [ ] Setup user-scope install performs legacy preflight.
- [ ] Setup system-scope install performs legacy preflight.
- [ ] Affirmative legacy-removal consent removes before installation.
- [ ] Explicit decline preserves the legacy unit and preserves the parent's install behavior.
- [ ] Direct CLI install retains its existing noninteractive behavior.

## Architectural

- [ ] Systemd runtime parsing remains in `gateway/systemd_runtime.py`.
- [ ] Unit identity decoding remains in `gateway/service_identity.py`.
- [ ] Systemd installation mechanics remain in `gateway/systemd_lifecycle.py`.
- [ ] Interactive legacy policy is CLI-owned.
- [ ] Shared legacy policy lives in a topical CLI sibling, preferably `gateway_setup_service.py`, not as new facade growth.
- [ ] No compatibility shim is added.
- [ ] No `gateway/ -> hermes_cli.gateway*` reverse dependency is introduced.
- [ ] No launchd/Windows behavior changes.

## Verification

- [ ] Every new invariant test is red on reviewed head `4eb21abf` for the defect it proves.
- [ ] Focused systemd/setup/service tests pass on repaired head.
- [ ] Phase 2 boundary test passes.
- [ ] Ruff passes on changed Python files.
- [ ] `git diff --check` passes.
- [ ] Exact repaired SHA is posted for re-review.
- [ ] Hosted validation status is reported truthfully and separately from local verification.

---

# 10. Commit shape

Prefer **one review-repair commit** once all three defects are proven:

```
fix(gateway): close Phase 2 systemd review regressions
```

Rationale: these are three behavior-preservation corrections to the same reviewed Phase 2 extraction head, and the reviewer needs one exact SHA on which all three findings are simultaneously closed.

If concurrent work makes an atomic commit unsafe, three narrowly ordered commits are acceptable:

1. `fix(gateway): parse systemd manager environment framing`
2. `fix(gateway): restore systemd environment codec identity`
3. `fix(gateway): restore setup legacy-service preflight`

In either case, the final review request must point to the final combined SHA and rerun the complete focused gate there.

---

# 11. Completion definition

This repair is not complete when the code "looks fixed."

It is complete when the final Phase 2 head demonstrates all three restored parent contracts through the new owners:

```
manager output
    -> gateway.systemd_runtime
    -> correct selected HERMES_HOME

systemd_env_line(value)
    -> unit file
    -> gateway.service_identity
    -> identical value

setup / explicit CLI policy
    -> one CLI legacy preflight
    -> gateway.systemd_lifecycle.install
    -> no accidental legacy/new-service coexistence caused by bypass
```

That is the review closure boundary. Anything beyond it belongs in a separate follow-up.
