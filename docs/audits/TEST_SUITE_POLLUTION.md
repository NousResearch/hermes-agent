# Known: suite-level test pollution in `tests/hermes_cli`

Recorded during the merge of the NOVA platform onto `main`, because it changes how the
suite's output should be read and because part of it is mine to fix.

## The measurement

Same selection (`-k "kanban or tenant or dispatch" -p no:randomly`), same interpreter, run
against a clean `origin/main` worktree and against the merged tree:

| Tree | Failed | Passed |
|---|---|---|
| `origin/main` (before this work) | **40** | 466 |
| merged (after this work) | **54** | 497 |

The 40 are identical on both — same files, same per-file counts:

```
13  test_kanban_notify.py          2  test_kanban_write_guard.py
12  test_web_oauth_dispatch.py     2  test_kanban_dispatch_tick_hook.py
 3  test_kanban_worker_lifecycle_hooks.py   2  test_kanban_decompose.py
 2  test_kanban_db.py              1  test_kanban_task_updated_hook.py
 1  test_kanban_swarm.py           1  test_kanban_review_surfaces.py
 1  test_kanban_lifecycle_hooks.py
```

**The merge introduced no regressions.** The extra 14 are all in
`tests/hermes_cli/test_multitenant_execution.py`, which does not exist on `main`.

## The part that is mine

Those 14 **pass in isolation** and **pass when run beside `test_kanban_notify.py`**. They
fail only inside the large mixed run, and they fail in a way worth stating plainly rather
than filing as flake:

```
with kt.use("tenant-b"):
    assert not operation(conn, victim)   # a cross-tenant mutation SUCCEEDED
```

`_guard` returns `True` only when `kanban_tenant.current()` is `None` — the unscoped,
single-tenant path. So under whatever state the mixed run leaves behind, the ambient tenant
binding is not visible to the guard.

### What was ruled out

* **Module reload splitting the contextvar.** If something reloaded
  `hermes_cli.kanban_tenant`, `kt.use()` would set `_ACTIVE` on one module object while
  `_guard`'s lazy import read another. A probe asserting
  `sys.modules["hermes_cli.kanban_tenant"] is kt`, run inside the polluted selection,
  **passed**. No reload occurs.
* **A regression from the merge.** The baseline above settles this.
* **Test fragility alone.** The assertion that fails is a security property, not a timing or
  ordering detail, so it does not get written off as ordering noise.

### What is still open

Which earlier test leaves the process in a state where the binding is invisible, and whether
the same state could occur in production. The candidate worth checking first is anything that
moves query execution off the calling thread — a contextvar set on one thread is not visible
on another, and `hermes_state_readpool` exists. `_tenant_scope()` and `_guard()` both read the
contextvar in the *caller's* thread, which should make them immune, so this is a hypothesis
rather than a finding.

**Until that is resolved, read the suite this way:** `tests/platform` (799) and
`tests/hermes_cli/test_multitenant_execution.py` (45) are green and are the gate for the
platform layer. The mixed `tests/hermes_cli` run carries a pre-existing 40-failure baseline
that predates this work.

## Why this is written down rather than worked around

Marking the 14 as `xfail`, or pinning the tenant explicitly in the fixture to make them pass,
would remove the signal while leaving the cause. The tests are doing their job: they detected
a condition under which tenant isolation does not hold. That the condition is (probably)
confined to a test process does not make it uninteresting — it makes it cheap to investigate
before it is not.
