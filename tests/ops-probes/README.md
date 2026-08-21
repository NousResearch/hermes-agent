# OPS-PROBE Layer

Cross-cutting kanban contract acceptance fixtures. Unlike
`tests/hermes_cli/` (single-helper unit tests), the probes drive the
full dispatcher surface end-to-end and emit structured JSON
pass/fail evidence.

## Catalog

| Probe | File | What it pins |
|-------|------|--------------|
| Priority inheritance | `test_priority_inherit.py` | C1–C10 (DB layer) + C11–C14 (CLI subprocess). The `child_priority` flag, per-child overrides, NULL/COALESCE, type coercion (bool/str), and dispatch ordering. |
| Dispatch order | `test_priority_dispatch_order.py` | `ORDER BY priority DESC, created_at ASC` plus the created_at tie-break. |

## How to run

```
# All probes
pytest tests/ops-probes/ -v

# Just the priority-inherit fixture
pytest tests/ops-probes/test_priority_inherit.py -v

# Combined with the hermetic unit tests
pytest tests/hermes_cli/test_kanban_decompose_priority.py \
       tests/ops-probes/ -v
```

## How to add a new probe

1. Create `tests/ops-probes/test_<name>.py`.
2. Use `kb.connect_closing(db_path=Path(...))` explicitly — pytest
   sanitises env vars between tests, so falling back to env-derived
   DB paths silently writes to the workspace default. The pattern in
   `test_priority_inherit.py` is the reference.
3. Each test must emit one structured JSON line on stdout via
   `_emit({...})` so CI can grep pass/fail evidence (case, layer,
   scenario, expected, observed, pass).
4. Subprocess tests (Layer 2) MUST launch a fresh
   `python hermes ...` invocation so the on-disk module graph is
   loaded — that is the whole point: it catches the
   `sys.modules` staleness class of regression that an in-process
   probe would miss.
5. Keep the wallclock for the whole file under 30 s on a warm
   cache. Use the `tmp_kanban` fixture from `conftest.py` for
   hermetic DBs and the `_run_cli` helper for subprocess probes.

## What this layer is NOT

- Not a substitute for the unit tests in `tests/hermes_cli/`. The
  probes target cross-cutting contracts; the unit tests pin
  individual helpers.
- Not a load test or a scheduler race test. The dispatcher surface
  itself is exercised separately in
  `tests/hermes_cli/test_kanban_dispatcher*.py`.
- Not a regression for OS-level priority-inheriting mutexes
  (`PTHREAD_PRIO_INHERIT`) — different problem. See the parent
  spec's §0 "Topic clarification" for why.