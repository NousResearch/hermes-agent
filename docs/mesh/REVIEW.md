# `hermes mesh` — scaffold v0.2 (review-incorporated)

This is the **structural scaffold** for the `hermes mesh` subcommand we're proposing to upstream into `NousResearch/hermes-agent`. Companion design doc: `~/PhoebeVault/Phoebe/projects/mesh-provisioner-design.md` (v0.3).

**Status:** v0.2 — Claude's review (Q1–Q6 + cross-cutting) folded in. Function bodies remain stubs; the *shape* is locked. Next pass: implementations against a real Pi.

## Changelog from v0.1

| Source | What changed |
|---|---|
| Q1 | Added `hermes mesh ssh-setup <host>` command |
| Q2 | Re-exported types from `mesh/__init__.py` (`NodeSpec`, `ControllerConfig`, `ProbeResult`, `NodeStatus`, `DriftState`) |
| Q3 | New `ProbeResult` dataclass; `render_role()` now takes `facts: ProbeResult`; `$node_user` flows from probe, not operator input |
| Q4 | `bare` template: `datetime.now(timezone.utc)`, no explicit TLS version pin, `socket.gethostname()` not `getfqdn()`, alive semantics documented in `manifest.yaml`, explicit `reconnect_delay_set`, `clean_session` trade documented |
| Q5 | `NodeStatus` gains `metadata_mismatch: bool` + `mismatch_detail: str` (orthogonal axis, not a 5th state) |
| Q6 | `add_node` defaults `rollback_on_failure=True`; CLI adds `--no-rollback` flag; `_rollback` docstring documents marker-guarded, step-wise, idempotent, best-effort policy |
| Cross-cutting | `_Redacted(str)` wrapper for `broker_password`; `# DO NOT LOG` notes near vars_; v0.2 TODO note on capabilities-as-JSON; `template_dirs` precedence documented at construction site |

## What's real here

| File | What's complete | What's stubbed |
|---|---|---|
| `hermes_cli/mesh.py` | Full CLI surface (9 commands), argparse wiring, docstrings | All `cmd_*` bodies |
| `mesh/__init__.py` | Public API re-exports | — |
| `mesh/provisioner.py` | `NodeSpec`, `ControllerConfig`, `ProbeResult`, `_Redacted`; SSH/SCP primitives; `probe()`; `render_role()` (real `string.Template`); `add_node()` orchestration shape + rollback flow | `deploy`, `validate_and_register` body, `init_controller`, `ssh_setup`, `remove_node`, `_rollback`, `_render_env` v0.2 caps work |
| `mesh/registry.py` | Module docstring + signatures | All bodies |
| `mesh/validation.py` | `NodeStatus` (incl. mismatch fields), `DriftState` literal, taxonomy docs | `wait_for_heartbeat`, `drift_check` |
| `mesh/templates/roles/bare/` | **Fully runnable** reference role | — |

## Deliberately deferred to v0.2+ (with notes in code)

- `hermes mesh upgrade <host>` — re-render + redeploy after role-template change. `remove + add` works.
- `hermes mesh ping <host>` — narrow alive check. Subset of `status`.
- Capabilities as JSON array in `.env` — `_render_env` has TODO comment.
- `watchdog` and `compute` role templates — hold until `bare` proves the pipeline against a real node.

## What's deliberately NOT here yet

- Implementation bodies (next pass — against a real Pi)
- Tests (`tests/`)
- `docs/mesh/{README,ADDING_A_ROLE,SECURITY}.md`

## How to read this

`hermes_cli/mesh.py` → `mesh/provisioner.py` → `mesh/templates/roles/bare/script.py.tmpl`. The other two `mesh/*.py` files are signature + dataclass only.

— Phoebe, 2026-05-20 ~10:50 ET
