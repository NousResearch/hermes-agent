# Statute PM Operationalization Proposal

This phase makes the local SQLite control plane operational for deterministic statute PM dispatch plumbing:

```text
default/Galt -> statutepm control profile -> statute-worker -> statutepm -> default/Galt
```

It does not implement autonomous LLM statute-worker execution. `--handler deterministic` is the operational worker path; `--handler agent` returns structured `not_implemented` JSON and exits non-zero.

## Fixed Identities

- Durable PM control profile: `statutepm`
- PM runtime profile: `nj-statutes-pm`
- Worker control/runtime profile: `statute-worker`
- Mapping shim: `hermes_cli/control_runtime.py`

The `statutepm -> nj-statutes-pm` mapping is compatibility glue for the current runtime profile layout. `statutepm` remains the only control-plane claimer for PM dispatches until a future identity migration is designed and reviewed.

## Commands

Bootstrap:

```bash
hermes control bootstrap-statutepm --live --seed-instances \
  --pm-profile-id statutepm \
  --worker-profile statute-worker
```

One PM cycle:

```bash
hermes control pm run \
  --live \
  --pm-profile-id statutepm \
  --pm-instance-id statutepm:bootstrap \
  --pm-runtime-profile nj-statutes-pm \
  --worker-profile statute-worker \
  --once
```

Looping PM:

```bash
hermes control pm run \
  --live \
  --pm-profile-id statutepm \
  --pm-instance-id statutepm:bootstrap \
  --pm-runtime-profile nj-statutes-pm \
  --worker-profile statute-worker \
  --loop \
  --poll-interval-s 5 \
  --child-timeout-s 600
```

Readiness:

```bash
hermes control readiness statutepm --live-check
```

Deterministic live smoke, only after literal approval `approved: live db smoke`:

```bash
hermes control live-smoke statutepm \
  --live \
  --deterministic \
  --smoke-tag "statutepm-dispatch-smoke" \
  --idempotency-key "statutepm-dispatch-smoke-v1"
```

Authority cutover, only after separate literal approval `approved: control_db cutover`:

```bash
hermes control mode control_db --live \
  --actor-profile default \
  --actor-instance-id default:bootstrap
```

Rollback:

```bash
hermes control mode shadow --live \
  --actor-profile default \
  --actor-instance-id default:bootstrap
```

## Mode Semantics

- `legacy`: DB exists but is not authoritative.
- `shadow`: DB may be written for tests, audit, and smoke proof, but production consumers must not treat rows as sole authority.
- `control_db`: approved consumers may use DB rows as authoritative coordination state.

## Verification

Operational readiness requires:

- `control pm run --help` parses.
- `control worker run --help` parses.
- `hermes -p statute-worker ... --help` dry-run spawnability passes.
- Isolated `control smoke-test statutepm` passes.
- Focused control-plane and Discord decommission tests pass.
- Approved live deterministic idle and dispatch smokes pass, or remain queued behind the approval gate.
