# Statute PM Control Plane

This local control-plane path lets `default` delegate statute work to `statutepm`, and lets `statutepm` dispatch scoped child work to ephemeral `statute-worker` instances through the SQLite control DB. Discord mirroring and delivery are not part of this rollout path.

`statutepm` is the durable control-plane PM profile id. The current Hermes runtime profile that executes PM code is `nj-statutes-pm`; `hermes_cli/control_runtime.py` contains the explicit `statutepm -> nj-statutes-pm` compatibility shim. Until a later identity migration removes that shim, only `statutepm` claims control-plane dispatches addressed to the PM.

## Isolated Smoke

Run the deterministic no-LLM smoke test against a temporary root:

```bash
python -m hermes_cli.main control smoke-test statutepm
```

Run readiness against an isolated root:

```bash
python -m hermes_cli.main control readiness statutepm
```

The readiness command reports `implementation_ready`, `live_ready`, `deterministic_operational_ready`, `agent_worker_ready`, runtime profile mapping, subprocess spawnability, bootstrap lease age/expiry, required pytest commands, and the strict-mode cutover command. It does not perform cutover.

## Bootstrap

Seed profiles, routes, and stable bootstrap instances in an isolated root:

```bash
ROOT=$(mktemp -d)
python -m hermes_cli.main control bootstrap-statutepm --root "$ROOT" --seed-instances
```

Live bootstrap is intentionally explicit and must not be run without supervisor approval:

```bash
hermes control bootstrap-statutepm --live --seed-instances \
  --pm-profile-id statutepm \
  --worker-profile statute-worker
```

The v1 seeded non-worker instance IDs are `default:bootstrap` and `statutepm:bootstrap`; they are online with a finite one-hour lease and metadata marking them as bootstrap seeded.

## PM Runner

Run one deterministic PM cycle against an isolated root:

```bash
python -m hermes_cli.main control pm run \
  --root "$ROOT" \
  --pm-profile-id statutepm \
  --pm-instance-id statutepm:bootstrap \
  --pm-runtime-profile nj-statutes-pm \
  --worker-profile statute-worker \
  --once
```

Loop mode heartbeats `statutepm:bootstrap`, reaps expired dispatch leases, emits JSONL events, and exits cleanly on interrupt:

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

## Scope Contract

Runtime statute work uses `statute_dispatch_v1`. Child dispatches cannot widen the parent `repo_root`, `allowed_paths`, `task_permissions`, or strict constraints such as `no_live_db_mutation` and `no_push`. This is a DB payload contract, not a filesystem or Git sandbox.

## Live Readiness

Live readiness is read-only:

```bash
python -m hermes_cli.main control readiness statutepm --live-check
```

It checks required profiles/routes in the live DB and reports the bootstrap command if they are missing. It also reports whether the real Hermes runtime profile `statute-worker` exists; readiness does not create it silently.

## Live Smoke and Gates

Live DB smoke mutates the live SQLite control DB and must not run without the literal approval `approved: live db smoke`:

```bash
hermes control live-smoke statutepm \
  --live \
  --deterministic \
  --smoke-tag "statutepm-dispatch-smoke" \
  --idempotency-key "statutepm-dispatch-smoke-v1"
```

The smoke creates a harmless parent dispatch from `default:bootstrap` to `statutepm`, runs one PM cycle, spawns `hermes -p statute-worker control worker run ... --handler deterministic`, and verifies the parent dispatch, child dispatch, result row, artifact row, and PM status message. Smoke artifacts are constrained to `control-plane/smoke/<smoke-tag>/`.

Authority modes:

- `legacy`: the DB exists but is not authoritative.
- `shadow`: tests, audit, and smoke may write DB rows, but production consumers must not treat rows as sole authority.
- `control_db`: approved consumers may use DB rows as authoritative coordination state.

Strict authority cutover remains manual:

```bash
hermes control mode control_db --live --actor-profile default --actor-instance-id <live-admin-instance>
```

Cutover requires the separate literal approval `approved: control_db cutover`. Roll back with:

```bash
hermes control mode shadow --live --actor-profile default --actor-instance-id default:bootstrap
```

`--handler deterministic` is the only operational worker handler in this phase. `--handler agent` returns structured `not_implemented` JSON with a non-zero exit; real LLM statute-worker autonomy requires a later prompt, sandbox, artifact-contract, and review gate.
