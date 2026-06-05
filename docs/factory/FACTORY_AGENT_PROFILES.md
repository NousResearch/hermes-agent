# SitioUno Software Factory — Agent Profiles

This document separates two concepts that are easy to confuse:

1. Database agents: rows in `software_factory.factory_agents` / local Factory DB. They are the canonical roster, routing metadata, metrics identity and audit identity.
2. Hermes profiles: executable worker identities under `~/.hermes/profiles/<name>/` with their own `config.yaml`, `SOUL.md`, memory, skills and wrapper command.

A Factory agent is not operational until both exist. The DB tells Zeus who should do what; the Hermes profile is the runnable process that can actually receive Kanban work.

## Created executable profiles

All 12 Factory v1 agents now exist as real Hermes profiles and wrapper commands:

| Profile | Runtime role | Primary outputs |
| --- | --- | --- |
| `factory-orchestrator` | Factory controller: intake, task graph, routing, gates, metrics | intake, task graph, benchmark, delivery report |
| `product-analyst` | PRD, functional spec, acceptance criteria | `FUNCTIONAL_SPEC.md`, `ACCEPTANCE_CRITERIA.md` |
| `solution-architect` | technical blueprint, DB/API/security design | `TECHNICAL_BLUEPRINT.md`, ADRs |
| `implementation-planner` | epics, task graph, dependencies, criteria of done | `IMPLEMENTATION_PLAN.md`, Kanban graph |
| `claude-builder` | complex implementation/refactor lane | diffs, commands, tests |
| `codex-builder` | bounded fixes/tests/reviews lane | diffs, commands, tests |
| `openhands-lab` | sandbox experiments/heavy validation | sandbox report, evidence |
| `quality-reviewer` | independent quality/spec gate | `QUALITY_REVIEW.md`, block/approve gate |
| `security-reviewer` | security/PII/payments/auth gate | `SECURITY_REVIEW.md` |
| `qa-verifier` | smoke/regression/browser verification | `QA_REPORT.md`, evidence |
| `devops-release` | CI/CD, migrations, release checks | `RELEASE_REPORT.md` |
| `factory-reporter` | executive synthesis and benchmark reporting | `DELIVERY_REPORT.md`, `ENGINE_BENCHMARK.md` |

## Runtime configuration

Each profile was cloned from `default`, then customized:

- `SOUL.md` rewritten with role mission, expected outputs and checkpoint format.
- `config.yaml` `agent.name` set to the role name.
- `config.yaml` `terminal.cwd` set to `/home/jean/Projects/hermes-agent-original`.
- `config.yaml` `terminal.env_passthrough` includes `FLEET_REGISTRY_DATABASE_URL`.
- `config.yaml` `toolsets` narrowed per role so the profile has the tools it needs.

## Factory method lanes

For each project the Factory should create three logical lanes:

1. `<project>-zeus`: Zeus Native Factory lane using deterministic DB/Kanban/gates.
2. `<project>-bmad`: BMAD/Hybrid lane for PRD/architecture/stories/adversarial review.
3. `<project>-integration`: integration lane that compares evidence and merges the selected result.

Implementation workers must use isolated branches/worktrees per lane.

## Verification commands

```bash
hermes profile list
hermes factory agents --json
hermes factory status --json
```

Expected current state:

- 12 DB agents.
- 12 executable Hermes profiles.
- Profile wrappers under `~/.local/bin/<profile>`.

## Important operational rule

Do not treat legacy OpenClaw/OpenClaw-style swarm state as source of truth. The source of truth for the new SitioUno Factory is:

1. Kanban board for work state.
2. `software_factory` Postgres schema / local Factory SQLite for audit, gates and metrics.
3. Hermes profile `SOUL.md` for role behavior.
