# Hermes Agent — Development Guide

Read every matching area guide before editing; overlapping routes apply together. Area
`AGENTS.md` files also load automatically in their directories. `python scripts/check` caps
this file at 12k chars, root-to-area chains at 30k; disclose detail through task links.

## Design invariants — the review lens

- **Per-conversation prompt caching is sacred.** Mutating past context, swapping toolsets,
  reloading memories or rebuilding the system prompt mid-conversation breaks the cached prefix and
  multiplies the user's cost; the ONE exception is context compression. Slash commands that change
  system-prompt state defer to the next session, with an opt-in `--now` (`/skills install --now`).
- **The core is a narrow waist; capability lives at the edges.** Every model tool is sent on
  every API call, so the bar for a new *core* tool is high. New capability should arrive as a
  CLI command + skill, a service-gated tool, or a plugin — not as core surface.

## Contribution rubric

Intent layer, for contributors and the triage sweeper (which may only close on
`implemented_on_main`, `cannot_reproduce` or `incoherent`; taste-based closes are a maintainer's
call, and when in doubt a PR stays open). Before planning, proposing, implementing, reviewing or
triaging a contribution — new capability included — read
[contributing.md § Contribution rubric](website/docs/developer-guide/contributing.md#contribution-rubric):
wanted/rejected patterns, verify-the-premise, footprint ladder.

**Security:** [SECURITY.md](SECURITY.md) is the scope authority. A §3.1 finding goes private (GitHub Security
Advisories or security@nousresearch.com), never into a public issue, PR, commit or comment; §3.2
hardening is ordinary public work. Name the §2 boundary crossed, with a repro on `main`.

## Task rules

Code changes and reviews: read the common rules in [CODING_STANDARDS.md](CODING_STANDARDS.md#task-index),
then only applicable branches. Tests, environment and dependencies: use that index for the relevant
branches. Documentation-only work needs no coding-standards preload.

## Routing — working in X → read

The filesystem is canonical.

| Area | Read |
|---|---|
| run_agent.py, agent/ | [AGENTS.md](agent/AGENTS.md) |
| cli.py, hermes_cli/ | [AGENTS.md](hermes_cli/AGENTS.md) |
| gateway/ | [AGENTS.md](gateway/AGENTS.md) |
| gateway/platforms/ (new adapter) | [ADDING_A_PLATFORM.md](gateway/platforms/ADDING_A_PLATFORM.md) |
| tools/, toolsets.py, model_tools.py | [AGENTS.md](tools/AGENTS.md) |
| plugins/, hermes_cli/plugins*.py | [AGENTS.md](plugins/AGENTS.md) |
| plugin-catalog/ | [README.md](plugin-catalog/README.md) |
| tui_gateway/, ui-tui/ | [AGENTS.md](tui_gateway/AGENTS.md) |
| web/, hermes_cli/web_routers/ | [AGENTS.md](web/AGENTS.md) |
| apps/desktop/ | [AGENTS.md](apps/desktop/AGENTS.md) |
| apps/desktop/src/ (frontend) | [AGENTS.md](apps/desktop/src/AGENTS.md) |
| skills/, optional-skills/, agent/curator*.py | [AGENTS.md](skills/AGENTS.md) |
| cron/, kanban | [AGENTS.md](cron/AGENTS.md) |
| tests/ | [AGENTS.md](tests/AGENTS.md) |
| pm/, pyproject.toml, uv.lock | [AGENTS.md](pm/AGENTS.md) |
| hermes_platform/ | [AGENTS.md](hermes_platform/AGENTS.md) |
| hermes_state*.py (facade + siblings) | [CODING_STANDARDS.md § Facades and siblings](CODING_STANDARDS.md#facades-and-siblings) |
| hermes_constants.py | [CODING_STANDARDS.md § Paths, profiles and machine facts](CODING_STANDARDS.md#paths-profiles-and-machine-facts) |

Long-form background: `website/docs/developer-guide/`; workflow rules (PR/issue/review/salvage): the `hermes-agent-dev` skill.
