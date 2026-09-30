# Reference assets

These are versioned specification inputs, not runtime assets.

- [Native guide adaptations](native-guide-adaptations.diff): the native
  `skills/autonomous-ai-agents/hermes-agent/` material ported to `guides/employee/`.
  The skill frontmatter is omitted; compatible references/templates are copied.
  Runtime-specific memory, browser and service-connection references are local
  additions. Native source: skill version 3.2.0, Hermes Agent + Teknium, MIT;
  latest source-directory commit `99768fd1274203a821c2eea42b7d02f2c1440672`.
- [Employee guide adaptations](guide-adaptations.diff): retained responsibility
  wording, with path/name substitutions normalized out.
- [Hindsight configuration](hindsight-config.json): image pin, API environment
  and bank template extracted without importing infrastructure code. Pool sizes
  are resolved from the reference production configuration (development matches).
  Secret values are not included. Deployment expressions remain visibly marked.

Provenance: local reference checkout `/Users/markvasilyev/ai-employee`, commit
`6f4f56aae0`. Hindsight source files: `infra/hermes_infra/stacks/hindsight.py`, `infra/environments/production.json`,
`infra/hindsight/Dockerfile`, and `infra/hindsight/hindsight_bank_reconciler.py`.
The source working tree was clean when inspected. Preserve these reference
values; do not auto-refresh them from a moving branch.

The master spec defines the fork's product independently of this provenance.
The reference Hindsight snapshot is not a complete Railway launch manifest:
translate endpoints, authentication and deployment inputs as explicitly agreed,
and verify pinned-image behavior and bank reconciliation before deployment.
