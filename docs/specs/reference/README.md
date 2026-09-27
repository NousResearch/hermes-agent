# Reference assets

These are versioned specification inputs, not runtime assets.

- [Connections guide](connections-guide.md): user-approved shortened local guide.
  `<connections-root>` is resolved during implementation; all other wording is
  the approved draft.
- [Hindsight configuration](hindsight-config.json): image pin, API environment
  and bank template extracted without importing infrastructure code. Pool sizes
  are resolved from the reference production configuration (development matches).
  Secret values are not included. Deployment expressions remain visibly marked.

Provenance: local reference checkout `/Users/markvasilyev/ai-employee`, commit
`6f4f56aae0`. Relevant source files: `guides/connections/guide.md`,
`infra/hermes_infra/stacks/hindsight.py`, `infra/environments/production.json`,
`infra/hindsight/Dockerfile`, and `infra/hindsight/hindsight_bank_reconciler.py`.
The source working tree was clean when inspected. Preserve these reference
values; do not auto-refresh them from a moving branch.

The master spec defines the fork's product independently of this provenance.
The reference Hindsight snapshot is not a complete Railway launch manifest:
translate endpoints, authentication and deployment inputs as explicitly agreed,
and verify pinned-image behavior and bank reconciliation before deployment.
