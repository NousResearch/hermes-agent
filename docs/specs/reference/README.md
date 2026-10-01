# Reference assets

These are versioned specification inputs, not runtime assets.

- [Native guide adaptations](native-guide-adaptations.diff): the native
  `skills/autonomous-ai-agents/hermes-agent/` material ported to `guides/employee/`.
  The skill frontmatter is omitted; compatible references/templates are copied.
  Runtime-specific memory, browser and service-connection references are local
  additions. Native source: skill version 3.2.0, Hermes Agent + Teknium, MIT;
  latest source-directory commit `99768fd1274203a821c2eea42b7d02f2c1440672`.
- [Employee guide adaptations](guide-adaptations.diff): retained responsibility
  and file-keeping wording, with path/name substitutions normalized out.
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

Connection guide provenance: native baseline
`6e69a8933adda7dbbff7cf3009a259a4524477e9`:

- `skills/productivity/google-workspace/`: v1.2.0, Nous Research, MIT.
  Guide and Gmail search reference copied; four helper scripts copied byte for
  byte. Corrected native guide drift: the setup helper has fixed requested scopes,
  plain-text output and explicit auth-URL retries, not service/JSON flags. Daily-brief workflow omitted because responsibilities own work/schedules.
- `skills/email/himalaya/`: v1.1.0, community, MIT. Guide and references copied.
  Examples target Himalaya 1.2.x; corrected account/flag argument placement and
  named configure account. Credentials use command/keyring examples.
- `skills/software-development/github/`: v2.0.0, Ben Barclay / Hermes, MIT.
  Connection-only adaptation of SKILL.md and references/auth.md; uses native gh
  login/setup-git. Task/review workflows and custom device/token helper scripts
  are not copied. Native gh already owns authentication.

[Connection adaptations](connection-guide-adaptations.diff) record prose changes
against those sources (frontmatter omitted). Paths point to shipped guides,
setup asks only for missing inputs, writes respect existing responsibility/user
authority, and manuals contain operation rather than setup instructions.
Google credentials retain native per-Hermes-home storage; gh/Himalaya keep their
own native configuration locations. No hosted gateway or connection seeds ship.
