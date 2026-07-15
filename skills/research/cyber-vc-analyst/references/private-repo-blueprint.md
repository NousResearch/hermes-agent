# Private Repo Blueprint

This skill can act as the Hermes runtime surface for a broader private cyber
investment workflow repository.

## Recommended Repository Shape

```text
cyber-vc-analyst/
  README.md
  docs/
    workflow-overview.md
    slack-usage.md
    vault-integration.md
    ros-mcp-setup.md
    analyst-operating-guide.md
    workflow-phases.md
    research-depth.md
    research-state.md
  prompts/
    company-analysis.md
    theme-analysis.md
    compare-analysis.md
    triage-analysis.md
    competitors-analysis.md
  schemas/
    company-analysis.schema.yaml
    theme-analysis.schema.yaml
    compare-analysis.schema.yaml
    triage-analysis.schema.yaml
    competitors-analysis.schema.yaml
    evidence-ledger.schema.yaml
  fixtures/
    company-red-access.input.yaml
    company-red-access.expected.md
    theme-soc-automation-ai-soc.input.yaml
    theme-soc-automation-ai-soc.expected.md
    compare-red-access-vs-noma.input.yaml
    compare-red-access-vs-noma.expected.md
    triage-red-access.input.yaml
    triage-red-access.expected.md
    competitors-browser-security.input.yaml
    competitors-browser-security.expected.md
  release-notes/
    2026-07-02-v1.1.0.md
    2026-07-03-v1.2.0.md
  research-state/
  skills/
    cyber-vc-analyst/
    cyber-vc-company/
    cyber-vc-theme/
    cyber-vc-compare/
    cyber-vc-triage/
    cyber-vc-competitors/
```

## Ownership Split

Use the private repo as the system of record for:

- canonical prompts
- schemas
- fixtures
- release notes
- operator docs

Use the Hermes bundled skill as the runtime copy that:

- receives Slack invocations
- resolves vault context
- uses ROS MCP when available
- writes durable memo outputs

## Synchronization Rule

Do not treat Hermes as the canonical authoring surface for this workflow.
Instead:

1. Evolve the prompts and schemas in the private repo.
2. Sync the runtime-facing skill copy into Hermes when you want the agent to
   use the new behavior.
3. Keep private fixtures and research artifacts out of public or shared runtime
   repos unless they are intentionally sanitized.

## Suggested Workflow Modules

Beyond single-company analysis, the private repo can support:

- thematic memos
- comparison workflows
- Slack triage workflows
- competitor landscapes
- diligence-gap detection
- market maps
- meeting-note to company-brief conversion
- portfolio watchlist updates

## Release Discipline

Track these changes explicitly:

- prompt changes
- taxonomy changes
- schema changes
- MCP dependency changes
- Slack workflow changes
- known limitations
