# Skill Governance

Hermes can apply provenance-aware governance to skill selection and skill search ranking.

The configuration lives under `skills.governance` in `config.yaml`:

```yaml
skills:
  governance:
    registry_path: governance/skills-registry.yaml
    task_class: ardyn_engineering
    protected_task_classes:
      - ardyn_engineering
    retrieval_ranking: true
```

Fields:

- `registry_path`: path to a machine-readable registry file. Relative paths resolve against `HERMES_HOME`.
- `task_class`: optional ambient task-class tag used by preload, automatic selection, and retrieval ranking.
- `protected_task_classes`: task classes that fail closed for non-current skills.
- `retrieval_ranking`: when true, supported skills-hub retrieval surfaces rank results with governance classification awareness.

## Registry Schema

The registry is YAML (or JSON with the same shape):

```yaml
version: 1
skills:
  - name: ModernCurrent
    classification: CURRENT
    aliases: [modern-current]
    provenance:
      source: qualified-registry
      lineage: current-v3
  - name: ToolTrust
    classification: COMPATIBILITY_ONLY
    aliases: [tooltrust]
    provenance:
      source: legacy-catalog
      lineage: tooltrust-v1
  - name: PREMP
    classification: STALE
    aliases: [premp]
    provenance:
      source: legacy-catalog
      lineage: premp-v1
```

Supported `classification` values:

- `CURRENT`
- `COMPATIBILITY_ONLY`
- `STALE`
- `CONFLICTING`
- `UNKNOWN`

## Decision Rules

For protected task classes:

- `CURRENT`: allowed
- `COMPATIBILITY_ONLY`: allowed only with explicit historical intent
- `UNKNOWN`: rejected
- `STALE`: rejected
- `CONFLICTING`: rejected

For unprotected task classes, selection remains permissive and the classification is still exposed for logs and ranking.

Current integrations:

- CLI/TUI skill preload (`--skills`, startup preload path)
- Gateway automatic channel/topic skill bindings
- Skills-hub retrieval ordering

## Deterministic Tests

Targeted tests for the governance facility:

```bash
pytest tests/agent/test_skill_governance.py
pytest tests/agent/test_skill_commands.py tests/gateway/test_slack_channel_skills.py
```

The governance test fixture uses a temporary `HERMES_HOME` and proves:

- `ToolTrust` (`COMPATIBILITY_ONLY`) is rejected from protected automatic/preload selection unless historical intent is explicit
- `PREMP` (`STALE`) is rejected
- an unregistered skill is treated as `UNKNOWN` and rejected
- `ModernCurrent` (`CURRENT`) is accepted
