# Hermes OS Architecture-First Framework

## Purpose

Hermes OS must act as product manager, systems architect, principal engineer, and technical program manager before it acts as a coder.

The target flow is:

```text
Idea
  -> Grill-Me
  -> Architecture
  -> Domain Model
  -> Workflow
  -> Dashboard
  -> Specification
  -> Tasks
  -> Execution
  -> Validation
  -> Continuous Improvement
```

## Core Rule

Coding starts only after architecture review. If a project lacks business framing, domain models, workflows, dashboard requirements, metrics, approvals, or agent boundaries, Hermes OS should warn, block, or request architecture work before implementation.

## Source Of Truth

Hermes OS owns:

- Projects.
- Tasks.
- Workflows.
- State.
- Dashboards.
- Metrics.
- Approvals.
- Specifications.
- History.
- Memory.

Official Hermes Agent owns:

- Reasoning.
- Tool use.
- Delegated execution.
- Worker coordination.
- Runtime memory cache.

Agents produce artifacts. Hermes OS validates, stores, and displays them.

## Required Project Documents

Every architecture-first project receives:

- `PROJECT.md`
- `DOMAIN.md`
- `WORKFLOWS.md`
- `DASHBOARD.md`
- `METRICS.md`
- `APPROVALS.md`
- `AGENTS.md`
- `TASKS.md`
- `DECISIONS.md`
- `ROADMAP.md`
- `ARCHITECTURE.md`

## Review Outputs

`hermes architect review <project>` should produce:

- Architecture score.
- Critical gaps.
- Missing entities.
- Missing workflows.
- Missing metrics.
- Missing dashboards.
- Missing approvals.
- Missing schemas.
- Automation opportunities.
- Recommended improvements.
- Priority roadmap.

## Migration Targets

Initial existing-project reviews:

- Workspace projects.
- workspace project.
- workspace project.
- workspace project.

Each review should generate missing documentation, workflows, dashboards, schemas, approvals, and an improvement roadmap.
