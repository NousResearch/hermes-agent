# Kanban Video Orchestrator

Plan, set up, and monitor a multi-agent video production pipeline backed by Hermes Kanban.

## What it does

This package helps you:

- scope a video brief
- design a role-appropriate team
- generate a shared-workspace `setup.sh`
- fire the initial director task
- monitor execution and intervene when tasks stall

## Package layout

```text
SKILL.md
plan.schema.json
assets/
  brief.md.tmpl
  setup.sh.tmpl
  soul.md.tmpl
scripts/
  bootstrap_pipeline.py
  monitor.py
examples/
  example-plan-product-teaser.json
  example-plan-ascii-music-video.json
  example-plan-manim-explainer.json
docs/
  dry-run-checklist.md
```

## Quick start

1. Edit one of the example plans and replace placeholder asset paths.
2. Generate the project artifacts:

```bash
python scripts/bootstrap_pipeline.py examples/example-plan-product-teaser.json   --out setup.sh   --brief-out brief.md   --team-out TEAM.md
```

3. Review `brief.md`, `TEAM.md`, and `setup.sh`.
4. Run the setup:

```bash
bash setup.sh
```

5. Monitor the tenant:

```bash
hermes kanban watch --tenant q3-product-teaser
hermes kanban list  --tenant q3-product-teaser
hermes kanban stats
python scripts/monitor.py --tenant q3-product-teaser --once
```

> Note: current Hermes `kanban stats` is board-scoped and has no `--tenant` flag; use `kanban list --tenant ...` or `scripts/monitor.py` for tenant-specific monitoring.

## Validation and schema

If your local generator includes the extended CLI, you can also use:

```bash
python scripts/bootstrap_pipeline.py --schema-out plan.schema.json
python scripts/bootstrap_pipeline.py examples/example-plan-product-teaser.json --validate-only
```

## Notes

- All project tasks should use one shared `dir:` workspace.
- All project tasks should use one tenant slug consistently.
- The director decomposes and routes work; specialists execute it.
