# Team & Task Graph — Fixture Product Teaser

## Team

- `director` — decompose the video into specialist tasks without executing renderer work (loads `kanban-orchestrator`)
- `ascii-renderer` — render the title card and product beat as deterministic ASCII frames (loads `ascii-video`)
- `editor` — assemble rendered scenes into the final video artifact (no skills required)
- `reviewer` — verify artifact duration, resolution, and brief compliance (no skills required)

## Task Graph

```
T0  director — decompose
T1    ascii-renderer — scene 1: ASCII title card: Fixture Product Teaser (parents: T0)
T2    ascii-renderer — scene 2: Product silhouette resolves into call to action (parents: T0)
T3    editor — assemble + mux (parents: T1, T2)
T4    reviewer — final QA (parent: T3)
```

## Per-task workspace requirement

All `kanban_create` calls MUST pass:
```
workspace_kind="dir"
workspace_path="$HOME/projects/video-pipeline/fixture-product-teaser"
tenant="fixture-product-teaser"
```