---
name: manim-data-viz-bridge
description: "Bridge matplotlib/seaborn high-end charts into Manim CE explainer videos. Use when the user wants a single coherent prompt turned into a data visualization plus animated mathematical or technical narration (3B1B-style pacing)."
version: 0.1.0
license: MIT
metadata:
  hermes:
    tags: [manim, matplotlib, seaborn, data-viz, video, animation, bridge]
    related_skills: [manim-video]
---

# Manim Data Visualization Bridge

## Role

Turn **one natural-language request** into:

1. A **structured** `VizSpecification` (filled by the agent from the prompt).
2. A **static PNG** chart (matplotlib + seaborn).
3. A **Manim Community Edition** scene script that reveals the chart and optional `MathTex` steps.

The Python package under this folder is **stateless**: it only writes inside a `tempfile` work directory and returns paths plus a suggested `manim` CLI line.

## Single-prompt workflow (agent-side)

1. Parse the user prompt into `DataVizBridgeRequest`: populate `viz.data` (exactly one of `records`, `x`+`y`, or `matrix`).
2. Choose `chart_type` and axis labels; add 0–12 short LaTeX fragments (no `$`) in `manim.tex_explanations` for the math/tech story after the chart appears.
3. Call `ManimDataVizBridge().build(request)`.
4. Run the returned `render_suggestion` with cwd = `work_dir` (requires Manim CE + deps on the user machine).

## Files

| File | Purpose |
|------|---------|
| `models.py` | Pydantic IO models (`agentskills.io`-friendly field descriptions). |
| `interface.py` | `DataVizBridgeProtocol` for alternate implementations. |
| `chart_engine.py` | seaborn/matplotlib → PNG. |
| `manim_emitter.py` | PNG → Manim `Scene` source. |
| `connector.py` | Orchestration. |

## Constraints

- No absolute paths in emitted Manim source; chart is referenced by filename inside `work_dir`.
- Extend chart styles by editing `chart_engine._draw`, not the agent loop.

See `references/bridge-patterns.md` for data shape examples.
