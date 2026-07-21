# Blender Scripts — Agent Instructions

This directory contains fork-specific Blender scripts for city destruction
simulation, compositor pipeline setup, and voice narration generation.
They are **not** part of the official Hermes repo — merge exclusions apply.

## Contents

| File | Description |
|------|-------------|
| `render_no_outputfile.py` | Final working render script using `CompositorNodeOutputFile` |
| `setup_compositor_final.py` | Blender 5.2 compositor: Glare bloom → AlphaOver white fade → PNG output |
| `city_destruction_sim.py` | Full scene: buildings, ground, camera, lighting, animation |
| `city_destruction_sim_fixed.py` | Corrected version of the above |
| `make_narration.py` | VOICEVOX narration generation (7.2s short version) |
| `make_narration2.py` | VOICEVOX narration generation (39s full version) |
| `narration_text.txt` | Narration script text |

## Upstream merge note

These scripts are fork-specific and should **not** be included in PRs to
NousResearch/hermes-agent. See `fork/AGENTS.md`.
