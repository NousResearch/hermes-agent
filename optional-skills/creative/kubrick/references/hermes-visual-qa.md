# Hermes Closed-Loop Visual QA

Kubrick compares a structured visual observation against the expected graph or storyboard frame state. It does not inspect images by itself and does not call a vision model. Any observer may produce the observation packet, but the comparison and correction stages remain deterministic.

## Sequence

1. Generate a frame from a validated adapter packet.
2. Produce a `visual-observation.schema.yaml` packet from human review or an optional vision model.
3. Compare expected and observed state with `compare_visual_observation.py`.
4. Inspect dimension-specific scores and mismatches.
5. Build a targeted regeneration packet with `build_visual_correction_packet.py`.
6. Regenerate while preserving graph identity and all passing dimensions.

## Commands

```bash
python scripts/compare_visual_observation.py \
  --expected out/kubrick/storyboard-symbolic-state.yaml \
  --observation out/kubrick/frame-002-observation.yaml \
  --output out/kubrick/frame-002-fidelity-report.json

python scripts/build_visual_correction_packet.py \
  --report out/kubrick/frame-002-fidelity-report.json \
  --adapter-packet out/kubrick/model-adapter-packet.yaml \
  --output out/kubrick/frame-002-correction.yaml
```

## Fidelity dimensions

- geometry,
- node state,
- motif ownership,
- object state,
- light state,
- material state,
- residue,
- convergence,
- continuity.

Scores remain separate. A high average cannot hide a lost owner, missing residue, graph mismatch, or failed convergence site.

## Rules

- graph identity mismatch returns `NOT_COMPUTABLE`,
- high-severity state mismatches require revision,
- passing dimensions become preservation locks,
- correction packets may request only observed repairs,
- adapters and observers may not rewrite symbolic intent,
- named esoterica remains outside audience-facing prompts,
- image generation and visual observation providers remain optional.
