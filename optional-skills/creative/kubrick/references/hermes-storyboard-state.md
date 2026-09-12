# Hermes Storyboard State

Kubrick preserves a shared private graph across frames while allowing controlled, observable state mutation.

## Sequence

1. Compile or build a valid private motif graph.
2. Define a frame plan containing only intended state deltas.
3. Run `propagate_graph_state.py` to inherit graph identity, node state, ownership, residue, and convergence.
4. Run `compare_frame_state.py` to inspect adjacent transitions.
5. Repair prohibited resets or lost residue before model adaptation.

## Commands

```bash
python scripts/propagate_graph_state.py \
  --graph out/kubrick/motif-graph.private.yaml \
  --plan project/storyboard-plan.yaml \
  --output out/kubrick/storyboard-symbolic-state.yaml

python scripts/compare_frame_state.py \
  --storyboard out/kubrick/storyboard-symbolic-state.yaml \
  --output out/kubrick/storyboard-transition-report.json
```

## Frame plan contract

Each frame may mutate:

- node states,
- motif ownership,
- object state,
- light state,
- material state,
- residue additions,
- convergence-site participation.

State is inherited unless explicitly changed. Residue is monotonic by default. Returning a node to its graph initial state is treated as a prohibited reset unless `allow_resets: true` is explicitly supplied.

## Hermes rules

- shared `graph_id` is immutable across the storyboard,
- every recurrence must mutate or carry a justified persistent function,
- object damage and repair remain visible until explicitly resolved,
- ownership transfers must survive subsequent frames,
- convergence sites may move, but disappearance is recorded,
- private pattern and lexicon links never enter audience prompts,
- model adapters consume this neutral state packet; they do not rewrite it.
