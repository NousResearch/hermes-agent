# Continuity System & Ledger

## Continuity Ledger (maintain live)

```yaml
continuity:
  timeline: [chronological events with sources]
  character_locations: {char: location + time}
  injuries: {char: description + status}
  wardrobe: {char: current state + changes}
  props: {item: location + condition + last seen}
  knowledge_by_character: {char: [facts they possess]}
  promises:
  secrets:
  unresolved_questions:
  relationships: {pair: status + recent change}
  world_rules: [active rules with exceptions]
  environmental_state:
  emotional_state: {char: current register}  # only as summary; dramatize in scenes
  setup_payoff_pairs: [{setup_scene, payoff_scene, status}]
  approved_visual_details: []
  approved_dialogue: []
```

## Audit Triggers

Before any new scene or revision:
- Cross-check all ledger entries that touch the scene.
- Flag contradictions.

## Detection Rules (run explicitly)

- Impossible travel / time compression without justification.
- Knowledge used before acquisition.
- Props/wardrobe that appear/disappear or change state without cause.
- Emotional resets (character acts as if prior scene never happened).
- Duplicated revelations.
- Forgotten promises or debts.
- Violated world rules.
- Characters acting on information they lack.
- Setup without payoff or premature payoff.
- Inconsistent injuries or physical state.

## Repair Process

1. Identify exact conflict(s) with evidence.
2. Propose smallest viable repair (prefer changing the later material).
3. List all affected downstream scenes/ledger entries.
4. If canon conflict: surface to user before applying.
5. Update ledger after repair.

## Provenance in Ledger

Every entry carries OBSERVED / INFERRED / SPECULATIVE tag + source reference.

## Handoff

Scene contracts must list `continuity_requirements`. After writing, update ledger deltas.
