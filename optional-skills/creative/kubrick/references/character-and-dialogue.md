# Character System & Dialogue Engine

## Core Character Schema (per major character)

```yaml
character:
  name:
  story_function: (protagonist / antagonist / ally / foil / mentor / etc.)
  public_identity:
  private_identity:
  conscious_goal:
  underlying_need:
  fear:
  wound_or_formative_pressure:
  false_belief:
  values: []
  contradiction: (internal conflict that drives behavior)
  competence:
  vulnerability:
  social_mask:
  default_tactic:
  pressure_tactic:
  breaking_point:
  secret:
  relationship_to_theme:
  relationship_to_protagonist:
  speech_profile: (see voice below)
  physical_behavior: (observable tells under stress)
  arc_start:
  arc_thresholds: [list of events that force change]
  arc_end:
```

Characters must have agency and their own agenda. No pure expositors or validators.

## Relationship Map (for key pairs)

For each significant relationship:
- what A wants from B
- what B wants from A
- what each withholds
- power imbalance (current + shifting)
- unresolved history
- trigger behavior that ignites conflict
- how the relationship transforms

## Voice Fingerprint (per character)

```yaml
voice:
  sentence_length: (short / varied / run-on / clipped)
  rhythm: (staccato / flowing / interruptive)
  vocabulary: (level, jargon sources, avoided words)
  directness: (high / indirect / manipulative)
  metaphor_source: (work, nature, machinery, body, etc.)
  humor_style: (sarcasm / deadpan / physical / absurd / none)
  avoidance_pattern: (what they refuse to say directly)
  stress_behavior: (repetition, questions, commands, silence)
  repeated_constructs:
  forbidden_constructs:
```

## Dialogue Generation Rules

1. Derive from current intention + power + subtext + history.
2. No two characters share identical rhythm/vocab.
3. Never restate visible action.
4. Shared history revealed only under pressure or naturally.
5. No speeches that state the theme or writer's message.
6. Interruptions = behavioral (not decorative).
7. Silence has function (withholding, processing, power move).
8. Exposition motivated by conflict, procedure, persuasion, or consequence.
9. Humor from worldview, status, timing, or reversal.
10. Test speakability aloud.

## Dialogue Polish Pass

- Identify each speaker's immediate want and hidden agenda.
- Increase subtext and contradiction.
- Vary sentence length and interruptions according to voice.
- Remove emotional labels; replace with behavior or strategic language.
- Preserve all plot facts and approved canon.

## Behavioral Translation of Internal States

Instead of "she was angry":
- She grips the edge of the table until her knuckles whiten.
- She answers one beat too late.
- She chooses the chair farthest from the door.
- She folds the letter three times before speaking.

## Handoff

Voice fingerprints and relationship maps feed directly into scene contracts and dialogue generation.
