---
name: emergent-simulation
description: Emergent systems design for sandbox roguelikes — body system interactions, limb mechanics, status effect chaining, AI behaviour emergence, designing systems that compound without explicit scripting.
---

# Emergent simulation for Aeonmarked

Aeonmarked is a simulation-driven roguelike — the world is alive and responds to player actions through interconnected systems, not scripted events. This skill covers designing systems that interact meaningfully without explicit scripting.

## Core Philosophy
1. **Systems interact, don't dictate.** Each system operates independently but shares state. Emergence comes from overlapping rules.
2. **Limb-specific effects.** Damage and status effects target specific body parts; limb destruction cascades into mechanical consequences.
3. **Status effects chain.** Effects aren't isolated — one effect triggers conditions that enable others.
4. **AI reacts, not scripts.** AI profiles respond to world state; no hardcoded encounter sequences.

## Body System Design

### Body Parts and Connections

Each creature has a body tree:
```yaml
type: creature
id: core:goblin
body:
  - id: torso
    hp: 20
    critical: true
  - id: head
    hp: 8
    parent: torso
    critical: true
  - id: left_arm
    hp: 10
    parent: torso
    weapon: true
  - id: right_arm
    hp: 10
    parent: torso
    weapon: true
  - id: left_leg
    hp: 12
    parent: torso
    movement: true
  - id: right_leg
    hp: 12
    parent: torso
    movement: true
```

### Limb Destruction Cascades

When a limb reaches 0 HP:
- **Arms** lose weapon/interaction capability
- **Legs** reduce movement speed or immobilize
- **Head** or **Torso** → creature dies (critical)
- Destroyed limbs may cause bleeding, pain, or secondary effects

## Status Effect Chaining

### Effect Categories
- **Direct:** Bleeding, poison, burning — each tick applies damage or debuff
- **Conditional:** Stunned, frozen, blinded — block specific actions
- **Environmental:** Slippery surface, smoke, toxic gas — affect terrain

### Chain Examples
```
Fire damage → burning → creature panics → moves into hazardous terrain → more damage
Bleeding → creature leaves blood trail → attracts predators → new encounter
Blinded → attacks hit wrong body part → unintended cascading damage
```

## AI Behaviour Emergence

### AI Profiles (Not Scripts)

AI profiles define **preferences and thresholds**, not action sequences:
```yaml
type: ai_profile
id: core:aggressive
personality: aggressive
target_priority: [weakest, closest]
flee_threshold: 0.3  # HP ratio
use_items_threshold: 0.5
aggression: 0.9
```

### Emergent Behaviours

These arise from system interactions:
- **Pack hunting** → multiple low-damage creatures gang up → limb destruction on player
- **Fleeing wounded creatures** → lead player into dangerous zones → trap by accident
- **Territorial disputes** → two hostile creatures fight each other → opportunistic player
- **Scavenging** → creatures pick up dropped items → unexpected encounters later

## Designing Emergent Systems

### Rules for Good Emergence
1. **Each system has one job.** Don't mix combat logic with world simulation.
2. **State is shared openly.** Any system reads any relevant state; no artificial barriers.
3. **Effects compose, don't override.** Multiple effects stack and interact; last one doesn't win.
4. **Test for edge cases, not just paths.** Emergent systems have unpredictable paths.

### Common Anti-Patterns
- **Scripted encounters.** "First time in this area, spawn X, say Y, do Z" kills emergence.
- **Overlapping status logic.** Don't hardcode "poison + fire = explosion" — let the systems handle it through generic mechanics.
- **Single-point-of-failure systems.** If one system crashes, it shouldn't bring down the simulation.

## Simulation Architecture Patterns

### Turn-Based Determinism

Every action is a deterministic state transition:
```
Input → Intent Queue → Simulation Step → World State Update → Output
```

This means:
- **Replays work.** Same seed + same inputs = same world state.
- **Undo works.** Roll back simulation steps without side-effect headaches.
- **AI doesn't "cheat."** It only reacts to what would be observable.

## Checklist for New Emergent Systems
- [ ] System has a single, clear responsibility
- [ ] System reads/writes only through public interfaces
- [ ] Effects compose correctly with existing status effects
- [ ] Limd-specific consequences defined
- [ ] AI can react to the system's state (not hardcoded)
- [ ] Deterministic behaviour across all paths
- [ ] Tested for cascading edge cases
