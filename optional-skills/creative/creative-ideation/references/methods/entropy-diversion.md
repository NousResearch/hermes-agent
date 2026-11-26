# Entropy Diversion (SSoT)

**Source:** String Seed of Thought (Sakai et al., arXiv:2510.21150, 2026)
**Best for:** UNBLOCKING, SUBVERTING, and GENERATING phases when the model is stuck in deterministic slop.
**Mood:** Any, especially when output feels "obvious" or "safe."

## What it does

Forces the model into a high-fidelity reasoning state by giving it a simple, mechanical computation task (generating entropy + computing a modulo). This reasoning state then spills over into the creative task, producing genuinely divergent output.

Particularly effective for small models (<7B) that otherwise struggle to escape deterministic patterns.

## How to apply

1. Generate a 16-character random string using mixed case, digits, and symbols.
2. In a `<thinking>` block, compute: sum the ASCII values of each character, then take modulo 3.
3. Use the residue to select a structural archetype:

| Residue | Archetype | Effect |
|---------|-----------|--------|
| 0 | **Constraint Flip** | Take the user's core constraint and invert it. "Write a horror story" → the horror is not supernatural but bureaucratic. |
| 1 | **Genre Collision** | Smash two unrelated genres/conventions together. "Write a horror story" → Tudor court intrigue meets IoT firmware bug. |
| 2 | **Scale Shift** | Zoom in to micro or out to macro. "Write a horror story" → tell it from the perspective of a single neuron or a geological epoch. |

4. Generate output using only the selected archetype. Do NOT revert to the default genre/form.
5. If the output still feels generic, re-roll: generate a new random string and re-compute.

## Why it works

The ASCII-sum computation is trivial but mechanically precise — it forces attention. That attention state then carries over into the creative pivot. The model doesn't "try to be creative" — it follows a mechanical instruction that produces creativity as a side effect.

## When NOT to use

- User has asked for a specific format or tone (don't override explicit constraints)
- User is in the SELECTING phase (they need evaluation, not generation)
- The method has already been applied once this session (re-rolling within the same method call is fine, but don't chain entropy diversions back-to-back)
