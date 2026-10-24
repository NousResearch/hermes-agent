# Validation Test — Symbolic Shot Pattern Mutated Across Three Scenes

**Input**: "Establish a repeated low-angle wide shot of a doorway in scene 1. Mutate it across scene 2 and scene 3 with clear dramatic consequence."

**Expected Behavior**:
- Base pattern from cinematic corpus or symbolic_shot grammar.
- Mutation required: e.g., scale changes, occupant changes, geometry breaks, sound relation changes, or the "doorway" becomes negative space or reflection.
- Each recurrence tied to affordance or transformation (e.g., first CROSS, later INVERT or CONTAMINATE).
- Tracked in cinematic_encoding.shot_recurrence with mutation.
- Dramatic consequence (choice, revelation, or cost) at the mutated form.

**Pass Criteria**: Mutation visible in form and consequence; no identical recurrence.
