# ASCII Cinema Scene Grammar

Use this reference after defining the narrative arc and before writing the
artifact. A scene is coherent when meaning, motion, background, and caption all
teach the same idea.

## Four Aligned Layers

### Meaning layer

What should the viewer understand after this scene?

### Motion layer

How does the focal subject express the concept through blink, bob, pulse, scan,
trail, orbit, or another restrained behavior?

### Background layer

What ambient visual language reinforces the concept without competing with the
focal subject?

### Caption layer

Why does this moment matter, and what should the viewer understand next?

If one layer communicates a different concept, redesign the scene before coding.

## Scene Pattern Catalog

### Wrong model / chaos

- Meaning: isolated prompting creates noise rather than leverage.
- Motion: unstable pulse, interrupted path, or repeated reset.
- Background: drifting prompts, broken streams, incomplete fragments.
- Caption rhythm: name the cost, then introduce the need for a new model.

### Context / memory

- Meaning: durable context gives action continuity.
- Motion: steady pulse or anchoring movement.
- Background: blocks, rails, linked nodes, stable coordinates.
- Caption rhythm: show what becomes possible when state persists.

### Inspect / analysis

- Meaning: current reality must be read before it is changed.
- Motion: scan beam, sensor sweep, or moving crosshair.
- Background: repository trees, logs, maps, evidence handles.
- Caption rhythm: contrast evidence with guessing.

### Plan / structure

- Meaning: dependencies and gates make execution controllable.
- Motion: measured movement from node to node.
- Background: boxes, arrows, bounded routes, approval gates.
- Caption rhythm: show how sequence reduces blast radius.

### Act / tools

- Meaning: tools produce evidence, not just prose.
- Motion: engaged pulse, command trail, or focused handoff.
- Background: terminal surfaces, files, browser frames, test signals.
- Caption rhythm: identify the action and the read-back that proves it.

### Preserve / forge

- Meaning: a successful solution becomes a reusable system.
- Motion: assembly, joining, or crystallization.
- Background: modules, templates, lattices, structured sparks.
- Caption rhythm: show the conversion from one-off work to leverage.

### Automate / orbit

- Meaning: repeatable work can run with bounded oversight.
- Motion: orbit, recurring marker, or controlled cycle.
- Background: rings, scheduler dots, loops, checkpoints.
- Caption rhythm: distinguish automation from uncontrolled autonomy.

### Transformation / release

- Meaning: the viewer now sees or operates differently.
- Motion: calm forward movement or resolved pulse.
- Background: simplified field, open path, coherent system map.
- Caption rhythm: state the new capability and next action.

## Caption Rhythm

For each scene use:

- a short declarative title;
- one or two sentences of explanation;
- at most two compact metadata tags;
- language that advances the story rather than labels the drawing.

Read all captions in sequence without the visuals. They should still form a
complete argument.

## Motion Contract

### Timer ownership

Playback has one owner and at most one active timer. Play, pause, restart, scene
selection, and wrap behavior must all update the same state. Restart clears the
existing timer before returning to the first scene. When the implementation also
uses `requestAnimationFrame`, keep its handle and call `cancelAnimationFrame` on
restart or teardown. Verification should exercise rapid toggles and assert that no
more than one playback timer or animation loop remains active.

### Reduced motion

When `prefers-reduced-motion: reduce` is active:

- autoplay remains off;
- nonessential transitions are disabled;
- the visible control state says paused when no timer exists;
- direct scene selection, restart, and manual navigation remain available.

Reduced motion changes delivery, not access to the narrative.

## Interaction Contract

### Keyboard

Use native buttons for every control. Enter and Space activation, visible focus,
and `aria-current` for the active scene should work without custom key emulation.

### Pointer

Targets must remain large enough to activate on mobile. Rapid play/pause and
scene selection must not create duplicate timers or skip state.

## Responsive Contract

### Narrow screens

Check both page overflow and the width of the longest rendered ASCII row. A
container can fit while the final glyph is clipped. When a row is too wide,
reduce grid density or font size within a readable bound.

Keep intentionally wide comparisons inside a labeled horizontal-scroll region.
The document body itself must not scroll sideways.

## Visual Restraint

Prefer:

- one dominant focal subject;
- one background language per scene;
- one readable caption block;
- a coherent palette across the full arc;
- motion that serves meaning.

Avoid:

- several simultaneous ideas in one frame;
- oversized control panels;
- random particles without narrative purpose;
- color as the only active-state signal;
- tiny text used merely to force the grid to fit.

## Scene Review Card

Before accepting a scene, answer:

```text
MEANING: What is taught?
FOCAL MOTION: What feels alive?
BACKGROUND: What reinforces the idea?
CAPTION: Why does it matter?
CONTROL STATE: What happens if playback is paused?
NARROW SCREEN: Is the complete glyph row visible?
```

The scene is ready only when all six answers agree with the intended lesson.
