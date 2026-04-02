---
name: ascii-cinema
description: Create browser-playable ASCII cinema explainers — living-agent scenes, scene-aware backgrounds, captions, and interactive controls in a single HTML file.
version: 1.0.0
author: Mustafa Sarac
license: MIT
metadata:
  hermes:
    tags: [ASCII, Cinema, HTML, Animation, Storytelling, Explainers, Creative]
    related_skills: [ascii-video, ascii-art, excalidraw]
---

# ASCII Cinema

ASCII Cinema is for browser-playable ASCII storytelling.

Use it when the user wants:
- an ASCII art explainer embedded in HTML
- a "living agent" visual instead of a static diagram
- a narrative walkthrough with multiple scenes
- a retro-terminal microsite that actually teaches something
- an animated ASCII landing page for a tool, workflow, or concept

This skill is not for generating MP4/GIF video files.
That is what `ascii-video` is for.

This skill is for building a self-contained HTML/CSS/JS artifact that behaves like a small cinematic experience in the browser.

## When to Use

Use this skill when the task involves one or more of these:

- the user explicitly asks for ASCII cinema, ASCII explainer, ASCII landing page, or terminal-style storytelling
- the page should have a living animated agent
- the page should show scene-by-scene progression
- the background should change with the concept being explained
- the output should be shareable as a public microsite
- the goal is education, launch marketing, or product storytelling rather than pure decoration

Do not use this skill when:
- the user needs a real rendered video file → use `ascii-video`
- a static diagram is enough → use `excalidraw`
- a simple ASCII banner is enough → use `ascii-art`

## Core Principle

ASCII Cinema is not decoration.
It is narrative interface.

Every scene must answer:
- what is happening?
- why does it matter?
- what should the viewer understand next?

A good ASCII cinema piece has:
- a clear story arc
- scene-aware backgrounds
- a living focal subject
- restrained controls
- readable captions
- rhythm
- visual coherence from start to finish

## Output Shape

Default output is a single self-contained HTML file with:
- inline CSS
- inline JavaScript
- no build step required
- browser-playable animation loop
- keyboard/mouse-safe controls
- mobile-friendly layout

Preferred structure:

1. hero / framing section
2. cinematic ASCII section
3. supporting explanation sections
4. closing transformation or CTA

## ASCII Cinema Design Rules

### 1. Keep the agent alive
The focal subject should not be static.
Give it at least two or three kinds of motion, for example:
- blink
- bob
- pulse
- trail
- scan beam
- orbit motion

The motion should fit the scene.

### 2. Backgrounds must be scene-aware
Do not use one generic terminal background for the whole piece.
Each scene should have a background language that matches the concept:
- chaos scene → prompt noise, drifting fragments, broken message streams
- context scene → memory blocks, rails, anchors, stable nodes
- inspect scene → repo trees, scans, crosshairs, logs
- planning scene → boxes, arrows, dependencies, pathways
- execution scene → tool panes, commands, browser frames, file grids
- automation scene → rings, cycles, scheduler dots, orbits

### 3. Captions explain, not merely label
Each scene should have:
- short scene title
- one explanatory caption
- one or two compact metadata tags at most

Do not overload the viewer.

### 4. Controls must be restrained
Use only the controls that genuinely improve use:
- play/pause
- restart
- optional speak/narrate
- optional scene jump list

Avoid dashboard clutter.

### 5. The layout must preserve focus
The viewer's attention should go here, in this order:
1. main cinematic frame
2. current scene title/caption
3. scene progression / navigation
4. playback controls

If controls compete with the scene, simplify them.

### 6. The piece should be screenshot-friendly
The final page should work both as:
- a live microsite
- a source for still screenshots or carousel slices

That means:
- strong section boundaries
- clean spacing
- no fragile hover-only meaning
- no dependence on hidden state for understanding

## Recommended Workflow

### Step 1: Define the narrative arc
Before writing code, define:
- audience
- core message
- transformation
- scene list
- final takeaway

A useful default story arc:
1. wrong model
2. reframe
3. context
4. inspect
5. plan
6. act with tools
7. preserve and automate
8. transformation / CTA

### Step 2: Map each scene to a visual grammar
For each scene decide:
- background motif
- agent behavior
- caption title
- caption body
- accent color/mood

Do not improvise scene visuals after coding starts.

### Step 3: Build the ASCII renderer
Use a grid-based renderer in JavaScript.
Recommended primitives:
- `makeGrid()`
- `setChar()`
- `writeText()`
- `writeCentered()`
- `overlay()`
- per-scene draw functions like `drawChaos()`, `drawInspect()`, `drawOrbit()`

Render into a `<pre>` block with `white-space: pre` and monospace font.

### Step 4: Separate scene logic from layout logic
Keep these distinct:
- scene data array
- renderer functions
- UI update functions
- layout/style layer

The page should be editable without touching the scene engine unnecessarily.

### Step 5: Add restrained playback features
Recommended:
- autoplay loop
- play/pause
- restart
- scene counter
- progress bar
- optional speech synthesis

Optional means optional.
Use only if it improves comprehension.

### Step 6: Verify manually
Check:
- no duplicated IDs
- JS syntax valid
- no overflow in cinematic section
- scene list readable
- mobile layout sane
- playback controls not dominant
- captions readable at a glance
- every scene background actually matches the concept

## Scene Schema

A useful scene object shape:

```js
{
  id: 'inspect-state',
  short: 'Inspect',
  title: 'A good agent inspects before acting.',
  themeLabel: 'repo map',
  action: 'Inspect current reality',
  background: 'directory trees, document rails, scan lines and map grids',
  narration: 'Instead of guessing, the agent reads the current state first.',
  duration: 7000,
  mode: 'inspect',
  pulse: 'scanning',
  mood: 'analytic'
}
```

## HTML/CSS Guidance

### Typography
Use a strong contrast between:
- elegant display font for large headings
- monospace for terminal / labels / scene chrome
- clean sans serif for body copy

### Layout
Preferred structure for the cinematic section:
- section header
  - left: title and framing copy
  - right: minimal playback controls
- content row
  - left: cinematic terminal
  - right: compact scene navigation

### Avoid
- giant sidebars
- long control explanations inside the cinematic block
- duplicate playback controls in multiple places
- excessive metadata chips
- random neon cyberpunk aesthetics unless the story calls for it

## Pitfalls

### Pitfall 1: One background for all scenes
This makes the piece feel decorative instead of narrative.

### Pitfall 2: Too many controls
ASCII cinema should feel composed, not like a control dashboard.

### Pitfall 3: Static focal subject
If the agent never feels alive, the entire illusion collapses.

### Pitfall 4: Explanations that are too short or too long
Too short = confusing.
Too long = heavy and slow.

### Pitfall 5: Scene list that overwhelms the page
Navigation should support the cinematic frame, not compete with it.

### Pitfall 6: Treating ASCII as the whole experience
ASCII is the medium, not the message.
The story still matters more.

## Verification Checklist

Before finishing, verify:
- [ ] The main scene renders correctly in a browser
- [ ] Every scene changes both content and background logic
- [ ] The agent appears alive
- [ ] Captions teach the process clearly
- [ ] The controls are minimal and useful
- [ ] The cinematic section feels premium, not gimmicky
- [ ] The output can be deployed as a public static site
- [ ] JS passes syntax validation
- [ ] No duplicate element IDs exist

## Deliverable Standard

A good ASCII Cinema artifact should feel like:
- a tiny product demo
- a launch asset
- an interactive explainer
- a cinematic terminal page someone wants to share

Not just a cute ASCII animation.
