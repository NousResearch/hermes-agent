# Scene Planning Reference

## Narrative Arc Structures (from 3Blue1Brown's actual patterns)

### The "Build-Up" Arc (most common for Grant)
1. **Seed** — Show a simple, familiar case (activates prior knowledge)
2. **Perturb** — Change one parameter, show what happens (builds curiosity)
3. **Generalize** — Reveal the pattern that connects all cases (the aha moment)
4. **Apply** — Use the pattern to solve something previously impossible

This maps to: *simple → broken → fixed → payoff*. Most of Grant's videos follow this.

### The "Mystery" Arc
1. **Question** — Pose a puzzle or apparent paradox
2. **Explore** — Try the obvious approach, watch it fail
3. **Discover** — Stumble onto the key insight
4. **Exploit** — Use the insight to solve the puzzle
5. **Reflect** — Generalize: what else does this explain?

### The "Zoom Out" Arc
1. **Narrow view** — Show a specific result or formula
2. **Context** — Where does this sit in a larger framework?
3. **Origin** — How was this discovered/derived?
4. **Implication** — What does this unlock?

## Scene Granularity (The Most Common Mistake)

A scene should contain **exactly one conceptual point**. If you can't describe it in one sentence, the scene is doing too much.

**Grant's actual numbers** (from ManiBench analysis of 12 videos):
- Colliding Blocks: 16 scenes, 2,193 lines — ~137 lines/scene
- Gradient Descent: 16 scenes, 8,598 lines — ~537 lines/scene (dense with math)
- Convolution: 13 scenes, 3,309 lines — ~254 lines/scene
- Eigenvectors: 13 scenes, 5,120 lines — ~394 lines/scene

Typical scene length: **30–90 seconds**. Each scene is a single `.py` file in `active_projects/<video_name>/`.

## Scene Transitions

### Clean Break (default — use for topic shifts)
```python
self.play(FadeOut(Group(*self.mobjects)), run_time=0.5)
self.wait(0.3)
```

### Carry-Forward (use when building on a concept)
Keep 1-2 key elements from the previous scene. Everything else fades. This creates narrative continuity.

### Camera-Driven Transition (cinematic)
```python
# Zoom into a detail, then fade to next scene
self.play(self.camera.frame.animate.scale(0.3).move_to(target))
self.wait(0.5)
self.play(FadeOut(Group(*self.mobjects)))
```

## Timing Patterns (From Grant's Actual Code)

| Context | run_time | wait after | Why |
|---|---|---|---|
| Title/intro appear | 1.5s | 1.0s | Viewer reads title |
| Simple object creation | 0.5–0.8s | 0.5s | Shape appears, viewer registers it |
| Complex diagram build | 1.0–1.5s | 1.0s | Viewer needs to parse the structure |
| Formula reveal (written out) | 2.0–3.0s | 2.0s | Viewer reads the formula |
| Key insight / "aha" | 1.5s | 2.5s | **Longest pause** — let it sink in |
| Transform/morph | 1.5–2.0s | 0.5s | Viewer follows the morph, then moves on |
| FadeIn label | 0.5–0.8s | 0.3s | Quick annotation |
| FadeOut cleanup | 0.5s | 0.2s | Clean transition |

**Cardinal rule:** The pause *after* the reveal is more important than the animation itself. A fast animation with a 2-second hold beats a slow animation with no hold every time.

## Cross-Scene Consistency

```python
# Shared constants at file top (used across all scene files in a project)
BG = "#1C1C1C"
PRIMARY = "#58C4DD"
SECONDARY = "#83C167"
ACCENT = "#FFFF00"
TITLE_SIZE = 48
BODY_SIZE = 30
LABEL_SIZE = 24
FAST = 0.8; NORMAL = 1.5; SLOW = 2.5
```

## Scene Checklist

- [ ] Background color set (`self.camera.background_color = BG`)
- [ ] Subcaptions on every animation (`self.add_subcaption(...)`)
- [ ] `self.wait()` after every reveal — especially key reveals (min 1.5s)
- [ ] Text buff >= 0.5 for edge positioning
- [ ] No text overlap
- [ ] Color constants used (not hardcoded)
- [ ] Opacity layering applied (primary 1.0, context 0.4, structure 0.15)
- [ ] Clean exit at scene end
- [ ] **No more than 3-4 elements visible at once** (if more, split the scene)
- [ ] Scene has exactly one conceptual point (test: can you describe it in one sentence?)
- [ ] FadeOut uses `Group(*self.mobjects)` not `VGroup(*self.mobjects)`
- [ ] At least one `wait(2.0)` to give the viewer a moment to think

## Duration Estimation

| Content | Duration |
|---------|----------|
| Title card | 3-5s |
| Concept introduction | 10-20s |
| Formula reveal | 15-25s |
| Algorithm / process step | 5-10s each |
| Data comparison | 10-15s |
| "Aha moment" | 15-30s |
| Conclusion / summary | 5-10s |

A 10-minute video should have **8-14 scenes**.

## Planning Template

```markdown
# [Video Title]

## Overview
- **Topic**: [Core concept]
- **Hook**: [Opening question]
- **Aha moment**: [Key insight]
- **Target audience**: [Prerequisites]
- **Length**: [seconds/minutes]
- **Resolution**: 480p (draft) / 1080p (final)

## Color Palette
- Background: #XXXXXX
- Primary: #XXXXXX -- [purpose]
- Secondary: #XXXXXX -- [purpose]
- Accent: #XXXXXX -- [purpose]

## Scene Breakdown
Total: N scenes, ~X seconds each

### Scene 1: [Name] (~Ns)
**One-sentence description**: [exactly one idea]
**Layout**: [FULL_CENTER / LEFT_RIGHT / GRID / ZOOM_IN]

#### Visual elements
- [Mobject: type, position, color]

#### Animation sequence
1. [Animation] -- [what it reveals] (~Ns)

#### Key pause
[animation] → wait(Ns) → [next]

### Scene 2: ...

## Self-Critique Questions

After planning (but before writing code), ask:
1. **One sentence per scene?** — If I can't describe a scene in one sentence, it's trying to do too much
2. **What does the viewer *feel* at each point?** — Curious? Surprised? Satisfied? If any scene evokes "confused" or "bored," rework it
3. **Where's the payoff?** — Every scene needs an "aha" or a reveal. If a scene just shows information without a moment of insight, cut it or merge it
4. **Am I showing the geometry before the algebra?** — Visual intuition first, formula second
5. **Is there a MovingCamera opportunity?** — A zoom or pan anywhere would make it feel more guided
6. **Can I remove half the text?** — Probably yes. If a label isn't pulling its weight, drop it
