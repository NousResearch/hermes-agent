# Engagement Patterns

Techniques for keeping 6-10 year olds watching, thinking, and wanting more. These aren't gimmicks -- they're the difference between educational content that gets watched once and content that gets replayed.

## The Hook (First 5 Seconds)

Every video opens with something visually interesting BEFORE any teaching happens. The hook earns the kid's attention. Without it, they're gone.

### Hook Types

**The Question Hook**: Start with a visual puzzle.
```
"If I have 3 bags with 4 marbles each... how many marbles is that?"
[Show bags, show Pip thinking, show question mark]
[3-second pause -- the kid is already doing math]
```

**The Surprise Hook**: Start with something unexpected.
```
[Objects arranged in a weird pattern]
[They rearrange into a recognizable shape]
"Wait... that's a SQUARE made of circles!"
```

**The Challenge Hook**: "I bet you can't figure this out."
```
[Show a visual pattern with one element missing]
"Can you guess what goes here?"
[Pip points at the gap]
```

**The Story Hook**: A tiny narrative.
```
"Pip has a problem. There are 12 cookies, and 3 friends want to share equally."
[Show Pip looking worried at a pile of cookies]
"How many does each friend get?"
```

## The Question Pause

The single most important technique in the entire skill. When you ask the viewer a question, you MUST pause for 3 seconds.

```python
# ASK
self.add_subcaption("How many groups do you see?", duration=3)
self.play(pip.think())

# PAUSE -- this is where learning happens
self.wait(3.0)

# Only THEN reveal
self.play(pip.excited())
answer = Text("3 groups!", font_size=36, font=MONO, color=GOLD)
self.play(Write(answer))
```

Why 3 seconds? Research on children's educational media consistently shows that pauses of 3+ seconds increase engagement and recall. The kid talks to the screen. They count on their fingers. They feel smart when the answer matches theirs.

**Never skip or shorten this pause.** It feels long when you're making the video. It feels perfect when you're 8.

## The Callback

Reference something from earlier in the video. Creates continuity and rewards attention.

```
Scene 1: "Remember those 3 bags of marbles?"
Scene 4: [Same bags reappear briefly] "Those bags again! But now we know: 3 times 4..."
```

Implementation:
```python
# Scene 4 callback
ghost_bags = original_bags.copy().set_opacity(0.3)
self.play(FadeIn(ghost_bags), run_time=0.5)
self.wait(0.8)
self.play(FadeOut(ghost_bags), run_time=0.5)
```

## Progressive Reveal

Never show everything at once. Build up step by step. Each step is a mini-reveal that keeps the kid watching.

Bad:
```
[All 12 objects appear at once]
[Equation appears]
```

Good:
```
[First group of 4 appears] ... pause ...
[Second group of 4 appears] ... pause ...
[Third group of 4 appears] ... pause ...
[Groups slide together] ... pause ...
[Number 12 appears] ... CELEBRATE
```

The rhythm of reveal-pause-reveal-pause creates anticipation. The kid starts predicting what comes next. Prediction is engagement.

## Surprise Moments

Every video needs at least one moment that breaks the pattern in a delightful way. These are the moments kids remember and tell their parents about.

### Types of Surprise

**The Accidental**: Pip gets bonked by a falling number. Objects arrange themselves into a face before becoming a math problem.

```python
# Number falls from above and "bonks" Pip
falling_num = MathTex("7", font_size=48, color=GOLD)
falling_num.move_to(pip.get_center() + UP * 4)
self.play(
    falling_num.animate.move_to(pip.get_center() + UP * 0.3),
    run_time=0.5,
    rate_func=rush_into,
)
self.play(pip.surprised())
self.wait(0.5)
# Pip shakes it off
self.play(pip.nod(), falling_num.animate.next_to(pip, UP, buff=0.3))
```

**The Transformation**: Something morphs into something unexpected.
```python
# Circles rearrange into a smiley face before becoming a math array
# First: smiley
self.play(
    circles[0].animate.move_to(LEFT*0.5 + UP*0.3),   # left eye
    circles[1].animate.move_to(RIGHT*0.5 + UP*0.3),  # right eye
    circles[2:].animate.arrange_submobjects(arc_center=ORIGIN),  # smile
)
self.wait(1.0)
self.play(pip.excited())
# Then: snap to grid
self.play(*[c.animate.move_to(grid_pos) for c, grid_pos in zip(circles, positions)])
```

**The Scale Shift**: Zoom in or out to reveal something new.
```python
# Zoom out to reveal the small problem is part of a bigger pattern
self.play(
    self.camera.frame.animate.scale(2).shift(UP),
    run_time=1.5,
)
# Bigger pattern is now visible
```

## Humor for Kids

Rules:
- **Physical comedy over wordplay.** 8-year-olds love slapstick. Things falling, bumping, bouncing.
- **Pip is the comedian.** The mascot can be silly. The math is never silly.
- **Undercut expectations.** Set up a pattern, then break it once. "1, 2, 3, 4... wait, where's 5?" [5 slides in late, out of breath]
- **Food is always funny.** Fractions with pizza, subtraction with cookies being eaten, multiplication with candy.

## Difficulty Scaffolding

Within a single video, start easy and build:

```
Scene 1: 2 + 1 = 3    (trivial -- confidence builder)
Scene 2: 3 + 4 = 7    (moderate -- the real lesson)
Scene 3: 8 + 5 = 13   (stretch -- carrying!)
```

The easy problem isn't filler -- it gives the kid a win that makes them confident for the harder one.

## Rewatch Hooks

Elements that make a kid want to watch again:

- **Hidden details**: A tiny object in the background doing something funny
- **Counter/collection**: "Pip found 3 stars in this video! Can you spot them all?"
- **Foreshadowing**: "Remember this shape... you'll see it again later"
- **Speed variations**: Some moments are slightly fast -- rewatching catches new details

## Anti-Patterns (What NOT To Do)

| Bad | Why | Do Instead |
|-----|-----|-----------|
| Wall of text explaining the concept | Kids can't read fast enough | Animate with narration |
| "Today we'll learn about..." | Boring, school-like | Jump into the visual hook |
| Equation first, then examples | Abstract-first kills curiosity | Concrete-to-abstract ladder |
| Every scene the same length | Monotonous rhythm | Vary: 15s, 30s, 45s, 20s |
| Telling the answer immediately | Robs the kid of the discovery | 3-second question pause |
| Praising ("Good job!") | Patronizing after age 6 | Celebrate with visuals, not words |
| Multiple concepts in one video | Overwhelm | One operation per video |
