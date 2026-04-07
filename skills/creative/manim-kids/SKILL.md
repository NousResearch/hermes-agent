---
name: manim-kids
description: "Animated math and science videos for elementary-age kids (ages 6-10) using Manim Community Edition. Mascot-driven, voice-first, pattern-discovery pedagogy designed for visual learners and neurodivergent kids who need more than worksheets. Produces short, vivid, dopamine-aware explainer clips that make abstract concepts physical and obvious."
version: 1.0.0
---

# Manim Kids -- Animated Learning for Young Minds

## Who This Is For

Kids ages 6-10. Especially the ones the classroom isn't reaching -- the visual thinkers, the pattern-spotters, the kids who zone out during worksheets but light up when you show them WHY something works. Neurodivergent kids. Gifted kids. Kids who need to SEE the math before they can DO the math.

These videos don't replace a teacher. They give a kid a private "aha moment" they can replay.

## Creative Standard

**This is a show, not a lecture.** Every video should feel like a favorite YouTube channel, not a classroom smartboard. The bar is "my kid asks to watch another one."

**The mascot is the teacher.** Pip (a friendly bouncing circle with expressive dot-eyes) appears in every video. Pip points at things, reacts to reveals, celebrates correct answers, and occasionally gets confused (on purpose) so the kid can feel smarter than the character. Pip is never condescending. Pip is a curious friend who's figuring things out alongside the viewer.

**Voice-first, text-second.** Eight-year-olds read slowly. Narration carries the content. On-screen text is for labels, equations, and punchlines -- never paragraphs. If you have to choose between a sentence of text and an animation, animate.

**30-second rule.** No scene runs longer than 30-45 seconds. Each scene is one idea. A "video" is 3-8 scenes, total runtime 2-5 minutes. Attention is precious -- earn it every half-minute.

**Show, don't tell. Then ask.** Never explain a concept with words first. Show the visual pattern. Let it sit. THEN name it. The equation is the last thing, never the first.

**Pattern discovery over instruction.** Don't say "3 times 4 equals 12." Show three groups of four objects. Show them merge. Show the number 12 appear. Let the kid's brain do the connecting. The "aha" is the product -- manufacture it carefully.

**Surprise and humor.** Every video needs at least one unexpected moment. Pip gets bonked by a falling number. Objects arrange themselves into a smiley face before becoming a math problem. A fraction pizza gets "eaten" with a cartoon chomp sound effect label. Delight is a learning accelerator.

**Reward the brain.** Celebration animations after every reveal: confetti bursts, Pip doing a spin, stars radiating outward, a satisfying color wash. Not patronizing -- genuinely satisfying. Think video game feedback, not gold star stickers.

## Design Principles

### 1. Concrete-to-Abstract Ladder

Every concept gets four passes, always in this order:

```
REAL OBJECT  -->  SIMPLE SHAPE  -->  DIAGRAM  -->  SYMBOL
  (pizza)        (colored circle)   (fraction bar)  (1/2)
```

Never skip a rung. Never start at the symbol. The whole point is building the bridge from physical intuition to mathematical notation.

### 2. Color Is Meaning

Operations and concepts get permanent colors. Every video, every time:

| Concept | Color | Hex |
|---------|-------|-----|
| Addition / combining | Blue | `#58C4DD` |
| Subtraction / removing | Coral red | `#FF6B6B` |
| Multiplication / groups | Green | `#83C167` |
| Division / splitting | Purple | `#B07CD8` |
| Equals / result | Gold | `#FFD93D` |
| Question / unknown | Orange | `#FF9F43` |
| Pip (mascot) | White with blue outline | `#FFFFFF` / `#58C4DD` |

A kid who watches ten of these videos will start FEELING that green means groups before they consciously know it. That's the goal.

### 3. Spatial Consistency

Math builds left-to-right, top-to-bottom. Inputs on the left, outputs on the right. The thing being explained is always center stage. Supporting context lives at the edges. Never surprise a kid with something appearing behind where they were looking.

### 4. Motion Vocabulary

Consistent animation meanings:

| Motion | Meaning |
|--------|---------|
| **Grow from center** | New concept appearing |
| **Slide together** | Combining / adding |
| **Slide apart** | Separating / subtracting |
| **Replicate + arrange** | Multiplication (groups forming) |
| **Split with line** | Division / fractions |
| **Transform/morph** | "This IS that" (equivalence) |
| **Bounce** | Emphasis, counting |
| **Fade to dim** | "Remember this is here, but look over here now" |
| **Confetti burst** | Celebration / correct answer |

### 5. The Pip System

Pip is built from simple Manim primitives -- a `Circle` body, two `Dot` eyes, and optional expression arcs. See `references/mascot.md` for full implementation.

Pip states:
- **Neutral** -- watching, present
- **Excited** -- eyes widen (scale up), small bounce
- **Thinking** -- one eye slightly up, head tilt (rotate)
- **Surprised** -- eyes big, jump back
- **Celebrating** -- spin + confetti
- **Pointing** -- arrow extends from body toward target

Pip lives in the bottom-right quadrant by default. Never overlaps the main content. Moves to point at things, then returns home.

### 6. Narration Pacing

| Moment | Pace | Pause after |
|--------|------|-------------|
| Introducing something new | Slow, clear | 2.0s |
| Counting objects | Rhythmic, bouncy | 0.5s per item |
| Asking a question | Slightly slower, rising tone | 3.0s (let them think!) |
| Revealing the answer | Normal | 1.5s |
| "Did you see that?" callback | Warm, conspiratorial | 1.0s |
| Celebration | Upbeat, quick | 1.0s |

The 3-second pause after questions is sacred. That's where learning happens -- in the kid's head, not on the screen.

### 7. Repetition With Variation

The same concept shown three ways:

1. **Physical metaphor** (apples, pizza, toy cars -- concrete objects the kid knows)
2. **Abstract shapes** (colored circles, blocks, bars)
3. **Mathematical notation** (numbers, operators, equations)

Each pass is a new scene with different visuals but the same underlying structure. The kid's brain pattern-matches across representations. That IS understanding.

## Stack

Same as `manim-video`:

| Layer | Tool | Purpose |
|-------|------|---------|
| Core | Manim Community Edition | Scene rendering, animation engine |
| Math | LaTeX (texlive/MiKTeX) | Equation rendering via `MathTex` |
| Video I/O | ffmpeg | Scene stitching, format conversion |
| TTS (optional) | ElevenLabs / edge-tts | Warm, friendly narration voice |

## Pipeline

```
CONCEPT --> LADDER --> SCRIPT --> RENDER --> STITCH --> NARRATE (optional)
```

1. **CONCEPT** -- Identify the math/science concept and the target misconception
2. **LADDER** -- Map the concrete-to-abstract progression (4 rungs)
3. **SCRIPT** -- Write `script.py` with short scenes, Pip appearances, rewards
4. **RENDER** -- `manim -ql script.py` for draft, `-qm` for delivery (720p is fine for kids)
5. **STITCH** -- ffmpeg concat into final clip
6. **NARRATE** (optional) -- Add TTS voiceover synced to animations

## Project Structure

```
project-name/
  concept.md             # What we're teaching and why
  script.py              # All scenes
  concat.txt             # ffmpeg scene list
  final.mp4              # Output
  media/
    videos/script/720p30/
```

## Scene Templates

### Template: Concept Introduction (Scene 1 of any video)

```python
class Scene1_Hook(Scene):
    def construct(self):
        self.camera.background_color = BG

        # Pip enters
        pip = Pip().to_corner(DR, buff=0.8)
        self.play(pip.enter(), run_time=0.8)
        self.wait(0.5)

        # Show the real-world object / situation
        # (customize: pizza, apples, toy cars, etc.)
        objects = VGroup(*[create_object() for _ in range(N)])
        objects.arrange_in_grid(rows=R, buff=0.4).move_to(ORIGIN)
        self.play(LaggedStart(*[GrowFromCenter(o) for o in objects], lag_ratio=0.1))
        self.wait(1.0)

        # Pip reacts
        self.play(pip.look_at(objects))
        self.wait(0.5)

        # Optional: narration cue
        self.add_subcaption("Hmm, how many do we have here?", duration=3)
        self.play(pip.think())
        self.wait(3.0)  # LET THE KID COUNT
```

### Template: Reveal + Celebration (Final scene)

```python
class SceneN_Reveal(Scene):
    def construct(self):
        self.camera.background_color = BG

        pip = Pip().to_corner(DR, buff=0.8)
        self.add(pip)

        # Show the equation forming from the visual
        equation = MathTex(r"3 \times 4 = 12", font_size=56, color=GOLD)
        self.play(Write(equation), run_time=2.0)
        self.wait(1.5)

        # Pip celebrates
        self.play(pip.celebrate())

        # Confetti burst
        confetti = create_confetti(equation.get_center())
        self.play(*[GrowFromCenter(c) for c in confetti], run_time=0.5)
        self.wait(0.3)
        self.play(*[FadeOut(c, shift=UP*0.5) for c in confetti], run_time=1.0)

        self.wait(1.0)
        self.play(FadeOut(Group(*self.mobjects)))
```

## Visual Specifications

### Typography

```python
MONO = "Menlo"  # or "DejaVu Sans Mono"

# Sizes tuned for young readers
TITLE_SIZE = 56    # Big, bold, unmissable
LABEL_SIZE = 36    # Object labels, counts
EQUATION_SIZE = 48 # Final equations
CAPTION_SIZE = 28  # Subcaptions (if any on-screen text)
```

All text uses monospace. Minimum font_size=24 for readability on tablets/phones.

### Color Palette

```python
# Background
BG = "#1A1A2E"       # Deep navy -- easy on eyes, high contrast

# Operation colors (NEVER change these across videos)
BLUE = "#58C4DD"     # Addition
RED = "#FF6B6B"      # Subtraction
GREEN = "#83C167"    # Multiplication
PURPLE = "#B07CD8"   # Division
GOLD = "#FFD93D"     # Results / equals
ORANGE = "#FF9F43"   # Questions / unknowns

# Pip
PIP_BODY = "#FFFFFF"
PIP_OUTLINE = "#58C4DD"
PIP_EYES = "#1A1A2E"

# Supporting
DIM = 0.3            # Opacity for background/context elements
BRIGHT = 1.0         # Opacity for focus elements
```

### Animation Timing

| Animation | run_time | wait() after |
|-----------|----------|-------------|
| Pip enters | 0.8s | 0.5s |
| Object appears | 0.6s | 0.3s |
| Objects arrange | 1.2s | 1.0s |
| Key reveal | 2.0s | 2.0s |
| Question pause | -- | 3.0s |
| Answer reveal | 1.5s | 1.5s |
| Celebration | 0.5s | 1.0s |
| Scene cleanup | 0.5s | 0.3s |

### Rendering

**Draft**: `manim -ql` (480p 15fps) -- for iteration
**Delivery**: `manim -qm` (720p 30fps) -- final output for kids
**Never -qh** unless specifically requested. Kids watch on tablets. 720p is perfect.

## Grade-Level Concept Map

See `references/elementary-math.md` for the full concept progression, but here's the K-3 core:

### Third Grade Focus

| Concept | Visual Metaphor | Key Animation |
|---------|----------------|---------------|
| Multiplication as groups | Groups of objects forming | Slide-together, count bounce |
| Division as sharing | Objects dealing into piles | Split animation |
| Fractions (1/2, 1/3, 1/4) | Pizza, chocolate bar, pie | Cut line, color halves |
| Area (length x width) | Grid of square tiles filling | Tile-by-tile fill |
| Number line | Road with mileposts | Pip walking/hopping along |
| Rounding | Number line with "gravity wells" | Numbers sliding to nearest 10 |
| Telling time | Analog clock with moving hands | Hands sweeping, Pip pointing |
| Perimeter | Ant walking around a shape | Traced path with counter |

## Reference Documents

Load detailed implementation guides on demand:

| Reference | Contents |
|-----------|----------|
| `mascot` | Pip implementation -- Circle body, Dot eyes, expression states, enter/exit/point/celebrate methods, positioning |
| `rewards` | Celebration animations -- confetti, star burst, color wash, Pip spin, sound effect cues |
| `concrete-to-abstract` | The 4-rung ladder pattern with examples for each math operation |
| `elementary-math` | Full K-5 concept map with visual metaphors and animation patterns per topic |
| `engagement` | Hooks, humor patterns, surprise reveals, question-pause technique, callback structure |
| `voice-and-pacing` | Narration-first workflow, TTS integration, pacing tables, script writing for young audiences |

## Critical Implementation Notes

### From manim-video (still apply)

- Raw strings for LaTeX: `MathTex(r"\frac{1}{2}")`
- `buff >= 0.5` on all `.to_edge()` calls
- `self.camera.background_color = BG` in every scene
- `Group(*self.mobjects)` for mixed-type FadeOut (not VGroup)
- Use `ReplacementTransform` to swap, not `Write` on top
- Monospace fonts only (Pango kerning bug with proportional fonts)

### Kids-specific

- **No dense text.** Maximum 6 words on screen at once. Prefer 2-3.
- **No abstract-first.** Never show an equation without first showing what it represents visually.
- **Count everything.** When objects appear, bounce-count them. Numbers appearing one by one as Pip "counts" is deeply satisfying.
- **Pause after questions.** 3 seconds minimum. The screen should feel like it's waiting for the kid to think.
- **Big, bold, round.** Prefer circles and rounded rectangles over sharp geometry. Feels friendlier.
- **Left-to-right flow.** Always. Math reads left to right for this age group.
- **One operation per video.** Don't mix addition and multiplication in the same video. Build one concept solidly.
