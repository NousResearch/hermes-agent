# Concrete-to-Abstract Ladder

The core pedagogical pattern. Every math concept is taught through four progressive representations, always in the same order. Never skip a rung.

## The Four Rungs

```
RUNG 1: REAL OBJECT      -- Something the kid knows from life
RUNG 2: SIMPLE SHAPE      -- Geometric abstraction (circles, blocks)
RUNG 3: DIAGRAM            -- Structured visual (number line, bar, grid)
RUNG 4: SYMBOL             -- Mathematical notation (digits, operators)
```

Each rung is its own scene (30-45 seconds). The full ladder takes 3-5 scenes depending on complexity.

## Why This Order Matters

Kids build mental models from physical experience upward. When you show `3 + 4 = 7` first, the equation is arbitrary -- just symbols to memorize. When you show 3 apples joining 4 apples and the kid counts 7, THEN show the equation, the symbols have meaning anchored in physical reality.

Neurodivergent and visual-spatial learners especially need this grounding. The abstract symbol is the LAST thing they should see, not the first.

## Example: Multiplication (3 x 4 = 12)

### Rung 1: Real Objects (Scene)

Show 3 bags. Each bag opens to reveal 4 toy cars.

```python
# Three paper bags appear
bags = VGroup(*[create_bag() for _ in range(3)])
bags.arrange(RIGHT, buff=1.0).move_to(UP * 0.5)
self.play(LaggedStart(*[GrowFromCenter(b) for b in bags], lag_ratio=0.2))

# Each bag "opens" and 4 toy cars slide out
for bag in bags:
    cars = VGroup(*[create_car(color=GREEN) for _ in range(4)])
    cars.arrange(DOWN, buff=0.2).next_to(bag, DOWN, buff=0.3)
    self.play(
        bag.animate.set_opacity(0.4),
        LaggedStart(*[FadeIn(c, shift=DOWN*0.3) for c in cars], lag_ratio=0.1),
        run_time=1.0,
    )
self.wait(1.5)

# Pip: "Hmm, how many cars is that?"
self.add_subcaption("Hmm, how many cars is that?", duration=3)
self.play(pip.think())
self.wait(3.0)  # LET THEM COUNT
```

### Rung 2: Simple Shapes (Scene)

Same structure, now with colored dots.

```python
# 3 groups of 4 green circles
groups = VGroup()
for i in range(3):
    group = VGroup(*[
        Circle(radius=0.2, color=GREEN, fill_opacity=0.8)
        for _ in range(4)
    ])
    group.arrange_in_grid(rows=2, cols=2, buff=0.15)
    groups.add(group)
groups.arrange(RIGHT, buff=0.8).move_to(ORIGIN)

# Groups appear one at a time
for g in groups:
    self.play(LaggedStart(*[GrowFromCenter(c) for c in g], lag_ratio=0.05))
    self.wait(0.5)

# Draw dashed boundary around each group
for g in groups:
    border = SurroundingRectangle(g, color=GREEN, buff=0.15, corner_radius=0.1)
    border.set_stroke(opacity=0.5)
    self.play(Create(border), run_time=0.4)

# Label groups: "3 groups"
label = Text("3 groups of 4", font_size=LABEL_SIZE, font=MONO, color=GREEN)
label.to_edge(UP, buff=0.5)
self.play(FadeIn(label))
self.wait(1.5)
```

### Rung 3: Diagram (Scene)

Array/grid representation -- a 3x4 grid of squares.

```python
# 3 rows, 4 columns grid
grid = VGroup()
for row in range(3):
    for col in range(4):
        sq = Square(side_length=0.5, color=GREEN, fill_opacity=0.6)
        sq.move_to(RIGHT * col * 0.6 + DOWN * row * 0.6)
        grid.add(sq)
grid.center()

# Fill row by row with bounce count
for row in range(3):
    row_squares = grid[row*4 : (row+1)*4]
    self.play(
        LaggedStart(*[GrowFromCenter(s) for s in row_squares], lag_ratio=0.08),
        run_time=0.6,
    )
    # Row label
    row_label = Text(f"4", font_size=24, font=MONO, color=GREEN)
    row_label.next_to(row_squares, RIGHT, buff=0.3)
    self.play(FadeIn(row_label), run_time=0.3)

# Column count
col_label = Text("3 rows", font_size=24, font=MONO, color=GREEN)
col_label.next_to(grid, LEFT, buff=0.5)
self.play(FadeIn(col_label))

# Total
total = Text("= 12", font_size=36, font=MONO, color=GOLD)
total.next_to(grid, DOWN, buff=0.5)
self.play(Write(total), run_time=1.0)
self.wait(1.5)
```

### Rung 4: Symbol (Scene)

The equation, built piece by piece from what they just saw.

```python
# "3" appears (green, like the groups)
three = MathTex("3", font_size=56, color=GREEN)
three.move_to(LEFT * 2)
self.play(Write(three), run_time=0.6)
self.add_subcaption("3...", duration=1)
self.wait(0.5)

# "x" appears
times = MathTex(r"\times", font_size=56, color=WHITE)
times.next_to(three, RIGHT, buff=0.3)
self.play(Write(times), run_time=0.4)
self.add_subcaption("times...", duration=0.8)
self.wait(0.3)

# "4" appears (green, like the group size)
four = MathTex("4", font_size=56, color=GREEN)
four.next_to(times, RIGHT, buff=0.3)
self.play(Write(four), run_time=0.6)
self.add_subcaption("4...", duration=1)
self.wait(0.5)

# "=" appears
equals = MathTex("=", font_size=56, color=WHITE)
equals.next_to(four, RIGHT, buff=0.3)
self.play(Write(equals), run_time=0.3)

# "12" appears in GOLD -- the big reveal
twelve = MathTex("12", font_size=56, color=GOLD)
twelve.next_to(equals, RIGHT, buff=0.3)
self.play(Write(twelve), run_time=1.0)
self.add_subcaption("equals twelve!", duration=2)

# CELEBRATE
self.play(pip.celebrate())
play_confetti(self, twelve.get_center())
self.wait(1.5)
```

## Ladder Patterns for Each Operation

### Addition: Slide Together

| Rung | Visual | Animation |
|------|--------|-----------|
| Real | 3 red apples + 4 green apples | Groups slide together |
| Shape | Blue circles merge into one group | Slide + recount |
| Diagram | Number line: hop from 3, land on 7 | Pip walks along |
| Symbol | `3 + 4 = 7` built left-to-right | Write piece by piece |

### Subtraction: Take Away

| Rung | Visual | Animation |
|------|--------|-----------|
| Real | 7 cookies, 3 get eaten (chomp!) | FadeOut with humor |
| Shape | Red circles disappear from group | Shrink + vanish |
| Diagram | Number line: hop backward from 7 | Pip walks back |
| Symbol | `7 - 3 = 4` | Coral-red coloring |

### Division: Fair Sharing

| Rung | Visual | Animation |
|------|--------|-----------|
| Real | 12 candies dealt to 3 kids | Objects slide to recipients |
| Shape | Purple circles sort into equal piles | Rearrange animation |
| Diagram | Bar split into 3 equal sections | Cut lines appear |
| Symbol | `12 / 3 = 4` | Purple coloring |

### Fractions: Parts of a Whole

| Rung | Visual | Animation |
|------|--------|-----------|
| Real | Pizza cut in half, one slice colored | Cut line, fill half |
| Shape | Circle with colored sector | Arc + fill |
| Diagram | Fraction bar (rectangle split) | Sections fill |
| Symbol | `1/2` with MathTex | Fraction appears |

## Transition Between Rungs

Never hard-cut between rungs. Use a brief morph or dissolve:

```python
# Morph real objects into abstract shapes
for apple, circle in zip(apples, shapes):
    self.play(
        ReplacementTransform(apple, circle),
        run_time=0.8,
    )
```

This visual continuity reinforces that the abstract shape IS the real object -- just simplified.

## Common Mistakes

- **Starting at Rung 3 or 4.** "Let me show you on the number line" -- no. Show real objects first.
- **Skipping Rung 2.** The intermediate abstraction is where the real learning happens. It bridges physical to diagrammatic.
- **Rushing Rung 4.** The equation should feel like a reward, not an assignment. Slow reveal, gold color, celebration.
- **Same objects every time.** Vary: apples, cars, stars, cookies, blocks. Same structure, different wrapper.
