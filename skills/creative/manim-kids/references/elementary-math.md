# Elementary Math Concept Map

Grade-level concept progressions with visual metaphors and recommended animation patterns. Each entry maps a math concept to the concrete objects, shapes, diagrams, and animations that make it click for young learners.

## Kindergarten (Ages 5-6)

### Counting (1-20)

**Visual**: Objects appear one at a time with bounce. Number counter increments.
**Animation**: `GrowFromCenter` each object + `play_bounce_count()`.
**Metaphor**: Counting toys into a toy box, stars appearing in the sky.
**Pip**: Nods with each count.

```
Objects:  *  * *  * * *  (appear one by one)
Counter:  1  2 3  4 5 6  (increments)
```

### More / Less / Same

**Visual**: Two groups side by side. One is clearly bigger. Lines connect matching pairs -- leftover objects highlight.
**Animation**: Matching pairs slide toward each other and connect with lines. Unmatched objects glow.
**Metaphor**: Two plates of cookies. "Who has more?"

### Shapes

**Visual**: Shape appears, label appears, shape morphs into a real object (circle -> wheel, square -> window).
**Animation**: `Create(shape)` then `ReplacementTransform(shape, real_object)`.
**Pip**: Points at each shape.

---

## First Grade (Ages 6-7)

### Addition (sums to 20)

**Visual**: Two groups of objects slide together and merge.
**Animation**: Left group + right group -> `animate.shift()` together, recount.
**Color**: BLUE for all addition elements.
**Ladder**:
1. Real: 3 toy cars + 5 toy cars
2. Shape: Blue circles merging
3. Diagram: Number line -- hop from 3, count 5 hops, land on 8
4. Symbol: `3 + 5 = 8`

### Subtraction (from 20)

**Visual**: Group of objects, some get removed (fly away, get eaten, disappear).
**Animation**: `FadeOut(objects, shift=UP)` or shrink + vanish for humor.
**Color**: RED/CORAL for subtraction elements.
**Ladder**:
1. Real: 8 cookies, 3 get eaten (cartoon chomp subcaption)
2. Shape: Red circles shrink away
3. Diagram: Number line -- hop backward
4. Symbol: `8 - 3 = 5`

### Place Value (tens and ones)

**Visual**: Loose blocks that snap into rods of 10. "23 = 2 rods + 3 loose blocks."
**Animation**: 10 blocks slide together and `ReplacementTransform` into a rod.
**Metaphor**: Bundling sticks, stacking cups.

---

## Second Grade (Ages 7-8)

### Addition with carrying

**Visual**: Ones column fills past 10 -> 10 ones morph into 1 ten-rod that slides to the tens column.
**Animation**: Step-by-step column addition with visual carry.
**Key moment**: The "carry" is shown as a physical movement, not a written digit.

### Skip counting (2s, 5s, 10s)

**Visual**: Number line with Pip hopping in fixed jumps. Landing spots light up.
**Animation**: Pip hops along number line, each landing creates a splash effect.
**Pattern discovery**: "What do you notice about these numbers?"

### Measurement

**Visual**: Object next to a ruler. Unit markers count off. "The pencil is 6 units long."
**Animation**: Unit blocks tile along the object, counter increments.
**Metaphor**: Pip walking along the object, counting steps.

### Money (coins and bills)

**Visual**: Coins with value labels. Coins slide together, values add up.
**Color**: Gold for all money. Values in white.
**Key pattern**: Quarter (25) + quarter (25) = 50. Show with bounce count by 25s.

---

## Third Grade (Ages 8-9) -- PRIMARY TARGET

### Multiplication as groups

**THE core concept for this age. Treat it as the most important topic.**

**Visual**: N groups of M objects. Groups form, then merge. Array grid forms.
**Color**: GREEN for all multiplication elements.
**Multiple representations**:
1. Groups of objects (bags of marbles, boxes of crayons)
2. Array grid (rows x columns of dots/squares)
3. Number line with repeated jumps
4. Equation with `x` or `times` symbol

**Key animations**:
- Groups forming from scattered objects (`arrange_in_grid`)
- Array filling row by row
- Pip counting by the group size ("4... 8... 12!")

```python
# Multiplication as repeated addition (number line)
for jump in range(3):
    arc = ArcBetweenPoints(
        start=number_line.n2p(jump * 4),
        end=number_line.n2p((jump + 1) * 4),
        angle=-0.5,
        color=GREEN,
    )
    self.play(Create(arc), run_time=0.6)
    # Landing label
    label = Text(str((jump + 1) * 4), font_size=24, font=MONO, color=GOLD)
    label.next_to(number_line.n2p((jump + 1) * 4), DOWN, buff=0.3)
    self.play(FadeIn(label))
```

### Division as sharing

**Visual**: Objects dealt out to recipients one at a time (round-robin).
**Color**: PURPLE for division.
**Animation**: Objects slide from center pile to recipient positions. Counter per recipient increments.
**Key insight**: Division is the reverse of multiplication. Show 12 / 3 = 4, then immediately show 3 x 4 = 12 with the same objects.

### Fractions (1/2, 1/3, 1/4, unit fractions)

**Visual**: Whole object (pizza, chocolate bar) gets cut. Parts are colored.
**Color**: Use the operation color of the context (e.g., GREEN if dividing pizza into equal groups).
**Animations**:
- Cut line appears (DashedLine)
- One part fills with color, others dim
- Fraction label appears: `MathTex(r"\frac{1}{4}")`

**Key concept**: "The bottom number is how many EQUAL pieces. The top number is how many you HAVE."

```python
# Pizza fraction
pizza = Circle(radius=1.5, color=ORANGE, fill_opacity=0.8)
# Cut into 4
for i in range(4):
    angle = i * TAU / 4
    cut = Line(
        start=pizza.get_center(),
        end=pizza.get_center() + np.array([1.5*np.cos(angle), 1.5*np.sin(angle), 0]),
        color=WHITE, stroke_width=2,
    )
    self.play(Create(cut), run_time=0.4)

# Color one slice
slice_highlight = AnnularSector(
    inner_radius=0, outer_radius=1.5,
    angle=TAU/4, start_angle=0,
    color=GOLD, fill_opacity=0.6,
).move_to(pizza.get_center())
self.play(FadeIn(slice_highlight))

# Label
frac = MathTex(r"\frac{1}{4}", font_size=48, color=GOLD)
frac.next_to(pizza, RIGHT, buff=1.0)
self.play(Write(frac))
```

### Area (length x width)

**Visual**: Rectangle fills with unit square tiles, row by row.
**Animation**: Tiles fill left-to-right, row-by-row. Counter shows running total.
**Connection to multiplication**: "3 rows of 5 tiles = 3 x 5 = 15 square units."
**Pip**: Points at rows, counts.

### Rounding (to nearest 10, 100)

**Visual**: Number line with "gravity wells" at 10, 20, 30... Numbers slide to the nearest well.
**Animation**: Number appears, wobbles, then slides (with a satisfying snap) to the nearest ten.
**Key insight**: "Is 37 closer to 30 or 40? Look at the ones digit."
**Pip**: Points at the midpoint (5) -- "this is the tipping point!"

### Telling time

**Visual**: Analog clock with hands. Hour hand is thick and short, minute hand is thin and long.
**Animation**: Hands sweep to positions. Labels appear.
**Key pattern**: Clock face is a number line bent into a circle. Show this transformation.

### Perimeter

**Visual**: Ant walks around the outside of a shape. Each side gets a length label. Counter adds up.
**Animation**: Traced path with Pip "walking" the perimeter. Side lengths bounce as they're added.
**Metaphor**: "How far does the ant walk to get all the way around?"

---

## Fourth-Fifth Grade Preview (Ages 9-11)

These are stretch concepts for advanced third graders:

### Multi-digit multiplication

**Visual**: Area model (rectangle split into parts). 23 x 14 = (20+3)(10+4).
**Animation**: Rectangle splits into 4 sub-rectangles, each calculated, then summed.

### Equivalent fractions

**Visual**: Same pizza, different number of cuts. 1/2 = 2/4 = 4/8.
**Animation**: Additional cuts appear, but the colored area stays the same.

### Decimals

**Visual**: Place value chart extending right past the decimal point. 10ths column = one rod cut into 10 pieces.
**Animation**: Rod from place value splits into 10 equal pieces.

### Negative numbers

**Visual**: Number line extending left past zero. Temperature metaphor (thermometer going below zero).
**Animation**: Number line slides to reveal negative territory. Pip shivers.

---

## Animation Patterns Quick Reference

| Concept | Primary animation | Color |
|---------|------------------|-------|
| Counting | Bounce count | GOLD (counter) |
| Addition | Slide together | BLUE |
| Subtraction | FadeOut / shrink | RED |
| Multiplication | Array fill / group form | GREEN |
| Division | Deal out / split | PURPLE |
| Fractions | Cut + color fill | Context-dependent |
| Number line | Hop with arcs | Context-dependent |
| Place value | Bundle into rods | BLUE (ones), GREEN (tens) |
| Area | Tile fill | GREEN |
| Perimeter | Trace walk | ORANGE |
| Rounding | Gravity slide + snap | GOLD |
