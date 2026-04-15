# Mobjects Reference

Everything visible on screen is a Mobject. They have position, color, opacity, and can be animated.

## Text

```python
title = Text("Hello World", font_size=48, color=BLUE)
eq = MathTex(r"E = mc^2", font_size=40)

# Multi-part (for selective coloring)
eq = MathTex(r"a^2", r"+", r"b^2", r"=", r"c^2")
eq[0].set_color(RED)
eq[4].set_color(BLUE)

# Mixed text and math
t = Tex(r"The area is $\pi r^2$", font_size=36)

# Styled markup
t = MarkupText('<span foreground="#58C4DD">Blue</span> text', font_size=30)
```

**Always use raw strings (`r""`) for any string with backslashes.**

## Shapes

```python
circle = Circle(radius=1, color=BLUE, fill_opacity=0.5)
square = Square(side_length=2, color=RED)
rect = Rectangle(width=4, height=2, color=GREEN)
dot = Dot(point=ORIGIN, radius=0.08, color=YELLOW)
line = Line(LEFT * 2, RIGHT * 2, color=WHITE)
arrow = Arrow(LEFT, RIGHT, color=ORANGE)
rrect = RoundedRectangle(corner_radius=0.3, width=4, height=2)
brace = Brace(rect, DOWN, color=YELLOW)
```

## Positioning

```python
mob.move_to(ORIGIN)                        # center
mob.move_to(UP * 2 + RIGHT)               # relative
label.next_to(circle, DOWN, buff=0.3)     # next to another
title.to_edge(UP, buff=0.5)               # screen edge (buff >= 0.5!)
mob.to_corner(UL, buff=0.5)               # corner
```

## VGroup vs Group

**VGroup** is for collections of shapes (VMobjects only — Circle, Square, Arrow, Line, MathTex):
```python
shapes = VGroup(circle, square, arrow)
shapes.arrange(DOWN, buff=0.5)
shapes.set_color(BLUE)
```

**Group** is for mixed collections (Text + shapes, or any Mobject types):
```python
# Text objects are Mobjects, not VMobjects — use Group when mixing
labeled_shape = Group(circle, Text("Label").next_to(circle, DOWN))
labeled_shape.move_to(ORIGIN)

# FadeOut everything on screen (may contain mixed types)
self.play(FadeOut(Group(*self.mobjects)))
```

**Rule: if your group contains any `Text()` objects, use `Group`, not `VGroup`.** VGroup will raise a TypeError on Manim CE v0.20+. MathTex and Tex are VMobjects and work with VGroup.

Both support `arrange()`, `arrange_in_grid()`, `set_opacity()`, `shift()`, `scale()`, `move_to()`.

## Styling

```python
mob.set_color(BLUE)
mob.set_fill(RED, opacity=0.5)
mob.set_stroke(WHITE, width=2)
mob.set_opacity(0.4)
mob.set_z_index(1)                         # layering
```

## Specialized Mobjects

```python
nl = NumberLine(x_range=[-3, 3, 1], length=8, include_numbers=True)
table = Table([["A", "B"], ["C", "D"]], row_labels=[Text("R1"), Text("R2")])
code = Code("example.py", tab_width=4, font_size=20, language="python")
highlight = SurroundingRectangle(target, color=YELLOW, buff=0.2)
bg = BackgroundRectangle(equation, fill_opacity=0.7, buff=0.2)
```

## Custom Mobjects

```python
class NetworkNode(Group):
    def __init__(self, label_text, color=BLUE, **kwargs):
        super().__init__(**kwargs)
        self.circle = Circle(radius=0.4, color=color, fill_opacity=0.3)
        self.label = Text(label_text, font_size=20).move_to(self.circle)
        self.add(self.circle, self.label)
```

## Constants

Directions: `UP, DOWN, LEFT, RIGHT, ORIGIN, UL, UR, DL, DR`
Colors: `RED, BLUE, GREEN, YELLOW, WHITE, GRAY, ORANGE, PINK, PURPLE, TEAL, GOLD`
Frame: `config.frame_width = 14.222, config.frame_height = 8.0`
