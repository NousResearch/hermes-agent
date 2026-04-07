# Pip -- The Mascot System

Pip is a friendly circle character built from basic Manim primitives. Pip appears in every manim-kids video as a curious companion who learns alongside the viewer.

## Construction

```python
class Pip(VGroup):
    """Friendly circle mascot with expressive dot-eyes."""

    def __init__(self, radius=0.4, **kwargs):
        super().__init__(**kwargs)
        self.body = Circle(
            radius=radius,
            color=PIP_OUTLINE,
            fill_color=PIP_BODY,
            fill_opacity=1,
            stroke_width=3,
        )
        # Eyes -- two dots, slightly above center
        eye_y = radius * 0.15
        eye_spread = radius * 0.35
        self.left_eye = Dot(
            point=LEFT * eye_spread + UP * eye_y,
            radius=0.06,
            color=PIP_EYES,
        )
        self.right_eye = Dot(
            point=RIGHT * eye_spread + UP * eye_y,
            radius=0.06,
            color=PIP_EYES,
        )
        self.add(self.body, self.left_eye, self.right_eye)

    # --- Expressions ---

    def enter(self):
        """Bouncy entrance from below."""
        self.shift(DOWN * 2)
        return Succession(
            self.animate.shift(UP * 2.3),
            self.animate.shift(DOWN * 0.3),
            rate_func=smooth,
        )

    def exit(self):
        """Slide out downward."""
        return self.animate.shift(DOWN * 2).set_opacity(0)

    def look_at(self, target):
        """Eyes shift slightly toward target mobject."""
        direction = normalize(target.get_center() - self.get_center())
        shift = direction * 0.05
        return AnimationGroup(
            self.left_eye.animate.shift(shift),
            self.right_eye.animate.shift(shift),
            run_time=0.3,
        )

    def look_center(self):
        """Reset eyes to neutral position."""
        return AnimationGroup(
            self.left_eye.animate.move_to(
                self.body.get_center() + LEFT * 0.35 * self.body.radius + UP * 0.15 * self.body.radius
            ),
            self.right_eye.animate.move_to(
                self.body.get_center() + RIGHT * 0.35 * self.body.radius + UP * 0.15 * self.body.radius
            ),
            run_time=0.3,
        )

    def think(self):
        """Head tilt + one eye up = thinking."""
        return Succession(
            self.animate.rotate(0.15),
            Wait(0.5),
        )

    def think_reset(self):
        """Return from thinking pose."""
        return self.animate.rotate(-0.15)

    def excited(self):
        """Eyes widen + small bounce."""
        return Succession(
            AnimationGroup(
                self.left_eye.animate.scale(1.5),
                self.right_eye.animate.scale(1.5),
                self.animate.shift(UP * 0.15),
            ),
            AnimationGroup(
                self.left_eye.animate.scale(1 / 1.5),
                self.right_eye.animate.scale(1 / 1.5),
                self.animate.shift(DOWN * 0.15),
            ),
            run_time=0.6,
        )

    def surprised(self):
        """Eyes go big + jump back."""
        return Succession(
            AnimationGroup(
                self.left_eye.animate.scale(2.0),
                self.right_eye.animate.scale(2.0),
                self.animate.shift(LEFT * 0.3 + UP * 0.2),
            ),
            AnimationGroup(
                self.left_eye.animate.scale(0.5),
                self.right_eye.animate.scale(0.5),
                self.animate.shift(RIGHT * 0.3 + DOWN * 0.2),
            ),
            run_time=0.8,
        )

    def celebrate(self):
        """Full spin + return to rest."""
        return Rotate(self, angle=TAU, run_time=0.6, rate_func=smooth)

    def nod(self):
        """Small vertical bob = agreement/counting."""
        return Succession(
            self.animate.shift(DOWN * 0.1),
            self.animate.shift(UP * 0.1),
            run_time=0.3,
        )

    def point_at(self, target, scene):
        """Extend an arrow from Pip toward target, hold, retract."""
        arrow = Arrow(
            start=self.get_center(),
            end=target.get_center(),
            color=PIP_OUTLINE,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.15,
        )
        return Succession(
            Create(arrow, run_time=0.4),
            Wait(1.5),
            FadeOut(arrow, run_time=0.3),
        )
```

## Positioning Rules

- **Default position**: bottom-right corner (`self.to_corner(DR, buff=0.8)`)
- **Never overlap content**: Pip is a companion, not a distraction
- **Moves to point**: When pointing at something, Pip shifts toward it, points, then returns
- **Scale**: Default radius 0.4. Scale down to 0.3 if the scene is crowded.

## Expression Choreography

Match Pip's expressions to the narrative beat:

| Narrative moment | Pip expression | Timing |
|-----------------|---------------|--------|
| Video opens | `enter()` | First 1s |
| New objects appear | `look_at(objects)` | As objects land |
| "How many?" question | `think()` | During question pause |
| Kid should be counting | `nod()` per item | Bounce with each count |
| Answer revealed | `excited()` | 0.3s before equation appears |
| Final equation shown | `celebrate()` | Immediately after |
| Scene ends | `exit()` or just stay | Last 0.5s |

## Multiple Pips

For comparison scenes (e.g., showing two different approaches), use two Pips with different outline colors:

```python
pip_a = Pip().set_color(BLUE).to_corner(DL, buff=0.8)
pip_b = Pip().set_color(GREEN).to_corner(DR, buff=0.8)
```

## Accessibility

- Pip's expressions are purely geometric (scale, position, rotation) -- no color-dependent states
- Works at any resolution
- No text on Pip -- expressions are self-evident
- High contrast against dark background (white body, blue outline)
