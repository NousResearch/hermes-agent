# Reward Animations

Celebration animations that fire after key reveals. These are the dopamine hits -- they signal "you got it!" and make the kid want to keep watching.

## Design Philosophy

- **Earned, not given.** Rewards follow a reveal or answer, never just a scene transition.
- **Fast and punchy.** 0.5s animation, 1.0s hold, 0.5s fadeout. Total: 2s max.
- **Varied per video.** Don't use the same celebration every time. Rotate through 3-4 types.
- **Never patronizing.** Think "video game level complete," not "good job, sweetie."

## Confetti Burst

The signature celebration. Colored circles scatter outward from a point.

```python
def create_confetti(center, n=20, spread=3.0):
    """Create confetti dots ready to burst outward from center."""
    colors = [BLUE, RED, GREEN, GOLD, PURPLE, ORANGE]
    confetti = VGroup()
    for i in range(n):
        dot = Dot(
            point=center,
            radius=np.random.uniform(0.04, 0.1),
            color=colors[i % len(colors)],
        )
        # Random target position in a circle around center
        angle = np.random.uniform(0, TAU)
        dist = np.random.uniform(0.5, spread)
        dot.target_pos = center + np.array([
            dist * np.cos(angle),
            dist * np.sin(angle),
            0,
        ])
        confetti.add(dot)
    return confetti


def play_confetti(scene, center, n=20):
    """Full confetti animation: burst outward, fade upward."""
    confetti = create_confetti(center, n)
    scene.add(*confetti)

    # Burst outward
    scene.play(
        *[dot.animate.move_to(dot.target_pos) for dot in confetti],
        run_time=0.5,
        rate_func=rush_from,
    )

    # Float up and fade
    scene.play(
        *[dot.animate.shift(UP * 0.5).set_opacity(0) for dot in confetti],
        run_time=0.8,
        rate_func=smooth,
    )
    scene.remove(*confetti)
```

## Star Burst

Stars radiate outward from the answer. More "magical" than confetti.

```python
def play_star_burst(scene, center, n=8):
    """Stars radiate outward in a ring."""
    stars = VGroup()
    for i in range(n):
        star = Star(
            n=5,
            outer_radius=0.15,
            inner_radius=0.07,
            color=GOLD,
            fill_opacity=1,
        ).move_to(center)
        angle = i * TAU / n
        star.target_pos = center + np.array([
            1.5 * np.cos(angle),
            1.5 * np.sin(angle),
            0,
        ])
        stars.add(star)

    scene.play(
        *[GrowFromCenter(s) for s in stars],
        run_time=0.3,
    )
    scene.play(
        *[s.animate.move_to(s.target_pos).set_opacity(0).scale(0.3) for s in stars],
        run_time=0.7,
        rate_func=rush_from,
    )
    scene.remove(*stars)
```

## Color Wash

The background briefly pulses with a warm color. Subtle but satisfying.

```python
def play_color_wash(scene, color=GOLD, opacity=0.15):
    """Background pulses with a warm color."""
    wash = Rectangle(
        width=config.frame_width,
        height=config.frame_height,
        fill_color=color,
        fill_opacity=0,
        stroke_width=0,
    )
    scene.play(
        wash.animate.set_fill(opacity=opacity),
        run_time=0.3,
    )
    scene.play(
        wash.animate.set_fill(opacity=0),
        run_time=0.7,
    )
    scene.remove(wash)
```

## Pip Spin + Confetti Combo

The full celebration -- use for the final reveal of each video.

```python
def play_full_celebration(scene, pip, answer_mobject):
    """Pip celebrates + confetti from the answer. The big finish."""
    scene.play(pip.celebrate(), run_time=0.6)
    play_confetti(scene, answer_mobject.get_center(), n=25)
    scene.wait(1.0)
```

## Bounce Counter

Not a celebration per se, but deeply satisfying -- objects bounce one by one as a number increments.

```python
def play_bounce_count(scene, objects, counter_pos=UP * 2.5):
    """Bounce each object and increment a visible counter."""
    counter = Integer(0, font_size=48, color=GOLD).move_to(counter_pos)
    scene.play(FadeIn(counter))

    for i, obj in enumerate(objects):
        new_counter = Integer(i + 1, font_size=48, color=GOLD).move_to(counter_pos)
        scene.play(
            obj.animate.shift(UP * 0.2).set_color(GOLD),
            run_time=0.15,
        )
        scene.play(
            obj.animate.shift(DOWN * 0.2).set_color(obj.original_color),
            Transform(counter, new_counter),
            run_time=0.15,
        )
    scene.wait(0.5)
    return counter
```

## When To Use What

| Moment | Reward | Intensity |
|--------|--------|-----------|
| Counting complete | Bounce counter | Low |
| Intermediate step | Color wash | Low |
| Key insight revealed | Star burst | Medium |
| Final answer | Pip spin + confetti | High |
| Video complete | Full celebration | Maximum |

## Sound Effect Cues

Manim can't play audio during rendering, but subcaptions can cue sound effects for post-production:

```python
self.add_subcaption("[SFX: pop]", duration=0.5)   # Object appears
self.add_subcaption("[SFX: ding]", duration=0.5)   # Correct answer
self.add_subcaption("[SFX: whoosh]", duration=0.5) # Confetti burst
self.add_subcaption("[SFX: boing]", duration=0.3)  # Bounce count
```
