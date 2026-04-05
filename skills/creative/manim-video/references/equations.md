# Equations and LaTeX Reference

## Basic LaTeX

```python
eq = MathTex(r"E = mc^2")
eq = MathTex(r"f(x) &= x^2 + 2x + 1 \\ &= (x + 1)^2")  # multi-line aligned
```

**Always use raw strings (`r""`).**

## Step-by-Step Derivations

```python
step1 = MathTex(r"a^2 + b^2 = c^2")
step2 = MathTex(r"a^2 = c^2 - b^2")
self.play(Write(step1), run_time=1.5)
self.wait(1.5)
self.play(TransformMatchingTex(step1, step2), run_time=1.5)
```

## Selective Color

```python
eq = MathTex(r"a^2", r"+", r"b^2", r"=", r"c^2")
eq[0].set_color(RED)
eq[4].set_color(GREEN)
```

## Building Incrementally

```python
parts = MathTex(r"f(x)", r"=", r"\sum_{n=0}^{\infty}", r"\frac{f^{(n)}(a)}{n!}", r"(x-a)^n")
self.play(Write(parts[0:2]))
self.wait(0.5)
self.play(Write(parts[2]))
self.wait(0.5)
self.play(Write(parts[3:]))
```

## Highlighting

```python
highlight = SurroundingRectangle(eq[2], color=YELLOW, buff=0.1)
self.play(Create(highlight))
self.play(Indicate(eq[4], color=YELLOW))
```

## Annotation

```python
brace = Brace(eq, DOWN, color=YELLOW)
label = brace.get_text("Fundamental Theorem", font_size=24)
self.play(GrowFromCenter(brace), Write(label))
```

## Common LaTeX

```python
MathTex(r"\frac{a}{b}")                  # fraction
MathTex(r"\alpha, \beta, \gamma")         # Greek
MathTex(r"\sum_{i=1}^{n} x_i")           # summation
MathTex(r"\int_{0}^{\infty} e^{-x} dx")  # integral
MathTex(r"\vec{v}")                       # vector
MathTex(r"\lim_{x \to \infty} f(x)")    # limit
```

## Derivation Pattern

```python
class DerivationScene(Scene):
    def construct(self):
        self.camera.background_color = BG
        s1 = MathTex(r"ax^2 + bx + c = 0")
        self.play(Write(s1))
        self.wait(1.5)
        s2 = MathTex(r"x^2 + \frac{b}{a}x + \frac{c}{a} = 0")
        s2.next_to(s1, DOWN, buff=0.8)
        self.play(s1.animate.set_opacity(0.4), TransformMatchingTex(s1.copy(), s2))
```
