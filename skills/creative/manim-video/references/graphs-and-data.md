# Graphs, Plots, and Data Visualization

## Axes

```python
axes = Axes(
    x_range=[-3, 3, 1], y_range=[-2, 2, 1],
    x_length=8, y_length=5,
    axis_config={"include_numbers": True, "font_size": 24}
)
axes.set_opacity(0.15)  # structural element
x_label = axes.get_x_axis_label(r"x")
```

## Plotting

```python
graph = axes.plot(lambda x: x**2, color=BLUE)
graph_label = axes.get_graph_label(graph, label=r"x^2", x_val=2)
area = axes.get_area(graph, x_range=[0, 2], color=BLUE, opacity=0.3)
```

## Animated Plotting

```python
self.play(Create(graph), run_time=3)  # trace the graph

# Moving dot along curve
dot = Dot(color=YELLOW).move_to(axes.c2p(0, 0))
self.play(MoveAlongPath(dot, graph), run_time=3)

# Dynamic parameter
tracker = ValueTracker(1)
dynamic = always_redraw(lambda: axes.plot(lambda x: tracker.get_value() * x**2, color=BLUE))
self.add(dynamic)
self.play(tracker.animate.set_value(3), run_time=2)
```

## Bar Charts

```python
chart = BarChart(
    values=[4, 6, 2, 8, 5], bar_names=["A", "B", "C", "D", "E"],
    y_range=[0, 10, 2], bar_colors=[RED, GREEN, BLUE, YELLOW, PURPLE]
)
self.play(Create(chart), run_time=2)
self.play(chart.animate.change_bar_values([6, 3, 7, 4, 9]))
```

## Number Lines

```python
nl = NumberLine(x_range=[0, 10, 1], length=10, include_numbers=True)
pointer = Arrow(nl.n2p(3) + UP * 0.5, nl.n2p(3), color=RED, buff=0)
tracker = ValueTracker(3)
pointer.add_updater(lambda m: m.put_start_and_end_on(
    nl.n2p(tracker.get_value()) + UP * 0.5, nl.n2p(tracker.get_value())))
self.play(tracker.animate.set_value(8), run_time=2)
```

## Animated Counters

```python
counter = DecimalNumber(0, font_size=72, num_decimal_places=0)
self.play(counter.animate.set_value(1000), run_time=3, rate_func=rush_from)
```

## Algorithm Visualization Pattern

```python
values = [5, 2, 8, 1, 9, 3]
bars = VGroup(*[
    Rectangle(width=0.6, height=v * 0.4, color=BLUE, fill_opacity=0.7)
    for v in values
]).arrange(RIGHT, buff=0.2, aligned_edge=DOWN).move_to(ORIGIN)
self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.1))
# Highlight, swap, etc.
```

## Data Story Pattern

```python
# Before/After comparison
before = BarChart(values=[3, 5, 2], bar_colors=[RED]*3).shift(LEFT * 3)
after = BarChart(values=[8, 9, 7], bar_colors=[GREEN]*3).shift(RIGHT * 3)
self.play(Create(before)); self.wait(1)
self.play(Create(after)); self.wait(1)
arrow = Arrow(before.get_right(), after.get_left(), color=YELLOW)
label = Text("+167%", font_size=36, color=YELLOW).next_to(arrow, UP)
self.play(GrowArrow(arrow), Write(label))
```
