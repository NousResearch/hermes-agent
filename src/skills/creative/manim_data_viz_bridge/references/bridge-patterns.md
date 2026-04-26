# Data shapes for `VizSpecification`

## XY line / scatter / regression

```json
{
  "chart_type": "regression",
  "title": "Linear trend in noisy samples",
  "x_label": "t",
  "y_label": "f(t)",
  "data": { "records": [
    {"x": 0, "y": 0.1}, {"x": 1, "y": 2.1}, {"x": 2, "y": 3.8}
  ]}
}
```

## Heatmap

```json
{
  "chart_type": "heatmap",
  "title": "Kernel weights",
  "data": { "matrix": [[1, 0, 2], [0, 3, 1], [2, 1, 0]] }
}
```

## Bar chart

```json
{
  "chart_type": "bar",
  "title": "Latency by region",
  "data": { "records": [
    {"category": "EU", "value": 42}, {"category": "US", "value": 55}
  ]}
}
```

## LaTeX overlays

Set `manim.tex_explanations` to fragments such as `r"\\hat{y} = X\\beta + \\epsilon"` (no surrounding `$`).
