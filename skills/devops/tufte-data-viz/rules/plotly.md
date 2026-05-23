# Plotly: Tufte Configuration

Load this file when generating any Plotly chart code (Python or JavaScript).

## Tufte template

Register once, use everywhere:

```python
import plotly.graph_objects as go
import plotly.io as pio

tufte_template = go.layout.Template(
    layout=go.Layout(
        font=dict(
            family='"Palatino Linotype", Palatino, Georgia, serif',
            size=13,
            color="#111",
        ),
        paper_bgcolor="#fffff8",
        plot_bgcolor="#fffff8",
        title=dict(
            font=dict(size=20, color="#111"),
            x=0.0,
            xanchor="left",
        ),
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=True,
            linewidth=0.5,
            linecolor="#ccc",
            tickfont=dict(family="system-ui, sans-serif", size=11, color="#999"),
            ticks="inside",
            ticklen=3,
            tickwidth=0.5,
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            tickfont=dict(family="system-ui, sans-serif", size=11, color="#999"),
            ticks="inside",
            ticklen=3,
            tickwidth=0.5,
        ),
        margin=dict(l=60, r=80, t=80, b=50),
        hoverlabel=dict(
            bgcolor="rgba(255,255,248,0.9)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(family="system-ui, sans-serif", size=12, color="#333"),
        ),
    )
)

pio.templates["tufte"] = tufte_template
pio.templates.default = "tufte"
```

## Color constants

```python
TUFTE = {
    "bg": "#fffff8",
    "text": "#111",
    "text_secondary": "#666",
    "text_tertiary": "#999",
    "axis": "#ccc",
    "series_default": "#666",
    "highlight": "#e41a1c",
    "categorical": ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"],
}
```

## Direct labeling (replace legend)

Add annotations at the last data point of each series:

```python
def add_direct_labels(fig, traces_data):
    """
    Add direct labels at the rightmost point of each trace.

    traces_data: list of dicts with keys 'x', 'y', 'name', 'color'
    """
    for trace in traces_data:
        fig.add_annotation(
            x=trace["x"][-1],
            y=trace["y"][-1],
            text=f"  {trace['name']}",
            showarrow=False,
            xanchor="left",
            font=dict(
                family='"Palatino Linotype", Palatino, Georgia, serif',
                size=13,
                color=trace.get("color", TUFTE["text"]),
            ),
        )
```

## Annotation helper

```python
def annotate_point(fig, x, y, text):
    """Annotate a notable point on the chart."""
    fig.add_annotation(
        x=x, y=y,
        text=text,
        showarrow=True,
        arrowhead=0,
        arrowwidth=0.5,
        arrowcolor="#ccc",
        ax=0, ay=-30,
        font=dict(
            family='"Palatino Linotype", Palatino, Georgia, serif',
            size=12,
            color="#333",
        ),
    )
```

## Range-frame axes

Constrain axis range to the data:

```python
def range_frame(fig, x_data, y_data, padding=0.02):
    """Set axis range to span only the data extent."""
    x_range = max(x_data) - min(x_data)
    y_range = max(y_data) - min(y_data)
    fig.update_xaxes(range=[
        min(x_data) - x_range * padding,
        max(x_data) + x_range * padding,
    ])
    fig.update_yaxes(range=[
        min(y_data) - y_range * padding,
        max(y_data) + y_range * padding,
    ])
```

## Complete example: line chart (graph_objects)

```python
import plotly.graph_objects as go

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
revenue = [4200, 4800, 5100, 4900, 5600, 6200]
target = [4000, 4200, 4400, 4600, 4800, 5000]

fig = go.Figure()

# Target line (gray, dashed)
fig.add_trace(go.Scatter(
    x=months, y=target, mode="lines",
    line=dict(color=TUFTE["series_default"], width=1, dash="dash"),
    hovertemplate="%{y:$,.0f}<extra>Target</extra>",
))

# Revenue line (highlighted)
fig.add_trace(go.Scatter(
    x=months, y=revenue, mode="lines",
    line=dict(color=TUFTE["highlight"], width=2),
    hovertemplate="%{y:$,.0f}<extra>Revenue</extra>",
))

# Direct labels
add_direct_labels(fig, [
    {"x": months, "y": revenue, "name": f"Revenue: ${revenue[-1]:,}", "color": TUFTE["highlight"]},
    {"x": months, "y": target, "name": f"Target: ${target[-1]:,}", "color": TUFTE["text_secondary"]},
])

# Annotate peak
peak_idx = revenue.index(max(revenue))
annotate_point(fig, months[peak_idx], revenue[peak_idx], f"Peak: ${revenue[peak_idx]:,}")

fig.update_layout(
    title=dict(text="Monthly Revenue vs. Target"),
    width=750, height=500,
)

fig.show()
```

## Complete example: horizontal bar chart (plotly.express)

```python
import plotly.express as px
import pandas as pd

df = pd.DataFrame({
    "product": ["Product A", "Product B", "Product C", "Product D", "Product E"],
    "revenue": [42000, 38000, 27000, 19000, 12000],
}).sort_values("revenue")

fig = px.bar(
    df, x="revenue", y="product", orientation="h",
    text="revenue",
    color_discrete_sequence=[TUFTE["series_default"]],
)

fig.update_traces(
    texttemplate="$%{text:,.0f}",
    textposition="outside",
    textfont=dict(family='"Palatino Linotype", Georgia, serif', size=13, color=TUFTE["text_secondary"]),
)

fig.update_layout(
    title=dict(text="Revenue by Product"),
    xaxis=dict(visible=False),  # Values are on bars — axis is redundant
    yaxis=dict(showline=False, tickfont=dict(size=13, color=TUFTE["text"])),
    width=750, height=350,
)

fig.show()
```

## Plotly.js (JavaScript) equivalent

```javascript
const tufteLayout = {
  paper_bgcolor: '#fffff8',
  plot_bgcolor: '#fffff8',
  font: { family: '"Palatino Linotype", Palatino, Georgia, serif', size: 13, color: '#111' },
  showlegend: false,
  xaxis: { showgrid: false, zeroline: false, linewidth: 0.5, linecolor: '#ccc', tickfont: { size: 11, color: '#999' } },
  yaxis: { showgrid: false, zeroline: false, showline: false, tickfont: { size: 11, color: '#999' } },
  hoverlabel: { bgcolor: 'rgba(255,255,248,0.9)', bordercolor: 'transparent', font: { size: 12, color: '#333' } },
  margin: { l: 60, r: 80, t: 60, b: 50 },
};
```
