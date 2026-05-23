"""
Tufte-styled charts using Plotly with a reusable template.

Demonstrates: custom Plotly template, no legend, direct annotations,
off-white background, serif fonts, minimal tooltip.
"""

import plotly.graph_objects as go
import plotly.io as pio

# --- Register Tufte template -------------------------------------------------

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
        margin=dict(l=60, r=100, t=80, b=50),
        hoverlabel=dict(
            bgcolor="rgba(255,255,248,0.9)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(family="system-ui, sans-serif", size=12, color="#333"),
        ),
    )
)

pio.templates["tufte"] = tufte_template
pio.templates.default = "tufte"

TUFTE = {
    "series_default": "#666",
    "highlight": "#e41a1c",
    "text_secondary": "#666",
}


# --- Example 1: Line chart with direct labels --------------------------------

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
revenue = [4200, 4800, 5100, 4900, 5600, 6200]
target = [4000, 4200, 4400, 4600, 4800, 5000]

fig = go.Figure()

# Target line (gray dashed)
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

# Direct labels at endpoints
fig.add_annotation(
    x=months[-1], y=revenue[-1],
    text=f"  Revenue: ${revenue[-1]:,}",
    showarrow=False, xanchor="left",
    font=dict(size=13, color=TUFTE["highlight"]),
)
fig.add_annotation(
    x=months[-1], y=target[-1],
    text=f"  Target: ${target[-1]:,}",
    showarrow=False, xanchor="left",
    font=dict(size=13, color=TUFTE["text_secondary"]),
)

# Annotate peak
peak_idx = revenue.index(max(revenue))
fig.add_annotation(
    x=months[peak_idx], y=revenue[peak_idx],
    text=f"Peak: ${revenue[peak_idx]:,}",
    showarrow=True, arrowhead=0, arrowwidth=0.5, arrowcolor="#ccc",
    ax=0, ay=-30,
    font=dict(size=12, color="#333", family='"Palatino Linotype", Georgia, serif'),
)

fig.update_layout(
    title="Monthly Revenue vs. Target",
    width=750, height=500,
)

fig.write_html("tufte-plotly-line.html")
fig.show()


# --- Example 2: Horizontal bar chart -----------------------------------------

import plotly.express as px
import pandas as pd

df = pd.DataFrame({
    "product": ["Product A", "Product B", "Product C", "Product D", "Product E"],
    "revenue": [42000, 38000, 27000, 19000, 12000],
}).sort_values("revenue")

fig2 = px.bar(
    df, x="revenue", y="product", orientation="h",
    text="revenue",
    color_discrete_sequence=[TUFTE["series_default"]],
)

fig2.update_traces(
    texttemplate="$%{text:,.0f}",
    textposition="outside",
    textfont=dict(
        family='"Palatino Linotype", Georgia, serif',
        size=13,
        color=TUFTE["text_secondary"],
    ),
)

fig2.update_layout(
    title="Revenue by Product",
    xaxis=dict(visible=False),
    yaxis=dict(
        showline=False,
        tickfont=dict(size=13, color="#111"),
    ),
    width=750, height=350,
)

fig2.write_html("tufte-plotly-bar.html")
fig2.show()
