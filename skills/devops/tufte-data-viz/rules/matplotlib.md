# matplotlib: Tufte Configuration

Load this file when generating any matplotlib or seaborn chart code.

## rcParams setup

Apply these at the top of any script or notebook:

```python
import matplotlib.pyplot as plt
import matplotlib as mpl

TUFTE_RC = {
    # Font
    "font.family": "serif",
    "font.serif": ["Palatino", "Palatino Linotype", "Georgia", "DejaVu Serif"],
    "font.size": 12,

    # Figure
    "figure.facecolor": "#fffff8",
    "figure.figsize": (9, 6),  # 1.5:1 aspect ratio
    "figure.dpi": 150,

    # Axes
    "axes.facecolor": "#fffff8",
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.5,
    "axes.labelcolor": "#666666",
    "axes.labelsize": 12,
    "axes.titlesize": 18,
    "axes.titleweight": "normal",
    "axes.titlepad": 16,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,

    # Ticks
    "xtick.color": "#999999",
    "ytick.color": "#999999",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,

    # Lines
    "lines.linewidth": 1.5,
    "lines.markersize": 3,

    # Legend (in case it's ever used)
    "legend.frameon": False,
    "legend.fontsize": 11,

    # Grid (off by default, but configured if manually enabled)
    "grid.color": "#eeeeee",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.5,

    # Savefig
    "savefig.facecolor": "#fffff8",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
}

plt.rcParams.update(TUFTE_RC)
```

## Tufte helper functions

### Range-frame axes

Constrain axis lines to span only the data range:

```python
def tufte_axes(ax, x_data, y_data):
    """Apply Tufte range-frame to an axes object."""
    # Remove top and right spines (should already be off via rcParams)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Range-frame: axis spans only the data range
    ax.spines["bottom"].set_bounds(min(x_data), max(x_data))
    ax.spines["left"].set_bounds(min(y_data), max(y_data))

    # Inward ticks
    ax.tick_params(direction="in", length=3, width=0.5)

    return ax
```

### Direct labeling (replace legend)

```python
def direct_label(ax, x, y, label, color="#111", offset=(8, 0)):
    """Add a direct label at the last point of a series."""
    ax.annotate(
        label,
        xy=(x[-1], y[-1]),
        xytext=offset,
        textcoords="offset points",
        fontsize=12,
        color=color,
        fontfamily="serif",
        va="center",
    )
```

### Annotation helper

```python
def annotate_point(ax, x, y, text, color="#333"):
    """Annotate a notable data point with an arrow."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(0, 24),
        textcoords="offset points",
        fontsize=11,
        fontstyle="italic",
        color=color,
        fontfamily="serif",
        ha="center",
        arrowprops=dict(arrowstyle="-", color="#cccccc", lw=0.5),
    )
```

### Sparkline

```python
def sparkline(ax, data, color="#666", highlight_endpoints=True):
    """Draw a minimal sparkline on the given axes."""
    ax.plot(data, color=color, linewidth=1)
    ax.set_xlim(0, len(data) - 1)

    if highlight_endpoints:
        # Min and max dots
        min_idx = data.index(min(data))
        max_idx = data.index(max(data))
        ax.plot(min_idx, data[min_idx], "o", color="#e15759", markersize=2)
        ax.plot(max_idx, data[max_idx], "o", color="#4e79a7", markersize=2)
        # Endpoint dot
        ax.plot(len(data) - 1, data[-1], "o", color=color, markersize=2)

    # Remove all chrome
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
```

### Rug marks (dot-dash plot)

```python
def rug_marks(ax, x_data, y_data, color="#999999", size=3):
    """Add rug marks along axes showing marginal distributions."""
    y_min = ax.get_ylim()[0]
    x_min = ax.get_xlim()[0]
    ax.plot(x_data, [y_min] * len(x_data), "|", color=color, markersize=size, alpha=0.5)
    ax.plot([x_min] * len(y_data), y_data, "_", color=color, markersize=size, alpha=0.5)
```

## Color constants

```python
TUFTE_COLORS = {
    "bg": "#fffff8",
    "text": "#111111",
    "text_secondary": "#666666",
    "text_tertiary": "#999999",
    "axis": "#cccccc",
    "grid": "#eeeeee",
    "series_default": "#666666",
    "highlight": "#e41a1c",
    "categorical": ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"],
}
```

## Complete example: annotated time series

```python
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(TUFTE_RC)

# Data
months = np.arange(1, 13)
month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
revenue = [42, 48, 51, 49, 56, 62, 58, 65, 71, 68, 75, 82]
target = [40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62]

fig, ax = plt.subplots()

# Plot data
ax.plot(months, target, color="#999999", linewidth=1, linestyle="--", label="_nolegend_")
ax.plot(months, revenue, color=TUFTE_COLORS["highlight"], linewidth=2)

# Range-frame
tufte_axes(ax, months, revenue + target)

# Direct labels (not legend)
direct_label(ax, months, revenue, "Revenue", color=TUFTE_COLORS["highlight"])
direct_label(ax, months, target, "Target", color="#999999")

# Annotate the peak
peak_idx = revenue.index(max(revenue))
annotate_point(ax, months[peak_idx], revenue[peak_idx], f"Peak: ${revenue[peak_idx]}k")

# Title (not ax.set_title — manual positioning for Tufte style)
fig.text(0.12, 0.95, "Monthly Revenue vs. Target, 2025",
         fontsize=18, fontfamily="serif", color=TUFTE_COLORS["text"])
fig.text(0.12, 0.91, "Revenue exceeded target every month, accelerating in H2",
         fontsize=13, fontfamily="serif", color=TUFTE_COLORS["text_secondary"])

# X-axis labels
ax.set_xticks(months)
ax.set_xticklabels(month_labels)
ax.set_ylabel("Revenue ($k)", fontsize=12, color=TUFTE_COLORS["text_secondary"])

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()
```

## Complete example: horizontal bar chart

```python
import matplotlib.pyplot as plt

plt.rcParams.update(TUFTE_RC)

categories = ["Product A", "Product B", "Product C", "Product D", "Product E"]
values = [42, 38, 27, 19, 12]

# Sort by value
sorted_pairs = sorted(zip(values, categories), reverse=True)
values, categories = zip(*sorted_pairs)

fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.barh(categories, values, color=TUFTE_COLORS["series_default"], height=0.6)

# Direct value labels
for bar, val in zip(bars, values):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"${val}k", va="center", fontsize=12, color=TUFTE_COLORS["text_secondary"],
            fontfamily="serif")

# Remove all spines
for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_xticks([])  # Values are on the bars — x-axis is redundant
ax.tick_params(left=False)  # Remove y tick marks, keep labels

fig.text(0.04, 0.95, "Revenue by Product",
         fontsize=18, fontfamily="serif", color=TUFTE_COLORS["text"])

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.show()
```

## Seaborn integration

```python
import seaborn as sns

# Apply Tufte-compatible seaborn theme
sns.set_theme(style="ticks", rc=TUFTE_RC)

# After any seaborn plot, clean up:
sns.despine()  # Removes top and right spines
ax = plt.gca()
ax.get_legend().remove()  # Seaborn adds legends by default — remove

# For range-frame after a seaborn plot:
tufte_axes(ax, x_data, y_data)
```

## Small multiples

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

metrics = {"Revenue": revenue_data, "Users": user_data, "Conversion": conv_data}

for ax, (name, data) in zip(axes, metrics.items()):
    ax.plot(data, color=TUFTE_COLORS["series_default"], linewidth=1.5)
    ax.set_title(name, fontsize=14, fontfamily="serif", fontweight="normal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)

plt.tight_layout()
plt.show()
```
