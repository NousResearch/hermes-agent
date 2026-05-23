"""
Tufte-styled annotated time series using matplotlib.

Demonstrates: range-frame axes, direct labels (no legend), annotation of peak,
off-white background, serif fonts, no gridlines.
"""

import matplotlib.pyplot as plt
import numpy as np

# --- Tufte rcParams -----------------------------------------------------------

TUFTE_RC = {
    "font.family": "serif",
    "font.serif": ["Palatino", "Palatino Linotype", "Georgia", "DejaVu Serif"],
    "font.size": 12,
    "figure.facecolor": "#fffff8",
    "figure.figsize": (9, 6),
    "figure.dpi": 150,
    "axes.facecolor": "#fffff8",
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.5,
    "axes.labelcolor": "#666666",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
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
    "lines.linewidth": 1.5,
    "savefig.facecolor": "#fffff8",
    "savefig.bbox": "tight",
}

plt.rcParams.update(TUFTE_RC)

COLORS = {
    "text": "#111111",
    "text_secondary": "#666666",
    "series_default": "#666666",
    "highlight": "#e41a1c",
    "axis": "#cccccc",
}


# --- Helper functions ---------------------------------------------------------

def tufte_axes(ax, x_data, y_data):
    """Apply Tufte range-frame to axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_bounds(min(x_data), max(x_data))
    ax.spines["left"].set_bounds(min(y_data), max(y_data))
    ax.tick_params(direction="in", length=3, width=0.5)
    return ax


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


# --- Data ---------------------------------------------------------------------

months = np.arange(1, 13)
month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
revenue = [42, 48, 51, 49, 56, 62, 58, 65, 71, 68, 75, 82]
target = [40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62]


# --- Plot ---------------------------------------------------------------------

fig, ax = plt.subplots()

# Target line (gray dashed)
ax.plot(months, target, color=COLORS["series_default"], linewidth=1, linestyle="--")

# Revenue line (highlighted)
ax.plot(months, revenue, color=COLORS["highlight"], linewidth=2)

# Range-frame axes
all_y = revenue + target
tufte_axes(ax, months, all_y)

# Direct labels (not legend)
direct_label(ax, months, revenue, "Revenue", color=COLORS["highlight"])
direct_label(ax, months, target, "Target", color=COLORS["series_default"])

# Annotate the peak
peak_idx = revenue.index(max(revenue))
ax.annotate(
    f"Peak: ${revenue[peak_idx]}k",
    xy=(months[peak_idx], revenue[peak_idx]),
    xytext=(0, 20),
    textcoords="offset points",
    fontsize=11,
    fontstyle="italic",
    color="#333",
    fontfamily="serif",
    ha="center",
    arrowprops=dict(arrowstyle="-", color="#ccc", lw=0.5),
)

# Axis labels
ax.set_xticks(months)
ax.set_xticklabels(month_labels)
ax.set_ylabel("Revenue ($k)", fontsize=12, color=COLORS["text_secondary"])

# Title
fig.text(
    0.12, 0.95,
    "Revenue Exceeded Target Every Month, Accelerating in H2",
    fontsize=18, fontfamily="serif", color=COLORS["text"],
)
fig.text(
    0.12, 0.91,
    "Monthly revenue vs. target, 2025",
    fontsize=13, fontfamily="serif", color=COLORS["text_secondary"],
)

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig("tufte-line-chart.png", dpi=150)
plt.show()
