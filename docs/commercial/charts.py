"""Charts and diagrams for the playbook, rendered to PNG at print resolution.

Every number plotted here is either sourced (and cited in the figure's own caption in the
document) or an explicitly labelled illustrative assumption. No chart invents a market
fact; the two economic charts carry "ILLUSTRATIVE" in the title itself, so the label
survives being screenshotted out of the document.

Colour follows the data-viz method: magnitude gets one hue, identity gets the fixed
categorical order (slots 1 and 2), and the palette was validated against this document's
panel colour rather than assumed.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

OUT = "docs/commercial/assets"

SURFACE = "#141C2E"
TEXT = "#E8ECF4"
TEXT_MUTED = "#9FB0CC"
GRID = "#243047"
SERIES_1 = "#3987E5"
SERIES_2 = "#D95926"
AQUA = "#199E70"
# One-hue sequential ramp for magnitude, light -> dark (data-viz blue ramp).
RAMP = ["#86B6EF", "#6DA7EC", "#5598E7", "#3987E5", "#2A78D6", "#256ABF", "#1C5CAB",
        "#184F95", "#104281", "#0D366B"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": TEXT,
    "axes.labelcolor": TEXT_MUTED,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "axes.edgecolor": GRID,
})


def _clean(ax, *, xgrid=False, ygrid=False):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, labelsize=8)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)


def vertical_scores():
    """Composite market-attractiveness score per vertical.

    One measure across ranked categories with long names, so: horizontal bars, sorted,
    one hue darkening with rank, value direct-labelled on every bar (twelve bars, no
    legend — a single series names itself in the title).
    """
    data = [
        ("Accountancy practices", 8.4),
        ("Property / block management", 7.5),
        ("Recruitment agencies", 7.3),
        ("IT managed service providers", 7.1),
        ("Digital / marketing agencies", 6.8),
        ("Logistics & freight forwarding", 6.5),
        ("Professional services (other)", 6.4),
        ("E-commerce operations", 6.0),
        ("Legal firms", 5.2),
        ("Financial services (FCA-regulated)", 4.4),
    ]
    data.sort(key=lambda r: r[1])
    labels = [d[0] for d in data]
    values = [d[1] for d in data]
    ramp = list(reversed(RAMP))[: len(values)]

    fig, ax = plt.subplots(figsize=(8.6, 4.5), dpi=220)
    bars = ax.barh(labels, values, height=0.6,
                   color=[ramp[i] for i in range(len(values))])
    for bar, v in zip(bars, values):
        ax.text(v + 0.12, bar.get_y() + bar.get_height() / 2, f"{v:.1f}",
                va="center", ha="left", fontsize=8.6, color=TEXT, fontweight="bold")
    ax.set_xlim(0, 10)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    _clean(ax, xgrid=True)
    ax.set_xlabel("Weighted composite score (1–10)", fontsize=8, labelpad=8)
    ax.tick_params(axis="y", labelsize=8.6)
    fig.tight_layout(pad=0.6)
    fig.savefig(f"{OUT}/market-scores.png", facecolor=SURFACE)
    plt.close(fig)


def economic_impact():
    """Illustrative hours per month on five accountancy workflows, before and after.

    Two series and therefore a legend, plus direct labels on the marks — identity is never
    carried by colour alone. The title carries the ILLUSTRATIVE label so it cannot be
    cropped away from the numbers.
    """
    rows = [
        ("Chasing client records", 46, 12),
        ("Routine client email replies", 38, 14),
        ("Internal 'where do I find…' questions", 22, 6),
        ("Onboarding paperwork & checklists", 18, 7),
        ("Deadline tracking & reminders", 14, 3),
    ]
    labels = [r[0] for r in rows]
    before = [r[1] for r in rows]
    after = [r[2] for r in rows]
    y = range(len(rows))
    h = 0.34

    fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=220)
    # Offsets are negated because the axis is inverted below: "Today" must sit above
    # "With NOVA" in each pair, matching the order the legend reads them.
    b1 = ax.barh([i - h / 2 - 0.02 for i in y], before, height=h,
                 color=SERIES_1, label="Today (manual)")
    b2 = ax.barh([i + h / 2 + 0.02 for i in y], after, height=h,
                 color=SERIES_2, label="With a governed workforce (assumed)")
    for bars, vals in ((b1, before), (b2, after)):
        for bar, v in zip(bars, vals):
            ax.text(v + 0.7, bar.get_y() + bar.get_height() / 2, f"{v}h",
                    va="center", ha="left", fontsize=7.8, color=TEXT)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8.4)
    # Largest at the top: the reader's eye starts there and the list is ranked.
    ax.invert_yaxis()
    ax.set_xlim(0, 56)
    _clean(ax, xgrid=True)
    ax.set_xlabel("Hours per month, one 30-person practice", fontsize=8, labelpad=8)
    ax.set_title("ILLUSTRATIVE ASSUMPTION — not a measured result", fontsize=8.6,
                 color="#FAB219", loc="left", pad=28, fontweight="bold")
    # Above the plot, never inside it: a legend sitting over the bars is the most common
    # way a two-series bar chart becomes unreadable at the long end.
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=2, frameon=False,
                    fontsize=8, handlelength=1.1, handleheight=0.9, columnspacing=1.6)
    for t in leg.get_texts():
        t.set_color(TEXT_MUTED)
    fig.tight_layout(pad=0.6)
    fig.savefig(f"{OUT}/economic-impact.png", facecolor=SURFACE)
    plt.close(fig)


def revenue_ramp():
    """Illustrative monthly recurring revenue under one stated pricing assumption.

    Change over time, one measure: a line. One series, so no legend — the title names it.
    The final point is direct-labelled; intermediate points are not, because a number on
    every point is noise.
    """
    months = list(range(1, 13))
    # Stated assumption, also written in the document beside this figure:
    # customers live at month end, £1,400/month each.
    customers = [0, 1, 1, 2, 3, 4, 5, 6, 8, 9, 11, 13]
    mrr = [c * 1400 for c in customers]

    fig, ax = plt.subplots(figsize=(8.6, 3.5), dpi=220)
    ax.plot(months, mrr, color=SERIES_1, linewidth=2.0, marker="o", markersize=5,
            markerfacecolor=SERIES_1, markeredgecolor=SURFACE, markeredgewidth=1.6)
    ax.fill_between(months, mrr, color=SERIES_1, alpha=0.13)
    ax.annotate(f"£{mrr[-1]:,}/mo\n({customers[-1]} customers)",
                xy=(12, mrr[-1]), xytext=(-10, -46), textcoords="offset points",
                fontsize=8.6, color=TEXT, fontweight="bold", ha="right")
    ax.set_xticks(months)
    ax.set_xticklabels([f"M{m}" for m in months], fontsize=8)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"£{int(v):,}"))
    ax.set_ylim(0, max(mrr) * 1.28)
    _clean(ax, ygrid=True)
    ax.set_title("ILLUSTRATIVE SCENARIO — not a forecast. Assumes £1,400/month per "
                 "customer, no churn.", fontsize=8.4, color="#FAB219", loc="left",
                 pad=12, fontweight="bold")
    fig.tight_layout(pad=0.6)
    fig.savefig(f"{OUT}/revenue-ramp.png", facecolor=SURFACE)
    plt.close(fig)


def capability_status():
    """How many audited capabilities sit on each rung of the evidence ladder.

    Magnitude across ordered categories — the order is the ladder itself, so it is not
    re-sorted by value. One hue, deliberately: these are rungs of one scale, not identities.
    """
    rungs = [
        ("Declared", 6),
        ("Wired", 5),
        ("Enforced", 7),
        ("Tested", 21),
        ("Live local proven", 14),
        ("Live FIELD proven", 0),
    ]
    labels = [r[0] for r in rungs]
    values = [r[1] for r in rungs]
    colours = ["#5598E7", "#3987E5", "#2A78D6", "#256ABF", "#199E70", "#E8563F"]

    fig, ax = plt.subplots(figsize=(8.6, 3.0), dpi=220)
    bars = ax.bar(labels, values, width=0.55, color=colours)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, str(v), ha="center",
                fontsize=9, color=TEXT, fontweight="bold")
    # Placed above the (absent) zero bar's value label rather than on top of it.
    ax.text(len(values) - 1, 3.4, "nothing has run\non real AWS yet", ha="center",
            fontsize=7.6, color="#E8563F", fontweight="bold", linespacing=1.4)
    ax.set_ylim(0, max(values) * 1.3)
    _clean(ax, ygrid=True)
    ax.tick_params(axis="x", labelsize=8.2)
    ax.set_ylabel("Capabilities at this rung", fontsize=8, labelpad=8)
    fig.tight_layout(pad=0.6)
    fig.savefig(f"{OUT}/capability-status.png", facecolor=SURFACE)
    plt.close(fig)


def positioning_matrix():
    """Where NOVA sits against adjacent categories.

    Two axes a buyer actually chooses on: how much governance the platform imposes, and
    whose infrastructure it runs in. Labelled points, no legend.
    """
    points = [
        ("Zapier / Make", 1.5, 2.3, TEXT_MUTED),
        ("Microsoft / Salesforce copilots", 4.2, 1.8, TEXT_MUTED),
        ("Hosted agent platforms", 5.0, 3.4, TEXT_MUTED),
        ("AI automation agencies", 2.2, 4.3, TEXT_MUTED),
        ("Enterprise AI platforms", 7.6, 6.2, TEXT_MUTED),
        ("NOVA", 7.8, 8.6, SERIES_1),
    ]
    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=220)
    for name, x, y, colour in points:
        is_nova = name == "NOVA"
        ax.scatter([x], [y], s=210 if is_nova else 120, color=colour,
                   edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.annotate(name, (x, y), xytext=(0, 13 if is_nova else -17),
                    textcoords="offset points", ha="center",
                    fontsize=8.6 if is_nova else 7.8,
                    color=TEXT if is_nova else TEXT_MUTED,
                    fontweight="bold" if is_nova else "normal")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.set_xlabel("Governance depth  →  policy, approvals, audit, per-agent permissions",
                  fontsize=8, labelpad=10)
    ax.set_ylabel("Runs in the customer's own infrastructure  →", fontsize=8, labelpad=10)
    # An axis with no ticks needs its own reference marks, or the points float.
    ax.axvline(5, color=GRID, linewidth=0.9, linestyle=(0, (4, 4)), zorder=1)
    ax.axhline(5, color=GRID, linewidth=0.9, linestyle=(0, (4, 4)), zorder=1)
    for x, y, note in ((0.4, 9.4, "Your infrastructure,\nlittle governance"),
                       (9.6, 9.4, "Your infrastructure,\ngoverned"),
                       (0.4, 0.5, "Someone else's cloud,\nlittle governance"),
                       (9.6, 0.5, "Someone else's cloud,\ngoverned")):
        ax.text(x, y, note, fontsize=6.6, color="#54648A", ha="left" if x < 5 else "right",
                va="top" if y > 5 else "bottom", linespacing=1.5)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.6)
    fig.savefig(f"{OUT}/positioning.png", facecolor=SURFACE)
    plt.close(fig)


def architecture():
    """The layer diagram: who calls what, and where the boundary between NOVA and the
    runtime actually falls. Drawn rather than described because the boundary is the single
    most misunderstood thing about this product."""
    fig, ax = plt.subplots(figsize=(8.6, 6.4), dpi=220)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    layers = [
        (86, "Customer", "Partners, practice managers, operations staff", "#1B2438", TEXT_MUTED),
        (73, "NOVA Control Center", "React dashboard · strict CSP · bearer auth · viewer/admin",
         "#1D3352", SERIES_1),
        (60, "NOVA Governance", "Policy compile → decide · RBAC · approvals · write-ahead audit",
         "#1D3352", SERIES_1),
        (47, "NOVA Workforce", "Agent specs · objectives · routing · governed automations",
         "#1D3352", SERIES_1),
        (34, "Hermes Runtime  (open source, ~600k LOC — NOVA does not own this)",
         "Profiles · task board · dispatcher · cron ticker · plugins · MCP · channels",
         "#2A2033", "#D95926"),
        (21, "AWS account — the customer's own",
         "EC2 · EBS+KMS · ECR · IAM+boundary · SSM · CloudWatch · Secrets Manager · Bedrock",
         "#1B2438", TEXT_MUTED),
    ]
    for y, title, sub, fill, accent in layers:
        ax.add_patch(plt.Rectangle((6, y), 88, 10.4, facecolor=fill, edgecolor=GRID,
                                   linewidth=1.0, zorder=2))
        ax.add_patch(plt.Rectangle((6, y), 1.2, 10.4, facecolor=accent, edgecolor="none",
                                   zorder=3))
        ax.text(10, y + 6.6, title, fontsize=9.4, color=TEXT, fontweight="bold", zorder=4)
        ax.text(10, y + 2.6, sub, fontsize=7.2, color=TEXT_MUTED, zorder=4)

    # One arrow per gap. Each box spans y..y+10.4, so the gap below a box at y_upper
    # runs from y_upper down to y_lower+10.4 — 2.6 units, and the arrow fills it.
    for upper in (86, 73, 60, 47, 34):
        ax.annotate("", xy=(50, upper - 2.5), xytext=(50, upper - 0.1),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.4",
                                    color="#5A7099", linewidth=1.5))

    ax.text(50, 12.5, "Customer systems reached from inside that account:",
            fontsize=7.6, color=TEXT_MUTED, ha="center")
    ax.text(50, 8.0,
            "S3-mirrored knowledge   ·   22 channel platforms   ·   65 MCP servers   "
            "·   36 grantable plugins",
            fontsize=8, color=TEXT, ha="center", fontweight="bold")
    ax.text(50, 3.4,
            "NOVA holds no customer credential. Each integration is a separate IAM role "
            "assumed with an ExternalId.",
            fontsize=7, color=TEXT_MUTED, ha="center", style="italic")

    fig.tight_layout(pad=0.3)
    fig.savefig(f"{OUT}/architecture.png", facecolor=SURFACE)
    plt.close(fig)


def funnel():
    """The acquisition funnel as counts, with the conversion written between stages.

    A funnel is a sequence of magnitudes, so the bars are one hue darkening down the
    sequence and every stage is direct-labelled — the shape alone never carries a number.
    """
    stages = [
        ("Prospects researched", 100),
        ("Contacted (multi-touch)", 100),
        ("Replied", 20),
        ("Discovery call", 10),
        ("Demo", 6),
        ("Pilot proposal", 3),
        ("Pilot signed", 1),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 3.6), dpi=220)
    labels = [s[0] for s in stages]
    values = [s[1] for s in stages]
    colours = RAMP[2:2 + len(values)]
    bars = ax.barh(range(len(values))[::-1], values, height=0.62, color=colours)
    for bar, v in zip(bars, values):
        ax.text(v + 1.4, bar.get_y() + bar.get_height() / 2, str(v), va="center",
                fontsize=8.6, color=TEXT, fontweight="bold")
    ax.set_yticks(range(len(values))[::-1])
    ax.set_yticklabels(labels, fontsize=8.4)
    ax.set_xlim(0, 118)
    _clean(ax, xgrid=True)
    ax.set_title("ILLUSTRATIVE PLANNING TARGET — set from cold-outreach norms, "
                 "not from NOVA's own results", fontsize=8, color="#FAB219", loc="left",
                 pad=12, fontweight="bold")
    fig.tight_layout(pad=0.6)
    fig.savefig(f"{OUT}/funnel.png", facecolor=SURFACE)
    plt.close(fig)


def build_all():
    vertical_scores()
    economic_impact()
    revenue_ramp()
    capability_status()
    positioning_matrix()
    architecture()
    funnel()


if __name__ == "__main__":
    build_all()
    print("charts written to", OUT)
