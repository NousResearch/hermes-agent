"""
Feishu Card Builder — compose interactive card messages for Feishu/Lark.

Provides pure-Python builder classes that construct Feishu interactive card
JSON (msg_type=interactive) without manual JSON construction.

Usage:
    from feishu_card_builder import MetricCard, TableCard, ListCard

    card = MetricCard("📊 Report", [
        {"label": "Total", "value": "1,247", "color": "blue"},
        {"label": "Success", "value": "98.3%", "color": "green"},
    ])
    # Pass card.to_dict() to the gateway send method
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

# ── Color constants ───────────────────────────────────────

CardColor = Literal["blue", "green", "red", "yellow", "orange", "purple", "grey", "default"]


# ── Base class ────────────────────────────────────────────

class BaseCard:
    """Base class for all Feishu interactive cards."""

    def __init__(self, header: str, color: CardColor = "blue"):
        self.header = header
        self.color = color
        self._elements: List[Dict[str, Any]] = []

    def _make_header(self) -> Dict[str, Any]:
        return {
            "title": {"tag": "plain_text", "content": self.header},
            "template": self.color,
        }

    def _add_div(self, text: str) -> None:
        self._elements.append({
            "tag": "div",
            "text": {"tag": "lark_md", "content": text},
        })

    def _add_hr(self) -> None:
        self._elements.append({"tag": "hr"})

    def _add_note(self, text: str) -> None:
        self._elements.append({
            "tag": "note",
            "element": {"tag": "plain_text", "content": text},
        })

    def to_dict(self) -> Dict[str, Any]:
        """Return the complete Feishu interactive card JSON."""
        return {
            "config": {"wide_screen_mode": False},
            "header": self._make_header(),
            "elements": self._elements,
        }


# ── Metric Card ───────────────────────────────────────────

class MetricCard(BaseCard):
    """A card displaying key-value metric pairs.

    Example:
        MetricCard("📊 Status", [
            {"label": "Uptime", "value": "99.9%", "color": "green"},
            {"label": "Errors", "value": "0", "color": "default"},
        ])
    """

    def __init__(
        self,
        header: str,
        metrics: List[Dict[str, str]],
        color: CardColor = "blue",
        note: str = "",
    ):
        super().__init__(header, color)
        self._build(metrics, note)

    def _build(self, metrics: List[Dict[str, str]], note: str) -> None:
        lines = []
        for m in metrics:
            label = m.get("label", "")
            value = m.get("value", "")
            # Use bold for the label, plain for value
            lines.append(f"**{label}**　{value}")
        self._add_div("\n".join(lines))

        if note:
            self._add_note(note)


# ── Table Card ────────────────────────────────────────────

class TableCard(BaseCard):
    """A card with a header row and data rows, styled as a table.

    Note: Feishu does not have a native <table> element in card JSON 2.0.
    This renders tables as markdown-style aligned rows in a div, which
    Feishu's post API renders as bordered grids (as of mid-2026).

    Example:
        TableCard("📋 Rankings", ["Rank", "Name", "Score"], [
            ["1", "Alice", "98"],
            ["2", "Bob", "87"],
        ])
    """

    def __init__(
        self,
        header: str,
        columns: List[str],
        rows: List[List[str]],
        color: CardColor = "blue",
        note: str = "",
    ):
        super().__init__(header, color)
        self._build(columns, rows, note)

    def _build(self, columns: List[str], rows: List[List[str]], note: str) -> None:
        # Build markdown table
        lines = []
        # Header row
        lines.append("| " + " | ".join(columns) + " |")
        # Separator row
        lines.append("| " + " | ".join("---" for _ in columns) + " |")
        # Data rows
        for row in rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")

        self._add_div("\n".join(lines))

        if note:
            self._add_note(note)


# ── List Card ─────────────────────────────────────────────

class ListCard(BaseCard):
    """A card with a bulleted list of items.

    Example:
        ListCard("📌 Tasks", [
            ("✅", "Review PR", "Pending review by Alice"),
            ("⏳", "Run tests", "Waiting for CI"),
        ])
    """

    def __init__(
        self,
        header: str,
        items: List[tuple[str, str, Optional[str]]],
        color: CardColor = "blue",
        note: str = "",
    ):
        """items: list of (icon, title, description_or_None)"""
        super().__init__(header, color)
        self._build(items, note)

    def _build(self, items: List[tuple[str, str, Optional[str]]], note: str) -> None:
        lines = []
        for icon, title, desc in items:
            if desc:
                lines.append(f"{icon} **{title}** — {desc}")
            else:
                lines.append(f"{icon} **{title}**")

        self._add_div("\n".join(lines))

        if note:
            self._add_note(note)


# ── Convenience ───────────────────────────────────────────

def build_metric_card(
    header: str,
    label: str,
    value: str,
    color: CardColor = "blue",
) -> Dict[str, Any]:
    """Quick one-liner for a single-metric card."""
    return MetricCard(header, [{"label": label, "value": value}], color=color).to_dict()


# ── CLI ───────────────────────────────────────────────────

def _cli():
    """CLI entry point for agent usage via terminal."""
    import argparse
    import sys
    import json

    parser = argparse.ArgumentParser(description="Build Feishu interactive cards")
    sub = parser.add_subparsers(dest="command", required=True)

    # metric
    m = sub.add_parser("metric", help="Build a metric/KPI card")
    m.add_argument("--header", required=True)
    m.add_argument("--color", default="blue", choices=["blue","green","red","yellow","orange","purple","grey"])
    m.add_argument("--items", nargs="+", required=True,
                   help='Items in "label:value" format, e.g. "Uptime:99.9%"')

    # table
    t = sub.add_parser("table", help="Build a table card")
    t.add_argument("--header", required=True)
    t.add_argument("--color", default="blue")
    t.add_argument("--columns", required=True, help='Comma-separated: "Rank,Name,Score"')
    t.add_argument("--rows", nargs="+", required=True,
                   help='Rows in comma-separated format: "1,Alice,98" "2,Bob,87"')

    # list
    l = sub.add_parser("list", help="Build a list card")
    l.add_argument("--header", required=True)
    l.add_argument("--color", default="blue")
    l.add_argument("--items", nargs="+", required=True,
                   help='Items in "icon:title:desc" or "icon:title" format')

    args = parser.parse_args()

    if args.command == "metric":
        metrics = []
        for item in args.items:
            if ":" in item:
                label, value = item.split(":", 1)
                metrics.append({"label": label, "value": value})
        card = MetricCard(args.header, metrics, color=args.color)
    elif args.command == "table":
        columns = [c.strip() for c in args.columns.split(",")]
        rows = [r.split(",") for r in args.rows]
        card = TableCard(args.header, columns, rows, color=args.color)
    elif args.command == "list":
        items_list = []
        for item in args.items:
            parts = item.split(":", 2)
            icon = parts[0]
            title = parts[1] if len(parts) > 1 else ""
            desc = parts[2] if len(parts) > 2 else None
            items_list.append((icon, title, desc))
        card = ListCard(args.header, items_list, color=args.color)

    print(json.dumps(card.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
