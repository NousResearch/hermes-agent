#!/usr/bin/env python3
"""Cell-level diff between two spreadsheet files.

Usage:
    python diff.py old.xlsx new.xlsx
    python diff.py old.xlsx new.xlsx --sheet "Sales"
    python diff.py old.csv new.csv --format markdown
"""

import argparse
import csv
import json
import os
import sys


def _read_as_grid(path, sheet=None):
    """Read a file into a 2D list of cell values (list of rows)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        import openpyxl

        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            ws = wb[sheet] if sheet else wb.active
            return [list(row) for row in ws.iter_rows(values_only=True)]
        finally:
            wb.close()
    else:
        delimiter = "\t" if ext == ".tsv" else ","
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter=delimiter)
            return [row for row in reader]


def _col_index_to_letter(index):
    """Convert 0-based column index to Excel column letter(s)."""
    result = ""
    index += 1
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def diff_sheets(old_path, new_path, sheet=None):
    """Compute cell-level differences between two spreadsheet files.

    Args:
        old_path: Path to the original file.
        new_path: Path to the new file.
        sheet: Sheet name (xlsx only).

    Returns:
        List of change dicts: {"cell": "A1", "type": "changed"|"added"|"removed",
                                "old": value, "new": value}
    """
    old_grid = _read_as_grid(old_path, sheet=sheet)
    new_grid = _read_as_grid(new_path, sheet=sheet)

    max_rows = max(len(old_grid), len(new_grid))
    max_cols = 0
    for row in old_grid + new_grid:
        if len(row) > max_cols:
            max_cols = len(row)

    changes = []
    for r in range(max_rows):
        old_row = old_grid[r] if r < len(old_grid) else []
        new_row = new_grid[r] if r < len(new_grid) else []
        row_max_cols = max(len(old_row), len(new_row))

        for c in range(row_max_cols):
            old_val = old_row[c] if c < len(old_row) else None
            new_val = new_row[c] if c < len(new_row) else None

            # Normalize for comparison
            old_str = str(old_val) if old_val is not None else ""
            new_str = str(new_val) if new_val is not None else ""

            if old_str != new_str:
                cell_ref = f"{_col_index_to_letter(c)}{r + 1}"
                if old_val is None or old_str == "":
                    change_type = "added"
                elif new_val is None or new_str == "":
                    change_type = "removed"
                else:
                    change_type = "changed"
                changes.append({
                    "cell": cell_ref,
                    "type": change_type,
                    "old": old_val,
                    "new": new_val,
                })

    return changes


def format_markdown(changes):
    """Format diff results as a markdown table."""
    if not changes:
        return "No differences found."
    lines = ["| Cell | Type | Old | New |", "| --- | --- | --- | --- |"]
    for ch in changes:
        old = str(ch["old"]) if ch["old"] is not None else ""
        new = str(ch["new"]) if ch["new"] is not None else ""
        lines.append(f"| {ch['cell']} | {ch['type']} | {old} | {new} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Cell-level diff between two spreadsheet files")
    parser.add_argument("old", help="Path to original file")
    parser.add_argument("new", help="Path to new file")
    parser.add_argument("--sheet", default=None, help="Sheet name (xlsx only)")
    parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        help="Output format (default: json)",
    )
    args = parser.parse_args()

    changes = diff_sheets(args.old, args.new, sheet=args.sheet)

    if args.format == "markdown":
        print(format_markdown(changes))
    else:
        print(json.dumps(changes, indent=2, default=str))


if __name__ == "__main__":
    main()
