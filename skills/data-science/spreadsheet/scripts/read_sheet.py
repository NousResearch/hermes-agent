#!/usr/bin/env python3
"""Read xlsx/csv/tsv files and output structured data.

Usage:
    python read_sheet.py data.xlsx
    python read_sheet.py data.xlsx --sheet "Sales" --format markdown
    python read_sheet.py data.csv --range A1:D20
"""

import argparse
import csv
import json
import os
import sys


def _col_letter_to_index(letter):
    """Convert column letter(s) to 0-based index. A=0, B=1, ..., Z=25, AA=26."""
    result = 0
    for ch in letter.upper():
        result = result * 26 + (ord(ch) - ord("A") + 1)
    return result - 1


def _parse_range(range_str):
    """Parse a range like 'A1:D20' into (min_row, min_col, max_row, max_col) all 0-based."""
    import re

    m = re.match(r"([A-Za-z]+)(\d+):([A-Za-z]+)(\d+)", range_str)
    if not m:
        raise ValueError(f"Invalid range: {range_str}")
    min_col = _col_letter_to_index(m.group(1))
    min_row = int(m.group(2)) - 1
    max_col = _col_letter_to_index(m.group(3))
    max_row = int(m.group(4)) - 1
    return min_row, min_col, max_row, max_col


def read_xlsx(path, sheet=None, cell_range=None):
    """Read an xlsx file and return a list of row dicts.

    Args:
        path: Path to the xlsx file.
        sheet: Sheet name to read. Defaults to the active sheet.
        cell_range: Optional range string like 'A1:D20'.

    Returns:
        List of dicts where keys are column headers.
    """
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        ws = wb[sheet] if sheet else wb.active
        rows = list(ws.iter_rows(values_only=True))
    finally:
        wb.close()

    if not rows:
        return []

    if cell_range:
        min_row, min_col, max_row, max_col = _parse_range(cell_range)
        rows = [
            tuple(row[min_col : max_col + 1] if len(row) > min_col else ())
            for row in rows[min_row : max_row + 1]
        ]

    headers = [str(h) if h is not None else f"col_{i}" for i, h in enumerate(rows[0])]
    result = []
    for row in rows[1:]:
        record = {}
        for i, header in enumerate(headers):
            val = row[i] if i < len(row) else None
            record[header] = val
        result.append(record)
    return result


def read_csv(path, delimiter=","):
    """Read a CSV/TSV file and return a list of row dicts.

    Args:
        path: Path to the csv/tsv file.
        delimiter: Field delimiter (default: comma).

    Returns:
        List of dicts where keys are column headers.
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        return [dict(row) for row in reader]


def _detect_delimiter(path):
    """Guess delimiter from file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".tsv":
        return "\t"
    return ","


def read_sheet(path, sheet=None, cell_range=None):
    """Read any supported spreadsheet format.

    Dispatches to read_xlsx or read_csv based on file extension.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return read_xlsx(path, sheet=sheet, cell_range=cell_range)
    elif ext in (".csv", ".tsv"):
        return read_csv(path, delimiter=_detect_delimiter(path))
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def format_markdown(rows):
    """Format row dicts as a markdown table."""
    if not rows:
        return "(empty)"
    headers = list(rows[0].keys())
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        vals = [str(row.get(h, "")) for h in headers]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Read spreadsheet files to structured output")
    parser.add_argument("path", help="Path to xlsx, csv, or tsv file")
    parser.add_argument("--sheet", default=None, help="Sheet name (xlsx only)")
    parser.add_argument("--range", default=None, help="Cell range like A1:D20")
    parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        help="Output format (default: json)",
    )
    args = parser.parse_args()

    rows = read_sheet(args.path, sheet=args.sheet, cell_range=args.range)

    if args.format == "markdown":
        print(format_markdown(rows))
    else:
        print(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
