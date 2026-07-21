#!/usr/bin/env python3
"""Write JSON data to xlsx or csv files.

Usage:
    python write_sheet.py output.xlsx --input data.json
    echo '[{"a":1}]' | python write_sheet.py output.csv
    python write_sheet.py output.xlsx --input data.json --sheet Results
"""

import argparse
import csv
import json
import os
import sys


def write_xlsx(data, path, sheet_name="Sheet1"):
    """Write a list of row dicts to an xlsx file.

    Args:
        data: List of dicts with consistent keys.
        path: Output file path.
        sheet_name: Name of the sheet to create.
    """
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet_name

    if not data:
        wb.save(path)
        return

    headers = list(data[0].keys())
    ws.append(headers)
    for row in data:
        ws.append([row.get(h) for h in headers])

    wb.save(path)


def write_csv(data, path, delimiter=","):
    """Write a list of row dicts to a CSV file.

    Args:
        data: List of dicts with consistent keys.
        path: Output file path.
        delimiter: Field delimiter (default: comma).
    """
    if not data:
        with open(path, "w", newline="", encoding="utf-8") as f:
            pass
        return

    headers = list(data[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(data)


def write_sheet(data, path, sheet_name="Sheet1"):
    """Write data to the appropriate format based on file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        write_xlsx(data, path, sheet_name=sheet_name)
    elif ext == ".tsv":
        write_csv(data, path, delimiter="\t")
    else:
        write_csv(data, path)


def main():
    parser = argparse.ArgumentParser(description="Write JSON data to spreadsheet files")
    parser.add_argument("path", help="Output file path (.xlsx, .csv, .tsv)")
    parser.add_argument("--input", dest="input_file", default=None, help="JSON input file (reads stdin if omitted)")
    parser.add_argument("--sheet", default="Sheet1", help="Sheet name (xlsx only, default: Sheet1)")
    args = parser.parse_args()

    if args.input_file:
        with open(args.input_file, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json.load(sys.stdin)

    if not isinstance(data, list):
        print("Error: input must be a JSON array of objects", file=sys.stderr)
        sys.exit(1)

    write_sheet(data, args.path, sheet_name=args.sheet)
    print(f"Wrote {len(data)} rows to {args.path}")


if __name__ == "__main__":
    main()
