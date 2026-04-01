#!/usr/bin/env python3
"""Create pivot tables from spreadsheet data.

Usage:
    python pivot.py sales.xlsx --rows Category --values Revenue --aggfunc sum
    python pivot.py sales.xlsx --rows Category --cols Month --values Revenue --aggfunc sum --output pivot.xlsx
"""

import argparse
import json
import os
import sys


def create_pivot(path, rows, cols=None, values=None, aggfunc="sum", sheet=None):
    """Create a pivot table from spreadsheet data.

    Args:
        path: Path to input file (xlsx/csv).
        rows: Column name(s) for pivot rows.
        cols: Optional column name(s) for pivot columns.
        values: Column name for values to aggregate.
        aggfunc: Aggregation function (sum, mean, count, min, max, median).
        sheet: Sheet name (xlsx only).

    Returns:
        List of row dicts representing the pivot table.
    """
    import pandas as pd

    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name=sheet or 0)
    elif ext == ".tsv":
        df = pd.read_csv(path, delimiter="\t")
    else:
        df = pd.read_csv(path)

    if isinstance(rows, str):
        rows = [rows]
    if isinstance(cols, str):
        cols = [cols]

    pivot_args = {"index": rows, "aggfunc": aggfunc}
    if cols:
        pivot_args["columns"] = cols
    if values:
        pivot_args["values"] = values

    pt = pd.pivot_table(df, **pivot_args)
    pt = pt.reset_index()

    # Flatten MultiIndex columns if present
    if hasattr(pt.columns, "levels"):
        pt.columns = ["_".join(str(c) for c in col).strip("_") for col in pt.columns.values]

    return json.loads(pt.to_json(orient="records", default_handler=str))


def main():
    parser = argparse.ArgumentParser(description="Create pivot tables from spreadsheet data")
    parser.add_argument("path", help="Input file path")
    parser.add_argument("--rows", required=True, nargs="+", help="Row grouping column(s)")
    parser.add_argument("--cols", nargs="+", default=None, help="Column grouping column(s)")
    parser.add_argument("--values", default=None, help="Values column to aggregate")
    parser.add_argument(
        "--aggfunc",
        default="sum",
        choices=["sum", "mean", "count", "min", "max", "median"],
        help="Aggregation function (default: sum)",
    )
    parser.add_argument("--sheet", default=None, help="Sheet name (xlsx only)")
    parser.add_argument("--output", default=None, help="Output file path (writes xlsx/csv)")
    args = parser.parse_args()

    result = create_pivot(
        args.path,
        rows=args.rows,
        cols=args.cols,
        values=args.values,
        aggfunc=args.aggfunc,
        sheet=args.sheet,
    )

    if args.output:
        import pandas as pd

        df = pd.DataFrame(result)
        ext = os.path.splitext(args.output)[1].lower()
        if ext in (".xlsx", ".xls"):
            df.to_excel(args.output, index=False)
        else:
            df.to_csv(args.output, index=False)
        print(f"Pivot table written to {args.output} ({len(result)} rows)")
    else:
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
