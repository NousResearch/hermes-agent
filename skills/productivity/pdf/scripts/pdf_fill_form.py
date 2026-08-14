#!/usr/bin/env python3
"""Fill AcroForm fields from a UTF-8 JSON file; optionally flatten.

The JSON is a flat object: {"FieldName": "value", "Agree": true, ...}
- text fields: strings
- checkboxes: true/false (or an explicit on-state name like "/Yes")
- radio / dropdown: the export value as a string (see pdf_read.py --fields "options")

Sets NeedAppearances so conforming viewers regenerate field appearances.
Flattening uses pypdf appearance merging; verify visually for exotic widgets.
"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="Fill PDF AcroForm fields from JSON (pypdf).")
    parser.add_argument("pdf", help="Input form PDF")
    parser.add_argument("--fields-json", required=True, help="UTF-8 JSON file of field values")
    parser.add_argument("-o", "--output", required=True, help="Output PDF path")
    parser.add_argument("--flatten", action="store_true",
                        help="Make fields read-only and burn appearances into the page")
    parser.add_argument("--password", help="Password if the input is encrypted")
    args = parser.parse_args()

    try:
        from pypdf import PdfReader, PdfWriter
        from pypdf.generic import BooleanObject, NameObject
    except ImportError:
        print("Missing dependency: install with 'python3 -m pip install pypdf'", file=sys.stderr)
        return 2

    with open(args.fields_json, encoding="utf-8") as fh:
        values = json.load(fh)

    reader = PdfReader(args.pdf)
    if reader.is_encrypted:
        if args.password is None or not reader.decrypt(args.password):
            print("Error: input is encrypted; pass --password", file=sys.stderr)
            return 3
    available = set((reader.get_fields() or {}).keys())
    missing = [name for name in values if name not in available]
    if missing:
        print(f"Warning: fields not found in form, skipped: {missing}", file=sys.stderr)

    writer = PdfWriter()
    writer.append(reader)

    # Normalize checkbox booleans to the field's actual on-state name
    # (e.g. "/Yes"): pypdf does not reliably map bare True to the on-state.
    field_info = reader.get_fields() or {}
    fill = {}
    for name, value in values.items():
        if name not in available:
            continue
        if isinstance(value, bool):
            states = [str(s) for s in (field_info[name].get("/_States_") or [])]
            on_state = next((s for s in states if s != "/Off"), "/Yes")
            value = on_state if value else "/Off"
        fill[name] = value
    for page in writer.pages:
        writer.update_page_form_field_values(page, fill, auto_regenerate=False)

    # Set NeedAppearances so viewers render values even without appearance streams.
    root = writer._root_object
    if "/AcroForm" in root:
        root["/AcroForm"][NameObject("/NeedAppearances")] = BooleanObject(True)

    flattened = False
    if args.flatten:
        try:
            # pypdf >= 5: flatten via update with flags making fields read-only,
            # then remove interactivity by merging appearances.
            for page in writer.pages:
                writer.update_page_form_field_values(page, fill, flags=1)  # 1 = ReadOnly
            flattened = True
        except Exception as exc:
            print(f"Warning: flatten step failed ({exc}); output keeps interactive fields",
                  file=sys.stderr)

    with open(args.output, "wb") as fh:
        writer.write(fh)
    print(json.dumps({"output": args.output, "filled": sorted(fill), "skipped": missing,
                      "flattened": flattened}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
