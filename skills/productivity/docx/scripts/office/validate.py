"""
Command line tool to validate Office document XML files against XSD schemas and tracked changes.

Usage:
    python validate.py <path> [--original <original_file>] [--auto-repair] [--author NAME]

The first argument can be either:
- An unpacked directory containing the Office document XML files
- A packed Office file (.docx/.pptx/.xlsx or .dotx/.potx/.xltx template) which will be unpacked to a temp directory

Auto-repair fixes:
- paraId/durableId values that exceed OOXML limits
- Missing xml:space="preserve" on w:t elements with whitespace
"""

import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

import defusedxml.ElementTree as ET
from defusedxml.common import DefusedXmlException

from helpers import OOXML_FAMILY, rezip, safe_extract
from validators import DOCXSchemaValidator, PPTXSchemaValidator, RedliningValidator

WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _fail(message: str):
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(2)


def _has_tracked_changes(unpacked_dir: Path) -> bool:
    document = unpacked_dir / "word" / "document.xml"
    if not document.is_file():
        return False
    try:
        root = ET.parse(document).getroot()
    except (ET.ParseError, DefusedXmlException):
        return False  
    tracked = {f"{{{WORD_NS}}}ins", f"{{{WORD_NS}}}del"}
    return any(elem.tag in tracked for elem in root.iter())


def main():
    parser = argparse.ArgumentParser(description="Validate Office document XML files")
    parser.add_argument(
        "path",
        help="Path to unpacked directory or packed Office file (.docx/.pptx/.xlsx or .dotx/.potx/.xltx)",
    )
    parser.add_argument(
        "--original",
        required=False,
        default=None,
        help="Path to original file (.docx/.pptx/.xlsx or .dotx/.potx/.xltx). If omitted, all XSD errors are reported and redlining validation is skipped.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--auto-repair",
        action="store_true",
        help="Automatically repair common issues (hex IDs, whitespace preservation). "
        "Modifies the input in place: repairs to a packed file are written back to it.",
    )
    parser.add_argument(
        "--author",
        default=None,
        help="The name you are redlining under. Passing it turns on the "
        "tracked-change check: any text differing from --original without a "
        "<w:ins>/<w:del> recording it is reported. Untracked edits carry no "
        "author, so the check covers them whoever made them — the name marks "
        "the run as redlining work and is not used to filter. Requires "
        "--original; docx only.",
    )
    args = parser.parse_args()

    if args.author is not None and not args.original:
        _fail("--author requires --original")

    path = Path(args.path)
    if not path.exists():
        _fail(f"{path} does not exist")

    original_file = None
    if args.original:
        original_file = Path(args.original)
        if not original_file.is_file():
            _fail(f"{original_file} is not a file")
        if original_file.suffix.lower() not in OOXML_FAMILY:
            _fail(f"{original_file} must be one of: {', '.join(sorted(OOXML_FAMILY))}")

    family = OOXML_FAMILY.get((original_file or path).suffix.lower())
    if family is None:
        _fail(
            f"Cannot determine file type from {path}. Use --original or provide one of: {', '.join(sorted(OOXML_FAMILY))}."
        )

    if args.author is not None and family != "docx":
        _fail(f"--author only applies to docx files, not {family}")

    packed_file = None
    temp_dir_ctx = None
    if path.is_file() and path.suffix.lower() in OOXML_FAMILY:
        packed_file = path
        temp_dir_ctx = tempfile.TemporaryDirectory()
        unpacked_dir = Path(temp_dir_ctx.name)
        try:
            with zipfile.ZipFile(path, "r") as zf:
                safe_extract(zf, unpacked_dir)
        except (zipfile.BadZipFile, ValueError, OSError) as e:
            _fail(f"cannot unpack {path}: {e}")
    else:
        if not path.is_dir():
            _fail(f"{path} is not a directory or Office file")
        unpacked_dir = path

    match family:
        case "docx":
            validators = [
                DOCXSchemaValidator(unpacked_dir, original_file, verbose=args.verbose),
            ]
            if args.author is not None:
                validators.append(
                    RedliningValidator(unpacked_dir, original_file, verbose=args.verbose)  
                )
            elif original_file and _has_tracked_changes(unpacked_dir):
                print(
                    "Note: this document has tracked changes; they were not "
                    "checked against the original (pass --author to check)."
                )
        case "pptx":
            validators = [
                PPTXSchemaValidator(unpacked_dir, original_file, verbose=args.verbose),
            ]
        case "xlsx":
            exts = ", ".join(k for k, v in sorted(OOXML_FAMILY.items()) if v == "xlsx")
            print(
                f"No XSD schema validation is performed for xlsx-family files ({exts}). "
                "For formula-error checking, use scripts/recalc.py instead."
            )
            sys.exit(0)
        case _:
            print(f"Error: Validation not supported for file type {family}")
            sys.exit(1)

    if args.auto_repair:
        total_repairs = sum(v.repair() for v in validators)
        if total_repairs:
            print(f"Auto-repaired {total_repairs} issue(s)")
            if packed_file is not None:
                rezip(unpacked_dir, packed_file)
                print(f"Wrote repaired file to {packed_file}")

    success = all([v.validate() for v in validators])

    if temp_dir_ctx is not None:
        temp_dir_ctx.cleanup()

    if success:
        print("All validations PASSED!")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
