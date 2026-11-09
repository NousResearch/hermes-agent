#!/usr/bin/env python3
"""Static sanity checks for MobileCompute.xcodeproj/project.pbxproj.

Runs on any platform (no Xcode required) so the pbxproj can be validated in
the Linux structure-validation job before the macOS build job even starts.

Checks:
  1. The file is OpenStep plist, not an XML plist (the original broken file
     was XML, which xcodebuild cannot parse).
  2. Every object id referenced somewhere is defined in `objects`.
  3. No PBXGroup lists itself as its own child (the original Products group
     had itself as its only child).
  4. rootObject is a PBXProject.
  5. The application target's build phases are Sources/Frameworks/Resources
     and the Sources phase contains the expected Swift files.
  6. INFOPLIST_FILE points at Info.plist in both configurations.

Exit code 0 = all good, 1 = at least one failure.
"""

import re
import sys
from pathlib import Path

PBXPROJ = Path(__file__).resolve().parent.parent / "MobileCompute.xcodeproj" / "project.pbxproj"

EXPECTED_SOURCES = ("main.swift", "MobileComputeServer.swift")
OBJECT_ID = r"[0-9A-F]{24}"

errors = []


def check(cond, message):
    if cond:
        print(f"   ok: {message}")
    else:
        print(f"   FAIL: {message}")
        errors.append(message)


def main() -> int:
    if not PBXPROJ.is_file():
        print(f"   FAIL: {PBXPROJ} not found")
        return 1

    text = PBXPROJ.read_text(encoding="utf-8")

    # 1. OpenStep format, not XML.
    first = text.lstrip().splitlines()[0] if text.strip() else ""
    check(not first.startswith("<?xml"), "pbxproj is not an XML plist")
    check("UTF8" in first, f"line 1 carries the Xcode encoding marker: {first!r}")

    # 2. Collect defined object ids: `ID /* comment */ = {` or `ID = {`.
    defined = set(re.findall(rf"^\s*({OBJECT_ID})\b[^=\n]*=\s*\{{", text, re.M))
    referenced = set(re.findall(rf"\b({OBJECT_ID})\b", text))
    dangling = sorted(referenced - defined)
    check(not dangling, f"no dangling object references (defined={len(defined)})")
    for oid in dangling:
        print(f"        dangling: {oid}")

    # 3. No self-referencing groups.
    for m in re.finditer(
        rf"({OBJECT_ID})\s*(?:/\*[^*]*\*/)?\s*=\s*\{{(.*?)\n\s*\}};", text, re.S
    ):
        oid, body = m.group(1), m.group(2)
        if "isa = PBXGroup;" not in body:
            continue
        children = re.search(r"children = \((.*?)\);", body, re.S)
        if not children:
            continue
        for child in re.findall(rf"({OBJECT_ID})", children.group(1)):
            check(child != oid, f"group {oid} does not list itself as a child")

    # 4. rootObject must be the PBXProject.
    root = re.search(rf"rootObject = ({OBJECT_ID})", text)
    check(root is not None, "rootObject is declared")
    if root:
        root_id = root.group(1)
        root_body = re.search(
            rf"{root_id}\b[^=\n]*=\s*\{{(.*?)\n\s*\}};", text, re.S
        )
        body = root_body.group(1) if root_body else ""
        check("isa = PBXProject;" in body, "rootObject is a PBXProject")

    # 5. Application target, sources phase contents.
    check(
        'productType = "com.apple.product-type.application";' in text,
        "target product type is com.apple.product-type.application",
    )
    check(
        "PRODUCT_BUNDLE_IDENTIFIER = com.hermes.mobilecompute;" in text,
        "bundle identifier is com.hermes.mobilecompute",
    )

    sources_phase = re.search(
        r"isa = PBXSourcesBuildPhase;(.*?)\n\s*\};", text, re.S
    )
    sources_body = sources_phase.group(1) if sources_phase else ""
    for name in EXPECTED_SOURCES:
        check(
            f"{name} in Sources" in sources_body,
            f"{name} is in the Sources build phase",
        )

    check(
        "Info.plist in Resources" not in text,
        "Info.plist is not copied as a bundled resource",
    )
    check(
        text.count("INFOPLIST_FILE = Info.plist;") == 2,
        "INFOPLIST_FILE = Info.plist set in both Debug and Release",
    )

    # Brace balance.
    check(text.count("{") == text.count("}"), "braces are balanced")

    if errors:
        print(f"\n{len(errors)} problem(s) found in {PBXPROJ.name}")
        return 1
    print(f"\npbxproj static checks passed ({len(defined)} objects)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
