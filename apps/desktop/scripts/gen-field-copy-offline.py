#!/usr/bin/env python3
"""Generate src/i18n/es-field-copy.ts OFFLINE from the existing disk cache.

Uses /tmp/hermes_es_field_cache.json (english -> spanish). For any string not
in cache, falls back to the original English (so the file is always complete
and valid). Replaces the network translator.
"""
import json
import re
from pathlib import Path

CONST = Path("/home/the_wolf/.hermes/hermes-agent/apps/desktop/src/app/settings/constants.ts")
OUT = Path("/home/the_wolf/.hermes/hermes-agent/apps/desktop/src/i18n/es-field-copy.ts")
CACHE_PATH = Path("/tmp/hermes_es_field_cache.json")

CACHE = {}
if CACHE_PATH.exists():
    try:
        CACHE = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        CACHE = {}


def tr_literal(s: str) -> str:
    inner = s[1:-1]
    es = CACHE.get(inner, inner)  # fallback to English if missing
    return "'" + es + "'"


def extract_call(src: str, fn_name: str) -> str:
    idx = src.find(fn_name + "(")
    i = src.index("{", idx)
    depth = 0
    for j in range(i, len(src)):
        c = src[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return src[i + 1:j]
    raise RuntimeError("unbalanced")


text = CONST.read_text(encoding="utf-8")
labels_block = extract_call(text, "defineFieldCopy")
second = text.find("defineFieldCopy(", text.find("defineFieldCopy(") + 1)
desc_block = extract_call(text[second:], "defineFieldCopy")

missing = 0
total = 0
out = open(OUT, "w", encoding="utf-8")
out.write(
    "// AUTO-GENERATED Spanish mirror of FIELD_LABELS / FIELD_DESCRIPTIONS from\n"
    "// apps/desktop/src/app/settings/constants.ts. Imported by src/i18n/es.ts so the\n"
    "// Desktop Settings form renders in Spanish without altering the shared\n"
    "// (TUI/CLI) constants. Regenerate with scripts/translate-field-copy.py.\n"
    "export const FIELD_LABELS_ES = {\n"
)
for line in labels_block.splitlines():
    m = re.match(r"^(\s*)([\w.]+):\s*('[^']*')\s*,?$", line)
    if m:
        total += 1
        if m.group(3)[1:-1] not in CACHE:
            missing += 1
        out.write(f"{m.group(1)}{m.group(2)}: {tr_literal(m.group(3))},\n")
    else:
        out.write(line + "\n")
out.write("} as const\n\n")
out.write("export const FIELD_DESCRIPTIONS_ES = {\n")
for line in desc_block.splitlines():
    m = re.match(r"^(\s*)([\w.]+):\s*('[^']*')\s*,?$", line)
    if m:
        total += 1
        if m.group(3)[1:-1] not in CACHE:
            missing += 1
        out.write(f"{m.group(1)}{m.group(2)}: {tr_literal(m.group(3))},\n")
    else:
        out.write(line + "\n")
out.write("} as const\n")
out.close()
print(f"WROTE {OUT}: {total} fields, {missing} fell back to English (not in cache)", flush=True)
