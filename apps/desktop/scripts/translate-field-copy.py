#!/usr/bin/env python3
"""Generate src/i18n/es-field-copy.ts: Spanish FIELD_LABELS + FIELD_DESCRIPTIONS.
Mirrors apps/desktop/src/app/settings/constants.ts labels, translated, so the
Desktop es locale can render the Settings form in Spanish without mutating the
shared (TUI/CLI) constants.ts. Writes incrementally + disk cache (resumable).
"""
import json
import re
import time
import urllib.parse
import urllib.request
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


def gtx(text: str) -> str:
    if not text.strip():
        return text
    if text in CACHE:
        return CACHE[text]
    for attempt in range(5):
        try:
            q = urllib.parse.quote(text)
            url = ("https://translate.googleapis.com/translate_a/single?client=gtx"
                   "&sl=en&tl=es&dt=t&q=" + q)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=12) as r:
                data = json.loads(r.read().decode("utf-8"))
            res = "".join(seg[0] for seg in data[0] if seg and seg[0])
            CACHE[text] = res
            return res
        except Exception:
            if attempt == 4:
                CACHE[text] = text
                return text
            time.sleep(0.8 * (attempt + 1))
    return text


def extract_call(src: str, fn_name: str) -> str:
    idx = src.find(fn_name + "(")
    if idx < 0:
        raise RuntimeError(f"{fn_name} not found")
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


def tr_literal(s: str) -> str:
    return "'" + gtx(s[1:-1]) + "'"


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
        out.write(f"{m.group(1)}{m.group(2)}: {tr_literal(m.group(3))},\n")
    else:
        out.write(line + "\n")
    CACHE_PATH.write_text(json.dumps(CACHE, ensure_ascii=False), encoding="utf-8")
out.write("} as const\n\n")
out.write("export const FIELD_DESCRIPTIONS_ES = {\n")
for line in desc_block.splitlines():
    m = re.match(r"^(\s*)([\w.]+):\s*('[^']*')\s*,?$", line)
    if m:
        out.write(f"{m.group(1)}{m.group(2)}: {tr_literal(m.group(3))},\n")
    else:
        out.write(line + "\n")
    CACHE_PATH.write_text(json.dumps(CACHE, ensure_ascii=False), encoding="utf-8")
out.write("} as const\n")
out.close()
print(f"WROTE {OUT}", flush=True)
