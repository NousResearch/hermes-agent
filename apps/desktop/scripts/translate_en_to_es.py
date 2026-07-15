#!/usr/bin/env python3
"""Translate en.ts -> es.ts LINE BY LINE, writing incrementally + disk cache.

Robust: writes each translated line immediately (flush), caches translations in
/tmp/hermes_es_cache.json, and skips already-translated lines on rerun.
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

SRC = Path("/home/the_wolf/.hermes/hermes-agent/apps/desktop/src/i18n/en.ts")
OUT = Path("/home/the_wolf/.hermes/hermes-agent/apps/desktop/src/i18n/es.ts")
CACHE_PATH = Path("/tmp/hermes_es_cache.json")

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


def tr_text(text: str) -> str:
    parts = re.split(r"(\$\{(?:[^{}]|\{[^{}]*\})*\})", text)
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            out.append(part)
        elif part.strip():
            out.append(gtx(part))
        else:
            out.append(part)
    return "".join(out)


def tr_literal(s: str) -> str:
    if s.startswith("`"):
        return "`" + tr_text(s[1:-1]) + "`"
    return "'" + tr_text(s[1:-1]) + "'"


def tr_inline_object(obj: str) -> str:
    return re.sub(r"'[^'\\]*'|`[^`\\]*`", lambda m: tr_literal(m.group(0)), obj)


def tr_any_strings_keep_keys(line: str) -> str:
    def repl(m):
        s = m.group(0)
        after = line[m.end():].lstrip()
        if after.startswith(":"):
            return s
        return tr_literal(s)
    return re.sub(r"'[^'\\]*'|`[^`\\]*`", repl, line)


PROP_RE = re.compile(r"^(\s*)([\w.'\- ]+):(\s*)(.*)$")
ARROW_TPL_RE = re.compile(r"^(?:[\w$]+|\([^)]*\))\s*=>\s*(['`].*)$")

SAVE_EVERY = 25


def main():
    lines = SRC.read_text(encoding="utf-8").splitlines()
    out = open(OUT, "w", encoding="utf-8")
    n = 0
    for raw in lines:
        stripped = raw.strip()
        m = PROP_RE.match(raw)
        is_prop = bool(m) and not stripped.startswith("//") and not stripped.startswith("import") \
            and "Translations" not in raw and "export const" not in raw
        if is_prop:
            indent, key, _, value = m.group(1), m.group(2), m.group(3), m.group(4).rstrip()
            if value.rstrip(",") in ("FIELD_LABELS", "FIELD_DESCRIPTIONS"):
                out.write(raw + "\n"); continue
            if value == "" or value in ("{", "}", "},", "{},", "[]"):
                out.write(raw + "\n"); continue
            if re.match(r"^[\d]+\s*,?$", value):
                out.write(raw + "\n"); continue
            am = ARROW_TPL_RE.match(value)
            if am:
                lit = am.group(1).rstrip(",")
                new = value[: len(value) - (1 if value.endswith(",") else 0)].replace(am.group(1).rstrip(","), tr_literal(lit))
                out.write(f"{indent}{key}: {new}" + ("," if value.endswith(",") else "") + "\n"); n += 1
            elif (value.startswith("'") and value.endswith("'")) or (value.startswith("`") and value.endswith("`")):
                comma = "," if value.endswith(",") else ""
                out.write(f"{indent}{key}: {tr_literal(value.rstrip(','))}{comma}\n"); n += 1
            elif value.startswith("{"):
                comma = "," if value.endswith(",") else ""
                out.write(f"{indent}{key}: {tr_inline_object(value.rstrip(','))}{comma}\n"); n += 1
            else:
                out.write(tr_any_strings_keep_keys(raw) + "\n"); n += 1
        else:
            out.write(raw + "\n")
        n_local = n
        if n_local % SAVE_EVERY == 0:
            out.flush()
            CACHE_PATH.write_text(json.dumps(CACHE, ensure_ascii=False), encoding="utf-8")
    out.write("\n")
    out.close()
    CACHE_PATH.write_text(json.dumps(CACHE, ensure_ascii=False), encoding="utf-8")
    print(f"DONE: {n} strings translated -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
