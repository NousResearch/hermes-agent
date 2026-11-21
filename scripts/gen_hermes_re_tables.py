"""Regenerate or verify hermes_re/_tables.py: where the running Python and the installed RE2 disagree.

    python scripts/gen_hermes_re_tables.py            regenerate the entry for this interpreter's Unicode version
                                                      and the installed RE2's Unicode generation
    python scripts/gen_hermes_re_tables.py --check    exit 1 when the shipped entry differs from a fresh computation
    add --predicates to compute the Python side from str.isalnum / isdecimal / isspace (no stdlib re)

Needs google-re2. The reference is stdlib `re` (allowed in scripts/, never inside the facade).
One google-re2 version can carry different Unicode data: wheels (and a libre2 built without ICU)
use RE2's bundled tables, a libre2 built with RE2_USE_ICU (Alpine, so the musl CI lane) the linked
ICU's. Run this once per such build; each run adds or checks the entry for ITS generation.
Importing this file does nothing: ``compute()`` does the work (about two seconds).
"""
import importlib.metadata
import json
import re as std
import sys
import time
import unicodedata
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "scripts" / "hermes_re_tables.json"
TABLES_PATH = ROOT / "hermes_re" / "_tables.py"
DELTAS = ("W_ADD", "W_SUB", "D_ADD", "D_SUB")
PREDICATES = {r"\w": lambda ch: ch.isalnum() or ch == "_", r"\d": str.isdecimal, r"\s": str.isspace}


def _all_text() -> str:
    """Every code point except the surrogates, in order."""
    return "".join(chr(code) for code in range(0x110000) if not 0xD800 <= code <= 0xDFFF)


def re2_set(source: str, data: bytes) -> set:
    """The code points RE2 matches with the one-character class ``source``."""
    from re2 import _re2  # here, not at module top: importing this script must not need the engine

    options = _re2.RE2.Options()
    options.log_errors = False
    program = _re2.RE2(source.encode(), options)
    if not program.ok():
        raise ValueError(program.error())
    out, pos, size, unanchored = set(), 0, len(data), _re2.RE2.Anchor.UNANCHORED
    while pos <= size:
        start, end = program.Match(unanchored, data, pos, size)[0]
        if start < 0:
            break
        out.add(ord(data[start:end].decode()))
        pos = end
    return out


def std_set(source: str, text: str, predicates: bool) -> set:
    """The code points Python matches with ``source``: by stdlib `re`, or by the str predicates."""
    if predicates:
        return {ord(ch) for ch in text if PREDICATES[source](ch)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {ord(ch) for ch in std.findall(source, text)}


def ranges(codes) -> list:
    out = []
    for code in sorted(codes):
        if out and out[-1][1] == code - 1:
            out[-1][1] = code
        else:
            out.append([code, code])
    return out


def generation() -> str:
    """The Unicode generation of the installed RE2's ``\\p{..}`` data (the facade's own probe)."""
    sys.path.insert(0, str(ROOT))
    from hermes_re._engine import unicode_generation

    return unicode_generation()


def compute(predicates: bool = False) -> dict:
    """The table entry for the running interpreter and the installed RE2 build."""
    text = _all_text()
    data = text.encode()
    w, d = std_set(r"\w", text, predicates), std_set(r"\d", text, predicates)
    rw, rd = re2_set(r"[\p{L}\p{N}_]", data), re2_set(r"\p{Nd}", data)
    return {"python": "{}.{}".format(*sys.version_info[:2]), "generation": generation(),
            "re2": importlib.metadata.version("google-re2"),
            "W_ADD": ranges(w - rw), "W_SUB": ranges(rw - w), "D_ADD": ranges(d - rd), "D_SUB": ranges(rd - d),
            "SPACE": ranges(std_set(r"\s", text, predicates)),
            "counts": {"W_ADD": len(w - rw), "W_SUB": len(rw - w), "D_ADD": len(d - rd), "D_SUB": len(rd - d)}}


def same(shipped, fresh: dict) -> bool:
    """Whether the shipped table for this Unicode version (``hermes_re._tables.TABLES[version]`` or
    ``None``) has ``fresh``'s SPACE and, for ``fresh``'s RE2 generation, its deltas."""
    if shipped is None or [list(pair) for pair in shipped["SPACE"]] != fresh["SPACE"]:
        return False
    entry = shipped["RE2"].get(fresh["generation"])
    return entry is not None and all([list(pair) for pair in entry[key]] == fresh[key] for key in DELTAS)


def merge(tables: dict, version: str, fresh: dict) -> None:
    """Record ``fresh`` in the JSON form of the tables, keeping the other generations' entries."""
    table = tables.setdefault(version, {"RE2": {}})
    table.update(python=fresh["python"], SPACE=fresh["SPACE"])
    table["RE2"][fresh["generation"]] = {key: fresh[key] for key in ("re2", "counts", *DELTAS)}


def _rows(pairs, indent: str) -> list:
    body, row = [], indent
    for lo, hi in pairs:
        item = f"(0x{lo:X}, 0x{hi:X}), "
        if len(row) + len(item) > 108:
            body.append(row.rstrip())
            row = indent
        row += item
    if row.strip():
        body.append(row.rstrip())
    return body


def _field(key: str, pairs, indent: str) -> str:
    body = _rows(pairs, indent + "    ")
    return f"{indent}{key!r}: (" + ("\n" + "\n".join(body) + "\n" + indent if body else "") + "),"


def render(tables: dict) -> str:
    lines = ['r"""GENERATED by scripts/gen_hermes_re_tables.py; do not edit. Static Unicode delta tables.',
             "", "Per Unicode version of the running Python (``unicodedata.unidata_version``): Python's `\\s`",
             r"(SPACE) and, per Unicode generation of RE2's ``\p{..}`` data (``_engine.unicode_generation()``:",
             "RE2's bundled tables, or ICU's in a libre2 built with RE2_USE_ICU), the code points where",
             r"Python's `\w` / `\d` and RE2's ``[\p{L}\p{N}_]`` / ``\p{Nd}`` disagree, as sorted inclusive ranges.",
             "ADD = Python yes, RE2 no. SUB = RE2 yes, Python no. The translation unites the generations'",
             "deltas (``_translate_class``), so it is exact under each listed generation; under one not",
             "listed `\\w` / `\\d` follow that RE2's own data where it differs (``--check`` fails there).",
             '"""', "", "TABLES = {"]
    for version in sorted(tables):
        table = tables[version]
        lines.append(f"    {version!r}: {{  # Python {table['python']}")
        lines.append(_field("SPACE", table["SPACE"], "        "))
        lines.append("        'RE2': {")
        for name in sorted(table["RE2"]):
            entry = table["RE2"][name]
            lines.append(f"            {name!r}: {{  # google-re2 {entry['re2']}, code points: {entry['counts']}")
            lines += [_field(key, entry[key], "                ") for key in DELTAS]
            lines.append("            },")
        lines.append("        },")
        lines.append("    },")
    lines.append("}")
    return "\n".join(lines) + "\n"


def main(argv) -> int:
    version, predicates = unicodedata.unidata_version, "--predicates" in argv
    started = time.perf_counter()
    fresh = compute(predicates)
    print(f"   computed in {time.perf_counter() - started:.2f} s from", "str predicates" if predicates else "stdlib re")
    print(f"unicodedata.unidata_version={version} python={fresh['python']} google-re2={fresh['re2']} "
          f"RE2 Unicode generation={fresh['generation']}")
    print("   code points:", fresh["counts"], "| ranges:", {key: len(fresh[key]) for key in (*DELTAS, "SPACE")})
    if "--check" in argv:
        from hermes_re import _tables

        current = same(_tables.TABLES.get(version), fresh)
        print("   table for this version and generation:", "UP TO DATE" if current else "STALE OR MISSING -> run this script")
        return 0 if current else 1
    tables = json.loads(JSON_PATH.read_text(encoding="utf-8-sig")) if JSON_PATH.exists() else {}
    merge(tables, version, fresh)
    JSON_PATH.write_text(json.dumps(tables, indent=0), encoding="utf-8")
    TABLES_PATH.write_text(render(tables), encoding="utf-8")
    print("   wrote hermes_re/_tables.py:", {name: sorted(table["RE2"]) for name, table in sorted(tables.items())})
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
