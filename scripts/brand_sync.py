#!/usr/bin/env python3
"""Gera os consumidores de marca a partir de brand/brand.json e brand/palette.json.

Uso:  python scripts/brand_sync.py [--check]

--check  não escreve; falha se os arquivos gerados estiverem desatualizados.
Saída:  brand/generated/brand.json, brand/generated/brand.py, brand/generated/brand.ts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRAND_DIR = ROOT / "brand"
OUT_DIR = BRAND_DIR / "generated"

HEADER = "GERADO por scripts/brand_sync.py a partir de brand/brand.json — não editar à mão.\n"


def load() -> tuple[dict, dict]:
    brand = json.loads((BRAND_DIR / "brand.json").read_text(encoding="utf-8"))
    palette = json.loads((BRAND_DIR / "palette.json").read_text(encoding="utf-8"))
    return brand, palette


def resolved_colors(palette: dict) -> dict[str, str]:
    prim = palette["primitives"]
    out: dict[str, str] = {}
    for token, ref in palette["semantic"].items():
        if ref is None:
            continue
        out[token] = prim.get(ref, ref)
    return out


def render_python(brand: dict, colors: dict[str, str]) -> str:
    lines = [
        '"""' + HEADER + '"""',
        "",
        "from __future__ import annotations",
        "",
        "BRAND = {",
        f'    "name": {brand["name"]!r},',
        f'    "product_name": {brand["productName"]!r},',
        f'    "slug": {brand["slug"]!r},',
        f'    "package_name": {brand["packageName"]!r},',
        f'    "cli_command": {brand["cliCommand"]!r},',
        f'    "owner": {brand["owner"]!r},',
        "}",
        "",
        "COLORS = {",
    ]
    for k, v in colors.items():
        lines.append(f"    {k!r}: {v!r},")
    lines += ["}", ""]
    return "\n".join(lines)


def render_ts(brand: dict, colors: dict[str, str]) -> str:
    lines = [
        f"// {HEADER}".rstrip(),
        "",
        "export const BRAND = {",
        f"  name: {json.dumps(brand['name'])},",
        f"  productName: {json.dumps(brand['productName'])},",
        f"  slug: {json.dumps(brand['slug'])},",
        f"  packageName: {json.dumps(brand['packageName'])},",
        f"  cliCommand: {json.dumps(brand['cliCommand'])},",
        f"  owner: {json.dumps(brand['owner'])},",
        "} as const",
        "",
        "export const COLORS = {",
    ]
    for k, v in colors.items():
        lines.append(f"  {k}: {json.dumps(v)},")
    lines += ["} as const", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    brand, palette = load()
    colors = resolved_colors(palette)
    outputs = {
        "brand.json": json.dumps({"brand": brand, "colors": colors}, indent=2, ensure_ascii=False) + "\n",
        "brand.py": render_python(brand, colors),
        "brand.ts": render_ts(brand, colors),
    }

    stale: list[str] = []
    for name, content in outputs.items():
        target = OUT_DIR / name
        if target.exists() and target.read_text(encoding="utf-8") == content:
            continue
        stale.append(name)
        if not args.check:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

    if args.check and stale:
        print(f"brand_sync: desatualizado: {', '.join(stale)}", file=sys.stderr)
        return 1

    print(f"brand_sync: {'verificado' if args.check else 'gerado'} ({len(outputs)} arquivos, {len(colors)} cores)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
