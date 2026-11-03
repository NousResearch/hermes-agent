#!/usr/bin/env python3
"""Gate de marca do B-3PO.

Verifica, sem dependências externas:
  1. brand.json e palette.json existem, parseiam e têm as chaves obrigatórias;
  2. todos os assets declarados existem;
  3. os 20 mascotes do manifest existem, sao PNG 512x512 e batem com o catalogo;
  4. contraste WCAG AA dos tokens semanticos (texto >= 4.5:1, accent >= 3:1);
  5. cor proibida (#00BFFF) nao entrou como primitiva;
  6. o mapa de rename não habilita nível 3 (proibido pela Deliberação 02).

Uso: python scripts/brand_check.py
Sai com código 1 em qualquer falha.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRAND = ROOT / "brand"
FAILS: list[str] = []
CHECKS = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    status = "PASSA" if ok else "FALHA"
    print(f"[{status}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def png_size(path: Path) -> tuple[int, int] | None:
    with path.open("rb") as fh:
        head = fh.read(24)
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", head[16:24])


def luminance(hex_color: str) -> float:
    hex_color = hex_color.lstrip("#")
    channels = [int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    # 1. estrutura
    brand_path, palette_path = BRAND / "brand.json", BRAND / "palette.json"
    check("brand.json existe", brand_path.exists(), str(brand_path))
    check("palette.json existe", palette_path.exists(), str(palette_path))
    if FAILS:
        return 1

    brand = json.loads(brand_path.read_text(encoding="utf-8-sig"))
    palette = json.loads(palette_path.read_text(encoding="utf-8-sig"))

    required = ["name", "productName", "slug", "packageName", "cliCommand", "owner", "assets", "palette"]
    missing = [k for k in required if k not in brand]
    check("brand.json tem chaves obrigatórias", not missing, f"faltando: {missing}" if missing else "")

    # 2. assets
    assets = brand.get("assets", {})
    missing_assets = [v for k, v in assets.items() if k != "avatarsDir" and not (BRAND / v).exists()]
    check("assets declarados existem", not missing_assets, f"faltando: {missing_assets}" if missing_assets else "")

    # 3. mascotes
    avatars_dir = BRAND / assets.get("avatarsDir", "avatars")
    manifest_path = BRAND / assets.get("avatarManifest", "avatars/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig")) if manifest_path.exists() else []
    pngs = sorted(avatars_dir.glob("*.png"))
    check("manifest tem 20 mascotes", len(manifest) == 20, f"encontrados {len(manifest)}")
    check("20 PNGs em avatars/", len(pngs) == 20, f"encontrados {len(pngs)}")

    ids = {a["id"] for a in manifest}
    files = {p.stem for p in pngs}
    check("ids do manifest == arquivos", ids == files, f"só no manifest: {sorted(ids - files)}; só em disco: {sorted(files - ids)}")

    bad_size = [p.name for p in pngs if png_size(p) != (512, 512)]
    check("mascotes sao 512x512", not bad_size, f"fora do padrão: {bad_size}" if bad_size else "")

    # 4. contraste
    prim = palette.get("primitives", {})
    sem = {k: prim.get(v, v) for k, v in palette.get("semantic", {}).items() if v}
    bg = sem.get("bg")
    if not bg:
        check("token bg definido", False)
    else:
        for token, minimum in (("text", 4.5), ("textMuted", 4.5), ("textDim", 3.0), ("accent", 3.0)):
            fg = sem.get(token)
            if not fg:
                check(f"contraste {token}/bg", False, "token ausente")
                continue
            ratio = contrast(fg, bg)
            check(f"contraste {token}/bg >= {minimum}", ratio >= minimum, f"{ratio:.2f}:1")

    # 5. cor proibida
    banned = {"#00BFFF", "#000000", "#FFFFFF"}
    used = {v.upper() for v in prim.values()}
    check("sem cor provisória como primitiva", not (used & banned), f"proibidas: {sorted(used & banned)}")

    # 6. rename map
    rm_path = BRAND / "rename-map.json"
    if rm_path.exists():
        rm = json.loads(rm_path.read_text(encoding="utf-8-sig"))
        check("rename-map não habilita nível 3", rm.get("level") != 3 and rm.get("forbidden", {}).get("level3_renames") is True,
              f"level={rm.get('level')}")
    else:
        check("rename-map.json existe", False)

    print(f"\nbrand_check: {CHECKS} checagens, {len(FAILS)} falha(s)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())



