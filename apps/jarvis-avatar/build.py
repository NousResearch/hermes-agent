#!/usr/bin/env python3
"""Gera dist/jarvis.html: a página inteira num arquivo só (CSS, JS, fontes e imagens embutidos).

Serve pra publicar como artifact ou abrir com duplo clique (file://), onde os módulos ES e as
imagens soltas não carregam. Só biblioteca padrão; os módulos são concatenados na ordem de
dependência abaixo, sem bundler, então cada nome de topo precisa ser único entre eles.

    python3 build.py            # escreve dist/jarvis.html
"""
from __future__ import annotations

import base64
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODULES = ["config", "util", "shaders", "audio", "pose", "shapes", "main"]
MIME = {".webp": "image/webp", ".png": "image/png", ".woff2": "font/woff2"}

IMPORT_RE = re.compile(r"^import\s.+?\sfrom\s+'\./[\w-]+\.js';\s*$", re.M)
TOP_DECL_RE = re.compile(r"^(?:export\s+)?(?:async\s+)?(?:const|let|function)\s+([\w$]+)", re.M)


def data_uri(path: Path) -> str:
    return f"data:{MIME[path.suffix]};base64,{base64.b64encode(path.read_bytes()).decode()}"


def bundle_js() -> str:
    parts, owner = [], {}
    for name in MODULES:
        src = (ROOT / "src" / f"{name}.js").read_text(encoding="utf-8")
        for decl in set(TOP_DECL_RE.findall(src)):
            if decl in owner:
                sys.exit(f"build: '{decl}' declarado em {owner[decl]}.js e {name}.js")
            owner[decl] = name
        src = IMPORT_RE.sub("", src)
        if re.search(r"^\s*(import|export)\b(?!\s+(const|let|function|async))", src, re.M):
            sys.exit(f"build: {name}.js tem import/export que este build não entende (use uma linha por import)")
        src = re.sub(r"^export\s+", "", src, flags=re.M)
        parts.append(f"/* ---- {name}.js ---- */\n{src.strip()}\n")
    js = "\n".join(parts)
    js = re.sub(r"'(assets/img/[\w.-]+)'", lambda m: f"'{data_uri(ROOT / m.group(1))}'", js)
    if "</script" in js:
        sys.exit("build: o JS contém '</script' e quebraria o HTML")
    return "(() => {\n'use strict';\n" + js + "})();\n"


def inline_css() -> str:
    css = (ROOT / "src" / "styles.css").read_text(encoding="utf-8")
    return re.sub(r"url\(\.\./(assets/fonts/[\w.-]+)\)", lambda m: f"url({data_uri(ROOT / m.group(1))})", css)


def main() -> None:
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    html = re.sub(r"[ \t]*<!--[^>]*-->\n", "", html)
    html = re.sub(r'[ \t]*<link rel="preload"[^>]*>\n', "", html)
    html = html.replace('<link rel="stylesheet" href="src/styles.css">', f"<style>\n{inline_css()}</style>")
    html = html.replace('<script type="module" src="src/main.js"></script>', f"<script>\n{bundle_js()}</script>")
    if 'src="src/' in html or 'href="src/' in html:
        sys.exit("build: sobrou referência a src/ no HTML")
    out = ROOT / "dist" / "jarvis.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"{out.relative_to(ROOT)}: {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
