#!/usr/bin/env python3
"""把新的 line.document.* i18n key 加到所有 locale 檔案。

用一次性腳本（上游常見做法）：en.yaml 是基準，每個語言都要有**完全相同**的
key set 與 placeholder（tests/agent/test_i18n.py 會驗證）。

用法: python3 scripts/add_line_document_keys.py [--check]
"""
from __future__ import annotations

import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCALES = os.path.join(REPO, "locales")

# lang -> (attachment_alt, download_hint)
T = {
    "af": ("Aanhangsel: {display_name}", "Tik die knoppie hieronder om {display_name} te laai"),
    "ar": ("مرفق: {display_name}", "اضغط على الزر أدناه لتنزيل {display_name}"),
    "de": ("Anhang: {display_name}", "Tippe auf die Schaltfläche unten, um {display_name} herunterzuladen"),
    "es": ("Adjunto: {display_name}", "Pulsa el botón de abajo para descargar {display_name}"),
    "fr": ("Pièce jointe : {display_name}", "Appuyez sur le bouton ci-dessous pour télécharger {display_name}"),
    "ga": ("Ceangalt: {display_name}", "Cliceáil an cnaipe thíos chun {display_name} a íoslódáil"),
    "hu": ("Melléklet: {display_name}", "Koppints a gomb alább a {display_name} letöltéséhez"),
    "it": ("Allegato: {display_name}", "Tocca il pulsante qui sotto per scaricare {display_name}"),
    "ja": ("添付ファイル: {display_name}", "下のボタンを押して {display_name} をダウンロード"),
    "ko": ("첨부 파일: {display_name}", "아래 버튼을 눌러 {display_name}을(를) 다운로드하세요"),
    "pt": ("Anexo: {display_name}", "Toque no botão abaixo para baixar {display_name}"),
    "ru": ("Вложение: {display_name}", "Нажмите кнопку ниже, чтобы скачать {display_name}"),
    "tr": ("Ek: {display_name}", "İndirmek için aşağıdaki düğmeye dokunun: {display_name}"),
    "uk": ("Вкладення: {display_name}", "Натисніть кнопку нижче, щоб завантажити {display_name}"),
    "zh": ("附件：{display_name}", "点击下方按钮下载 {display_name}"),
    "zh-hant": ("附件：{display_name}", "點擊下方按鈕下載 {display_name}"),
}

ANCHOR = "    standalone:\n"
BLOCK = "    document:\n      attachment_alt: \"{alt}\"\n      download_hint: \"{hint}\"\n"


def escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def process(path: str, alt: str, hint: str, check_only: bool) -> bool:
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if "    document:\n" in src and "attachment_alt" in src:
        print(f"  = {os.path.basename(path):<16} 已有 document key，略過")
        return True
    idx = src.find(ANCHOR)
    if idx < 0:
        print(f"  ✗ {os.path.basename(path):<16} 找不到 standalone 錨點")
        return False
    # 找到 standalone 區塊的結尾（下一個同層級 key）
    tail = src[idx + len(ANCHOR):]
    m = re.search(r"\n  [a-z_]+:", tail)
    end = idx + len(ANCHOR) + (m.start() if m else len(tail))
    block = BLOCK.format(alt=escape(alt), hint=escape(hint))
    out = src[:end] + "\n" + block + src[end:]
    if check_only:
        print(f"  ? {os.path.basename(path):<16} 需要加入")
        return False
    with open(path, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"  ✓ {os.path.basename(path):<16} 已加入")
    return True


def main() -> int:
    check_only = "--check" in sys.argv
    ok = True
    for lang, (alt, hint) in sorted(T.items()):
        path = os.path.join(LOCALES, f"{lang}.yaml")
        if not os.path.exists(path):
            print(f"  - {lang}.yaml 不存在，略過")
            continue
        ok = process(path, alt, hint, check_only) and ok
    print()
    print("（en.yaml 已手動加入，作為基準）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
