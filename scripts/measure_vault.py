#!/usr/bin/env python3
"""วัดสุขภาพลิงก์ของ Obsidian vault

สคริปต์นี้อ่าน vault จาก argument แรกเท่านั้น ไม่เดา path และไม่เขียนกลับเข้า
vault เอง ยกเว้นผู้ใช้ระบุ `--json PATH` เป็นไฟล์ผลลัพธ์ที่ต้องการ
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlparse


EXCLUDED_DIRS = {".git", ".obsidian", ".trash"}
NAV_FILES = {"readme.md", "_index.md", "index.md", "moc.md"}
WIKI_LINK_RE = re.compile(r"!?\[\[([^\]]+)\]\]")
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="measure_vault.py",
        usage="measure_vault.py vault_root [--json PATH]",
        description="วัดสุขภาพลิงก์ของ Obsidian vault",
    )
    parser.add_argument("vault_root")
    parser.add_argument("--json", dest="json_path")
    return parser.parse_args(argv)


def iter_note_paths(vault_root: Path) -> list[Path]:
    note_paths: list[Path] = []
    for root, dirnames, filenames in os.walk(vault_root):
        dirnames[:] = [name for name in dirnames if name not in EXCLUDED_DIRS]
        root_path = Path(root)
        for filename in filenames:
            if filename.lower().endswith(".md"):
                note_paths.append(root_path / filename)

    return sorted(note_paths, key=lambda path: path.relative_to(vault_root).as_posix().lower())


def normalize_target(raw_target: str) -> str:
    target = unquote(raw_target.strip().strip("<>"))
    target = target.split("|", 1)[0]
    target = target.split("#", 1)[0]
    target = target.split("^", 1)[0]
    target = target.strip()
    if not target:
        return ""

    name = Path(target.replace("\\", "/")).name
    if name.lower().endswith(".md"):
        name = name[:-3]
    return name.strip()


def is_external_target(raw_target: str) -> bool:
    parsed = urlparse(raw_target.strip().strip("<>"))
    return bool(parsed.scheme and parsed.scheme != "file")


def extract_links(markdown: str) -> list[str]:
    links: list[str] = []

    for match in WIKI_LINK_RE.finditer(markdown):
        target = normalize_target(match.group(1))
        if target:
            links.append(target)

    for match in MARKDOWN_LINK_RE.finditer(markdown):
        raw_target = match.group(1)
        if is_external_target(raw_target):
            continue
        target = normalize_target(raw_target)
        if target:
            links.append(target)

    return links


def relpath(path: Path, vault_root: Path) -> str:
    return path.relative_to(vault_root).as_posix()


def folder_name(path: Path, vault_root: Path) -> str:
    relative = path.relative_to(vault_root).as_posix()
    return "." if relative == "." else relative


def measure_vault(vault_root: Path) -> dict[str, object]:
    notes = iter_note_paths(vault_root)
    note_names = {path.stem for path in notes}
    incoming = Counter()
    outgoing_by_note: dict[str, list[str]] = {}
    broken_samples: list[dict[str, str]] = []
    total_links = 0
    broken_links = 0

    for note_path in notes:
        note_relpath = relpath(note_path, vault_root)
        text = note_path.read_text(encoding="utf-8", errors="replace")
        links = extract_links(text)
        outgoing_by_note[note_relpath] = links
        total_links += len(links)

        for target in links:
            if target in note_names:
                incoming[target] += 1
            else:
                broken_links += 1
                if len(broken_samples) < 15:
                    broken_samples.append({"in": note_relpath, "target": target})

    no_outgoing_list = sorted(
        [path for path, links in outgoing_by_note.items() if not links],
        key=str.lower,
    )
    no_incoming_list = sorted(
        [relpath(path, vault_root) for path in notes if incoming[path.stem] == 0],
        key=str.lower,
    )
    orphan_list = sorted(set(no_outgoing_list) & set(no_incoming_list), key=str.lower)

    folders = sorted(
        {path.parent for path in notes},
        key=lambda path: path.relative_to(vault_root).as_posix().lower(),
    )
    folders_without_nav_list: list[str] = []
    for folder in folders:
        nav_exists = any(
            child.is_file() and child.name.lower() in NAV_FILES
            for child in folder.iterdir()
        )
        if not nav_exists:
            folders_without_nav_list.append(folder_name(folder, vault_root))

    return {
        "total_notes": len(notes),
        "total_links": total_links,
        "broken_links": broken_links,
        "broken_samples": broken_samples,
        "no_outgoing": len(no_outgoing_list),
        "no_incoming": len(no_incoming_list),
        "true_orphans": len(orphan_list),
        "folders_with_notes": len(folders),
        "folders_without_nav": len(folders_without_nav_list),
        "orphan_list": orphan_list,
    }


def print_summary(metrics: dict[str, object]) -> None:
    print("สรุปสุขภาพลิงก์ของ Obsidian vault")
    print(f"- จำนวนโน้ตทั้งหมด: {metrics['total_notes']}")
    print(f"- จำนวนลิงก์ทั้งหมด: {metrics['total_links']}")
    print(f"- ลิงก์เสีย: {metrics['broken_links']}")
    print(f"- โน้ตที่ไม่มีลิงก์ออก: {metrics['no_outgoing']}")
    print(f"- โน้ตที่ไม่มีลิงก์เข้า: {metrics['no_incoming']}")
    print(f"- โน้ตกำพร้าจริง: {metrics['true_orphans']}")
    print(f"- โฟลเดอร์ที่มีโน้ต: {metrics['folders_with_notes']}")
    print(f"- โฟลเดอร์ที่ไม่มีไฟล์นำทาง: {metrics['folders_without_nav']}")

    samples = metrics["broken_samples"]
    if samples:
        print("\nตัวอย่างลิงก์เสีย:")
        for sample in samples:
            print(f"- {sample['in']} -> {sample['target']}")


def write_json(metrics: dict[str, object], json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: measure_vault.py vault_root [--json PATH]")
        return 2

    args = parse_args(argv)
    vault_root = Path(args.vault_root).expanduser().resolve()
    if not vault_root.is_dir():
        print(f"error: vault_root ไม่ใช่โฟลเดอร์: {vault_root}", file=sys.stderr)
        return 2

    metrics = measure_vault(vault_root)
    print_summary(metrics)

    if args.json_path:
        json_path = Path(args.json_path).expanduser().resolve()
        write_json(metrics, json_path)
        print(f"\nเขียน JSON แล้ว: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
