#!/usr/bin/env python3
"""ตรวจสัญญา Hermes Worktree Lifecycle โดยไม่แก้ไฟล์."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REQUIRED_HEADINGS = [
    "## 1. ขอบเขตและแหล่งจริง",
    "## 3. โครงสร้างตำแหน่ง",
    "## 5. สมุดทะเบียนกลาง",
    "## 6. วงจรสถานะ",
    "## 7. สิทธิ์เขียนและการส่งต่อเครื่อง",
    "## 8. กฎเมื่อขาดการเชื่อมต่อ",
    "## 9. Runtime namespace",
    "## 10. พื้นที่และงบ",
    "## 11. Cleanup gate 6/6",
    "## 12. คำสั่งกลาง",
    "## 13. ความปลอดภัย",
    "## 14. การเชื่อม Shortcut",
    "## 15. PDCA",
    "## 16. เหตุการณ์ที่ต้องถูกปฏิเสธ",
    "## 17. เกณฑ์ประกาศใช้",
    "## 18. Decision tokens",
]

REQUIRED_PHRASES = [
    "หนึ่งโครงการมีพื้นที่หลักหนึ่งแห่ง",
    "หนึ่งงานเขียนมี task worktree ของตัวเอง",
    "หนึ่ง task มีเครื่องถือสิทธิ์เขียนได้ครั้งละหนึ่งเครื่อง",
    "~/Documents/Worktrees/<project-id>/<staff-id>/<task-id>-<slug>",
    "/home/linux-nat/.worktree/<project-id>/<staff-id>/<task-id>-<slug>",
    "task/<staff-id>/<task-id>-<slug>",
    "CREATED → ACTIVE",
    "CLEANUP_READY → QUARANTINED → ARCHIVED",
    "QUARANTINED 72 ชั่วโมง",
    "ตรวจเบาทุก 24 ชั่วโมง",
    "เสนอ cleanup ทุก 168 ชั่วโมง",
    "hermes worktree open",
    "hermes worktree cleanup --dry-run",
    "WTL_BLOCKED",
    "Shortcut visibility เห็นกฎ WTL 30/30",
    "โซน A — AI ทำต่อเองจนจบเฟส",
    "โซน B — รวบขออนุมัติระดับ Phase ครั้งเดียว",
]

SHORTCUTS = [
    "Use Act-As",
    "Use Comply",
    "Use Summary",
    "Use Scan Feature",
    "Use AI Relay",
    "Use Viber Structure",
    "Use Viber Audit",
    "Use Impeccable",
    "Use Blog Auto",
    "Use WOW Resource",
    "Use Flow Guardian",
    "Use New Chat",
    "Use Close Chat",
    "Use Save Git",
    "Use Merge to Production",
    "Use Continue",
    "Use Move Folder",
    "Review Chat",
    "Use AI Pair",
    "Use Business Plan",
    "Use SaaS Opus Master Prompt",
    "Use BusinessPlan",
    "Use OverviewProgress",
    "Use FeatureSpec",
    "Use DesignSystem",
    "Use Create Design System",
    "Use Hermes Structure",
    "Use Create Content",
    "Use QA QC",
    "Use SonarQube",
]

CONFLICT_PHRASES = [
    "หลาย Cursor/AI ใช้โฟลเดอร์เดียวกันได้",
    "ห้ามเสนอ ห้ามสร้าง และห้ามสั่งสร้าง Git worktree ใหม่",
    "งานใหม่สร้างได้เฉพาะ branch ภายใน registered folder เดิม",
    "allow_multiple_writers: true",
    "cleanup_by_age: true",
    "allow_direct_worktree_delete: true",
]


def validate_text(text):
    missing_headings = [item for item in REQUIRED_HEADINGS if item not in text]
    missing_phrases = [item for item in REQUIRED_PHRASES if item not in text]
    missing_shortcuts = [item for item in SHORTCUTS if item not in text]
    conflicts = [item for item in CONFLICT_PHRASES if item in text]
    ok = not (missing_headings or missing_phrases or missing_shortcuts or conflicts)
    return {
        "ok": ok,
        "required_headings": len(REQUIRED_HEADINGS),
        "required_phrases": len(REQUIRED_PHRASES),
        "shortcut_count": len(SHORTCUTS),
        "missing_headings": missing_headings,
        "missing_phrases": missing_phrases,
        "missing_shortcuts": missing_shortcuts,
        "conflicts": conflicts,
    }


def main():
    parser = argparse.ArgumentParser(description="ตรวจ Worktree Lifecycle Contract")
    parser.add_argument("contract", help="ไฟล์สัญญาที่ต้องตรวจ")
    parser.add_argument("--json", action="store_true", help="คืนผล JSON")
    args = parser.parse_args()

    path = Path(args.contract).expanduser().resolve()
    if not path.is_file():
        result = {"ok": False, "error": "contract_not_found", "path": str(path)}
    else:
        result = validate_text(path.read_text(encoding="utf-8"))
        result["path"] = str(path)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("WTL_CONTRACT_OK" if result.get("ok") else "WTL_CONTRACT_BLOCKED")
        for key in ("missing_headings", "missing_phrases", "missing_shortcuts", "conflicts"):
            for item in result.get(key, []):
                print("- {}: {}".format(key, item))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
