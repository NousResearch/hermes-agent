---
title: Use Move Folder
version: v1.1
aliases:
  - use-move-folder
  - Move Folder
  - move-folder
  - movefolder
  - ใช้ Move Folder
  - ย้ายโฟลเดอร์
  - จัดเรียง Folder
  - จัดเรียงโฟลเดอร์
tags:
  - prompt-shortcuts
  - server-cleanup
  - folder-move
status: active
updated: 2026-06-25
reviewed_by:
  - claude-opus (draft)
  - gpt-5.5 codex (cross-check)
source_of_truth: /home/linux-nat/.codex/use-move-folder/project-registry
---

# Use Move Folder

Use this shortcut when the owner asks to continue the VPS folder cleanup, folder move, retention review, workspace cleanup, backup/archive cleanup, or disk-space cleanup workflow that already exists on the `linux-nat` VPS.

Do not invent the workflow from memory. The operational source of truth is the VPS project registry:

```text
/home/linux-nat/.codex/use-move-folder/project-registry
```

## Required Loading Order

When `Use Move Folder` or a close alias is invoked, read these files first from the VPS, in this exact order:

1. `/home/linux-nat/.codex/use-move-folder/project-registry/LATEST_NEW_CHAT_RESUME.md`
2. `/home/linux-nat/.codex/use-move-folder/project-registry/NO_TOUCH_POLICY.md`
3. `/home/linux-nat/.codex/use-move-folder/project-registry/REPORT_STYLE_REQUIREMENTS.md`
4. `/home/linux-nat/.codex/use-move-folder/project-registry/NEW_CHAT_CONTINUATION_RULES_20260621T141500Z.md`
5. `/home/linux-nat/.codex/use-move-folder/project-registry/ACCELERATED_CLEANUP_OPERATING_MODEL_20260621T140340Z.md`
6. The latest relevant audit, plan, or execute report for the requested scope.

If the VPS is unavailable, say clearly that the shortcut exists but the live registry cannot be read yet. Do not continue from stale local memory, and do not fabricate any project owner, folder size, or scan result from memory. State `UNKNOWN - blocked-by-evidence` instead.

## Core Rules

- Answer in Thai first.
- Use evidence only. Do not guess project ownership from path names.
- Every cleanup or review table must include project owner, size, risk, evidence, and recommendation/status.
- Treat unknown ownership as `UNKNOWN - blocked-by-evidence`.
- Do not scan, move, delete, back up, chmod, chown, restart, or otherwise operate inside protected/no-touch roots unless the owner gives a new exact approval.
- Broad reviews must exclude protected roots and no-touch names.
- Separate findings into `safe-only`, `owner-decision`, and `dangerous/hold`.
- Never delete, move, back up, or restart without explicit owner approval for the exact scope.
- Use larger safe-only batches when possible, but always run a second-pass check and show exact commands before execution.

## Operational Safety Gate (บังคับทุกครั้งก่อน move/ย้ายจริง)

ก่อนเสนอหรือรันงานย้าย/ลบ/สำรองจริง ต้องผ่าน 6 ด่านนี้ตามลำดับ ทุกด่าน:

1. **อ่าน live registry สดก่อน** — โดยเฉพาะ `NO_TOUCH_POLICY.md` ฉบับบน VPS เสมอ ห้ามยึด list ในไฟล์นี้ตัดสิน (ดูหมายเหตุ snapshot ข้างล่าง)
2. **คลี่ path จริง (`realpath`)** — แปลง source และ destination เป็น path จริงด้วย `realpath` ก่อน เพื่อกัน symlink (ลิงก์ลัด) ชี้เข้า no-touch root โดยไม่รู้ตัว ตรวจว่าทั้ง source และ dest หลังคลี่แล้วไม่ตกอยู่ในรายชื่อ protected
3. **ตรวจ no-touch + DEC-039** — เทียบกับ `NO_TOUCH_POLICY.md` สด และกฎ `DEC-039` (1 โปรเจกต์ = 1 โฟลเดอร์) ก่อนเสนอย้าย ต้องค้นหาโฟลเดอร์ canonical ของโปรเจกต์เดิมก่อน ห้ามย้ายไปสร้างโฟลเดอร์ที่หน้าที่ทับซ้อนของเดิมใน `/srv/projects`
4. **ตรวจขอบเขต filesystem (mount boundary)** — ถ้า source กับ dest อยู่คนละ filesystem/คนละ mount การ `mv` จะกลายเป็น copy+delete ที่เสี่ยงข้อมูลขาดกลางทาง ให้ใช้ `rsync --dry-run` ก่อนทุกครั้งสำหรับงานข้าม filesystem
5. **แสดงคำสั่งจริง + แผนถอนกลับ (rollback) + วิธีตรวจหลังย้าย (verify)** — print exact command ที่จะรัน พร้อมบอกว่าถ้าพังจะกู้คืนอย่างไร และจะ verify ผลย้ายอย่างไร (เช่น เทียบจำนวนไฟล์/ขนาดต้นทาง-ปลายทาง)
6. **รอ owner approval แบบ exact scope** — รอเจ้าของงานพิมพ์อนุมัติเฉพาะขอบเขตนั้น ห้ามเหมาว่า "OK ครั้งก่อน" ครอบรอบนี้

## Decision Tokens (ปิดท้ายทุกเฟส 1 token)

ทุกเฟสต้องปิดด้วย 1 token เดียวที่บอกผลตัดสิน เพื่อให้เจ้าของงานอ่านปุ๊บรู้ทันทีว่าต้องทำอะไรต่อ:

- `MOVE_SAFE_BATCH_PROPOSED` — มี batch ที่ปลอดภัย เสนอให้เจ้าของงานตรวจ (proposed) ยังไม่ใช่สั่งรันได้เอง ต้องรอ approval ก่อน
- `MOVE_OWNER_DECISION_REQUIRED` — มีของที่เจ้าของงานต้องตัดสินเอง (เช่น ownership ไม่ชัด หรือเสี่ยงปานกลาง)
- `MOVE_BLOCKED_NO_TOUCH` — งานนี้ชนกับ no-touch root หรือไม่มีหลักฐานพอ หยุด รอคำสั่งใหม่

## Current Known Protected Roots (snapshot เฉยๆ ไม่ใช่ตัวตัดสิน)

หมายเหตุ: รายชื่อข้างล่างนี้เป็น **snapshot** ไว้อ้างอิงเร็วเท่านั้น ไม่ใช่แหล่งตัดสิน ตัวตัดสินคือ `NO_TOUCH_POLICY.md` ฉบับสดบน VPS เสมอ ถ้าอ่านไฟล์สดแล้วต่างจากรายชื่อนี้ ให้ยึดไฟล์สด และห้ามใช้ snapshot นี้ปิดงาน

```text
/home/linux-nat/SynerryTools
/home/linux-nat/OfficeProjects
/home/linux-nat/SaaS Products
/home/linux-nat/Customer Projects
/home/linux-nat/Personal Projects
/home/linux-nat/SynerryTools/newwebengine2026
/srv/projects
```

`/srv/projects` is treated as protected unless the owner gives exact approval for a path or project.

## Required Footer

Every phase response must end with:

```text
สถานะ: ...
ผลตรวจ: ...
ต้องทำต่อ: ...
รออนุมัติ: ...
Decision: <MOVE_SAFE_BATCH_PROPOSED | MOVE_OWNER_DECISION_REQUIRED | MOVE_BLOCKED_NO_TOUCH>
```

## Purpose

This shortcut exists to connect Hermes Prompt Shortcut Registry to the real Move Folder workflow that was created and used on the VPS. It prevents new AI chats from falsely claiming that `Use Move Folder` does not exist just because it is stored under Codex runtime state instead of the Hermes registry.
