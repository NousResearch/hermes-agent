---
title: Use Move Folder
version: v2.1
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
updated: 2026-06-26
schema: memory-schema-v1.1
reviewed_by:
  - claude-opus (draft v2.0/v2.1)
  - gpt-5.5 codex (cross-check 2 รอบ · MUST-1/2 + S-1..S-5 ผ่าน)
source_of_truth: /home/linux-nat/.codex/use-move-folder/project-registry
---

# Use Move Folder (v2.1)

คู่กับ Memory Schema v1.1 · เช็ก schema version ตอนเริ่ม ไม่ตรง = เตือน
ใช้เมื่อเจ้าของสั่งจัดระเบียบ/ย้ายโฟลเดอร์บน VPS linux-nat · งานนี้คุม `.ssh` + service จริง = พลาด = ล็อกเครื่อง/prod ดับ

> ⚠️ **เงื่อนไขก่อนปล่อยรันกลางคืนแบบไม่มีคนดู:** prompt นี้คือ "กฎให้ AI ทำตาม" ไม่ใช่กลไกบังคับจริง · การ์ดความปลอดภัยทุกตัว (dot-guard / no-touch / ref-scan / free-space / verify / stop-batch) **ต้องฝังในตัวย้าย (executor) เป็นโค้ดก่อน** จึงปล่อยรันกลางคืนไม่มีคนดูได้ · ระหว่างที่ executor ยังไม่เสร็จ = จัดจริงได้เฉพาะตอนเจ้าของตื่นดูอยู่ (เหตุผลเดียวกับ AI Relay FIX-1: prompt ที่ถูก ≠ การบังคับที่ปลอดภัย)

## ลำดับอ่านบังคับ (จาก VPS ก่อนเสมอ)
อ่านจาก `/home/linux-nat/.codex/use-move-folder/project-registry` ตามลำดับ:
1. LATEST_NEW_CHAT_RESUME.md  2. NO_TOUCH_POLICY.md  3. REPORT_STYLE_REQUIREMENTS.md
4. NEW_CHAT_CONTINUATION_RULES_*.md  5. ACCELERATED_CLEANUP_OPERATING_MODEL_*.md  6. รายงานล่าสุดของขอบเขตที่ขอ
VPS เข้าไม่ได้ = "UNKNOWN - blocked-by-evidence" · ห้ามทำต่อจากความจำเก่า ห้ามมั่ว owner/ขนาด/ผลสแกน · ตอบไทยก่อน · ใช้หลักฐานเท่านั้น ห้ามเดา ownership จากชื่อ path

## โครงสร้างเป้าหมาย (ห้ามเว้นวรรคในชื่อ)
SynerryTools/ · OfficeProjects/ · SaaSProducts/ · CustomerProjects/ · PersonalProjects/ · ConfigCenter/ · Archive/ · _Inbox/
กฎ: 1 โปรเจกต์ = 1 โฟลเดอร์ (DEC-039)

## MUST-1 · no-touch แบบ fail-closed (กฎเหมารวม ไม่ใช่ลิสต์)
- **กฎเหมารวม: ทุกชื่อที่ขึ้นต้นด้วย `.` ที่ระดับ `/home/linux-nat/` = no-touch อัตโนมัติเสมอ** (ครอบ dotfile/dotdir ใหม่ในอนาคต + `.env` + บัญชี AI ทุกตัว) — ของที่ลืมจะปลอดภัยเองโดยปริยาย · scope เฉพาะระดับ home root (ไม่บล็อก `.git/.env` ที่อยู่ในโปรเจกต์เวลาย้ายทั้งโปรเจกต์)
- **ลงมือเป็น allowlist:** ย้าย/แตะได้ **เฉพาะ** รายการที่อยู่ในแผนที่เจ้าของอนุมัติแล้ว · นอกแผน = ไม่แตะ
- protected roots (5 root + /srv/projects): **ห้ามลบ/ย้ายตัว root หรือแตะของเดิมในนั้นโดยไม่อนุมัติ · แต่ "วางโปรเจกต์ใหม่เข้าไปตามแผน" = ทำได้**

## ConfigCenter ที่ถูก
ไม่ใช่ขน dotfile บ้าน (ย้ายแล้วล็อกอินพัง) · คือ config ที่ใช้ร่วมจริง + ตัวชี้ symlink (ของจริงอยู่ home เดิม) + secret รวมศูนย์ (private-credentials perm 700 · infisical เป็น service อย่าย้ายดื้อ)

## ด่านความปลอดภัย 9 ด่าน (ก่อนย้าย/ลบ/เปลี่ยนชื่อจริงทุกครั้ง)
1. อ่าน NO_TOUCH_POLICY.md สด
2. realpath คลี่ source/dest กัน symlink ชี้เข้า no-touch
3. เทียบ no-touch (กฎ `.` เหมารวม) + DEC-039 (หา canonical folder เดิม)
4. เช็กข้าม filesystem + เช็ก free space ปลายทาง ≥ ขนาดที่ย้าย + เผื่อ (ไม่พอ = MOVE_BLOCKED)
5. แสดงคำสั่งจริง + rollback + วิธี verify
6. รอ owner approval แบบ exact scope
7. เช็ก service กำลังรันที่ path นั้น (lsof/pm2 list/systemd) · มีรัน = MOVE_BLOCKED ห้ามย้ายขณะรัน
8. ย้ายไม่ลบ ตามชนิด: same-fs = `mv` · cross-fs = `rsync -a` (รักษา perm/owner/timestamp) **แล้วเก็บ source ไว้ ไม่ลบกลางคืน** + verify checksum/จำนวนไฟล์ + rollback/ledger เก็บใน `.hermes/` (no-touch · ไม่ใช่ในโฟลเดอร์ที่กำลังย้าย) · secret คง perm 600/700 + verify perm หลังย้าย
9. **[rename guard]** ก่อน rename/ย้ายทุก path → `grep -r` หาการอ้างอิง path เก่าใน pm2/systemd/nginx-caddy/crontab/.env/โค้ด · ทำ symlink ชั่วคราว old→new ตอนย้าย · อัปเดต reference ครบก่อนถอด symlink · ref ค้าง = MOVE_OWNER_DECISION_REQUIRED ห้ามถอด

## งานกลางคืน 2 เฟส (แก้ "10 วันได้ 5 โฟลเดอร์")
- เฟส 1 (ตื่น): สร้างแผนจัดของทั้งราก → เจ้าของยืนยัน bucket รอบเดียว
- เฟส 2 (กลางคืน auto · เฉพาะเมื่อ executor พร้อม): รันชุดปลอดภัย (ownership ชัด + ย้อนกลับได้ + ไม่ใช่ `.` / protected / no service รัน / มี free space / ไม่มี ref ค้าง) · กำกวม = พักเป็น MOVE_OWNER_DECISION_REQUIRED · **ไม่ลบอะไร** · verify ไม่ตรงแม้ 1 รายการ = หยุดทั้ง batch

## ทำหลายโปรเจกต์พร้อมกัน
ได้ เฉพาะคนละ top-level root + claim บังคับ path ไม่ทับ · ห้าม 2 ตัวลุย tree เดียวกัน

## กันรกถาวร 4 ชั้น
1. Placement Policy: ห้ามสร้างโฟลเดอร์ใหม่ที่ราก
2. ตัวยาม (guardian) cron: **รายงานก่อน** · auto-ย้ายเข้า _Inbox เฉพาะชนิดปลอดภัย · **ห้ามแตะ: ขึ้นต้น `.` / `*.pid *.lock *.sock` / ไฟล์ mtime < 30 นาที** · guardian กับตัวย้ายกลางคืน **ห้ามรันพร้อมกัน**
3. AI ทุกตัว (New/Continue/Relay/Move) อ่าน Placement Policy → ปฏิเสธสร้างโฟลเดอร์ที่ราก
4. _Inbox = ทางออกถูกกฎ

## เกาะสัญญากลาง
อ่าน memory/plan + เคารพ claim/worktree + เขียน ledger Schema §13 · autonomy mapping (Schema §12): ย้าย/Archive ในขอบเขตอนุมัติ+ย้อนกลับได้ = auto · ลบจริง / rename ที่มี ref / แตะ protected / แตะของที่รัน = ขอคน

## Decision Token + Footer
MOVE_SAFE_BATCH_PROPOSED · MOVE_OWNER_DECISION_REQUIRED · MOVE_BLOCKED_NO_TOUCH
Footer ทุกเฟส: สถานะ / ผลตรวจ / ต้องทำต่อ / รออนุมัติ / Decision: \<token\>

## Worktree Lifecycle v1

อ่าน `worktree-lifecycle-contract.md` ก่อนใช้ Prompt นี้ · registered worktree roots และ task worktree ทุกสถานะเป็น no-touch สำหรับ Move Folder · การสำรวจ/ย้ายทะเบียน/cleanup ส่งให้ `hermes worktree scan/import/cleanup`; Shortcut นี้ห้ามย้าย เปลี่ยนชื่อ หรือลบเอง

## Changelog
- v2.1 (2026-06-26): ผ่าน cross-check 2 รอบ · MUST-1 no-touch fail-closed (ทุก `.` = no-touch + allowlist) · MUST-2 ด่าน 9 สแกน ref + symlink เปลี่ยนผ่าน · S-1..S-5 (guardian ปลอดภัย/free space/cross-fs เก็บ source/rollback ใน .hermes รักษา perm/stop batch) · แยก protected (วางของใหม่เข้าได้ ห้ามแตะของเดิม) · เพิ่มเงื่อนไข executor ก่อนปล่อยรันกลางคืน
- v2.0 (2026-06-26): เกาะ Schema v1.1 + no-touch dotfile + 5 root + ด่าน 7-8 + กลางคืน 2 เฟส + กันรก 4 ชั้น
- v1.1 (2026-06-25): router ไป VPS registry + 6 ด่าน

## Purpose
เชื่อมทะเบียน Hermes เข้ากับระบบ Move Folder จริงบน VPS · กัน AI แชทใหม่อ้างว่า shortcut นี้ไม่มี
