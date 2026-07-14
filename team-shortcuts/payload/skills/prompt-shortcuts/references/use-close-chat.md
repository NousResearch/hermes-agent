---
title: Use Close Chat
aliases:
  - Use Close Chat
  - use-close-chat
  - Close Chat
  - close-chat
  - ใช้ Close Chat
  - ปิดแชท
  - ปิดงานแชท
  - จบแชท
tags:
  - prompt-shortcuts
  - close-chat
  - handoff
  - session-memory
  - context-management
status: active
version: 2.4
updated: 2026-07-14
schema: memory-schema-v1.2
pairs_with: use-new-chat >= 1.8
---

# Use Close Chat (v2.4 · 2026-07-14)

คู่กับ Use New Chat ≥ v1.8 · อ้าง Memory Schema v1.2 · เช็ก schema version ตอนเริ่ม · ไม่ตรง = เตือน + ห้ามเขียนไฟล์ความจำจนกว่าจะอ่าน schema ล่าสุด

ปิดแชทแบบปลอดภัย + เขียน "ความจำถาวร" ให้แชทหน้าอ่านตอนเปิด · แก้ปัญหา AI ลืมว่าทำอะไร แก้อะไร ถึงไหน + กัน AI โกหกว่าเสร็จทั้งที่ยังไม่ตรวจ

> บทบาทไม่ทับกัน: Close Chat = พรีวิว+ปิดงาน+เขียน memory · Review Chat = alias โหมดพรีวิวของ Close Chat · Save Git = ตรวจการส่ง Git · Merge to Production = merge/deploy · New Chat = เปิดงาน+อ่าน memory
> Close Chat **ไม่ push/merge/deploy เอง** ชี้ไป Use Save Git / Use Merge to Production

## Prompt

```text
Use Close Chat

คู่กับ Use New Chat ≥ v1.8 · อ้าง Memory Schema v1.2 · เช็ก schema version ตอนเริ่ม · ไม่ตรง = เตือน + ห้ามเขียนไฟล์ความจำจนกว่าจะอ่าน schema ล่าสุด
ปิดแชทแบบปลอดภัย + เขียน "ความจำถาวร" ให้แชทหน้าอ่านตอนเปิด · แก้ปัญหา AI ลืมว่าทำอะไร แก้อะไร ถึงไหน + กัน AI โกหกว่าเสร็จทั้งที่ยังไม่ตรวจ
บทบาทไม่ทับกัน: Close Chat = พรีวิว+ปิดงาน+เขียน memory · Review Chat = alias โหมดพรีวิวของ Close Chat · Save Git = ตรวจการส่ง Git · Merge to Production = merge/deploy · New Chat = เปิดงาน+อ่าน memory
Close Chat ไม่ push/merge/deploy เอง ชี้ไป Use Save Git / Use Merge to Production

[โหมดเดียว 2 ช่วง — ยุบงานซ้ำจาก Review Chat]
- ช่วง PREVIEW: ตรวจบทสนทนา/หลักฐาน/งานค้าง แล้วแสดงว่าจะเขียนไฟล์ใดและเพิ่มอะไร · ยังไม่เขียน
- ช่วง CLOSE: เมื่อเจ้าของสั่งปิดและบันทึกชัด หรืออนุมัติพรีวิวแล้ว จึงเขียนความจำและออก Decision Token
- ผู้ใช้เรียก `Review Chat` = รันเฉพาะ PREVIEW แล้วหยุดก่อนเขียน · ห้ามสร้างข้อความส่งต่อยาวซ้ำกับ memory; ให้ข้อความเปิดแชทสั้นที่ชี้ไฟล์หลัก
- ผู้ใช้เรียก `Use Close Chat` โดยยังไม่มีสิทธิ์เขียน = ทำ PREVIEW ให้ครบแล้วขออนุมัติครั้งเดียวสำหรับชุดไฟล์ทั้งหมด ไม่ถามรายไฟล์

[Save Git Evidence Receipt — ใช้หลักฐานเดิมโดยไม่ตรวจหนักซ้ำ]
- ถ้ามีผล Use Save Git จากงานเดียวกัน ให้รับได้เฉพาะเมื่อ project/task/branch/HEAD SHA ตรง และ `git status --short` รอบปิดยืนยันว่าไม่มีไฟล์เปลี่ยนหลังใบตรวจ
- ใบตรวจตรง = ใช้ผล build/test/CI/VPS/health จากใบนั้นได้ ไม่รันด่านหนักซ้ำ · ยังต้องรัน git status สดเสมอ
- ไม่มีใบตรวจ/ค่าไม่ตรง/จะกล่าวอ้างพร้อม push-merge-deploy = เรียก Use Save Git ตาม stage ที่เกี่ยว
- แชทไม่มีการเปลี่ยน Git และไม่มีการส่งโค้ด = Save Git เป็น N/A ห้ามเรียก 5 ด่านโดยไม่จำเป็น

กฎบังคับ (เหมือน New Chat — เพราะนี่คือด่านสุดท้าย ยิ่งต้องเข้ม):
- เจ้าของงานอาจเป็น non-dev → อธิบายภาษาคน · ทุกคำสั่งให้เขารันต้องก๊อปวางได้ · ห้ามถามว่า "ใช้ test ตัวไหน" ค้นเอง
- ห้ามบอกว่า "ปิดแล้ว/clean/ผ่าน" โดยไม่รัน command จริง · ทุกค่ามาจาก command จริง
- ไม่มี output จริงแปะ = ถือเป็น claimed อัตโนมัติ (กฎเหล็กกันโกหก)
- redact secret ทุกอย่างก่อนเขียน memory (Schema §7)
- แนบ Evidence footer (timestamp/host/cwd/commands) ท้ายรายงานเสมอ

Pre-Close Gate (6 ด่าน):

1. Commit ค้างไหม — รัน `git status --short --branch` จริง · มีไฟล์ค้าง = ยังปิด clean ไม่ได้ → ถามเจ้าของ: commit / stash / discard (เสนอคำสั่งก๊อปวางทั้ง 3 ทาง)
2. ต้อง push/merge/deploy ไหม — ถ้าใช่ ชี้ไป Use Save Git / Use Merge to Production · Close ไม่ทำเอง
3. Quality Gate ผ่านจริงไหม — ค้น gate จาก repo เอง (Schema §5: package.json/Makefile/pyproject/CI) แล้วรันจริง + แปะ output · gate ใดไม่มีในโปรเจกต์ = N/A · ไม่เจอ gate เลย = บอกตรง ๆ ไม่เดาว่าผ่าน
4. ไล่ทุก task: verified หรือ claimed — ตัดสินตามบันไดหลักฐาน (Schema §4) · claimed ที่ยังไม่ตรวจ = งานค้าง · ห้ามเลื่อน claimed ขึ้น verified เอง
5. Deploy verify (เฉพาะถ้ารอบนี้ถึงขั้น merge→main) — โปรเจกต์ deploy แบบ CI/CD auto-on-merge:
   - ตรวจสถานะ CI run ของ SHA ที่ merge (เช่น `gh run list` สำหรับ GitHub / `glab ci status` หรือ pipeline API สำหรับ GitLab) → ต้องเขียวถึงนับ deploy verified
   - ถ้ามี health endpoint → ยิงดูตอบ 200 ไหม + เทียบ live SHA กับ SHA ที่ตั้งใจ deploy
   - CI ยังรัน/แดง/ไม่เทียบ SHA = `NEED_OWNER_ACTION_BEFORE_CLOSE` ห้ามบอกว่า deploy แล้ว
6. Claim + dirty — ปลด claim ของงานนี้ (release/เปลี่ยนเป็น handoff) · dirty ที่เป็นงานนี้ต้องตัดสินก่อน · dirty ของคนอื่น = ไม่แตะ รายงาน
   - ถ้าจบที่ `NEED_OWNER_ACTION_BEFORE_CLOSE` → ห้ามปลด claim (กันงานหลุดให้คนอื่นทับทั้งที่ยังไม่จบ)

สรุป prompt ทั้งแชท (กัน AI ลืมคำสั่งเจ้าของ):
ไล่ prompt ทั้งแชทของเจ้าของงาน สรุปเป็น: เป้าหมายเดิม / คำสั่งที่เปลี่ยนทิศ / decision สำคัญ / ข้อห้าม → ข้อไหนยังไม่ทำหรือตกหล่น เอามาแสดง

เขียนความจำถาวร (review-before-write — เสนอก่อน รออนุมัติถ้าไม่มีคำสั่งชัด):
เขียนตาม Memory Schema v1.2 §1 (**ความจำทำงานต่อ = `.project/` ที่เดียว · ห้ามเขียน `.hermes/` หรือ root อีก**) · redact secret ก่อนเขียนทุกครั้ง (§7)

1. `.project/OverviewProgress.md` = **แหล่งจริงของสถานะ — อัปเดตทุกครั้งก่อนอย่างอื่น** (Schema §1c)
   - เช็กบรรทัดแรก: ป้าย `> memory-schema: v1.2` + สารบัญบังคับอ่าน (ไม่มี = เติมให้)
   - อัปเดต 4 หัวข้อบนสุด: `สถานะล่าสุด` (ขั้นไหน/SHA/gate) · `งานถัดไป` (1-3 ข้อ) · `ข้อห้าม/กติกาล็อก` · `งานค้าง/ส่งต่อ` (claimed + next_owner) + pointer ไป session log
   - ประวัติรอบนี้ต่อท้ายส่วนล่าง · ห้ามดันสถานะปัจจุบันจมลงล่าง
   - ถ้าไฟล์ถูกแก้หลังเริ่มแชท → อ่าน diff ก่อนเขียน · ชนกับคนอื่น = เขียนเข้า review queue ก่อน ไม่ทับทันที
   - [Migration Schema §1b] ถ้ายังเจอ `handoff.md`/`.hermes/active.md`/`.hermes/decisions.md`/`.hermes/plan.md` ที่มีเนื้อหาจริง → ย้ายเข้า `.project/` + ทำไฟล์เก่าเป็น stub ชี้ทางใหม่ (ห้ามลบ) + แก้จุดอ้างทางเก่าในไฟล์กฎราก ก่อนปิด
   - [ด่านไฟล์เข้า git จริง — Schema §1b เพิ่ม 2026-07-05] หลังเขียน/ย้ายไฟล์ `.project/` รัน `git check-ignore -v .project/*.md` (ต้องไม่เจอ) + `git ls-files .project/` (ต้องเห็นครบ) · โดนซ่อน → เจาะช่องอนุญาต `!.project/` + `!.project/**` ใน `.gitignore` แล้วตรวจซ้ำ · ไฟล์ความจำไม่เข้า git = ปิดแชท clean ไม่ได้
2. session log (path ตาม Schema) เนื้อขั้นต่ำ:
   - เป้าหมายรอบนี้
   - Changed-files table: | file | owner | changed_by | reason | verification(verified/claimed+หลักฐาน) | risk | next_owner |
   - Decision log: ตัดสินใจอะไร + ทำไมเลือกทางนี้
   - Quality gate ที่รัน + output จริง
   - Deploy: merge SHA / CI status / live SHA (ถ้ามี)
   - งานค้าง (รวม claimed ที่ยังไม่ตรวจ) + เจ้าของถัดไป
   - ความเสี่ยงที่เหลือ
   - next step + ข้อความเปิดแชทหน้า (พร้อมก๊อป)
3. อัปเดต pointer + ไฟล์ประสาน (สำคัญ — New Chat อ่านพวกนี้):
   - `latest-close.md` (pointer ต่อ staff ตาม Schema §6)
   - `.project/decisions.md` = append decision สำคัญรอบนี้ (ห้ามเขียนทับของเดิม)
   - (งานกำลังทำ/ค้าง ไม่มีไฟล์แยกแล้ว — อยู่ในหัวข้อ `งานค้าง/ส่งต่อ` ของ OverviewProgress ข้อ 1)

Business Plan Sync (capability-based — ทำเฉพาะเมื่อมี `.project/BusinessPlan.md`):
ถามตัวเอง: รอบนี้มี feature ใหม่ / ราคาเปลี่ยน / เจอ insight ลูกค้าใหม่ / คู่แข่งเปลี่ยน ที่ขยับแผนธุรกิจไหม
- ไม่มี → ระบุใน Output ว่า "Business Plan: รอบนี้ไม่มีการเปลี่ยนด้านธุรกิจ"
- มี → อัป [fact] ใน `.project/BusinessPlan.md` + ต่อท้าย log ใน `.project/BusinessPlan-Full.md` (ห้ามทับ log เก่า)
ห้ามบังคับอัปเดตทั้งที่รอบนั้นไม่มีอะไรกระทบธุรกิจ (กันไฟล์รก/ข้อมูลลม) · รายละเอียดเทมเพลตอยู่ใน Use BusinessPlan

QA/QC Scan Sync (capability-based — ทำเฉพาะเมื่อมี `.project/qaqc-scan.md` · คู่กับ Use QA QC ≥ v1.0):
ถามตัวเอง: แชทนี้มีการสแกน / แก้ปัญหาจากตาราง QA/QC / งานแก้ QQF-* เปลี่ยนสถานะไหม
- ไม่มี → ระบุใน Output ว่า "QA/QC: รอบนี้ไม่มีการสแกน/แก้"
- มี → อัปเดต `.project/qaqc-scan.md`: สถานะรายหมวด + ตารางปัญหาคงค้าง (ตัดเฉพาะข้อที่ verified มีแถว gate-run) + ประวัติรอบ (append ใหม่บนสุด) · สถานะ "verified" ใน qaqc-scan.md ใช้บันไดหลักฐานเดียวกับข้อ 4 ห้ามหย่อนกว่า

Decision Token ปิด (ตาม Schema §2):
- `CLOSED_CLEAN` — commit ครบ ไม่มีค้าง gate เขียว (CI เขียวถ้ามี deploy)
- `CLOSED_WITH_PENDING` — ปิดได้แต่มีงานค้าง/claimed · ระบุชัดว่าค้างอะไร
- `NEED_OWNER_ACTION_BEFORE_CLOSE` — ยังปิดไม่ได้ (dirty/ยัง commit/ยัง merge/CI แดง) · ห้ามบอกว่าปิดแล้ว · ห้ามปลด claim · แสดงคำสั่งแก้แบบก๊อปวาง

รูปแบบ Output บังคับ:

Close Chat Report
สรุปภาษาคน: ทำอะไรไป / เหลืออะไร / ลืมคำสั่งไหนไหม
Tasks: <task> = verified(หลักฐาน) | claimed(เหตุที่ยังไม่ตรวจ)   ← ไล่ทุกอัน
Quality Gate: <gate ที่ค้นเจอ> = ผล + output (หรือ N/A / ไม่พบ gate)
Deploy: merge SHA / CI status / live SHA   (หรือ N/A ถ้ายังไม่ถึง merge)
Memory written: session log path / OverviewProgress(4 หัวข้อบน) / decisions.md(+N บรรทัด) / migration(ถ้าย้ายไฟล์เก่า)
Business Plan: <อัปอะไรใน .project/BusinessPlan(.md/-Full.md) หรือ "ไม่มีการเปลี่ยนด้านธุรกิจ" หรือ N/A ถ้าไม่มีไฟล์>
QA/QC: <อัปอะไรใน .project/qaqc-scan.md หรือ "รอบนี้ไม่มีการสแกน/แก้" หรือ N/A ถ้าไม่มีไฟล์>
Decision Token: <token> + เหตุผล
งานค้าง: <รายการ> + เจ้าของถัดไป
ข้อความเปิดแชทหน้า: <ก๊อปวางได้>
Evidence: timestamp / host / cwd / commands ที่รันจริง

ตัวอย่าง CLEAN: Decision Token: CLOSED_CLEAN · commit ครบบน feature/x · test+lint+build เขียว (output แปะแล้ว) · merge→main CI เขียว live SHA ตรง · ปลด claim แล้ว
ตัวอย่าง BLOCKED: Decision Token: NEED_OWNER_ACTION_BEFORE_CLOSE · มี 2 ไฟล์ค้างยังไม่ commit + CI run ของ merge ล่าสุดยัง failing · ยังไม่ปิด ไม่ปลด claim · รัน: git add -A && git commit -m "..." แล้วรอ CI เขียวก่อนปิดใหม่

ข้อห้าม: บอก "ปิดแล้ว" โดยไม่รัน git/CI จริง · เลื่อน claimed เป็น verified โดยไม่มี output · เขียน secret ลง memory · ปลด claim ตอนยังไม่จบ · สรุปด้วยศัพท์เทคนิคโดยไม่แปล
```

## Worktree Lifecycle v1

อ่าน `worktree-lifecycle-contract.md` ก่อนใช้ Prompt นี้ · ปิดสิทธิ์ด้วย `hermes worktree close`; ถ้า merged จึงประเมิน `cleanup` 6/6 แบบ dry-run · ห้ามลบทันที และต้องรายงาน quarantine/cleanup state ใน Close Chat Report

## Changelog

- v2.4 (2026-07-14): ยุบ Review Chat เป็นโหมด PREVIEW ของ Close Chat · เพิ่ม Save Git Evidence Receipt เพื่อลดการตรวจซ้ำ โดยยังบังคับ git status สดและตรวจค่า project/task/branch/SHA ก่อนรับหลักฐาน
- v2.3 (2026-07-10): เพิ่มขั้น QA/QC Scan Sync (capability-based — เฉพาะโปรเจกต์ที่มี `.project/qaqc-scan.md`) + บรรทัด QA/QC ใน Output · คู่กับ Use QA QC v1.0 (แผน QAQC-P4-I2) · verified ใน qaqc-scan.md ใช้บันไดหลักฐานเดิม ไม่หย่อน
- v2.2 (2026-07-05): เกาะ Memory Schema v1.2 — เลิกเขียน `handoff.md`/`.hermes/active.md`/`.hermes/decisions.md` · แหล่งจริงของสถานะ = `.project/OverviewProgress.md` (4 หัวข้อบนสุด อัปเดตทุกครั้งก่อนอย่างอื่น) + `.project/decisions.md` (append) · เพิ่มขั้นย้ายไฟล์เก่าเป็น stub ตอนปิด · schema ไม่ตรง = ห้ามเขียนความจำ (คำสั่งเจ้าของ 2026-07-05 · ตรวจข้ามค่าย Grok+Codex)
- v2.1 (2026-06-28): เพิ่มขั้น Business Plan Sync (capability-based) · ปิดแชทแล้วถ้ามี `.project/BusinessPlan.md` ให้เช็คว่ารอบนี้มีอะไรกระทบแผนธุรกิจไหม → อัป BusinessPlan(.md/-Full.md) เฉพาะเมื่อเปลี่ยนจริง + เพิ่มบรรทัด Business Plan ใน Output · ผูกกับ Use BusinessPlan v1.0 (ชุด Project OS)
- v2.0 (2026-06-26): เขียนใหม่ทั้งฉบับให้เกาะ Memory Schema v1.1 · เพิ่ม Pre-Close Gate เป็น 6 ด่าน (Quality Gate auto-detect ค้นเอง + Deploy verify ผ่านสถานะ CI ของ merge SHA) · ผูก verified/claimed กับบันไดหลักฐาน · ห้ามปลด claim ตอน NEED_OWNER_ACTION · เขียน memory เพิ่ม active.md + decisions.md (append) · Output format ตายตัว + Evidence footer · เช็ก schema version ตอนเริ่ม
- v1.0 (2026-06-24): สร้างใหม่ตามโจทย์เจ้าของงาน (แก้ AI ลืมงานข้ามแชท) · Pre-Close Gate 5 ด่าน · handoff สั้น + session log แยกคน/วัน/branch + latest-close pointer · Decision token 3 แบบ · ไม่ push/merge เอง

## Graph Links

- Parent hub: [[skills/prompt-shortcuts/Prompt Shortcuts|Prompt Shortcuts]]
- Registry: [[ai-context/prompt-shortcut-registry|Prompt Shortcut Registry]]
- Schema: [[skills/prompt-shortcuts/references/memory-schema|Memory Schema v1.2]]
- Pair: [[skills/prompt-shortcuts/references/use-new-chat|Use New Chat]]
- Related: [[skills/prompt-shortcuts/references/review-chat|Review Chat]]
- คู่มือเจ้าของงาน: [[skills/prompt-shortcuts/references/memory-system-owner-guide-th|ระบบความจำข้ามแชท — คู่มือเจ้าของงาน]]
