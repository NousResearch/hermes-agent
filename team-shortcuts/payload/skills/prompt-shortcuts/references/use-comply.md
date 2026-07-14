---
title: Use Comply
aliases:
  - Use Comply
  - use-comply
  - Comply
  - comply
  - ใช้ Comply
  - ทำ Comply
  - แตกเฟส
  - ทำตารางเปอร์เซ็นต์
tags:
  - prompt-shortcuts
  - planning
  - compliance
  - phase-tracking
status: active
version: "3.2"
updated: 2026-07-14
schema: memory-schema-v1.2
---

# Use Comply (v3.2 · 2026-07-14)

คู่กับ Memory Schema v1.2 · เช็ก schema version ตอนเริ่ม · ไม่ตรง = เตือน + ห้ามเขียนไฟล์ความจำจนกว่าจะอ่าน schema ล่าสุด

> เปลี่ยนจาก v2.1: อ่าน plan/ความจำก่อน · ค้น gate เอง · ใช้ id ร่วม · กฎเหล็ก "ไม่มี output = claimed" · CI = ตัววัด deploy · redact

## Shortcut

```text
Use Comply
```

## Prompt

```text
Use Comply กับงานนี้

แตกงานเป็น phase และ issue ย่อยละเอียด กัน AI ลืม/ข้ามขั้น/ทำไม่ครบ/รายงานเสร็จทั้งที่ยังไม่ตรวจจริง

[กฎ non-dev] อธิบายภาษาคน · ห้ามถามว่าใช้ test ตัวไหน — ค้นเอง (Schema §5) · หลักฐานก๊อปวางได้

[ขั้น 0 — อ่านก่อนแตกงาน]
- ถ้ามี `.project/plan.md` (จาก Act-As · Schema v1.2) → แตก issue ใต้ phase_id เดิม ไม่ตั้งเฟสใหม่ซ้ำ · เจอแต่ `.hermes/plan.md` เก่า = ทำ Migration ตาม Schema §1b ก่อน (ย้าย + stub ห้ามลบ)
- อ่าน `.project/OverviewProgress.md` (4 หัวข้อบนสุด) + decision token (Schema §2) · `NEED_OWNER_ACTION_BEFORE_CLOSE` = เตือนก่อนลุย

[เกณฑ์ "งานใหญ่"] เข้าข้อใดข้อหนึ่ง: เกิน 3 เฟส / เกิน 8 issue / แตะหลายโมดูลหรือข้าม ownership / มี deploy หรือ migration / มีข้อมูลสำคัญ-เงิน-ความปลอดภัย → ทยอยส่งทีละเฟส (เว้นแต่สั่งให้ทำต่ออัตโนมัติ) · เล็กกว่านี้ส่งรวดเดียวได้ แต่ยังต้องมีหลักฐาน

[สถานะมาตรฐาน — ใช้ 6 คำ] not_started / in_progress / blocked / failed / verified / unknown
- verified = รันตรวจจริงแล้วผ่าน (ใช้ได้เฉพาะมีหลักฐาน) · failed = ตรวจแล้วไม่ผ่าน · blocked = ติดเงื่อนไข · unknown = ข้อมูลไม่พอ
- ไม่แน่ใจ = blocked/unknown · ห้ามเดาว่า verified
- [กฎเหล็ก Schema §3] ไม่มี output จริงแปะ = ถือเป็น claimed (ไม่ใช่ verified) อัตโนมัติ
- [Schema §11] ตอนคิด % และตอนส่งให้ Close: verified เท่านั้น = เสร็จ · 5 สถานะที่เหลือ = งานค้าง

[สูตร %] ราย issue: verified=100% นอกนั้น=0% · % เฟส = verified÷ทั้งหมดในเฟส ×100 · % รวม = verified÷ทั้งหมด ×100 · ปัดจำนวนเต็ม · ห้าม % จากความรู้สึก

[Two-Zone Gate — จำแนกก่อนส่งให้ Use Continue]
- ทุก issue ต้องมี `zone` ก่อนเริ่มทำงาน: `ZONE_A` หรือ `ZONE_B` · ไม่มี zone = ยังเริ่มไม่ได้
- `ZONE_A` = งานไม่มีผลภายนอกหรือย้อนกลับได้ในขอบเขตที่อนุมัติ เช่น อ่าน/วิเคราะห์/แก้ไฟล์ใน allowed_paths/เขียน test หรือ doc/รัน test-lint-build · Use Continue ทำต่อเองได้ทั้ง Phase โดยไม่ถามราย issue
- `ZONE_B` = งานข้าม scope/ownership, พื้นที่ dirty ของคนอื่น, dependency/config ที่มีผลกว้าง, push/merge/deploy, production, migration, ลบถาวร, เงิน, secret หรือสื่อสารภายนอก
- รวม `ZONE_B` ที่มีเป้าหมายเดียวกันเป็น `approval_phase` เดียว · เจ้าของอนุมัติครั้งเดียวครอบทุก issue ใน Phase ที่ระบุ `task_id + branch + base_sha + allowed_paths`
- ถ้า path/branch/SHA/ผลภายนอกเปลี่ยน = สิทธิ์เดิมหมดอายุ ต้องจัด Phase ใหม่ · ห้ามขยายสิทธิ์เงียบ
- การอนุมัติ Zone A หรือ Phase ไม่ยกเลิกด่านจริง: worktree/claim/secret/test/save-git/production ยังต้องผ่านตามชนิดงาน

ลำดับการทำ:

1. แยกเป้าหมาย — ผลสุดท้าย / scope / out of scope / ถ้าเสร็จผู้ใช้ได้อะไร

2. แตก phase — แต่ละเฟส: `phase_id` (จาก plan.md ถ้ามี) / ชื่อ / เป้าหมาย / รายการ issue / เกณฑ์ผ่าน / วิธีตรวจ / output

3. แตก issue — แต่ละ issue: `issue_id` (P1-I1…) / ชื่อ / รายละเอียด / ไฟล์เกี่ยวข้อง / เงื่อนไขเสร็จ / วิธีตรวจ / สถานะ / blocker / `zone` / `risk_reason` / `approval_phase` / `allowed_paths` / `external_effect`

4. ตาราง comply — | phase_id | issue_id | zone | approval_phase | สถานะ | % | หลักฐาน | (หลักฐานละเอียดเฉพาะ verified/blocked/failed)

5. Verification — ค้น gate เอง (ใหม่)
   - งานโค้ด: ค้น quality gate จาก repo เอง (Schema §5: package.json/Makefile/CI) แล้วรันจริง + แปะ output · localhost เปิดได้จริง
   - deploy = ดูสถานะ CI run ของ SHA ที่ merge (ไม่ใช่ endpoint ลอย ๆ) + health 200 + live SHA ตรง (Schema §4)
   - งานเอกสาร/ความรู้: ไฟล์มีจริง / ลิงก์ถูก / เนื้อครบตาม issue / ไม่มีศัพท์ไม่แปล / มีส่วนให้รีวิว
   - ไม่พบ gate เลย = รายงานตรง ๆ ว่าไม่มีตัวตรวจอัตโนมัติ ไม่เดาว่าผ่าน

[แบบฟอร์มหลักฐาน — ทุก issue verified ต้องครบ · redact secret §7]
คำสั่งที่รัน / ผลลัพธ์ / ไฟล์หรือ URL / วันเวลา / ข้อจำกัด
ตรวจไม่ได้ → เขียนเหตุผล (ไม่มี dependency/สิทธิ์/env) · ห้ามใส่ "ผ่าน" ถ้าไม่ได้ตรวจจริง

6. รายงานท้ายเฟส — สรุปทำอะไร / verified กี่ตัว / ค้างกี่ตัว / % เฟส / หลักฐาน / ความเสี่ยง / ทำต่ออะไร
   - ห้ามบอก 100% ถ้ายังมี issue ที่สถานะไม่ใช่ verified
   - งานใหญ่ห้ามส่งก้อนเดียว ทยอยทีละเฟส ตรวจครบก่อนบอกเสร็จ

ข้อห้าม: เดา verified · % จากความรู้สึก · เขียน secret ลงหลักฐาน · ถามเจ้าของว่าใช้ test ตัวไหน · ตั้ง phase ใหม่ทับ id จาก plan.md · ศัพท์เทคนิคไม่แปล
```

## Worktree Lifecycle v1

อ่าน `worktree-lifecycle-contract.md` ก่อนใช้ Prompt นี้ · ทุก Issue ที่เขียนไฟล์ต้องผูก `task_id + worktree + machine + evidence`; ตารางเปอร์เซ็นต์แยกสถานะ lifecycle และห้ามนับ verified ถ้า WTL gate ไม่พร้อม

## Changelog

- v3.2 (2026-07-14): เพิ่ม Two-Zone Gate ตามคำสั่งเจ้าของ · ทุก issue ต้องเป็น ZONE_A/ZONE_B · Zone A ทำต่อเองได้ทั้ง Phase · Zone B รวมขออนุมัติครั้งเดียวต่อ approval_phase แทนการถามราย issue
- v3.1 (2026-07-05): เกาะ Memory Schema v1.2 — อ่านแผนจาก `.project/plan.md` (เดิม `.hermes/plan.md`) + อ่านสถานะจาก `.project/OverviewProgress.md` · เจอไฟล์เก่า = Migration §1b ก่อน · schema ไม่ตรง = ห้ามเขียนความจำ (คำสั่งเจ้าของ 2026-07-05)
- v3.0 (2026-06-26): เกาะ Memory Schema v1.1 · เพิ่มขั้น 0 อ่าน plan.md + handoff + token ก่อนแตกงาน (แตก issue ใต้ phase_id เดิม) · ค้น quality gate เอง (§5) · ใช้ issue_id ร่วม (P1-I1 ตาม §10) · กฎเหล็ก "ไม่มี output = claimed" (§3) · จับคู่สถานะ §11 · deploy วัดจาก CI run ของ merge SHA (§4) · redact secret (§7) · กฎ non-dev
- v2.1 (2026-06-24): เพิ่มสูตร % (นับเฉพาะ verified) · สถานะมาตรฐาน 6 ค่า + แยก failed กับ blocked · เกณฑ์ "งานใหญ่" · แบบฟอร์มหลักฐาน
- v2.0 (2026-05-28): อัปเกรดเป็น prompt v2

## Graph Links

- Parent hub: [[skills/prompt-shortcuts/Prompt Shortcuts|Prompt Shortcuts]]
- Registry: [[ai-context/prompt-shortcut-registry|Prompt Shortcut Registry]]
- Schema: [[skills/prompt-shortcuts/references/memory-schema|Memory Schema v1.1]]
- Prev in lifecycle: [[skills/prompt-shortcuts/references/use-act-as|Use Act-As]]
- Next in lifecycle: [[skills/prompt-shortcuts/references/use-continue|Use Continue]]
