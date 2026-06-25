---
title: Use Continue
aliases:
  - Use Continue
  - use-continue
  - Continue
  - continue
  - ทำต่อ
  - ทำต่อเอง
  - ทำงานต่อ
  - ทำต่ออัตโนมัติ
  - ไม่ต้องรอผม
  - Go to Sleep
  - go-to-sleep
  - Sleep Mode
  - sleep-mode
  - เข้าโหมดนอน
  - โหมดนอน
tags:
  - prompt-shortcuts
  - autonomous-work
  - phase-tracking
  - completion
status: active
version: 3.1
updated: 2026-06-24
replaces: go-to-sleep
---

# Use Continue

## Shortcut

```text
Use Continue
```

## Prompt

```text
Use Continue

ทำงานต่อเองโดยไม่ต้องรอผม ทำเป็นระบบทีละเฟส ปิดแต่ละเฟสให้ครบตามนิยาม 100%
ก่อนข้ามเฟส · งานรอบนี้ยึด 1 เป้าหมายหลักเท่านั้น

[ระดับอิสระ 3 ชั้น — ตัดสินก่อนทำทุกอย่าง]

ชั้น 1 ทำเองได้เลย:
อ่านไฟล์ / แก้โค้ดใน scope รอบนี้ / รันเทส / เขียนไฟล์ทดสอบ / เขียน doc

ชั้น 2 รายงานในแชทแล้วทำต่อได้ (ไม่ต้องรอ) แต่ต้องแนบหลักฐานก่อนทำ:
- ลง dependency → แนบ lockfile diff + ผลตรวจ license/security
- ลบไฟล์จำนวนมาก → แนบรายการ dry-run (ไฟล์ที่จะลบ) ก่อน
- แตะไฟล์ข้าม ownership → แนบสรุปผลกระทบต่อเจ้าของเดิม
- แก้ config ที่ไม่ใช่ production

ชั้น 3A auto ผ่านด่านอัตโนมัติ (ทำต่อได้เองไม่ต้องรอคน ถ้าด่านผ่าน):
รายการ: git commit/push/merge/tag/release / deploy production / แก้ CI/CD /
แก้ auth-security policy / production config / migration ข้อมูล / เรียก external API จริง
กติกาด่าน (บังคับทุกข้อ ห้ามข้าม):
1. ด่านที่ต้องรัน (ตายตัว): push/merge = save-git merge-gate · deploy = ship-gate ·
   migration = สำรอง + ทดสอบ restore จริง + verify checksum ก่อน ·
   auth/CI/external API = รันบน staging หรือ dry-run ที่ไม่ใช้ของจริงก่อน
2. ผลด่านต้องมาจากเครื่องมือจริง (exit code / ผลแบบมีโครงสร้าง) — ห้าม AI ตีความว่า SAFE เอง
3. fail-closed: ด่านหาไม่เจอ / timeout / อ่านผลไม่ออก / ผลไม่ใช่ SAFE ชัดเจน = ถือว่า BLOCKED ทันที
4. ผูกด่านกับคำสั่งจริง: บันทึก target (repo/branch/commit SHA/env/service/migration id)
   คำสั่งที่ทำจริงต้องตรงกับ target ที่ด่านตรวจ (ด่านตรวจ commit A ห้าม push commit B)
5. จำกัดขอบเขต: auto ได้เฉพาะ path/service/env/branch ที่ประกาศไว้รอบนี้ เกินจากนี้ = หยุด
6. deploy production ต้องมี health check + คำสั่ง rollback พร้อม + เงื่อนไขหยุดอัตโนมัติถ้า health เพี้ยน
ด่านตอบ SAFE → ทำต่อเลย · BLOCKED → หยุด รายงานชั้นที่ติด
ทุกการกระทำชั้น 3A เขียน ledger (append-only ห้ามแก้ย้อนหลัง):
เวลา / คำสั่ง / ด่านที่รัน / exit code / commit SHA / ผล

ชั้น 3B ต้องขออนุมัติคนชัดก่อนเสมอ (ด่านอัตโนมัติตรวจแทนไม่ได้ · แม้ด่านบอก SAFE):
ใช้เงิน / ส่งอีเมลหรือทำแทนบุคคลภายนอก / เปิดเผย secret / ลบไฟล์สำคัญถาวร

[นิยาม "ว้าว"]
เหนือความคาดหมาย / สร้างประสบการณ์ใหม่ / สิ่งที่ผู้ใช้ไม่เคยเจอ
แต่ "ว้าว" ใช้เป็นเกณฑ์เลือกวิธีที่ดีที่สุดภายใน scope เท่านั้น
ห้ามใช้เป็นเหตุผลขยาย scope · งานนอกเป้าหมายให้บันทึกเป็น backlog เสนอ ไม่ทำเอง

[นิยาม "ผ่าน 100%"]
= ทุก issue ในเฟส verified จริง (test ผ่าน / lint ผ่าน / manual check ผ่าน + มีหลักฐาน)
% เฟส = issue ที่ verified ÷ issue ทั้งหมด · วัดเป็นเลขไม่ได้ให้ใช้ PASS/BLOCKED แทน
ทำไม่ได้จริงให้แยกเป็น blocker ห้ามนับรวม 100%

วิธีทำ:
1. อ่าน context (กติกาโปรเจกต์ / memory / tracking / สถานะ / เป้าหมายรอบนี้)
2. แตกเฟส (เป้าหมาย / issue / วิธีตรวจ / เงื่อนไขผ่าน / output)
3. ทำทีละเฟส จบและตรวจก่อนข้าม · fail ให้แก้จนผ่านหรือระบุ blocker
4. บันทึกเหตุผล + trade-off ทุกครั้งที่ตัดสินใจแทนผู้ใช้
   - เรื่องสำคัญที่ไม่รู้จริง (กระทบทางเลือกหลัก) = หยุดถาม
   - รายละเอียดเล็กระดับวิธีเขียน = เลือกเองได้ แล้วบันทึกเหตุผล

[STOP ทันที]
- worktree dirty เฉพาะไฟล์ที่จะเข้าไปแตะ หรือมีงานคนอื่นค้างเสี่ยงทับ
- จะแตะงานชั้น 3B หรือด่านชั้น 3A ตอบ BLOCKED
- error class เดิม 3 ครั้งไม่ผ่าน
- เจอ secret ใน diff
- ต้องเดาเรื่องสำคัญเพราะ context ไม่พอ
→ หยุด รายงาน ถาม 1 คำถาม

ส่งงานจบ (closeout):
| Phase | เป้าหมาย | % หรือสถานะ | หลักฐาน |
+ สิ่งที่ทำ / ไฟล์ที่แก้ / คำสั่งตรวจ / เรื่องที่เลือกแทนผู้ใช้ /
  ความเสี่ยงค้าง / next step เดียวที่แนะนำสุด
(ช่อง % ใส่ BLOCKED หรือ NOT VERIFIED แทนได้ ถ้าวัดเป็นเลขไม่ได้)
```

## Legacy Names

ชื่อเก่า `Go to Sleep`, `Sleep Mode`, `เข้าโหมดนอน`, และ `โหมดนอน` เป็น alias เพื่อความเข้ากันได้ย้อนหลังเท่านั้น ชื่อหลักที่ต้องใช้และแสดงให้ผู้ใช้เห็นคือ `Use Continue`

## Changelog

- v3.1 (2026-06-24): ผ่านการตรวจ 2 AI (Claude ร่าง · Codex cross-check ด้านความปลอดภัย) · เปลี่ยนงานเสี่ยงสูงจาก "ต้องขออนุมัติคน" เป็น "auto ผ่านด่านอัตโนมัติ" (ชั้น 3A) ตามคำสั่งเจ้าของงาน · ด่านต้อง fail-closed + ผลจากเครื่องมือจริง + ผูกกับคำสั่งจริง + จำกัดขอบเขต · คงงานย้อนยากสุด (เงิน/อีเมล/secret/ลบถาวร) ไว้ที่ชั้น 3B · เพิ่มนิยาม "ว้าว" + นิยาม 100% + STOP rule + closeout
- v3.0 (2026-06-07): เปลี่ยนชื่อจาก Go to Sleep เป็น Use Continue

## Graph Links

- Parent hub: [[skills/README|skills]]
- Router: [[00-Center/docs/AI_SKILL_ROUTER|AI Skill Router]]
- Graph: [[00-Center/docs/SKILL_GRAPH|Skill Graph]]
