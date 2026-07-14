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
version: "4.5"
updated: 2026-07-14
schema: memory-schema-v1.2
replaces: go-to-sleep
---

# Use Continue (v4.5 · 2026-07-14)

คู่กับ Memory Schema v1.2 · เช็ก schema version ตอนเริ่ม · ไม่ตรง = เตือน + ห้ามเขียนไฟล์ความจำจนกว่าจะอ่าน schema ล่าสุด

> เปลี่ยนจาก v3.1: prod merge/deploy ออกจาก auto (ต้องขอคน · flag ALLOW_AUTO_PROD=OFF) · อ่าน memory+token ก่อนลุย · ledger ผูกเข้า memory · ค้น gate เอง · เคารพ claim/worktree · redact

## Shortcut

```text
Use Continue
```

## Prompt

```text
Use Continue

ทำงานต่อเองโดยไม่ต้องรอผม ทีละเฟส ปิดแต่ละเฟสให้ครบ 100% ก่อนข้าม · งานรอบนี้ยึด 1 เป้าหมายหลักเท่านั้น

[กฎ non-dev] เจ้าของเป็น non-dev → อธิบายภาษาคน · ค้น gate เอง · ทุกครั้งที่ต้องขอคน เสนอคำสั่ง/ทางเลือกก๊อปวางได้

[ขั้น 0 — อ่านก่อนลงมือ]
- อ่าน `.project/plan.md` (เฟส+id) + `.project/OverviewProgress.md` (4 หัวข้อบนสุด: สถานะ/งานถัดไป/ข้อห้าม/งานค้าง) + `.project/decisions.md` + decision token (Schema §2)
- เจอไฟล์ความจำเก่าใน `.hermes/` หรือ root ที่ยังมีเนื้อหาจริง → Migration ตาม Schema §1b ก่อนลุย (ย้าย + stub ห้ามลบ + แก้จุดอ้าง)
- `NEED_OWNER_ACTION_BEFORE_CLOSE` รอบก่อน = หยุด เตือนเจ้าของ ห้ามลุยต่อ จนกว่าเคลียร์
- เคารพ claim/worktree: route ไป worktree ตัวเอง · path ซ้อน claim คนอื่น = STOP · ไม่แตะ worktree คนอื่น
- อ้าง phase_id/issue_id เดิมจาก plan/comply ตลอด ไม่ตั้งใหม่ · ถ้า plan.md ประกาศ plan_id (เช่น GRD) เลขงานต้องขึ้นต้นด้วย plan_id นั้น · เลขที่ไม่มีใน plan.md = ห้ามทำ

[กฎ re-anchor — กันลืมแผนหลังคำถามแทรก · แผน GRD 2026-07-07]
หลังตอบคำถามแทรก / ออกนอกเรื่อง / สลับงาน — ก่อนแตะไฟล์หรือลงมือครั้งถัดไป ต้องเปิด `.project/plan.md` ทวน "เฟสปัจจุบัน + ข้อห้าม" ก่อนเสมอ · ห้ามทำต่อจากความจำในแชท (ปฐมเหตุ: AI ตอบคำถามเสร็จแล้วลืมแผน ทำโปรเจกต์พัง · ละเมิด "ตอบโดยไม่ทวนโจทย์" 3,790 ครั้ง)

[Two-Zone Execution — ลดการกดอนุมัติรายนาที]
- รับตารางจาก Use Comply ที่ติดป้ายทุก issue เป็น `ZONE_A` หรือ `ZONE_B` · ไม่มีป้าย = กลับไปจำแนกก่อน ไม่ถามเจ้าของราย issue
- `ZONE_A`: ทำต่อเองทั้ง Phase จน verified 100% — อ่าน/วิเคราะห์/แก้ใน allowed_paths/เขียน test-doc/รัน test-lint-build/แก้ผลตรวจใน scope เดิม · ห้ามหยุดขออนุมัติระหว่าง issue
- `ZONE_B`: ยังไม่ทำ · รวบ issue ที่เกี่ยวกันเป็น `approval_phase` พร้อมคำสั่ง ผลกระทบ วิธีย้อนกลับ และเงื่อนไขหยุด แล้วขอเจ้าของครั้งเดียวต่อ Phase
- อนุมัติ Phase แล้วทำทุก issue ใน Phase ต่อเองได้ ตราบใดที่ `task_id + branch + base_sha + allowed_paths + external_effect` ยังตรง
- หยุดและจัด Phase ใหม่เฉพาะเมื่อ scope/path/branch/SHA เปลี่ยน, ชน ownership, พบ secret, ด่านจริง BLOCKED, จะแตะ production/เงิน/บุคคลภายนอก/ลบถาวร
- คำถามย่อย การเลือกวิธีภายใน scope และการแก้ test ที่ไม่ผ่าน ไม่ใช่เหตุให้ขออนุมัติใหม่ · เลือกทางที่ปลอดภัยสุดแล้วบันทึกเหตุผล

[ระดับอิสระ 3 ชั้น — ตัดสินก่อนทำทุกอย่าง · อิง Schema §12]
ชั้น 1 = ZONE_A ทำเองได้เลย: อ่านไฟล์ / แก้โค้ดใน scope รอบนี้ / รันเทส / เขียนไฟล์ทดสอบ / เขียน doc
ชั้น 2 รายงานแล้วทำต่อได้ (แนบหลักฐานก่อน): ลง dependency (+ lockfile diff + license/security) / ลบหลายไฟล์ (+ dry-run list) / แตะไฟล์ข้าม ownership (+ สรุปผลกระทบ) / แก้ config ที่ไม่ใช่ production
ชั้น 3A auto ผ่านด่านอัตโนมัติได้: push ขึ้น branch ตัวเอง / tag / deploy staging / แก้ CI ที่ไม่กระทบ prod / external API บน sandbox

ชั้น 2/3A จัดเป็น ZONE_B ก่อนเสมอ · เมื่อ approval_phase ได้รับอนุมัติและด่านเครื่องมือคืน SAFE จึงทำต่อได้โดยไม่ถามซ้ำราย issue

[Phase Write Permit จาก Use New Chat — หนึ่งสิทธิ์ต่อ Phase ไม่ใช่ต่อ issue]
- ก่อนแก้ไฟล์ครั้งแรกของ Phase ต้องมี `task_id / approval_phase / branch / base_sha / allowed_paths / owner_approval / claim_status` ที่ยังตรงกับ Git จริง
- สิทธิ์เดียวใช้ได้กับทุก issue ใน Phase และ allowed_paths เดิม · ห้ามถามอนุมัติซ้ำระหว่าง Phase
- คำสั่งใหม่ เปลี่ยนเป้าหมาย เพิ่ม path เปลี่ยน branch/SHA หรือกลับมาหลังบริบทไม่แน่นอน ต้องตรวจ branch/status/claim ใหม่ แต่ถ้าทุกค่าตรงให้ทำต่อได้ทันที ไม่ต้องขอเจ้าของซ้ำ
- ถ้าเป็น `NEW_WRITABLE_TASK` ต้องกลับไป WTL Gate ให้ Manager แสดง dry-run และขออนุมัติก่อน `--apply` · ห้ามถือว่า Use Continue อนุญาตใช้ task/branch เดิมกับทุกงาน

[เปลี่ยนสำคัญ] ต้องขอคนเสมอ (ค่าตั้งต้น · แม้ด่าน SAFE):
- merge → main · deploy production · migration prod → เพราะ merge→main = CI/CD ดันขึ้น prod ที่มีผู้ใช้จริงอัตโนมัติ
- เปิด auto ได้เฉพาะเมื่อเจ้าของตั้ง `ALLOW_AUTO_PROD=ON` ชัดเจน (Schema §12) · เปิดแล้วยังต้องผ่านด่าน+ledger

ชั้น 3B ต้องขอคนเสมอ: ใช้เงิน / ส่งอีเมลหรือทำแทนบุคคลภายนอก / เปิดเผย secret / ลบไฟล์สำคัญถาวร

[กติกาด่าน 3A — บังคับทุกข้อ]
1. ด่านตายตัว: push/merge = save-git merge-gate · deploy = ship-gate · migration = สำรอง+ทดสอบ restore จริง+verify checksum ก่อน · auth/CI/external API = staging หรือ dry-run ก่อน
2. ผลด่านจากเครื่องมือจริง (exit code/ผลมีโครงสร้าง) — ห้าม AI ตีความว่า SAFE เอง
3. fail-closed: ด่านหาไม่เจอ/timeout/อ่านผลไม่ออก/ไม่ใช่ SAFE ชัดเจน = BLOCKED ทันที
4. ผูกด่านกับคำสั่งจริง: บันทึก target (repo/branch/SHA/env/service/migration id) · ด่านตรวจ commit A ห้าม push commit B
5. จำกัดขอบเขต: auto เฉพาะ path/service/env/branch ที่ประกาศรอบนี้ เกิน = หยุด
6. deploy ต้องมี health check + คำสั่ง rollback พร้อม + เงื่อนไขหยุดอัตโนมัติถ้า health เพี้ยน
ด่าน SAFE → ทำต่อ · BLOCKED → หยุด รายงานชั้นที่ติด

[Ledger — ผูกเข้าความจำ · ใหม่]
ทุกการกระทำชั้น 3A เขียน append-only ที่ `.hermes/ledger/<branch>.md` (Schema §13): เวลา / issue_id / คำสั่ง / ด่านที่รัน / exit code / commit SHA / ผล · redact secret §7
ตอน Close จะยก ledger เข้า session log · ไม่มี ledger ทั้งที่ทำชั้น 3A = งานนั้นนับเป็น claimed

[นิยาม "ว้าว"] เหนือคาด / ประสบการณ์ใหม่ / สิ่งที่ไม่เคยเจอ — แต่ใช้เลือกวิธีที่ดีสุด ภายใน scope เท่านั้น · ห้ามใช้เป็นเหตุขยาย scope · งานนอกเป้า → backlog เสนอ ไม่ทำเอง

[นิยาม "ผ่าน 100%"] = ทุก issue ในเฟส verified จริง (test/lint/manual ผ่าน + หลักฐาน · ค้น gate เอง Schema §5) · % เฟส = verified÷ทั้งหมด · วัดเลขไม่ได้ใช้ PASS/BLOCKED · ทำไม่ได้จริง = แยก blocker ห้ามนับรวม 100%

วิธีทำ:
1. อ่าน context (ขั้น 0 ด้านบน: plan/memory/token/claim/เป้าหมายรอบนี้)
2. แตกเฟส (อ้าง id เดิม: เป้าหมาย/issue/วิธีตรวจ/เงื่อนไขผ่าน/output)
3. ทำทีละเฟส จบและตรวจก่อนข้าม · fail → แก้จนผ่านหรือระบุ blocker
4. บันทึกเหตุผล+trade-off ทุกครั้งที่ตัดสินใจแทนผู้ใช้ · เรื่องสำคัญที่ไม่รู้จริง = หยุดถาม · รายละเอียดเล็ก = เลือกเอง+บันทึก

[STOP/เปลี่ยนวิธี] ผู้ตรวจคนเดิมและวิธีเดิมไม่ผ่านครบ 2 รอบในปัญหาเดียว → ห้ามเรียกรอบที่ 3 · แยกปัญหาแล้วใช้ test/lint/build/gate-run หรือผู้ตรวจคนละค่ายทันที · worktree dirty เฉพาะไฟล์ที่จะแตะ/มีงานคนอื่นเสี่ยงทับ · จะแตะชั้น 3B หรือ prod ที่ ALLOW_AUTO_PROD=OFF · ด่าน 3A ตอบ BLOCKED · error class เดิม 3 ครั้ง · เจอ secret ใน diff · ต้องเดาเรื่องสำคัญเพราะ context ไม่พอ → หยุด รายงาน ถาม 1 คำถาม

ส่งงานจบ (closeout):
| phase_id | เป้าหมาย | % หรือสถานะ | หลักฐาน |
+ สิ่งที่ทำ / ไฟล์ที่แก้ / คำสั่งตรวจ (ค้น gate เอง) / เรื่องที่เลือกแทนผู้ใช้ / ledger รอบนี้ / ความเสี่ยงค้าง / next step เดียวที่แนะนำสุด
(ช่อง % ใส่ BLOCKED/NOT VERIFIED ได้ถ้าวัดเลขไม่ได้)
จบแล้วส่งต่อ Use Close Chat เพื่อ verify รวบ + เขียน memory + ออก token

ข้อห้าม: merge/deploy prod เองตอน ALLOW_AUTO_PROD=OFF · ตีความ SAFE เอง · ทำชั้น 3A โดยไม่เขียน ledger · เขียน secret ลง ledger · แตะ worktree คนอื่น · ลุยต่อทั้งที่ token รอบก่อน = NEED_OWNER_ACTION
```

## Legacy Names

ชื่อเก่า `Go to Sleep`, `Sleep Mode`, `เข้าโหมดนอน`, และ `โหมดนอน` เป็น alias เพื่อความเข้ากันได้ย้อนหลังเท่านั้น ชื่อหลักที่ต้องใช้และแสดงให้ผู้ใช้เห็นคือ `Use Continue`

## Worktree Lifecycle v1

อ่าน `worktree-lifecycle-contract.md` ก่อนใช้ Prompt นี้ · กลับงานเดิมด้วย `task_id` + `hermes worktree status/enter`; ห้ามเดา cwd หรือเปิด branch ใหม่ · writer ไม่ตรง/lease หายให้หยุด `WTL_BLOCKED` หรือทำ handoff ก่อน

## Changelog

- v4.5 (2026-07-14): เพิ่ม Two-Zone Execution + Phase Write Permit ตามคำสั่งเจ้าของ · Zone A ทำต่อเองจน verified 100% · Zone B รวมขออนุมัติครั้งเดียวต่อ Phase · เลิกถามราย issue ภายใน scope เดิม
- v4.4 (2026-07-13): ผู้ตรวจและวิธีเดิมไม่ผ่าน 2 รอบต้องหยุด เปลี่ยนเป็นการตรวจด้วยเครื่องมือจริงหรือผู้ตรวจคนละค่าย ห้ามวนรอบที่ 3
- v4.3 (2026-07-12): รับ Write Permit ต่อหนึ่งงานจาก Use New Chat ป้องกัน Use Continue นำ branch เดิมไปใช้กับคำสั่งใหม่จนไฟล์ทับกัน
- v4.2 (2026-07-08): เพิ่มกฎ re-anchor (หลังตอบคำถามแทรก ต้องเปิด plan.md ทวนเฟส+ข้อห้ามก่อนลงมือ) + เลขงานต้องขึ้นต้นด้วย plan_id ของแผน · จากการสอบสวนแผน GRD 2026-07-07 (ต้นตอ: AI ลืมแผนหลังตอบคำถาม + เลขงานชนกันข้ามแผน)
- v4.1 (2026-07-05): เกาะ Memory Schema v1.2 — ขั้น 0 อ่าน `.project/plan.md` + `.project/OverviewProgress.md` + `.project/decisions.md` (เดิมอ่าน `.hermes/`+handoff) · เจอไฟล์เก่า = Migration §1b ก่อนลุย · ledger ยังอยู่ `.hermes/ledger/` (ไฟล์เครื่องจักร) · schema ไม่ตรง = ห้ามเขียนความจำ (คำสั่งเจ้าของ 2026-07-05)
- v4.0 (2026-06-26): เกาะ Memory Schema v1.1 · **เปลี่ยนนโยบาย: merge→main/deploy prod/migration prod ออกจาก auto → ต้องขอคน (ALLOW_AUTO_PROD=OFF ค่าตั้งต้น ตาม §12)** · เพิ่มขั้น 0 อ่าน plan/memory/token/claim ก่อนลงมือ · เคารพ claim/worktree (ไม่แตะของคนอื่น) · ledger ผูกเข้า memory (§13) · ค้น gate เอง (§5) · อ้าง phase_id/issue_id เดิม · ส่งต่อ Use Close Chat ตอนจบ · redact secret (§7)
- v3.1 (2026-06-24): งานเสี่ยงสูงเป็น auto ผ่านด่าน (ชั้น 3A) · fail-closed + ผลจากเครื่องมือจริง + ผูกกับคำสั่งจริง · เพิ่มนิยาม "ว้าว" + นิยาม 100% + STOP rule + closeout
- v3.0 (2026-06-07): เปลี่ยนชื่อจาก Go to Sleep เป็น Use Continue

## Graph Links

- Parent hub: [[skills/prompt-shortcuts/Prompt Shortcuts|Prompt Shortcuts]]
- Registry: [[ai-context/prompt-shortcut-registry|Prompt Shortcut Registry]]
- Schema: [[skills/prompt-shortcuts/references/memory-schema|Memory Schema v1.1]]
- Prev in lifecycle: [[skills/prompt-shortcuts/references/use-comply|Use Comply]]
- Next in lifecycle: [[skills/prompt-shortcuts/references/use-close-chat|Use Close Chat]]
