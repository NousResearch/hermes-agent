---
title: Use OverviewProgress
aliases:
  - Use OverviewProgress
  - use-overviewprogress
  - Use Overview Progress
  - OverviewProgress
  - ใช้ OverviewProgress
  - สร้างไฟล์ภาพรวมงาน
  - อัปเดตภาพรวม project
  - ภาพรวมความคืบหน้า
tags:
  - prompt-shortcuts
  - overview-progress
  - project-os
  - project-memory
  - standard-file
status: active
version: "1.1"
created: 2026-06-28
updated: 2026-07-05
schema: memory-schema-v1.2
pairs_with: use-new-chat >= 1.8, use-close-chat >= 2.2
---

# Use OverviewProgress (v1.1 · 2026-07-05)

ไฟล์ภาพรวม + ความคืบหน้ามาตรฐานต่อ project — ไฟล์ที่ 2/4 ของชุด Project OS
เป็นแกนกลางกัน AI ลืมภาพรวมงานข้ามแชท · อ่านตอนเปิดแชท (New Chat) · อัปตอนปิดแชท (Close Chat)
ตาม Memory Schema v1.2 ไฟล์นี้คือ **แหล่งจริงของสถานะโปรเจกต์** และยุบ `handoff.md` + `active.md` เดิมเข้ามาไว้ที่เดียว

## Shortcut

```text
Use OverviewProgress
```

## Prompt

```text
Use OverviewProgress

เป้าหมาย: ทำให้ทุก project มีไฟล์ภาพรวมเดียวที่บอก "project นี้ทำอะไร / ถึงไหน /
ค้างอะไร / เสร็จอะไร" เพื่อให้ AI ตัวไหนเปิดแชทมาก็เข้าใจงานทันที ไม่ต้องเริ่มจากศูนย์
คู่กับ Use New Chat >= v1.8 (อ่าน) + Use Close Chat >= v2.2 (เขียน) · อ้าง Memory Schema v1.2 (§1c โครงบังคับ)

[กฎพื้นฐาน]
- ผู้ใช้พิมพ์ไทย ตอบไทย · ศัพท์เทคนิคแปลเป็นภาษาคนทันที
- ทุกบรรทัดสถานะติดป้าย [fact] (ตรวจของจริงแล้ว) / [assumption] (ยังเดา) / [ยังไม่มีข้อมูล]
- reality-overrides-token (Schema §2): เชื่อ git/CI/SHA ของจริง มากกว่าข้อความที่เขียนไว้รอบก่อน
- redact secret ก่อนเขียนทุกครั้ง

[พกพาได้ทุก project — บังคับ]
- หา root เองด้วย `git rev-parse --show-toplevel` · ห้ามฝัง path/IP/ชื่อเครื่อง
- ที่เก็บมาตรฐาน: `<ราก project>/.project/OverviewProgress.md`
- [ด่านไฟล์เข้า git จริง — บังคับหลังสร้าง/แก้] รัน `git check-ignore -v .project/OverviewProgress.md`
  (ต้องไม่เจอ) + `git ls-files .project/` (ต้องเห็นไฟล์) · ถ้าโดนซ่อนโดย .gitignore →
  เจาะช่องอนุญาตเพิ่มบรรทัด `!.project/` และ `!.project/**` แล้วตรวจซ้ำ ·
  ไฟล์ไม่เข้า git = ความจำหายข้ามเครื่องแน่นอน = งานยังไม่เสร็จ ห้ามรายงานว่าเสร็จ

[ขั้นที่ 1 — หาไฟล์]
ดูว่ามี .project/OverviewProgress.md ไหม · เจอไฟล์ความจำเก่า (.hermes/active.md /
handoff.md ที่ root) → ทำ Migration ตาม Schema §1b (ยุบเข้าไฟล์นี้ + ทำ stub ห้ามลบ)

[ขั้นที่ 2A — ยังไม่มี → สร้าง]
1. สแกน project อ่านของจริง: README, โครงโฟลเดอร์, AGENTS.md/CLAUDE.md,
   ไฟล์ความจำเก่าตาม §1b, `git log -10`, branch, PR/CI
2. เขียนตามเทมเพลตด้านล่าง (โครงบังคับ §1c) · งานล่าสุดถึงไหนให้อิง git จริง ไม่เชื่อ handoff อย่างเดียว
3. ข้อมูลไม่พอ → [assumption]/[ยังไม่มีข้อมูล] ไม่แต่งเติม

[ขั้นที่ 2B — มีอยู่แล้ว → อัปเดต]
1. เช็กป้าย `> memory-schema:` บรรทัดแรก (ไม่ตรง v1.2 = อัปโครงก่อน) แล้วอ่านไฟล์เดิม
2. เทียบกับ git จริง (commit/branch/SHA ล่าสุด) + งานในแชทรอบนี้
3. อัป 4 หัวข้อบนสุดก่อนเสมอ · ย้ายงานเป็น "เสร็จ" เฉพาะที่ verified จริง (หลักฐาน Schema §4)
4. ประวัติยาวลงล่างเท่านั้น ห้ามดันสถานะจม

[ขั้นที่ 3 — รายงาน]
สรุปภาษาคนสั้น ๆ ว่าเปลี่ยนอะไร (ก่อน→หลัง) + แนบ Evidence (timestamp/host/cwd/commands)

═══════════ เทมเพลต OverviewProgress.md (โครงบังคับ Schema §1c) ═══════════
> memory-schema: v1.2
> อ่านตามลำดับ: plan.md → decisions.md → <ไฟล์ source of truth ของ project>

# Overview & Progress — <ชื่อ project>
อัปเดตล่าสุด: <วันที่> · commit ล่าสุด: <SHA> · branch: <ชื่อ> · ป้าย: [fact]/[assumption]

## สถานะล่าสุด
<process ขั้นไหน / SHA ล่าสุด / gate ล่าสุดผ่านไหม>  [fact]

## งานถัดไป
1. <งานถัดไป 1-3 ข้อ เรียงลำดับ — แชทใหม่เริ่มข้อแรกได้ทันที>

## ข้อห้าม/กติกาล็อก
- <สิ่งที่ AI ห้ามทำในโปรเจกต์นี้>

## งานค้าง/ส่งต่อ
- <claimed + งานรอเจ้าของ + ใครถือต่อ>

## project นี้คืออะไร (2-3 บรรทัด)
<ทำอะไร ใครใช้ คุณค่าหลัก>  [fact/assumption]

## เสร็จแล้ว (verified) + ประวัติ
- <งาน> — หลักฐาน: <test/SHA/curl>  [fact]

ข้อห้าม: ย้ายงานขึ้น "เสร็จแล้ว" โดยไม่มีหลักฐาน · เชื่อข้อความเก่าโดยไม่เทียบ git · เขียน secret · ปิดงานโดยไม่ผ่านด่านไฟล์เข้า git จริง · สรุปด้วยศัพท์เทคนิคโดยไม่แปล
```

## Worktree Lifecycle v1

อ่าน `worktree-lifecycle-contract.md` ก่อนใช้ Prompt นี้ · OverviewProgress ต้องเก็บ task/machine/worktree/branch/lifecycle/writer/cleanup และอ่านไฟล์แผน tracking ก่อนเสมอ เพื่อให้แชทถัดไปกลับพื้นที่เดิมได้

## Changelog

- v1.1 (2026-07-05): **กู้คืนจากการถูกถอนกลับ** (commit `fff10805b` ถอน `f22b6f3bd` เมื่อ 2026-06-28 โดยไม่มีบันทึกเหตุผล — ไฟล์หายจากระบบ 7 วันโดยความจำยังจดว่า "ครบ 4/4") · อัปเข้า Memory Schema v1.2: เทมเพลตใช้โครงบังคับ §1c (ป้ายเวอร์ชัน + สารบัญ + 4 หัวข้อบนสุด) · ยุบ handoff/active เดิมเข้าไฟล์นี้ · เพิ่มด่าน "ไฟล์เข้า git จริง" (บทเรียน .gitignore ซ่อน .md จากอีก project 2026-07-05)
- v1.0 (2026-06-28): ไฟล์ที่ 2/4 ของ Project OS · ภาพรวม+ความคืบหน้าต่อ project · อ่านตอน New Chat / เขียนตอน Close Chat · อิง git จริง (reality-overrides-token) · ป้าย [fact]/[assumption] · พกพาได้ทุก project

## Graph Links

- Parent hub: [[skills/prompt-shortcuts/Prompt Shortcuts|Prompt Shortcuts]]
- Registry: [[ai-context/prompt-shortcut-registry|Prompt Shortcut Registry]]
- Schema: [[skills/prompt-shortcuts/references/memory-schema|Memory Schema v1.2]]
- Pair: [[skills/prompt-shortcuts/references/use-new-chat|Use New Chat]] · [[skills/prompt-shortcuts/references/use-close-chat|Use Close Chat]]
- Sibling: [[skills/prompt-shortcuts/references/use-businessplan|Use BusinessPlan]] · [[skills/prompt-shortcuts/references/use-featurespec|Use FeatureSpec]] · [[skills/prompt-shortcuts/references/use-designsystem|Use DesignSystem]]
