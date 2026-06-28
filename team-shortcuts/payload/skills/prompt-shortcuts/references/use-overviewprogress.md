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
version: "1.0"
created: 2026-06-28
updated: 2026-06-28
schema: memory-schema-v1.1
pairs_with: use-new-chat >= 1.8, use-close-chat >= 2.2
---

# Use OverviewProgress (v1.0 · 2026-06-28)

ไฟล์ภาพรวม + ความคืบหน้ามาตรฐานต่อ project — ไฟล์ที่ 2/4 ของชุด Project OS
เป็นแกนกลางกัน AI ลืมภาพรวมงานข้ามแชท · อ่านตอนเปิดแชท (New Chat) · อัปตอนปิดแชท (Close Chat)

## Shortcut

```text
Use OverviewProgress
```

## Prompt

```text
Use OverviewProgress

เป้าหมาย: ทำให้ทุก project มีไฟล์ภาพรวมเดียวที่บอก "project นี้ทำอะไร / ถึงไหน /
ค้างอะไร / เสร็จอะไร" เพื่อให้ AI ตัวไหนเปิดแชทมาก็เข้าใจงานทันที ไม่ต้องเริ่มจากศูนย์
คู่กับ Use New Chat >= v1.8 (อ่าน) + Use Close Chat >= v2.2 (เขียน) · อ้าง Memory Schema v1.1

[กฎพื้นฐาน]
- ผู้ใช้พิมพ์ไทย ตอบไทย · ศัพท์เทคนิคแปลเป็นภาษาคนทันที
- ทุกบรรทัดสถานะติดป้าย [fact] (ตรวจของจริงแล้ว) / [assumption] (ยังเดา) / [ยังไม่มีข้อมูล]
- reality-overrides-token (Schema §2): เชื่อ git/CI/SHA ของจริง มากกว่าข้อความที่เขียนไว้รอบก่อน
- redact secret ก่อนเขียนทุกครั้ง

[พกพาได้ทุก project — บังคับ]
- หา root เองด้วย `git rev-parse --show-toplevel` · ห้ามฝัง path/IP/ชื่อเครื่อง
- ที่เก็บมาตรฐาน: `<ราก project>/.project/OverviewProgress.md`

[ขั้นที่ 1 — หาไฟล์]
ดูว่ามี .project/OverviewProgress.md ไหม

[ขั้นที่ 2A — ยังไม่มี → สร้าง]
1. สแกน project อ่านของจริง: README, โครงโฟลเดอร์, AGENTS.md/CLAUDE.md,
   .hermes/context.md /active.md /handoff.md /decisions.md, `git log -10`, branch, PR/CI
2. เขียนภาพรวมตามเทมเพลตด้านล่าง · งานล่าสุดถึงไหนให้อิง git จริง ไม่เชื่อ handoff อย่างเดียว
3. ข้อมูลไม่พอ → [assumption]/[ยังไม่มีข้อมูล] ไม่แต่งเติม

[ขั้นที่ 2B — มีอยู่แล้ว → อัปเดต]
1. อ่านไฟล์เดิม
2. เทียบกับ git จริง (commit/branch/SHA ล่าสุด) + งานในแชทรอบนี้
3. ย้ายงานจาก "กำลังทำ/ค้าง" → "เสร็จแล้ว" เฉพาะที่ verified จริง (มีหลักฐาน Schema §4)
4. อัปบรรทัด "อัปเดตล่าสุด" + SHA

[ขั้นที่ 3 — รายงาน]
สรุปภาษาคนสั้น ๆ ว่าเปลี่ยนอะไร (ก่อน→หลัง) + แนบ Evidence (timestamp/host/cwd/commands)

═══════════ เทมเพลต OverviewProgress.md ═══════════
# Overview & Progress — <ชื่อ project>
อัปเดตล่าสุด: <วันที่> · commit ล่าสุด: <SHA> · branch: <ชื่อ> · ป้าย: [fact]/[assumption]

## project นี้คืออะไร (2-3 บรรทัด)
<ทำอะไร ใครใช้ คุณค่าหลัก>  [fact/assumption]

## สถานะรวม
<ขั้นไหนของงาน เช่น กำลังพัฒนา / ใช้งานจริง / พักไว้>  [fact]

## เสร็จแล้ว (verified)
- <งาน> — หลักฐาน: <test/SHA/curl>  [fact]

## กำลังทำ
- <งาน> — ใคร/branch ไหน

## ค้าง / ทำต่อ (เรียงตามลำดับ)
1. <งานถัดไป 1 ข้อที่ชัดที่สุด>

## ความเสี่ยง / สิ่งที่ยังไม่รู้
- ...

ข้อห้าม: ย้ายงานขึ้น "เสร็จแล้ว" โดยไม่มีหลักฐาน · เชื่อข้อความเก่าโดยไม่เทียบ git · เขียน secret · สรุปด้วยศัพท์เทคนิคโดยไม่แปล
```

## Changelog

- v1.0 (2026-06-28): ไฟล์ที่ 2/4 ของ Project OS · ภาพรวม+ความคืบหน้าต่อ project · อ่านตอน New Chat / เขียนตอน Close Chat · อิง git จริง (reality-overrides-token) · ป้าย [fact]/[assumption] · พกพาได้ทุก project

## Graph Links

- Parent hub: [[skills/prompt-shortcuts/Prompt Shortcuts|Prompt Shortcuts]]
- Registry: [[ai-context/prompt-shortcut-registry|Prompt Shortcut Registry]]
- Schema: [[skills/prompt-shortcuts/references/memory-schema|Memory Schema v1.1]]
- Pair: [[skills/prompt-shortcuts/references/use-new-chat|Use New Chat]] · [[skills/prompt-shortcuts/references/use-close-chat|Use Close Chat]]
- Sibling: [[skills/prompt-shortcuts/references/use-businessplan|Use BusinessPlan]]
