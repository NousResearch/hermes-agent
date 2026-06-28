---
title: Use DesignSystem
aliases:
  - Use DesignSystem
  - use-designsystem
  - Use Design System File
  - DesignSystem
  - ใช้ DesignSystem
  - สร้างไฟล์ Design System
  - อัปเดต Design System project
  - ตรวจดีไซน์ตามมาตรฐาน
tags:
  - prompt-shortcuts
  - design-system
  - project-os
  - project-memory
  - standard-file
status: active
version: "1.0"
created: 2026-06-28
updated: 2026-06-28
schema: memory-schema-v1.1
related: use-create-design-system
---

# Use DesignSystem (v1.0 · 2026-06-28)

ไฟล์ระบบดีไซน์มาตรฐานต่อ project — ไฟล์ที่ 4/4 ของชุด Project OS
รวม CI (อัตลักษณ์แบรนด์ · สี/ฟอนต์/ระยะ) ของ project ไว้ที่เดียว ให้ AI อ่านแล้ว
ตรวจ + ปรับหน้าจอของ project ให้ตรงระบบดีไซน์ที่ตกลงไว้

> ต่างจาก `Use Create Design System` (เครื่องสร้างระบบดีไซน์ใหม่ทั้งชุด 5 เฟส)
> ตัวนี้ = ไฟล์สรุประบบดีไซน์ประจำ project + ตัวช่วยตรวจว่าโค้ดตรงดีไซน์ไหม · เรียก Use Create Design System เมื่อต้องออกแบบใหม่

## Shortcut

```text
Use DesignSystem
```

## Prompt

```text
Use DesignSystem

เป้าหมาย: ทำให้ทุก project มีไฟล์ระบบดีไซน์มาตรฐานที่ AI อ่านแล้วรู้ว่า project นี้
ใช้สี/ฟอนต์/ระยะ/ส่วนประกอบอะไร แล้วใช้ตรวจ+ปรับโค้ดให้ตรงดีไซน์ · กัน AI คิดสีเอง
คู่กับมาตรฐานกลาง 60-Design/design-systems/design-system-standard.md · อ้าง Memory Schema v1.1

[กฎพื้นฐาน]
- ผู้ใช้พิมพ์ไทย ตอบไทย · ศัพท์เทคนิคแปลทันที
- ทุกค่าดีไซน์ต้องมาจากของจริงใน project (ไฟล์ธีม/token/CSS/tailwind config) · ไม่มี = [ยังไม่มีข้อมูล] ไม่คิดสีเอง
- ค่าที่ตรวจจากโค้ดจริงติดป้าย [fact] · ค่าที่เสนอ/ยังไม่ยืนยันติด [assumption]

[พกพาได้ทุก project — บังคับ]
- หา root เองด้วย git · ห้ามฝัง path/IP
- ที่เก็บมาตรฐาน: `<ราก project>/.project/DesignSystem.md`
- ถ้ามีไฟล์ token เครื่องอ่านได้ (เช่น tokens.json แบบ W3C DTCG) ให้ชี้ไปไฟล์นั้น เพื่อให้ตรวจค่าตรงเป๊ะได้ (ไม่ใช่ดูด้วยตา)

[ขั้นที่ 1 — หาไฟล์ + อ่านแหล่งดีไซน์]
1. ดูว่ามี .project/DesignSystem.md ไหม
2. อ่านแหล่งดีไซน์จริงใน project: ไฟล์ธีม (เช่น themes/*.css), tailwind/uno config,
   token json, ตัวแปรสี OKLCH/HEX, ฟอนต์, design doc · + มาตรฐานกลางถ้าอ้างถึง

[ขั้นที่ 2A — ยังไม่มี → สร้าง]
1. สรุปค่าจริงที่เจอลงเทมเพลตด้านล่าง (สี/ฟอนต์/ระยะ/ส่วนประกอบ/โหมดสว่าง-มืด)
2. เทียบกับมาตรฐานกลาง (DTCG 3-tier, WCAG 2.2 AA) → ระบุช่องที่ขาด/ไม่ผ่าน contrast
3. ไม่มีระบบดีไซน์ในโค้ด → สร้างโครงเปล่า + เสนอให้เรียก Use Create Design System

[ขั้นที่ 2B — มีอยู่แล้ว → อัปเดต + ตรวจ]
1. อ่านไฟล์เดิม + เทียบกับโค้ดดีไซน์ปัจจุบัน
2. โหมดตรวจ: หา "จุดที่โค้ดไม่ตรงระบบดีไซน์" (เช่น สี #hex ลอย ๆ ที่ไม่ใช่ token,
   contrast ต่ำกว่า AA, ฟอนต์นอกชุด) → ลงรายการ + path:line
3. เสนอวิธีแก้ให้ตรงดีไซน์ (ไม่แก้เองจนกว่าเจ้าของสั่ง · งาน UI ต้องตรวจภาพก่อน claim)

[ขั้นที่ 3 — รายงาน]
สรุป: ค่าหลักของระบบดีไซน์ + จุดที่โค้ดไม่ตรง N จุด + Evidence

═══════════ เทมเพลต DesignSystem.md ═══════════
# Design System — <ชื่อ project>
อัปเดตล่าสุด: <วันที่> · แหล่งจริง: <ไฟล์ธีม/token ที่อ้าง> · มาตรฐานกลาง: design-system-standard.md

## CI / สีแบรนด์
| บทบาท | ค่า (token/HEX/OKLCH) | ที่มา (path) | ป้าย |
|---|---|---|---|
| primary | ... | ... | [fact/assumption] |

## ตัวอักษร (typography)
- ฟอนต์หลัก/รอง · ขนาด · น้ำหนัก  [fact]

## ระยะ + เลย์เอาต์ (spacing/grid)
- สเกลระยะ · radius · เงา  [fact]

## ส่วนประกอบหลัก (components)
- ปุ่ม / ฟอร์ม / การ์ด / ตาราง · สถานะ (ปกติ/ชี้/กด/ปิด)  [fact]

## โหมดสว่าง-มืด + การเข้าถึง (a11y)
- รองรับ dark mode ไหม · ผล contrast WCAG 2.2 AA  [fact]

## ไฟล์ token เครื่องอ่านได้ (ถ้ามี)
- <path ไป tokens.json> — ใช้ตรวจค่าตรงเป๊ะ

## จุดที่โค้ดยังไม่ตรงระบบดีไซน์ (อัปทุกครั้งที่ตรวจ)
- <path:line> — ปัญหา + วิธีแก้

ข้อห้าม: คิดสี/ฟอนต์ใหม่เองที่ไม่ได้มาจากโค้ด/เจ้าของ · claim ว่าหน้าจอตรงดีไซน์โดยไม่ตรวจภาพ · แก้ธีมกลางโดยไม่ถาม (เทียบ INC-9085 in-place vs showcase) · สรุปด้วยศัพท์เทคนิคโดยไม่แปล
```

## Changelog

- v1.0 (2026-06-28): ไฟล์ที่ 4/4 ของ Project OS · สรุประบบดีไซน์/CI ประจำ project จากของจริงในโค้ด + โหมดตรวจหาจุดที่โค้ดไม่ตรงดีไซน์ · ผูกมาตรฐานกลาง design-system-standard.md + DTCG token · เรียก Use Create Design System เมื่อต้องออกแบบใหม่ · พกพาได้ทุก project

## Graph Links

- Parent hub: [[skills/prompt-shortcuts/Prompt Shortcuts|Prompt Shortcuts]]
- Registry: [[ai-context/prompt-shortcut-registry|Prompt Shortcut Registry]]
- Engine: [[skills/prompt-shortcuts/references/use-create-design-system|Use Create Design System]]
- Standard: [[60-Design/design-systems/design-system-standard|design-system-standard]]
- Sibling: [[skills/prompt-shortcuts/references/use-businessplan|Use BusinessPlan]] · [[skills/prompt-shortcuts/references/use-overviewprogress|Use OverviewProgress]] · [[skills/prompt-shortcuts/references/use-featurespec|Use FeatureSpec]]
