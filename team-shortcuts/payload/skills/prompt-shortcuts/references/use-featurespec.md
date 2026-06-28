---
title: Use FeatureSpec
aliases:
  - Use FeatureSpec
  - use-featurespec
  - Use Feature Spec
  - FeatureSpec
  - ใช้ FeatureSpec
  - สแกนฟีเจอร์ project
  - บัญชีฟีเจอร์
  - อัปเดตรายการฟีเจอร์
tags:
  - prompt-shortcuts
  - feature-spec
  - project-os
  - project-memory
  - standard-file
status: active
version: "1.0"
created: 2026-06-28
updated: 2026-06-28
schema: memory-schema-v1.1
pairs_with: use-close-chat >= 2.2
related: use-scan-feature
---

# Use FeatureSpec (v1.0 · 2026-06-28)

ไฟล์บัญชีฟีเจอร์มาตรฐานต่อ project — ไฟล์ที่ 3/4 ของชุด Project OS
ให้ AI สแกนของจริงแล้วลงบัญชีว่า project นี้มีฟีเจอร์อะไร · อันไหนของจริง อันไหนยังหลอก (mock)

> ใช้ `Use Scan Feature` (ที่มีอยู่) เป็น "เครื่องสแกน" เมื่อต้องการสแกนลึกเฟสต่อเฟส · ตัวนี้คือไฟล์มาตรฐานที่เก็บผลให้ AI ทุกตัวอ่าน

## Shortcut

```text
Use FeatureSpec
```

## Prompt

```text
Use FeatureSpec

เป้าหมาย: ทำให้ทุก project มีบัญชีฟีเจอร์มาตรฐานที่ตรงกับโค้ดจริง เพื่อให้ AI รู้ว่า
มีอะไรทำแล้ว อะไรยังเป็นของหลอก (mock) อะไรยังไม่ได้ทำ · กัน AI เดาว่าฟีเจอร์มี/ไม่มี
คู่กับ Use Close Chat >= v2.2 · อ้าง Memory Schema v1.1

[กฎพื้นฐาน]
- ผู้ใช้พิมพ์ไทย ตอบไทย · ศัพท์เทคนิคแปลทันที
- ทุกฟีเจอร์ต้องมีหลักฐานจากโค้ดจริง (path:line) · ไม่มีหลักฐาน = ห้ามลงว่า "ของจริง"
- สถานะฟีเจอร์ 3 แบบ: real (ทำจริงใช้ได้) / mock (มีโครงแต่ยังหลอก/ค่าตายตัว) / planned (ยังไม่ทำ)

[พกพาได้ทุก project — บังคับ]
- หา root เองด้วย git · ห้ามฝัง path/IP
- ที่เก็บมาตรฐาน: `<ราก project>/.project/FeatureSpec.md`

[ขั้นที่ 1 — หาไฟล์]
ดูว่ามี .project/FeatureSpec.md ไหม

[ขั้นที่ 2A — ยังไม่มี → สร้าง]
1. สแกนโค้ดจริง: route/endpoint, หน้าจอ/หน้าเว็บ, command, ปลั๊กอิน, service, job
2. แต่ละฟีเจอร์ระบุ: ชื่อ · สถานะ (real/mock/planned) · ที่อยู่ในโค้ด (path:line) · หลักฐาน
3. ของที่ดูเหมือนทำงานแต่ค่าตายตัว/TODO/return ปลอม → ลงเป็น mock ไม่ใช่ real
4. ไม่ชัด → planned หรือ [ตรวจไม่ได้] ไม่เดาว่า real

[ขั้นที่ 2B — มีอยู่แล้ว → อัปเดต]
1. อ่านไฟล์เดิม
2. เทียบกับโค้ดปัจจุบัน: ฟีเจอร์ใหม่เพิ่มแถว · ฟีเจอร์ที่เลื่อนจาก mock→real ต้องมีหลักฐาน
3. ฟีเจอร์ที่ถูกลบออกจากโค้ด → มาร์คว่า removed ไม่ลบประวัติทิ้งเฉยๆ

[ขั้นที่ 3 — รายงาน]
สรุปสั้น: ฟีเจอร์ทั้งหมด N ตัว · real X / mock Y / planned Z + Evidence

═══════════ เทมเพลต FeatureSpec.md ═══════════
# Feature Spec — <ชื่อ project>
อัปเดตล่าสุด: <วันที่> · สแกนโดย: <AI/คน> · สรุป: real X / mock Y / planned Z

## บัญชีฟีเจอร์
| ฟีเจอร์ | สถานะ | ที่อยู่ในโค้ด (path:line) | หลักฐาน |
|---|---|---|---|
| <ชื่อ> | real/mock/planned | <path:line> | <test/curl/อ่านโค้ด> |

## Reality Matrix (สรุปของจริง vs ของหลอก)
- ของจริงใช้ได้เลย: <รายการ>
- ของหลอก/ยังไม่ครบ: <รายการ> + ขาดอะไร
- ยังไม่ได้ทำ: <รายการ>

ข้อห้าม: ลงฟีเจอร์เป็น real โดยไม่มี path:line + หลักฐาน · เดาว่าทำงานจากชื่อฟังก์ชัน · ลบประวัติฟีเจอร์เดิมทิ้ง · สรุปด้วยศัพท์เทคนิคโดยไม่แปล
```

## Changelog

- v1.0 (2026-06-28): ไฟล์ที่ 3/4 ของ Project OS · บัญชีฟีเจอร์ที่ตรงโค้ดจริง · สถานะ real/mock/planned + หลักฐาน path:line · Reality Matrix · ใช้ Use Scan Feature เป็นเครื่องสแกนลึก · พกพาได้ทุก project

## Graph Links

- Parent hub: [[skills/prompt-shortcuts/Prompt Shortcuts|Prompt Shortcuts]]
- Registry: [[ai-context/prompt-shortcut-registry|Prompt Shortcut Registry]]
- Engine: [[skills/prompt-shortcuts/references/use-scan-feature|Use Scan Feature]]
- Pair: [[skills/prompt-shortcuts/references/use-close-chat|Use Close Chat]]
- Sibling: [[skills/prompt-shortcuts/references/use-businessplan|Use BusinessPlan]] · [[skills/prompt-shortcuts/references/use-overviewprogress|Use OverviewProgress]]
