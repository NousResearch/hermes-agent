---
title: Use BusinessPlan
aliases:
  - Use BusinessPlan
  - use-businessplan
  - Use BusinessPlan File
  - Use Project BusinessPlan
  - BusinessPlan File
  - ใช้ BusinessPlan
  - สร้างไฟล์แผนธุรกิจ
  - สแกนแผนธุรกิจ project
  - อัปเดตแผนธุรกิจ project
tags:
  - prompt-shortcuts
  - business-plan
  - project-os
  - project-memory
  - standard-file
status: active
version: "1.0"
created: 2026-06-28
updated: 2026-06-28
schema: memory-schema-v1.1
pairs_with: use-close-chat >= 2.1
uses_engine:
  - use-business-plan
  - use-saas-opus-master-prompt
---

# Use BusinessPlan (v1.0 · 2026-06-28)

ไฟล์แผนธุรกิจมาตรฐานต่อ project — เป็น 1 ใน 4 ไฟล์ของชุด Project OS
(BusinessPlan · OverviewProgress · FeatureSpec · DesignSystem)

> ต่างจาก `Use Business Plan` (เว้นวรรค) ที่เป็น "เครื่องวางแผนธุรกิจ" รับโจทย์ดิบมาวิเคราะห์ออกแผน
> `Use BusinessPlan` (ติดกัน) ตัวนี้ = สร้าง/อัปเดต "ไฟล์ความจำธุรกิจ" ฝังในแต่ละ project ให้ AI ทุกตัวอ่านเข้าใจธุรกิจของ project นั้นทันที · มันเรียกเครื่องวางแผนข้างต้นมาเป็น "เครื่องยนต์" ไม่เขียนตรรกะวางแผนซ้ำ

## Shortcut

```text
Use BusinessPlan
```

## Prompt

```text
Use BusinessPlan

เป้าหมาย: ทำให้ทุก project มีไฟล์แผนธุรกิจมาตรฐานที่ AI อ่านแล้วเข้าใจธุรกิจของ
project นั้นทันที · ใช้ได้กับทุก project ทั้งบน VPS และเครื่องโน้ตบุ๊ก
คู่กับ Use Close Chat >= v2.1 · อ้าง Memory Schema v1.1 · เช็ก schema version ตอนเริ่ม ไม่ตรง = เตือน

[กฎพื้นฐาน]
- ผู้ใช้พิมพ์ไทย ตอบไทย · ศัพท์เทคนิคแปลเป็นภาษาคนทันที
- เจ้าของงานอาจไม่ใช่ dev → อธิบายภาษาคน · ห้ามถามว่า "ใช้เครื่องมือตัวไหน" ค้นเอง
- ทุกบรรทัดในไฟล์ติดป้าย [fact] (ตรวจของจริงแล้ว) หรือ [assumption] (ยังเดา) หรือ [ยังไม่มีข้อมูล]
- ห้ามเขียนไฟล์ทับของจริงจนกว่าเจ้าของอนุมัติ (ยกเว้นโหมดอัปเดตย่อยที่ Use Close Chat เรียก)
- redact secret ก่อนเขียนทุกครั้ง (Schema §7)

[พกพาได้ทุก project — บังคับ]
- หาตำแหน่ง project เองด้วย `git rev-parse --show-toplevel` · ห้ามฝัง path/IP/ชื่อเครื่องตายตัว
- ที่เก็บมาตรฐาน: `<ราก project>/.project/BusinessPlan.md` (สั้น) + `<ราก project>/.project/BusinessPlan-Full.md` (ลึก)
- ไม่ใช่ git repo → ใช้โฟลเดอร์ปัจจุบันเป็นราก + รายงานว่าไม่ใช่ repo
- ถ้าโปรเจกต์ยังไม่มีโฟลเดอร์ .project/ → สร้างโฟลเดอร์นี้ (ที่เดียว ไม่วางไฟล์กระจายที่ราก)

[ขั้นที่ 1 — หาไฟล์]
ดูว่ามี .project/BusinessPlan.md และ .project/BusinessPlan-Full.md ไหม

[ขั้นที่ 2A — ยังไม่มี → สร้าง]
1. สแกน project อ่านของจริง: README, โค้ดหลัก, หน้าเว็บ/หน้าจอ, เอกสารธุรกิจ/การตลาด,
   .hermes/context.md /active.md /decisions.md, AGENTS.md/CLAUDE.md ถ้ามี
2. เรียก "เครื่องวางแผนธุรกิจ" ที่มีอยู่ (Use Business Plan · ถ้าเป็น SaaS ใช้โหมด SaaS)
   มาวิเคราะห์เต็ม → ผลเต็มลง BusinessPlan-Full.md (ตามเทมเพลตเต็มด้านล่าง)
3. กลั่นเหลือแก่นสั้นลง BusinessPlan.md (ตามเทมเพลตสั้นด้านล่าง) — คุมให้ ~1 หน้า
   เหตุผล: ไฟล์ที่ยาวเกิน AI จะไม่อ่าน (อ่าน/ทำตามได้จริงราว 150-200 บรรทัด)
4. ข้อมูลไม่พอ → ใส่ [assumption] หรือ [ยังไม่มีข้อมูล] + รวมเป็น "คำถามค้างถึงเจ้าของ"
   ห้ามแต่งเรื่องเติมให้ดูครบ
5. project ที่ไม่มีมุมธุรกิจ (เช่น เครื่องมือภายใน) → สร้างโครงเปล่า + คำถาม ไม่ยัดเรื่องแต่ง

[ขั้นที่ 2B — มีอยู่แล้ว → อัปเดต]
1. อ่าน 2 ไฟล์เดิม
2. เทียบกับของจริงใน project ตอนนี้ (โค้ด/หน้าเว็บ/เอกสารที่เปลี่ยนไป)
3. อัปเฉพาะที่เปลี่ยน · ขยับป้าย [assumption] → [fact] เมื่อพิสูจน์ได้จริง
4. ของที่ขัดกับความจริงปัจจุบัน → แก้ + โน้ตสั้นว่าแก้เพราะอะไร
5. ห้ามเขียนทับ log ประวัติเดิมใน Full → ต่อท้ายเท่านั้น

[ขั้นที่ 3 — รายงานเจ้าของ]
สรุปภาษาคนสั้น ๆ ว่า สร้าง/อัปเดตอะไรบ้าง (ก่อน→หลัง) + คำถามค้าง + ขออนุมัติก่อนเขียนจริง
แนบ Evidence ท้ายรายงาน: timestamp / host / cwd / commands ที่รัน

═══════════ เทมเพลต BusinessPlan.md (สั้น · อ่านทุกครั้ง) ═══════════
# Business Plan — <ชื่อ project>
อัปเดตล่าสุด: <วันที่> · ตรวจกับของจริงโดย: <AI/คน> · ป้าย: [fact]/[assumption]/[ยังไม่มีข้อมูล]

## โมเดลธุรกิจ 1 บรรทัด
<ขายอะไร เก็บเงินยังไง>  [fact/assumption]

## ลูกค้า (ใครใช้ / ใครจ่าย / ใครอนุมัติ)
- ...

## ปัญหาที่แก้ + ความต้องการที่ยังไม่มีใครตอบ (unmet need)
- ...

## คุณค่าที่วัดได้ (ตัวเลขจริง เช่น ลดเวลา 80%, ต้นทุน -30%)
- ...

## ราคา / รายได้
- ...

## คู่แข่ง + จุดที่เราชนะ (moat = กำแพงกันลอก)
- ...

## กลยุทธ์การตลาดย่อ
- ...

## ความเสี่ยง + แผนสำรอง (Plan B)
- ...

## สิ่งที่ยังไม่รู้ (คำถามค้างถึงเจ้าของ project)
- ...

> วิเคราะห์เต็มอยู่ที่ .project/BusinessPlan-Full.md

═══════════ เทมเพลต BusinessPlan-Full.md (ลึก · อัปตอนปิดงาน) ═══════════
# Business Plan (ฉบับเต็ม) — <ชื่อ project>
อัปเดตล่าสุด: <วันที่>

## ผลวิเคราะห์เต็ม (จากเครื่องวางแผน)
<360 องศา · persona · customer journey · คู่แข่ง · กลยุทธ์ชนะ · pitch · pricing · ความเสี่ยง>

## ประวัติการเปลี่ยนแผน (ต่อท้ายเรื่อย ๆ ห้ามลบของเก่า)
- <วันที่> · <แชต/งานที่ทำ> · เปลี่ยนอะไร · กระทบแผนตรงไหน

═══════════ ผูกกับ Use Close Chat (ขั้น Business Plan Sync) ═══════════
ตอนปิดแชท ถ้ามี .project/BusinessPlan.md ให้ถามตัวเอง:
"รอบนี้มี feature ใหม่ / ราคาเปลี่ยน / เจอ insight ลูกค้าใหม่ / คู่แข่งเปลี่ยน ที่ขยับแผนธุรกิจไหม"
- ไม่มี → เขียนใน Report ว่า "รอบนี้ไม่มีการเปลี่ยนด้านธุรกิจ"
- มี → อัป [fact] ใน BusinessPlan.md + ต่อท้าย log ใน BusinessPlan-Full.md
แล้วออก Report:
  สรุปการอัปเดตแผนธุรกิจรอบนี้
  - BusinessPlan.md: <เปลี่ยนอะไร ก่อน→หลัง หรือ "ไม่เปลี่ยน">
  - BusinessPlan-Full.md: <เพิ่ม log อะไร หรือ "ไม่เพิ่ม">
  - คำถามค้างถึงเจ้าของ: <ถ้ามี>

ข้อห้าม: เขียนไฟล์ทับโดยไม่อนุมัติ · แต่งข้อมูลธุรกิจที่ไม่มีหลักฐาน · ทำไฟล์สั้นให้ยาวจน AI ไม่อ่าน · บังคับอัปเดตทั้งที่รอบนั้นไม่มีอะไรกระทบธุรกิจ · สรุปด้วยศัพท์เทคนิคโดยไม่แปล
```

## Changelog

- v1.0 (2026-06-28): ตัวแรกของชุด Project OS · ไฟล์แผนธุรกิจมาตรฐาน 2 ชั้น (สั้น/ลึก) · เช็ค→สร้าง/อัปเดต · พกพาได้ทุก project (หา root จาก git ไม่ฝัง path) · เรียกเครื่องวางแผนเดิม (Use Business Plan / SaaS Opus) เป็นเครื่องยนต์ · ผูก Business Plan Sync เข้า Use Close Chat v2.1 · แยกชื่อชัดจาก `Use Business Plan` (เครื่องวางแผน)

## Graph Links

- Parent hub: [[skills/prompt-shortcuts/Prompt Shortcuts|Prompt Shortcuts]]
- Registry: [[ai-context/prompt-shortcut-registry|Prompt Shortcut Registry]]
- Schema: [[skills/prompt-shortcuts/references/memory-schema|Memory Schema v1.1]]
- Engine: [[skills/prompt-shortcuts/references/use-business-plan|Use Business Plan]] · [[skills/prompt-shortcuts/references/use-saas-opus-master-prompt|Use SaaS Opus Master Prompt]]
- Pair: [[skills/prompt-shortcuts/references/use-close-chat|Use Close Chat]]
