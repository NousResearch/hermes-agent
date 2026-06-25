---
title: Use Impeccable
aliases:
  - use-impeccable
  - Impeccable
  - ใช้ Impeccable
  - ตรวจ UI Slop
  - แก้ AI Slop
tags:
  - prompt-shortcuts
  - design-quality
  - ui-review
status: active
version: 1.2
created: 2026-06-01
updated: 2026-06-24
related:
  - "[[50-Playbooks/ai-design-team-without-ui-execution-bottleneck|AI Design Team Without UI Execution Bottleneck]]"
  - "[[00-Center/docs/Synerry_Design_System|Synerry Design System]]"
---

# Use Impeccable

Shortcut หลัก: `Use Impeccable` · Aliases: `use-impeccable`, `Impeccable`, `ใช้ Impeccable`, `ตรวจ UI Slop`, `แก้ AI Slop`

## Purpose

ตรวจและปรับคุณภาพ UI เพื่อลดงานที่ดูเป็น AI slop · AI จัดการขั้นตอนเอง เจ้าของงานไม่ต้องเลือก shortcut ย่อย

## Decision Tree (ตัดสินก่อนทำ — กันเพิ่ม dependency มั่ว)

1. มี target ชัดไหม (project/หน้า/path/component) · ไม่ชัด → ถามสั้น ๆ ว่าตรวจอะไร
2. เป็นงาน frontend/UI ไหม · ไม่ใช่ → ไม่ติดตั้งอะไร บอกว่าไม่เกี่ยว
3. repo มี Impeccable แล้วไหม · มี → ใช้เลย
4. ยังไม่มี → ติดตั้งได้อย่างปลอดภัยไหม (ดู Install Safety) · ปลอดภัย+อนุมัติ → ติดตั้ง · ไม่ → ทำ manual UI audit ตาม rubric แทน
5. ถ้า `npx impeccable detect` fail / ไม่ใช่ Node frontend / ไม่มี package.json → ทำ manual audit

## Install Safety (กันแตะ dependency โดยไม่ควร)

"ติดตั้งปลอดภัย" = มี `package.json` + lockfile ตรง package manager จริง + ไม่ใช่ production freeze + ไม่มี policy ห้ามเพิ่ม dependency
- ติดตั้งเป็น devDependency เท่านั้น · ต้องขออนุมัติเจ้าของงานก่อนเพิ่ม dependency เสมอ (ไม่ติดตั้งเงียบ)
- เช็ก branch/worktree dirty ก่อน
- แม้ใช้ `npx` แบบชั่วคราว ก็ต้องขออนุมัติก่อน เพราะมีการดาวน์โหลด/รัน package ภายนอก (network/exec risk)

## Workflow

1. ระบุเป้าหมายจากบริบทล่าสุด (project/path/URL/หน้า admin/component)
2. รันตรวจที่เหมาะ (`npx impeccable detect` หรือเจาะ path) ตาม Decision Tree
3. งาน frontend/admin/dashboard/landing/product/component → โหลดกรอบ design ร่วม (ดู Design System Source)
4. แปลผลเป็นภาษาคน: ปัญหาจริงคืออะไร / กระทบ UX-credibility-production ยังไง / แก้ทันทีจุดไหน / เป็นหนี้ UI จุดไหน
5. ถ้าสั่งให้แก้ → แก้แบบ scoped: ไม่เปลี่ยน business logic เกินจำเป็น / ไม่สร้าง design ใหม่มั่ว / ไม่ refactor ใหญ่เกินโจทย์ / รักษา design system เดิม
6. ตรวจผลหลังแก้ด้วยหลักฐานจริง (command output / screenshot / browser check)
7. สรุปภาษาไทยให้เจ้าของงานตัดสินใจได้

## Design System Source (ลำดับค้นหา)

repo-local design system ก่อน → ถ้าไม่มี ใช้ `$HERMES_OBSIDIAN_ROOT/00-Center/docs/Synerry_Design_System` + `$HERMES_OBSIDIAN_ROOT/50-Playbooks/ai-design-team-without-ui-execution-bottleneck`
- หาไม่เจอทั้งคู่ → บอกว่าไม่พบ ทำ audit จากหลักการทั่วไป + มาร์ค unverified
- มี design system → ถือเป็นกรอบหลัก · AI ห้ามคิดสี/spacing/component ใหม่เองโดยไม่มีเหตุผล

## Decision Rules

- `AI slop` = UI ดูเหมือน generate ทั่วไป (gradient ม่วง-ฟ้าซ้ำ, card หนาเกิน, side accent bar, glassmorphism เกิน, hierarchy แบน, spacing ไม่จริง, motion jank, image แตก/placeholder)
- `Blocking issue` = route/login/nav/form/responsive/accessibility พัง, text overlap, layout broken → **แก้ก่อน UI debt เสมอ**
- `UI debt` = หน้าตา/ประสบการณ์ที่ควรแก้ แต่ระบบยังไม่พัง
- ผล Impeccable ขัดบริบทจริง → ใช้วิจารณญาณ ไม่แก้ตามเครื่องมือแบบตาบอด
- เว็บราชการ/enterprise/SaaS/admin → UI นิ่ง อ่านง่าย ทำงานจริง ไม่แต่งแบบ landing page
- AI = visual execution + cleanup · เจ้าของงาน/lead = taste gate (ตัดสินว่าดี/เหมาะ brand/เหมาะผู้ใช้)

## Scoring Rubric (100 คะแนน — แทน "80%" ลอย ๆ)

| หมวด | เต็ม |
|---|--:|
| layout / hierarchy | 25 |
| responsive | 20 |
| accessibility | 20 |
| design-system adherence | 20 |
| credibility / brand fit | 15 |

- ให้คะแนนเฉพาะหมวดที่มีหลักฐาน (screenshot/หน้าที่ตรวจจริง) · หมวดที่ตรวจไม่ได้ = mark `unverified` ไม่ใช่ให้ 0
- ต่ำกว่า 80 (จากหมวดที่ตรวจได้) = ห้ามสรุปว่าใช้จริงพร้อม production แม้ command จะผ่าน

## Output Standard (บังคับรูปแบบ กัน AI แต่งคะแนน)

1. สถานะ: ตรวจแล้ว / ติดตั้งแล้ว / แก้แล้ว / รอเป้าหมาย
2. คะแนนรวม + คะแนนรายหมวด + หลักฐานแต่ละหมวด (หมวดที่ตรวจไม่ได้ = unverified)
3. รายการ Blocking issue
4. รายการ UI debt (เก็บไว้แก้ทีหลัง)
5. สิ่งที่ตรวจไม่ได้ + เหตุผล
6. สิ่งที่แก้ไป + หลักฐานหลังแก้
7. งานค้าง/ความเสี่ยงที่เหลือ
ห้ามตอบด้วยคำสั่งเครื่องมืออย่างเดียว (เช่น `detect passed`) โดยไม่แปลผลกระทบ

## Owner Preference

ใช้คำเดียว `Use Impeccable` · ไม่ต้องมี shortcut ย่อย (Audit/Polish/Harden/Gate) · AI เลือกวิธีทำเอง

## Changelog

- v1.2 (2026-06-24): ผ่านตรวจ 2 AI (Claude+Codex) · เพิ่ม Decision Tree + manual audit fallback เมื่อเครื่องมือใช้ไม่ได้/ไม่ใช่ frontend · Install Safety (devDependency + ขออนุมัติก่อนเพิ่ม dependency · npx ก็ต้องขออนุมัติ) · Scoring Rubric 100 คะแนนแทน "80%" ลอย ๆ (หมวดตรวจไม่ได้ = unverified) · Design System ค้น repo-local ก่อน แล้ว $HERMES_OBSIDIAN_ROOT · บังคับรูปแบบ output กันแต่งคะแนน
- v1.1 (2026-06-06): เวอร์ชันก่อนหน้า

## Graph Links

- Parent hub: [[skills/prompt-shortcuts/Prompt Shortcuts|Prompt Shortcuts]]
- Registry: [[ai-context/prompt-shortcut-registry|Prompt Shortcut Registry]]
```
