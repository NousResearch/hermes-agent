---
title: Use WOW Resource
aliases:
  - Use WOW Resource
  - use-wow-resource
  - WOW Resource
  - wow-resource
  - ใช้ WOW Resource
  - ใช้ WOW
  - WOW Layout
  - WOW Menu
  - WOW Script
  - WOW Design
  - WOW Web Engine
tags:
  - prompt-shortcuts
  - wow-system
  - web-design-intelligence
  - resource-selection
status: active
version: 1.1
updated: 2026-06-24
---

# Use WOW Resource

## Invocation

ใช้ shortcut นี้เมื่อเจ้าของงานขอแนวทางดีไซน์เว็บระดับพรีเมียม layout/UI/UX/animation/script/menu/hover/module/Web Engine/เว็บภาครัฐ/เว็บองค์กร/pitch-ready/award-ready ผ่านวลีสั้น เช่น:

```text
Use WOW Resource กับโจทย์นี้
ใช้ WOW Resource กับโจทย์นี้
ใช้ WOW กับโจทย์นี้
WOW Layout: ...
WOW Menu: ...
WOW Script: ...
WOW Design: ...
WOW Web Engine: ...
```

## Core Contract

คุณไม่ใช่คนก๊อปสคริปต์ · คุณคือคนคัดทรัพยากรและแปลเป็นทิศทางดีไซน์เฉพาะโปรเจกต์

เมื่อ shortcut นี้ถูกเรียก ต้อง:

1. เข้าใจเป้าหมายโปรเจกต์ กลุ่มผู้ใช้ การรับรู้แบรนด์ น้ำหนักเนื้อหา technical stack และเป้าส่งมอบ
2. อ่านเฉพาะแหล่ง WOW System + Web Design Intelligence ที่เกี่ยวข้องจาก Obsidian (ไม่โหลดทั้ง vault)
3. คัด layout/design/interaction/animation/script/component/module ที่เหมาะ
4. ปฏิเสธตัวที่ generic, ไม่เข้าโจทย์, flashy เกิน, ไม่ accessible, อ่อนสำหรับงาน enterprise/ภาครัฐ, หรือไม่คุ้ม implementation cost
5. แปลงเป็น concept เฉพาะโปรเจกต์ · ห้ามก๊อป script/layout/visual ตรง ๆ
6. อธิบายว่าเลือกอะไร ปฏิเสธอะไร เพราะอะไร
7. จบงานทำ usage log ให้ WOW/WDI เรียนรู้ว่า resource ไหนช่วย/พลาด/ควรกัก
8. ถ้าต้องแก้โค้ด/ไฟล์ ทำ Selection Brief ก่อน เว้นแต่เจ้าของสั่งทำเลย

## Path Policy (พกพาข้ามเครื่อง)

ทุก path ใช้ตัวแปร `$HERMES_OBSIDIAN_ROOT` เป็นราก:
- Mac = `~/ObsidianVault/HermesAgent` · VPS = `/home/linux-nat/ObsidianVault/HermesAgent`
- ค่าทั้งสองนี้คือ "ค่าที่ต้องตั้งในตัวแปร" ไม่ใช่ fallback ให้เดาเอง
- ถ้าตัวแปรไม่ถูกตั้ง หรือหาไฟล์ไม่เจอ → รายงานว่า "ต้องตั้งค่า root ก่อน" แล้วตอบจากโจทย์แบบ unverified ได้ แต่ห้ามอ้างว่าอ่านคลังแล้ว

## Required Source Routing

เริ่มจากชุดเล็กที่สุดที่เป็นประโยชน์ · อ่านเสมอ:

- `$HERMES_OBSIDIAN_ROOT/skills/wow-design-master/SKILL.md`
- `$HERMES_OBSIDIAN_ROOT/knowledge/web-design-intelligence/ai-use-index.md`
- `$HERMES_OBSIDIAN_ROOT/knowledge/web-design-intelligence/retrieval-rules.md`

แล้วเลือกเพิ่มตามงาน (path ต่อท้าย `$HERMES_OBSIDIAN_ROOT/`):

| Task signal | Add these sources |
|---|---|
| Web Engine, theme, layout-lang, SiteSection | `skills/wow-design-master/references/WOW_WEB_ENGINE_BRIDGE.md`, `knowledge/web-design-intelligence/modern-stack-map.md` |
| layout, homepage, information-heavy site | `skills/wow-design-master/references/WOW_LAYOUT_PATTERN_LIBRARY.md` |
| style, premium look, non-generic | `skills/wow-design-master/references/WOW_STYLE_INTELLIGENCE.md` |
| government, public service, award-ready | `skills/wow-design-master/references/WOW_GLOBAL_AWARD_SYSTEM.md` |
| TOR, bid, pitch, committee | `skills/wow-design-master/references/WOW_PITCHING_SYSTEM.md` |
| audit, old site, AI-looking UI | `skills/wow-design-master/references/WOW_SITE_AUDIT_SYSTEM.md` |
| menu, tab, accordion, nav, hover | `knowledge/web-design-intelligence/script-bank/accordion-tab-menu.md` + project-local menu docs |
| animation, scroll, motion, hover | `knowledge/web-design-intelligence/script-bank/animation-motion.md` |
| chart, table, data display | `knowledge/web-design-intelligence/modern-stack-map.md` + WDI chart/table notes |
| carousel, gallery, media, SVG, video | search WDI script-bank หมวดที่ตรง |

ถ้างานอยู่ใน repo ให้อ่าน adapter ของโปรเจกต์ก่อนถ้ามี: `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `QWEN.md`, `.cursorrules`, `.cursor/rules/*.mdc`, `.hermes/context.md`

## เกณฑ์คะแนนคัด Resource

ให้คะแนนแต่ละ resource 1-5 ต่อเกณฑ์ (สูง = ดี):
- fit กับโจทย์ / accessibility / ความน่าเชื่อถือ enterprise / reusability
- implementation cost: **กลับด้าน — ต้นทุนต่ำ = คะแนนสูง**
เลือกตัวคะแนนรวมสูงสุด + เหตุผล 1 บรรทัดต่อ resource · ผูกกับ status: verified-ready > adapt-with-plan > เลี่ยง quarantine/retired
แสดงเป็นตารางสั้น 3-5 resource เท่านั้น (กัน brief ยาวเกิน)

## Output: Selection Brief

ก่อน implement ส่ง brief นี้ (ภาษาไทย เว้นแต่เจ้าของใช้ภาษาอื่น):

```text
WOW Resource Selection

โจทย์ที่เข้าใจ:
เป้าหมายธุรกิจ/หน่วยงาน:
ผู้ใช้หลัก:
Brand Perception:
Source Access Status: (อ่านแหล่งไหนได้/ไม่ได้ — read / failed / missing / permission-denied)
Obsidian sources read:
Resource ที่เลือก (+ คะแนนรวม): 
Resource ที่ไม่เลือก:
เหตุผลการเลือก:
แนวทางดัดแปลง:
ความเสี่ยง:
จุดที่จะนำไปใช้:
ขั้นต่อไป:
```

กระชับไว้ · เป้าคือคุณภาพการตัดสินใจ ไม่ใช่รายงานยาว
ถ้าอ่านคลัง WOW/WDI ไม่สำเร็จ ห้ามเรียกผลว่า "Resource ที่เลือก" ให้ใช้ "แนวทางชั่วคราวจากโจทย์" และ status = unverified เท่านั้น

## Output: Usage Log After Work

จบงานแนบ log นี้ · ถ้าเจ้าของสั่งให้เขียนลง Obsidian และกติกาโปรเจกต์อนุญาต ให้ append เดือนปัจจุบันที่ `$HERMES_OBSIDIAN_ROOT/knowledge/web-design-intelligence/usage-log/`

```json
{
  "date": "YYYY-MM-DD",
  "project": "project name",
  "task": "short task name",
  "resources_used": ["resource-id"],
  "resources_rejected": ["resource-id"],
  "outcome": "planned | implemented | rejected | reworked",
  "human_rating": null,
  "rework_needed": false,
  "error": null,
  "notes": "why these resources helped or failed"
}
```

ถ้า resource ทำให้ผลออกมา generic/ผิด/ไม่ accessible/ช้า/ต้องรื้อ ให้แนะนำ `quarantine` เป็นภาษาไทย + เหตุผล

## Implementation Rules

เมื่ออนุมัติให้ implement:
- ใช้ resource เป็นแรงบันดาลใจ+logic ไม่ใช่วางผลก๊อป
- ปรับ pattern ให้เข้ากับเนื้อหา/IA/แบรนด์/accessibility/stack ของโปรเจกต์
- เลือก stack ทันสมัยที่เข้ากับ repo (React, Vue, Node, Tailwind, GSAP, Framer Motion, Radix/shadcn, TanStack, ECharts, Swiper/Embla ฯลฯ) · เลี่ยง jQuery plugin เว้นแต่ legacy หรือสั่งชัด
- ปฏิเสธของ generic (underline-only nav, plain dropdown, opacity fade, card grid ไร้กลยุทธ์, random hover scale, animation ไร้คุณค่า, ก๊อป CodePen)
- งาน frontend ตรวจ behavior ถ้าทำได้: screenshot, responsive, accessibility พื้นฐาน, ไม่มี text overlap, ไม่มีปุ่มตาย
- ใช้ PDCA registry ถ้ามี: `$HERMES_OBSIDIAN_ROOT/knowledge/web-design-intelligence/registry/resource-registry.json` · เลือก verified-ready ก่อน แล้ว adapt-with-plan · เลี่ยง quarantine/retired เว้นใช้เป็นตัวอย่างเชิงลบ

## Decision Rule (แยก 2 กรณีให้ชัด)

- **ข้อมูลโจทย์ไม่พอ** (เช่นไม่รู้กลุ่มผู้ใช้) → สมมติได้ + ติดป้าย assumption + ถาม 1 คำถามเฉพาะเมื่อคำตอบเปลี่ยน resource ที่เลือก
- **อ่านคลัง WOW/WDI ไม่สำเร็จ** (ไฟล์หาย/เปิดไม่ได้/ตัวแปร root ไม่ตั้ง) → ห้ามเดาว่าอ่านแล้ว · ต้องรายงานว่าอ่านแหล่งไหนไม่ได้ + ตอบจากโจทย์เท่าที่มี + มาร์ค unverified

## Minimal User Examples

```text
ใช้ WOW Resource กับโจทย์นี้:
ออกแบบ Mouse Hover Menu สำหรับเว็บกรมระดับประเทศ มีข้อมูลเยอะ
```

```text
WOW Layout:
เว็บบริษัท B2B ราคาแพง ต้องดูน่าเชื่อถือ ไม่ใช่ landing page ทั่วไป
```

## Changelog

- v1.1 (2026-06-24): ผ่านตรวจ 2 AI (Claude+Codex) · เปลี่ยน path ฝังเครื่องเป็นตัวแปร $HERMES_OBSIDIAN_ROOT (พนักงาน/VPS ใช้ได้) · แก้ Decision Rule แยก "โจทย์ไม่พอ" กับ "อ่านคลังไม่ได้" (กันเดาว่าอ่านแล้ว) · เพิ่มเกณฑ์คะแนนคัด resource 1-5 (ต้นทุนต่ำ=คะแนนสูง) · เพิ่ม Source Access Status ใน brief · อ่านคลังไม่ได้ห้ามเรียก "resource ที่เลือก"
- v1.0 (2026-05-31): เวอร์ชันแรก

## Graph Links

- Parent hub: [[skills/prompt-shortcuts/Prompt Shortcuts|Prompt Shortcuts]]
- Registry: [[ai-context/prompt-shortcut-registry|Prompt Shortcut Registry]]
```
