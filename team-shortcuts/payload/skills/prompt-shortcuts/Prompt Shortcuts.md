---
title: Prompt Shortcuts
aliases:
  - prompt-shortcuts
  - Prompt Library
  - Standard Prompts
  - Use Act-As
  - use-act-as
  - Use Act As
  - Act-As
  - act-as
  - Use Comply
  - use-comply
  - Comply
  - comply
  - Use Summary
  - use-summary
  - Summary
  - summary
  - Use Scan Feature
  - use-scan-feature
  - Scan Feature
  - scan-feature
  - Use Opus Plan
  - use-opus-plan
  - Opus Plan
  - opus-plan
  - Use Business Plan
  - use-business-plan
  - Business Plan
  - business-plan
  - Use Viber Structure
  - use-viber-structure
  - Viber Structure
  - viber-structure
  - Use Viber Audit
  - use-viber-audit
  - Viber Audit
  - viber-audit
  - Viber Enterprise Standard
  - Use WOW Resource
  - Use Blog Auto
  - use-blog-auto
  - Blog Auto
  - blog-auto
  - ใช้ Blog Auto
  - เขียนบล็อกอัตโนมัติ
  - ทำบล็อกจากงานนี้
  - ส่งเข้า Hi Logic Labs
  - use-wow-resource
  - WOW Resource
  - wow-resource
  - WOW Layout
  - WOW Menu
  - WOW Script
  - WOW Design
  - WOW Web Engine
  - Use Flow Guardian
  - use-flow-guardian
  - Flow Guardian
  - Safe Flow
  - New Chat Gate
  - Use New Chat
  - use-new-chat
  - Start New Chat
  - New Chat Startup
  - Initialize Hermes Agent chat
  - เริ่ม New Chat
  - เปิด New Chat
  - เริ่มแชทใหม่
  - เปิดแชทใหม่
  - Use Save Git
  - use-save-git
  - Save Git
  - save-git
  - ใช้ Save Git
  - เซฟ Git
  - ก่อน push
  - ก่อน merge
  - ก่อน deploy
  - Git Safe Flow
  - GitLab Deploy Safe Flow
  - Use Ship Gate
  - Use Continue
  - use-continue
  - Continue
  - continue
  - ทำต่อ
  - ทำต่อเอง
  - ทำงานต่อ
  - ไม่ต้องรอผม
  - Use Move Folder
  - use-move-folder
  - Move Folder
  - move-folder
  - movefolder
  - ใช้ Move Folder
  - ย้ายโฟลเดอร์
  - จัดเรียง Folder
  - จัดเรียงโฟลเดอร์
  - Go to Sleep
  - go-to-sleep
  - Sleep Mode
  - sleep-mode
  - Review Chat
  - review-chat
  - Chat Review
  - chat-review
tags:
  - skills
  - knowledge
  - prompt-shortcuts
  - hermesnous
status: active
source_of_truth: skills/prompt-shortcuts
runtime_path: ~/.codex/skills/prompt-shortcuts
registry: ai-context/prompt-shortcut-registry
updated: 2026-06-22
---

# Prompt Shortcuts

ศูนย์รวม shortcut สำหรับเรียก prompt มาตรฐานที่ใช้บ่อย โดยไม่ต้อง copy/paste prompt ยาวทุกครั้ง

> [!important]
> Registry กลางสำหรับ AI ทุกตัวอยู่ที่ [[ai-context/prompt-shortcut-registry|Prompt Shortcut Registry]]
> Prompt ตัวจริงอยู่ใน `references/` และเป็น source of truth

## Operating Rule

ชุดนี้ทำตาม [[10-Knowledge/Knowledge Operating Rules|Knowledge Operating Rules]]

## ใช้งานกับ Codex

เรียก shortcut หลัก:

```text
Use Act-As
Use Comply
Use Summary
Use Scan Feature
Use Opus Plan
Use AI Pair
Use Business Plan
Use Viber Structure
Use Viber Audit
Use Blog Auto
Use WOW Resource
Use Flow Guardian
Use New Chat
Use Save Git
Use Continue
Use Move Folder
Review Chat
```

หรือแบบชัดเจน:

```text
ใช้ $prompt-shortcuts Use Act-As กับงานนี้
ใช้ $prompt-shortcuts Use Comply กับงานนี้
ใช้ $prompt-shortcuts Use Summary กับ link + content นี้
ใช้ $prompt-shortcuts Use Scan Feature กับ repo นี้
ใช้ $prompt-shortcuts Use Opus Plan กับงานนี้
ใช้ $prompt-shortcuts Use AI Pair กับงานนี้
ใช้ $prompt-shortcuts Use Business Plan กับโจทย์ธุรกิจ / การตลาด / pitch / เว็บไซต์นี้
ใช้ $prompt-shortcuts Use Viber Structure กับโปรเจกต์ Viber Code นี้
ใช้ $prompt-shortcuts Use Viber Audit กับโปรเจกต์นี้ หรือทุกโปรเจกต์ใน Viber Project
ใช้ $prompt-shortcuts Use Impeccable กับงาน UI นี้
ใช้ $prompt-shortcuts Use Blog Auto เพื่อสรุปองค์ความรู้จากงานนี้ไปเป็น draft บล็อก Hi Logic Labs และส่งต่อ Content Factory แบบรออนุมัติ
ใช้ $prompt-shortcuts Use WOW Resource กับโจทย์ layout / design / script / Web Engine นี้
ใช้ $prompt-shortcuts Use Flow Guardian เพื่อบังคับ worktree/branch gate, no-write audit, approval, verify, tracking, และ handoff
ใช้ $prompt-shortcuts Use New Chat เพื่อเริ่มแชทใหม่แบบตรวจ project/worktree/branch/dirty/VPS ก่อนตอบ
ใช้ $prompt-shortcuts Use Save Git ก่อน push / merge / deploy เพื่อตรวจ Git, worktree, dirty files, GitLab main, VPS SHA, health endpoint และคืน SAFE/STOP decision
ใช้ $prompt-shortcuts Use Continue กับงานนี้
ใช้ $prompt-shortcuts Use Move Folder กับงานจัดเรียงโฟลเดอร์/cleanup บน VPS โดยอ่าน registry จริงใน `/home/linux-nat/.codex/use-move-folder/project-registry` ก่อนทำงาน
ใช้ $prompt-shortcuts Review Chat กับแชทนี้
```

## Shortcut Map

| Shortcut | Aliases | Prompt |
| --- | --- | --- |
| `Use Act-As` | `use-act-as`, `Use Act As`, `Act-As`, `act-as`, `ใช้ Act-As`, `กำหนดบทบาท`, `เรียกทีมผู้เชี่ยวชาญ` | [[skills/prompt-shortcuts/references/use-act-as|use-act-as]] |
| `Use Comply` | `use-comply`, `Comply`, `comply`, `ใช้ Comply`, `ทำ Comply`, `แตกเฟส`, `ทำตารางเปอร์เซ็นต์` | [[skills/prompt-shortcuts/references/use-comply|use-comply]] |
| `Use Summary` | `use-summary`, `Summary`, `summary`, `ใช้ Summary`, `สรุป`, `สรุปลิงก์`, `วิเคราะห์บทความ`, `สรุปข้อมูล` | [[skills/prompt-shortcuts/references/use-summary|use-summary]] |
| `Use Scan Feature` | `use-scan-feature`, `Scan Feature`, `scan-feature`, `สแกนฟีเจอร์`, `ตรวจฟีเจอร์`, `บัญชีฟีเจอร์` | [[skills/prompt-shortcuts/references/use-scan-feature|use-scan-feature]] |
| `Use AI Pair` | `use-ai-pair`, `AI Pair`, `ai-pair`, `Use Pair AI`, `Pair AI`, `pair-ai`, `ใช้ AI Pair`, `ใช้ Pair AI`, `จับคู่ AI เขียนตรวจ` | [[skills/prompt-shortcuts/references/use-ai-pair|use-ai-pair]] |
| `Use Business Plan` | `use-business-plan`, `Business Plan`, `business-plan`, `ใช้ Business Plan`, `รีวิวโจทย์ธุรกิจ`, `วางแผนธุรกิจ`, `วางแผนการตลาด`, `วางแผน Pitch`, `งานประมูล` | [[skills/prompt-shortcuts/references/use-business-plan|use-business-plan]] |
| `Use Viber Structure` | `use-viber-structure`, `Viber Structure`, `viber-structure`, `ใช้ Viber Structure`, `โครงสร้าง Viber`, `วางโครงสร้าง Viber Code`, `วางแผน Viber Code`, `Vibe Code Enterprise` | [[skills/prompt-shortcuts/references/use-viber-structure|use-viber-structure]] |
| `Use Viber Audit` | `use-viber-audit`, `Viber Audit`, `viber-audit`, `Use Viber Standard Audit`, `Use Viber Compliance`, `ใช้ Viber Audit`, `ตรวจ Viber Standard`, `ตรวจ Viber Enterprise`, `ตรวจมาตรฐาน Viber`, `Viber Enterprise Standard` | [[skills/prompt-shortcuts/references/use-viber-audit|use-viber-audit]] |
| `Use Impeccable` | `use-impeccable`, `Impeccable`, `ใช้ Impeccable`, `ตรวจ UI Slop`, `แก้ AI Slop` | [[skills/prompt-shortcuts/references/use-impeccable|use-impeccable]] |
| `Use Blog Auto` | `use-blog-auto`, `Blog Auto`, `blog-auto`, `ใช้ Blog Auto`, `เขียนบล็อกอัตโนมัติ`, `ทำบล็อกจากงานนี้`, `ส่งเข้า Hi Logic Labs` | [[skills/prompt-shortcuts/references/use-blog-auto|use-blog-auto]] |
| `Use WOW Resource` | `use-wow-resource`, `WOW Resource`, `wow-resource`, `ใช้ WOW Resource`, `ใช้ WOW`, `WOW Layout`, `WOW Menu`, `WOW Script`, `WOW Design`, `WOW Web Engine` | [[skills/prompt-shortcuts/references/use-wow-resource|use-wow-resource]] |
| `Use Flow Guardian` | `use-flow-guardian`, `Flow Guardian`, `Safe Flow`, `New Chat Gate`, `ใช้ Flow Guardian`, `ใช้ Safe Flow`, `เปิด Flow Guardian`, `ตรวจ worktree`, `กัน AI แก้งานทับกัน` | [[skills/prompt-shortcuts/references/use-flow-guardian|use-flow-guardian]] |
| `Use New Chat` | `use-new-chat`, `Start New Chat`, `New Chat Startup`, `Initialize Hermes Agent chat`, `เริ่ม New Chat`, `เปิด New Chat`, `เริ่มแชทใหม่`, `เปิดแชทใหม่` | [[skills/prompt-shortcuts/references/use-new-chat|use-new-chat]] |
| `Use Save Git` | `use-save-git`, `Save Git`, `save-git`, `ใช้ Save Git`, `เซฟ Git`, `ก่อน push`, `ก่อน merge`, `ก่อน deploy`, `Git Safe Flow`, `GitLab Deploy Safe Flow`, `Use GitLab Deploy Safe Flow`, `Use Ship Gate` | [[skills/prompt-shortcuts/references/use-save-git|use-save-git]] |
| `Use Continue` | `use-continue`, `Continue`, `continue`, `ทำต่อ`, `ทำต่อเอง`, `ทำงานต่อ`, `ทำต่ออัตโนมัติ`, `ไม่ต้องรอผม`, legacy: `Go to Sleep`, `go-to-sleep`, `Sleep Mode`, `sleep-mode`, `เข้าโหมดนอน`, `โหมดนอน` | [[skills/prompt-shortcuts/references/use-continue|use-continue]] |
| `Use Move Folder` | `use-move-folder`, `Move Folder`, `move-folder`, `movefolder`, `ใช้ Move Folder`, `ย้ายโฟลเดอร์`, `จัดเรียง Folder`, `จัดเรียงโฟลเดอร์` | [[skills/prompt-shortcuts/references/use-move-folder|use-move-folder]] |
| `Review Chat` | `review-chat`, `Chat Review`, `chat-review`, `รีวิวแชท`, `ตรวจแชท`, `สรุปส่งต่อ`, `สรุปเปิดแชทใหม่` | [[skills/prompt-shortcuts/references/review-chat|review-chat]] |

## ใช้งานกับ AI ตัวอื่น

Claude Code, Gemini, Qwen, Cursor และ AI ตัวอื่นให้เริ่มจาก [[ai-context/prompt-shortcut-registry|Prompt Shortcut Registry]] แล้วเปิด prompt file ตาม mapping ข้างบน

## ไฟล์หลัก

- [[skills/prompt-shortcuts/SKILL|SKILL]] - ตัวกำหนดวิธีเรียกใช้ prompt shortcut
- [[skills/prompt-shortcuts/references/use-act-as|use-act-as]] - prompt เต็มสำหรับกำหนด role และแบ่งงานแบบลึก
- [[skills/prompt-shortcuts/references/use-comply|use-comply]] - prompt เต็มสำหรับแตก phase, issue checklist, compliance %, และตรวจ localhost/VPS
- [[skills/prompt-shortcuts/references/use-summary|use-summary]] - prompt สำหรับสรุปและวิเคราะห์ link + content พร้อมเสนอทางเลือกก่อนบันทึก
- [[skills/prompt-shortcuts/references/use-scan-feature|use-scan-feature]] - prompt สำหรับสแกน repo จริงเป็น phase และสกัดบัญชีฟีเจอร์พร้อมหลักฐาน/สถานะจริง-ปลอม
- [[skills/prompt-shortcuts/references/use-ai-pair|use-ai-pair]] - prompt สำหรับทีม 3 AI ค่าเริ่มต้น Claude วางแผน/ตรวจท้าย · Codex เขียน · Qwen ตรวจ read-only พร้อมสร้าง Coder Brief, Review Packet, และ handoff ทันทีเมื่อข้อมูลพอ ไม่หยุดถามซ้ำ
- [[skills/prompt-shortcuts/references/use-business-plan|use-business-plan]] - prompt สำหรับรีวิวโจทย์ธุรกิจ การตลาด pitch งานประมูล เว็บไซต์ และแตกเป็น role/phase/issue ก่อนลงมือ
- [[skills/prompt-shortcuts/references/use-viber-structure|use-viber-structure]] - prompt สำหรับแปลงมาตรฐาน Viber Code / Vibe Code Enterprise เป็นโครงโปรเจกต์ เอกสาร phase/issue tracker และ quality gates
- [[skills/prompt-shortcuts/references/use-viber-audit|use-viber-audit]] - prompt สำหรับตรวจโปรเจกต์เดียวหรือทุกโปรเจกต์เทียบ Viber Enterprise Standard พร้อม gap, คะแนน, หลักฐาน, และ tracker ระหว่างทำงาน
- [[skills/prompt-shortcuts/references/use-impeccable|use-impeccable]] - prompt เดียวสำหรับให้ AI ใช้ Impeccable ตรวจ ติดตั้ง สแกน อธิบาย แก้ หรือวางแผนลด UI/AI slop โดยไม่แตก shortcut ย่อย
- [[skills/prompt-shortcuts/references/use-blog-auto|use-blog-auto]] - prompt สำหรับสกัดองค์ความรู้จากงานจริงไปเป็น draft บล็อก Hi Logic Labs, ทำ privacy gate, Obsidian index, และส่งต่อ Content Factory แบบ draft-first
- [[skills/prompt-shortcuts/references/use-wow-resource|use-wow-resource]] - prompt สำหรับให้ AI อ่าน WOW System และ Web Design Intelligence แล้วคัด resource ให้เหมาะกับโจทย์โดยไม่ copy script ตรงๆ
- [[skills/prompt-shortcuts/references/use-flow-guardian|use-flow-guardian]] - prompt สำหรับบังคับ Home OS Agent safe flow, worktree/branch safety, no-write audit, approval, verify, tracking, และ handoff
- [[skills/prompt-shortcuts/references/use-new-chat|use-new-chat]] - prompt สำหรับเริ่มแชทใหม่แบบต้องตรวจ project, worktree, branch, dirty status, local/VPS equality, service และ endpoint ก่อนตอบ
- [[skills/prompt-shortcuts/references/use-save-git|use-save-git]] - prompt สำหรับ safe Git/GitLab/VPS gate ก่อน push, merge, deploy หรือสรุป readiness โดยต้องคืน SAFE/STOP decision
- [[skills/prompt-shortcuts/references/use-continue|use-continue]] - prompt เต็มสำหรับทำงานต่อเองทีละเฟสจนผ่าน 100%
- [[skills/prompt-shortcuts/references/use-move-folder|use-move-folder]] - prompt สำหรับเชื่อม Hermes กับ workflow จัดเรียงโฟลเดอร์/cleanup จริงบน VPS ที่ `/home/linux-nat/.codex/use-move-folder/project-registry`
- [[skills/prompt-shortcuts/references/go-to-sleep|go-to-sleep]] - alias เก่าสำหรับความเข้ากันได้ย้อนหลัง ให้ชี้ไปใช้ `Use Continue`
- [[skills/prompt-shortcuts/references/review-chat|review-chat]] - prompt เต็มสำหรับรีวิวแชท, ปิดงานค้าง, อัปเดต handoff, และเตรียมข้อความเปิด new chat

## โครงสร้างจริง

ไฟล์จริงอยู่ใน HermesAgent vault:

```text
skills/prompt-shortcuts/
```

Codex runtime path:

```text
~/.codex/skills/prompt-shortcuts
```

เป็น symlink กลับมาที่ vault นี้ เพื่อให้ HermesNous เป็นศูนย์กลางความรู้และ Codex ยังโหลด shortcut ได้เหมือนเดิม

## Graph Links

- Parent hub: [[skills/README|skills]]
- Router: [[00-Center/docs/AI_SKILL_ROUTER|AI Skill Router]]
- Graph: [[00-Center/docs/SKILL_GRAPH|Skill Graph]]
