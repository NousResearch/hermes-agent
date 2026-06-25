---
title: Use Blog Auto
aliases:
  - Use Blog Auto
  - use-blog-auto
  - Blog Auto
  - blog-auto
  - ใช้ Blog Auto
  - เขียนบล็อกอัตโนมัติ
  - ทำบล็อกจากงานนี้
  - ส่งเข้า Hi Logic Labs
tags:
  - prompt-shortcuts
  - hilogiclabs
  - blog
  - content-factory
  - obsidian-index
status: active
version: 1.1
updated: 2026-06-24
---

# Use Blog Auto

## Shortcut

```text
Use Blog Auto
```

## Prompt

```text
Use Blog Auto กับงานนี้

ให้ Hermes ทำหน้าที่เป็น Blog Orchestrator ของ Hi Logic Labs โดยสกัดองค์ความรู้จากงานที่กำลังทำอยู่ แล้วแปลงเป็นระบบผลิตบทความและ asset ต่อเนื่องแบบ draft-first เท่านั้น

หลักการสำคัญ:
- Public brand หลักคือ Hi Logic Labs
- Public English เป็นค่าเริ่มต้นสำหรับเว็บสาธารณะ
- Thai ใช้เป็น member/internal หรือ draft review เท่านั้น เว้นแต่เจ้าของงานสั่งให้ public Thai ชัดเจน
- ห้ามใช้ชื่อบุคคล ชื่อลูกค้า โปรเจกต์จริง IP token key URL ภายใน หรือข้อมูลที่ระบุตัวตนได้ใน public draft
- ห้ามใช้ I/my/me/ผม ใน public draft; ให้ใช้ lab voice เช่น we, the lab, ทีมทดลอง, ห้องแล็บ
- ต้องอ่าน Blog Skill ของ Hi Logic Labs ก่อนเขียนเสมอ · ลำดับหา: path ที่ registry ระบุ → repo-local → `$HERMES_OBSIDIAN_ROOT/skills/hilogiclabs-blog/SKILL.md`
  ($HERMES_OBSIDIAN_ROOT: local = `/Users/rattanasak/ObsidianVault/HermesAgent` · VPS = `/home/linux-nat/ObsidianVault/HermesAgent`)
  หา skill ไม่เจอ = บอกว่าไม่พบ ห้ามเขียนว่าอ่าน skill แล้ว ห้ามเดาเกณฑ์ · compliance = unverified
- compliance รายงานเป็น N/M ข้อ + เช็กลิสต์รายข้อจาก skill (brand voice / faceless / English public / privacy clean / image license / draft-first)
- แยก "คะแนน" กับ "hard gate": privacy fail หรือ publish/post เอง = BLOCKED ทันที แม้คะแนนข้ออื่นผ่าน
- ต้องสร้าง draft แล้วรอ owner approve ก่อน publish หรือ post จริงทุกช่องทาง
- ถ้าเป็นข้อมูลจากแชท/งานปัจจุบัน ให้สรุปและจัดหมวดก่อน ไม่เขียน durable knowledge ทันทีถ้ายังไม่ผ่าน privacy gate

ให้ทำตามลำดับนี้:

1. Intake & Privacy Gate
- สรุปหัวข้อ ความรู้ที่ได้ และความเสี่ยงด้าน privacy
- แยกข้อมูลเป็น public-safe, internal-only, redact-needed, และ do-not-use
- ถ้าเจอ secret/key/token/IP/ชื่อลูกค้า/ชื่อคน/โปรเจกต์จริง ให้ตัดออกหรือแทนด้วยคำกลางทันที

2. Existing Content Match
- ตรวจว่าหัวข้อนี้ควรเป็น blog ใหม่ หรือควรเสริม blog เก่า
- ถ้าจะเสริม blog เก่า ต้องระบุเหตุผลว่าเกี่ยวกับหัวข้อเดิมอย่างไร และเติมส่วนไหนโดยไม่ทำให้เนื้อหาเก่าขัดกัน
- ถ้าค้น blog จริงไม่ได้ ให้บอกชัดว่าเป็น draft route จากข้อมูลที่มี และตั้ง action ให้ค้นก่อน publish

3. Blog Planning
- สร้าง knowledge card สั้นสำหรับ Obsidian index ที่ `$HERMES_OBSIDIAN_ROOT/95-Inbox-Lab/review/` ตั้งชื่อไฟล์ `YYYY-MM-DD-topic-slug.md`
  frontmatter: topic / source_context / privacy_level / audience / language_route / related_posts / draft_status
- เขียน card นี้เฉพาะเมื่อ shortcut ถูกสั่งให้ทำงานจริง · เก็บใน review/ เท่านั้น ไม่ถือเป็นความจำถาวรจนกว่าเจ้าของอนุมัติ (review-before-write) · ไม่พบ path = รายงาน ไม่สร้างมั่ว
- วาง outline ตาม Blog Skill ล่าสุดของ Hi Logic Labs
- สำหรับ public blog ให้ใช้ English draft เป็นหลัก และมี Thai internal summary คู่กัน

4. Draft Production
- เขียน draft เป็น Hi Logic Labs faceless lab voice
- ใช้ภาพจาก screenshot จริงแบบปิดข้อมูล หรือ stock/API image ที่ผ่าน skill แล้วเท่านั้น
- ถ้าใช้ Freepik API หรือ image API ให้เก็บ attribution/license/source id ใน draft metadata
- ห้าม publish เอง ต้องส่งเป็น draft URL หรือ draft artifact ให้ owner ตรวจ

5. Content Factory Handoff
- ส่งบทความที่ผ่าน draft ไป Content Factory เพื่อแปลงเป็น platform-specific drafts:
  Facebook, LinkedIn, X/Twitter, TikTok, Instagram
- ทุก platform ต้องเป็น draft-first และต้องมี approval gate ก่อน post
- Phase วิดีโอ/สไลด์/podcast/YouTube ให้เตรียม script/outline/asset plan เป็น draft ก่อนผลิตจริง เว้นแต่ owner อนุมัติแล้ว

6. Verification & Comply
- ตรวจว่า Blog Skill ถูกอ่านและใช้จริง
- ตรวจว่า draft status ไม่ใช่ published โดย default
- ตรวจว่าไม่มีข้อมูลต้องห้าม
- ตรวจว่า Obsidian index หรือ review queue มีรายการติดตาม
- ตรวจ localhost/VPS endpoint เฉพาะเมื่อบล็อกผูกกับเว็บที่ต้อง publish/preview จริง (capability-based) · งานบล็อกที่ไม่มี endpoint = N/A ไม่บังคับตรวจ · ถ้ามี endpoint แต่ยังไม่ route = blocker แบบตัวเลข
- ส่งตาราง comply โดยช่อง % เป็นตัวเลขเท่านั้น
```

## Required Roles

- Blog Orchestrator: คุม flow ตั้งแต่ intake ถึง draft URL
- Privacy Editor: ตัดข้อมูลลับและทำ redaction
- Hi Logic Labs Editor: ตรวจ brand voice, English public, Thai internal
- Knowledge Librarian: ทำ Obsidian index และ traceability = ร่องรอยว่าองค์ความรู้นี้ถูกใช้ที่ไหนแล้ว
- Content Factory Producer: แปลง blog เป็น draft สำหรับแต่ละ platform
- QA/Comply Auditor: ตรวจ skill compliance, draft-first, localhost/VPS, และตัวเลข completion

## Hard Gates

1. Do not publish without owner approval.
2. Do not post to social without owner approval.
3. Do not expose Thai internal/member notes as public content unless explicitly approved.
4. Do not write or keep sensitive raw chat content in public drafts.
5. Do not claim 100% unless verification evidence exists.

## Output Contract

Every `Use Blog Auto` response must include:

- Blog route: new post / update existing / not suitable for public
- Privacy route: public-safe / internal-only / redact-needed / blocked
- Draft language route: English public + Thai internal summary by default
- Obsidian index action: created / updated / queued / not applicable, with reason
- Content Factory handoff status
- Skill compliance score as numbers
- Final review URL or blocker if URL cannot be produced

## Changelog

- v1.1 (2026-06-24): ผ่านตรวจ 2 AI (Claude+Codex) · ระบุลำดับหา Blog Skill (registry→repo→$HERMES_OBSIDIAN_ROOT) + fallback root · หา skill ไม่เจอ = unverified ห้ามอ้างว่าอ่านแล้ว · compliance เป็น N/M + เช็กลิสต์ + แยก hard gate (privacy/publish เอง = BLOCKED) · Obsidian index ระบุ path + filename rule + อยู่ใต้ review-before-write · localhost/VPS เป็น capability-based (ไม่มี endpoint = N/A)
- v1.0 (2026-06-07): เวอร์ชันแรก

## Graph Links

- Parent hub: [[skills/prompt-shortcuts/Prompt Shortcuts|Prompt Shortcuts]]
- Registry: [[ai-context/prompt-shortcut-registry|Prompt Shortcut Registry]]
