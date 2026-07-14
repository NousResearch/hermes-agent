---
title: Review Chat
aliases:
  - Review Chat
  - review-chat
  - Chat Review
  - chat-review
  - รีวิวแชท
  - ตรวจแชท
  - สรุปส่งต่อ
  - สรุปเปิดแชทใหม่
tags:
  - prompt-shortcuts
  - chat-review
  - handoff
  - context-management
status: active
version: 2.4
updated: 2026-07-14
schema: memory-schema-v1.2
routes_to: use-close-chat-preview
---

# Review Chat (v2.4 · Preview Alias)

## Shortcut

```text
Review Chat
```

## Prompt

```text
Review Chat

เรียก `Use Close Chat` v2.4 ในโหมด `PREVIEW_ONLY`

กฎบังคับ:
- ตรวจบทสนทนา งานค้าง หลักฐานจริง ไฟล์สำคัญ และความเสี่ยงตาม Use Close Chat ทุกข้อที่เกี่ยวข้อง
- แยก "ตรวจจากหลักฐานจริง" ออกจาก "สรุปจากบทสนทนา"
- แสดงรายการไฟล์ความจำที่จะอัปเดตและสิ่งที่จะเพิ่ม แต่ห้ามเขียนไฟล์ ห้ามปลด claim และห้ามเปลี่ยน Git
- สร้างข้อความเปิดแชทใหม่แบบสั้น โดยชี้ `.project/OverviewProgress.md`, `.project/decisions.md` และ session log ล่าสุดแทนการคัดลอกความจำยาวซ้ำอีกชุด
- ไม่ประเมินเปอร์เซ็นต์ context window ถ้าหน้าจอไม่แสดงค่าจริง; ให้บอกเพียงว่าเห็นหรือไม่เห็นค่า
- ผลลัพธ์ต้องจบด้วย `PREVIEW_READY_FOR_CLOSE` และคำขออนุมัติครั้งเดียวสำหรับชุดไฟล์ทั้งหมดเมื่อจำเป็น

ห้ามทำขั้น CLOSE หรือเขียน memory จนกว่าเจ้าของสั่ง `Use Close Chat` หรืออนุมัติพรีวิวชัดเจน
```

## Worktree Lifecycle v1

อ่าน `worktree-lifecycle-contract.md` ก่อนใช้ Prompt นี้ · เพิ่มหลักฐาน `task_id / machine_id / worktree / branch / writer / lifecycle / cleanup` ในรายงาน · Review Chat อ่านสถานะและเสนอ action เท่านั้น เว้นแต่เจ้าของอนุมัติการเขียน/ปิดชัดเจน

## Changelog

- v2.4 (2026-07-14): ยุบ Review Chat เป็น alias โหมด PREVIEW_ONLY ของ Use Close Chat v2.4 · ตัดการอ่าน/สรุป/สร้างความจำซ้ำ · ไม่เดาเปอร์เซ็นต์ context ที่มองไม่เห็น
- v2.3 (2026-07-05): ผูก Memory Schema v1.2 และใช้ review-before-write

## Graph Links

- Parent hub: [[skills/README|skills]]
- Close engine: [[skills/prompt-shortcuts/references/use-close-chat|Use Close Chat]]
- Schema: [[skills/prompt-shortcuts/references/memory-schema|Memory Schema v1.2]]
