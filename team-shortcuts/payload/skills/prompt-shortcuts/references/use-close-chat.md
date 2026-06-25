---
title: Use Close Chat
aliases:
  - Use Close Chat
  - use-close-chat
  - Close Chat
  - close-chat
  - ใช้ Close Chat
  - ปิดแชท
  - ปิดงานแชท
  - จบแชท
tags:
  - prompt-shortcuts
  - close-chat
  - handoff
  - session-memory
  - context-management
status: active
version: 1.0
created: 2026-06-24
---

# Use Close Chat

ปิดแชทแบบปลอดภัย + เขียน "ความจำถาวร" ให้แชทหน้าอ่านตอนเปิด · แก้ปัญหา AI ลืมว่าทำอะไร แก้อะไร ถึงไหน เมื่อข้ามแชท

> คู่กับ Use New Chat: เปิด = New Chat (อ่านความจำล่าสุดก่อนเริ่ม) · ปิด = Close Chat (เขียนความจำ)
> บทบาทไม่ทับกัน: Close Chat = ปิดงาน+เขียน memory · Review Chat = ตรวจคุณภาพงาน/claim · Save Git = commit/push · New Chat = เปิดงาน+อ่าน memory
> Close Chat **ไม่ push/merge เอง** ชี้ไป Use Save Git / Use Merge to Production

## Pre-Close Gate (5 ด่านสั้น + ค่าปลอดภัยตั้งต้น)

1. **Commit แล้วยัง** — รัน `git status --short --branch` จริง · มีไฟล์ค้าง = ยังปิด clean ไม่ได้ · ถามเจ้าของงานจะ commit / stash / discard
2. **ต้อง push / merge / แจ้ง merger ไหม** — ถ้าใช่ ชี้ไป Use Save Git (merge-gate) หรือ Use Merge to Production · Close Chat ไม่ทำเอง
3. **แต่ละงานในแชท "ทำจริง" ไหม** — ไล่ทุก task · แยก `verified` (มีหลักฐานตรวจจริง) กับ `claimed` (แค่ AI บอกว่าเสร็จ) · claimed ที่ยังไม่ตรวจ = นับเป็นงานค้าง
4. **AI ลืมอะไรไหม** — สรุป prompt ทั้งแชทของเจ้าของงานเป็น: เป้าหมายเดิม / คำสั่งที่เปลี่ยนทิศ / decision สำคัญ / ข้อห้าม → ข้อไหนยังไม่ทำหรือตกหล่น เอามาแสดง
5. **Claim + dirty** — ปลด claim ของงานนี้ (`claim release` หรือเปลี่ยนเป็น handoff) · dirty ที่เป็นงานนี้ต้องตัดสินก่อน · dirty ของคนอื่น = ไม่แตะ รายงาน

## เขียนความจำถาวร (review-before-write — เสนอก่อน รออนุมัติถ้าไม่มีคำสั่งชัด)

1. `handoff.md` = สถานะล่าสุดของโปรเจกต์ (สั้น) + ลิงก์ไป session log · ไม่ใช่ที่ทิ้ง log ทุกแชท
   - ถ้า handoff ถูกแก้หลังเริ่มแชท → อ่าน diff ก่อนเขียน · ชนกับคนอื่น = เขียนเข้า review queue ก่อน ไม่ทับทันที
2. session log = `$HERMES_OBSIDIAN_ROOT/projects/<project>/session-logs/YYYY-MM-DD-<staff>-<branch>.md`
   ($HERMES_OBSIDIAN_ROOT: local = `/Users/rattanasak/ObsidianVault/HermesAgent` · VPS = `/home/linux-nat/ObsidianVault/HermesAgent`)
   - อัปเดต `latest-close.md` (ตัวชี้ session ล่าสุดต่อโปรเจกต์) ให้ New Chat อ่านถูกไฟล์
3. เนื้อ session log ขั้นต่ำ:
   - เป้าหมายรอบนี้
   - Changed-files table: `| file | owner | changed_by | reason | verification | risk | next_owner |`
   - Decision log: การตัดสินใจสำคัญ + ทำไมเลือกทางนี้
   - คำสั่งตรวจ + ผลจริง
   - งานค้าง + เจ้าของถัดไป
   - ความเสี่ยงที่เหลือ
   - next step + ข้อความเปิดแชทหน้า

## Decision Token ปิด (บอกแชทหน้าให้ทำตัวยังไง)

- `CLOSED_CLEAN` — commit ครบ ไม่มีค้าง · New Chat ทำงานต่อได้เลย
- `CLOSED_WITH_PENDING` — ปิดได้แต่มีงานค้าง · New Chat ต้องอ่านงานค้างก่อนเริ่ม
- `NEED_OWNER_ACTION_BEFORE_CLOSE` — ยังปิดไม่ได้ (dirty ไม่ตัดสิน / ต้อง commit / ต้อง merge) · ห้ามบอกว่าปิดแล้ว

## Output

- สรุปภาษาคน (ทำอะไรไป / เหลืออะไร / ลืมอะไรไหม)
- Decision token
- ลิงก์ session log + handoff ที่อัปเดต (เป็น path ข้อความ ไม่ใช่ลิงก์คลิกถ้าอยู่นอกโปรเจกต์)
- งานค้าง + เจ้าของถัดไป
- ข้อความเปิดแชทหน้า (พร้อมใช้)

## Changelog

- v1.0 (2026-06-24): สร้างใหม่ตามโจทย์เจ้าของงาน (แก้ AI ลืมงานข้ามแชท) · ผ่านตรวจ 2 AI (Claude+Codex) · Pre-Close Gate 5 ด่าน · เขียนความจำถาวร (handoff สั้น + session log แยกคน/วัน/branch + latest-close pointer ให้ New Chat อ่าน) · changed-files + decision log สำหรับทีม · กันชน handoff คนอื่น · Decision token ปิด 3 แบบ · ไม่ push/merge เอง

## Graph Links

- Parent hub: [[skills/prompt-shortcuts/Prompt Shortcuts|Prompt Shortcuts]]
- Registry: [[ai-context/prompt-shortcut-registry|Prompt Shortcut Registry]]
- Pair: [[skills/prompt-shortcuts/references/use-new-chat|Use New Chat]]
- Related: [[skills/prompt-shortcuts/references/review-chat|Review Chat]]
