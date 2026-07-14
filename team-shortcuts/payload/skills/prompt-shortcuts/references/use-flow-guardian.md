---
title: Use Flow Guardian
aliases:
  - use-flow-guardian
  - Flow Guardian
  - Safe Flow
  - ใช้ Flow Guardian
  - ใช้ Safe Flow
  - เปิด Flow Guardian
  - ตรวจ worktree
  - กัน AI แก้งานทับกัน
tags:
  - prompt-shortcut
  - flow-guardian
status: active
created: 2026-06-06
updated: 2026-06-24
---

# Use Flow Guardian

บังคับวินัยความปลอดภัย "ระหว่างทำงาน" ในโปรเจกต์ · กัน AI แก้งานทับกัน + กันลงมือก่อนอนุมัติ

> ความต่างจาก Use New Chat: New Chat = ด่านตอน "เปิดแชท" ตรวจสถานะเริ่มต้นครั้งแรก ·
> Flow Guardian = วินัย "ต่อเนื่องระหว่างทำงาน" ใช้ startup report ล่าสุด · ถ้าไม่มีหรือ stale ค่อยเรียก New Chat ใหม่
> (alias "New Chat Gate" ถูกถอดออก → ใช้ Use New Chat แทน)

## Required Context (path จริง พกพาได้)

อ่านจาก `$HERMES_OBSIDIAN_ROOT/ai-context/` (local = `/Users/rattanasak/ObsidianVault/HermesAgent` · VPS = `/home/linux-nat/ObsidianVault/HermesAgent`):
- `home-os-agent.md` · `ai-workflow-guardian-policy.md` · `ai-worktree-branch-safety-rules.md` · `ai-approval-gates.md`
- repo-local: project adapter ที่ตรง (AGENTS.md / .hermes/context.md)
หา root/ไฟล์ไม่เจอ → รายงาน path ที่หา · ห้าม fallback ไปความจำ ห้ามทำเหมือนอ่านแล้ว

## Flow (ลำดับงาน)

1. รายงานสถานะจริง (รันคำสั่ง ไม่เดา):
   `pwd` · `git rev-parse --show-toplevel` · `git branch --show-current` · `git status --short --branch` · `git worktree list` · `git rev-parse HEAD` · `git remote -v`
2. ฟีเจอร์ใหม่ → ถามว่าใช้ worktree/branch เดิม หรือสร้างใหม่ (ใช้ข้อความมาตรฐานด้านล่าง)
3. ตรวจการจองงาน (ใช้ระบบเดียวกับ New Chat): `claim list` → เช็ก path overlap → `claim acquire` ก่อนเริ่มแก้ → จบงาน `claim release` หรือเปลี่ยนเป็น handoff
4. งาน "เสี่ยง" → ทำ no-write audit (อ่านอย่างเดียว) ก่อนแก้
5. เข้า approval gate → ขออนุมัติเจ้าของงานก่อน
6. ทำเฉพาะ scope ที่อนุมัติ
7. verify ด้วยหลักฐานจริงก่อนบอกเสร็จ
8. อัปเดต tracking/handoff (review-before-write — เสนอก่อน รออนุมัติถ้าไม่มีคำสั่งชัด)

## นิยาม "เสี่ยง" (มี threshold ไม่ตีความกว้าง)

เข้าข่ายเสี่ยง = ข้อใดข้อหนึ่ง: แตะมากกว่า 1 โมดูล / ไฟล์ข้าม ownership / migration / deploy / แตะ auth-security / config / secret / แก้เกิน ~10 ไฟล์
→ งานเสี่ยงต้อง no-write audit + ขออนุมัติก่อน · งานเล็กในโมดูลเดียวไม่ต้อง

## STOP rule (หยุดเฉพาะงานเขียน/เสี่ยง — งานอ่านยังทำได้)

หยุด "เขียน/แก้/deploy" (ไม่ใช่หยุดคิดหรือ audit) เมื่อ:
- ไม่มี context/adapter หรือไม่รู้ branch ปลอดภัย
- worktree dirty ที่เป็นงานคนอื่น
- claim ชนกับคนอื่น
- จะแตะงานเสี่ยงโดยยังไม่อนุมัติ
→ หยุด รายงานสาเหตุ + สิ่งที่ต้องให้เจ้าของงานตัดสิน

## ข้อความถามเจ้าของงานสำหรับฟีเจอร์ใหม่

```text
ตอนนี้คุณกำลังทำงานอยู่ที่:
Worktree: <path>
Branch: <branch>
สถานะไฟล์: <clean / dirty / unknown>

งานนี้เป็นฟีเจอร์ใหม่ ต้องการให้ใช้ worktree/branch นี้ต่อ หรือสร้างใหม่เพื่อกันงานทับกันครับ?
```

## Handoff template (ขั้นต่ำ)

```text
worktree:
branch:
changed files:
verification (คำสั่ง + ผลจริง):
result:
risk ที่เหลือ:
next step:
protected areas (ห้ามแตะ):
```

## Worktree Lifecycle v1

อ่าน `worktree-lifecycle-contract.md` ก่อนใช้ Prompt นี้ · ก่อนเขียนต้องตรวจ `task_id + machine_id + writer lease + branch + runtime namespace` ด้วย `hermes worktree status` · task ใหม่ส่งให้ Manager เปิดพื้นที่; ห้าม Guardian สร้างหรือสลับเอง

## Changelog

- v1.1 (2026-06-24): ผ่านตรวจ 2 AI (Claude+Codex) · context เป็น path จริง $HERMES_OBSIDIAN_ROOT (หาไม่เจอ=รายงาน ไม่เดา) · ใส่คำสั่ง git จริง (+pwd/HEAD/remote) · นิยาม "เสี่ยง" มี threshold · STOP เฉพาะงานเขียน/เสี่ยง (อ่านได้) · เพิ่ม handoff template · เชื่อมระบบจองงาน claim เดียวกับ New Chat · ชี้ความต่างจาก New Chat + ถอด alias "New Chat Gate"
- v1.0 (2026-06-06): เวอร์ชันแรก

## Links

- [[ai-context/home-os-agent|Home OS Agent]]
- Registry: [[ai-context/prompt-shortcut-registry|Prompt Shortcut Registry]]
- Related: [[skills/prompt-shortcuts/references/use-new-chat|Use New Chat]]
