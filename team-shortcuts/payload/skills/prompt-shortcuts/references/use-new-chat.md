---
title: Use New Chat
aliases:
  - Use New Chat
  - use-new-chat
  - Start New Chat
  - start-new-chat
  - New Chat Startup
  - new-chat-startup
  - Initialize Hermes Agent chat
  - เริ่ม New Chat
  - เปิด New Chat
  - เริ่มแชทใหม่
  - เปิดแชทใหม่
tags:
  - prompt-shortcuts
  - new-chat
  - startup-gate
  - flow-guardian
status: active
version: 1.6
updated: 2026-06-24
---

# Use New Chat

## Shortcut

```text
Use New Chat
```

## Prompt

```text
Use New Chat

เป้าหมาย: เริ่มแชทใหม่ให้พร้อมทำงานจริง โดยตรวจของจริงก่อน ไม่ใช่แค่ตอบว่า "พร้อมรับคำสั่ง"
หลักการ: capability-based — มีอะไรตรวจอันนั้น · ไม่มีก็ใส่ N/A ไม่ใช่บังคับมีทุกช่อง

กฎบังคับ:
- ห้ามตอบ "พร้อมรับคำสั่ง" ก่อนตรวจสถานะจริง
- ห้ามเดา project / worktree / branch / dirty / remote / VPS / service · ตรวจด้วย command จริงก่อนสรุป
- ทุกค่าที่รายงานต้องมาจาก command จริง · แนบ timestamp + host + cwd ท้าย report (กัน report ที่ดูเหมือนตรวจแต่ไม่ได้ตรวจ)
- ผู้ใช้พิมพ์ไทย ตอบไทย แปลศัพท์เทคนิคทันที
- ห้ามแก้ไฟล์ก่อนส่ง startup report เว้นแต่ผู้ใช้สั่งทำต่อเอง
- redact secret ใน remote URL เสมอ (เช่น token ใน https://token@... ให้แสดงเป็น https://***@...)

แหล่ง config (ลำดับความสำคัญ · ห้าม fallback ไปค่า default ของโปรเจกต์อื่น):
project config > repo runbook > AGENTS.md/.hermes context > ถามเจ้าของงาน

ขั้นตอน (ทำเท่าที่โปรเจกต์มี):

0. อ่านความจำล่าสุดก่อนเริ่ม (บังคับ — คู่กับ Use Close Chat แก้ AI ลืมงานข้ามแชท)
- เปิดอ่าน `handoff.md` + session log ล่าสุด (ตามตัวชี้ `latest-close.md` ของโปรเจกต์) ก่อนตอบงานเสมอ
- สรุปให้เจ้าของงานเห็นว่า แชทก่อนทำอะไรค้าง / decision อะไร / งานค้างของใคร
- ถ้า decision token ล่าสุด = CLOSED_WITH_PENDING ต้องอ่านงานค้างก่อนเริ่ม · = NEED_OWNER_ACTION_BEFORE_CLOSE แปลว่ารอบก่อนยังไม่ปิดจริง ให้เตือน
- หา latest-close/handoff ไม่เจอ = บอกว่าไม่พบความจำรอบก่อน ไม่เดาว่าทำอะไรไป
- บังคับสรุปออกมา 3 บรรทัดตายตัว (ไม่ครบ = ยังไม่เริ่มงาน):
  · Process อยู่ขั้นไหน
  · งานล่าสุดทำถึงไหน (commit/PR/SHA ล่าสุด)
  · สิ่งที่ต้องทำต่อ 1 ข้อ
- "งานถึงไหน" ต้องเทียบ `git log -5` + branch ปัจจุบัน + PR/timestamp ด้วย ไม่เชื่อ handoff/session log อย่างเดียว (อาจ stale)

1. Project Detection
- project path ปัจจุบัน + repo name
- execution target: local หรือ VPS (ถ้า shell อยู่ local แต่ target เป็น VPS ต้องรายงานทั้งสองค่า ห้ามบอกว่า "อยู่บน VPS แล้ว")
- repo-local context ที่เจอ: AGENTS.md / .hermes/context.md / active.md / decisions.md

2. Git Gate
- git rev-parse --show-toplevel / git branch --show-current / git status --short --branch / git worktree list
- แปลผล clean หรือ dirty
- ถ้าไม่ใช่ git repo / เป็น nested git / monorepo ที่ไม่ชัด → สถานะ NEED_OWNER_INPUT ห้ามเดา

3. Remote Gate
- ตรวจ remote · ถ้ามีหลาย remote ให้รายงาน candidate ทั้งหมด แล้วถามว่าอันไหนคือ remote หลัก ห้ามเลือกมั่ว
- เทียบ local HEAD กับ remote หลัก (เมื่อรู้แน่)

4. Staff / Worktree Routing Gate — [ทำเฉพาะเมื่อ project config ระบุว่าใช้ worktree/ownership routing]
- ระบุ staff id (เช่น nat/may/mind) · ถ้าไม่รู้ → ถามก่อน ห้าม fallback ไป worktree คนอื่น
- ตรวจว่า staff id + project มี worktree จริงไหม (ถ้ามี scripts/hermes_worktree_route.py ให้รัน)
- ไม่เจอ worktree → รายงาน missing worktree หยุดก่อนแก้ไฟล์
- โปรเจกต์ทั่วไปที่ config ไม่ได้ระบุ routing → ข้ามขั้นนี้ (ไม่ต้องถาม staff id)

5. VPS / Runtime Gate — [capability-based]
- โปรเจกต์มี VPS/service ตาม config → ตรวจ service/endpoint ที่ config ระบุเท่านั้น (ห้าม port-scan กว้าง)
- โปรเจกต์ไม่มี VPS → ใส่ "N/A: โปรเจกต์นี้ไม่มี VPS"
- ตรวจไม่ได้ → ระบุว่าไม่ได้ตรวจ เพราะอะไร ห้ามบอกว่าใช้ได้

6. Risk Gate
- dirty files → แยกว่าเป็นไฟล์แก้ค้างหรือไฟล์ใหม่
- งานถัดไปเป็น feature/deploy/migration/security/multi-file → บอกว่าต้อง audit ก่อนแก้
- worktree dirty + งานเป็น feature ใหม่ → หยุดถามหรือเลือก worktree ปลอดภัยกว่า

[Hermes Agent เท่านั้น — ตรวจจาก repo marker ไม่ใช่ชื่อโฟลเดอร์]
- ถ้า project config ยืนยันว่าเป็น Hermes Agent: Mac = local controller · VPS target ตามที่ runbook ระบุ · ใช้ ssh/enter script ของโปรเจกต์ ไม่ hardcode IP/path ใน prompt
- กฎแสดงผล branch: ถ้า branch จริงเป็นชื่อพนักงาน (เช่น nat) แสดงในรายงานเป็น "Update Feature (branch จริง: nat)" — เป็น config เฉพาะ Hermes ไม่ใช่กฎสากล · ห้ามเปลี่ยนชื่อ branch git จริง

[New Feature Branch Gate — บังคับสร้าง branch ใหม่ก่อนแก้โค้ดฟีเจอร์ใหม่]
เป็น "ฟีเจอร์ใหม่" เมื่อ: เพิ่ม behavior/UI/command/tool/config/integration ใหม่ หรือเจ้าของงานใช้คำว่า build/add/create/implement/ทำ
เป็น "งานต่อเดิม" ต้องมีหลักฐาน: handoff ระบุ branch เดิม / task-PR เดิม / commit ล่าสุดตรงกับงานนั้น · ไม่มีหลักฐาน = ถือเป็นงานใหม่ ห้ามเดาว่าต่อ

ก่อนแก้โค้ดทุกครั้งเช็ก:
- branch ปัจจุบันเป็น main/master/develop/shared/release/ของคนอื่น + เป็นฟีเจอร์ใหม่ → STOP ห้ามแก้โค้ด
- detached HEAD → STOP เสมอ
- dirty worktree → STOP ถามจะพา changes ไป branch ใหม่ / stash / ให้เจ้าของจัดการก่อน
ต้องสร้าง branch ใหม่ก่อน: ทีม/VPS = `worktree/<staff>/<slug>` · local = `feature/<slug>` (มี ticket id ใส่ในชื่อ) ·
ชื่อชนของเดิม = เสนอชื่อใหม่ ไม่เขียนทับ · เสนอชื่อ + ขอยืนยันก่อนสร้าง

ยกเว้น (ไม่บังคับสร้าง branch · กัน over-block):
- งานอ่านอย่างเดียว / audit / review / explain / รัน test / grep / ดู git
- แก้ typo หรือ docs เล็กมาก ถ้าเจ้าของยืนยันให้ทำบน branch ปัจจุบัน
- hotfix เร่งด่วน → ใช้ `hotfix/<slug>`
- งานต่อ PR/branch เดิมที่มีหลักฐานชัด (ยืนยันจาก branch/handoff/PR/task id ก่อน)

[Team-Safe Flow — กันงานทับซ้อน/dirty เมื่อหลายคน+หลาย AI ทำร่วมกัน]
ทำเมื่อ project config ระบุว่าเป็นงานทีม/มี worktree routing · งานแค่ถาม-ไม่แก้ repo ข้ามได้

ลำดับ:
1. ระบุ staff id → route ไป worktree ของตัวเองด้วย hermes_worktree_route.py · ห้ามไป worktree คนอื่น
2. อ่านไฟล์ประสานทีม (อ่านอย่างเดียว ตามลำดับ): handoff.md → .hermes/active.md → .hermes/decisions.md
3. เช็กการจองงาน: รัน `hermes_worktree_route.py claim list` (ห้ามอ่าน/เขียน markdown จอง้เอง)
   - งานที่จะทำมี path "ซ้อนทับ" กับ claim ของคนอื่นไหม (เทียบแบบ path overlap เช่น gateway/ ซ้อน gateway/run.py ไม่ใช่ชื่อตรงเป๊ะ)
   - ชน → STOP รายงานว่าใครจอง path ไหน
   - เจอ claim หมดอายุ (เลย expires_at) → สถานะ STALE_CLAIM_REVIEW · ไฟล์เสี่ยง (config/migration/deploy/secret) ห้ามยึดอัตโนมัติ ต้องถามเจ้าของ
4. เช็ก dirty บน worktree ตัวเอง: dirty + ไม่ใช่งานที่ตัวเองค้าง → STOP (อาจเป็นงานคนอื่น/รอบก่อน)
5. จองงานก่อนเริ่มแก้: `hermes_worktree_route.py claim acquire` (staff_id/project/worktree/branch/issue/paths/expires_at)
   - claim = "ตั้งใจจะเริ่มแก้" เท่านั้น · แหล่งจริงของ "แก้อะไรไปแล้ว" คือ git status/branch — ต้องเทียบทั้งสองเสมอ
6. ปลด claim เมื่อจบ: หลัง commit/handoff รัน `claim release` หรือเปลี่ยนเป็น handoff
ถ้าไม่มีคำสั่ง claim/registry → อ่าน handoff/active.md ได้ แต่รายงานว่า claim เป็น unverified ห้ามทำเหมือนล็อกแล้ว

รูปแบบคำตอบบังคับ:

New Chat Startup Report
Project: path / repo / execution target / local controller / context loaded
Git: worktree / branch / dirty / HEAD / remote match (หรือ candidate ถ้ายังไม่รู้)
Routing: staff id / target worktree / route status (หรือ N/A ถ้าโปรเจกต์ไม่ใช้ routing)
VPS / Runtime: service / endpoint (หรือ N/A)
Evidence: timestamp / host / cwd / commands ที่รัน
Readiness: READY / BLOCKED / NEED_OWNER_INPUT + เหตุผล + งานที่ควรทำต่อ

ตัวอย่าง READY (โปรเจกต์ทั่วไปไม่มี VPS):
Readiness: READY · git clean บน main ตรง origin · ไม่มี VPS (N/A) · พร้อมรับงานต่อ

ตัวอย่าง BLOCKED:
Readiness: NEED_OWNER_INPUT · มี 2 remote (origin, fork) ยังไม่รู้อันไหนหลัก + ไม่รู้ staff id ของ worktree routing · ขอ 2 ข้อนี้ก่อน

ข้อห้าม: ตอบแค่ "พร้อมรับคำสั่ง" · บอก clean โดยไม่รัน git status · บอก VPS ใช้ได้โดยไม่ตรวจ endpoint · สรุปด้วยศัพท์เทคนิคโดยไม่แปล
```

## Changelog

- v1.6 (2026-06-24): ผ่านตรวจ 2 AI · แก้ 2 ปัญหาเจ้าของงาน — (1) บังคับสรุปบริบทเก่า 3 บรรทัด (Process ขั้นไหน/งานล่าสุดถึงไหน/ต้องทำต่ออะไร) + เทียบ git log ไม่เชื่อ handoff อย่างเดียว (2) เพิ่ม New Feature Branch Gate (hard gate): ฟีเจอร์ใหม่บน main/shared/detached/dirty = STOP ต้องสร้าง branch ใหม่ (worktree/<staff>/<slug> หรือ feature/<slug>) ก่อนแก้โค้ด · ยกเว้นงานอ่าน/typo/hotfix/งานต่อที่มีหลักฐาน (กัน over-block)
- v1.5 (2026-06-24): ปิดวงจรความจำคู่กับ Use Close Chat · เพิ่มขั้น 0 บังคับอ่าน handoff.md + session log ล่าสุด (ตาม latest-close.md) ก่อนเริ่มงานเสมอ → แก้ AI ลืมงานข้ามแชท · อ่าน decision token รอบก่อน (CLOSED_WITH_PENDING/NEED_OWNER_ACTION) · หาความจำไม่เจอ = บอก ไม่เดา
- v1.4 (2026-06-24): เพิ่ม Team-Safe Flow กันงานทับซ้อน/dirty เมื่อหลายคน+หลาย AI ทำร่วมกัน (ผ่านตรวจ Codex) · จองงานผ่านคำสั่ง `hermes_worktree_route.py claim acquire/release/list` (มี file lock + expires_at) ไม่ให้ AI เขียน markdown จองเอง · เช็กชนแบบ path overlap · claim หมดอายุ = STALE_CLAIM_REVIEW · git/branch เป็นแหล่งจริงของ "แก้ไปแล้ว" claim เป็นแค่ "ตั้งใจจะแก้" · ปลด claim เมื่อ commit/handoff
- v1.3 (2026-06-24): ผ่านตรวจ 2 AI (Claude+Codex) · เปลี่ยนเป็น capability-based gate (มีอะไรตรวจอันนั้น ไม่มีใส่ N/A) · แยก Core gate (ทุกโปรเจกต์) ออกจาก Hermes-only · เลิก hardcode IP/port/path/ssh (อ่านจาก config/runbook ของโปรเจกต์) · staff id/worktree/VPS บังคับเฉพาะโปรเจกต์ที่ config ระบุ · เพิ่ม Input contract (ไม่รู้ staff id/remote = ถามก่อน) · redact secret ใน remote URL · เพิ่ม Evidence (timestamp/host/cwd) กัน report ปลอม · Readiness เป็น token READY/BLOCKED/NEED_OWNER_INPUT + ตัวอย่าง
- v1.2 (2026-06-06): เวอร์ชันก่อนหน้า

## Graph Links

- Parent hub: [[skills/prompt-shortcuts/Prompt Shortcuts|Prompt Shortcuts]]
- Registry: [[ai-context/prompt-shortcut-registry|Prompt Shortcut Registry]]
- Related: [[ai-context/session-start-contract|Session Start Contract]]
```
