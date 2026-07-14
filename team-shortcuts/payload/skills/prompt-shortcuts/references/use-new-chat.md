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
version: "2.7"
updated: 2026-07-14
schema: memory-schema-v1.2
pairs_with: use-close-chat >= 2.2
includes: use-ai-relay >= 2.9
---

# Use New Chat (v2.7 · 2026-07-14)

คู่กับ Use Close Chat ≥ v2.2 · อ้าง Memory Schema v1.2 · เช็ก schema version ตอนเริ่ม · ไม่ตรง = เตือน + ห้ามเขียนไฟล์ความจำจนกว่าจะอ่าน schema ล่าสุด

## Shortcut

```text
Use New Chat
```

## Prompt

```text
Use New Chat

คู่กับ Use Close Chat ≥ v2.2 · อ้าง Memory Schema v1.2 · เช็ก schema version ตอนเริ่ม · ไม่ตรง = เตือน + ห้ามเขียนไฟล์ความจำจนกว่าจะอ่าน schema ล่าสุด
เป้าหมาย: เริ่มแชทใหม่ให้พร้อมทำงานจริง โดยตรวจของจริงก่อน + เรียนรู้ทั้งโปรเจกต์ ไม่ใช่แค่ตอบ "พร้อมรับคำสั่ง"
หลักการ: capability-based — มีอะไรตรวจอันนั้น · ไม่มีก็ N/A

[Use New Chat = New Chat Gate + AI Relay Startup]
- ผู้ใช้พิมพ์ `Use New Chat` ครั้งเดียว = ทำขั้นเปิดแชททั้งหมด + เปิดโหมดตรวจความพร้อมของ Use AI Relay อัตโนมัติ
- AI Relay Startup ตรวจทะเบียน/โปรแกรม/รุ่น/บัญชี/สถานะพัก/งานค้างจากของจริง แต่ห้ามยิง AI เพียงเพราะเปิดแชท
- รายงาน AI Relay: ตัวคิด/ตัวเขียน/ตัวตรวจใดพร้อม ไม่พร้อม ค้าง พัก หรือรุ่นไม่เข้ากัน
- ถ้า Relay ไม่พร้อม ยังอ่าน/ตรวจได้ แต่ห้ามเริ่มงานเขียนที่ต้องใช้ Relay จนกว่าจะรายงานข้อจำกัดให้เจ้าของรู้
- ส่งกฎตรวจไม่ผ่าน 2 รอบไปยัง AI Relay: ผู้ตรวจและวิธีเดิมใช้ได้สูงสุด 2 รอบต่อปัญหา · ถ้ายังไม่ผ่านต้องแยกปัญหาและเปลี่ยนเป็น test/lint/build/gate-run หรือผู้ตรวจคนละค่าย ห้ามเรียกรอบที่ 3

[Fresh Close Receipt — ลดการตรวจซ้ำแต่ไม่เชื่อความจำลอย ๆ]
- อ่าน `latest-close.md` ก่อน แล้วเทียบ receipt กับ project/workspace/branch/HEAD SHA ของจริง
- ไม่ว่าปิดแบบใด ต้องรันขั้นต่ำสดเสมอ: cwd + git root + branch + `git status --short --branch` + memory schema + hook doctor
- receipt ตรงและ `CLOSED_CLEAN` = ใช้ผล quality/CI/VPS ที่แนบไว้ได้โดยไม่รันด่านหนักซ้ำ เว้นแต่งานใหม่ต้องใช้ความสามารถนั้นหรือสถานะเปลี่ยน
- receipt ไม่ตรง/ไม่มี/เป็น PENDING หรือ NEED_OWNER_ACTION = ตรวจเต็มตาม capability และรายงานความต่าง
- ห้ามข้าม Git ขั้นพื้นฐานเพียงเพราะ token บอก CLEAN

[Hook Health Gate — ตรวจจริง ไม่ใช่ดูว่ามีไฟล์]
- รัน `hermes-hook-doctor` ตอนเปิดแชต · ต้องได้ `ok=true` ครบ 3 ด่าน: ภาษาคน / ผู้ตรวจอิสระก่อนปิดงานโค้ด / หลักฐานครบตาม Prompt
- ถ้าคำสั่งหายหรือด่านใดไม่ผ่าน ให้รายงาน `HOOK_HEALTH_BLOCKED` และห้ามอ้างว่างานเขียนพร้อมส่ง · ไม่ต้องคัดลอกกฎเต็มเข้าคำตอบ

[คำถามเลือกรูปแบบการใช้ AI — ถามเมื่อกำลังจะใช้งานหลาย AI เท่านั้น]
หลังแสดง New Chat Startup Report ให้ถาม 1/2 เมื่อผู้ใช้สั่งงานที่ต้องใช้ AI Relay/หลาย AI และแชทนี้ยังไม่มีคำตอบ ถ้ารอบนั้นเป็นงานอ่านทั่วไป งาน AI เดียว หรือยังไม่มีงานเป้าหมาย ให้ยังไม่ถาม:
1. แบ่ง AI คนละขั้น: Opus ศึกษา/วิเคราะห์/วางแผน → AI ตัวที่สองผลิตผลงานหรือลงมือทำ → AI ตัวที่สามตรวจผลงาน
2. ใช้ AI สองตัวในขั้นศึกษา/วิเคราะห์: AI ตัวหลักสร้าง output (ผลลัพธ์หลัก) → AI อีกตัวรีวิวเพื่อตรวจข้อผิดพลาดก่อนยอมรับผล แล้วจึงส่งเข้าสายงานถัดไป
เมื่อเข้าเงื่อนไขหลาย AI: ห้ามเรียก AI เพื่อวิเคราะห์งาน ห้ามสร้างแผน และห้ามเริ่มงานเขียนจนกว่าผู้ใช้เลือก 1 หรือ 2 · เมื่อเลือกแล้วให้ส่งค่าต่อไปยัง AI Relay โดยไม่ถามซ้ำในแชทเดียวกัน
คำถามนี้เลือก “รูปแบบแบ่งหน้าที่” ไม่ใช่ถามว่าใช้สมองเดียวหรือสมองคู่ · ห้ามตีความโหมด 1 ว่า Opus ทำทุกขั้นคนเดียว · ห้ามบังคับโหมด 2 เป็นค่าถาวรทุกงาน

[ด่านก่อนเขียนทุก Phase — ตรวจซ้ำเมื่อขอบเขตจริงเปลี่ยน ไม่ถามราย issue]
- การอนุมัติ branch ใช้ได้กับ `task_id + approval_phase + ขอบเขต + รายการ path` หนึ่งชุด · ทุก issue ใน Phase เดิมใช้สิทธิ์เดียวกันได้ ไม่ต้องขอซ้ำ
- คำสั่งผู้ใช้ก้อนใหม่ การเปลี่ยนหัวข้อ/ขอบเขต/path/branch/SHA หรือหลังย่อบริบทที่ตรวจค่าเดิมไม่ได้ = ต้องตรวจใหม่เสมอ
- ก่อนการแก้ไฟล์ครั้งแรกของทุกงาน และก่อนส่ง AI ตัวเขียนผ่าน Relay ให้รันใหม่: `git branch --show-current` + `git status --short --branch` + ตรวจ branch/งานค้างใน `.project/OverviewProgress.md` + ตรวจ claim เมื่อโปรเจกต์มีระบบจอง
- จัดประเภททุกงานเป็น `CONTINUATION` หรือ `NEW_WRITABLE_TASK` พร้อมหลักฐาน:
  - `CONTINUATION` ใช้ branch เดิมได้เฉพาะเมื่อ task/issue เดิมตรงกัน, branch/PR เดิมมีหลักฐาน, ขอบเขต path เดิมไม่เปลี่ยน และ dirty ทั้งหมดเป็นของงานนั้น
  - ไม่ครบข้อใดข้อหนึ่ง = `NEW_WRITABLE_TASK` · ต้องให้ Worktree Manager แสดง dry-run และรอเจ้าของอนุมัติก่อน `--apply`/เขียน
- คำว่า “ทำต่อ”, “แก้เพิ่ม”, “อีกเรื่อง”, หรือการอยู่ในแชตเดิม ไม่ใช่หลักฐานว่าเป็นงานต่อเดิม
- ก่อนเขียนให้ประกาศ `Phase Write Permit`: task_id / approval_phase / branch / base SHA / allowed paths / owner approval / claim status · ไม่มีหรือค่าเปลี่ยน = ห้ามเขียน
- ถ้าโปรเจกต์ไม่มีระบบ claim ให้ใช้ `hermes-write-permit acquire ...` เป็นตัวล็อกกลางต่อโฟลเดอร์ · ผล `workspace_locked` = มีอีกงานถือสิทธิ์ ห้ามเขียน · ก่อนเขียนแต่ละรอบใช้ `check` และเมื่อจบ/ส่งต่อใช้ `release` · ไม่พบคำสั่งนี้ = ติดตั้งชุด Shortcut ใหม่ ห้ามทำเหมือนมีล็อก
- งานอ่าน ตรวจ อธิบาย และรันทดสอบที่ไม่แก้ไฟล์ไม่ต้องสร้าง branch แต่ถ้าเครื่องมือกำลังจะเขียนไฟล์ให้กลับเข้าด่านนี้ทันที

[Fixed Workspace Policy — กฎเจ้าของ 2026-07-12]
- หนึ่งพนักงาน + หนึ่งโปรเจกต์ = หนึ่งโฟลเดอร์ทำงานประจำที่อยู่ในทะเบียน
- Use New Chat ต้องแสดง current folder + git root + registered folder + folder match + branch + dirty ก่อนรับงานเขียน
- ห้าม Shortcut หรือ AI สร้าง/switch branch/worktree ด้วย Git เอง; งานใหม่ต้องผ่าน Worktree Manager
- ถ้า registered folder ไม่มี = `MISSING_REGISTERED_WORKSPACE` · หยุดและรายงานเจ้าของ ห้าม fallback ไป main/คนอื่นหรือสร้างใหม่
- งานใหม่สร้างเป็น task worktree ใต้ registered root หลัง dry-run ผ่านและเจ้าของอนุมัติ
- แต่ละ task แยกโฟลเดอร์; task หนึ่งมี writer lease ได้ครั้งละหนึ่งเครื่อง ส่วน reviewer อ่านอย่างเดียว

กฎบังคับ:
- เจ้าของงานอาจเป็น non-dev → อธิบายภาษาคน · เสนอคำสั่งก๊อปวางได้ · ห้ามถามว่า "ใช้ test ตัวไหน" ค้นเอง (Schema §5)
- ห้ามตอบ "พร้อมรับคำสั่ง" ก่อนตรวจสถานะจริง
- ห้ามเดา project/worktree/branch/dirty/remote/VPS/CI · ตรวจด้วย command จริงก่อนสรุป
- ทุกค่าจาก command จริง · แนบ Evidence (timestamp/host/cwd/commands) ท้ายรายงาน
- ผู้ใช้พิมพ์ไทย ตอบไทย แปลศัพท์ทันที
- ห้ามแก้ไฟล์ก่อนส่ง startup report เว้นแต่เจ้าของสั่งทำต่อเอง
- redact secret ใน remote URL เสมอ (https://***@...)

แหล่ง config (ห้าม fallback ไปค่า default ของโปรเจกต์อื่น):
project config > repo runbook > AGENTS.md/.hermes context > ถามเจ้าของ

ขั้นตอน (ทำเท่าที่โปรเจกต์มี):

0. อ่านความจำ + เรียนรู้ทั้งโปรเจกต์ (บังคับ)

0a. ความจำรอบล่าสุด (แก้ AI ลืมงานข้ามแชท · Schema v1.2: อ่าน `.project/` ที่เดียวต้องไปต่อได้):
- เปิด `.project/OverviewProgress.md` (เช็กป้าย `> memory-schema:` บรรทัดแรก + อ่าน 4 หัวข้อบนสุด: สถานะล่าสุด/งานถัดไป/ข้อห้าม/งานค้าง-ส่งต่อ) → อ่านตามสารบัญบังคับ: `.project/plan.md` + `.project/decisions.md` → session log ล่าสุด (ตาม `latest-close.md` ของ staff ตัวเอง — Schema §6)
- ถ้าพบความจำเก่า, memory-audit, QA/QC หรือโหมดงานทีม ให้อ่านเฉพาะหัวข้อที่ตรงเงื่อนไขใน `use-new-chat-conditional-gates.md` · ห้ามโหลดไฟล์นั้นเมื่อไม่เข้าเงื่อนไข
- memory guard: ยืนยันว่า memory ที่อ่านเป็นของ project + worktree นี้จริงก่อนใช้ · ไม่ใช่ = ไม่ใช้ รายงาน
- หา latest-close/handoff ไม่เจอ = บอกว่าไม่พบความจำรอบก่อน ไม่เดาว่าทำอะไรไป
- reality-overrides-token (Schema §2): เทียบ token รอบก่อนกับของจริง (git/CI/SHA)
  - token = `CLOSED_CLEAN` แต่เจอ dirty / SHA ไม่ตรง / CI แดง → เชื่อของจริง + ตีธงว่ารอบก่อนปิดไม่ตรงจริง
  - `CLOSED_WITH_PENDING` → อ่านงานค้าง + claimed ก่อนเริ่ม
  - `NEED_OWNER_ACTION_BEFORE_CLOSE` → รอบก่อนยังไม่ปิดจริง เตือนเจ้าของ

0b. เรียนรู้ทั้งโปรเจกต์ (แก้ AI ไม่อ่านภาพรวม):
- อ่าน + สรุปสั้น: `AGENTS.md` / runbook / architecture / `.project/decisions.md` (สะสมทั้งหมด ไม่ใช่แค่ decision ล่าสุด) / ไฟล์ source of truth ตามสารบัญใน OverviewProgress
- สรุปออกมาให้เจ้าของเห็นว่าโปรเจกต์นี้คืออะไร / โครงหลัก / ข้อตกลงสำคัญที่ตัดสินไปแล้ว

0c. สรุปบังคับ 3 บรรทัด (ไม่ครบ = ยังไม่เริ่มงาน):
- Process อยู่ขั้นไหน
- งานล่าสุดถึงไหน (commit/PR/SHA ล่าสุด — เทียบ `git log -5` + branch + PR/timestamp ไม่เชื่อ handoff อย่างเดียวเพราะอาจ stale)
- สิ่งที่ต้องทำต่อ 1 ข้อ

1. Project Detection
- project path + repo name · execution target (local/VPS — ถ้า shell local แต่ target VPS รายงานทั้งสอง ห้ามบอก "อยู่บน VPS แล้ว") · context ที่เจอ (AGENTS.md/.hermes/*)

2. Git Gate
- `git rev-parse --show-toplevel` / `branch --show-current` / `status --short --branch` / `worktree list` → แปลผล clean/dirty
- ไม่ใช่ git repo / nested / monorepo ไม่ชัด → `NEED_OWNER_INPUT` ห้ามเดา

3. Remote Gate
- ตรวจ remote · หลาย remote = รายงาน candidate ทั้งหมด แล้วถามอันไหนหลัก ห้ามเลือกมั่ว · เทียบ local HEAD กับ remote หลัก

4. Staff / Worktree Routing Gate — [เฉพาะเมื่อ project config ใช้ worktree/ownership routing]
- ระบุ staff id (nat/may/mind) · ไม่รู้ → ถาม ห้าม fallback ไป worktree คนอื่น
- ตรวจว่า staff id + project มี worktree จริง (รัน `scripts/hermes_worktree_route.py` ถ้ามี) · ไม่เจอ = รายงาน missing หยุดก่อนแก้
- โปรเจกต์ที่ config ไม่ระบุ routing → ข้ามขั้นนี้

5. Quality Gate Detection (ค้นเอง · เจ้าของไม่ต้องบอก)
- ค้นว่ามี gate อะไร (Schema §5: package.json/Makefile/pyproject/CI) → รายงานว่าโปรเจกต์ตรวจงานด้วยอะไร
- ยังไม่ต้องรันตอนเปิด (รันตอน Close/ก่อน commit) แต่ต้องรู้ว่ามีอะไร เพื่อบอกเจ้าของได้ว่าจะ verify งานยังไง
- ไม่พบ gate = รายงานตรง ๆ ว่าโปรเจกต์นี้ไม่มีตัวตรวจอัตโนมัติ (ความเสี่ยงที่ต้องรู้)

6. VPS / Runtime + Deploy Gate — [capability-based · โปรเจกต์ deploy แบบ CI/CD on-merge]
- service/endpoint ตาม config เท่านั้น (ห้าม port-scan กว้าง) · ไม่มี VPS → N/A
- เทียบ deployed SHA กับ main/branch: prod ตามโค้ดล่าสุดทันไหม
  - ดู CI run ล่าสุดของ main (เช่น `gh run list` สำหรับ GitHub / `glab ci status` หรือ pipeline API สำหรับ GitLab) เขียวไหม · live SHA = HEAD ของ main ไหม
  - main นำหน้า live SHA = มีโค้ดที่ merge แล้วแต่ยังไม่ขึ้น prod → flag
- ตรวจไม่ได้ = ระบุว่าไม่ได้ตรวจเพราะอะไร ห้ามบอกว่าใช้ได้

7. Risk Gate
- dirty files → แยกไฟล์แก้ค้าง vs ไฟล์ใหม่
- งานถัดไปเป็น feature/deploy/migration/security/multi-file → ต้อง audit ก่อนแก้
- worktree dirty + งานเป็น feature ใหม่ → หยุด ห้าม stash/ย้ายไฟล์/สร้าง worktree หนี · ปิดหรือส่งต่องานเดิมก่อน

[Hermes Agent เท่านั้น — ตรวจจาก repo marker ไม่ใช่ชื่อโฟลเดอร์]
- ถ้า config ยืนยันเป็น Hermes: Mac = local controller · VPS target ตาม runbook · ใช้ ssh/enter script ของโปรเจกต์ ไม่ hardcode IP/path
- branch จริงเป็นชื่อพนักงาน (เช่น nat) → แสดงเป็น "Update Feature (branch จริง: nat)" · ห้ามเปลี่ยนชื่อ branch git จริง

[New Feature Branch Gate — บังคับสร้าง branch ใหม่ก่อนแก้โค้ดฟีเจอร์ใหม่]
"ฟีเจอร์ใหม่" = เพิ่ม behavior/UI/command/tool/config/integration ใหม่ หรือเจ้าของใช้คำ build/add/create/implement/ทำ
"งานต่อเดิม" ต้องมีหลักฐาน: handoff ระบุ branch เดิม / task-PR เดิม / commit ล่าสุดตรง · ไม่มีหลักฐาน = ถือเป็นงานใหม่
ก่อนแก้โค้ดเช็ก:
- branch เป็น main/master/develop/shared/release/ของคนอื่น + ฟีเจอร์ใหม่ → STOP
- detached HEAD → STOP เสมอ
- dirty worktree → STOP ใน task นั้น · ห้ามย้าย changes/stash · รายงานว่าเป็นงานใครและปิด/พัก/ส่งต่อ หรือเสนอ task ใหม่ผ่าน Manager
สร้าง task worktree ใหม่ผ่าน Manager เท่านั้น: branch มาตรฐาน `task/<staff-id>/<task-id>-<slug>` · ชื่อ/path ชนให้ Manager บล็อก · ต้องแสดง dry-run และขอยืนยันก่อน `--apply`
ยกเว้น: อ่านอย่างเดียว/audit/review/explain/test/grep/git · typo/docs เล็ก (เจ้าของยืนยัน) · hotfix → `hotfix/<slug>` · งานต่อ PR เดิมที่มีหลักฐาน

[Team-Safe Flow] ถ้า config เป็นงานทีม/มี routing ให้อ่านหัวข้อ Team Claim Gate ใน `use-new-chat-conditional-gates.md` แล้วบังคับจอง path ก่อนทุกงานเขียน · งานอ่านอย่างเดียวข้ามได้

รูปแบบคำตอบบังคับ:

New Chat Startup Report
Project: path / repo / execution target / local controller / context loaded
Workspace: current folder / git root / registered folder / folder match
Project Overview (3 บรรทัด): โปรเจกต์คืออะไร / โครงหลัก / ข้อตกลงสำคัญ
Memory: process ขั้นไหน / งานล่าสุดถึงไหน(SHA) / ต้องทำต่อ 1 ข้อ / token รอบก่อน(+ธงถ้าไม่ตรงจริง)
Git: worktree / branch / dirty / HEAD / remote match (หรือ candidate)
Routing: staff id / target worktree / route status (หรือ N/A)
AI Relay: brain / coder / reviewer readiness / account state / version compatibility / queue state
Branch Gate: current branch / new-or-continuation / proposed branch (branch only; never a new worktree)
Quality Gate: gate ที่ค้นเจอ (หรือ "ไม่พบตัวตรวจอัตโนมัติ")
VPS / Deploy: service / endpoint / live SHA vs main (หรือ N/A)
Evidence: timestamp / host / cwd / commands ที่รัน
Readiness: READY / BLOCKED / NEED_OWNER_INPUT + เหตุผล + งานที่ควรทำต่อ

ตัวอย่าง READY: Readiness: READY · git clean บน feature/login ตรง origin · main live SHA ตรง CI เขียว · พร้อมรับงานต่อ
ตัวอย่าง BLOCKED: Readiness: BLOCKED · registered folder หายหรือมี dirty จากงานอื่น · ห้ามเริ่มงานเขียน
ตัวอย่าง NEED_OWNER_INPUT: Readiness: NEED_OWNER_INPUT · มี 2 remote ยังไม่รู้อันไหนหลัก + ไม่รู้ staff id · ขอข้อมูลที่ขาดก่อน

ข้อห้าม: ตอบแค่ "พร้อมรับคำสั่ง" · บอก clean โดยไม่รัน git status · บอก deploy ใช้ได้โดยไม่เทียบ SHA/CI · เชื่อ token โดยไม่เทียบของจริง · ถามเจ้าของว่าใช้ test ตัวไหน · สรุปด้วยศัพท์เทคนิคโดยไม่แปล
```

## Worktree Lifecycle v1 (มีผลเหนือกฎ fixed-folder รุ่นเก่า)

อ่าน `worktree-lifecycle-contract.md` ก่อนใช้ Prompt นี้ · task เดิมใช้ `hermes worktree status/enter`; task ใหม่ใช้ `open` แบบ dry-run แล้ว `--apply` หลังอนุมัติ · ห้าม Shortcut สร้าง branch/worktree เอง · รายงาน task/machine/writer/runtime และ Decision Token WTL ทุกครั้ง

## Changelog

- v2.7 (2026-07-14): เพิ่ม Fresh Close Receipt ลดการตรวจ CI/VPS ซ้ำแต่ยังบังคับ Git ขั้นพื้นฐานสด · ถามโหมด AI 1/2 เฉพาะเมื่อจะใช้หลาย AI · เปลี่ยน Write Permit เป็นสิทธิ์ระดับ Phase ไม่ถามราย issue
- v2.6 (2026-07-13): ส่งกฎหยุดหลังตรวจไม่ผ่าน 2 รอบให้ AI Relay · บังคับแยกปัญหาและเปลี่ยนวิธีตรวจแทนการเรียกผู้ตรวจเดิมรอบที่ 3
- v2.5 (2026-07-12): เพิ่ม Hook Health Gate ตรวจสด 3 ด่าน ป้องกันกรณีมีไฟล์ Hook แต่ถูกปิดหรือปล่อยผ่าน
- v2.4 (2026-07-12): เพิ่มด่านก่อนเขียนทุกงานหลังพบปัญหา branch ถูกอนุมัติครั้งเดียวแล้วนำไปใช้ซ้ำจนไฟล์ทับกัน · สิทธิ์ผูก task+scope+paths · งานใหม่ทุกก้อนต้องตรวจ branch/dirty/claim และขออนุมัติ branch ใหม่ · แก้ตัวอย่าง BLOCKED ที่ใช้ token ผิด
- ประวัติก่อน v2.4 อยู่ใน Git history และสรุปเงื่อนไขใน `use-new-chat-conditional-gates.md` เพื่อไม่ให้ AI อ่านประวัติที่ไม่ใช้ทุกแชต

## Graph Links

- Parent hub: [[skills/prompt-shortcuts/Prompt Shortcuts|Prompt Shortcuts]]
- Registry: [[ai-context/prompt-shortcut-registry|Prompt Shortcut Registry]]
- Schema: [[skills/prompt-shortcuts/references/memory-schema|Memory Schema v1.1]]
- Pair: [[skills/prompt-shortcuts/references/use-close-chat|Use Close Chat]]
- Related: [[ai-context/session-start-contract|Session Start Contract]]
- คู่มือเจ้าของงาน: [[skills/prompt-shortcuts/references/memory-system-owner-guide-th|ระบบความจำข้ามแชท — คู่มือเจ้าของงาน]]
