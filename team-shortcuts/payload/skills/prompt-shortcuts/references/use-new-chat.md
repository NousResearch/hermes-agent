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
version: "2.0"
updated: 2026-07-10
schema: memory-schema-v1.2
pairs_with: use-close-chat >= 2.2
---

# Use New Chat (v2.0 · 2026-07-10)

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
- [Migration Gate — เจอของเก่า ต้องย้ายก่อนเริ่มงาน (Schema §1b)]: ถ้าเจอ `.hermes/plan.md` / `.hermes/active.md` / `.hermes/decisions.md` / `handoff.md` ที่ root ที่ยังมีเนื้อหาจริง → ย้ายเข้า `.project/` ตามผัง §1b + ทำไฟล์เก่าเป็น stub ชี้ทางใหม่ (ห้ามลบ) + grep แก้จุดอ้างทางเก่าในไฟล์กฎราก/`hermes.project.yaml` + รายงานเจ้าของว่าย้ายแล้ว · ห้ามเขียนความจำใหม่ลงที่เก่าเด็ดขาด
- [ด่านไฟล์เข้า git จริง (Schema §1b · เพิ่ม 2026-07-05)]: หลังย้าย/สร้างไฟล์ `.project/` ทุกครั้ง รัน `git check-ignore -v .project/<ไฟล์>` (ต้องไม่เจอ) + `git ls-files .project/` (ต้องเห็นครบ) · ถ้า `.gitignore` ซ่อน (เช่นกฎกวาด `*.md`) → เจาะช่องอนุญาต `!.project/` + `!.project/**` แล้วตรวจซ้ำ + รายงานผลใน startup report · ไฟล์ไม่เข้า git = ความจำหายข้ามเครื่อง ห้ามนับว่าย้ายเสร็จ
- [ตัวเทียบความจำกับของจริง — GRD-P2 2026-07-08]: ถ้า repo มี `scripts/memory-audit/memory_audit.py` → รัน `python3 scripts/memory-audit/memory_audit.py` 1 ครั้ง แล้วรายงานผลใน startup report (exit 1 = ความจำโกหก/ไฟล์หลุด git ต้องเคลียร์ก่อนเริ่มงาน · exit 2 = มีเตือน อ่านแล้วไปต่อได้) · repo ที่ไม่มีสคริปต์ = ข้าม
- [QA/QC Center — เพิ่ม 2026-07-10 คู่กับ Use QA QC ≥ v1.0]: ถ้าโปรเจกต์มี `.project/qaqc-scan.md` → อ่าน 4 ส่วน: "สถานะรายหมวด" + "ตารางปัญหาคงค้าง" + **"แผนแก้ (QQF-*)" + "ประวัติรอบล่าสุด"** แล้วรายงานใน startup report (สแกนแล้วกี่หมวด/16 · ค้างแก้กี่ข้อ ระดับไหน · งานแก้ QQF ค้างสถานะไหน มีแถว gate-run แล้วยัง · รอบล่าสุดเมื่อไหร่) · ไม่มีไฟล์ = ข้าม ไม่ต้องพูดถึง
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
- worktree dirty + งานเป็น feature ใหม่ → หยุดถาม/เลือก worktree ปลอดภัยกว่า

[Hermes Agent เท่านั้น — ตรวจจาก repo marker ไม่ใช่ชื่อโฟลเดอร์]
- ถ้า config ยืนยันเป็น Hermes: Mac = local controller · VPS target ตาม runbook · ใช้ ssh/enter script ของโปรเจกต์ ไม่ hardcode IP/path
- branch จริงเป็นชื่อพนักงาน (เช่น nat) → แสดงเป็น "Update Feature (branch จริง: nat)" · ห้ามเปลี่ยนชื่อ branch git จริง

[New Feature Branch Gate — บังคับสร้าง branch ใหม่ก่อนแก้โค้ดฟีเจอร์ใหม่]
"ฟีเจอร์ใหม่" = เพิ่ม behavior/UI/command/tool/config/integration ใหม่ หรือเจ้าของใช้คำ build/add/create/implement/ทำ
"งานต่อเดิม" ต้องมีหลักฐาน: handoff ระบุ branch เดิม / task-PR เดิม / commit ล่าสุดตรง · ไม่มีหลักฐาน = ถือเป็นงานใหม่
ก่อนแก้โค้ดเช็ก:
- branch เป็น main/master/develop/shared/release/ของคนอื่น + ฟีเจอร์ใหม่ → STOP
- detached HEAD → STOP เสมอ
- dirty worktree → STOP ถามจะพา changes ไป branch ใหม่ / stash / ให้เจ้าของจัดการ
สร้าง branch ใหม่: ทีม/VPS = `worktree/<staff>/<slug>` · local = `feature/<slug>` (มี ticket id) · ชื่อชน = เสนอชื่อใหม่ ไม่ทับ · เสนอ+ขอยืนยันก่อนสร้าง
ยกเว้น: อ่านอย่างเดียว/audit/review/explain/test/grep/git · typo/docs เล็ก (เจ้าของยืนยัน) · hotfix → `hotfix/<slug>` · งานต่อ PR เดิมที่มีหลักฐาน

[Team-Safe Flow — กันงานทับซ้อน/dirty เมื่อหลายคน+หลาย AI]
ทำเมื่อ config เป็นงานทีม/มี worktree routing · งานแค่ถาม-ไม่แก้ ข้ามได้
1. ระบุ staff id → route ไป worktree ตัวเองด้วย `hermes_worktree_route.py` · ห้ามไป worktree คนอื่น
2. อ่านไฟล์ประสาน (อ่านอย่างเดียว): `.project/OverviewProgress.md` (หัวข้อ งานค้าง/ส่งต่อ) → `.project/decisions.md`
3. เช็กการจอง: `hermes_worktree_route.py claim list` (ห้ามอ่าน/เขียน markdown จองเอง)
   - path "ซ้อนทับ" claim คนอื่นไหม (เทียบ path overlap เช่น `gateway/` ซ้อน `gateway/run.py`) · ชน → STOP รายงานใครจอง path ไหน
   - claim หมดอายุ → `STALE_CLAIM_REVIEW` · ไฟล์เสี่ยง (config/migration/deploy/secret) ห้ามยึดอัตโนมัติ ถามเจ้าของ
4. เช็ก dirty บน worktree ตัวเอง: dirty + ไม่ใช่งานตัวเองค้าง → STOP
5. จองก่อนแก้: `claim acquire` (staff/project/worktree/branch/issue/paths/expires_at)
   - claim = "ตั้งใจจะเริ่มแก้" · แหล่งจริงของ "แก้อะไรไปแล้ว" = git status/branch — เทียบทั้งสองเสมอ
6. ปลด claim เมื่อจบ: หลัง commit/handoff → `claim release` หรือเปลี่ยนเป็น handoff
ไม่มี claim/registry → อ่าน handoff/active.md ได้ แต่รายงาน claim เป็น unverified ห้ามทำเหมือนล็อกแล้ว

รูปแบบคำตอบบังคับ:

New Chat Startup Report
Project: path / repo / execution target / local controller / context loaded
Project Overview (3 บรรทัด): โปรเจกต์คืออะไร / โครงหลัก / ข้อตกลงสำคัญ
Memory: process ขั้นไหน / งานล่าสุดถึงไหน(SHA) / ต้องทำต่อ 1 ข้อ / token รอบก่อน(+ธงถ้าไม่ตรงจริง)
Git: worktree / branch / dirty / HEAD / remote match (หรือ candidate)
Routing: staff id / target worktree / route status (หรือ N/A)
Quality Gate: gate ที่ค้นเจอ (หรือ "ไม่พบตัวตรวจอัตโนมัติ")
VPS / Deploy: service / endpoint / live SHA vs main (หรือ N/A)
Evidence: timestamp / host / cwd / commands ที่รัน
Readiness: READY / BLOCKED / NEED_OWNER_INPUT + เหตุผล + งานที่ควรทำต่อ

ตัวอย่าง READY: Readiness: READY · git clean บน feature/login ตรง origin · main live SHA ตรง CI เขียว · พร้อมรับงานต่อ
ตัวอย่าง BLOCKED: Readiness: NEED_OWNER_INPUT · มี 2 remote ยังไม่รู้อันไหนหลัก + ไม่รู้ staff id · ขอ 2 ข้อนี้ก่อน

ข้อห้าม: ตอบแค่ "พร้อมรับคำสั่ง" · บอก clean โดยไม่รัน git status · บอก deploy ใช้ได้โดยไม่เทียบ SHA/CI · เชื่อ token โดยไม่เทียบของจริง · ถามเจ้าของว่าใช้ test ตัวไหน · สรุปด้วยศัพท์เทคนิคโดยไม่แปล
```

## Changelog

- v2.0 (2026-07-10): เพิ่มขั้นอ่าน QA/QC Center (`.project/qaqc-scan.md` — capability-based เฉพาะโปรเจกต์ที่มีไฟล์) รายงานหมวดที่สแกนแล้ว/ค้างแก้ใน startup report · คู่กับ Use QA QC v1.0 (แผน QAQC-P4-I2)
- v1.9 (2026-07-08): เพิ่มขั้นรัน memory-audit ตอนเปิดแชท (repo ที่มี `scripts/memory-audit/`) — ตัวเทียบความจำกับของจริงจากแผน GRD-P2 · จับ SHA ถูก revert เงียบ / ไฟล์ความจำหลุด git / เลขงานกำพร้า
- v1.8 (2026-07-05): เกาะ Memory Schema v1.2 — ความจำทำงานต่ออยู่ `.project/` ที่เดียว (OverviewProgress 4 หัวข้อบนสุด + plan.md + decisions.md) · เพิ่ม Migration Gate ย้ายไฟล์เก่าจาก `.hermes/`/root เป็น stub + แก้จุดอ้างทางเก่าในรอบเดียว · schema ไม่ตรง = ห้ามเขียนความจำ · ปฐมเหตุ: AI อ่านไม่ครบ 2 โฟลเดอร์แล้วทำงานมั่วซ้ำหลาย project (คำสั่งเจ้าของ 2026-07-05)
- v1.7 (2026-06-26): เขียนใหม่ทั้งฉบับให้เกาะ Memory Schema v1.1 · เพิ่มขั้น 0b เรียนรู้ทั้งโปรเจกต์ (อ่านคู่มือ/โครงสร้าง/decisions ทั้งหมด สรุปภาพรวม) · ยก reality-overrides-token เป็นกฎชัด · เพิ่ม memory guard (ยืนยัน memory เป็นของ project+worktree นี้) · เพิ่ม Quality Gate Detection ตอนเปิด · เพิ่ม Deploy Gate เทียบ live SHA กับ main · latest-close แยก pointer ต่อ staff · เพิ่ม Project Overview ใน output · เช็ก schema version
- v1.6 (2026-06-24): บังคับสรุปบริบทเก่า 3 บรรทัด + เทียบ git log · เพิ่ม New Feature Branch Gate (ฟีเจอร์ใหม่บน main/shared/detached/dirty = STOP)
- v1.5 (2026-06-24): ปิดวงจรความจำคู่กับ Use Close Chat · ขั้น 0 อ่าน handoff + session log + decision token รอบก่อน
- v1.4 (2026-06-24): เพิ่ม Team-Safe Flow กันงานทับซ้อนเมื่อหลายคน+หลาย AI · จองงานผ่าน claim acquire/release/list
- v1.3 (2026-06-24): เปลี่ยนเป็น capability-based gate · เลิก hardcode IP/port/path · เพิ่ม Evidence · Readiness token
- v1.2 (2026-06-06): เวอร์ชันก่อนหน้า

## Graph Links

- Parent hub: [[skills/prompt-shortcuts/Prompt Shortcuts|Prompt Shortcuts]]
- Registry: [[ai-context/prompt-shortcut-registry|Prompt Shortcut Registry]]
- Schema: [[skills/prompt-shortcuts/references/memory-schema|Memory Schema v1.1]]
- Pair: [[skills/prompt-shortcuts/references/use-close-chat|Use Close Chat]]
- Related: [[ai-context/session-start-contract|Session Start Contract]]
- คู่มือเจ้าของงาน: [[skills/prompt-shortcuts/references/memory-system-owner-guide-th|ระบบความจำข้ามแชท — คู่มือเจ้าของงาน]]
