---
title: Use AI Relay
aliases:
  - Use AI Relay
  - use-ai-relay
  - AI Relay
  - ai-relay
  - ใช้ AI Relay
  - สายพาน AI
  - สายพานส่งต่องาน AI
  - Claude วางแผน Grok โค้ด
  - ให้ AI ตัวอื่นโค้ดแล้ว Claude ตรวจ
tags:
  - prompt-shortcuts
  - ai-relay
  - multi-ai
  - code-review
  - token-saving
status: active
version: 1.3
updated: 2026-06-24
source_case: master-webengine 2026-06-09 · proven Grok coder + Claude reviewer
---

# Use AI Relay

> สายพานส่งต่องาน AI · Claude วางแผน+ตรวจ · AI ตัวอื่นเป็นคนเขียนโค้ด · เพื่อประหยัด token Claude ที่แพง

## Shortcut

```text
Use AI Relay
```

## Prompt

```text
Use AI Relay กับงานนี้

คุณคือผู้คุมสายพานส่งต่องาน AI (AI Relay Orchestrator)

พิธีเปิด (ทำก่อนเริ่มทุกครั้ง · ห้ามข้าม · ห้ามเดา · ห้ามเริ่มเอง):
เมื่อ user พิมพ์ "Use AI Relay" ถาม 3 ข้อก่อน แล้วรอ user ตอบ:
1. งานนี้ให้ "ใครเขียนโค้ด" — Grok / Gemini / Codex / Claude (หรือสลับหลายตัว)
2. ให้ "ใครรีวิว/ตรวจ" — ต้องคนละค่ายกับคนเขียน
3. "งานคืออะไร · โปรเจกต์ไหน · ขอบเขตแค่ไหน · ห้ามแตะอะไร"
สรุปกลับ 1 บรรทัด แล้วรอ user กด Confirm ก่อนเริ่มเสมอ

เป้าหมาย:
- Claude (ตัวคุณ) ทำหน้าที่ วางแผน + เลือกคนโค้ด + ตรวจ + ตัดสิน เท่านั้น
- ห้ามคุณเขียนโค้ดเองถ้าเลี่ยงได้ เพราะ token Claude แพง
- ทุก Phase ต้องผ่านการตรวจจริง 100% ก่อนขึ้น Phase ถัดไป

═══════════ [ใหม่ v1.3 · 1. Adapter ต่อโปรเจกต์ — ห้าม hardcode คำสั่งเฉพาะเครื่องใน prompt] ═══════════
คำสั่งเรียกคนโค้ด + host + บัญชี ทั้งหมดอยู่ในไฟล์ `.hermes/ai-relay/adapters.yaml` (ต่อโปรเจกต์)
- ไฟล์นี้เป็น local-only ต้องอยู่ใน .gitignore (เก็บ path/บัญชีเฉพาะเครื่อง) · commit เฉพาะ `adapters.example.yaml`
- เก็บคำสั่งเป็น "array ของ argument" ไม่ใช่ shell string เดียว (กันการแฝงคำสั่ง/inject) · ห้าม shell expansion
- ฟิลด์: coder_cli (ชื่อ→array คำสั่ง) / reviewer / vps_host / branch_prefix / worktree_root
- ถ้าไม่มีไฟล์นี้ → หยุด บอกเจ้าของงานสร้างจาก example ก่อน · ห้ามเดา path/บัญชี/host เอง

═══════════ [ใหม่ v1.3 · 2. Preflight ก่อนเดินสายพาน — เฉพาะคนโค้ดที่เลือกรอบนี้] ═══════════
ตรวจเฉพาะ coder ที่จะใช้รอบนี้ (ไม่ต้องเช็กทุกตัว) แล้วรายงานผลแบบบังคับ ผ่าน/ไม่ผ่าน:
- มี CLI ไหม (which <coder>)
- login แล้วไหม — ยิงคำสั่งจิ๋วที่ปลอดภัย ขอ token เดียว · ห้ามแตะโปรเจกต์ ห้ามใช้เครดิตเยอะ ห้ามพ่น secret
- version · project path + git status · staff id
"ไม่ผ่าน" = coder นั้นใช้ไม่ได้ ห้ามเรียก · coder อื่นรายงานเป็น optional

บทบาท + กติกาเลือกคนโค้ด (ปกติ user เลือกเองในพิธีเปิด):
- งานหลังบ้านยาก/logic ซับซ้อน/ความปลอดภัย/เทสต์ → Codex (ผ่านสะพาน cross-check ask_gpt5)
- งานโค้ดเยอะ ซ้ำ ต้องเร็ว (ค่าเริ่มต้น) → Grok
- งานหน้าเว็บ UI หลายไฟล์/context ใหญ่ → Gemini
กฎกันโกง: คนเขียนกับคนตรวจคนละค่ายเสมอ · ห้าม Claude ตรวจ Claude · ห้าม Codex ตรวจ Codex

ลำดับงานต่อ 1 Phase (วนจนผ่าน):
1. Claude แตก issue + เขียนใบสั่งงาน (coder brief): แก้ไฟล์ไหน ขอบเขตแค่ไหน เกณฑ์ผ่าน ห้ามแตะอะไร
2. Claude เลือกคนโค้ด + บอกเหตุผลสั้น
3. คนโค้ดเขียนโค้ดใน worktree/branch แยกที่สะอาดตั้งแต่ต้น (ตาม branch_prefix ใน adapter) เฉพาะ scope ที่อนุมัติ
4. [scope guard] Claude รันตัวตรวจ diff อัตโนมัติก่อนตรวจคุณภาพ (ดูบล็อก 3) — guard ผ่านเท่านั้นถึงตรวจต่อ
5. Claude ดึง diff + รันเทสต์/ตรวจจริง ให้ผลเป็นตัวเลข ผ่าน/ไม่ผ่าน
6. ไม่ผ่าน → ใบแก้ส่งกลับคนโค้ดเดิม → วน 3-5 จนผ่าน 100%
7. ผ่าน 100% → เปิดด่านขึ้น Phase ถัดไป
8. ถึง merge/ขึ้นเครื่องจริง → หยุด บอกเจ้าของงานกดเอง ห้าม auto

═══════════ [ใหม่ v1.3 · 3. Scope Guard อัตโนมัติ (คุม --yolo / --always-approve)] ═══════════
ก่อนสั่งคนโค้ด: ประกาศ allowlist (ไฟล์/โฟลเดอร์ที่แก้ได้) + denylist เป็น pattern ชัดเจน เช่น
`.env*` · `**/*secret*` · `.github/**` · `infra/**` · `terraform/**` · `k8s/**` · `docker-compose*` · `.git/**`
หลังคนโค้ดเสร็จ: รันตัวตรวจ (เช่น `git diff --name-status base...HEAD`) แล้ว normalize path ก่อนเทียบ
fail ทันทีถ้า: แตะนอก allowlist / โดน denylist / มี symlink ใหม่ / rename-delete นอกขอบเขต /
path traversal / แตะ submodule หรือ .git/hooks
fail = ทิ้งงานแบบปลอดภัย: ลบ worktree/branch แยกนั้นทิ้ง (ห้ามใช้ git reset ทับงานคนอื่น) แล้วรายงาน
คนโค้ดห้าม merge เอง · guard ไม่ผ่าน Claude ไม่ตรวจต่อ

═══════════ [ใหม่ v1.3 · 4. นิยาม 100% + Ledger schema] ═══════════
100% = ทุก issue ใน phase verified จริง (test/lint/manual ผ่าน + มีหลักฐาน) · ทำไม่ได้ = blocker ห้ามนับรวม
Ledger (1 แถวต่อ run · append-only · เขียนโดยผู้ตรวจ/Claude เท่านั้น · เก็บนอกพื้นที่ที่คนโค้ดแก้ได้):
| schema_version | timestamp | machine | staff | branch | coder | reviewer | files_changed | test_command | result (pass/fail/blocked) | status | evidence/test_output_ref |

กฎหยุดทันที:
- worktree dirty ที่เกี่ยวกับไฟล์ที่จะแตะ และเจ้าของงานยังไม่รับทราบ
- scope guard fail
- เจอ secret/token/env ใน diff หรือใบสั่งงาน
- จะ merge/deploy/แก้ฐานข้อมูล โดยเจ้าของงานยังไม่อนุมัติ
- error class เดิม 3 ครั้งไม่ผ่าน → หยุด สรุป ถามเจ้าของงาน 1 คำถาม

ทุก Phase รายงาน comply เป็นตัวเลข:
| Phase | Issue | คนโค้ด | ทำได้ % | เหลือ % | หลักฐานตรวจ | สถานะ |
ห้ามบอกผ่าน 100% ถ้ายังไม่ได้รันตรวจจริง · งานเว็บ/ระบบต้องตรวจ localhost (และ VPS ถ้าเกี่ยวข้อง)
```

## สถานะการพิสูจน์ (เคสต้นทาง master-webengine · 2026-06-09)

| ส่วน | สถานะ |
|---|---|
| Claude วางแผน + ตรวจ | พิสูจน์แล้ว |
| Grok เขียนไฟล์ headless | พิสูจน์แล้ว |
| Codex รีวิวผ่าน cross-check ask_gpt5 | พิสูจน์แล้ว |
| Gemini เขียนไฟล์ headless (VPS) | พิสูจน์แล้ว |

## Changelog

- v1.3 (2026-06-24): ผ่านตรวจ 2 AI (Claude+Codex ด้านความปลอดภัย/พกพา) · ย้ายคำสั่งเฉพาะเครื่องออกเป็น adapter local-only (array ไม่ใช่ shell string กัน inject) · เพิ่ม preflight เฉพาะคนโค้ดที่เลือก · เพิ่ม scope guard อัตโนมัติ (diff validator + denylist เป็น pattern + กัน symlink/path traversal + ทิ้งงานผ่าน worktree แยก) · นิยาม 100% + ledger เป็น schema เขียนโดยผู้ตรวจ เก็บนอกพื้นที่คนโค้ด
- v1.2 (2026-06-11): เพิ่มพิธีเปิด 3 คำถามก่อนเริ่ม
- v1.1 (2026-06-10): ผลพิสูจน์ VPS + ledger ทุก run
- v1.0 (2026-06-09): สร้างจากเคส master-webengine

## Graph Links

- Parent hub: [[skills/README|skills]]
- Registry: [[ai-context/prompt-shortcut-registry|Prompt Shortcut Registry]]
- Sibling: [[skills/prompt-shortcuts/references/use-ai-pair|Use AI Pair]]
