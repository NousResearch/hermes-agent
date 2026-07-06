---
title: Memory Schema v1.2
aliases:
  - Memory Schema
  - memory-schema
  - สัญญาความจำกลาง
  - schema ความจำข้ามแชท
  - สัญญากลางของทั้งระบบ
tags:
  - prompt-shortcuts
  - memory-schema
  - session-memory
  - context-management
  - lifecycle
status: active
version: "1.2"
updated: 2026-07-05
---

# Memory Schema v1.2 (สัญญากลางของทั้งระบบ · 2026-07-05)

> เปลี่ยนจาก v1.1: **ความจำที่ใช้ทำงานต่อ ย้ายไปรวมที่ `.project/` ที่เดียว** (คำสั่งเจ้าของ 2026-07-05 · แก้โรค "สมองแยกซีก" ที่ AI อ่านไม่ครบ 2 โฟลเดอร์แล้วทำงานมั่ว — เกิดซ้ำหลาย project) · `.hermes/` เหลือเฉพาะไฟล์เครื่องจักร
> เปลี่ยนจาก v1: เดิมคุม New/Close เท่านั้น · v1.1 คุมทั้ง 6 ตัว (เพิ่ม §9–§13 ให้ Act-As/Comply/Continue เกาะร่วม)

แหล่งความจริงร่วมของทั้งระบบ · แก้รูปแบบ/ฟิลด์/นโยบายใด ๆ ให้แก้ที่นี่ที่เดียว แล้ว bump เวอร์ชัน ทุก prompt ต้องอ้างไฟล์นี้

> เป้าหมาย: กันแต่ละ shortcut "พูดคนละภาษา" จน verify/ความจำ/การส่งต่องานพังเงียบ ๆ

---

## 0. ผู้ใช้ระบบเป็น non-dev (กฎครอบทั้งระบบ)

- เจ้าของงานอาจไม่ใช่โปรแกรมเมอร์ → **ห้ามถามว่า "ใช้ test/build ตัวไหน"** AI ต้องค้นจาก repo เอง (§5)
- ทุกคำสั่งที่ให้เจ้าของรัน = คำสั่งเต็มก๊อปวางได้ + บอกว่าจะเกิดอะไรเป็นภาษาคน
- รายงานภาษาไทย แปลศัพท์เทคนิคทันที

## 1. ไฟล์ในระบบ · ใครเขียน · ใครอ่าน

**กฎทองข้อเดียว (ใหม่ v1.2): "เปิดแชทใหม่ อ่าน `.project/` ที่เดียว ต้องทำงานต่อได้ทันที"** — ทุกอย่างที่จำเป็นต่อการไปต่อ (แผน · สถานะ · งานค้าง · การตัดสินใจ) อยู่ `.project/` ทั้งหมด · `.hermes/` เหลือเฉพาะไฟล์เครื่องจักรที่ไม่ต้องอ่านเพื่อไปต่อ

| ไฟล์ | ที่อยู่ | เขียนโดย | อ่านโดย | หน้าที่ |
|---|---|---|---|---|
| `.project/OverviewProgress.md` | repo `.project/` | Close | New, Act-As, Comply, Continue | ภาพรวม + สถานะล่าสุด + งานกำลังทำ/ค้าง + ส่งต่อ (ยุบ `handoff.md` + `active.md` เดิมเข้าไฟล์นี้ไฟล์เดียว) + pointer ไป session log |
| `.project/plan.md` | repo `.project/` | **Act-As** | **Comply, Continue**, New | แผนเฟส+id ที่อนุมัติแล้ว (§9–§10) |
| `.project/decisions.md` | repo `.project/` | Close (**append**) | New, **Act-As**, Continue | log การตัดสินใจสะสม |
| session log | `$ROOT/projects/<project>/session-logs/YYYY-MM-DD-<staff>-<branch>.md` | Close | New | บันทึกเต็มรอบนั้น (รวม ledger จาก Continue) |
| `latest-close.md` | `$ROOT/projects/<project>/` | Close | New | pointer ต่อ staff (§6) |
| `.hermes/ai-relay/` + `.hermes/ledger/` | repo `.hermes/` | โค้ด (relay-call/gate-run) + Continue | relay-report, Close (ยก ledger เข้า session log) | ไฟล์เครื่องจักรล้วน: ตั้งค่าสายพาน · ledger · briefs |
| `AGENTS.md` / runbook | repo | คน/AI | New | บริบทถาวร |

`$ROOT` = `$HERMES_OBSIDIAN_ROOT`: local `/Users/rattanasak/ObsidianVault/HermesAgent` · VPS `/home/linux-nat/ObsidianVault/HermesAgent`

> New/Act-As/Comply/Continue **อ่านอย่างเดียว** ทุกไฟล์ความจำ · มีแต่ Close ที่เขียน memory ถาวร · ข้อยกเว้น: Act-As เขียน `.project/plan.md` ได้เมื่อเจ้าของอนุมัติแผน · Continue เขียน ledger (§13)

### 1b. กติกาย้ายของเก่า (Migration v1.1 → v1.2 · ใช้กับทุก project เก่า 30-40 ตัว)

- **"อ่านได้สองที่ · เขียนที่ใหม่เท่านั้น"**: ทุก shortcut อ่าน `.project/` ก่อน · ถ้าไม่เจอแล้วไปเจอไฟล์เก่า (`.hermes/plan.md` / `.hermes/active.md` / `.hermes/decisions.md` / `handoff.md` ที่ root) → **ย้ายเนื้อหามาที่ใหม่ทันทีในรอบนั้น** (plan→`.project/plan.md` · active+handoff→ยุบเข้า `.project/OverviewProgress.md` · decisions→`.project/decisions.md`) + รายงานเจ้าของว่าย้ายแล้ว
- **ห้ามลบไฟล์เก่า** — แทนด้วย stub (ไฟล์ชี้ทาง) 2-3 บรรทัด: "ย้ายแล้ว → ไปอ่าน `.project/<ชื่อไฟล์>`" · กันสคริปต์/เครื่องมือเก่าที่ยังฝังทางเดิมพังเงียบ (ผลตรวจข้ามค่าย 2026-07-05)
- **ย้ายแล้วต้องแก้จุดอ้างทางเก่าใน repo รอบเดียวกัน**: grep หา `.hermes/plan.md`, `.hermes/active.md`, `.hermes/decisions.md`, `handoff.md` แล้วแก้ให้ชี้ที่ใหม่ — อย่างน้อยต้องครอบ: ไฟล์กฎที่ราก (`AGENTS.md`/`CLAUDE.md`/`GEMINI.md`/`QWEN.md`) + `.project/OverviewProgress.md` + `.project/hermes.project.yaml` (`required_files`) · จุดอ้างค้าง = ธงแดง รายงานเจ้าของ (ผลตรวจ Grok 2026-07-05)
- **ด่านไฟล์เข้า git จริง (เพิ่ม 2026-07-05)**: หลังสร้าง/ย้ายไฟล์ `.project/` ต้องรัน `git check-ignore -v` กับทุกไฟล์ (ต้องไม่เจอสักไฟล์) + `git ls-files .project/` (ต้องเห็นครบ) · ถ้า `.gitignore` ซ่อนไฟล์ (เช่นกฎกวาด `*.md` ทั้ง repo) → เจาะช่องอนุญาตเพิ่ม `!.project/` + `!.project/**` แล้วตรวจซ้ำ · ไฟล์ไม่เข้า git = ความจำหายข้ามเครื่องเงียบ ๆ = งานยังไม่เสร็จ ห้ามรายงานว่าเสร็จ (บทเรียนจริง 2026-07-05: repo ที่กฎซ่อน `*.md` ทำให้ไฟล์ความจำอยู่เครื่องเดียว)
- ห้ามเขียนไฟล์ความจำใหม่ลง `.hermes/` หรือ root อีกเด็ดขาด (ยกเว้นไฟล์เครื่องจักร §1)
- project ที่ไม่มีโฟลเดอร์ `.project/` → สร้างได้เลย (มาตรฐานกลาง) ไม่ต้องถาม
- เจอทั้งสองที่พร้อมกัน (ไฟล์ใหม่+เก่า) → ที่ใหม่คือของจริง · ของเก่าอ่านเทียบได้แต่ห้ามใช้ทับที่ใหม่ · แจ้งเจ้าของว่าเจอซ้ำ

### 1c. โครงบังคับของ `.project/OverviewProgress.md` (กันไฟล์บวมแล้วสถานะจม)

บรรทัดแรกของไฟล์ต้องเป็นป้ายเวอร์ชัน: `> memory-schema: v1.2` (ทุก shortcut เช็กบรรทัดนี้ก่อนอ่านเนื้อ)
บรรทัดถัดมาต้องเป็น **สารบัญบังคับอ่าน** (manifest): `> อ่านตามลำดับ: plan.md → decisions.md → <ไฟล์ source of truth ของ project>` — กัน AI อ่านแค่ไฟล์เดียวใน `.project/` แล้วภาพขาด (ผลตรวจ Grok 2026-07-05)
หัวข้อคงที่ 4 หัวข้อต้องอยู่ **บนสุด** เสมอ · ประวัติยาวลงล่างเท่านั้น:

1. `## สถานะล่าสุด` — process ขั้นไหน / commit-SHA ล่าสุด / gate ล่าสุดผ่านไหม
2. `## งานถัดไป` — 1-3 ข้อ เรียงตามลำดับ (แชทใหม่เริ่มจากข้อแรกได้ทันที)
3. `## ข้อห้าม/กติกาล็อก` — สิ่งที่ AI ห้ามทำในโปรเจกต์นี้ (เช่น wireframe = ดีไซน์จบ)
4. `## งานค้าง/ส่งต่อ` — claimed + งานรอเจ้าของ + ใครถือต่อ
(ถัดลงไปเป็นภาพรวมโปรเจกต์ + ประวัติ — ยาวได้ไม่จำกัด)

> Close ต้องอัปเดต 4 หัวข้อบนนี้ **ทุกครั้ง** ก่อนเขียนอย่างอื่น · ไฟล์นี้คือแหล่งจริงของสถานะ — session log ฝั่ง vault เป็นสำเนาละเอียด ไม่ใช่แหล่งจริง (ถ้าขัดกัน เชื่อ `.project/` + git)

## 2. Decision Token (Close เขียน → ทุกตัวอ่านก่อนทำงาน)

| Token | แปลว่า | ทุกตัวต้องทำก่อนเริ่ม |
|---|---|---|
| `CLOSED_CLEAN` | ไม่มีค้าง CI เขียว | ทำต่อได้ |
| `CLOSED_WITH_PENDING` | มีงานค้าง/claimed | อ่านงานค้างก่อน |
| `NEED_OWNER_ACTION_BEFORE_CLOSE` | รอบก่อนยังไม่ปิดจริง | **เตือนเจ้าของ + ห้ามลุยต่อจนกว่าจะเคลียร์** |

**reality-overrides-token:** token ขัดกับของจริง (git/CI/SHA) → เชื่อของจริง + ตีธงว่ารอบก่อนปิดไม่ตรง

## 3. verified vs claimed (หัวใจกันโกหก · ใช้ทุกตัว)

- **verified** = มี output จริงแปะตามบันได §4 เท่านั้น
- **claimed** = บอกว่าเสร็จ แต่ไม่มี output
- **กฎเหล็ก: ไม่มี output จริงแปะ = claimed อัตโนมัติ** · ห้ามเลื่อน claimed ขึ้น verified เอง · claimed = นับเป็นงานค้างเสมอ

## 4. บันไดหลักฐาน (Evidence Ladder · ตามชนิดงาน)

| ชนิดงาน | verified ต่อเมื่อ (แปะของจริง) |
|---|---|
| แก้โค้ด | commit SHA + ผล quality gate ที่ค้นเจอ (§5) แปะ output |
| migration/schema | คำสั่ง migrate + output + ตรวจ schema จริงหลังรัน |
| deploy (CI/CD on merge) | merge SHA + CI run ของ SHA นั้น = เขียว + (ถ้ามี) health 200 + live SHA ตรง |
| config/env | diff + ยืนยันค่าโหลดจริง (redact secret §7) |
| ตรวจไม่ได้ในแชทนี้ | **บังคับ claimed** |

## 5. Quality Gate Auto-Detection (AI ค้นเอง)

ค้นตามลำดับจนเจอ รันเฉพาะที่มี ที่เหลือ N/A:

1. `package.json` scripts (test/lint/typecheck/build) → 2. `Makefile` → 3. `pyproject/tox/pytest` → 4. `.github/workflows/*` (รันชุดเดียวกับ CI) → 5. ไม่เจอ = รายงานตรง ๆ ว่าไม่มีตัวตรวจอัตโนมัติ ไม่เดาว่าผ่าน

> deploy verified วัดจาก **สถานะ CI run** ของ SHA ที่ merge ไม่ใช่ SSH เข้าไปดู

## 6. กัน pointer ชนกัน

`latest-close.md` แยก pointer **ต่อ staff** (block `nat:`/`may:`/`mind:`) · New เลือกของ staff ตัวเอง ไม่รู้ staff = ถาม

## 7. Redaction (บังคับก่อนเขียน memory/ledger/หลักฐาน)

redact token/password/connection string/private key เสมอ · remote URL → `https://***@...` · ไม่แน่ใจว่าลับไหม = ถือว่าลับ

---

## 8. Version Compatibility

Schema v1.2 ↔ Close ≥ v2.2 ↔ New ≥ v1.8 ↔ Act-As ≥ v3.1 ↔ Comply ≥ v3.1 ↔ Continue ≥ v4.1 ↔ Review Chat ≥ v2.3
ทุก prompt เช็ก schema version ตอนเริ่ม · ไม่ตรง = **เตือน + ห้ามเขียนไฟล์ความจำใด ๆ จนกว่าจะอ่าน schema ล่าสุดแล้ว** (แค่เตือนเฉย ๆ ไม่พอ — AI จะข้าม · ผลตรวจข้ามค่าย 2026-07-05)

---

## 9. Lifecycle — ตัวไหนต่อตัวไหน (ใหม่ใน v1.1)

```
New (เปิด+อ่านความจำ)
  → Act-As (วางแผนเป็นเฟส+id → เขียน .project/plan.md เมื่ออนุมัติ)
    → Comply (แตก issue ใต้แต่ละเฟส + % + หลักฐาน)
      → Continue (ลงมือทีละเฟส อัตโนมัติ + ledger)
        → Close (verify รวบ + เขียน memory + token)
```

แต่ละตัวอ่าน output ของตัวก่อนหน้า (ผ่านไฟล์ §1) ไม่เริ่มจากศูนย์ · ข้ามขั้นได้ถ้างานเล็ก แต่ตัวที่รันต้องอ่านความจำ+token ก่อนเสมอ (§2)

## 10. Shared ID (ใหม่ · ให้ส่งต่องานไม่หลุด)

- เฟส = `P1, P2…` · issue = `P1-I1, P1-I2…`
- Act-As ตั้ง id ในแผน → Comply แตก issue ใต้ id เดิม → Continue อ้าง id ตอนลงมือ+ใน ledger → Close ยก id ไปลง changed-files/งานค้าง
- id เดียวตามงานตลอดสาย = handoff ไม่เสียข้อมูล

## 11. Status Mapping (ใหม่ · ให้ Close อ่าน Comply รู้เรื่อง)

Comply ใช้ 6 สถานะ · เวลา Close/New สรุปความจำ ให้ยุบเป็น 2 ตาม §3:

| Comply status | นับเป็น (§3) | เข้า %100 ไหม | Close ปฏิบัติ |
|---|---|---|---|
| `verified` | verified | ✓ | งานเสร็จ |
| `failed` / `blocked` / `in_progress` / `not_started` / `unknown` | claimed/ไม่เสร็จ | ✗ | **งานค้าง** (ระบุ next_owner) |

> ใช้สถานะละเอียด 6 คำได้ใน Comply เพื่อสื่อสาร แต่ตอนคิด % และตอนปิด ยึด verified อย่างเดียว = 100%

## 12. Autonomy Policy (ใหม่ · ใช้ร่วมกับ Continue §3 ชั้น)

ค่าตั้งต้นของทีมนี้ (non-dev owner + SaaS มีผู้ใช้จริง + CI/CD auto-deploy on merge):

| การกระทำ | ชั้น | auto ได้ไหม |
|---|---|---|
| อ่าน/แก้โค้ดใน scope/รันเทส/เขียน doc | 1 | auto |
| push ขึ้น branch ตัวเอง / deploy **staging** / ลง dependency (มีหลักฐาน) | 2–3A | auto ถ้าด่านผ่าน |
| **merge → main / deploy production / migration prod** | **ต้องขอคน** | **ค่าตั้งต้น = ปิด auto** (เพราะ merge→main = ขึ้น prod อัตโนมัติ ที่มีผู้ใช้จริง) |
| ใช้เงิน/ส่งเมล/เปิด secret/ลบถาวร | 3B | ขอคนเสมอ |

> flag `ALLOW_AUTO_PROD` (ค่าตั้งต้น = OFF) · เปิดเมื่อเจ้าของสั่งชัดเท่านั้น · เปิดแล้วยังต้องผ่าน fail-closed gate + ledger

## 13. Ledger (ใหม่ · กันงาน auto หายตอนปิด)

- Continue เขียน append-only: `.hermes/ledger/<branch>.md` (เวลา/คำสั่ง/ด่าน/exit code/SHA/ผล · redact §7)
- ตอน Close: ยก ledger รอบนี้เข้า session log → ความจำเห็นว่า AI ทำอะไรเองไปบ้าง
- ไม่มี ledger ทั้งที่มีการกระทำชั้น 3A = ถือว่างานนั้น **claimed** (ไม่มีหลักฐาน)

## Changelog

- v1.2 (2026-07-05): **ย้ายความจำที่ใช้ทำงานต่อไป `.project/` ที่เดียว** ตามคำสั่งเจ้าของ (ปฐมเหตุ: AI อ่านไม่ครบ 2 โฟลเดอร์แล้วทำงานมั่วซ้ำหลาย project · เคสหนักสุด: สร้าง UI ใหม่ทับ wireframe ที่ล็อกแล้วใน EA Farm) · `.project/plan.md` (เดิม `.hermes/plan.md`) · `.project/OverviewProgress.md` ยุบ handoff+active + โครงบังคับ 4 หัวข้อบนสุด + ป้ายเวอร์ชันบรรทัดแรก (§1c) · `.project/decisions.md` (เดิม `.hermes/decisions.md`) · กติกาย้ายของเก่า "อ่านได้สองที่ เขียนที่ใหม่เท่านั้น + stub ห้ามลบ" (§1b) · schema ไม่ตรง = ห้ามเขียนความจำ (§8) · ผ่านตรวจข้ามค่าย Grok+Codex 2026-07-05
- v1.1 (2026-06-26): ขยายจากคุม New/Close เป็นคุมทั้ง 6 ตัว · เพิ่มไฟล์ `.hermes/plan.md` (Act-As เขียน) · §9 Lifecycle · §10 Shared ID · §11 Status Mapping · §12 Autonomy Policy (ค่าตั้งต้นปิด auto prod) · §13 Ledger · ขยาย version lock เป็น 6 ตัว
- v1 (2026-06-26): สัญญากลางฉบับแรก คุม New/Close

## Graph Links

- Parent hub: [[skills/prompt-shortcuts/Prompt Shortcuts|Prompt Shortcuts]]
- Registry: [[ai-context/prompt-shortcut-registry|Prompt Shortcut Registry]]
- Writers/Readers: [[skills/prompt-shortcuts/references/use-close-chat|Use Close Chat]] · [[skills/prompt-shortcuts/references/use-new-chat|Use New Chat]] · [[skills/prompt-shortcuts/references/use-act-as|Use Act-As]] · [[skills/prompt-shortcuts/references/use-comply|Use Comply]] · [[skills/prompt-shortcuts/references/use-continue|Use Continue]]
- คู่มือเจ้าของงาน: [[skills/prompt-shortcuts/references/memory-system-owner-guide-th|ระบบความจำข้ามแชท — คู่มือเจ้าของงาน]]
