# Session Log — 2026-07-14 · Use Migrate Web (Shortcut ใหม่)

> staff: nat · branch: `control_webengine_flow` · เริ่มด้วย Fable (ออกแบบ) → สลับ Opus (ปิดแชท) · schema v1.2

## เป้าหมายรอบนี้
สร้าง Shortcut `Use Migrate Web` — สายการผลิตเว็บจากโรงงาน Root Admin ตาม Flow 13 ขั้น ใช้ทุกโปรเจกต์ (RSF/DRA/ContentThailand) · กัน AI โกหก/ข้าม Flow ด้วยเครื่องตัดสิน · ทีม ~10 คนใช้พร้อมกัน notebook+VPS

## Changed-files table
| file | changed_by | reason | verification | risk | next_owner |
|---|---|---|---|---|---|
| `.project/mw-spec-draft.md` | Fable | SPEC v1.2 ตารางแม่ 55/55 | verified: mw-spec-check PASS + Codex 2 รอบ | prompt ยังไม่ใช้จริง — P4 | Opus |
| `team-shortcuts/payload/.../use-migrate-web.md` | Fable | prompt หลัก v1.0 | verified: Codex 2 รอบ + grep | ยังไม่รันจริง | Opus |
| `team-shortcuts/payload/.../use-migrate-web-flow13.md` | Fable | เนื้อ Flow เต็ม 439 บรรทัด | verified: sha256 ตรงต้นฉบับ 100% | ต้นทางแก้ = ออกรุ่นใหม่ (มีด่าน checksum) | Opus |
| `scripts/mw-spec-check.py` | Fable | เครื่องตรวจโครงสร้าง spec | verified: fail→pass พิสูจน์แล้ว | ยังไม่มี unit test แยก | Opus (P3-I2) |
| `.project/mw-flow-baseline.md` | Fable | sha256 ต้นทาง 3 ไฟล์ | verified: shasum ตรง | — | — |
| `team-shortcuts/payload/ai-context/prompt-shortcut-registry.md` | Fable | +1 แถวทะเบียน | verified: grep=1 | ทะเบียนคลัง Obsidian ยังไม่เพิ่ม (เจ้าของ push เอง) | เจ้าของ |
| `.project/plan.md` | Fable/Opus | แผน MW + แผนส่งมอบ I2 | verified: grep | plan-anchor อ่าน QAQC แรก → MW ใช้ --no-plan | Opus |
| `.project/OverviewProgress.md` | Fable/Opus | สถานะล่าสุด MW | verified: อ่านได้ | — | — |

## Decision log
- ต่อยอด Flow 13 ขั้น v2 ที่ผ่านทีมรีวิว 24/24 ไม่เขียนใหม่ (คุ้ม+ทีมไม่ต้องเรียนใหม่)
- โครง 2 ชั้น: แกน Flow กลาง (Hermes) + Project Profile `.work/` ต่อเว็บ
- "เสร็จ" = เครื่องตัดสิน (menu-gate/page-check/gate-run) หรือคนยืนยัน — ไม่ใช่ AI พูด
- ผู้ตรวจ AI ครบ 2 รอบไม่ผ่าน → เปลี่ยนเป็นเครื่องตรวจ (mw-spec-check) ตามกติกา relay v2.16 — ใช้จริงสำเร็จ

## Quality gate
- `save-git --stage merge-gate` = **SAFE_TO_MERGE** (ด่าน 1-2 pass · CI skip-ok documented) · commit 1 · files 8 · in scope
- `mw-spec-check.py` = PASS (ตารางแม่ 55/55 · [G] ครบ · จุดเคาะ 13/13 · baseline ตรง)
- CI: skip-ok (workflow เป็น upstream NousResearch — ไม่บังคับกับ push นี้)

## Deploy
N/A — ยังไม่ merge (งานหลายเฟส MW-P1..P5 = 1 PR ตอนจบ · ตอนนี้ P3-I1)

## งานค้าง + เจ้าของถัดไป
- **MW-P3-I2 (Opus)**: เครื่องมือ 7 ตัว (เริ่ม work-locks) — แผนเต็มใน plan.md
- MW-P3-I3/I4 · P4 ทดสอบจริง RSF · P5 ปิด 1 PR (เจ้าของกด merge)
- เจ้าของ: push คลัง Obsidian (เพิ่มแถวทะเบียน use-migrate-web)

## ความเสี่ยงที่เหลือ
- prompt ยังไม่ผ่านการใช้จริง (P4) · เครื่องมือ 7 ตัวยังเป็นสัญญา (สถานะ P3-I2 ระบุชัดใน prompt)
- relay review mode พังบนเครื่องนี้ (portal + flag เก่า grok 1.0.1) → ใช้ cross-check MCP แทน · จดใน plan

## next step + ข้อความเปิดแชทหน้า (ก๊อปได้)
```
Use New Chat
แล้ว Use AI Relay ทำ MW-P3-I2 ตามแผนส่งมอบใน .project/plan.md (เครื่องมือ 7 ตัว เริ่ม work-locks)
```
