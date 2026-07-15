# Session Log — 2026-07-15 · MW-P4 จบ + MW-P6 Flow Enforcement + team-ready

> memory-schema: v1.2 · staff: nat · แชท: Opus (สลับ Fable ช่วงวิเคราะห์ตามคำสั่งเจ้าของ) · branch: `feature/mw-flow-gate` + `feature/mw-team-ready` (merged แล้วทั้งคู่)

## เป้าหมายรอบนี้ (วิวัฒน์ตามคำสั่งเจ้าของ)
1. เริ่ม: MW-P4 รัน mw-backend-check กับ RoadSafeFund จริง
2. เจ้าของสั่งเพิ่ม: แบ่งงาน 2 โซน (ไม่เสี่ยง = AI ทำเลย ไม่ต้องกด OK ทีละครั้ง)
3. เหตุการณ์: AI ข้าม flow เอง (เดา workflow 6 ขั้น ไม่เปิด flow13) → เจ้าของสั่ง **แก้ต้นเหตุ: บังคับ flow ด้วยเครื่อง + พิสูจน์ว่าข้ามไม่ได้/โกหกไม่ได้**
4. เป้าสุดท้ายที่เจ้าของประกาศ: **ทุกอย่าง 100% พร้อมใช้ → เจ้าของทดสอบผ่านก่อน → ค่อยประกาศทีม** — สำเร็จครบ

## Changed-files table
| file | owner | changed_by | reason | verification | risk | next_owner |
|---|---|---|---|---|---|---|
| scripts/mw/flow_eval.py + flow-rules.yaml + flow_gate.py | MW | Codex (relay I2) | เครื่องคุมลำดับ 13 ขั้น สถานะคำนวณสด | verified — pytest 20 เคส + exit codes ตรวจเอง | slug/path ผ่าน validate แล้ว | ทีมใช้ |
| team-shortcuts/hooks/enforce-flow-gate.py + install-team-hooks.py + flow_gate.py(guard-write) | MW | Codex (relay I3) | hook PreToolUse บังคับที่ตัวโปรเจกต์ | verified — เทสต์รวม + demo 4 ฉาก block จริง | กันคนตั้งใจโกงมือไม่ได้ 100% (ตรวจย้อนหลังได้) | ทีมติดตั้ง |
| scripts/mw/menu_gate.py + mw_doctor.py | MW | Codex (relay I4) | ผูกด่านปิดเมนู + เช็คติดตั้ง | verified — mw 291 passed ของเดิมไม่พัง | ต้องส่ง --menu ตอนใช้จริง | ทีมใช้ |
| tests/team_shortcuts/test_new_chat_write_permit.py | payload | Opus | เทสต์ตรึงรุ่นเก่า 2.6/4.4 → อัปตามไฟล์จริง 2.7/4.5 | verified — 310 passed 0 failed | — | — |
| team-shortcuts/OWNER-ACCEPTANCE-MW.md | MW | Opus | ชุดทดสอบรับงาน 5 ข้อ | verified — เจ้าของกดผ่าน 5/5 (tier 5) | — | เก็บใช้รอบหน้า |
| VPS /home/linux-nat/mw-p4/* (config+wrapper) | MW-P4 | Opus | รัน backend-check จริงกับ RSF 78 | verified — 3/3 PASS + neg 2/2 + form_cycle PASS + TEST-MW ลบเหลือ 0 | ไฟล์อยู่นอก repo (VPS) | ทีม/เจ้าของ |
| hermes-standard/* (curse tracker) + adapters (กฎ Use ...) | เซสชันอื่น | Opus (เก็บกันหาย) | WIP เซสชันอื่นค้าง ไม่ commit | tests 37+23+2/2 เขียวก่อน commit | **ยังไม่ merged** อยู่ feature/spec-central + control_webengine_flow | เซสชันเจ้าของงานนั้น |

## Decision log
- **ต้นเหตุ AI ข้าม flow** (Fable วิเคราะห์ + Codex ตรวจค้าน): (1) ตัวบังคับจริงมีแค่ menu-gate ปลายทาง — G5 ที่เหลือเป็นตัวหนังสือ (2) งานที่เป็น migrate-web แต่ไม่พิมพ์ชื่อ shortcut = หลุดทุกด่าน (3) ไม่มีสถานะต่อเมนูที่เครื่องอ่านได้
- **ดีไซน์ตาม Codex เสริม**: สถานะคำนวณสดจากหลักฐาน (ไม่มี state ไฟล์ให้ปลอม) · กติกากลางไฟล์เดียว evaluator เดียว · ตรวจเนื้อไฟล์ไม่ใช่แค่มีไฟล์ · บังคับที่ตัวโปรเจกต์ (.work/profile.yaml) · checkpoint เจ้าของคง 4 จุดเดิม
- **API จริงของ synerry-engine ใช้ prefix `/api/v1`** (main.ts setGlobalPrefix) — ไม่ใช่ `/api` · จุดนี้ทำ 404 ตอนแรก
- **hook เก่ารุ่น MVP (13 ก.ค.) บนเครื่องเจ้าของไม่คุม Bash** — อัปเป็นรุ่นใหม่แล้วผ่าน installer (เจ้าของรันเอง)
- ลบ `.bak` 1 ไฟล์หลังพิสูจน์ diff ว่าเนื้อตรงกับ git history 100% (ปลดล็อกด่าน push)

## Quality gate ที่รัน + ผลจริง
- `pytest tests/scripts/mw/ tests/team_shortcuts/ -q` → **310 passed, 0 failed** (รันสดตอนปิด)
- demo block จริง: เขียนข้ามขั้น exit 2 · shell แอบเขียน exit 2 · M0 ตามลำดับ exit 0 · โปรเจกต์อื่น exit 0
- backend-check RSF: อ่าน 3/3 PASS · negative 2/2 FAIL ถูกต้อง · form_cycle PASS · TEST-MW เหลือ 0

## Deploy
- ไม่มี CI/CD auto-deploy ใน repo นี้ (gh run list ว่าง) = N/A · merged: PR #42 (`c7af9fd44`) + PR #43 (`706b278c3`) — เจ้าของกดทั้งคู่ · hook ติดเครื่องเจ้าของแล้ว (ยืนยัน diff IDENTICAL + demo ผ่าน hook จริง)

## งานค้าง + เจ้าของถัดไป
- branch เซสชันอื่นยังไม่ merged (spec-central + control_webengine_flow) → เซสชันงานนั้น/เจ้าของ
- ทีมติดตั้งจริงทุกเครื่อง = claimed · mw-setup.sh บน VPS ยังไม่ยืนยัน → ทีม/รอบหน้า
- เมนูแรกจริง RSF ต้องเริ่ม FW-P0 (สร้าง .work/profile.yaml — เจ้าของล็อกค่า)

## ความเสี่ยงที่เหลือ
- flow-gate คุมเฉพาะโปรเจกต์ที่มี `.work/profile.yaml` — โปรเจกต์ที่ยังไม่เปิด FW-P0 ยังไม่ถูกคุม
- ชุดเทสต์เต็ม repo ยังแดง 683 เคสที่ฐาน (ปัญหาเก่า ไม่เกี่ยวรอบนี้)

## ข้อความเปิดแชทหน้า (ก๊อปวางได้)
Use New Chat
งานต่อ: เดินเมนูแรกจริงบน RoadSafeFund ตาม Use Migrate Web — เริ่ม FW-P0 (สร้าง .work/profile.yaml เจ้าของล็อกค่า) แล้วเข้า M0 เมนูแรก · flow-gate + hook ใช้งานแล้ว (ทดสอบรับงานผ่าน 5/5 · 2026-07-15) · อ่าน .project/OverviewProgress.md + session-log-2026-07-15-flow-enforcement.md
