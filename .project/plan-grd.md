> [เก็บประวัติ 2026-07-10] แผน GRD จบครบ 4 เฟส merged แล้ว · ไฟล์นี้ย้ายมาจาก plan.md เพื่อเปิดทางแผน active ใหม่ (QAQC) · คิว GRD-P5..P9 ยังอยู่ท้ายไฟล์นี้ · จะเริ่มคิวไหน ให้ยกกลับไป plan.md ก่อน (plan-anchor อ่าน plan.md เท่านั้น)

# Plan — GRD · ระบบกันแผนหาย กัน AI ทำงานมั่ว (อนุมัติ 2026-07-07 · "อนุมัติ" ในแชท Fable)

> memory-schema: v1.2 · **plan_id: GRD** · จาก Use Act-As v3.1 (Fable วิเคราะห์จาก log จริง 3 สาย · เจ้าของอนุมัติทั้ง 4 เฟสรวดเดียว 2026-07-07)
> ทีม: **Fable = สมองวางแผน/ตัดสิน** (ข้อยกเว้นที่เจ้าของสั่งชัดในแชท 2026-07-07 — ไม่ใช่ค่าเริ่มต้นของ relay) · **Codex = เขียนโค้ด** (`relay-call --tool codex`) · **Claude = ตรวจ** (`relay-call --tool opus` · คนละค่ายกับคนเขียน) · **gate-run = ผู้ตัดสิน verified** · เจ้าของ = กด merge
> branch: `feature/plan-guardrails` (แตกจาก main `5aa135e7f`) · งานหลายเฟส = **1 PR เดียว** · ห้าม merge→main / deploy เอง (`ALLOW_AUTO_PROD=OFF`)
> แผนเก่า "Hermes ใช้งานจริงทั้งระบบ" (2026-07-05): P0-P1 เสร็จ merged แล้ว · งานค้าง P2-P6 เดิมถูกดูดเข้าเป็น GRD-P5 ถึง GRD-P9 ท้ายไฟล์นี้ ไม่มีอะไรหาย

## กติกาเหล็กของแผนนี้ — ทุก AI ที่มาทำงานต่อ ต้องทำตามก่อนแตะไฟล์ใดๆ

1. **เลขงานต้องขึ้นต้นด้วย plan_id** เช่น `GRD-P1-I1` · เลขงานที่ไม่มีในไฟล์นี้ = ห้ามทำ ห้ามเดา ห้ามตั้งเลขเอง (ต้นตอเดิม: เลข P1-I1 ถูกใช้ซ้ำ 3 แผนคนละเรื่อง จน AI ทำงานผิดแผน)
2. **กลับมาอ่านแผนก่อนลงมือทุกครั้ง** — หลังตอบคำถามแทรก / ออกนอกเรื่อง / สลับงาน ต้องเปิดไฟล์นี้ทวน "เฟสปัจจุบัน + ข้อห้าม" ก่อนแตะไฟล์ (ต้นตอเดิม: AI ตอบคำถามเสร็จแล้วทำต่อจากความจำแชทที่เลือน จนหลุดแผน)
3. **verified = มีแถว gate-run เท่านั้น** · reviewer บอกผ่าน = ข้อมูลประกอบ ไม่ใช่ verified · ไม่มีแถว gate-run = claimed (ยังไม่นับเสร็จ ห้ามรายงานว่าเสร็จ)
4. **ใบสั่งงานทุกใบเป็น "สัญญางาน" (Task Contract)**: เลขงาน + ไฟล์ที่แตะได้ (allowed) + ไฟล์ห้ามแตะ (forbidden) + คำสั่งตรวจที่ตอบผ่าน/ตกได้เอง (exit 0/1) · ใบสั่งงานต้องฝังข้อความเฟส+ข้อห้ามจากไฟล์นี้ลงไปในตัว ไม่พึ่งความจำแชท
5. ห้ามแตะ: `scripts/jarvis-voice/` (สาย JARVIS แยก) · `design-system-standard-v2/` · `.claude/launch.json` · `.hermes/**` (coder ห้ามแตะเด็ดขาด — เป็นช่องปลอมหลักฐาน) · `.env*` / secret ทุกชนิด

## ต้นตอ 6 ข้อที่แผนนี้แก้ (สอบสวนจาก log จริง 2026-07-07 · หลักฐานเต็มใน decisions.md)

1. ไม่มีกติกาบังคับ "กลับมาอ่านแผน" หลังออกนอกเรื่อง (ละเมิด "ตอบโดยไม่ทวนโจทย์" 3,790 ครั้ง)
2. แผนเคยอยู่ในโฟลเดอร์ที่ git ไม่เห็น → แผนอยู่เครื่องเดียว (แก้แล้วด้วย Schema v1.2 แต่เพิ่งครอบ repo นี้)
3. ไม่มีตัวเทียบความจำกับของจริง (เคส revert เงียบ `fff10805b` ความจำโกหก 7 วัน)
4. แชทหลายสายทับกัน — สายหนึ่ง stash กวาดงานอีกสายทิ้ง 2 รอบ (เคส JARVIS)
5. ปิดงานแบบอ้างว่าเสร็จ — สมุด gate-run (`.hermes/ledger/`) ไม่เคยมีจริง + บั๊ก auth ปลอมค้าง 4 งาน (แก้แล้ว PR #15)
6. เลขงานชนกันข้ามแผน (`P1-I1` ใช้ซ้ำ 3 แผนใน call ledger)

---

## GRD-P1 — สัญญางานผูกแผน (แก้ต้นตอ 1, 6) · สถานะ: กำลังทำ

- **GRD-P1-I1** สคริปต์ `plan-anchor` (`scripts/ai-relay/plan-anchor.py`)
  - หน้าที่ (ก) ตรวจว่าเลขงานมีจริงใน `.project/plan.md` — ไม่มี = exit 1 (ข) โหมด `--emit-brief` พิมพ์หัวใบสั่งงาน: ข้อความเฟส + ข้อห้าม + allowed/forbidden ของเลขงานนั้น เอาไปแปะหัว brief
  - allowed: `scripts/ai-relay/plan-anchor.py`, `scripts/ai-relay/tests/test_plan_anchor.py`
  - forbidden: อื่นทั้งหมด (โดยเฉพาะ `relay-call.py`, `.hermes/**`)
  - verify: `python3 -m pytest scripts/ai-relay/tests/test_plan_anchor.py -q` → exit 0
- **GRD-P1-I2** ผูกด่านเข้า `relay-call.py`: ถ้า repo มี `.project/plan.md` ที่ประกาศ plan_id → task-id ต้องผ่าน plan-anchor ก่อนยิงงาน · ไม่ผ่าน = ไม่ยิง คืนสถานะ `off_plan` · มีทางหนีชัดเจน `--no-plan` (จดลง ledger ว่า off-plan โดยตั้งใจ เพื่องานจร) · ของเก่าที่ไม่มี plan.md = พฤติกรรมเดิมทุกอย่าง
  - allowed: `scripts/ai-relay/relay-call.py`, `scripts/ai-relay/tests/test_relay_fixes.py` (เพิ่มเทสต์), `scripts/ai-relay/tests/test_plan_anchor.py`
  - forbidden: adapter/accounts จริงใน `.hermes/**`
  - verify: `python3 -m pytest scripts/ai-relay/tests/ -q` → exit 0 (ของเก่าต้องไม่พัง)
- **GRD-P1-I3** เพิ่มกฎ "re-anchor" ลง prompt กลางใน vault (`use-ai-relay.md` + `use-continue.md`): หลังตอบคำถามแทรก ต้องเปิด plan.md ทวนเฟสก่อนลงมือ · ผู้ทำ: Fable เอง (ไฟล์ vault ไม่ใช่โค้ด ไม่ผ่าน relay)
  - verify: grep เจอข้อความกฎในทั้ง 2 ไฟล์ + จด SHA/เวลาแก้ใน ledger แชทนี้

## GRD-P2 — ตัวเทียบความจำกับของจริง `memory-audit` (แก้ต้นตอ 3) · สถานะ: รอ P1

- **GRD-P2-I1** สคริปต์ `scripts/memory-audit/memory_audit.py` + เทสต์
  - ตรวจ 4 อย่าง แล้วออกรายงานภาษาไทย + exit 0(ตรง)/1(เจอความจำโกหก):
    (ก) commit SHA ที่ OverviewProgress อ้าง มีจริง + ไม่ถูก revert ทีหลัง (สแกน `git log --grep=Revert` + เทียบไฟล์)
    (ข) ไฟล์ที่ความจำอ้างว่ามี — มีจริง + git เก็บจริง (`check-ignore`/`ls-files`)
    (ค) ป้าย `> memory-schema:` + plan_id ใน plan.md ยังครบ
    (ง) เลขงานในสมุด relay (`calls-*.md`) สังกัด plan_id ที่รู้จัก — เจอเลขกำพร้า = เตือน
  - allowed: `scripts/memory-audit/**`
  - forbidden: `.project/**` (สคริปต์อ่านอย่างเดียว ห้ามแก้ความจำเอง), `.hermes/**`
  - verify: `python3 -m pytest scripts/memory-audit/tests/ -q` → exit 0 + ยิงเคสจำลอง "อ้าง SHA ที่ถูก revert" แล้วจับได้
- **GRD-P2-I2** ต่อเข้าวงจรใช้งาน: เพิ่มบรรทัดสั่งรัน memory-audit ในขั้น 0 ของ `use-new-chat.md` (vault) + คำสั่งติดตั้งตัวตั้งเวลารายสัปดาห์ 1 บรรทัดให้เจ้าของก๊อปวาง · ผู้ทำ: Fable (vault) + Codex (สคริปต์ติดตั้ง ถ้าจำเป็น)
  - verify: grep เจอใน use-new-chat.md + คำสั่งติดตั้งรันแล้ว `launchctl list` เห็นรายการ (งานคน: เจ้าของกด)

## GRD-P3 — ด่านกันแชทซ้อนทับ/กวาดงานคนอื่น (แก้ต้นตอ 4) · สถานะ: รอ P1

- **GRD-P3-I1** ต่อยอดด่านใน `tools/approval.py` ตามแบบ phase-013 ที่พิสูจน์แล้ว: จับคำสั่งกลุ่ม "กวาดงานค้างทิ้ง" — `git stash push/clear/drop` · `git clean -f` · `git checkout -- .` · `git restore .` (ทั้งโฟลเดอร์) → บล็อกพร้อมข้อความ "อาจกวาดงานค้างของสายอื่น ต้องให้เจ้าของยืนยัน" · ไม่ false positive กับ `git stash list/show` และ `git restore <ไฟล์เดียวที่เพิ่งแก้เอง>`
  - allowed: `tools/approval.py`, `tests/tools/test_stash_sweep_guard.py` (ใหม่)
  - forbidden: ไฟล์ test เดิม (ห้ามแก้ให้ผ่าน) · ทุกอย่างนอก allowed
  - verify: `python3 -m pytest tests/tools/test_stash_sweep_guard.py tests/tools/test_workspace_retirement_guard.py tests/tools/test_command_guards.py -q` → exit 0 (ของเดิม 38 เคสต้องยังผ่าน)

## GRD-P4 — ล้างความจำเก่าให้ตรงจริง + รวมเล่มแผน (แก้ต้นตอ 2, 5) · สถานะ: ทำในแชทนี้โดย Fable

- **GRD-P4-I1** เขียน plan.md ฉบับนี้ + แก้ OverviewProgress 3 จุดที่ไม่ตรงจริง (HEAD เก่า · สถานะด่านกันลบโฟลเดอร์ที่จริงๆ merge แล้ว `f9fb0827f` · บรรทัดบทบาท Fable) + append decisions.md
  - verify: `git check-ignore -v .project/*` ว่าง + `git ls-files .project/` ครบ + commit SHA ใน PR เดียวกัน

---

## คิวถัดไป (ดูดจากแผนเก่า · ไม่มีอะไรหาย · เริ่มเมื่อ GRD-P1..P4 verified + เจ้าของสั่ง)

- **GRD-P5** Monitor Hub — URL เดียวบน VPS + การ์ดเช้าเข้า Lark (เดิม P5 · Codex เขียน สลับผู้ตรวจตามกติกา)
- **GRD-P6** มาตรฐานกลางวิ่งจริง 30-40 โปรเจกต์ ผ่าน safe_apply เท่านั้น (เดิม P2 · **รอเจ้าของส่ง "ปัญหาชุดสุดท้าย" ก่อนล็อกดีไซน์**)
- **GRD-P7** ปิดวงจรคำด่า/ความพลาด — ตัวจับใหม่ + วิเคราะห์ทุก 3 ชม. + ด่าซ้ำออกใบงานอัตโนมัติ (เดิม P3 · รอพร้อม P6)
- **GRD-P8** ย้ายน้ำหนัก hook จาก "เตือนหลังตอบ" เป็น "กั้นก่อนทำ" + ซ่อมตัวนับเฟ้อ (เดิม P4 · รอพร้อม P6)
- **GRD-P9** อัปรุ่น v0.17.0 → v0.18.0 — ต้องทำบัญชีของต่อเติม ~3,215 commit ก่อน + ซ้อมใน worktree `upgrade-audit` (เดิม P6)
- สายคู่ขนาน **JARVIS v2**: แผนแยกที่ `.project/FeatureSpec-jarvis-voice.md` (plan_id: jarvis-v2-phase-plan) · ขอบเขต `scripts/jarvis-voice/` เท่านั้น · รอเจ้าของทดสอบเสียง P0

## งานคน (เจ้าของ)

- กด merge PR ของแผน GRD นี้ (1 PR เดียว)
- สั่ง commit ไฟล์ JARVIS ที่ยัง untracked (`scripts/jarvis-voice/` + `FeatureSpec-jarvis-voice.md` — อยู่เครื่องเดียว เสี่ยงหาย) · หมายเหตุ: โฟลเดอร์นั้นมี `.venv-gemini` ทั้งก้อน ควรใส่ `.gitignore` ก่อน commit
- รันคำสั่งติดตั้ง memory-audit รายสัปดาห์ 1 บรรทัด (GRD-P2-I2 เตรียมให้)
- ส่ง "ปัญหาชุดสุดท้าย" เพื่อปลดล็อก GRD-P6..P8

## ความเสี่ยงค้าง

- ด่าน plan-anchor กันได้เฉพาะงานที่วิ่งผ่าน relay-call — AI ที่แก้ไฟล์ตรงๆ ไม่ผ่าน relay ยังต้องพึ่งกติกาเหล็กข้อ 2 (จะย้ายเป็นด่านเครื่องใน GRD-P8)
- memory-audit จับ "ความจำโกหก" ได้เฉพาะแบบที่นิยามไว้ 4 อย่าง — แบบใหม่ๆ ต้องเติม
- Grok บน VPS ยังไม่ล็อกอิน · Opus 4.7/4.6 ยังไม่ทดสอบ (ยกมาจากแผนเก่า)
