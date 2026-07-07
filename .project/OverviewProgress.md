> memory-schema: v1.2
> อ่านตามลำดับ: plan.md (plan_id: GRD) → decisions.md → hermes-standard/REQUIREMENTS.md (บัญชีความต้องการ 66 ข้อ)

# Overview & Progress — Hermes Agent
อัปเดตล่าสุด: 2026-07-07 · branch งาน: `feature/plan-guardrails` (แตกจาก main `5aa135e7f`) · ป้าย: [fact] เว้นแต่ระบุ

## สถานะล่าสุด
- **แผนใหม่ plan_id: GRD "ระบบกันแผนหาย กัน AI ทำงานมั่ว" อนุมัติแล้ว 2026-07-07** — Fable สอบสวนจาก log จริง 3 สาย เจอต้นตอ 6 ข้อ (รายละเอียดใน plan.md + decisions.md) · เจ้าของอนุมัติ 4 เฟสรวดเดียว · Codex เขียน + Claude ตรวจ + gate-run ตัดสิน [fact]
- PR #15 (แก้ auth ปลอมใน relay-call) merge เข้า main แล้ว — main HEAD = `5aa135e7f` [fact]
- แผนเก่า "Hermes ใช้งานจริงทั้งระบบ": P0-P1 เสร็จ merged (PR #12) · งานค้าง P2-P6 เดิมถูกดูดเข้า GRD-P5..P9 ไม่มีอะไรหาย [fact]
- สาย JARVIS v2: เปลือกเสียงใช้ได้จริง · แผน 8 เฟสอนุมัติแล้วที่ `.project/FeatureSpec-jarvis-voice.md` · รอเจ้าของทดสอบ P0 แล้วเปิดแชตใหม่ส่ง Use AI Relay [fact]

## งานถัดไป
1. เดินแผน GRD-P1 → P3 ผ่านสายพาน relay (Codex เขียน · Claude ตรวจ · gate-run ตัดสิน) → จบเป็น 1 PR ให้เจ้าของกด merge
2. GRD-P5 Monitor Hub เริ่มเมื่อ GRD-P1..P4 verified + เจ้าของสั่ง
3. GRD-P6..P8 รอเจ้าของส่ง "ปัญหาชุดสุดท้าย" ก่อนล็อกดีไซน์

## ข้อห้าม/กติกาล็อก
- ห้ามเขียนความจำทำงานต่อลง `.hermes/` หรือ root — เขียน `.project/` เท่านั้น (Schema v1.2)
- หลังสร้าง/ย้ายไฟล์ `.project/` ต้องผ่านด่าน `git check-ignore` + `git ls-files` ก่อนบอกเสร็จ
- **เลขงานต้องขึ้นต้นด้วย plan_id (เช่น GRD-P1-I1) · เลขที่ไม่มีใน plan.md = ห้ามทำ** · หลังตอบคำถามแทรก ต้องเปิด plan.md ทวนเฟสก่อนลงมือ (กติกาเหล็กของแผน GRD)
- ห้ามแตะ `scripts/jarvis-voice/` + `design-system-standard-v2/` + `.claude/launch.json` (งานเจ้าของค้าง · คนละสาย)
- ห้าม merge→main / deploy เอง — เจ้าของกด · งานหลายเฟส = 1 PR เดียว
- สมองแผน GRD = Fable ตามคำสั่งเจ้าของ 2026-07-07 (ข้อยกเว้นจากกติกา relay v2.7 ที่ปกติใช้ Opus) · Codex/Claude เขียน-ตรวจสลับค่ายผ่าน relay-call · **verified = มีแถว gate-run เท่านั้น**

## งานค้าง/ส่งต่อ
- รอเจ้าของ: ส่งปัญหาชุดสุดท้าย (ปลดล็อก GRD-P6..P8) · rotate GitLab token (ค้างจาก 2026-07-04) · สั่ง commit ไฟล์ JARVIS untracked (`scripts/jarvis-voice/` — ระวัง `.venv-gemini` ต้องใส่ .gitignore ก่อน) · กด merge PR แผน GRD เมื่อพร้อม
- claimed (ยังไม่ตรวจ): iptables :3010 ไม่ persistent ข้าม reboot · webhook pr-review ลงแค่ project เด็กฝึก 527
- ด่านกันลบโฟลเดอร์ทั้งก้อน (phase-013): **โค้ด+เทสต์เข้า main แล้ว (`f9fb0827f`) [fact — แก้ความจำเก่าที่จดว่ายังค้าง]** · ที่ยังค้างจริง = ยืนยันว่า VPS runtime รันโค้ดรุ่นที่มีด่านนี้ (ยัง unverified)
- feature ค้างจากตารางรีวิว Hermes 2026-07-03: F3-F8
- อัปรุ่น v0.18.0 = GRD-P9 (ยังไม่เริ่ม · ต้องทำบัญชีของต่อเติมก่อน)

---

## project นี้คืออะไร (2-3 บรรทัด)
ศูนย์เครื่องมือ AI ส่วนตัวของเจ้าของ (fork จาก NousResearch/hermes-agent v0.17.0 + ของต่อเติม ~3,215 commit): สายพาน AI Relay ประหยัดเงิน · ชุด shortcut คุมวินัยงาน · มาตรฐานกลาง 30-40 โปรเจกต์ (hermes-standard) · เครื่องมือคุมคุณภาพ (violation-audit, pr-review-gate, curse tracker) · gateway ให้ทีม 15 คนใช้บน VPS [fact]

## เสร็จแล้ว (verified) + ประวัติย่อ
- 2026-07-07: PR #15 แก้ auth ปลอม relay-call merge เข้า main (`5aa135e7f`) · สอบสวนต้นตอ AI มั่ว 6 ข้อ + แผน GRD อนุมัติ [fact]
- 2026-07-06: P0-P1 แผนเก่า merged — PR #12 (`da4689a58`) · Project OS ครบ 4/4 · ความจำอยู่ `.project/` [fact]
- 2026-07-05: กู้ shortcut Project OS 3 ตัว (ถูก revert `fff10805b` เมื่อ 2026-06-28 โดยความจำยังจดว่าครบ) + ด่านไฟล์เข้า git จริงทั้งระบบ (`f079acf47`) · relay-call เพิ่มนาฬิกากันค้าง — pytest 16/16
- 2026-07-04-05: relay P3 ครบ 4/4 — PR #8, #9, #10 merged · F1 violation-audit + F2 pr-review-gate ใช้จริง (tier 3) · AI Relay ยืนยันทั้ง notebook + VPS
- 2026-06-21: ด่านกันลบโฟลเดอร์งานทั้งก้อน phase-013 (Codex เขียน · ตรวจแล้ว 38+14 เทสต์) — เข้า main ที่ `f9fb0827f`
- ก่อนหน้า: ดู decisions.md + session log ใน vault (`projects/hermes-agent-dev/`)
