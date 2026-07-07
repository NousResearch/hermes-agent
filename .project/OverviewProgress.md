> memory-schema: v1.2
> อ่านตามลำดับ: plan.md (plan_id: GRD) → decisions.md → hermes-standard/REQUIREMENTS.md (บัญชีความต้องการ 66 ข้อ)

# Overview & Progress — Hermes Agent
อัปเดตล่าสุด: 2026-07-08 (คืน Use Continue) · branch งาน: `feature/plan-guardrails` (แตกจาก main `5aa135e7f`) · ป้าย: [fact] เว้นแต่ระบุ

## สถานะล่าสุด
- **แผน GRD ทำครบทั้ง 4 เฟสแล้ว รอเจ้าของกด merge PR เดียว** — P1 สัญญางานผูกแผน (plan-anchor + relay-call บังคับ + กฎ re-anchor ใน vault) · P2 memory-audit ตัวเทียบความจำ · P3 ด่านกัน stash กวาดงานคนอื่น · P4 ล้างความจำเก่า — ทุกชิ้นผ่านผู้ตรวจต่างค่าย + เทสต์ scoped เขียว (154 เคสรวม: 64 relay + 10 memory-audit + 80 guards) [fact]
- ตัวเขียนโค้ดจริงของรอบนี้: **Grok เป็นหลัก** (Codex ชนโควต้าตั้งแต่ใบแรก) · ใบแก้สุดท้าย Gemini · ผู้ตรวจ = Claude ทุกใบ [fact]
- **ชุดเทสต์เต็ม repo แดงอยู่ก่อนแล้ว**: `pytest -q` ที่ฐาน main = 683 failed / 24,193 passed (จุดตกอยู่ใน tests/cli, tests/gateway ที่งาน GRD ไม่ได้แตะ) — gate-run จดเป็นแถวแรกใน `.hermes/ledger/` แล้ว · เป็นงานซ่อมแยกรอบ [fact]
- PR #15 (แก้ auth ปลอมใน relay-call) merge เข้า main แล้ว — main HEAD = `5aa135e7f` [fact]
- สาย JARVIS v2: รอเจ้าของทดสอบเสียง P0 แล้วเปิดแชตใหม่ส่ง Use AI Relay [fact]

## งานถัดไป
1. **เจ้าของ: ตรวจ + กด merge PR แผน GRD** (branch `feature/plan-guardrails` · 5 commit)
2. GRD-P5 Monitor Hub เริ่มเมื่อ PR merge + เจ้าของสั่ง
3. GRD-P6..P8 รอเจ้าของส่ง "ปัญหาชุดสุดท้าย" ก่อนล็อกดีไซน์

## ข้อห้าม/กติกาล็อก
- ห้ามเขียนความจำทำงานต่อลง `.hermes/` หรือ root — เขียน `.project/` เท่านั้น (Schema v1.2)
- หลังสร้าง/ย้ายไฟล์ `.project/` ต้องผ่านด่าน `git check-ignore` + `git ls-files` ก่อนบอกเสร็จ
- **เลขงานต้องขึ้นต้นด้วย plan_id (เช่น GRD-P1-I1) · เลขที่ไม่มีใน plan.md = ห้ามทำ** · หลังตอบคำถามแทรก ต้องเปิด plan.md ทวนเฟสก่อนลงมือ (กติกาเหล็กของแผน GRD)
- ห้ามแตะ `scripts/jarvis-voice/` + `design-system-standard-v2/` + `.claude/launch.json` (งานเจ้าของค้าง · คนละสาย)
- ห้าม merge→main / deploy เอง — เจ้าของกด · งานหลายเฟส = 1 PR เดียว
- สมองแผน GRD = Fable ตามคำสั่งเจ้าของ 2026-07-07 (ข้อยกเว้นจากกติกา relay v2.7 ที่ปกติใช้ Opus) · Codex/Claude เขียน-ตรวจสลับค่ายผ่าน relay-call · **verified = มีแถว gate-run เท่านั้น**

## งานค้าง/ส่งต่อ
- รอเจ้าของ: **กด merge PR แผน GRD** · ส่งปัญหาชุดสุดท้าย (ปลดล็อก GRD-P6..P8) · rotate GitLab token (ค้างจาก 2026-07-04) · สั่ง commit ไฟล์ JARVIS untracked (`scripts/jarvis-voice/` — ระวัง `.venv-gemini` ต้องใส่ .gitignore ก่อน) · ติดตั้ง memory-audit รายสัปดาห์ (ถ้าต้องการ): `(crontab -l 2>/dev/null; echo '0 9 * * 1 cd "/Users/rattanasak/Documents/Viber Project/Tech Tools/Hermes Agent" && ./venv/bin/python scripts/memory-audit/memory_audit.py >> ~/.claude/ai-fail-stats/memory-audit.log 2>&1') | crontab -`
- **งานซ่อมแยกรอบ (ใหม่ 2026-07-08): ชุดเทสต์เต็ม repo แดง 683 เคสที่ฐาน main** — ทำให้ gate-run ตัดสิน pass ไม่ได้ทั้ง repo · ควรไล่ซ่อมหรือกำหนด gate ย่อยที่เขียวได้จริง (เสนอดูดเข้า GRD-P8)
- โควต้า AI คืน 2026-07-08: Codex + Grok ชนโควต้าทั้งคู่ช่วงดึก · Gemini crash ตอนจบแต่เขียนไฟล์สำเร็จ — เช็กโควต้าก่อนเริ่มงานใหญ่รอบถัดไป
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
