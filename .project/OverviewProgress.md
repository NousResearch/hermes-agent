> memory-schema: v1.2
> อ่านตามลำดับ: plan.md → decisions.md → hermes-standard/REQUIREMENTS.md (บัญชีความต้องการ 66 ข้อ)

# Overview & Progress — Hermes Agent
อัปเดตล่าสุด: 2026-07-06 · branch: `main` (HEAD `da4689a58`) · ป้าย: [fact] เว้นแต่ระบุ

## สถานะล่าสุด
- **P0-P1 ของแผน "Hermes ใช้งานจริงทั้งระบบ" เสร็จ merged แล้ว** — PR #12 merge เข้า main 2026-07-06 (`da4689a58`) · Project OS ครบ 4/4 · ความจำอยู่ `.project/` · ด่านไฟล์เข้า git จริงเข้ากติกากลาง 7 จุด [fact]
- PR #10 (relay P3) merged 2026-07-05 · แผนเก่า AI Relay Hardening v2 ปิด: P1-P3 verified (pytest 16/16) · P4 เดิมยุบเข้า P5 แผนใหม่ [fact]
- curse-tracker อัปคำด่า v2 + hook ai-fail-stats ขึ้น main แล้ว (`06dbc2a53` · test 14/14) — ISSUE-004 "ตัวจับแคบ" ปิดฝั่งโค้ดแล้ว [fact]
- สาย JARVIS v2 ผู้ช่วยเสียง (แชตแยก · 2026-07-06): เปลือกเสียงใช้ได้จริง (ปุ่มลัด+เปิดเองตอนเข้าเครื่อง+แก้ช้า/เสียงย้อน) · ระบบเก่า v1 ลบถาวร 9/9 · แผน 8 เฟส P0-P7 อนุมัติแล้วที่ `.project/FeatureSpec-jarvis-voice.md` · รอเจ้าของทดสอบ P0 แล้วส่ง Use AI Relay [fact]

## งานถัดไป
1. P5 Monitor Hub (URL เดียวบน VPS + การ์ดเช้าเข้า Lark) — ปลดล็อกแล้วเพราะ P0-P1 merged · เดินผ่านสายพาน (Codex เขียน · Grok ตรวจ) เมื่อเจ้าของสั่งเริ่ม
2. รอเจ้าของส่ง "ปัญหาชุดสุดท้าย" → ล็อกดีไซน์ P2-P4 (มาตรฐานกลางวิ่งจริง + วงจรคำด่า + ย้ายน้ำหนัก hook)
3. สาย JARVIS v2: รอเจ้าของทดสอบ P0 เสียง แล้วเปิดแชตใหม่ส่ง Use AI Relay ไล่ P1→P7

## ข้อห้าม/กติกาล็อก
- ห้ามเขียนความจำทำงานต่อลง `.hermes/` หรือ root อีก — เขียน `.project/` เท่านั้น (Schema v1.2)
- หลังสร้าง/ย้ายไฟล์ `.project/` ต้องผ่านด่าน `git check-ignore` + `git ls-files` ก่อนบอกเสร็จ
- ห้ามแตะ `scripts/jarvis-voice/` + `design-system-standard-v2/` + `.claude/launch.json` (งานเจ้าของค้างใน worktree · คนละงาน)
- ห้าม merge→main / deploy เอง — เจ้าของกด · งานหลายเฟส = 1 PR เดียว
- Fable ใช้เฉพาะงานเกรดยาก (catalog §6) · Codex/Grok เขียน-ตรวจสลับค่ายผ่าน relay-call · verified = gate-run เท่านั้น

## งานค้าง/ส่งต่อ
- รอเจ้าของ: ส่งปัญหาชุดสุดท้าย (ก่อนล็อก P2-P4) · rotate GitLab token (AI ทำเองได้ผ่าน server — ค้างจาก 2026-07-04) · สั่ง commit ไฟล์งาน JARVIS ที่ยัง untracked (`scripts/jarvis-voice/` + `.project/FeatureSpec-jarvis-voice.md` — อยู่เครื่องเดียว เสี่ยงหาย)
- claimed (ยังไม่ตรวจ): iptables :3010 ไม่ persistent ข้าม reboot · webhook pr-review ลงแค่ project เด็กฝึก 527
- feature ค้างจากตารางรีวิว Hermes 2026-07-03: F3-F8
- อัปรุ่น v0.18.0 = P6 ในแผน (ยังไม่เริ่ม · ต้องทำบัญชีของต่อเติมก่อน)
- **เหตุการณ์ P0 เปิดค้างตั้งแต่ 2026-06-07**: ต้องมีด่านกัน AI ลบโฟลเดอร์งาน/worktree ทั้งก้อน (เคยลบ EmailHunter workspace แล้วต้องกู้จาก branch snapshot) · รายละเอียด `.hermes/issues/phase-013-destructive-workspace-retirement-guard.md` · ควรถูกดูดเข้า P4 (ด่านกั้นก่อนทำ)

---

## project นี้คืออะไร (2-3 บรรทัด)
ศูนย์เครื่องมือ AI ส่วนตัวของเจ้าของ (fork จาก NousResearch/hermes-agent v0.17.0 + ของต่อเติม ~3,215 commit): สายพาน AI Relay ประหยัดเงิน · ชุด shortcut คุมวินัยงาน · มาตรฐานกลาง 30-40 โปรเจกต์ (hermes-standard) · เครื่องมือคุมคุณภาพ (violation-audit, pr-review-gate, curse tracker) · gateway ให้ทีม 15 คนใช้บน VPS [fact]

## เสร็จแล้ว (verified) + ประวัติย่อ
- 2026-07-06: P0-P1 merged เข้า main — PR #12 (`da4689a58`) · check-ignore ว่าง + ls-files 6/6 [fact]
- 2026-07-05: กู้ shortcut Project OS 3 ตัว (ถูก revert `fff10805b` เมื่อ 2026-06-28 โดยความจำยังจดว่าครบ) กลับเข้าคลัง+ทะเบียน+ชุดแจก · เพิ่มด่านไฟล์เข้า git จริงทั้งระบบ — หลักฐาน: commit `f079acf47`
- 2026-07-05: relay-call เพิ่มนาฬิกาปลุกกันค้าง (timeout) — pytest 16/16 · ผ่าน GPT-5.5 review (แผนเก่า P1)
- 2026-07-04-05: relay P3 ครบ 4/4 (registry.yaml · relay-status · relay-suggest · RELAY-RULES.md) — PR #8, #9 merged · #10 เปิดค้าง
- 2026-07-04: F1 violation-audit + F2 pr-review-gate ใช้จริง (tier 3) · AI Relay ยืนยันทั้ง notebook + VPS
- ก่อนหน้า: ดู decisions.md + session log ใน vault (`projects/hermes-agent-dev/`)
