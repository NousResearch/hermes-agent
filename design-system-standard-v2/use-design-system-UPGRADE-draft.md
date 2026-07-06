# ร่างอัปเกรดคำสั่ง Use Create Design System → รองรับ Use Design System (v2)

> ร่างไว้ก่อน · ยังไม่ทับไฟล์จริงในคลัง (`skills/prompt-shortcuts/references/use-create-design-system.md`)
> รอเจ้าของอนุมัติเช้าค่อย promote · กันแก้ระบบกลางตอนไม่มีคนเฝ้า

## สิ่งที่ต้องเปลี่ยน 5 จุด

1. **เพิ่มชื่อเรียก** — เพิ่ม alias `Use Design System`, `use-design-system`, `ใช้ Design System` เข้า frontmatter (ตอนนี้มีแต่ `Use Create Design System`)

2. **ผูกมาตรฐาน v2** — เปลี่ยน `standard_ref` ให้ชี้ชุดใหม่ `design-system-standard-v2/` (spec + emotion-matrix + checklist + tokens) แทนไฟล์เดี่ยวเดิม

3. **แทรกชั้นอารมณ์เป็นเฟสบังคับ** — เพิ่ม "Phase 0.5 · ตารางอารมณ์" ก่อนด่านสี:
   - กรอกลูกโซ่ แบรนด์→เป้าหมาย→persona→อารมณ์(5 แกน 1-5)→ฟังก์ชัน
   - แปลงเป็นค่า token ด้วยสูตรใน `spec/02-emotion-matrix.md`
   - ไม่ครบ = ไม่ผ่าน (กันอารมณ์ลอย)

4. **ตรรกะ ไม่มี=สร้าง / มี=อัปเดต (ชัดขึ้น)**
   - สแกนหา token เดิม (Phase 1 เดิม) → ไม่พบ = GREENFIELD สร้าง 2 โปรไฟล์จาก DTCG ต้นทาง
   - พบ = MIGRATION เทียบ `checklist/` ทีละข้อ → อัปเดตเฉพาะที่ ❌/⚠️ · คง alias เดิม
   - รันตัวตรวจ `tools/ds-check.py` ปิดงาน รายงานคะแนนใช้ token %

5. **ด่านหยุดก่อนแก้จริง (เพิ่มความปลอดภัย)**
   - ก่อนเขียนไฟล์จริง: ถาม in-place/showcase (INC-9085 เดิม) + ต้องผ่าน "โหมดเตือน" ของตัวตรวจก่อน
   - เว็บ production (DRA/ก.ล.ต./Content Thailand) = บังคับ MIGRATION + เทียบภาพก่อน/หลัง ห้าม GREENFIELD ทับ

## จุดที่ "ไม่แตะ" (คงเดิม)
- 5 เฟสเดิม + human color gate ยังอยู่ (แค่เสริมชั้นอารมณ์ก่อนด่านสี)
- กฎ migration INC-9085 เดิม
- โครงการ์ด non-dev (อธิบายภาษาคน)

## การจับคู่เฟส (กันสับสน)
- คำสั่งนี้ = "เอามาตรฐานไปใช้ต่อโปรเจกต์" (ทำซ้ำทุก repo)
- ชุด v2 8 เฟส = "สร้างมาตรฐานกลาง" (ทำครั้งเดียว — งานคืนนี้)
- ไม่ทับกัน
