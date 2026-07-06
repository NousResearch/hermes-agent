# มาตรฐาน Design System กลาง (v2) — ข้อกำหนด (Spec)

> ไฟล์กลางขององค์กร · ทุกโปรเจกต์ยึดตามนี้ · เรียกผ่าน `Use Design System`
> คู่กับ: [ตารางอารมณ์](02-emotion-matrix.md) · [เช็กลิสต์](../checklist/design-system-checklist.md) · [token](../tokens/)
> อิงสากล: W3C DTCG 2025.10 · WCAG 2.2 AA · Material 3 · IBM Carbon

## 1. หลักการ (บังคับ)
1. ทุกค่ามาจาก token · ห้าม hardcode สี/ขนาด/ระยะ ใน component
2. token 3 ชั้น: primitive → semantic → component · ห้ามใช้ primitive ตรงใน component
3. แยก 2 เลเยอร์: `ci` (แบรนด์/ลูกค้า) + `dna` (ระบบ)
4. ต้นทางเดียว = DTCG json → สร้าง css + ts อัตโนมัติ (กัน 3 ไฟล์เพี้ยน)
5. อารมณ์แบรนด์ต้องแปลงเป็นค่า token ตัวเลข (ดูตารางอารมณ์) ไม่ปล่อยลอย
6. เข้ากันได้กับของเก่าเสมอ · เปลี่ยนชื่อ token = คง alias + deprecated ≥ 2 รอบ

## 2. โครงสร้าง 2 โปรไฟล์
| | หน้าเว็บ (Front) | แอดมิน (Admin) |
|---|---|---|
| เน้น | อารมณ์ · ความสวย · โหลดเร็ว · SEO | ข้อมูลแน่น · ประสิทธิภาพ · งานเยอะ |
| ความหนาแน่น | โปร่ง (1.0×) | แน่น (0.75×) · แถว ≤ 40px |
| component เด่น | hero · CTA · บทความ · ราคา · footer | ตาราง · ฟอร์ม · แดชบอร์ด · ตัวกรอง · 5 สถานะ · สิทธิ์ |
| ระบบสากลอ้างอิง | Airbnb · Stripe · Vercel · gov | Carbon · Polaris · Ant · Cloudscape · Atlassian |
| ใช้ร่วมกัน | token/สี/ฟอนต์/อารมณ์/การเข้าถึง/โหมดสี | (ชุดเดียวกัน) |

## 3. 50 หัวข้อมาตรฐาน (4 ชั้น)
รายการเต็ม + ระดับบังคับ อยู่ใน [เช็กลิสต์](../checklist/design-system-checklist.md)
- ชั้น A รากฐาน token (17) · ชั้น B component (13) · ชั้น C รูปแบบ+เนื้อหา+เข้าถึง (11) · ชั้น D กำกับดูแล+ท่อผลิต (9)
- เกณฑ์ผ่าน: ข้อ 🔴 บังคับต้องครบ 100% (รวม 🔴A/🔴F ตามโปรไฟล์)

## 4. มาตรฐาน token (ยึด DTCG 2025.10)
- ชนิดพื้นฐาน 8: color(OKLCH) · dimension · fontFamily · fontWeight · duration · cubicBezier · number · string
- ชนิดประกอบ 6: border · strokeStyle · transition · shadow · gradient · typography
- ใช้คีย์ `$value` `$type` `$description` · อ้างค่าด้วย `{alias}`
- หมวดที่ต้องมีครบ: สี(+dark) · ตัวอักษร · spacing · sizing · grid · breakpoints · radius · elevation · z-index · opacity · motion · icon

## 5. การเข้าถึง (WCAG 2.2 AA · บังคับ)
- ข้อความ contrast ≥ 4.5:1 · UI ≥ 3:1 · target ≥ 24×24px (ผู้สูงอายุ ≥ 44px)
- focus เห็นชัด (2.4.11/2.4.13) · คีย์บอร์ดครบ · semantic HTML + ARIA
- ไม่สื่อด้วยสีเดียว · เคารพ `prefers-reduced-motion` · รองรับโหมดเข้มสูง

## 6. งานรัฐไทย (เมื่อเข้าเกณฑ์)
- DGA 3.0 · PDPA · กฎฟอนต์ไทย (TH/EN · line-height · ขั้นต่ำอ่านยาว · พ.ศ./ตัวเลขไทย)

## 7. การกำกับดูแล
- เจ้าของมาตรฐาน + ด่านรีวิว token ใหม่ · semver + changelog
- ตัวตรวจ hardcode (โหมดเตือน→เข้ม) · คะแนนการใช้จริงต่อโปรเจกต์
- ทดสอบใน CI: visual regression · axe a11y · contrast อัตโนมัติ

## 8. การนำไปใช้ (ผ่านคำสั่ง)
- โปรเจกต์ไม่มี DS → วิเคราะห์ → กรอกตารางอารมณ์ → สร้าง token 2 โปรไฟล์
- มีแล้ว → เทียบเช็กลิสต์ → อัปเดตเฉพาะที่ไม่ตรง (คง alias เดิม)
- ก่อนแก้ไฟล์จริง: ถาม in-place/showcase (INC-9085) + ทดลองโปรเจกต์ไม่ critical ก่อน
