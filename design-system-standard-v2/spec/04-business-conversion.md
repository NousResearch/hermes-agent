# 04 · Business & Conversion Layer — แปลงคลังธุรกิจ/การตลาดเป็นชิ้นส่วน UI มาตรฐาน

> สังเคราะห์จากคลังจริงของเจ้าของ (2026-07-05):
> `skills/prompt-shortcuts/references/use-business-plan.md` (Business 360: Persona/Journey 7 ขั้น · Emotion/Function/Motivation/WOW · Storytelling Canvas · WOW Pitching)
> `use-saas-opus-master-prompt.md` (15 หมวด: Positioning · Pricing · Marketing Funnel · Agentic Marketing OS · Pitch System)
> เป้า: SaaS + งานหลากหลาย 30-40 โปรเจกต์ ใช้ชิ้นส่วนชุดเดียวกัน

## หลักคิด
แผนธุรกิจบอก "ต้องสื่ออะไร" · Design System ต้องมี "ชิ้นส่วนสำเร็จรูปที่สื่อสิ่งนั้น"
ถ้าไม่มีชิ้นส่วนรองรับ ทุกโปรเจกต์จะสร้างหน้าราคา/หน้าขายใหม่จากศูนย์ = ช้า + ไม่เป็นมาตรฐาน + วัดผลไม่ได้

## G1 · โครงข้อความ Hero (จาก Positioning)
- แม่แบบ: Positioning statement (1 บรรทัด) + 3 ข้อความหลัก + 3 หลักฐาน (proof point)
- component: hero-headline · subhead · proof-strip (3 ช่อง) · CTA คู่ (หลัก/รอง)
- กติกา: headline ต้องมาจาก positioning จริงของโปรเจกต์ (ห้ามคำลอย "ดีที่สุด" ไม่มีตัวเลข)

## G2 · แม่แบบหน้าราคา (จาก Business Model & Pricing)
- component: ตาราง tier (3-4 คอลัมน์ + เน้นตัวแนะนำ) · สวิตช์รายเดือน/รายปี · แถว enterprise ("คุยกับทีมขาย") · FAQ ราคา · เปรียบเทียบ feature ต่อ tier
- ผูก token: tier แนะนำใช้ glow/accent จากชั้น E ได้เมื่อ preset เปิด

## G3 · ชิ้นส่วนตาม Funnel 7 ขั้น (awareness→referral)
| ขั้น | ชิ้นส่วนมาตรฐาน |
|---|---|
| awareness | hero + content card + SEO block |
| lead | ฟอร์มเก็บ lead สั้น (ชื่อ+อีเมล) + lead magnet card |
| demo/trial | ปุ่มทดลอง/นัดเดโม + ฟอร์มจองเวลา |
| activation | เช็กลิสต์เริ่มใช้ + progress + empty state ที่สอนใช้ |
| paid | หน้าราคา (G2) + upgrade prompt ในแอป |
| retention | แจ้งเตือนใช้งาน + สรุปคุณค่ารายเดือน (usage recap) |
| referral | บล็อกชวนเพื่อน + ส่วนลด/รางวัล |

## G4 · ชิ้นส่วนความเชื่อมั่น (Trust & Proof) — ตอบ Motivation "มั่นใจก่อนจ่าย" (งาน 10-20 ล้าน)
- แถบโลโก้ลูกค้า · การ์ดรีวิว/คำนิยม · การ์ด case study (ปัญหา→ผลลัพธ์เป็นตัวเลข) · ตัวเลขผลงานนับขึ้น (ผูก count-up จากคลังเอฟเฟกต์) · ป้ายมาตรฐาน/รางวัล (ISO/DGA/awards) · ทีมงาน+ใบหน้า

## G5 · Onboarding & Activation UI (จาก Journey ขั้น Onboarding→Usage)
- welcome screen · เช็กลิสต์ 3-5 ก้าวแรก · tooltip นำทางครั้งแรก · empty state แบบ "ชวนทำ" (ไม่ใช่แค่ว่างเปล่า) · แถบความคืบหน้าโปรไฟล์

## G6 · ลำดับเล่าเรื่องของหน้า (จาก Storytelling Canvas)
- แม่แบบลำดับ section หน้า landing: ปัญหา → ทางแก้ → WOW moment → หลักฐาน → ราคา → CTA ปิด
- แม่แบบหน้า pitch (จาก WOW Pitching): Insight WOW → Visual WOW (before/after) → ตัวเลข → ทีม → ข้อเสนอ
- กติกา: ทุกหน้า landing ต้องระบุว่าใช้ลำดับไหน (ห้ามเรียง section มั่ว)

## G7 · Lifecycle นอกแอป (จาก retention + ชั้น C10 เดิม)
- แม่แบบอีเมล: welcome / ใบเสนอราคา / นัดเดโม / renewal / win-back · ใช้ token ชุดเดียวกับเว็บ (สี/โลโก้/ฟอนต์)

## G8 · จุดวัดผล (จาก KPI + Learning loop + Agentic Marketing OS)
- ทุก CTA หลักต้องมีชื่อ event มาตรฐาน (`cta_hero_primary`, `pricing_tier_select`, ...) ฝังใน component ตั้งแต่แรก
- แดชบอร์ดหลังบ้านมี KPI card ตาม funnel (ใช้ B9 เดิม) · consent/PDPA banner เป็น component มาตรฐาน (ผูก C11)

## เกณฑ์ใช้ต่อโปรเจกต์
- SaaS/งานขาย = G1-G8 บังคับ · เว็บข้อมูล/ราชการ = G1, G4, G6, G8 พอ (ไม่มีหน้าราคา)
- แหล่งข้อความจริง: ดึงจากผล `Use Business Plan` / `Use BusinessPlan` (.project/BusinessPlan.md) ของโปรเจกต์นั้น — ห้ามเขียนคำโฆษณาเอง
