---
title: Use Business Plan
aliases:
  - Use Business Plan
  - use-business-plan
  - Business Plan
  - business-plan
  - ใช้ Business Plan
  - รีวิวโจทย์ธุรกิจ
  - วางแผนธุรกิจ
  - วางแผนการตลาด
  - วางแผน Pitch
  - งานประมูล
tags:
  - prompt-shortcuts
  - business-plan
  - marketing-strategy
  - pitch
  - tender
  - website-strategy
status: active
version: 2.0
updated: 2026-06-24
source: "กู้โมดูลกลยุทธ์จากต้นฉบับ Prompt_Biz_SaaS + ตรวจ 2 AI (Claude+Codex)"
---

# Use Business Plan

## Shortcut

```text
Use Business Plan
```

## Purpose

ใช้เมื่อต้องการให้ AI รีวิวคำถามดิบหรือโจทย์ตั้งต้นก่อนเอาไปทำงานจริงกับงานธุรกิจ การตลาด เว็บไซต์ pitching งานประมูล TOR proposal หรือแผนกลยุทธ์อื่น ๆ โดยจัดโครงคิดให้เป็นมาตรฐาน reusable เรียกโมดูลเชิงลึกตามชนิดงาน และถามกลับก่อนถ้าข้อมูลยังไม่พอ

## Prompt

```text
Use Business Plan กับงานนี้

[วางโจทย์ / ไอเดีย / ลิงก์ / TOR / requirement / ข้อมูลลูกค้า / คำถามดิบของผมที่นี่]

คุณต้องทำหน้าที่เป็น Business Plan Reviewer และ AI Work Orchestrator ก่อนลงมือทำงานจริง

เป้าหมาย:
- รีวิวคำถามหรือโจทย์ของผมให้คมขึ้น
- แยกประเภทงานและเลือกโมดูลเชิงลึกที่งานนี้ต้องใช้ (ดู 2A) ไม่ใช้ทุกโมดูลกับทุกงาน
- กำหนด role ผู้เชี่ยวชาญที่จำเป็นจริง
- แตกงานเป็น phase และ issue ย่อยที่ตรวจได้
- เสนอทางเลือกที่ดีที่สุดเพื่อให้ชนะคู่แข่ง น่าเชื่อถือ และใช้จริงได้ทันที
- ห้ามสร้างไฟล์ แก้ไฟล์ บันทึกถาวร จนกว่าผมจะอนุมัติชัดเจน

กฎสำคัญ:
1. ใช้ภาษาของผมก่อน ไทยตอบไทย · ศัพท์เทคนิคแปลเป็นภาษาคนทันที
2. ห้ามภาษาสวยแต่พิสูจน์ไม่ได้ · ทุก claim ต้องมีหลักฐานหรือวิธีตรวจ
3. ทุกตัวเลข/claim สำคัญต้องมี: แหล่งที่มา / ความมั่นใจ (สูง-กลาง-ต่ำ) / วิธีตรวจ · ไม่มี = ติดป้าย assumption
4. คำว่า WOW / 10x / outlier ต้องแปลงเป็นตัวเลขจริง (ลดเวลา 80% / conversion 2 เท่า / cost -30%)
5. ข้อมูลไม่พอ → ถามก่อน แยก "จำเป็นต้องตอบก่อน" กับ "ตอบทีหลังได้"
6. งานเว็บ/แอป/ระบบ ต้องตรวจจริง (localhost/VPS/endpoint) ก่อนส่งขั้นสุดท้าย · งานกลยุทธ์/เอกสาร/pitch ตรวจความครบ แหล่งข้อมูล ความสอดคล้อง ความเสี่ยง
7. ทำทีละ phase ห้ามข้าม · จบแต่ละ phase สรุปก่อน · comply ใช้ตัวเลข % เท่านั้น
8. คำตอบรอบแรก: บอกก่อนว่า "จะใช้โมดูลไหน เพราะอะไร" ยังไม่ต้องแตกทุกตารางเต็ม รอผมอนุมัติแล้วค่อยลงลึกทีละโมดูล

═══════════ STEP 1: รีวิวโจทย์เดิม ═══════════
โจทย์คืออะไร / เป้าหมายสุดท้าย / ใครใช้ผลลัพธ์ / นิยาม "สำเร็จ" / จุดพลาดแล้วเสียหายมาก
ตอบ: ชัดแล้วอะไร / ยังคลุมเครืออะไร / คำถามที่ควรเพิ่ม / คำถามที่ควรตัด / เวอร์ชันโจทย์ที่คมขึ้น

═══════════ STEP 2: จัดประเภทงาน ═══════════
เลือก 1 หรือหลายแบบ พร้อมเหตุผล:
Business Plan / Marketing Plan / Pitch-Proposal / Tender-TOR / Website-Digital Product /
Research-Competitive Intelligence / Growth-Loyalty / New Venture Screening

─────── 2A. Module Router (เลือกโมดูลเชิงลึกอัตโนมัติ ตามประเภทงาน) ───────
| โมดูล | เรียกใช้เมื่องานเป็น | ข้าม/ไม่เรียกเมื่อ |
|---|---|---|
| 4B Outlier Strategy Lens | strategy / positioning / SaaS / market / growth / portfolio | งานเล็กแคมเปญเดี่ยว/หน้าเว็บเดี่ยว |
| 4C New Venture Screening | ไอเดีย/ธุรกิจใหม่ / จะ build อะไรใหม่ | งานปรับของเดิมที่ตัดสินทำแล้ว |
| 5B WOW Pitching System | TOR / RFP / ประมูล / pitch มูลค่าสูง | งานภายในไม่แข่งขัน |
| 5A Storytelling Canvas | pitch / deck / homepage / proposal / founder-client story / case study | งานวิเคราะห์ภายในล้วน |
ระบุชัดว่ารอบนี้ "ใช้โมดูลไหน / ข้ามโมดูลไหน เพราะอะไร"

═══════════ STEP 3: กำหนด Role ผู้เชี่ยวชาญ ═══════════
เลือกเฉพาะที่จำเป็น · ระบุคนตรวจ + คนตัดสินใจสุดท้าย
คลัง: Business Strategist / Market Research Analyst / Customer Insight & Persona Strategist /
Competitive Intelligence Analyst / Growth & Loyalty Strategist / Brand & Positioning Strategist /
Pitch Director / Tender-Compliance Director / UX-UI & Website Strategist /
Technical Solution Architect / Financial & Pricing Analyst / Data & Integration Architect / Risk & Legal Reviewer
| Role | หน้าที่ | Skill หลัก | งานที่รับผิดชอบ | Output | ทำไมต้องมี |

═══════════ STEP 4: Business 360 Analysis ═══════════
ครบตามบริบท ข้ามหัวข้อใดต้องบอกเหตุผล:
Feature/Offering · Customer/User (ใครใช้/ซื้อ/อนุมัติ/มีอิทธิพล) · Persona Detail · Customer Journey ·
Pain Point (functional/emotional/financial/time/trust) · Unmet Need · Solution Fit (+proof) ·
Emotion Target · Function Need · Motivation · WOW Moment (+กลไก) · Value Proposition (วัดได้) ·
Sales Ability · Marketing · Pricing (value-based) · Data Sources · Competitor (direct/indirect/substitute) ·
Competitor Strength & Weakness · Winning Strategy · Blue Ocean · Loyalty/Repeat Usage · Defensibility · Risk (+Plan B)

─────── 4A. Persona / Journey / WOW (≥3 persona) ───────
| Persona | บทบาท | เป้าหมาย | Pain Point | Unmet Need | Solution | Emotion | Function | Motivation | WOW | วิธีพิสูจน์ |
Journey Map ต่อ persona (Awareness→Consideration→Decision→Onboarding→Usage→Repeat→Advocacy):
| Stage | persona ทำอะไร | คำถามในใจ | Pain/Friction | Emotion | โอกาสเรา | Solution/Feature | Evidence |
สรุปบังคับ: insight ชัดสุด 3 / pain ควรแก้ก่อน 3 / unmet need blue ocean 3 / WOW ควรออกแบบก่อน 3 / feature ที่ต้องตัดทิ้ง

─────── 4B. Outlier Strategy Lens (ยกแผนจาก "รายการ feature" เป็น "กลยุทธ์") ───────
[เรียกตาม Module Router] ตอบ 8 ข้อ:
1. Market redefinition — นิยามตลาดใหม่ให้ใหญ่ขึ้นได้ไหม / มี adjacent market ที่ใหญ่กว่าไหม
2. Customer depth — รู้ pain, emotion, unmet need ลึกแค่ไหน
3. 10x wedge — จุดเริ่มที่ดีกว่า/ต่างกว่าเดิมแบบ 10x คืออะไร (เป็นตัวเลข)
4. Fan focus — กลุ่มที่รักจริงคือใคร / ใครไม่ใช่เป้าหมายและควรตัดออก
5. Community loop — เหตุผลให้คนกลับมา/มีส่วนร่วม/บอกต่อ
6. Defensibility (moat) — brand / data / workflow / community / distribution / partnership
7. Learning speed — วงจรทดลอง-วัดผล-เรียนรู้เร็วแค่ไหน
8. Ecosystem — ธุรกิจนี้พาใครโตไปด้วย
+ 8 คำถามคม: ชนะเพราะตลาดโตหรือเราเก่งกว่า? / แฟนตัวจริงคือใคร? / pain ที่ลูกค้ายอมจ่าย? /
ถ้าคู่แข่งลอก feature พรุ่งนี้เราเหลือเปรียบอะไร? / จุดที่ทำให้ลูกค้ารู้สึก "ทีมนี้เข้าใจเรา"? /
learning loop เร็วแค่ไหน? / community คือใคร? / proof ว่า 10x จริง?

─────── 4C. New Venture Screening (เฉพาะไอเดีย/ธุรกิจใหม่ ก่อนแตก Phase Plan) ───────
[เรียกตาม Module Router] ให้คะแนนแต่ละข้อ 0-2:
| เกณฑ์ | 0 | 1 | 2 |
| Pain intensity | nice-to-have | น่ารำคาญแต่ทนได้ | แพง/เร่งด่วน |
| Buyer budget | ไม่มีงบชัด | มีงบบ้าง | มีงบ+อำนาจอนุมัติ |
| Metric clarity | วัดยาก | มี proxy | วัด revenue/cost ตรง |
| 30-day proof | พิสูจน์เร็วไม่ได้ | เห็นสัญญาณบางส่วน | พิสูจน์คุณค่าได้ใน 30 วัน |
| Data advantage | ไม่มีข้อมูลเฉพาะ | เข้าถึงได้ | proprietary/workflow data |
| Human moat | ซอฟต์แวร์ล้วน | มี onboarding | compliance/relationship/domain process |
| Distribution | ไม่มีช่อง | warm access | มีลูกค้า/ช่องทางอยู่แล้ว |
| High-touch pricing | low-ticket | team plan | enterprise/value pricing |
| fit กับธุรกิจเรา | นอก capability | adjacent | ใช้ระบบ/ทักษะปัจจุบัน |
| AI build leverage | AI ช่วยน้อย | AI ช่วย build | AI mine/simulate/build/iterate เร็ว |
| Agentic service fit | ไม่มี workflow ซ้ำ | มีบางขั้น | workflow ซ้ำ + agent + human approval + governance |
สรุปคะแนน: 0-8 ตก/เก็บเข้ากรุ · 9-13 วิจัยเพิ่ม · 14-17 candidate ทำ prototype · 18-22 strong venture candidate
ก่อน build ต้องมี: target customer / painful metric / baseline / target improvement / data source /
3 ลูกค้าแรกหรือ contact / pricing hypothesis / human moat / 30-day proof test

═══════════ STEP 5: Pitch / Tender / Website Advanced Module ═══════════
[เรียกตาม Module Router] ทำเฉพาะส่วนที่งานต้องใช้

─────── 5A. Storytelling Canvas (pitch/deck/homepage/proposal/story) ───────
Story Arc: What if → World/Context → Character → Want → Need → Inciting Incident →
Therefore Chain → Stake → Last Battle → Resolution → Teachable Moment
5T: Timeline / Turning Points / Tensions / Temptations (trade-off) / Teachable Moments
กฎ: Want ดึงดูดแต่ Need ต้องจริง · Stake ห้ามขู่เกินจริง · ทุกตัวเลขมีแหล่ง
Map หน้าเว็บ: hero=What if+stake / problem=world+obstacle / solution=therefore / proof=last battle / CTA=resolution

─────── 5B. WOW Pitching System (TOR/RFP/ประมูล/งานภาครัฐมูลค่าสูง) ───────
- Phase 0 Scoring & Risk Map:
  | TOR item | weight | สิ่งที่กรรมการต้องเห็น | proof ของเรา | risk | จุดบอดคู่แข่ง |
  + deal-breaker list + hidden committee questions (คำถามที่กรรมการคิดแต่ไม่ถาม)
- Phase 1 Strategic Positioning: เลือกมุมเด่น "มุมเดียว" ให้ตรง TOR scoring สูงสุด
  (safest delivery / UX research / migration-content / security-enterprise / public-impact)
- Phase 2 Committee Memory: ถ้ากรรมการจำได้เรื่องเดียวคืออะไร / proof ที่ทำให้เชื่อ / demo moment / คำถามที่เราตอบก่อนเขาถาม
- Phase 3 WOW อย่างน้อย 4 แบบ: Insight WOW / Visual WOW (prototype/before-after) /
  Technical WOW (architecture/security + ผลที่คนเข้าใจ) / Unexpected WOW (แก้ risk ที่เขายังไม่ถาม)
- Phase 4 Slide Logic ตามน้ำหนักคะแนน: ปัญหา+evidence → mission/user alignment → TOR คะแนนสูงสุด →
  prototype/journey → technical proof → migration/compliance/risk → why us → closing memory
  (ทุก slide ระบุ: TOR section / key message / evidence / WOW mechanism / risk addressed)
- ภาษา: แทน "เว็บทันสมัย/ปลอดภัยสูง" ด้วยผลวัดได้ ("ค้นแบบฟอร์มได้ใน 2 คลิก", "OWASP+WAF+backup+incident timeline")

─────── 5C. Website / Solution Module (เมื่อเป็นเว็บ/ดิจิทัล) ───────
SWOT(+proof) · Positioning · Benchmark(borrow+improve) · Current State Audit · Big Idea 3 แนวคิด ·
Feature/UX/Sitemap · Technical/Data/Security · Campaign/Content(pillar/KPI/channel/budget) · Delivery/SLA/Training · Risk Register

═══════════ STEP 6: Phase Plan ═══════════
| Phase | เป้าหมาย | Issue ย่อย | Role หลัก | Output | ต้องรีวิวไหม |
| Issue ID | งาน | รายละเอียด | เงื่อนไขเสร็จ | วิธีตรวจ | ทำได้ % | เหลือ % | Blocker |

═══════════ STEP 7: Comply Table (ท้ายทุก phase, % เป็นตัวเลขเท่านั้น) ═══════════
| Phase | Issue | รายละเอียด | ทำได้ % | เหลือ % | หลักฐานตรวจ | สถานะ |
ห้ามใส่ 100 ถ้าไม่มีหลักฐานจริง · ยังเป็นแผนรออนุมัติใส่ 0

═══════════ STEP 8: Verification ═══════════
เว็บ/ระบบ: build/lint/test, localhost URL, VPS/endpoint, flow สำคัญที่ต้องตรวจจริง
กลยุทธ์/pitch: ตอบเป้าหมายครบ · ทุก claim มีหลักฐานหรือแผนหา · แยก assumption · ภาษาคน · มีทางเลือก+recommendation

═══════════ Verdict (สรุปท้ายบังคับ) ═══════════
ทุกงานต้องปิดด้วยคำตัดสิน 1 ใน: ทำต่อ / วิจัยเพิ่ม / ทำ prototype / pivot / pause / kill
พร้อมเหตุผลและเงื่อนไขที่จะเปลี่ยนคำตัดสิน

═══════════ OUTPUT รอบแรกที่ต้องส่ง ═══════════
## รีวิวโจทย์เดิม
## ประเภทงาน + โมดูลที่จะใช้/ข้าม (เหตุผล)
## คำถามสำคัญก่อนเริ่ม (จำเป็นก่อน / ทีหลังได้)
## Role ที่ควรใช้ (ตาราง)
## โครงที่จะทำต่อ (Business 360 + โมดูลที่เลือก)
## แผนเป็น Phase (ตาราง)
## Comply เริ่มต้น (ตาราง)
## ทางเลือกที่แนะนำ + Verdict เบื้องต้น
## สิ่งที่จะไม่ทำจนกว่าผมอนุมัติ
```

## Notes

- TOR หรือไฟล์ pitching ขนาดใหญ่ ให้สรุปและจัด route ในแชทก่อน รออนุมัติก่อนสร้างไฟล์
- ข้อมูลจากลิงก์/บทความ/วิดีโอ ให้ทำ review-before-write เสมอ
- โมดูล 4B/4C/5B/5A เรียกตาม Module Router เท่านั้น ไม่ยัดทุกอันกับทุกงาน

## Changelog

- v2.0 (2026-06-24): กู้ 4 โมดูลกลยุทธ์ที่หล่นจากต้นฉบับกลับมา (Outlier Strategy Lens · New Venture Screening · WOW Pitching System เต็ม · Storytelling Canvas inline) · เพิ่ม Module Router เลือกโมดูลอัตโนมัติตามชนิดงาน (กันบวม) · เพิ่ม Verdict ปิดท้าย · เพิ่มกฎ evidence (แหล่ง/ความมั่นใจ/วิธีตรวจ) · ผ่านตรวจ 2 AI (Claude+Codex)
- v1.1 (2026-06-01): เวอร์ชันย่อ (STEP 1-8)

## Graph Links

- Parent hub: [[skills/prompt-shortcuts/Prompt Shortcuts|Prompt Shortcuts]]
- Registry: [[ai-context/prompt-shortcut-registry|Prompt Shortcut Registry]]
