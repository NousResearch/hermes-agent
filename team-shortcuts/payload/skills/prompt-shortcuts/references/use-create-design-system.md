---
title: Use Create Design System
aliases:
  - Use Create Design System
  - use-create-design-system
  - Create Design System
  - create-design-system
  - Use Design System
  - use-design-system
  - ใช้ Design System
  - ใช้ Create Design System
  - สร้าง Design System
  - ทำ Design System มาตรฐาน
  - วาง Design System ให้โปรเจกต์
tags:
  - prompt-shortcuts
  - design-system
  - tokens
  - accessibility
status: active
version: "2.4"
created: 2026-06-26
updated: 2026-07-05
schema: memory-schema-v1.1
standard_ref: design-system-standard-v2/
---

# Use Create Design System / Use Design System (v2.3 · 2026-07-05)

## เลือกตัวไหนในกลุ่มงานดีไซน์ (ป้ายบอกทาง · 2026-07-04)

| โจทย์ | ใช้ตัวไหน |
|---|---|
| ตรวจ/ซ่อมหน้าที่มีอยู่ (งานลวก AI slop, responsive พัง, ความสวยไม่ผ่าน) | `Use Impeccable` |
| วางหรือรื้อระบบดีไซน์ทั้งโปรเจกต์ (token, สี, component มาตรฐาน) | `Use Create Design System` |
| หาไอเดีย/แพทเทิร์น layout-เอฟเฟกต์ระดับพรีเมียมก่อนลงมือสร้าง | `Use WOW Resource` |

ลำดับที่ถูกเมื่อทำเว็บใหม่: WOW Resource (หาไอเดีย) → Create Design System (วางระบบ) → Impeccable (ตรวจก่อนส่ง)

คู่กับ Memory Schema v1.1 · เช็ก schema version ตอนเริ่ม ไม่ตรง = เตือน

> คำสั่งกลาง · ใช้ได้กับทุกโปรเจกต์ขององค์กร (30+ ตัว) ไม่ผูกกับโปรเจกต์ใดโปรเจกต์หนึ่ง
>
> หน้าที่: พิมพ์คำสั่งนี้ในโปรเจกต์ไหนก็ได้ แล้ว AI จะ (0) ถามก่อนว่าจะทำ Design System สำหรับ "หน้าเว็บ (Front)" หรือ "ระบบหลังบ้าน (Application)" (1) อ่านเป้าหมาย/แผนธุรกิจ → เสนอบุคลิกดีไซน์โดยอิงดีไซน์ระดับโลกตามโหมดที่เลือก (2) ค้นของเดิม + ประเมินช่องว่าง (3) หยุดถามเจ้าของเรื่องสีแบรนด์ 4 สี + น้ำหนัก แล้วต่อยอดเป็นชุดสีครบ (4) สร้างไฟล์มาตรฐาน + ปรับโปรเจกต์ให้ตรงมาตรฐานแบบไม่พังของเดิม
>
> มาตรฐานกลางตัวจริงที่คำสั่งนี้ยึด (v2.3): ชุด `design-system-standard-v2/` ใน repo Hermes Agent (spec + emotion-matrix + checklist 77 + tokens + tools + ENTRY-FOR-OTHER-PROJECTS.md) · **หาชุดได้ทั้ง Mac (โฟลเดอร์ local) และ VPS/เครื่องอื่น: `git clone --depth 1 https://github.com/rattanasak-ops/hermes-agent.git` แล้วใช้ `design-system-standard-v2/`** — อย่าฝัง path เครื่องเดียว · ไฟล์เก่า `60-Design/design-systems/design-system-standard.md` เป็นฉบับอ้างอิงประวัติ ไม่ใช่ตัวรัน

## ใหม่ v2.1 · เครื่องมือรันจบในตัว — โปรเจกต์ไม่ต้องอ่านมาตรฐานเอง

**หลักการส่งมอบ (คำสั่งเจ้าของ 2026-07-05):** คำสั่งนี้คือ "เครื่องมือ" ที่เอาไปรันใน 30-40 โปรเจกต์
ไม่ใช่เอกสารให้โปรเจกต์มาเปิดอ่าน · AI ที่รับคำสั่งนี้ต้อง**อ่านมาตรฐานแทนเจ้าของ**แล้ว**ลงมือสร้างของจริงลงในโปรเจกต์**:

สิ่งที่เครื่องมือ "วางลงในโปรเจกต์" เมื่อรันจบ:
1. `design/tokens/` — ไฟล์ token DTCG ของโปรเจกต์นั้น (คัดจาก seed กลาง + สีแบรนด์ + ผลลูกโซ่ F/G) + `build-tokens.mjs` + ผลสร้าง css/ts
2. `design/ds-check.py` + `.ds-allowlist` — ตัวตรวจ hardcode พร้อมใช้ (โหมดเตือนก่อน)
3. `.project/DesignSystem.md` — สรุปการตัดสินใจ 1 หน้า (โหมด · preset · ลูกโซ่ F · คะแนนล่าสุด) ให้ AI แชทถัดไปอ่านต่อ
ต้นทาง seed กลาง: `design-system-standard-v2/` ใน repo Hermes Agent (Mac local หรือ clone จาก GitHub — ดู path portable ข้างบน) · สมองของเครื่องมือ — เจ้าของไม่ต้องเปิด

### ลำดับรันเต็ม (v2.2): H → F → G → A → B/C → E → D
- **H · บริบทโครงการ (ก่อนทุกอย่าง · spec/05-project-identity.md):** หาอ่านเอกสารที่โปรเจกต์มีอยู่แล้ว — ไฟล์แบรนด์ canonical / แผนธุรกิจ / แผนการตลาด / roadmap → กรอกบัตรประจำตัวโครงการ (ชื่อ·วิสัยทัศน์·พันธกิจ·ภาษานำ·เฟส) + เหตุผลทุกสีผูกพันธกิจ + ตาราง pain→คำตอบดีไซน์ + ตรวจความสอดคล้องข้ามเอกสาร (โค้ด vs เอกสารขัดกัน = รายงาน ห้ามเลือกเอง) · ปิดงานดีไซน์ทุกครั้งต้องผ่าน Direction Check 4 คำถาม · ทุก showcase มีปุ่มสลับ TH/EN
- **F · คน/แบรนด์ (ก่อน token เสมอ):** กรอกลูกโซ่ แบรนด์(archetype 12)→เป้าหมาย→Persona(+floor)→**แรงจูงใจ**→อารมณ์ 6 แกน(รวม Luxe)→ฟังก์ชัน · ตาม `spec/02-emotion-matrix.md`
- **G · ธุรกิจ/conversion:** ถ้าเป็น SaaS/งานขาย เตรียมชิ้นส่วน G1-G8 (hero positioning · หน้าราคา · funnel · trust · onboarding · ลำดับเล่าเรื่อง · อีเมล · จุดวัดผล) ตาม `spec/04-business-conversion.md` · ข้อความดึงจากผล Use Business Plan ห้ามแต่งเอง
- **A-C · token + ของจริง:** สร้าง/อัปเดต token 3 ชั้น + component ตามเช็กลิสต์ 77 หัวข้อ (H8+A17+B13+C9+D9+E7+F6+G8)
- **E · ชั้นหรูตาม preset:** ถาม 1 คำถาม "งานนี้ระดับไหน: เรียบราชการ / องค์กรทั่วไป / พรีเมียม" → ตั้ง preset off/standard/premium (`spec/03-expressive-layer.md`) · Luxe ≥4 = เปิด glow/gradient/display/glass
- **D · กำกับดูแล:** ผูก ds-check เข้า CI (เตือนก่อน→เข้ม) + บันทึก `.project/DesignSystem.md`

### โหมดทำงานร่วม Use AI Relay (v2.7 · ไม่ใช้ Fable/Faber/Fiber 5 · คุณภาพเท่ากันด้วย "ด่าน" ไม่ใช่ตัวโมเดล)
คำสั่งนี้รันได้กับสมองทุกระดับ เพราะคุณภาพถูกบังคับด้วยด่านที่เป็นโค้ด ไม่ใช่ดุลยพินิจของโมเดล:
- **สมองวางแผน:** Opus 4.8 (ค่าเริ่มต้นของ relay) — อ่านชุดมาตรฐาน + กรอกลูกโซ่ H/F/G + เขียน brief · ไม่ใช้ Fable/Faber/Fiber 5 ในเส้นทางปัจจุบัน
- **คนเขียนโค้ด:** Codex หรือ Grok ผ่าน `relay-call --tool <coder> --task-id <งาน> --prompt-file <brief>` · brief ต้องแนบกติกาเหล็ก: ทุกข้อความ 2 ภาษา (data-th/data-en) · สีข้อความเฉพาะชุดที่ผ่าน AA · น้ำหนักฟอนต์ 300/400/500/600/700 · ทุกหัวข้อต้องมองเห็นบนหน้าโชว์
- **คนตรวจ:** AI อีกตัวที่ไม่ใช่คนเขียน (เขียน Codex → ตรวจ Grok และกลับกัน)
- **ด่านตัดสิน (บังคับผ่านก่อนปิดงาน · ใครทำก็ด่านเดียวกัน):** `ds-check.py` คะแนนใช้ token ≥95% + 0 จุดร้ายแรง · `contrast-check.mjs` exit 0 ทุกคู่ · `node build-tokens.mjs` exit 0 · เกรดดีไซน์ 10 มิติ ไม่มี D/F รวม ≥B · เช็กลิสต์ 77 ข้อ 🔴 ครบ 100%
- ผลลัพธ์: output คุณภาพเดียวกันไม่ว่าใครเขียน เพราะงานที่ไม่ผ่านด่าน = ไม่ปิด วนแก้จนผ่าน

### ก้าว 0 · อ่านโปรเจกต์ให้เข้าใจก่อน (บังคับ · ห้ามข้าม)
ก่อนแตะอะไร ต้องสแกนโปรเจกต์แล้วสรุปให้เจ้าของเห็นก่อน:
- ใช้ tech อะไร (framework, วิธีเขียน CSS, Tailwind/Storybook ไหม)
- มี token/ธีมเดิมไหม อยู่ที่ไหน · แบรนด์/สีเดิม · หน้าเด่น/หน้าที่ใช้บ่อย · กลุ่มผู้ใช้
- โปรเจกต์นี้เป็น production (เว็บลูกค้าจริง) หรือไม่ → ถ้าใช่ บังคับ MIGRATION ห้าม GREENFIELD ทับ
ไม่เข้าใจโปรเจกต์ = ยังไม่เริ่มทำ

### วง PDCA (วัดว่า AI ทำได้ดีแค่ไหน · ออกเป็นตัวเลขทุกรอบ)
| ขั้น | ทำอะไร | ตัวเลขที่วัด |
|---|---|---|
| P วางแผน | อ่านโปรเจกต์ + กรอกตารางอารมณ์ + เทียบเช็กลิสต์ 77 หัวข้อ | เช็กลิสต์ผ่าน % |
| D ลงมือ | สร้าง/อัปเดต token เข้าหน้าจริง (คง alias เดิม) | จำนวนหน้าที่ต่อ token |
| C ตรวจ | รัน `tools/ds-check.py` + ตรวจ contrast + เทียบภาพก่อน/หลัง | **คะแนนใช้ token %** · จุด hardcode เหลือ · contrast ผ่านกี่คู่ |
| A ปรับ | แก้จุดที่ยังฝังตายตัว แล้ววนซ้ำ C | วนจน token ≥ 95% + 0 จุดร้ายแรง |

### คุมดีไซน์ทั้งโปรเจกต์ (ทำให้ "ใช้ระบบเดียวกันจริง")
ผูก `ds-check.py --mode error` เข้า CI → ใครเขียนสี/ขนาดฝังตายตัวใหม่ = build แดง เข้าไม่ได้
= ทุกหน้าถูกบังคับให้ใช้ token จากคลังกลางเท่านั้น

## Shortcut

```text
Use Design System        # หรือ Use Create Design System (ตัวเดียวกัน)
```

## Prompt

```text
Use Create Design System กับโปรเจกต์นี้

คุณคือ Senior Design System Engineer ขององค์กร งานของคุณคือทำให้โปรเจกต์นี้มี design
system ที่ตรงกับ "มาตรฐานกลางขององค์กร" — ไม่ว่าโปรเจกต์นี้จะมีของเดิมอยู่แล้วหรือยังไม่มี

[กฎ non-dev] เจ้าของงานอาจไม่ใช่ dev → อธิบายภาษาคน · ห้ามถามว่าใช้ test/tool ตัวไหน (ค้นเอง)
· เสนอทางที่ดีที่สุดให้ตัดสินใจต่อได้ · ห้ามโยน option ดิบให้เลือกเอง

[ขั้นเลือกโหมด — ทำก่อนทุกอย่าง · บังคับ · หัวใจของคำสั่งนี้]
ถามเจ้าของเป็นข้อแรกสุด: "จะทำ Design System สำหรับแบบไหน"
  (ก) หน้าเว็บ (Front) — เว็บไซต์ข้อมูล/การตลาดที่คนทั่วไปเห็น
      ชิ้นส่วนเด่น: hero, ปุ่มเรียกร้อง (CTA), บทความ, การ์ดเนื้อหา, หน้าแพ็กเกจ/ราคา, footer
      เน้น: ความสวย, ความเร็วโหลด, SEO, ภาพใหญ่, ความโปร่ง
  (ข) ระบบหลังบ้าน (Application) — แอปจัดการข้อมูลภายใน/แอดมิน
      ชิ้นส่วนเด่น: ตารางข้อมูล (เรียง/กรอง/เลือกหลายแถว/แก้ในแถว), ฟอร์ม, แดชบอร์ด, ตัวกรอง,
      เมนูข้าง, สถานะ 5 แบบ (ว่าง/กำลังโหลด/ผิดพลาด/มีบางส่วน/สมบูรณ์), UI ที่รู้สิทธิ์ผู้ใช้
      เน้น: ความหนาแน่นสูง (compact), ข้อมูลเยอะ, ประสิทธิภาพ, ตารางหมื่นแถว (virtualization)
- โปรเจกต์มีทั้งสองแบบ = ทำทีละโหมด ไม่ปนกัน
- โหมดที่เลือกกำหนด: ชุดชิ้นส่วนที่สร้าง, รูปแบบ (pattern), ความหนาแน่น, และตัวอย่างดีไซน์ระดับโลกที่หยิบมาอ้าง
- ใช้ร่วมกันทั้งสองโหมด (ไม่ต้องเลือก): รากฐานสี/โทเคน, มาตรฐาน DTCG, WCAG 2.2 AA, กฎกันพังของเดิม
- บันทึกโหมดที่เลือกลงทุก output (เช่น PROJECT-DESIGN.md ระบุ `mode: front` หรือ `mode: app`)

[ขั้น 0 — อ่านความจำ + มาตรฐานก่อนวางแผน · บังคับ]
- อ่าน `.project/OverviewProgress.md` + `.project/plan.md` + `.project/decisions.md` (Memory Schema v1.2 · ไฟล์เก่า handoff/.hermes = fallback อ่านได้ ย้ายตาม §1b) · ห้ามรื้อ decision เดิมโดยไม่บอก
- อ่านมาตรฐานกลาง design-system-standard.md (ที่ราก repo ถ้ามี ไม่งั้นใช้ฉบับใน 60-Design/design-systems/)
- หาความจำไม่เจอ = บอกว่าไม่พบ ไม่เดาว่าเคยทำอะไร

[กฎเหล็ก — อ่านก่อนทำทุกครั้ง]
1. ห้ามแก้/ลบไฟล์จำนวนมากทันที — ทำ Phase 0–2 ให้เสร็จและ "เสนอแผน" ก่อนลงมือแก้จริง
2. ห้ามทำลายการใช้งานเดิม (เข้ากันได้กับของเก่า) — จะ rename token/component เดิม ให้คง alias เดิม + mark deprecated แทนการลบ
3. ทำเป็นรอบเล็ก ๆ commit ทีละหมวด ไม่รวบยอดทีเดียว
4. ก่อนเปลี่ยนแปลงเชิงทำลาย (ลบ/rename/ย้ายไฟล์จำนวนมาก) ต้องถามยืนยันก่อนเสมอ
5. ค่าทุกอย่างต้องมาจาก token ห้าม hardcode สี/ขนาด/ระยะ ลงใน component ตรง ๆ
6. [กฎ migration — INC-9085] โปรเจกต์ที่มีธีมอยู่แล้ว ต้องถามเจ้าของก่อนว่า "แก้ของเดิมในที่เดิม (in-place)
   หรือสร้างหน้าโชว์แยก (showcase) เก็บของเดิม" · ห้ามตีความเอง

# PHASE 0 — กำหนดทิศทางจากเป้าหมาย/แผนธุรกิจ (ขั้นเสนอแนว · อิงโหมดที่เลือก)
- อ่าน brief/แผนธุรกิจ/เป้าหมายโปรเจกต์ (ถ้าไม่มีไฟล์ ให้ถามเจ้าของสั้น ๆ ว่ากลุ่มเป้าหมาย/อารมณ์แบรนด์คืออะไร)
- ถอดเกณฑ์: ประเภทงาน (gov/fintech/enterprise/startup/luxury/content) · กลุ่มผู้ใช้ · ระดับพรีเมียม · ภาษาไทยไหม · ฟังก์ชันหลัก
- จับคู่กับคลังดีไซน์ระดับโลก (60-Design/design-systems/ + ~/.claude/design-library/ 15 ระบบพร้อมคะแนน)
  **กรองตัวเลือกตามโหมดที่เลือกในขั้นเลือกโหมด:**
    - โหมดหน้าเว็บ (front) → หยิบระบบที่เด่นด้านเนื้อหา/อารมณ์/การตลาด (เช่น Airbnb, Stripe, Vercel, แนวภาครัฐถ้าเป็นงาน gov)
    - โหมดหลังบ้าน (app) → หยิบระบบที่เด่นด้าน data-heavy/แอดมิน (เช่น IBM Carbon, Shopify Polaris, Ant Design, AWS Cloudscape, Atlassian)
  เลือก 3–5 ระบบที่เหมาะสุด → แสดงตารางเทียบ + เหตุผล → เสนอ "บุคลิกดีไซน์" 1 แนวที่แนะนำ
- หยุดรอเจ้าของยืนยันแนว ก่อนไป Phase 1

# PHASE 1 — ค้นหา (Discovery)
สแกนทั้งโปรเจกต์ว่ามี design system เดิมไหม ค้นด้วย pattern:
  มาตรฐาน: design-system-standard.md, DESIGN_SYSTEM*.md, CLAUDE.md, AGENTS.md, .cursor/rules/*
  Token/Theme: **/*token*.{css,json,js,ts}, **/ds-tokens.*, **/ci.*, **/dna.*, **/theme*.{css,ts,js,json},
               **/tokens/**, **/globals.css, tailwind.config.*, postcss.config.*, :root{ --... }
  หน้า DS: **/design-system*, **/style-guide*, .storybook/**, **/*.stories.*
  Component: **/components/**, **/ui/**, **/lib/components/**
  Multi-tenant: **/*tenant*, **/themes/**, **/brand*, TenantSwitcher*, ThemeProvider*
ระบุ tech stack (framework, วิธีเขียน CSS, มี Tailwind/Storybook ไหม) + จัดของที่เจอเข้าหมวดมาตรฐาน

# PHASE 2 — ประเมินช่องว่าง (Gap Audit)
รายงานเทียบของที่เจอกับมาตรฐาน แต่ละหมวดให้สถานะ: ✅ ตรงมาตรฐาน · ⚠️ มีแต่ไม่ครบ · ❌ ยังไม่มี
บอกด้วยว่าของเดิมส่วนไหน "ดี ควรเก็บ" และส่วนไหน "ควรปรับ"

# PHASE 3 — ด่านสี (Color Gate · ขั้นใหม่ · หัวใจของเจ้าของงาน)
- ถามเจ้าของ: "สีแบรนด์หลัก 4 สี จะใช้สีอะไร และน้ำหนัก/สัดส่วนเท่าไหร่" (เช่น สีหลัก/สีรอง/สีเน้น/สีกลาง)
- จาก 4 สีนั้น AI ต่อยอดให้ครบเป็นชุดสีมาตรฐานอัตโนมัติ:
    semantic เต็ม (action/success/warning/danger/info/neutral) + surface/text/border + โหมดสว่าง+มืด
- เช็กสีตัดกันพออ่านออกตาม WCAG 2.2 AA (ข้อความ ≥ 4.5:1, UI ≥ 3:1) → ถ้าตก ปรับน้ำหนักให้ผ่าน + แจ้งเจ้าของ
- สรุปเป็นตารางสีให้เจ้าของอนุมัติ "ทั้งชุด" (เจ้าของตัดสิน 4 สี · ได้ระบบสีครบ) → หยุดรอยืนยันก่อนไป Phase 4

# PHASE 4 — ตัดสินเส้นทาง + ลงมือ (ทำตามแผนที่อนุมัติ ทีละหมวด)
- ถ้า Phase 1 ไม่เจอ design system เลย → GREENFIELD: scaffold ใหม่ทั้งชุด (token 3 ชั้น → component → showcase)
- ถ้าเจอของเดิม → MIGRATION: ปรับของเดิมเข้ามาตรฐาน เก็บส่วนดี เปลี่ยนเฉพาะที่ไม่ตรง คงความเข้ากันได้เดิม
- ก่อนลงมือจริง: ยืนยันโหมด in-place/showcase (กฎเหล็กข้อ 6) แล้วเสนอแผน (ลำดับงาน + ไฟล์ที่จะแตะ + ความเสี่ยง) "หยุดรอยืนยัน"
ยึดมาตรฐานเหล่านี้ (รายละเอียดเต็มใน design-system-standard.md):
  - Token 3 ชั้น: primitive → semantic/alias → component · แยก 2 เลเยอร์ ci (แบรนด์/ลูกค้า) + dna (ระบบ/semantic)
       ห้ามใช้ primitive ตรง ๆ ใน component
  - รูปแบบไฟล์ token: W3C DTCG (2025.10) ใช้ $value/$type/$description แล้ว map เป็น CSS custom properties
  - Multi-tenant theming: แยก brand token ออกจาก system token สลับธีมรายลูกค้าได้ runtime ผ่าน CSS variables
  - โครงไฟล์: 1 component = 1 โฟลเดอร์/ไฟล์ + มี showcase/story (แบบ SEC) ไม่เอาไฟล์เดียวยาวรวมทุกอย่าง
  - Accessibility: WCAG 2.2 AA — contrast ≥ 4.5:1, target ≥ 24x24px, focus ชัด, keyboard ได้, semantic HTML + ARIA,
       เคารพ prefers-reduced-motion, ห้ามสื่อด้วยสีอย่างเดียว
  - **ชุดชิ้นส่วน + ความหนาแน่น = ตามโหมดที่เลือก (ใช้รากฐานสี/โทเคนชุดเดียวกัน แต่สร้างคนละชุด component):**
      - โหมดหน้าเว็บ (front) → hero, CTA, การ์ดเนื้อหา, บทความ, หน้าราคา, footer · ความหนาแน่นโปร่ง · effects/expressive ใช้เต็มที่
      - โหมดหลังบ้าน (app) → ตารางข้อมูล (เรียง/กรอง/เลือกหลายแถว/sticky header/virtualization), ฟอร์ม+validation, แดชบอร์ด,
        ตัวกรอง, เมนูข้าง, สถานะ 5 แบบ (ว่าง/โหลด/ผิดพลาด/บางส่วน/สมบูรณ์), UI ที่รู้สิทธิ์ผู้ใช้ · ความหนาแน่นสูง (compact) · effects คุมให้แน่น
  - Versioning: semver + changelog · ของเลิกใช้ = mark deprecated ไม่ลบทันที

# PHASE 5 — สรุปผล
ปิดงานด้วย: สรุปสิ่งที่เปลี่ยน, รายการไฟล์ที่สร้าง/แก้, สิ่งที่ยัง deprecated รอลบรอบหน้า,
checklist หมวดที่ยังขาด · แต่ละ claim ผูกหลักฐานตามบันได Schema §4 (เช่น "สีผ่าน WCAG: contrast 4.8:1")

# ทางลัดกลไก (ใช้ได้เหมือนกันทั้ง VPS + Notebook — AI รันให้เจ้าของ ไม่ต้องจำหลาย step)
มี `design-system-standard-v2/ds-adopt.sh` รวมงานกลไกไว้ 2 คำสั่ง:
- **ก่อนวางแผน:** `bash design-system-standard-v2/ds-adopt.sh prep <target-project>` — ติดตั้ง playwright (ครั้งแรก) + build token + วัด baseline
- **ก่อนปิดงาน:** `bash design-system-standard-v2/ds-adopt.sh check <target-project> <หน้าโชว์.html>` — รันด่านครบ (build + ds-check + brand-leak + contrast-audit) · **exit 1 = ยังไม่ผ่าน ห้ามปิดงาน**
- ชุดไม่มีในเครื่อง (VPS): ดึงก่อน `git clone --depth 1 https://github.com/rattanasak-ops/hermes-agent.git /tmp/hermes-ds && cp -r /tmp/hermes-ds/design-system-standard-v2 <โปรเจกต์>/`
เจ้าของแค่พิมพ์ `Use Design System` — AI รัน 2 คำสั่งนี้ให้เอง (เตรียมตอนต้น · ตรวจตอนปิด)

# เริ่มงาน
เริ่มจาก `ds-adopt.sh prep <target>` แล้ว "ขั้นเลือกโหมด" (ถามหน้าเว็บ/หลังบ้าน) → PHASE 0 รายงานแนวดีไซน์ให้เจ้าของก่อน อย่าเพิ่งแก้โค้ด · ปิดงานต้องผ่าน `ds-adopt.sh check` ก่อนเสมอ

ข้อห้าม: รื้อ decision เดิมโดยไม่บอก · แก้ไฟล์ก่อนอนุมัติแผน · hardcode สี · ตีความ in-place/showcase เอง
· เคลม "เสร็จ/ผ่าน" โดยไม่มีตัวเลขหลักฐาน · ศัพท์เทคนิคไม่แปลไทย
```

## เคล็ดการใช้งานจริง

- **รอบแรกของแต่ละโปรเจกต์**: ให้ AI รันถึงแค่ Phase 2 (รายงานช่องว่าง) ก่อน เจ้าของอ่าน/อนุมัติแผน ค่อยปล่อย Phase 4 — กัน AI รื้อมั่ว
- **เว็บลูกค้าจริง (production เช่น MOL/DRA)**: เน้น MIGRATION + เข้ากันได้กับของเดิม อย่าให้ GREENFIELD ทับของเดิม
- **อยากให้ทุกโปรเจกต์เหมือนกัน**: วาง design-system-standard.md + คำสั่งนี้ ลงทุก repo หรือทำเป็น shared template repo

## Changelog

- docs (2026-07-04): เพิ่มตารางป้ายบอกทางกลุ่มงานดีไซน์
- v1.1 (2026-06-26): เพิ่ม "ขั้นเลือกโหมด" เป็นข้อแรกสุด — เจ้าของเลือกได้ว่าจะทำ Design System สำหรับ "หน้าเว็บ (Front)" หรือ "ระบบหลังบ้าน (Application)" · โหมดที่เลือกกำหนดชุดชิ้นส่วน/รูปแบบ/ความหนาแน่น/ตัวอย่างดีไซน์ที่หยิบมาอ้าง (PHASE 0 + PHASE 4 แตกตามโหมด) · ใช้รากฐานสี/โทเคน/มาตรฐาน/กฎกันพังร่วมกัน · ย้ำว่าเป็นคำสั่งกลางใช้ทุกโปรเจกต์ 30+ ตัว ไม่ผูกโปรเจกต์เดียว
- v1.0 (2026-06-26): สร้างครั้งแรก · เย็บ 3 ส่วนเข้าด้วยกัน — Phase 0 (อ่านแผนธุรกิจ → เสนอแนวจากคลังดีไซน์ระดับโลก, ต่อยอดจากสกิล design-selector), Phase 1–2 + 4–5 (prompt มาตรฐาน 5 เฟสของเจ้าของงาน), Phase 3 ด่านสีใหม่ (ถาม 4 สีแบรนด์ → ต่อยอดชุดสีครบ + เช็ก WCAG) · ฝังกฎ migration INC-9085 · ยึดมาตรฐานกลาง design-system-standard.md

## Graph Links

- Parent hub: [[skills/prompt-shortcuts/Prompt Shortcuts|Prompt Shortcuts]]
- Registry: [[ai-context/prompt-shortcut-registry|Prompt Shortcut Registry]]
- Schema: [[skills/prompt-shortcuts/references/memory-schema|Memory Schema v1.1]]
- Standard: [[60-Design/design-systems/design-system-standard|Design System Standard]]
- Library: [[60-Design/design-systems/README|Design Systems Library]]

## v2.4 (2026-07-07) · บทเรียนจากสนามจริง (OneManFleet 3 วัน) — บังคับ

ทุก project ที่ใช้ shortcut นี้ ต้องได้ผลลัพธ์: **HTML โชว์ที่มองเห็นได้จริง (Front + Admin) + Token 2 โปรไฟล์ (front + admin)** และต้องผ่านด่านล่างนี้ก่อนปิดงาน (ผูกกับ `checklist ชั้น A+`):

1. **ตัวอ่านหลัก = 16px มาตรฐาน** (token body = `fontSize.16`) · ป้ายจิ๋วเท่านั้นที่เล็กกว่าได้
2. **contrast ผ่านหน้าเรนเดอร์จริง ทั้ง Front และ Admin = 0 fail** — รัน `node tools/contrast-audit-run.mjs <หน้า.html>` (headless · exit 0 = ผ่าน · ครั้งแรก `npm i playwright` ในโฟลเดอร์ tools) หรือวาง `tools/contrast-audit.js` ใน DevTools Console · `contrast-check.mjs` (static) เป็นแค่ pre-check เสริม ไม่พอ (เคยพลาดข้อความบนการ์ดพื้นสว่าง)
3. **เปิด HTML โชว์ได้จริงทั้ง 2 โปรไฟล์** — ส่ง URL/ไฟล์ให้เจ้าของกดดูได้ (คนไม่อ่านโค้ด)
4. **เช็คคำแปลไทยสื่อแบรนด์** + ห้ามตัวเข้มบนพื้นเข้ม / ตัวขาวเล็กบนแดงสด `#EC2C23`

⚠ **onemanfleet-ds.html = ตัวอย่างให้ดูโครงเท่านั้น ห้ามลอกสี/แบรนด์**: `design-system-standard-v2/preview/onemanfleet-ds.html` + `tokens/{front,admin}.tokens.json` เป็น **โครงตัวอย่าง 1 เคส** (สี/แบรนด์เป็นของ OneManFleet) — ลอกได้แค่ "โครงสร้าง/ลำดับชั้น/ชิ้นส่วน" เท่านั้น · **สี/แบรนด์/ข้อความต้องสร้างของโปรเจกต์เองผ่านด่านสี Phase 3** แล้วรัน `tools/brand-leak-check.sh` ให้ผ่าน (ต้องไม่เหลือสี OneManFleet `#E94560`/`#1A1A2E` ในหน้าโชว์โปรเจกต์)
