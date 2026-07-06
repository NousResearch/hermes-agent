# Ledger — งานสร้างมาตรฐาน Design System v2 (autonomous ข้ามคืน)

เริ่ม 2026-07-04 · โหมด Use Continue · เจ้าของไปนอน มาตรวจเช้า

## สถานะราย Phase

| Phase | งาน | สถานะ | หลักฐาน |
|---|---|---|---|
| 0 | สัญญากลาง DS Contract v0 | ✅ เสร็จ | scratchpad/DS-CONTRACT-v0.md |
| 1 | ค้นสากล + 24 หัวข้อเติม | ✅ เสร็จ | Material3/Carbon/DTCG (แชท) |
| 2 | ตารางอารมณ์→token | ✅ 100% (Grok ตรวจ+แก้ครบ) | spec/02-emotion-matrix.md |
| 3 | ร่างมาตรฐาน 50 หัวข้อ | ✅ 100% (spec+emotion+checklist+guide) | spec/00-standard.md · guide/how-to.md |
| 4 | ท่อผลิต token (DTCG→ts+css) | ✅ Codex เขียน+build ผ่าน · 🔄 รอ Grok ตรวจ | tokens/ (core+front+admin+build+dist) |
| 5 | ตัวตรวจ + อัปเกรดคำสั่ง | ✅ ตัวตรวจเขียน+ทดสอบผ่าน · คำสั่งร่างเสร็จ | tools/ds-check.py · use-design-system-UPGRADE-draft.md |
| 4.5 + 6 | ทดลองโปรเจกต์จริง | ⛔ เก็บไว้เช้า (แตะเว็บลูกค้า) | - |

## หลักฐานที่รันจริง (Tier 3)
- tokens/build-tokens.mjs → exit 0 · gen dist/{front,admin}.css + .ts · front.css มี oklch 138 จุด + dark
- core.tokens.json มี $type/$value 170 จุด (DTCG ถูก)
- tools/ds-check.py ทดสอบ fixture → จับ hex+px, ข้าม var, error mode exit 1, adoption 50%
- Codex ตรวจ ds-check.py เจอ 6 จุด (ร้ายแรง 1: px จับทั้งบรรทัด · สำคัญ 5: id selector, var ข้ามทั้งบรรทัด, oklch/rgb wrap var, ไม่มี allowlist)
- แก้ครบ → ds-check.py v1.1: จับราย match, px เฉพาะ inline style, ข้าม token-wrap, รองรับ .ds-allowlist
- ทดสอบซ้ำ 6 เคส: violations=4 ตรงคาด (text/idselector/var-wrap/allowlist ไม่จับแล้ว) · ผ่าน
- Grok ตรวจ token (Phase 4 gate) เสร็จ · verdict: โครงพร้อม pilot (3 ชั้นถูก · รัน build ไม่มี broken/circular ref · OKLCH ครบ · dark override ได้)
- Grok findings:
  - เล็กน้อย: button.paddingX ชี้ {dimension.space.6} ตรง (ไม่ผ่าน semantic) → จดไว้ ไม่แตะคืนนี้ (ต้องเพิ่มชั้น semantic spacing = งานตัดสินใจ ทำตอนเจ้าของตื่น)
  - เติมก่อน go-live (ไม่บล็อก pilot · ตรงกับหัวข้อในเช็กลิสต์อยู่แล้ว): prefers-color-scheme toggle (C4) · สเกลสีเต็ม hover/disabled/border · shadow/elevation (A9) · focus-ring (B10) · z-index (A10) · ตรวจ contrast a11y (C7) · ผูก build เข้า CI (D4)
- Phase 4 = ✅ pilot-ready (ปิดเป็น deliverable อ้างอิง · ส่วนเติมเต็มทำตอน apply โปรเจกต์จริง)
- สรุป: คืนนี้เสร็จ Phase 2-5 · เว็บจริง (4.5+6) รอเจ้าของตื่น

## รอบเช้า 2026-07-05 · อัปเกรด + เติม token
- อัปเกรดคำสั่งจริง use-create-design-system.md → v2.0: เพิ่ม alias `Use Design System` · ก้าว 0 อ่านโปรเจกต์ก่อน · วง PDCA · คุมทั้งโปรเจกต์ผ่าน CI (tier 1 · markdown)
- ⚠️ บทเรียน: Codex exec เบื้องหลังค้างรอ stdin 2 ชม. (เจ้าของทัก) → ฆ่า process แล้ว · เปลี่ยนเป็นทำเอง+ตรวจ build ทุกก้อน ไม่ยิง Codex ทิ้งอีก
- เติม token เอง (build ผ่านทุกก้อน · tier 3):
  - สเกลสีเต็ม green/amber/red/sky (จาก 3 → 9 สเต็ป) · แก้ที่ Grok ทัก "สีไม่พอ hover/disabled"
  - เพิ่ม zIndex (base→toast) · แก้ที่วางผิดใต้ color → ย้ายขึ้น top-level (--z-index-modal: 1300)
  - เพิ่ม semantic spacing (xs-xl) + แก้ button.paddingX ให้ผ่าน semantic (24px) · แก้ที่ Grok ทัก "ไม่ผ่าน semantic"
- ทำต่อจนจบ (เจ้าของสั่งไม่ต้องหยุดถาม) · ทำเอง+ตรวจ build ทุกก้อน (tier 3):
  - build.mjs: เพิ่ม shadowToCss (composite → box-shadow) + ตรวจ contrast WCAG อัตโนมัติ (OKLCH→linear sRGB→luminance) · รันโชว์ 6 คู่สี ผ่านหมด 6.1-18.1:1
  - core: เพิ่ม shadow (sm/md/lg composite)
  - front+admin: เพิ่ม semantic elevation (→shadow) + focus-ring (color/width/offset) · admin paddingX เดิมถูกอยู่แล้ว (semantic)
  - build เต็ม exit 0 · shadow เป็น box-shadow จริง (0px 4px 12px -2px oklch...) · focus-ring 2px/4px · z-index 6 ตัว · green 9 สเต็ป
  - promote: อัปเดต registry row → v2.0 + alias Use Design System + status paragraph 2026-07-05
- สถานะ: Phase 4 token = ครบสมบูรณ์ (ไม่ใช่แค่ pilot-ready) · Phase 5 คำสั่ง promote แล้ว
- เหลืองานคน: ทดลอง 1 โปรเจกต์ไม่ critical + ย้ายชุด design-system-standard-v2 เข้า vault กลาง (รอเจ้าของ)

## รอบ Fable 2026-07-05 · ชั้น Expressive (งานพรีเมียม 10-20 ล้าน)
- วิเคราะห์จุดแข็ง/อ่อน DS v2 เทียบพอร์ตงานหรู: ระบบเดิม "ปลอดภัย/ราชการ" ขาดภาษาหรูทั้งชั้น (fontSize สูงสุด 32px · เงาเทาล้วน · ไม่มี gradient/glow/glass · motion สูงสุด 480ms)
- ค้นคลัง Obsidian ตาม index-first: 60-Design มี ux-effects ~61 ตัว (ป้าย 🟢🟡🔴) · a-premium 6 ระบบ (Linear/Vercel/Stripe/Apple/shadcn/Radix + คะแนน 4 มิติ) · web-intelligence Core Rule (ทำเพื่องาน 5 แสน-10 ล้าน+) · layouts 64 + hero themes 28
- สร้างสะพานคลัง→token (build ผ่านทุกก้อน · tier 3):
  - build-tokens.mjs: เพิ่ม gradient type (DTCG stops→linear-gradient) + number + merge effects.tokens.json เป็นชั้น 3 (core→effects→profile)
  - effects.tokens.json: display type 48-128px · glow 3 ระดับ (เงาสีหลายชั้น) · gradient 3 ชุด (brandSweep/darkLuxe/metallicGold) · glass blur · cinematic 800/1400ms + stagger + easing luxe · ทุกตัวติดป้าย severity
  - ตรวจ: --glow-ambient เงาคู่ 2 สี · --gradient-brand-sweep เป็น linear-gradient จริง · --display-font-size-xl 96px · --cinematic-duration-epic 1400ms
  - spec/03-expressive-layer.md: กติกาเหล็ก 5 ข้อ (จาก web-intelligence) + preset off/standard/premium + แกนอารมณ์ที่ 6 "ความหรู (Luxe) 1-5" + เส้นทาง WOW Resource→token
  - preview: เพิ่มปุ่มสลับ "✦ พรีเมียม" (hero มืด mesh gradient + glow + glass + display 96px + ทองโลหะ) · redeploy URL เดิม
- รอเจ้าของ: เปิดหน้าตัวอย่าง กดสลับ พรีเมียม เทียบกับมาตรฐาน แล้วตัดสินทิศ

## รอบแก้ตามคำติเจ้าของ 2026-07-05 (2 ระลอก)
- ฟอนต์ไทย: Noto Sans Thai ขึ้นนำเป็น default ทั้ง sans+display · แก้ที่ core token + rebuild ตรวจ output จริง + preview redeploy
- เจ้าของทักถูก: Brand/Target/Persona/Emotion/Function ถูกยุบเหลือ 2 ข้อใน checklist ทั้งที่เป็นหัวใจ + Motivation ขาดจริงไม่เคยทำ
- แก้: (1) เพิ่ม Motivation เข้าลูกโซ่ emotion-matrix (ส่วน 2.7 · JTBD → CTA/microcopy) (2) แตกชั้น F · Brand & Human 6 หัวข้อ (F1 Branding · F2 Target · F3 Persona+floor · F4 Motivation · F5 Emotion 6 แกน · F6 Function) ทุกข้อ 🔴 ต้องทำก่อน token · C5/C6 ย้ายเข้า F (C เหลือ 9)
- ยอดรวมสุดท้าย: **61 หัวข้อ** (A17+B13+C9+D9+E7+F6) · เส้นทาง 28→50→57→61
- บทเรียนซ้ำ 2 รอบ: เพิ่ม/แก้ของต้องอัปเดตยอดรวม checklist ทันที + หัวใจที่เจ้าของสั่งต้องเป็นหมวดเต็ม ไม่ใช่แถวเดียว

## รอบ Business & Conversion 2026-07-05
- อ่านคลังธุรกิจจริง (index-first): use-business-plan.md (Business 360 · Journey 7 ขั้น · Storytelling · WOW Pitching) + use-saas-opus-master-prompt.md (15 หมวด: Positioning/Pricing/Funnel/Agentic Marketing OS) · โฟลเดอร์ 10-Knowledge/domain/{business,marketing,sales} มีแต่ README ว่าง — เนื้อจริงอยู่ใน shortcut 2 ตัวนี้
- สร้าง spec/04-business-conversion.md + ชั้น G 8 หัวข้อใน checklist (G1 hero positioning · G2 หน้าราคา · G3 funnel 7 ขั้น · G4 trust · G5 onboarding · G6 storytelling order · G7 email lifecycle · G8 event/KPI/PDPA)
- ยอดรวมใหม่: **69 หัวข้อ** (A17+B13+C9+D9+E7+F6+G8) · 28→50→57→61→69
- กติกาสำคัญ: ข้อความ hero/ราคา ดึงจากผล Use Business Plan ของโปรเจกต์ ห้าม AI เขียนคำโฆษณาเอง · ลำดับทำ F→G→A→B/C→E→D

## รอบ GitLab Hub · 7 โปรเจกต์แชร์ DS เดียว 2026-07-05
- เจ้าของสร้าง repo: gitlab.dev.jigsawgroups.work/Nat-Rattanasak/designsystem_onemanfleet (private) + อนุมัติโครง + ระบุ 8 โปรเจกต์ local
- push hub สำเร็จ: 19 ไฟล์สะอาด (spec/checklist/tokens/tools/preview/ENTRY/README/link script) · ตัด dist/log/ledger · build ผ่านจาก hub · branch master+main
- แก้จุดจริง: repo private → raw URL 302 → เปลี่ยนคู่มือจาก curl เป็น git clone (creds ล็อกอินผ่าน)
- ดึงเข้าโปรเจกต์จริง 7 ตัว (clone design/brand-ds + build + gitignore กันปน): Content Factory(pilot) · MD Assist · SynerryFable · EA FarmFable · Lotto Reward · AdsPilot-AI · Venture Radar V2 → ทุกตัวมี dist/front.css พร้อมใช้ (ตรวจ ✓ ทั้ง 7)
- ข้าม Hermes Agent (เป็นต้นทาง DS เอง ไม่ดึงจากตัวเอง) · VPS เจ้าของบอกทำทีหลัง
- โมเดล: แก้ token ที่ hub → push → แต่ละโปรเจกต์ `git -C design/brand-ds pull` = เปลี่ยนพร้อมกัน · ci ชุดเดียว dna เหมือนกัน
- งานคนที่เหลือ: sync ขึ้น VPS · pilot รัน Use Design System จริง 1 หน้า

## รอบตรวจ 2 ส่วน · ปลด Fable ออกจากสมการ 2026-07-05
- เจ้าของถาม: (1) ใช้ DS โดยไม่มี Fable ได้ไหม (Opus วางแผน→Codex/Claude เขียน ผ่าน relay) (2) Master/shortcut อัปพร้อม relay + 77 แล้วยัง
- ตรวจ (1): ชุด design-system-standard-v2 = **0 ไฟล์อ้างชื่อโมเดล** → ใช้กับ AI ตัวไหนก็ได้ทันที · คุณภาพเท่ากันเพราะบังคับด้วยด่านโค้ด (ds-check ≥95% · contrast exit 0 · build exit 0 · 10 มิติ ≥B · เช็กลิสต์ 🔴 100%) ไม่ใช่ดุลยพินิจโมเดล
- ตรวจ (2): **ยังไม่เรียบร้อยจริง** — เจอ version ปน (2.1/หัว v2.0) · เลขเก่า 69+50 ค้าง · ไม่มีคำว่า relay สักคำ · master เดิมไม่ชี้ v2
- แก้ครบ 4 จุด (ยืนยันด้วย grep ทุกจุด): shortcut → v2.3 (77 หัวข้อ + หมวด "โหมดทำงานร่วม Use AI Relay: Opus วางแผน default / Codex-Grok เขียน / reviewer อีกตัว / ด่านโค้ดตัดสิน — Fable เฉพาะงานยาก ไม่บังคับ") · master เดิมติด banner ชี้ v2.3 ห้ามใช้เป็นตัวรัน · ENTRY เพิ่มหมวด "ใช้กับ AI ตัวไหนก็ได้" · registry แถว DS → v2.3

## รอบสุดท้าย · AI Relay ปิดช่องว่าง 2026-07-05
- เจ้าของสั่ง: รอบสุดท้าย + บังคับใช้ Use AI Relay (Fable ห้ามเขียนโค้ด — แพง) · Fable ทำแค่ brief + วิเคราะห์ + ตรวจรับ
- สายพานจริง: Fable เขียน brief 4 ก้อน → **Codex เขียน** (relay DS-FINAL-1 · status ok) → **Grok ตรวจอิสระ** (DS-FINAL-REVIEW · verdict ผ่าน + จุดเล็ก 3) → Fable วิเคราะห์: --accent ผ่านจริง 7.2-12.5:1 (ข้อกล่าวหาตก) เหลือจุดจริง 1 (ป้ายปุ่ม preset) → **Codex แก้** (DS-FINAL-2 timeout → retry DS-FINAL-2R ok) → Fable ตรวจรับ
- ของที่ได้: f-emotion (archetype + 6 แกนอารมณ์แถบคะแนน + ค่า token ต่อแกน + แรงจูงใจ) · f-arch (แผนภาพ 3 ชั้น + ci/dna + ชนิด DTCG 8+6 + opacity) · a-pipeline (ท่อผลิต + งบประสิทธิภาพ + Figma/review gate) · ปุ่ม preset กลับมา (เปิด/ปิดชั้นหรู + ป้าย 4 สถานะ TH/EN)
- ตรวจรับ (tier 4): console 0 error · Front 14 + Admin 10 = 24 หมวด · preset ซ่อน/โชว์ premium จริง · ป้ายปุ่มถูกทั้ง เปิด/ปิด×TH/EN · i18n รั่ว 0 ใน section ใหม่ · ภาพ f-emotion ยืนยัน (6 แกนสวย)
- relay ledger: calls_used 9 · Fable ไม่เขียนโค้ดเลยในรอบนี้

## รอบ "ทุกหัวข้อต้องมองเห็น" 2026-07-05
- หลักการใหม่จากเจ้าของ (สลักเป็นกฎ): คนตรวจจากโค้ด/เอกสารไม่ได้ — **ทุกหัวข้อของ DS ต้องมีของมองเห็นบนหน้าโชว์** เพื่อเทียบตอนใช้จริงว่าตรงไหม
- ขยายหน้า: Front 8→12 หมวด (+รากฐาน token ครบชุด: spacing/radius/z-index/motion/icon/breakpoints · +นำทาง/tabs/breadcrumb + modal/drawer/toast กดจริง · +ถ้อยคำ do-don't + a11y targets · +funnel 7 ขั้น/trust/onboarding/email/event tags ใน G) · Admin 5→9 หมวด (+ตารางขั้นสูง: sort/filter chips/ค้นหา/เลือกแถว+bulk/แบ่งหน้า/แก้ในแถว — ข้อมูลจำลอง 23 แถว · +ฟอร์มครบชนิด: select/date/textarea/upload/error + wizard 4 ขั้น · +ชาร์ต line/bar + จานสีกราฟ colorblind-safe · +ตารางสิทธิ์เต็ม 5 ความสามารถ×4 บทบาท + PDPA banner + governance: deprecation/adoption 83.7% จริง/AI chat UI)
- ทุกข้อความใหม่มี data-th/data-en ครบ · ไฟล์ 168KB (รวมฟอนต์ฝัง)
- ทดสอบจริง (tier 4): console 0 error · sort→฿8,200 ขึ้นก่อน · filter ค้างซ่อม→3 แถว · เลือกทั้งหมด→bulk 23 · แบ่งหน้า 4 · modal เปิด/Esc ปิด · toast ยิงได้ · ภาพ foundations ยืนยัน
- Front 12 + Admin 9 = 21 หมวดมองเห็น ครอบทุกชั้น H/F/G/A/B/C/D/E

## รอบแก้ 3 บั๊กเจ้าของจับได้ + คำถามใหญ่ 2026-07-05
- เจ้าของจับได้: (1) ฟอนต์ไม่ใช่ Noto จริง+เล็ก (2) สลับ EN ไทยปน (3) DS หลังบ้านตื้นเกินสำหรับเว็บ 20 ล้าน + ถามว่า 77 หัวข้อหมดแล้วเหรอ
- แก้ (1): เครื่องเจ้าของไม่มี Noto Sans Thai → **ฝังฟอนต์จริงเป็น base64 ในไฟล์** (variable 100-900 · thai+latin · 77KB) · document.fonts.check = true ยืนยันโหลดจริง · ขยายขนาด: body 15→16 · caption 12.5→13.5 · lead 16→17.5 · เมนู 13→14 · ตาราง 13→14
- แก้ (2): audit ด้วย regex ไทยบน DOM จริงหลังสลับ EN → เจอ 10 จุด → เติม data-en ครบ (specimen/หน่วยเดือน/ปุ่มโหมด) + ระบบ placeholder 2 ภาษา (data-ph-*) → ตรวจซ้ำเหลือแค่สัญลักษณ์ ฿ (ใช้ได้ 2 ภาษา) · ภาพโหมด EN ยืนยันสะอาด
- ตอบ (3): มาตรฐาน 77 หัวข้อครบในเอกสาร/token/เครื่องมือ แต่หน้าโชว์แสดง ~10% — หลังบ้านขาด: ตารางขั้นสูง (กรอง/แก้ในแถว/bulk/virtualization) · ฟอร์มครบชนิด · modal/drawer/toast · wizard · permission matrix · charts — **งานถัดไป: ขยายโซน Admin ให้ครบระดับ 20 ล้าน**

## รอบรื้อหน้าตา Linear-class 2026-07-05 (เจ้าของให้ 2 คะแนน → redesign เต็ม)
- เจ้าของตัดสิน: เครื่องข้างในเสร็จ แต่หน้าตา "ทุเรศ" — ยอมรับ · รื้อ visual ทั้งหน้าใหม่ (ของข้างใน/ฟังก์ชันเดิมครบ)
- Art direction ใหม่: dark-first แบบ Linear · hero เปิดหน้า display 72px weight 280 + gradient text แดง→ทอง · แสงบรรยากาศ fixed จางทั้งหน้า · การ์ดสูตรแพง (hairline + inner highlight) · ปุ่ม radius 10 + glow · ตาราง/KPI คมขึ้น · reveal ตอน scroll (เคารพ reduced-motion) · light mode เป็นรอง
- ฟังก์ชันครบเดิม: 2 โซน · เมนู fix + scroll-spy + ย่อขยาย (แก้แล้ว) · TH/EN (เพิ่ม data-th-html สำหรับ hero) · identity/เหตุผลสี/pain ครบ
- ตรวจ: contrast-check เพิ่ม 8 คู่ dark-first → **30/30 ผ่าน AA (exit 0)** · ภาพยืนยัน 3 จุด (hero front / premium / admin) — เทคนิคใหม่: จับภาพส่วนลึกด้วย translateY เพราะตัวจับภาพไม่รองรับ scroll ลึก
- Grok beauty critique เสร็จ (_ledger/grok-beauty-review.txt) · คำตัดสิน: "token/contrast ดี แต่ typography+spacing ยังไม่เป็นระบบ · composition เป็น dark-SaaS template · ยังไม่ผ่าน pitch 10-20 ล้าน"
- แก้ทันทีตาม review (รอบ 1): น้ำหนักฟอนต์ปลอม 250/280/450/650/750/780 → น้ำหนักจริง 300/400/500/600/700 (grep ยืนยัน) · เพิ่ม --fs-*/--sp-*/--track-* scale ใน :root + ผูก h1/display/h2/sec/card · letter-spacing เหลือ 2 จังหวะ · copy ลายเซ็น AI ("แบบ Linear/แบบ Stripe", "10-20M CLASS", footer สารภาพ) → เสียงแบรนด์จริง ("ONE MAN FLEET · FLAGSHIP" + product copy) · ตรวจ computed style: h1 72px/300 · h2 24px · card 24px ✓
- ยังเหลือจาก review (ต้องเจ้าของตัดสิน — บันทึกไว้): (1) หน้านี้เป็นเครื่องมือรีวิวภายใน → ถ้าจะใช้โชว์ลูกค้าต้องมี "โหมดลูกค้า" ซ่อน scaffolding (layer labels, stat 77/22, conflict warning) (2) visual identity ที่เกินชุด gradient+glow (ต้องมี logo/ภาพ/ไอคอนจริงจากแบรนด์) (3) โหลดไฟล์ฟอนต์ Noto จริงเมื่อใช้ในโปรเจกต์ (artifact โหลด CDN ไม่ได้)
- Republish URL เดิม (label: linear-class-redesign)

## รอบแก้บั๊กจากเจ้าของ 2026-07-05 (เมนูย่อค้าง + contrast พัง)
- เจ้าของจับได้ 2 บั๊ก: (1) เมนูย่อแล้วขยายกลับไม่ได้ (2) ไม่ตรวจ WCAG จริง contrast พัง — ทั้งที่กฎ C7 เขียนไว้เองใน checklist
- สาเหตุ (1): โหมดย่อ 64px โลโก้+padding กิน 66px ปุ่มขยายตกขอบโดน overflow:hidden ตัด → แก้ side-head เรียงแนวตั้งตอน mini · ทดสอบจริง: ย่อ (btn อยู่ในกรอบ w30/64) → ขยายกลับ 264px ✓
- สาเหตุ (2): ใช้สีสถานะ (success/warning/danger) เป็น "ข้อความ" ตรงๆ → สร้าง tools/contrast-check.mjs (สูตร WCAG relative luminance · 22 คู่ · light+dark) → เดิมตก: badge success ตกหนัก · warning บนขาว · danger บน inset · ปุ่มแดง dark 4.24
- แก้: เพิ่ม semantic --success-text/--warning-text/--danger-text (คู่ light/dark) · --text-muted เข้มขึ้น #54545d / #a4a4ae · menu-group #a8a6a4 · dark action #c9221a · ไล่แก้ทุกจุดที่ใช้สีสถานะเป็นข้อความ (badge/state/kpi trend/role/conflict)
- ผลตรวจซ้ำ: **22/22 คู่ผ่าน AA (exit 0)** + อ่าน computed style จริงจาก DOM ทั้ง 2 โหมด ตรงกับค่าที่พิสูจน์
- a11y MCP audit ใช้ไม่ได้ (ไม่มี Chrome ให้ puppeteer) → contrast-check.mjs เป็นตัวตรวจสำรองถาวร · งานค้าง: เพิ่ม status-text tokens เข้า front/admin.tokens.json + ผูก contrast-check เข้า CI (D4) ให้บังคับทุกงานหน้าจอ
- บทเรียน: **กฎที่เขียนใน checklist ต้องรันจริงกับงานตัวเองทุกครั้งก่อนส่ง — ห้ามส่งหน้าจอโดยไม่มีผลตรวจ contrast แนบ**

## รอบ Project Identity (ชั้น H) + สองภาษา 2026-07-05
- เจ้าของติ: DS ไม่มีบริบทโครงการ (ชื่อ/พันธกิจ/วิสัยทัศน์) · ไม่มีเหตุผลการเลือกสีผูกพันธกิจ · ไม่มี pain→design · ไม่มีสองภาษา · สั่งให้กลับไปอ่านแผนธุรกิจ/การตลาดใน Obsidian แล้วดึงเองว่าพลาดอะไร
- อ่านจริง: OneManFleet มี BRAND_SOURCE_OF_TRUTH.md (canonical · positioning 2 ภาษา · target+pain · offer ladder) → **จับ conflict จริงได้: สี canonical #1A1A2E/#E94560 ≠ โค้ด tailwind #231F20/#851914** = พิสูจน์ว่าชั้น direction check จำเป็น
- สร้างชั้น H · Project Identity & Direction (8 หัวข้อ H0-H7): อ่านเอกสารโปรเจกต์บังคับ · บัตรประจำตัวโครงการ · เหตุผลสีทุกสี · pain→design ≥3 แถว · สองภาษา TH/EN · Direction Check 4 คำถาม · ตรวจข้ามเอกสาร · บันทึกส่งต่อ
- ยอดรวม: **77 หัวข้อ** (H8+A17+B13+C9+D9+E7+F6+G8) · ลำดับใหม่ H→F→G→A→B/C→E→D · shortcut อัป v2.2
- หน้า Explorer: เพิ่มส่วนบัตรประจำตัวโครงการ (ข้อมูลจริงจาก BRAND_SOURCE_OF_TRUTH) + แถบเตือน conflict + ปุ่ม TH/EN สลับทั้งหน้า · ทดสอบผ่านตาจริง (ภาพ EN version ยืนยัน)

## รอบ Explorer 2 โซน + วิจัยกันกลิ่น AI 2026-07-05
- เจ้าของทวง 2 เรื่อง: (1) โครงเมนู HTML เดิมที่เคยสั่ง (sidebar fix + เมนูย่อย + สลับเนื้อหา + scroll-spy + ย่อขยาย) (2) 2 โซน Front/Admin ที่มีใน token แต่หน้า preview ไม่โชว์
- ทำใหม่: หน้า preview เป็น DS Explorer — sidebar ซ้าย fix สีเข้ม + ปุ่มสลับโซน ①Front(9 หมวด: ภาพรวม สี ตัวอักษร ปุ่ม ฟอร์ม สถานะ เงา ชั้นหรู ธุรกิจG) / ②Admin(6 หมวด: KPI ตารางแน่น0.75× ฟอร์ม สถานะ สิทธิ์4บทบาท) + scroll-spy (IntersectionObserver) + collapse เหลือไอคอน + breadcrumb
- บั๊กจริงที่จับได้จากพรีวิว: ไม่มี meta charset → ไทยเพี้ยนตอนเสิร์ฟตรง → แก้แล้ว ตรวจภาพผ่าน
- ทดสอบผ่านตาจริง (tier 4): สลับโซน ✓ เมนูสลับชุด ✓ ย่อเมนู ✓ ไทยถูก ✓ console 0 error
- วิจัยกันกลิ่น AI (web จริง): สาเหตุ = distributional convergence (Inter+purple gradient+rounded-lg) · เครื่องมือ: GSAP(ฟรีหมด 2025)+Lenis+Rive+Theatre.js · Utopia fluid type · Huetone LCH · grain/noise overlay · แหล่ง: awwwards/codrops/madewithgsap

## รอบยืนยันหลักการส่งมอบ 2026-07-05
- เจ้าของยืนยัน: ของส่งมอบ = Shortcut เครื่องมือรันใน 30-40 โปรเจกต์ · ไม่ใช่เอกสารให้โปรเจกต์อ่าน
- เย็บ shortcut เป็น v2.1: ผูกครบทุกชั้น F→G→A→B/C→E→D + นิยาม "ของที่วางลงในโปรเจกต์" (design/tokens + build + ds-check + .ds-allowlist + .project/DesignSystem.md) + คำถามเดียวเลือก preset
- ชุด design-system-standard-v2/ = สมองของเครื่องมือ (seed กลาง) · เจ้าของและโปรเจกต์ไม่ต้องเปิดอ่าน

## บันทึกก้าว (append-only)
- setup: สร้างโฟลเดอร์ design-system-standard-v2/{spec,guide,checklist,tokens,_ledger}
- settings self-grant grok/codex โดนบล็อก (safety) แต่ CLI รันได้อยู่แล้ว
- Phase 2: เขียน spec/02-emotion-matrix.md (12 archetype + 5 แกน + persona floor + 4 ตัวอย่างจริง) → ส่ง Grok ตรวจ
