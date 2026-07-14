# Plan — QAQC · ระบบตรวจคุณภาพงาน "Use QA QC" (อนุมัติ 2026-07-10 · เจ้าของตอบครบ 3 คำถามในแชท Fable)

> memory-schema: v1.2 · **plan_id: QAQC** · จาก Use Act-As v3.1 (Fable 5 ออกแบบ · เจ้าของสั่งตรงให้ Fable เป็นสมองแผนนี้ 2026-07-10)
> แผนก่อนหน้า **GRD จบครบ 4 เฟส merged แล้ว** — เนื้อเต็ม + คิว GRD-P5..P9 ย้ายไป `.project/plan-grd.md` (ห้ามลบ · จะเริ่มคิวไหนให้ยกกลับมาไฟล์นี้ก่อน เพราะ plan-anchor อ่านไฟล์นี้ไฟล์เดียว)
> ทีม: **Fable 5 = สมองออกแบบ/สังเคราะห์/เขียนเอกสาร** · **Grok + Codex = กรรมการตรวจข้ามค่าย** (ผ่าน relay-call) · **เจ้าของ = อนุมัติผลแต่ละด่าน + push คลัง + กด merge**
> branch: `feature/use-qa-qc` (แตกจาก origin/main `f14fefca1`) · งานหลายเฟส = **1 PR เดียว** · ไฟล์ shortcut อยู่คลัง Obsidian (คนละ repo — เจ้าของ push เอง)

## กติกาเหล็กของแผนนี้ (สืบทอดจาก GRD ครบทุกข้อ)

1. **เลขงานขึ้นต้น `QAQC-`** และต้องมีจริงในไฟล์นี้ · เลขที่ไม่มี = ห้ามทำ (งานจรใช้ `--no-plan` ให้เห็นใน ledger)
2. **กลับมาอ่านแผนก่อนลงมือทุกครั้ง** หลังตอบคำถามแทรก/สลับงาน
3. **verified = แถว gate-run เท่านั้น** · งานเอกสาร/prompt (ไม่มี test อัตโนมัติ) = `manual_verified` + เจ้าของยืนยัน ตามกฎ no_gate ของ relay
4. ใบสั่งงานฝังข้อความเฟส + ข้อห้ามจากไฟล์นี้ลงไปในตัว
5. ห้ามแตะ: `scripts/jarvis-voice/` · `design-system-standard-v2/` · `.claude/launch.json` · `.hermes/**` · `.env*`/secret · **Master ViberQC = อ่านอย่างเดียวเด็ดขาด ห้ามแก้ไฟล์ใดในโปรเจกต์นั้น**

## โจทย์จากเจ้าของ (สัญญางาน — เช็กครบก่อนปิดแผน)

- shortcut `Use QA QC`: เมนู **2 แกน** — แกนช่วงงาน (25% / 50% / 75% / 100%) × แกนประเภทตรวจ · **เลือกหลายหมวดพร้อมกันได้** · ทุกเมนูมีคำแนะนำภาษา user · **Scan All อยู่ท้ายสุด + ด่านยืนยัน + ประมาณการค่าใช้จ่าย** (เจ้าของไม่อยากให้ใช้ แต่ต้องมี)
- AI 7 ตัว: เปิดใช้ก่อน **Opus 4.8 / Codex / Grok** · ช่องเสียบอนาคต **Gemini / DeepSeek / Qwen / GLM** — เพิ่ม AI = แก้ config ไม่แก้ prompt
- หัวข้อตรวจครบแบบ **Full System**: รวมหัวข้อเฉพาะของ AI ที่ยังไม่เปิดใช้ด้วย · **ไม่ซ้ำ-ไม่หาย** พิสูจน์ด้วยตัวเลข N/M ต่อแหล่ง (ไฟล์วิจัย Grok / ViberQC / มาตรฐานโลก)
- ตัวไหลงาน: Opus สแกนต้น → AI ตัวที่ 2 สแกนคู่ตรวจซึ่งกันและกัน → **ตาราง severity** (Critical/High/… + comment ผลกระทบการแก้ · เสี่ยงสูง = แยกไปเฟสท้ายสุด) → เขียน `.project/qaqc-scan.md` ก่อนแก้ → แตก Phase/Issue ส่ง **Use AI Relay** → AI ตัวที่ 3 แก้โค้ด → AI ตัวที่ยังไม่ถูกใช้รีวิว (มี 3 ตัว = กติกาสำรอง: คนรีวิวห้ามซ้ำคนแก้)
- เอกสารทุกชิ้นเขียนแบบ **AI อ่านต่อ-ดัดแปลงได้** (แยก "หลักการ" ออกจาก "ค่าเฉพาะโปรเจกต์") · ผล P1 ต้องเอาไปใช้**ปรับปรุง Master ViberQC** ต่อได้
- งบ: เจ้าของปล่อยเพื่อคุณภาพ แต่ต้องรายงาน**ความคุ้มค่าจริง**ใน P5

## QAQC-P1 — คลังหัวข้อครบ ไม่ซ้ำ ไม่หาย (ตารางแม่) · สถานะ: เสร็จ 2026-07-10 (รอเจ้าของอ่านอนุมัติ = manual_verified) — ผลอยู่ `QAQC-Master-Taxonomy.md` + `ViberQC-Diagnosis-2026-07-10.md` ในคลัง · บัญชีกันหาย S1 369/369 · S1b 34/34 · S2 747/747 · S3 13/13

- **QAQC-P1-I1** สกัดหัวข้อตรวจทั้งหมดจากไฟล์วิจัย Grok — `/Users/rattanasak/ObsidianVault/HermesAgent/20-Departments/Security/AI-Security-Testing/Sources/2026-07-10-ai-security-master-source.md` (1,809 บรรทัด) · ทุกหัวข้อ + AI ที่เก่ง + เครื่องมือที่ต้องมี + อ้างบรรทัด · ห้ามตัดทอน
- **QAQC-P1-I2** สกัด + วินิจฉัย **Master ViberQC** (อ่านอย่างเดียว): หมวด/checklist ที่ออกแบบไว้ · จุดที่กระจัดกระจาย/ซ้ำซ้อน (เจ้าของวินิจฉัยว่ายังไม่รัดกุม — หาหลักฐาน) · ของดีที่ควรเก็บ
- **QAQC-P1-I3** ค้นมาตรฐานโลก (เน็ต/GitHub/GitLab): ISTQB test types · OWASP ASVS/Top 10 · ISO/IEC 25010 · production-readiness / launch checklist ชั้นนำ · อ้าง URL ทุกแหล่ง
- **QAQC-P1-I4** Fable สังเคราะห์ **ตารางแม่**: หัวข้อ × หมวด × ช่วงงานที่เหมาะ (25/50/75/100%) × AI หลัก/สำรอง × แหล่งอ้าง + **ตัวเลขกันหาย N/M ต่อแหล่ง** → ไฟล์ในคลัง Obsidian (โฟลเดอร์ AI-Security-Testing)
- verify: ไฟล์ตารางแม่ + N/M ครบ 3 แหล่ง + เจ้าของอ่านอนุมัติ (manual_verified)

## QAQC-P2 — ตาราง AI ประจำหมวด + สายสำรอง · สถานะ: I1+I2 เสร็จ (ฝังใน taxonomy §2/§4) · **I3 ค้าง — กรรมการนอกค่ายล่มพร้อมกัน 3 ตัว 2026-07-10 (Grok=login พัง exit40 · Codex+Gemini=quota exit30) · ตามกติกา cross-check-จบรอบเดียว จะรีวิวรวบ P1+P2+P3+P4 ครั้งเดียวเมื่อโควต้าคืน**

- **QAQC-P2-I1** ต่อหมวด: AI หลักสแกน / สำรอง / คนแก้ / คนรีวิว + กติกาเมื่อโดนลิมิต/บัญชีหมดอายุ (สลับตามสาย) + ช่องเสียบ AI อนาคต — โครงแบบ accounts.yaml ของ relay
- **QAQC-P2-I2** กติกาสำรองช่วง 3 ตัว: **คนรีวิวห้ามซ้ำคนแก้** (กฎเหล็กขั้นต่ำ) · gate-run เป็นผู้ตัดสินจริงเสมอ ไม่ใช่ปากรีวิวเวอร์
- **QAQC-P2-I3** กรรมการข้ามค่าย: Grok + Codex ตรวจร่าง P1+P2 ผ่าน `relay-call --task-id QAQC-P2-I3` (เพดานรอบเต็มจากรอบล้มเหลว — งานรีวิวย้ายไปรันใต้ QAQC-P3-I3)
- **QAQC-P2-I4** [เพิ่ม 2026-07-10 — งานซ่อมที่ขวางกรรมการ] แก้ quota ปลอมใน `relay-call.py classify()`: QUOTA_RE จับคำ quota/rate limit ใน "เนื้อคำตอบยาว" ของงานที่พูดถึงเรื่องโควต้า → ใส่ตัวกันความยาว ≤250 แบบเดียวกับ auth (บทเรียน 2026-07-05) ทั้งฝั่ง exit 0 และ exit ≠ 0 + เทสต์กันถอยหลัง 2 ตัว
  - ผู้ทำ: Fable (ข้อยกเว้นไก่-ไข่: relay พังจนใช้ relay สั่งแก้ตัวเองไม่ได้ — คำตอบของ coder จะโดนบั๊กเดียวกันจับทิ้ง) · ส่ง Codex ตรวจ diff ตามด่านปกติก่อนปิด
  - verify: `./venv/bin/python -m pytest scripts/ai-relay/tests/ -q` → 68 passed / 1 failed (ตัวที่ failed = `test_run_once_returns_timeout_mark_on_timeout` **แดงบนโค้ดเดิมก่อนแก้ พิสูจน์ด้วย git stash แล้ว** — งานซ่อมแยกรอบ)
  - พบพ่วง: `~/.local/bin/relay-call` เป็น symlink ไปสำเนาพนักงาน `~/.hermes/ai-relay-tools` รุ่นก่อน GRD (ไม่มี plan-anchor/--no-plan) → หลัง merge ต้องรัน relay-setup.sh อัปสำเนา · grok CLI v1.0.1 ตัด flag เก่า → แก้ adapters.yaml แล้ว · grok headless ต้องมี API key (งานคนของเจ้าของ)
- verify: ผลตรวจ 2 ค่ายแนบในแชท + แก้ครบข้อ blocking (manual_verified)

## QAQC-P3 — ตัว prompt `Use QA QC` · สถานะ: **เปิดใช้แล้ว v1.1 active (คำสั่งเจ้าของ 2026-07-10 "จบให้ก่อน ใช้ Fable ไม่รอ Codex/Grok")** · I3 กรรมการต่างค่าย = เลื่อนเป็น hardening หลังใช้งาน (ไม่ขวาง) · Fable ตรวจปิดเอง + อุดข้อ 5f โหมด AI ตัวเดียว

- **QAQC-P3-I1** เมนู 2 แกน + เลือกหลายหมวด + คำแนะนำภาษา user ("งานอยู่ช่วงไหน ควรตรวจอะไร") + Scan All ด่านยืนยัน
- **QAQC-P3-I2** ตัวไหลงานเต็ม (สแกนคู่ → ตาราง severity → qaqc-scan.md → Relay → แก้ → รีวิว) + **ตัวประหยัด 3 ชั้น**: (ก) เครื่องมือฟรีรันก่อน AI อ่านผลสรุป (ข) ตัวสแกนที่ 2 ตรวจเฉพาะสิ่งที่ตัวแรกพบ+จุดไม่แน่ใจ (ค) เพดานงบ ledger ของ relay
- **QAQC-P3-I3** กรรมการข้ามค่ายตรวจ prompt ทั้งฉบับ ผ่าน `relay-call --task-id QAQC-P3-I3`
- verify: ไฟล์ prompt เต็มในคลัง + ผ่านกรรมการ + เจ้าของอนุมัติ (manual_verified)

## QAQC-P4 — ต่อระบบความจำ + ทะเบียน · สถานะ: เสร็จ 3/3 — แม่แบบ qaqc-scan.md ฝังใน shortcut · New Chat → v2.0 + Close Chat → v2.3 (เพิ่มขั้น QA/QC) · ทะเบียน 29→30 (สถานะ draft)

- **QAQC-P4-I1** แม่แบบ `.project/qaqc-scan.md` — โครง AI-readable: ป้าย schema/สถานะรายหมวด/ตาราง severity/ประวัติรอบสแกน/งานแก้ค้าง
- **QAQC-P4-I2** เชื่อม **Use New Chat** (อ่าน qaqc-scan.md ตอนเปิด — เฉพาะโปรเจกต์ที่มีไฟล์) + **Use Close Chat** (อัปเดตตอนปิด) · capability-based · bump เวอร์ชัน 2 ไฟล์นั้น
- **QAQC-P4-I3** ลงทะเบียน shortcut ในคลัง (29→30) + Graph Links
- verify: `git check-ignore` ว่าง + `git ls-files` เห็นไฟล์ .project ใหม่ (ตอน commit PR) + ทะเบียนมีแถวใหม่ + grep เจอบรรทัดเชื่อมใน use-new-chat.md / use-close-chat.md

## QAQC-P5 — นำร่อง Road Safe Fund + Root Admin · สถานะ: รอ P1-P4 จบ + เจ้าของสั่งเริ่ม

- รันจริง 1-2 หมวดต่อโปรเจกต์ (Road Safe Fund + Root Admin ก่อน · ต่อไป DRA, Content Thailand + SaaS/Web App 10+) → เก็บ token/เวลา/ผลจริง → ตารางความคุ้มค่า + ปรับเพดาน/ขอบเขต
- verify: แถว ledger จริง + ตาราง severity จริง + ตัวเลขงบจริง (tier 3)

## งานคน (เจ้าของ)

- อนุมัติตารางแม่ (จบ P1) · อนุมัติ prompt (จบ P3) · push คลัง Obsidian ขึ้น GitLab · กด merge PR repo นี้ · สั่งเริ่ม P5

## ความเสี่ยงค้าง

- plan.md active เปลี่ยนเป็น QAQC — เลขงาน GRD-P5..P9 ต้องยกกลับมาไฟล์นี้ก่อนเริ่ม (จดวิธีไว้หัว plan-grd.md แล้ว)
- เอกสาร ViberQC อาจใหญ่กว่าที่เห็น — สกัดไม่ครบรอบเดียวต้องรายงานส่วนที่ยังไม่ได้อ่าน ห้ามเงียบ
- Grok/Codex เคยชนโควต้าพร้อมกัน (2026-07-08) — ถ้ากรรมการเรียกไม่ได้ ให้รายงานและรอ ไม่ข้ามด่านตรวจ

---

# Plan — MW · Shortcut "Use Migrate Web" (เริ่ม 2026-07-14 · เจ้าของสั่งเดินในแชท Fable + Codex ตรวจคู่)

> memory-schema: v1.2 · **plan_id: MW** · จาก Use Act-As v3.1 (รีวิวเบื้องต้นส่งแล้ว 2026-07-14 · เจ้าของตอบ 3 คำถาม + ส่งไฟล์ที่ 1 + สั่งเดิน)
> ทีม: **Fable 5 = สมองคิด/วิเคราะห์/ออกแบบ (เจ้าของสั่งตรง)** · **Codex = ผู้ตรวจคู่ทุกใบ (relay-call --role review)** · Grok = สำรองถ้า Codex เสีย (รายงานการสลับเสมอ) · เจ้าของ = ตัดสินทุก checkpoint
> branch: `control_webengine_flow` (เจ้าของเปิดให้ 2026-07-14) · งานหลายเฟส = **1 PR เดียว**
> ขอบเขต: Shortcut ใช้กับ**ทุกโปรเจกต์ที่สร้างจาก Root Admin** (เริ่มจริงที่ RoadSafeFund) · ทีม ~10 คน pull จาก dev กลาง VPS ทำบนเครื่องตัวเอง + บางคน worktree บน VPS

## กติกาเหล็กของแผนนี้ (สืบทอด GRD/QAQC ครบ + เพิ่มของเจ้าของ)

1. เลขงานขึ้นต้น `MW-` และต้องมีจริงในไฟล์นี้ · เลขที่ไม่มี = ห้ามทำ
2. **ห้ามข้าม Flow · ห้ามตัดทอนข้อมูลเจ้าของ** — ทุกไฟล์ที่วิเคราะห์ต้องมีบัญชีนับ N/M · จะตัด/จัดลำดับใหม่ต้องเสนอเจ้าของตัดสินรายข้อ
3. ทุกใบวิเคราะห์/ออกแบบต้องผ่าน Codex ตรวจก่อนสรุปให้เจ้าของ (AI Relay โหมด 2)
4. verified = แถว gate-run เท่านั้น · งานเอกสาร = manual_verified + เจ้าของยืนยัน
5. จบแต่ละเฟส: ตาราง compile % ต่อหัวข้อ + ตารางว้าว (ดีขึ้นจากเดิมกี่ % วัดด้วยอะไร)
6. ห้ามแตะ: `.hermes/**` · `.env*`/secret · repo NewWebEngine2026 = **อ่านอย่างเดียว** (งานเขียนอยู่ Hermes Agent เท่านั้น)

## MW-P1 — วิเคราะห์ไฟล์เจ้าของทีละไฟล์ · สถานะ: I1-I3 เสร็จ (I2,I3 เจ้าของอนุมัติแล้ว · I1 มี 11 จุดเคาะค้าง) · รอไฟล์ชุดถัดไปหรือคำสั่งเข้า P2

### ผลสะสม (2026-07-14 · ทุกใบผ่านผู้ตรวจต่างค่าย Codex ผ่าน cross-check MCP เพราะ relay review mode มีบั๊ก)
- **I1 FLOW-13Steps-v2.docx.md** (439 บรรทัด อ่านครบ): ของใหม่เจ้าของแทรก **23 ข้อ** + 2 ข้อความต้องเก็บคำเดิม (ชุดคำถาม M0 บรรทัด 25 + hardcode บรรทัด 249) · **11 จุดเคาะรอเจ้าของตอบรายข้อ** (เด่น: DEC-167 เลขชนกันจริงกับ decisions.md · .work vs .project/ · เลขลิสต์แม่ห้ามพิมพ์มือ—เครื่องนับจริง=81 · Sitemap 2 docx → เสนอ "บัตรประจำเมนู 1 md/เมนู") · repo ต้นทางถูกแชทอื่นแก้สด (FLOW-JOURNEY-13STEP.md หายจาก .project/ แล้ว · checklist 74→81)
- **I2 Workshop Comment (PDF 15 หน้า + md)**: ปัญหา 28 ข้อจากทีม 5 คน → Flow ครอบ ✅7 🟡21 ❌0 · **เจ้าของอนุมัติ 4 ข้อเสนอ**: DS pattern กลาง 13 ตัว · ทุก pattern มี 6 ช่องบังคับ (หน้าตา/พฤติกรรม/ข้อมูล/สถานะผิดปกติ/การเข้าถึง/หลักฐานทดสอบ) · checklist ใหม่ ~10 ข้อ · แตก 28 กลุ่ม→~40 ข้อย่อยตอนทำบัญชี N/M
- **I3 TOR 3 โปรเจกต์** (CTH 58 REQ สร้างใหม่+catalog 2 ภาษา / DRA 92 REQ เสร็จ 88% + 7 minisite + OIT O1-O43 + TWCAG 2022 / RSF migrate + GOV-GAP 9 จุด): **เจ้าของอนุมัติ 12 ข้อเพิ่ม Flow + 8 ตัวกันเสี่ยง** — เด่น: โหมดหลัก+แทร็กเสริม (MIGRATE/REMEDIATE/BUILD + DATA/FORM/MINISITE/BILINGUAL) · RTM 3 ชั้น (มีข้อกำหนด→มีเทสต์→รันผ่านจริง) · Light Loop หน้าโครงซ้ำแบบกันข้ามขั้น · บัญชีคิวเมนู+ล็อก site/module ทีม 10 คน · เลเยอร์ปิดงวดราชการ · ผูก Use QA QC
- **บั๊ก infra พบระหว่างงาน (คิวซ่อมแยก · อย่าลืม)**: relay-call role review ยัด flag เก่า (--permission-mode/--no-subagents) ใส่ grok 1.0.1 → crash 1 วิ ทุกครั้ง · อยู่ทั้งตัวติดตั้ง ~/.hermes/ai-relay-tools และ repo main (`prepare_adapter_for_role`) · Codex ผ่าน portal ก็พังบนเครื่องนี้ · ทางเลี่ยงที่ใช้: cross-check MCP ask_gpt5

- **MW-P1-I4 เสร็จ**: ค้นคลัง Obsidian (60-Design 234 ไฟล์/42 module · QAQC Q01-Q16 178 หัวข้อ · playbooks) + 4 ข้อเสนอ ก-ง ผ่าน Codex · ข้อค้นพบ: VPS มี vault sync แล้ว (auto-push-vault.sh ทุก 3 นาที → GitLab + linux-nat:~/knowledge/hermes-agent-vault/) · ไฟล์ tracked อ่อนไหว 2 ตัวตรวจแล้ว 0 token จริง · ก่อนเปิดคลังให้ทีม clone ต้องตรวจประวัติ GitLab
- **MW-P1-I5 เสร็จ (เจ้าของแก้ข้อมูล)**: source ภาพจริง 4 ตัว = **Freepik + Recraft + Topaz + Magnific (magnific.com/user/api-keys = ต่อ API)** — ไม่มี "Opus" · เพิ่ม Image Source Registry (config-driven) + Benchmark Intake Gate (ห้าม copy 100% · วิเคราะห์แตกส่วน · ถามโจทย์ · แปลงเข้า DS)
- **MW-P1-I9 เสร็จ + เจ้าของเคาะ 2026-07-14: "ตามเสนอทั้งหมด ทำเลย"** — จุดเคาะ 13 จุดอนุมัติครบตามข้อเสนอ (1 ออกเลข DEC ใหม่กฎภาพ · 2 hardcode=ข้อยกเว้นอนุมัติรายกรณี · 3 เลขเครื่องนับสด · 4 Module แจ้งไฟล์เสียที่ Root Admin+ทดสอบรายเมนู · 5 ES ด่านโปรเจกต์+รายเมนู search · 6 .work=งานลูกค้า+deliverables · .project=ความจำ AI · ทั้งคู่เข้า git · 7 Pagination เข้า checklist · 8 ภาพคั่น/วิดีโอใต้เพดาน+reduced-motion · 9 แท็บลัดผูก WCAG+SEO · 10 GV: RSF ไม่ถามซ้ำ · 11 ตรวจเครื่องมือภาพจริงก่อนใช้ · 12 Light Loop เครื่อง 100% + คนสุ่มหน้าแรกทุกแบบ+20%/กลุ่ม · 13 รับข้อเสนอ I4 ทั้ง 4) · บัญชีรวมเข้า spec = 55 กลุ่มรายการ (I1:23+2 · I2:4 · I3:12+8 · I4:4 · I5:2)
- **MW-P1-I9** สรุปรวมทุกชุด: บัญชีความต้องการรวม + ตารางจุดเคาะรวม (รวม 11 จุดค้างของ I1)

- **MW-P1-I1** ไฟล์ 1 `FLOW-13Steps-v2.docx.md` (439 บรรทัด): บัญชี delta ที่เจ้าของแทรกเพิ่มจากไฟล์ repo + จุดขัด/ทับซ้อน + ตอบคำถามรวม/แยกไฟล์ Sitemap → Codex ตรวจ → เสนอเจ้าของตัดสิน
- **MW-P1-I2..I5** เสร็จครบ (I2 Workshop · I3 TOR · I4 คลัง Obsidian · I5 เจ้าของสั่งเพิ่ม — ดูผลสะสมด้านบน)
- **MW-P1-I9** เสร็จ — เจ้าของเคาะ 13 จุด อนุมัติทั้งหมด 2026-07-14 · **จุดเคาะ 11 ข้อของ I1 ถูกรวมและปิดในตาราง 13 จุดแล้ว (ไม่มีค้าง)**
- verify: บัญชี N/M ทุกไฟล์ + Codex verdict ทุกใบ + เจ้าของอนุมัติบัญชีรวม

## MW-P2 — ออกแบบ spec Shortcut · สถานะ: กำลังทำ · draft = `.project/mw-spec-draft.md`

- **MW-P2-I1** ร่าง spec v1.0 + Codex ตรวจรอบ 1 → FIX 8 จุด (ตารางแม่/I1 หาย 5/ขัดแย้ง 6/เครื่องมือ) — เสร็จ 2026-07-14
- **MW-P2-I2 เสร็จ 2026-07-14**: v1.1 → Codex รอบ 2 = FIX 3 เรื่อง (ครบ 2 รอบผู้ตรวจเดิม) → **เปลี่ยนวิธีตรวจเป็นเครื่องตามกติกา v2.16**: แก้ v1.2 (baseline sha256 จริง 3 ไฟล์ใน `.project/mw-flow-baseline.md` · [G] ทุกแถวผูก §10-1..10-10 · โปรโตคอลล็อกกลาง 6 ข้อ · สัญญารับมอบ P3 3 ข้อ §13) + สร้าง `scripts/mw-spec-check.py` (5 ด่าน) → **รันจริง PASS: ตารางแม่ 55/55 · [G] ครบ · § ครบ · จุดเคาะ 13/13 · baseline ตรง** (เครื่องจับ 21 จุดหลวมก่อนผ่าน — พิสูจน์ว่าด่านทำงาน)
- **MW-P2-I3 เสร็จ: เจ้าของอนุมัติ spec 2026-07-14 ("อนุมัติ spec") → P2 ปิด 3/3**

- โครง 2 ชั้น (แกน Flow กลาง ↔ Project Profile ต่อเว็บ) · ด่านกันข้าม/กันโกหก (ยก G5+menu-gate) · Write Permit ต่อเมนูกันทีมชนกัน · บทบาท AI capability-based · ตาราง compile+ว้าวต่อเฟส · การเรียกใช้ VPS+notebook
- verify: Codex ตรวจ spec + เจ้าของอนุมัติก่อนเขียนไฟล์จริง

## MW-P3 — เขียนไฟล์จริง · สถานะ: **เสร็จ 2026-07-14 · merged main (PR #35+#36)** — I1 prompt ✅ · I2 เครื่องมือ **7/7** ✅ (Grok เขียน · GPT-5 ตรวจ · mw suite 252) · I3 test-id map + §13.1 COMPLETE 32/32 ✅ · I4 สัญญา §13 ครบ 3/3 ✅

- **MW-P3-I1 เสร็จ 2026-07-14**: prompt `use-migrate-web.md` v1.0 + `use-migrate-web-flow13.md` (เนื้อต้นฉบับ 439 บรรทัดตรง 100% · embedded_sha256 คุมช่วง 31-469 verify ตรง · ตาราง delta 25 + กฎลำดับความสำคัญ + ตารางจับคู่พาธเก่า→ใหม่) + แถวทะเบียน payload registry · **Codex ตรวจ 2 รอบ**: รอบ 1 FIX 6 กลุ่ม→แก้ครบ · รอบ 2 ผ่าน 4/6 เหลือ 2 บรรทัด (รอยจาก shell กลืน backtick)→แก้+verify ด้วย grep/shasum เขียวครบ · บทเรียนจดแล้ว: heredoc ต้อง quote + replace ต้องมี assert
- **MW-P3-I2 — แผนส่งมอบ (Fable เขียน 2026-07-14 · เจ้าของสลับกลับ Opus ทำต่อตามกติกา relay v2.16)**
  - **บทบาทรอบใหม่:** Opus = สมองคุมสายพาน (วางแผน/เลือก coder/เขียน brief/ตรวจ/ตัดสิน) · coder = Codex หรือ Grok (เลือกตาม catalog §7 ต่อชนิดงาน) · reviewer = คนละค่ายกับ coder เสมอ · verified = gate-run เท่านั้น
  - **งาน = เครื่องมือ 7 ตัว · 1 ตัว = 1 issue** (สเปคเต็มอยู่ SPEC §10 + ตาราง [เครื่องมือประกอบ] ใน use-migrate-web.md):
    - MW-P3-I2d `work-locks` ล็อกกลางจองเมนูข้ามเครื่อง (ทำก่อน — ปลดล็อกทีม) · เกณฑ์รับมอบ: จำลอง 2 clone จองพร้อมกัน สำเร็จ 1/2 (ผลรันแนบ)
    - MW-P3-I2a `mw-menu-gate` เครื่องตัดสินปิดเมนู portable (checklist กลาง + .work config · เลขนับสด · exit 0/1) — หลักการจากต้นแบบ menu-gate.mjs ของ NewWebEngine แต่**ห้ามแตะ repo นั้น** (อ่านอย่างเดียว)
    - MW-P3-I2b `mw-page-check` ตรวจหน้า (soft-404 ใน rendered main content · ลิงก์ · ภาษา · เพดานไฟล์ hero≤300KB/content≤150KB/vdo≤2MB · pagination · sticky ไม่บัง · related logic · วิดีโอ attr autoplay/muted/reduced-motion) config ต่อโปรเจกต์
    - MW-P3-I2f `mw-doctor` ตรวจติดตั้ง notebook/VPS + smoke call เครื่องมือภาพ 4 source (Freepik/Recraft/Topaz/Magnific — 1 งานเล็กจริง) + ยิง relay จริง 1 ครั้ง
    - MW-P3-I2c `mw-rtm-report` นับ RTM 3 ชั้น (req-register → test cases → ผลรันจริง)
    - MW-P3-I2g `mw-wow-report` รวมตัวเลขเครื่อง before/after (LH/axe/gate N-M/RTM%/ขนาดไฟล์) — AI ห้ามกรอกเอง
    - MW-P3-I2e `mw-backend-check` (ใหญ่สุด — วงจรฟอร์ม/แจ้งไฟล์เสีย/siteId/dashboard ต้องมี engine จริง) → เสนอทำคู่ P4 บน RoadSafeFund
  - **ที่วางโค้ด:** `scripts/mw/` ใน repo นี้ + เทสต์ `tests/scripts/mw/` · ทุกตัวมี pytest + gate-run pass + reviewer ต่างค่าย
  - **ข้อจำกัดเครื่องนี้ที่ Opus ต้องรู้ (พิสูจน์แล้ววันนี้):** (1) relay-call role **review** พัง 2 ชั้น — portal crash + ยัด flag เก่าใส่ grok 1.0.1 → **ใช้ role code ผ่าน `AI_RELAY_ALLOW_LOCAL_CLI=1` ได้ · งานรีวิวใช้ cross-check MCP `ask_gpt5`** (2) plan-anchor อ่าน plan_id แรกของไฟล์ (QAQC) → เลขงาน MW ใช้ `--no-plan` ไปก่อน หรือขอเจ้าของสลับไฟล์แผน (3) Codex CLI ตรงใช้ได้ (ทดสอบ PONG แล้ว) (4) `hermes-write-permit` มีบนเครื่อง (5) ผู้ตรวจเดิม+วิธีเดิมได้ 2 รอบ — รอบ 3 ต้องเปลี่ยนเป็นเครื่องตรวจ (แบบที่ทำกับ mw-spec-check สำเร็จแล้ว)
  - **ก่อนปิด P3 ทั้งก้อน:** สัญญา §13 ครบ 3 ข้อ — ตาราง test ID ทุกแถว [G] (P3-I3) + จำลอง 2 clone + mw-spec-check PASS
- **MW-P3-I3** ผูก installer + ตาราง test ID ทุก [G] (สัญญา §13 ข้อ 1)
- **MW-P3-I4** รันสัญญา §13 ครบ 3 ข้อ → ส่งเจ้าของ
- verify: mw-spec-check PASS + Codex ตรวจ + ด่าน git + สัญญา §13

## MW-P4 — ทดสอบจริงบน RoadSafeFund + ติดตั้ง 2 ทาง (VPS/notebook) · สถานะ: **install เสร็จ (mw-setup.sh + install-shortcuts wiring · 2 เทสต์ · branch feature/mw-p4-install)** · **ค้าง: รันจริงกับ RoadSafeFund backend (ต้องมี API+DB+token เจ้าของ) + ยืนยันติดตั้งบน VPS**

- verify: เดิน flow จริงอย่างน้อย 1 เมนู หลักฐาน tier 3+ + เจ้าของเห็นของจริง

## MW-P5 — ปิดงาน: 1 PR + อัปความจำ · สถานะ: **PR merged แล้ว (2 PR) · อัป OverviewProgress+plan รอบนี้ · เหลือ Use Close Chat formalize + decisions**

## งานคน (เจ้าของ)

- ส่งไฟล์ 2-5 · ตัดสินจุดขัดรายข้อ · อนุมัติ spec (จบ P2) · กด merge PR (จบ P5)
