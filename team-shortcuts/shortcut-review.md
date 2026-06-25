# รายงานรีวิว Prompt Shortcut (AI Relay)

> สายพาน: Codex (GPT-5.5) รีวิวภายนอก · Claude สังเคราะห์+เสริมมุมคนละค่าย
> เริ่ม: รอบงานนี้ · ขอบเขต: อ่านอย่างเดียว ยังไม่แก้ไฟล์ Shortcut
> สมุดงาน: `.hermes/ai-relay/ledger.md`

แต่ละ Shortcut มี 4 ช่อง: เป้าหมาย · จุดแข็ง · จุดอ่อน · ควรแก้ (รวมความเห็น Codex + Claude แล้ว)

---

## 1. Use Act-As

**เป้าหมาย:** บังคับ AI ให้หยุดคิดก่อนลงมือ — วิเคราะห์งาน จัดทีม role ผู้เชี่ยวชาญ แบ่งความรับผิดชอบ วางเฟส และขออนุมัติก่อนทำงานถาวร ใช้กับงานยากที่ถ้ารีบทำจะหลุดบริบทหรือทำเกิน

**จุดแข็ง:**
- โครงสร้างดี เริ่มจาก "งานคืออะไร/สำเร็จคืออะไร/พลาดตรงไหนเสียหาย" ก่อนเลือก role
- สั่งเลือกเฉพาะ role ที่ทำให้งานดีขึ้นจริง กัน role ฟุ่มเฟือย
- แบ่งคนทำ/คนตรวจ/คนตัดสินสุดท้ายชัด
- มีด่านอนุมัติชัด ห้ามสร้าง/แก้ไฟล์จนเจ้าของงานอนุมัติ

**จุดอ่อน:**
- คำว่า "wow / ดีที่สุด / เหนือความคาดหมาย" นามธรรม ทำให้ AI ตอบฟุ้งหรือใหญ่เกินจริง
- ไม่มีเพดานจำนวน role / คำถาม / เฟส ผลลัพธ์ยาวไม่คงเส้นคงวา
- ไม่บังคับแยก "ข้อมูลที่รู้จริง" กับ "สมมติฐาน" — AI เติมบริบทเอง
- (Claude เสริม) สร้าง role ลอย ๆ ไม่ผูกกับทีม agent จริงในเครื่อง (มี Business Agent 20 ตัวอยู่แล้ว) + งานทับซ้อนกับ Use Opus Plan / Use AI Pair

**ควรแก้:**
- จำกัด "ใช้ 3-6 role เท่านั้น เกินต้องอธิบายเหตุผล"
- เพิ่มหัวข้อ "สมมติฐาน / สิ่งที่ยังไม่รู้ / ต้องขอเพิ่ม" กัน AI เดา
- แปลง wow เป็นเกณฑ์วัดได้ (เร็วขึ้น/ลดเสี่ยง/ใช้จริงทันที/ตรวจสอบได้)
- เพิ่ม "เกณฑ์พร้อมลงมือ" + (Claude) ผูก role เข้ากับ subagent จริงถ้ามี

---

## 2. Use Comply

**เป้าหมาย:** บังคับ AI ทำงานแบบตรวจย้อนหลังได้ — แตกเป้าหมายเป็น phase, issue, เกณฑ์เสร็จ, หลักฐาน, % คืบหน้า เพื่อแก้ปัญหา AI ข้ามขั้น/ลืม/โมเมว่าเสร็จ

**จุดแข็ง:**
- โครงครบสำหรับงานใหญ่ มี scope, out of scope, phase, issue, verification, รายงานท้ายเฟส
- แยกหลักฐานตามชนิดงาน (เว็บเช็ก localhost/VPS · เอกสารเช็กไฟล์/ลิงก์/ความครบ)
- บังคับรายงาน % และ blocker เห็นงานค้างชัด

**จุดอ่อน:**
- % ไม่มีสูตรคำนวณ AI ใส่ตัวเลขเองให้ดูดีได้
- "งานใหญ่" ไม่นิยามว่าใหญ่แค่ไหนจึงต้องทยอยส่ง
- หลักฐานตรวจไม่กำหนดรูปแบบ (คำสั่ง/ไฟล์/URL/test result/SHA)
- ไม่มีสถานะมาตรฐาน (not_started/in_progress/blocked/verified/failed)
- (Claude เสริม) ตาราง % ซ้ำกับ Use Continue และ Use Viber Structure — ควรชี้ความต่างให้ชัด

**ควรแก้:**
- เพิ่มสูตร %: นับสำเร็จเฉพาะ issue ที่ verified · phase % = เฉลี่ยถ่วงน้ำหนัก
- เพิ่มแบบฟอร์มหลักฐาน (คำสั่งที่รัน/ผล/ไฟล์หรือ URL/วันที่/ข้อจำกัด)
- เพิ่มเกณฑ์ "งานใหญ่" (เกิน 3 phase, เกิน 8 issue, แตะหลายไฟล์, มี deploy)
- เพิ่มกฎ "ไม่แน่ใจใส่ unknown/blocked ห้ามเดา"

---

## 3. Use Continue

**เป้าหมาย:** ให้ AI ทำงานต่อเองอัตโนมัติ ลดการรอผู้ใช้ บังคับทำเป็นเฟส มีหลักฐานตรวจจริง ห้ามอ้างเสร็จถ้ายังไม่ผ่านเงื่อนไข

**จุดแข็ง:**
- กรอบชัด: อ่านบริบท → แตกเฟส → ทำทีละเฟส → ตรวจ → รายงาน
- ข้อห้ามสำคัญครบ (secret, deploy production, ใช้เงิน, ส่งอีเมลจริง)
- บังคับบันทึกเหตุผล + trade-off ตรวจย้อนหลังได้
- รูปแบบส่งงานจบดี (% + หลักฐาน + คำสั่งตรวจ + ความเสี่ยง + เรื่องที่เลือกแทนผู้ใช้)

**จุดอ่อน:**
- "100%" เสี่ยงทำให้ AI ฝืนสรุปเกินหลักฐาน
- "ไม่ต้องรอผม" เสี่ยงเกินไปสำหรับงานกระทบสูง (migration, push/merge, แก้ config, ลง dependency, เรียก API จริง)
- เกณฑ์ "wow" กว้าง พา AI ขยาย scope
- ไม่มี stop rule ชัด (งบเวลา, test fail จาก environment, context ไม่พอ, worktree สกปรก)
- (Claude เสริม) ชนกับกฎเจ้าของงาน "Strict Scope 1 รอบ = 1 จุด" โดยตรง — autonomy ของ Continue อาจแหก scope lock

**ควรแก้:**
- เพิ่มระดับอิสระ 3 ชั้น: ทำเองได้ / ต้องรายงานก่อนทำ / ต้องขออนุมัติ (production, เงิน, data, git push-merge)
- นิยาม "ผ่าน 100%" เป็น checklist จริง (test ผ่าน/lint ผ่าน/manual ผ่าน) ไม่งั้นเป็น blocker
- เพิ่มกฎกัน scope creep (ห้ามขยายนอกเป้าหมายรอบนี้)
- เพิ่ม closeout สั้น: ทำอะไร/หลักฐาน/ไม่ได้ทำ/ความเสี่ยง/next step เดียว

---

## 4. Use Summary

**เป้าหมาย:** ให้ AI สรุป+วิเคราะห์+เสนอทางเลือกข้อมูล/ลิงก์ที่ผู้ใช้ส่งมา ก่อนบันทึกลงไฟล์/memory/registry เสมอ เพื่อกันความรู้ขยะ/กฎผิด/เครื่องมือเสี่ยงไหลเข้า Hermes โดยเจ้าของงานยังไม่ตัดสิน

**จุดแข็ง:**
- เส้นอนุมัติชัดมาก: วิเคราะห์ในแชทก่อน เขียนจริงหลังอนุมัติ
- บังคับ truth check (จริง/ไม่จริง/ยังไม่ยืนยัน) กัน AI เชื่อทุกอย่าง
- ครอบหลายชนิดงาน (ความรู้/plugin/upgrade/workflow/security/issue) + มีระบบหลังอนุมัติครบ (rollback, scheduled evaluation, dashboard sync, tags)

**จุดอ่อน:**
- หนักเกินไปสำหรับข้อมูลเล็ก ๆ บังคับ 8 ส่วนทุกครั้ง คำตอบยาว AI เติมเกินจริง
- truth check ไม่แยกว่า "ตรวจจากเนื้อหาที่ให้" หรือ "ตรวจจากเว็บภายนอก" ความมั่นใจแกว่ง
- "review queue" ทั้งห้ามใช้เลื่อนรีวิว และเป็นตัวเลือกมาตรฐาน — AI สับสน
- (Claude เสริม) บังคับรัน script dashboard ที่ผูก path เครื่องเจ้าของ (`/Users/rattanasak/...`) → พนักงานบนเครื่องอื่นรันไม่ได้ (โยงกับงานชุดติดตั้งทีมที่เพิ่งทำ)

**ควรแก้:**
- เพิ่มโหมด: สั้น / ปกติ / ลึก / install-or-upgrade ให้ 8 ส่วนใช้เฉพาะเคสที่ควรลึก
- นิยาม truth check 3 ชั้น (จากข้อมูลที่ให้ / จากแหล่งที่ตรวจเพิ่ม / ยังไม่ตรวจภายนอก)
- แยก 2 เฟสชัด: ก่อนอนุมัติ = review only · หลังอนุมัติ = write/install/sync
- เปลี่ยน "review queue" เป็น "คิวงานหลังรีวิว" + (Claude) เปลี่ยน path script เป็นอิงโฮมไดเรกทอรี

---

## 5. Use Scan Feature

**เป้าหมาย:** เปลี่ยน repo จริงให้เป็น "บัญชีฟีเจอร์ที่พิสูจน์ได้" ภาษาไทย พร้อมหลักฐาน path:line เพื่อส่งต่อให้ AI การตลาดใช้คิดต่อโดยไม่หลงโฆษณาของที่ยังไม่จริง

**จุดแข็ง:**
- กฎ EVIDENCE-OR-SILENCE แข็งมาก ลดการเดา ลดเสี่ยงเคลมเกินจริง
- 2 โหมดชัด (AI อ่าน repo เอง / chat paste) ใช้ข้ามเครื่องมือได้จริง
- บังคับเก็บ system capability (queue/cache/cron/webhook/multi-tenant) ซึ่งมักเป็น hidden value
- มี Reality Matrix + Mockup Risk + Hidden Gems + GATE หยุดให้คนอนุมัติ

**จุดอ่อน:**
- ไม่กำหนด "ขนาดฟีเจอร์" อาจแตกละเอียดเกินทุกปุ่ม หรือหยาบเกินเป็นโมดูลใหญ่
- discovery ใช้ grep/find เป็นหลัก อาจพลาด repo ใหญ่/monorepo ควรบังคับ rg ถ้ามี
- Evidence Coverage % ไม่ได้กำหนดตัวหาร ตีความเอง
- GATE ไม่ระบุรูปแบบคำตอบที่ต้องรอ (เช่น ต้องตอบ APPROVE_PHASE_2) AI อาจไหลต่อเอง
- (Claude เสริม) ไฟล์ยาว 348 บรรทัด กิน context — โหมด chat (paste) อาจเกินขีดจำกัดบางเครื่องมือ

**ควรแก้:**
- เพิ่ม Evidence Ledger = ตารางไฟล์ที่เปิดอ่านจริง + ช่วงบรรทัด
- นิยามระดับฟีเจอร์: capability area → feature → sub-feature → UI action
- เพิ่ม "Marketing-safe Claim Boundary" (เคลมได้/ต้องระวัง/ห้ามเคลม)
- ใส่ machine token ที่ GATE + resume protocol (phase id, artifact, คำสั่งอนุมัติ)

---

## 6. Use Opus Plan

**เป้าหมาย:** ให้ Opus เป็นหัวหน้าวางแผน (ไม่ลงมือโค้ดเอง) แปลงงานคลุมเครือเป็น phase/issue/risk/verification + prompt ส่งต่อ แล้วให้ router เลือก AI ตัวถัดไป สร้างร่องรอยส่งงานข้าม AI ที่ตรวจย้อนหลังได้

**จุดแข็ง:**
- แยกบทบาทชัด: planner / executor / reviewer ลดปัญหา AI ตัวเดียวทำเองตรวจเอง
- มี artifact จริง (.hermes/plans/*.md + .hermes/routes/*.json) ส่งต่องานได้
- บังคับ verification (ไม่เคลม 100%, web ตรวจ localhost, VPS ตรวจ VPS)
- มี safety gate (ห้าม deploy/ใช้เงิน/ส่งอีเมล/เปิด public service ถ้าไม่สั่งชัด)

**จุดอ่อน:**
- พึ่งสคริปต์เฉพาะทาง (multi_ai_workflow.py + router) มากเกิน ไม่มีไฟล์นี้ workflow สะดุดทันที
- พกพาข้ามเครื่องเปราะ: path .hermes/scripts/docs/Mac-VPS mirror + ชื่อ AI ผูกกับ ecosystem เฉพาะ
- router % เสี่ยงเป็น "คะแนนแต่งเอง" ถ้าไม่บังคับใช้ผลจากสคริปต์เท่านั้น
- ไม่มี contract สำหรับ failure state (context missing, route command failed, executor ไม่มีในเครื่อง)
- (Claude เสริม) ตรงกับจุดอ่อนที่เจอในงานชุดติดตั้งทีม — ผูก path เครื่องเจ้าของ ทำให้พนักงานเครื่องอื่นใช้ไม่ได้

**ควรแก้:**
- เพิ่ม preflight ก่อนวางแผน (ตรวจ project root, .hermes, สคริปต์, git status, executor ที่มีจริง)
- เพิ่ม fallback: ถ้าไม่มี router/script ให้เขียน plan แบบพกพาได้ แล้วบอกว่าต้องติดตั้ง workflow ก่อน
- ใส่ schema version + issue_id + status + blocking_reason ให้เครื่องอ่านได้
- แยก config executor ออกเป็น .hermes/multi-ai.yml ไม่ผูกตายกับ Codex/Qwen/Gemini

---

## 7. Use AI Pair

**เป้าหมาย:** ให้ AI หลายตัวทำงานบน issue เดียวโดยไม่ทับกัน ผ่านบทบาท Planner/Coder/Reviewer ล็อกด้วย branch + handoff.md มีหลักฐานตรวจ (diff, review packet) และ MR/CI เป็นด่านสุดท้าย

**จุดแข็ง:**
- บทบาทชัด: คนวางแผน/คนเขียน/คนตรวจ แยกหน้าที่ดี
- กฎ anti self-review: ผู้ตรวจคนละค่ายกับผู้เขียน
- reviewer read-only ลดเสี่ยงแก้ไฟล์ทับ
- ห้ามถามซ้ำเมื่อข้อมูลพอ ให้สร้าง brief/handoff ทันที

**จุดอ่อน:**
- ซ้ำกับ Use AI Relay มาก (โหมด Planner→Coder→Reviewer, anti self-review, branch, handoff, loop แก้-ตรวจ)
- ค่าคงที่ผูกเครื่องเกินไป (GitLab host, linux-nat, ชื่อ AI) พนักงานเครื่องอื่น/โปรเจกต์อื่นใช้ยาก
- บาง stop rule แข็งเกิน (dirty worktree ควรเป็น "หยุดรายงาน" ไม่ใช่หยุดทั้ง workflow)
- ไม่ชัดว่ารูปแบบไฟล์ .hermes/ai-pair มาตรฐานคืออะไร ใครเป็นเจ้าของการสร้าง

**ควรแก้:**
- แยกหน้าที่กับ AI Relay ให้ชัด: AI Pair = กติกาทั่วไปของทีม AI · AI Relay = โหมดประหยัด token
- เปลี่ยนค่าคงที่เป็น config ต่อโปรเจกต์ (.hermes/ai-pair/config.json: gitlab_url, vps_name, planner, coder, reviewer, branch_prefix)
- เพิ่ม decision tree: งานเล็ก 2 ตัว · งานเสี่ยง/หลายไฟล์ 3 ตัว · runtime ไม่พร้อม = handoff-only
- นิยาม artifact มาตรฐาน (planner-brief/coder-brief/review-packet/review-result/handoff) + ฟิลด์บังคับ

---

## 8. Use AI Relay

**เป้าหมาย:** ให้ Claude เป็นหัวหน้าคุมงาน (วางแผน/ตรวจ/ตัดสิน/เปิดด่าน) แต่ไม่เขียนโค้ดเองถ้าเลี่ยงได้ เพื่อลด token แพง ส่งงานเขียนให้ AI ถูกกว่า บังคับมีหลักฐานตรวจผ่าน ledger ทุกครั้ง

**จุดแข็ง:**
- พิธีเปิด 3 คำถามล็อกคนเขียน/คนตรวจ/ขอบเขตก่อนเริ่ม ลดการเดา
- กฎกันตรวจตัวเองชัด คนเขียน-คนตรวจคนละค่าย
- บังคับตรวจ diff + รันเทสต์ + รายงานตัวเลข + ห้ามเคลม 100% ลอย ๆ
- มี stop rule ครบ (dirty worktree, secret, deploy, database, error ซ้ำ 3 ครั้ง)

**จุดอ่อน:**
- พึ่ง CLI ภายนอกหลายตัว (Grok/Gemini/Codex bridge/Qwen/SSH/บัญชี Gemini เฉพาะเครื่อง) ย้ายข้ามเครื่อง/พนักงานพังง่าย
- path + บัญชีเฉพาะบุคคล (ssh myserver, ~/.gemini-accounts/jigsaw) ทำให้ prompt ไม่พกพา
- คำสั่ง --yolo / --always-approve เสี่ยงให้ AI เขียนไฟล์เองกว้าง ถ้า scope checker ไม่แข็งพอ
- "100%" ยังไม่ล็อกเกณฑ์ (test command/acceptance) อาจเป็นตัวเลขจากความรู้สึก
- (Claude เสริม) จุดอ่อนพกพานี้ตรงกับงานชุดติดตั้งทีมที่เพิ่งทำ — เป็นปัญหาเชิงระบบของหลาย Shortcut

**ควรแก้:**
- เพิ่ม preflight สคริปต์เช็ก: เครื่องนี้มี CLI ไหน, login แล้วไหม, version, project path, staff id
- แยกคำสั่งเฉพาะเครื่องเป็น config (.hermes/ai-relay/adapters.yaml) แทน hardcode ใน prompt
- ทำ ledger เป็น schema ชัด (run id, machine, staff, branch, coder, reviewer, files, test, result, status)
- เพิ่มตัวตรวจ scope ก่อน-หลังโค้ด (allowlist ไฟล์, denylist .env/secret, diff checker หยุดถ้าแตะนอกขอบเขต)

---

## 9. Use Business Plan

**เป้าหมาย:** ไม่ได้ทำแผนธุรกิจทันที แต่บังคับ AI ตรวจโจทย์และวางระบบก่อนลงมือ — แยกประเภทงาน เลือกผู้เชี่ยวชาญ แตกเฟส ระบุหลักฐาน รออนุมัติก่อนบันทึกถาวร

**จุดแข็ง:**
- โครงครบมาก ครอบธุรกิจ/การตลาด/pitch/ประมูล/เว็บ/research/growth
- กัน AI มั่วชัด: ทุกข้ออ้างต้องมีหลักฐานหรือวิธีตรวจ + ห้ามใส่ 100 ถ้าไม่มีหลักฐาน
- บังคับถามข้อมูลที่ขาดก่อน เหมาะกับโจทย์จริงมากกว่าตอบสูตรสำเร็จ
- นิยาม persona/journey/pain point/unmet need/WOW ละเอียด จับต้องได้

**จุดอ่อน:**
- prompt ใหญ่มาก หลายโมดูลซ้อน AI อาจตอบยาวเกิน หรือเลือกหัวข้อไม่เหมือนกันแต่ละงาน
- ส่วน Business 360 กับ 4A เนื้อหาทับกัน (persona/journey/pain point/WOW) ผลซ้ำ
- คำสั่ง PDCA ให้บันทึกผล ขัดกับกฎ "ห้ามบันทึกถาวรก่อนอนุมัติ" ถ้าไม่แยกเงื่อนไข
- ไม่มีตัวเลือกความลึก งานเล็กถูกวิเคราะห์หนักเกิน

**ควรแก้:**
- เพิ่ม Mode ต้น prompt: Quick Review / Full Plan / Tender-TOR / Website / Pitch Deck คุมขนาดคำตอบ
- รวม persona ที่ซ้ำให้เหลือแกนเดียว แต่ละโมดูลเรียกใช้ตามจำเป็น
- เพิ่ม "ข้อมูลขั้นต่ำที่ต้องมี" แยกตามประเภทงาน (pitch: ผู้ฟัง/เป้าหมาย/เวลา · tender: TOR/เกณฑ์/วันส่ง)
- แก้ PDCA ให้ "เสนอว่าจะบันทึกอะไร" ก่อน รออนุมัติแล้วค่อยบันทึก

---

## 10. Use SaaS Opus Master Prompt

**เป้าหมาย:** ให้ Opus 4.8 เป็น "ห้องวางแผนธุรกิจ SaaS" ครบวงจร (สินค้า/การตลาด/ราคา/pitch/ตัดสินทำต่อหรือฆ่าทิ้ง) บังคับทุกแผนมีหลักฐาน ขายได้จริง วัดผลได้จริง

**จุดแข็ง:**
- ครอบคลุมมาก: product, pricing, funnel, pitch, risk, competitor, portfolio decision
- ระบบกันมโนชัด: แยกหลักฐาน/สมมติฐาน/ความมั่นใจ/สิ่งที่ต้องตรวจเพิ่ม
- WOW proof แข็ง ห้ามใช้ WOW ลอย ๆ บังคับอธิบายกลไก + วิธีพิสูจน์
- decision gate ดี (ทำต่อ/pivot/merge/pause/kill) กันทำทุกไอเดียเท่ากัน

**จุดอ่อน:**
- prompt ใหญ่มาก ผลลัพธ์ยาวเกินถ้าโจทย์ต้องการแค่ pricing หรือ pitch
- ไม่มีลำดับ "ถามก่อน-ตอบทีหลัง" เข้มพอ AI อาจเริ่มจากข้อมูลไม่พอ
- Opportunity Score ให้ทุกข้อหนักเท่ากัน ทั้งที่ distribution/buyer budget อาจสำคัญกว่า
- ไม่บังคับรูปแบบอ้างหลักฐานชัด (ไฟล์/บรรทัด/ต้นทาง)

**ควรแก้:**
- เพิ่มโหมด: full audit / pricing only / pitch only / portfolio ranking / investor deck
- เพิ่ม "คำถามบังคับก่อนเริ่ม" 5-7 ข้อ (ลูกค้า/งบ/ช่องทางขาย/ไฟล์หลัก/เป้าหมาย)
- ถ่วงน้ำหนัก Opportunity Score (pain/budget/distribution/proof ใน 30 วัน > visual WOW)
- ทุก claim สำคัญต้องมี source + confidence % ไม่มีให้ติดป้าย assumption

---

## 11. Use Viber Structure

**เป้าหมาย:** แปลงงาน Viber/Vibe Code เป็นระบบ "สเปกก่อนโค้ด" ไม่ให้ AI เดา intent แล้วรีบเขียน ผลลัพธ์คือโครงโปรเจกต์ + ตาราง artifact + tracker เฟส/งานย่อย + ด่านตรวจคุณภาพ + กฎหลักฐาน

**จุดแข็ง:**
- กฎเหล็กชัด: ไม่มี requirement/user story/acceptance criteria ห้ามเริ่มโค้ด
- แยกบทบาท AI กับคนดี (intent, security, UAT, go-live, เงิน, ข้อมูลลูกค้า)
- บังคับหลักฐานจริง (build/lint/test/localhost/VPS/endpoint)
- มีโครง phase/issue + compliance table ตามงานต่อได้

**จุดอ่อน:**
- prompt ใหญ่ หนักเกินสำหรับโปรเจกต์เล็กถ้าไม่มีโหมด Light/Standard/Enterprise
- artifact matrix เยอะ แต่ไม่ระบุ "ขั้นต่ำที่ต้องมี" แยกตามชนิดงาน (landing/SaaS/API/mobile)
- Verify Command บังคับดี แต่ไม่มีตัวอย่างคำสั่งตาม stack (Python/Node/Docker/Next.js)
- อ้าง workflow ผ่าน Obsidian link ถ้า AI ตัวอื่นไม่มี vault จะทำตามไม่ครบ

**ควรแก้:**
- เพิ่มระดับ 3 แบบ: งานเล็ก/งานจริง/enterprise ไม่ให้ทุกงานแบกเอกสารเต็มชุด
- เพิ่ม template สั้นของ issue/gate/compliance table ที่คัดลอกใช้ได้ทันที
- เพิ่มตัวอย่างหลักฐาน "นับผ่าน/ไม่นับผ่าน" (screenshot/log/test output/endpoint response)
- เพิ่มกฎ: ข้อมูลไม่พอให้ถามไม่เกิน 3-5 ข้อก่อนสร้างเอกสาร

---

## 12. Use Viber Audit

**เป้าหมาย:** ตรวจโปรเจกต์จริงแบบมีหลักฐานก่อนให้คะแนน เทียบกับ Viber Enterprise Standard บอกว่า artifact/gate/tracker ขาดอะไร ต้องแก้อะไร ต่อยอดเป็น issue ได้

**จุดแข็ง:**
- ครอบคลุมมาก (spec, architecture, data, test, security, performance, release, governance)
- evidence-first ชัด ห้ามบอกผ่าน/100% ถ้าไม่อ่านหลักฐานจริง
- แยกงาน AI กับงานที่คนต้องเซ็น (UAT, security risk, go-live)
- มีระบบคะแนน + tracker ต่อยอดเป็น issue แก้จริงได้

**จุดอ่อน:**
- scope ใหญ่มาก สั่ง "ตรวจทุกโปรเจกต์" ใช้เวลานาน ผลล้น
- เกณฑ์คะแนนมีช่องตีความ (partial 50% อาจไม่สะท้อนความเสี่ยงจริง)
- ไม่กำหนดรูปแบบ issue output ที่ผูกกับ GitHub/GitLab/Asana
- ถ้าไม่มี project registry หรือ path ไม่ครบ งาน discovery เป็นคอขวด

**ควรแก้:**
- เพิ่มโหมดตรวจเร็ว/ตรวจเต็ม แยกงานเล็กกับ enterprise
- เพิ่ม template คะแนนละเอียดต่อหมวด (security/release/human sign-off)
- เริ่มจาก critical gate ก่อน (spec/security/test/release) แล้วค่อย 360 องศา
- เพิ่ม output มาตรฐานสร้าง issue (title/severity/owner/verify command/evidence link)

---

## 13. Use Impeccable

**เป้าหมาย:** เป็นคำสั่งเดียวสำหรับงานคุณภาพ UI ทั้งหมด ไม่ให้เจ้าของงานเลือก shortcut ย่อยเอง AI ตัดสินใจติดตั้ง/สแกน/แก้/รายงานเอง แปลผลเป็นภาษาคน

**จุดแข็ง:**
- โฟกัสดี: แยก blocking issue (พัง) ออกจาก UI debt (หนี้หน้าตา แก้ทีหลังได้)
- ลด AI slop ชัด (gradient สำเร็จรูป, card หนา, glassmorphism, hierarchy แบน)
- บทบาทถูก: AI cleanup · เจ้าของงานเป็น taste gate
- มีหลักฐานหลังแก้ (command, screenshot, browser check)

**จุดอ่อน:**
- ไม่กำหนดเกณฑ์ "ควรติดตั้ง Impeccable เมื่อไร" AI อาจติดตั้งในโปรเจกต์ที่ไม่ควรแตะ dependency
- ไม่มี fallback ถ้า npx impeccable detect ใช้ไม่ได้/network บล็อก/ไม่ใช่ Node-frontend
- "ต่ำกว่า 80% ห้ามบอกพร้อม production" ดี แต่ไม่มี rubric ว่า 80% คิดจากอะไร
- บอกต้องโหลด Synerry/AI Design Team แต่ไม่บอก path จริง/วิธีจัดการถ้าไฟล์หาย

**ควรแก้:**
- เพิ่ม decision tree: มี target ไหม / frontend ไหม / มี Impeccable ไหม / ติดตั้งได้ไหม / ถ้าไม่ได้ทำ manual audit
- เพิ่ม install safety policy (ตรวจ package manager, lockfile, branch/dirty, ขออนุมัติก่อนเพิ่ม dependency)
- เพิ่ม scoring rubric 100 คะแนน (layout 25/responsive 20/a11y 20/design-system 20/credibility 15)
- เพิ่ม output template ตายตัว (สถานะ/ปัญหาจริง/ผลกระทบ/แก้ทันที/หนี้ UI/หลักฐาน/ความเสี่ยง)

---

## 14. Use Blog Auto

**เป้าหมาย:** เปลี่ยนความรู้จากงานจริงเป็นบล็อก Hi Logic Labs แบบร่างก่อนเสมอ เน้นคัดกรองความลับ ทำดัชนี Obsidian ส่งต่อ Content Factory โดยไม่เผยแพร่ก่อนเจ้าของอนุมัติ

**จุดแข็ง:**
- ประตูความเป็นส่วนตัวชัด (public-safe/internal-only/redact-needed/do-not-use)
- กำหนดเสียงแบรนด์ดี (public ใช้ภาษาอังกฤษ lab voice ไม่ใช้ "ผม/I/me")
- บทบาทครบ (Blog Orchestrator, Privacy Editor, Knowledge Librarian, QA Auditor)
- บังคับ draft-first ทุกช่องทาง (social/video/slide/podcast/YouTube)

**จุดอ่อน:**
- บอกต้องอ่าน Blog Skill แต่ไม่ระบุ path ชัด AI หาไฟล์ผิดได้
- Skill compliance score ไม่บอกเกณฑ์คิดคะแนน % อาจมั่ว/เทียบกันไม่ได้
- Obsidian index action ไม่บอกตำแหน่งไฟล์/รูปแบบข้อมูลที่ต้องเขียน
- localhost/VPS check อาจไม่เกี่ยวกับงานบล็อกบางกรณี ควรมีเงื่อนไข

**ควรแก้:**
- ใส่ path ตายตัวของ Blog Skill, Content Factory prompt, Obsidian review queue
- เพิ่ม template output (privacy table, knowledge card, comply table, handoff checklist)
- แยก "รีวิวอย่างเดียว" กับ "สร้าง draft จริง" กัน AI เขียนไฟล์เกินคำสั่ง
- เกณฑ์คะแนน compliance เป็นข้อ ๆ (privacy 25/brand voice 20/draft-first 25/index-handoff 30)

---

## 15. Use WOW Resource

**เป้าหมาย:** ให้ AI เป็นคนคัดทรัพยากร WOW/Web Design Intelligence แล้วแปลเป็นทิศทางออกแบบเฉพาะโปรเจกต์ ไม่ใช่ก๊อป layout/script มาแปะ ปฏิเสธตัวที่ไม่เข้าพร้อมเหตุผล

**จุดแข็ง:**
- มี routing ชัดว่าโจทย์แบบไหนอ่านไฟล์ไหน ลดการโหลด vault ทั้งก้อน
- บังคับปฏิเสธงาน generic/flashy เกิน/ไม่ accessible/ไม่คุ้ม implementation cost
- มี Selection Brief ก่อนลงมือ เจ้าของงานตัดสินใจได้ก่อน AI แก้ไฟล์
- มี usage log เก็บร่องรอย resource ไหนใช้ได้/ไม่ได้

**จุดอ่อน:**
- พึ่ง absolute path ของ vault เฉพาะเครื่อง (/Users/rattanasak/...) ย้ายเครื่อง/VPS/AI ตัวอื่น สะดุดทันที
- ไม่มี fallback ชัดเมื่อไฟล์ WOW/WDI หาย/อ่านไม่ได้/อยู่คนละ mirror
- usage log เสี่ยงให้ AI เขียนความจำถาวรเร็วเกินถ้าไม่แยก "เสนอก่อน" กับ "บันทึกหลังอนุมัติ"
- ไม่บังคับรูปแบบคะแนน/สถานะ resource ให้แข็งพอสำหรับหลาย AI ใช้ร่วม

**ควรแก้:**
- เพิ่ม path resolver (local mac / VPS mirror / env var HERMES_OBSIDIAN_ROOT) แทน path ตายตัว
- เพิ่ม failure mode: อ่านคลังไม่ได้ให้ตอบจากโจทย์ + ระบุแหล่งที่อ่านไม่สำเร็จ ห้ามเดาว่าอ่านแล้ว
- เพิ่ม scoring rubric (fit/accessibility/enterprise trust/implementation cost/reusability)
- แยก Selection Brief กับ Usage Log ให้ log เป็น draft ในแชทก่อน

---

## 16. Use Flow Guardian

**เป้าหมาย:** บังคับ AI ตรวจสภาพงานก่อนลงมือ (worktree/branch/dirty status) แก้ปัญหา AI หลายตัวทำงานทับกัน ให้ถามเจ้าของก่อนเลือก branch/worktree สำหรับฟีเจอร์ใหม่

**จุดแข็ง:**
- ลำดับความปลอดภัยชัด: ตรวจสถานะ → audit ไม่เขียนไฟล์ → ขออนุมัติ → ทำเฉพาะ scope → ตรวจจริง → handoff
- มีข้อความมาตรฐานให้ถามเจ้าของงาน พฤติกรรมไม่ลอย
- ผูกกับ context หลัก (policy, startup gate, approval gate, worktree safety)

**จุดอ่อน:**
- ไม่ระบุคำสั่งตรวจจริง (git status/branch/worktree list) AI แต่ละตัวตรวจไม่เท่ากัน
- "risky/multi-file/approval gates apply" ตีความกว้างเกิน
- บอกให้ update tracking + handoff แต่ไม่กำหนดไฟล์ปลายทาง/รูปแบบ
- project adapter หาย ไม่มี STOP rule ชัด

**ควรแก้:**
- เพิ่ม checklist บังคับพร้อมคำสั่งจริง (path/branch/dirty/remote/worktree list/tracking file)
- เพิ่ม decision table: งานใหม่/bugfix/audit-only/deploy/หลาย AI พร้อมกัน ต้องถามหรือหยุดตรงไหน
- กำหนด tracking/handoff template (งานที่ทำ/ไฟล์ที่แตะ/verification/risk/next owner action)
- เพิ่ม fallback: อ่าน context/adapter ไม่ได้ ให้ "หยุดก่อน" พร้อมเหตุผล ไม่เดา

---

## 17. Use New Chat

**เป้าหมาย:** บังคับ AI เริ่มแชทใหม่ด้วยการเช็กของจริงก่อน (path/worktree/branch/dirty/local-VPS เท่ากัน/service) ไม่ตอบลอย ๆ ว่าพร้อม กัน AI ทำงานผิดที่ ผิด branch ผิดเครื่อง

**จุดแข็ง:**
- ครอบจุดเสี่ยงหลักครบ (project path, worktree, branch, dirty, remote/VPS/service)
- ข้อห้ามชัด (ห้ามบอก clean ถ้าไม่รัน git status, ห้ามบอก VPS ใช้ได้ถ้าไม่ตรวจ endpoint)
- เหมาะกับ Hermes Agent (Mac เป็น local controller, VPS เป็น target จริง)
- รูปแบบรายงานบังคับ ช่วยให้ทุก AI ตอบมาตรฐานเดียว

**จุดอ่อน:**
- ต้องรู้ staff id ก่อน แต่ไม่กำหนดวิธีเดาจากบริบท/ถามสั้นเมื่อไม่รู้
- ผูกกับ Hermes Agent/VPS มาก ใช้กับโปรเจกต์ทั่วไปมีช่องว่างเรื่อง service/endpoint
- "local/remote/VPS เท่ากันไหม" ไม่ระบุ remote หลัก สับสนถ้ามีหลาย remote (origin/fork/upstream)
- แสดง branch nat เป็น "Update Feature" อ่านง่ายขึ้นแต่เสี่ยงสับสนชื่อแสดงผลกับชื่อ Git จริง

**ควรแก้:**
- เพิ่ม Input contract: ถ้าไม่รู้ staff id/project/target ต้องถามอะไร 1-3 ข้อก่อน
- เพิ่มตาราง decision (READY / BLOCKED / NEED_OWNER_INPUT)
- เพิ่ม fallback สำหรับโปรเจกต์ทั่วไป (ไม่มี VPS ตรวจเฉพาะ local + remote + localhost)
- เพิ่มตัวอย่างรายงานที่ดี 1 ชุด + ตัวอย่าง blocked 1 ชุด

---

## 18. Use Save Git

**เป้าหมาย:** ทำให้ก่อน merge/deploy มีด่านตรวจจริง 5 ชั้น ไม่ใช่ความมั่นใจจาก AI หรือ test local อย่างเดียว เหลือคำตัดสินเดียว (เช่น SAFE_TO_MERGE) + Grid บอกชั้นที่บล็อก

**จุดแข็ง:**
- บังคับใช้คำสั่งเดียว (save-git --stage merge-gate/ship-gate) ลดการตีความมั่ว
- .savegit.json เป็น adapter ให้ด่านรู้จัก stack/project จริง
- Grid 5 ชั้นเห็นทันทีว่าบล็อกที่ local/MR/CI/VPS dry-run/production
- เช็ค commitSha ใน health กันเคส health 200 แต่ production รันโค้ดเก่า

**จุดอ่อน:**
- ถ้า .savegit.json ไม่ครบหรือ skip มากเกิน gate อาจดูผ่านทั้งที่ยังเสี่ยง
- ความน่าเชื่อถือขึ้นกับตัว save-git เอง ต้องมี exit code/log/artifact ตรวจย้อนหลังได้
- พึ่งสิทธิ์ GitLab/CI/VPS ถ้าสิทธิ์ไม่ครบ กลายเป็น OWNER_DECISION_REQUIRED บ่อย
- commitSha ดี แต่ถ้า service คืนค่าผิด/ไม่ผูกกับ build จริง ก็ยังหลอกได้

**ควรแก้:**
- เพิ่ม schema validation ของ .savegit.json + field บังคับตาม stage (merge-gate ต้องมี remote/target/CI)
- แยกผล skip เป็น "ยอมรับได้" กับ "เสี่ยงเพราะไม่มีข้อมูล" ไม่ให้ข้ามด่านสำคัญแบบเงียบ
- ให้ save-git ออกทั้ง human Grid + machine JSON ใช้ต่อใน CI/GitLab merge rule ได้
- เพิ่มหลักฐานรันจริง (timestamp/commit SHA/command summary/artifact path)

---

## 19. Use Merge to Production

**เป้าหมาย:** ทำให้ "ขึ้น production" มีด่านตรวจจริง ไม่ใช่กด merge แล้ว deploy ตามความรู้สึก เฉพาะ merger ที่อนุญาต (nat/namton/nam) รัน merge-gate + ship-gate deploy จาก origin/target คืน token + Grid 6 ชั้น

**จุดแข็ง:**
- decision token เดียว ตัดสินใจง่าย (deploy ได้/ห้าม/verify ไม่ได้)
- Grid 6 ชั้นบอกติดตรงไหน ตั้งแต่สิทธิ์คนสั่งถึง commit ที่รันบน VPS
- กฎ "deploy จาก origin/<target> เท่านั้น" กันเอา local/feature branch ขึ้น production
- ผูกกับ GitLab protected branch (ล็อกสิทธิ์ฝั่ง server) ไม่ใช่ shortcut เป็นตัวบังคับเดียว

**จุดอ่อน:**
- ตรวจ caller ไม่ชัดว่าผูกกับตัวตนอะไร (GitLab username/SSH user/chat ID หรือแค่ชื่อที่พิมพ์) เสี่ยงปลอมตัว
- ชื่อ shortcut "merge to production" อาจทำให้ AI เข้าใจผิดว่า merge แทนได้
- เว็บ static ที่ไม่มี commit endpoint verify ยาก ค้างที่ PRODUCTION_NOT_VERIFIED บ่อย
- ไม่มี rollback plan ชัดถ้า deploy แล้ว health fail หรือ commit ไม่ตรง

**ควรแก้:**
- เพิ่มกฎยืนยันตัวตน merger แบบผูกระบบจริง (GitLab actor + SSH user + allowlist ใน .savegit.json)
- ระบุชัด: AI ห้ามกด merge เอง เว้นแต่เจ้าของงานสั่ง + สิทธิ์ระบบอนุญาต
- deploy ผ่าน CI/CD จาก commit SHA เดียวกัน + ส่ง image digest/version ไป verify บน VPS
- เพิ่ม Grid ช่อง evidence (MR ID/pipeline ID/origin SHA/built SHA/image digest/health URL/เวลาตรวจ)

---

## 20. Use Move Folder

**เป้าหมาย:** ไม่ใช่ prompt คิดวิธีย้ายโฟลเดอร์ใหม่ แต่เป็นทางลัดให้ AI ไปต่อ workflow เดิมบน VPS บังคับอ่าน registry/checkpoint/นโยบายห้ามแตะ/หลักฐานขอบเขตก่อนลงมือ

**จุดแข็ง:**
- กัน AI เดาเองดี ระบุลำดับไฟล์ที่ต้องอ่านก่อนชัด
- safety gate แข็ง: ห้าม scan/move/delete/backup/restart ถ้าไม่มี approval exact scope
- แยกงาน safe-only / owner-decision / dangerous-hold เจ้าของตัดสินง่าย
- บังคับรายงานไทย + footer สถานะ ส่งต่อข้ามแชทได้

**จุดอ่อน:**
- พึ่ง path VPS เฉพาะมาก (/home/linux-nat/.codex/use-move-folder/...) เครื่อง/user เปลี่ยน หรือ path หาย = หยุดทันที
- ไม่มี fallback อ่านสำเนา read-only/mirror ถ้า VPS ล่ม AI หยุดแม้มีข้อมูลเก่าช่วยวางแผนได้
- "safe-only" ตีความหลวมได้ถ้าไม่มี dry-run command + diff ก่อน/หลัง + approval ราย path
- protected roots อยู่ใน prompt แต่บอกว่าไฟล์บน VPS authoritative ถ้าไม่ตรงกัน AI สับสน

**ควรแก้:**
- เพิ่ม "VPS unavailable mode": สรุปได้เฉพาะระดับวางแผน ห้ามปฏิบัติการ ติดป้าย stale
- เพิ่ม schema รายงาน (path/owner evidence/size before-after/action/command preview/rollback)
- เพิ่ม hard stop สำหรับคำสั่งเสี่ยง (rm/mv/rsync --delete/chmod -R/chown -R) ต้อง approval เฉพาะคำสั่ง
- แยก protected roots เป็นไฟล์เดียวที่ sync ได้ + บอกว่า local list เป็น snapshot ไม่ใช่แหล่งจริง

---

## 21. Review Chat

**เป้าหมาย:** ไม่ใช่แค่สรุปแชท แต่ปิดงานเป็นระบบ: ตรวจงานค้าง ตรวจหลักฐานจริง อัปเดต handoff เตรียมข้อความเปิดแชทใหม่ กันบริบทหลุดเมื่อย้ายแชท

**จุดแข็ง:**
- ลำดับชัด: ตรวจคำสั่งเดิม → ตรวจสถานะจริง → อัปเดตไฟล์ → สรุป → สร้าง prompt ต่อแชท
- บังคับแยก "เสร็จจริง" กับ "พูดว่าเสร็จ" ต้องมีหลักฐาน
- มี output ใช้ต่อได้ทันที (ข้อความเปิดแชทใหม่)

**จุดอ่อน:**
- "อัปเดตไฟล์ที่เกี่ยวข้องถ้าควร" เปิดช่องให้ AI เขียนไฟล์เองมากไป ควรล็อกให้ขออนุมัติถ้าไม่ชัด
- ไม่มีเกณฑ์บอกว่า handoff ไหน "relevant" AI อาจอัปเดตผิดไฟล์/ข้ามไฟล์สำคัญ
- Context window % มองไม่เห็นจริงในหลาย UI ต้องรายงานเป็นข้อจำกัด ไม่ควรเดาแรง

**ควรแก้:**
- เพิ่มกฎ: ไม่มีคำสั่งเขียนไฟล์ชัด ให้สรุปแนะนำก่อน ห้ามอัปเดต handoff อัตโนมัติ
- เพิ่มรายการไฟล์ handoff มาตรฐานตามชนิดงาน (HermesAgent/coding project/shortcut work)
- เพิ่มช่อง "หลักฐานที่ยังขาด" ในผลลัพธ์

---

# สรุปภาพรวม — รูปแบบจุดอ่อนที่ซ้ำกันทั้ง 21 ตัว

Codex + Claude เห็นตรงกันว่ามี 5 ปัญหาเชิงระบบที่โผล่ซ้ำในหลาย Shortcut ถ้าแก้ที่ส่วนกลางทีเดียวจะดีขึ้นพร้อมกันหลายตัว:

| # | รูปแบบจุดอ่อนที่ซ้ำ | โผล่ใน Shortcut | ทางแก้ส่วนกลาง |
|---|---|---|---|
| A | **ผูก path เครื่องเจ้าของ** (/Users/rattanasak/...) | Summary, Opus Plan, WOW Resource, Move Folder, AI Relay, AI Pair | ใช้ตัวแปร HERMES_OBSIDIAN_ROOT อิงโฮมไดเรกทอรี (โยงงานชุดติดตั้งทีมที่เพิ่งทำ) |
| B | **ไม่มีโหมดความลึก** (เล็ก/ปกติ/เต็ม) prompt ใหญ่ตอบยาวเกิน | Summary, Business Plan, SaaS Opus, Viber Structure, Viber Audit | เพิ่มหัว Mode ทุก Shortcut ใหญ่ |
| C | **% / คะแนนไม่มีสูตร** ตีความเอง | Comply, Continue, Impeccable, Blog Auto, Viber Audit | เพิ่ม scoring rubric + สูตร % ที่นับเฉพาะของที่ verified |
| D | **หลักฐาน/สถานะไม่มีรูปแบบตายตัว** | Comply, Scan Feature, Save Git, Flow Guardian | กำหนด evidence schema + สถานะมาตรฐานกลาง |
| E | **ไม่มี fallback เมื่ออ่านไฟล์/ระบบไม่ได้** AI เสี่ยงเดาว่าอ่านแล้ว | WOW Resource, Move Folder, Flow Guardian, Opus Plan | เพิ่มกฎ "อ่านไม่ได้ = หยุด/ติดป้าย ห้ามเดา" ทุกตัว

