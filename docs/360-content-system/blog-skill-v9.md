---
title: Blog Skill v9 — Hi Logic Labs Edition
status: phase-AB-draft
created: 2026-06-01
owner: พี่นัท (รัตนศักดิ์)
replaces: Idea2Logic/docs/blog-skills/ (v8 · 8 โมดูล 145 KB)
related:
  - 360-overview.md
  - privacy-gate-rules.md
  - knowledge-card-template.md
principle: เบา · ชัด · เขียนตามได้จริง · ไม่มีตัวบุคคล · อังกฤษ global-first
---

# Blog Skill v9 — Hi Logic Labs Edition

> เครื่องมือเขียนบล็อกของแบรนด์ Hi Logic Labs
> เล็กกว่า v8 ประมาณ 5-6 เท่า กฎไม่ขัดกันเอง ออกแบบใหม่เพื่อ 4 ปัญหาที่เจอจริง:
> ซับซ้อนเกิน · เสียงโยงตัวตน · HTML พังบ่อย · อังกฤษไม่เป็นธรรมชาติ

## 0. ใช้เมื่อไหร่

เมื่อจะเปลี่ยน "การ์ดความรู้" (จากตัวเก็บเกี่ยว เฟส 2) เป็นบทความบล็อกภาษาอังกฤษ
สำหรับเผยแพร่ในนาม Hi Logic Labs

อ่าน 3 ไฟล์นี้ก่อนเขียนทุกครั้ง: ไฟล์นี้ · `privacy-gate-rules.md` · การ์ดความรู้ต้นทาง

## 1. เสียงแบรนด์ (แก้ปัญหา "เสียงโยงตัวตน")

Hi Logic Labs คือ **ห้องทดลอง ไม่ใช่คน** เขียนในนามห้องทดลอง ไม่ใช่ในนามบุคคล

| กฎเสียง | ❌ ห้าม | ✅ ใช้ |
|---|---|---|
| ผู้พูด | "I fired 3 freelancers" (ตัวบุคคล) | "We ran this against a 3-person workflow" (ห้องทดลอง) |
| สรรพนาม | I / my / me (โยงคนเดียว) | we / the lab / here's |
| ตัวตน | ชื่อคน อายุ บริษัท | ไม่เอ่ยถึงเลย |
| ท่าที | กูรูสอน / อวดผลงานตัวเอง | คนทดลองที่เปิดผลให้ดู |

โทนรวม: **calm, sharp, evidence-first** — เหมือนห้องทดลองที่บอกว่า "เราลองมาแล้ว นี่คือผล"
ไม่ตื่นเต้นเกิน ไม่ขายฝัน ให้ตัวเลขและผลพูดแทน

## 2. กฎห้ามเด็ดขาด (privacy + persona)

1. ห้ามเอ่ยชื่อจริง ชื่อเล่น อายุ ของเจ้าของ
2. ห้ามเอ่ยชื่อบริษัท ชื่อลูกค้า มูลค่าโครงการ (ตาม `privacy-gate-rules.md`)
3. ห้ามใช้ "I" แบบบุคคล — ใช้ "we/the lab"
4. ห้ามอ้างว่าเป็น developer หรือเขียนโค้ดเอง — กรอบคือ "วางระบบให้ AI ทำ"
5. ทุกบทความต้องผ่านด่าน privacy gate ก่อนเผยแพร่

## 3. โครงบทความ (6 ส่วน · แก้ปัญหา "ซับซ้อนเกิน")

| ส่วน | เนื้อหา | ความยาว |
|---|---|---|
| 1. Hook | เปิดด้วยตัวเลขช็อก หรือปัญหาจริง 1-2 ประโยค | สั้น |
| 2. The problem | โจทย์ที่คนกลุ่มเป้าหมายเจอ | 2-3 ย่อหน้า |
| 3. What we tested | เราลองอะไร วางระบบยังไง | 3-5 ย่อหน้า |
| 4. The result | ผลจริง ตัวเลข ก่อน-หลัง (บอกว่าวัดจากอะไร) | 2-3 ย่อหน้า |
| 5. The takeaway | บทเรียนที่เอาไปใช้ต่อได้ | 1-2 ย่อหน้า |
| 6. Try this | ผู้อ่านลองทำอะไรได้ทันที | 1-3 bullet |

ปิดท้ายด้วยข้อคิด ไม่ใช่ "In conclusion" (ดูข้อ 4)

## 4. ภาษาอังกฤษ global-first (แก้ปัญหา "อังกฤษไม่เป็นธรรมชาติ")

เขียนอังกฤษตั้งแต่ต้น **ห้ามแปลจากไทย** ถ้าคิดเป็นไทยแล้วแปล จะอ่านออกว่าแปลมา

หลักสั้นๆ:
- ประโยคสั้น 1-3 ประโยคต่อย่อหน้า
- ใช้ contractions: it's, don't, we've, here's
- active voice เสมอ: "We built" ไม่ใช่ "was built"
- ตัวเลขเจาะจง: "37% faster" ไม่ใช่ "much faster"

**คำต้องห้าม (anti-AI words · ยกของดีเดิมมา):**
delve · leverage · utilize · comprehensive · robust · seamless · pivotal ·
groundbreaking · transformative · innovative · harness · furthermore · moreover ·
"it's worth noting" · "in today's fast-paced world" · "in conclusion" ·
game-changer · cutting-edge · "unlock the potential" · paradigm · compelling ·
unprecedented · tapestry · landscape (เชิงเปรียบเปรย)

ห้ามเปิดด้วย "In today's rapidly evolving..." · ห้ามปิดด้วย "I hope this helps"

## 5. ระบบภาพ (แก้ปัญหา "HTML พังบ่อย")

หลักใหม่: **ข้อความเป็นหลัก ภาพเป็นของเสริม**

- เขียนเป็น markdown ปกติ ไม่ฝัง HTML/CSS ซับซ้อน (ตัวที่พังบ่อยใน v8)
- ใส่ภาพเฉพาะที่จำเป็นจริง: หน้าจอผลงาน (เบลอข้อมูลลับ) หรือภาพเทียบก่อน-หลัง
- ตารางใช้ markdown table ธรรมดา
- ถ้าแพลตฟอร์มปลายทาง (เช่น Medium) รองรับแค่ภาพ + ข้อความ ก็พอแล้ว
- ภาพหน้าจอใช้ Playwright แคปตามเฟส 4 แล้วเบลอข้อมูลระบุตัวตนก่อนใส่

## 6. ต่อจากการ์ดความรู้ (input → blog)

วัตถุดิบมาจากการ์ดความรู้ (`knowledge-card-template.md`):
- ช่อง topic → หัวเรื่องบทความ
- ช่อง ปัญหา/วิธีแก้/ผลลัพธ์ → ส่วน 2-4 ของโครง
- ช่อง บทเรียน → ส่วน 5
- ช่อง วัตถุดิบคอนเทนต์ "มุมบล็อก" → ใช้ตั้ง hook
- ช่อง blog_decision → ต่อบล็อกเก่า หรือเขียนใหม่

## 7. ด่านตรวจก่อนเผยแพร่ (QC · สั้น ไม่ขัดกันเอง)

ก่อน publish ตรวจ 6 ข้อนี้ให้ผ่านทุกข้อ:

1. ไม่มีชื่อคน/บริษัท/ลูกค้า/มูลค่า (privacy gate)
2. ไม่มี "I" แบบบุคคล (ใช้ we/the lab)
3. ไม่มีคำต้องห้าม anti-AI (ข้อ 4)
4. ทุกตัวเลขบอกว่าวัดจากอะไร
5. ครบ 6 ส่วนของโครง
6. อ่านออกเสียงแล้วลื่น เหมือนเขียนอังกฤษตั้งแต่ต้น

## 8. ตัวอย่างเสียงที่ถูกต้อง (faceless lab)

❌ v8 เดิม (โยงตัวตน):
"I fired three freelancers. My workflows broke every Tuesday."

✅ v9 (ห้องทดลอง):
"Most domain checkers tell you a name is free when it's already taken. We tested
340 names and watched a fast tool report the wrong answer dozens of times. Here's
the two-layer check that actually works."

> สั้น · ไม่มีตัวคน · เปิดด้วยปัญหาจริง · ให้ตัวเลข · ชวนอ่านต่อ
