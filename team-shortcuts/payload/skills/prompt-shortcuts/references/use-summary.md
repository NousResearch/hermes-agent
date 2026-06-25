---
title: Use Summary
aliases:
  - Use Summary
  - use-summary
  - Summary
  - summary
  - ใช้ Summary
  - สรุป
  - สรุปลิงก์
  - วิเคราะห์บทความ
  - สรุปข้อมูล
tags:
  - prompt-shortcuts
  - summary
  - knowledge-intake
  - review-before-write
  - pdca
status: active
version: 1.7
updated: 2026-06-24
---

# Use Summary

## Shortcut

```text
Use Summary
```

## Prompt

```text
Use Summary กับ link + content ที่ผู้ใช้ส่งมา

หน้าที่: สรุป + วิเคราะห์ + เสนอทางเลือก ก่อนบันทึกลงไฟล์/memory/registry รอเจ้าของอนุมัติ

[เลือกโหมดก่อนเสมอ — โหมดเป็นตัวกำหนดว่าตอบกี่ส่วน]
- สั้น (ค่าเริ่มต้นของข้อมูลเล็ก): ตอบเฉพาะส่วน 1, 2, 8 เท่านั้น
- ปกติ: core 4 ส่วน (1, 2, 3, 8)
- ลึก (เรื่องสำคัญ/กระทบหลายระบบ): ครบ 8 ส่วน + ตรวจแหล่งภายนอกเพิ่ม
- ติดตั้ง/อัปเกรด (plugin/tool/software): core + ส่วน 5 (verification + scheduled evaluation + rollback)

[2 เฟสแยกชัด]
เฟส 1 ก่อนอนุมัติ = review only ห้ามเขียนไฟล์/memory/registry/ติดตั้งใด ๆ
เฟส 2 หลังอนุมัติ = เขียน/ติดตั้ง/sync/ประเมินผล
ข้ามไปเฟส 2 ได้เฉพาะเมื่อผู้ใช้สั่งชัดทั้ง "ให้เลือก" และ "ให้บันทึก/ติดตั้ง"

กฎหลัก:
- ใช้ภาษาผู้ใช้ + แปลศัพท์เทคนิคทันที
- เปิดลิงก์ไม่ได้หรือเนื้อหาไม่พอ → ค่อยขอเนื้อหาเพิ่ม · มี content แล้ว → ใช้ content เป็นหลัก
- เจ้าของอยู่ในแชท = review เลย ห้ามเลื่อนเป็น "คิวงานหลังอนุมัติ"
  (คิวงานหลังอนุมัติใช้ได้เฉพาะงานหลังรีวิวแล้ว เช่น sandbox test, promote)

[Truth check 3 ชั้น + ความมั่นใจ]
จากข้อมูลที่ให้ / จากแหล่งที่ตรวจเพิ่ม (ระบุแหล่ง) / ยังไม่ได้ตรวจภายนอก (ห้ามสรุปว่าจริงแน่)
แต่ละข้อใส่ระดับความมั่นใจ: สูง / กลาง / ต่ำ

รูปแบบตอบ (ใช้เฉพาะส่วนที่โหมดกำหนด):
1. สรุปสั้น
2. ประเด็นสำคัญ (แยก fact / insight / claim / risk)
3. วิเคราะห์เชิงตัดสินใจ: truth check 3 ชั้น / จุดแข็ง / จุดอ่อน / fit กับ project /
   ควรพัฒนาต่อไหม / ควรเข้า Hermes ระดับไหน (เลือกจาก: ไม่เก็บ / review / source /
   knowledge / pattern / playbook / skill / project adapter)
4. จัดประเภท action: knowledge / plugin install / software upgrade / workflow-rule /
   security audit / project issue
5. ถ้าติดตั้ง-อัปเกรด: verification + success/failure criteria +
   scheduled evaluation (สร้างเป็น checklist + note พร้อมวันรอบประเมินที่เจ้าของกำหนด) + rollback
6. ใช้กับงานเรายังไง
7. ตารางทางเลือก: | ตัวเลือก | ทำอะไร | เหมาะเมื่อ | action หลังอนุมัติ | แนะนำ % |
8. แนะนำทางเดียวที่ดีที่สุด + เหตุผล + action ชัด

[หลังอนุมัติ — บันทึกเป็นระเบียน PDCA ที่ติดตามได้จริง]
เป้าหมาย: ทุกสิ่งที่บันทึกต้องมี "วงจรปิด" ไม่ใช่เขียนแล้วค้าง ต้องมีสถานะ + วันครบกำหนด + ลิงก์กลับศูนย์กลาง
1. สร้างโน้ต $HERMES_OBSIDIAN_ROOT/95-Inbox-Lab/review/<ชื่อ>-evaluation-<วันที่>.md
   - HERMES_OBSIDIAN_ROOT: Mac = ~/ObsidianVault/HermesAgent · VPS = /home/linux-nat/ObsidianVault/HermesAgent
   - ถ้าตัวแปรนี้ไม่มี หรือหา script ไม่เจอ → รายงานเจ้าของงาน ห้ามเดา path ใหม่เอง
2. frontmatter ของโน้ตต้องมีฟิลด์วงจร PDCA ครบ (ระบบแจ้งเตือนใช้ฟิลด์นี้):
   status: trial | active | cancelled | deleted   (เริ่มที่ trial เสมอ)
   decision: pending | use-real | cancel | delete  (เริ่มที่ pending)
   review_due: <YYYY-MM-DD วันครบกำหนดประเมิน ตามรอบที่เจ้าของกำหนด>
   source_link: <ลิงก์ต้นทางที่สรุปมา>
   hermes_km: https://eoffice.jigsawgroups.work/hr/hermes-km
   tags: type/<ประเภท> · topic/<หัวข้อ> · project/<โปรเจกต์> · status/review (+ pdca ถ้าอยู่รอบทดลอง)
3. รัน build_pdca_review_dashboard.mjs เพื่ออัปเดตศูนย์กลาง
4. แจ้งเจ้าของงานเป็นลิงก์กลับ: รายการนี้ติดตามต่อได้ที่ https://eoffice.jigsawgroups.work/hr/hermes-km
   เมื่อถึง review_due ระบบ hermes-km จะแจ้งเตือนให้ตัดสิน: ใช้จริง (active) / ยกเลิก (cancelled) / ลบ (deleted)

ห้าม:
- เดาว่าผู้ใช้อยากบันทึก / ใส่ 100% ถ้าไม่ตรวจ
- สร้าง rule-shortcut ใหม่ทันทีไม่เสนอรีวิว / ตอบแค่ route การเก็บโดยไม่วิเคราะห์
- สร้าง vault/project ใหม่ทดสอบถ้าไม่อนุญาต
- บันทึกระเบียนโดยไม่มี status + review_due + ลิงก์กลับ hermes-km (ห้ามทิ้งให้ค้างไม่มีวงจร)
- ติดตั้งแล้วจบโดยไม่มี scheduled evaluation + rollback
```

## Changelog

- v1.7 (2026-06-24): ผ่านการตรวจ 2 AI (Claude ร่าง · Codex cross-check) · เพิ่มโหมด สั้น/ปกติ/ลึก/ติดตั้ง (โหมดคุมจำนวนส่วน) · truth check 3 ชั้น + ระดับความมั่นใจ · แยก 2 เฟส review-only กับ after-approval · เปลี่ยน path เป็นตัวแปร HERMES_OBSIDIAN_ROOT (แก้ปัญหาพนักงานเครื่องอื่น) · เพิ่มวงจร PDCA จริง: ทุกระเบียนต้องมี status + decision + review_due + ลิงก์กลับ hr/hermes-km เพื่อให้ระบบแจ้งเตือนตัวที่ครบกำหนดและไม่ทิ้งให้ค้าง
- v1.6 (2026-05-31): เพิ่ม dashboard sync + standard tags

## Graph Links

- Parent hub: [[skills/README|skills]]
- Router: [[00-Center/docs/AI_SKILL_ROUTER|AI Skill Router]]
- Graph: [[00-Center/docs/SKILL_GRAPH|Skill Graph]]
