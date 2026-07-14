---
title: Use Hermes Structure
aliases:
  - Use Hermes Structure
  - use-hermes-structure
  - Hermes Structure
  - ใช้ Hermes Structure
  - มาตรฐานกลาง Hermes
  - hermes standard
tags:
  - prompt-shortcuts
  - hermes-standard
  - central-standard
status: active
version: 1.0
updated: 2026-06-28
---

# Use Hermes Structure

คำสั่งเดียวคุมมาตรฐานกลาง Hermes ทุก project · พิมพ์คำเดียว แล้ว AI โชว์เมนูให้เลือก

## Prompt

```text
Use Hermes Structure

หน้าที่: เป็นประตูเดียวจัดการ "มาตรฐานกลาง Hermes" (โฟลเดอร์ hermes-standard/ ใน repo Hermes Agent)
เจ้าของงานอาจเป็น non-dev → อธิบายภาษาคน · รันคำสั่งให้เอง · หา path เอง · ห้ามให้เจ้าของพิมพ์คำสั่งยาว

[เมื่อถูกเรียก ให้โชว์เมนูนี้ก่อนเสมอ แล้วถามว่าจะเอาข้อไหน หรือ "ทั้งหมด"]

  1) ใช้/แจก   — เอามาตรฐานกลางใส่ project (project เดียว หรือทุกตัวในทะเบียน)
  2) อัปเดต    — แก้กฎกลางที่เดียว แล้วกระจายลงทุก project
  3) ตรวจสอบ   — สแกนสุขภาพไฟล์ + ดูสถานะรวม + ตารางคำด่า
  4) ทำทั้งหมด — รันข้อ 2 (อัปเดต) → 1 (แจก) → 3 (ตรวจ) ตามลำดับ

[map แต่ละข้อไปคำสั่งจริง · เครื่องมืออยู่ที่ hermes-standard/bin/]
ข้อ 1 ใช้/แจก:
- project เดียว:  python3 hermes-standard/bin/safe_apply.py <path> --init
- ทุก project:    python3 hermes-standard/bin/hermes_rollout.py --init   (อ่านจาก hermes-standard/projects.txt)
- ต้องใช้ safe_apply เสมอ (รันเกณฑ์ project ก่อน/หลัง · พังแล้วถอยกลับเอง) · ห้ามแก้ไฟล์ project ตรง ๆ

ข้อ 2 อัปเดต:
- แก้กฎกลาง "ที่เดียว" = hermes-standard/rules/central-block.md (ห้ามไปแก้ทีละ project)
- แล้วสั่ง:  python3 hermes-standard/bin/hermes_rollout.py   (sync · เขียนทับเฉพาะโซนกลาง ไม่แตะโซน project)

ข้อ 3 ตรวจสอบ:
- สุขภาพไฟล์:  python3 hermes-standard/bin/hermes_scan.py <root> --html hermes-standard/scheduler/scan-latest.html
- สถานะรวม:   เปิด hermes-standard/COMPLY.md
- คำด่า/ปัญหาซ้ำ: python3 hermes-standard/bin/curse_track.py report + python3 hermes-standard/bin/hermes_analyze.py --data <data>

ข้อ 4 ทำทั้งหมด: รัน 2 → 1 → 3 ต่อกัน แล้วสรุปผลรวมเป็นภาษาคน

กฎความปลอดภัย (ทุกข้อ):
- รันด้วยของจริง อ่านผลจริง ไม่เดา · บอกผลเป็นภาษาคน (project ไหนผ่าน/พัง/ถอยกลับ)
- ข้อ 1/2 เปลี่ยนไฟล์จริง → ถ้า safe_apply ขึ้น BROKE_ROLLED_BACK ให้รายงานว่า project ไหนถอยกลับ + เพราะอะไร
- งานที่ต้องรันบน VPS/เครื่องคนอื่น หรือเข้า ~/.claude (ติดด่าน relay) → บอกคำสั่งให้เจ้าของรันเอง ไม่แอบทำ
- ถ้าเจ้าของพิมพ์ "Use Hermes Structure ทั้งหมด" หรือระบุข้อมาเลย → ข้ามเมนู ทำข้อนั้นทันที
```

## Worktree Lifecycle v1

อ่าน `worktree-lifecycle-contract.md` ก่อนใช้ Prompt นี้ · การติดตั้ง/อัปเดตมาตรฐาน Hermes ต้องติดตั้ง contract, Manager route, registry path, tracking และคำสั่งตรวจ Notebook/VPS รุ่นเดียวกัน

## Changelog

- v1.0 (2026-06-28): สร้างใหม่ · ประตูเดียวคุมมาตรฐานกลาง Hermes (ใช้/อัปเดต/ตรวจ/ทั้งหมด) · ผูกกับ hermes-standard/bin/ 9 เครื่องมือ

## Graph Links

- Registry: [[ai-context/prompt-shortcut-registry|Prompt Shortcut Registry]]
- ที่เก็บเครื่องมือ: `Tech Tools/Hermes Agent/hermes-standard/`
