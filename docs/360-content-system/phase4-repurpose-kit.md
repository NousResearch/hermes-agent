---
title: เฟส 4 — ชุดแปลงเนื้อหา (blog → LinkedIn + slide)
status: phase-4-design
created: 2026-06-01
owner: พี่นัท (รัตนศักดิ์)
brand: Hi Logic Labs
related:
  - blog-skill-v9.md
  - sample-blog-domain-check.md
  - privacy-gate-rules.md
---

# เฟส 4 — ชุดแปลงเนื้อหา (Repurpose Kit)

> เปลี่ยนบล็อก 1 ใบ ให้กลายเป็นโพสต์ LinkedIn และโครงสไลด์ โดยใช้เสียงเดียวกัน
> (ห้องทดลอง · ไม่โชว์หน้า · อังกฤษ) ทุกชิ้นต้องผ่านด่าน privacy gate

## 1. หลักการแปลง

| ปลายทาง | ความยาว | จุดเด่น | เสียง |
|---|---|---|---|
| LinkedIn | สั้น (~120-180 คำ) | hook + 1 insight + ชวนอ่านบล็อก | we = the lab |
| Slide outline | 6-8 สไลด์ | 1 ความคิดต่อสไลด์ ตัวเลขเด่น | สำหรับวิดีโอไม่โชว์หน้า |

กฎร่วม: ไม่มีชื่อคน/บริษัท/ลูกค้า · ไม่มี "I" บุคคล · ตัวเลขบอกที่มา · ไม่มีคำต้องห้าม anti-AI

---

## 2. ตัวอย่างจริง — LinkedIn post (จาก sample-blog-domain-check)

```
Your domain checker is probably lying to you.

We screened 340 brand-name candidates. A fast lookup tool kept reporting
taken domains as "free" — dozens of them. Trust it, and you fall for a name
you can never own.

The fix is two layers, not one:

1. Fast scan to clear obvious dead ends.
2. A direct query to the registry that actually owns .com — reading for the
   exact "no match" signal.

Then add a control: one name you know is taken, one you know is free. Re-run
them every batch. If a control flips, your tool is rate-limited and feeding
you noise — stop trusting it.

Speed and accuracy aren't the same tool. Any automated check you put inside a
real decision needs a known-answer test running beside it.

Full breakdown → [link to blog]
```

---

## 3. ตัวอย่างจริง — Slide outline (สำหรับวิดีโอไม่โชว์หน้า)

```
Slide 1 (Title)
  "Your domain checker is lying to you"
  subtitle: what 340 name checks taught us

Slide 2 (The problem)
  One quick lookup feels authoritative.
  A registered domain with no nameservers looks exactly like a free one.

Slide 3 (The cost)
  Fast check says "free" → 3 days later it's taken
  → already on your pitch deck.

Slide 4 (What we did — 2 layers)
  Layer 1: fast bulk scan
  Layer 2: direct query to the .com registry

Slide 5 (The control)
  1 known-taken name + 1 known-free name
  Re-run every batch. Control flips = tool is lying.

Slide 6 (The result)
  340 candidates checked
  Clean real-word .com: ~100% taken
  Fast scan: wrong dozens of times

Slide 7 (Takeaway)
  Speed ≠ accuracy.
  Every automated check in a real decision needs a known-answer test beside it.

Slide 8 (CTA)
  Hi Logic Labs — full write-up at hilogiclabs.com
```

---

## 4. ขั้นต่อไป (เฟส 4 ส่วนที่เหลือ)

- Facebook version (โทนเล่าเรื่องกว่า LinkedIn เล็กน้อย)
- รวมหลายบล็อกเรื่องเดียวกัน → สคริปต์ podcast
- Slide outline → วิดีโอ (เสียงบรรยาย + สไลด์ ไม่โชว์หน้า)
- ย้าย pattern นี้ไปสร้างเป็นเครื่องมือจริงใน Content Factory
