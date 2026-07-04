# pr-review-gate — ด่านตรวจโค้ด AI ก่อน merge (F2)

## แก้ปัญหาอะไร

ทีมใช้สายพาน AI Relay ให้ AI ราคาถูก (Grok/Codex/Gemini) เขียนโค้ด
แต่ก่อนหน้านี้ "ตาที่สอง" มีเฉพาะในแชท — ถ้าพนักงานกด merge เอง โค้ดเข้า repo ทีมโดยไม่ผ่านการรีวิว
ด่านนี้ให้ AI อีกตัว (PR-Agent + Gemini 2.5 Flash ราคาถูก) รีวิวที่ระดับ PR/MR แล้วโพสต์ความเห็นภาษาไทยลงใน PR โดยตรง

ที่มา: เครื่องมือโอเพนซอร์ส [PR-Agent (Qodo Merge)](https://github.com/qodo-ai/pr-agent) ~11k ดาว · Apache 2.0

## ติดตั้ง (ครั้งเดียวต่อเครื่อง)

```bash
bash scripts/pr-review-gate/install-pr-review-gate.sh
```

ต้องมี: `GEMINI_API_KEY` ใน `~/.hermes/.env` (มีอยู่แล้วถ้าติดตั้ง AI Relay) และ `gh` login แล้ว (สำหรับ GitHub)

## ใช้งาน

```bash
pr-review https://github.com/<owner>/<repo>/pull/<เลข>          # รีวิว + ให้คะแนน + จุดเสี่ยง
pr-review <ลิงก์ PR> describe                                    # ให้ AI เขียนสรุปว่า PR นี้ทำอะไร
pr-review <ลิงก์ PR> improve                                     # ขอคำแนะนำแก้โค้ดเป็นจุดๆ
```

ผลรีวิวจะไปโผล่เป็นความเห็นใน PR นั้น (ภาษาไทย ตามที่ตั้งใน config)

## ตั้งค่า

- ไฟล์เครื่อง: `~/.hermes/pr-review-gate/pr_agent.toml` (สร้างให้ตอนติดตั้ง)
- ไฟล์ราย repo: วาง `.pr_agent.toml` ที่รากของ repo นั้น — PR-Agent อ่านเองอัตโนมัติ (repo นี้มีให้เป็นตัวอย่างแล้ว)
- เปลี่ยนรุ่น AI ชั่วคราว: `PR_REVIEW_MODEL="openrouter/anthropic/claude-haiku-4.5" pr-review <ลิงก์>`

## ใช้กับ GitLab ของทีม

1. สร้าง Personal Access Token ใน GitLab ของทีม (สิทธิ์ `api`) — งานคน ~5 นาที
   (GitLab → รูปโปรไฟล์ → Preferences → Access Tokens → เลือก scope `api`)
2. เพิ่ม 2 บรรทัดใน `~/.hermes/.env` บน VPS (URL ไม่ commit ลง repo เพราะเป็นข้อมูลภายใน):
   ```
   GITLAB__PERSONAL_ACCESS_TOKEN=<token>
   GITLAB_URL=https://<gitlab-ของทีม>
   ```
3. `pr-review <ลิงก์ MR>` — ตัว wrapper สลับเป็น GitLab ให้เองเมื่อลิงก์ไม่ใช่ github.com
4. ขั้นบังคับทั้งทีม (เฟสถัดไป): ตั้ง webhook ให้รีวิวอัตโนมัติทุก MR + ผูกเป็นด่านใน save-git stage 2

## ข้อจำกัดที่บอกตรงๆ

- รีวิวได้เฉพาะ PR/MR ที่เปิดแล้วบน GitHub/GitLab — ไม่รีวิว diff ในเครื่อง (นั่นเป็นหน้าที่ Codex gate เดิม)
- AI รีวิวไม่แทนคนตัดสิน: ความเห็นเป็นข้อมูลช่วยตัดสิน คนกด merge ยังเป็นผู้รับผิดชอบ
