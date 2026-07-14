---
title: Use Merge to Production
aliases:
  - Use Merge to Production
  - use-merge-to-production
  - Merge to Production
  - merge-to-production
  - ใช้ Merge to Production
  - ขึ้น production
  - deploy production
  - Ship to Production
tags:
  - prompt-shortcuts
  - git
  - gitlab
  - deploy
  - vps
  - production
  - merger-only
status: active
version: 1.1
updated: 2026-06-24
related: use-save-git
---

# Use Merge to Production

## Shortcut

```text
Use Merge to Production
```

## Purpose

shortcut นี้สำหรับ "คนที่มีสิทธิ์ merge" เท่านั้น ใช้ตอนจะรวมงานเข้า production
แล้ว deploy ขึ้นเซิร์ฟเวอร์จริง ต่อยอดจาก `Use Save Git` v2 แต่เดินครบสายถึง deploy

ต่างจาก `Use Save Git` ตรงไหน:

- `Use Save Git` = ทุกคนใช้ได้ · ตรวจถึงด่าน merge-gate (ก่อนกด merge)
- `Use Merge to Production` = เฉพาะคน merge · ตรวจถึง ship-gate + deploy + ยืนยัน production

`merger` = คนที่มีสิทธิ์รวมงานขึ้น production · `ship-gate` = ด่านตรวจครบก่อนขึ้นจริง

## Merger Allowlist + Identity (ยืนยันตัวตนจริง ไม่ใช่ชื่อที่พิมพ์)

- allowlist อ่านจาก `.savegit.json` (`mergers: [...]`, `deployers: [...]`) **เวอร์ชันบน `origin/<target>` เท่านั้น** ไม่ใช่จาก branch งาน
  (กัน privilege escalation: ถ้าอ่านจาก branch คนแก้ MR เติมชื่อตัวเองเป็น merger ได้)
- ถ้า MR แตะไฟล์ `.savegit.json` → block · ต้องให้เจ้าของงานอนุมัติแยก
- ยืนยันตัวตน merger จาก **GitLab authenticated actor** (API current user / merge_user) ไม่ใช่ชื่อที่ AI อ่านจากแชท
  · SSH user ใช้เป็นบริบทเสริมเท่านั้น (บัญชีร่วม เช่น deploy/ubuntu/root ผูกคนจริงไม่ได้)
- ผู้สั่งไม่ผ่าน → `BLOCKED_NOT_A_MERGER` บอกให้ใช้ `Use Save Git` แทน
- การบังคับจริง 2 ชั้น (server-side): GitLab **protected branch** + **protected environment / production approval**
  (merge ได้ ≠ ควร deploy ได้) · shortcut นี้เป็นด่านเสริมฝั่งคน

## Runtime · ขั้นตอนที่ AI ต้องเดิน

```bash
# 0) ยืนยันผู้สั่งเป็น merger ก่อน ถ้าไม่ใช่ หยุด
# 1) ตรวจก่อน merge
save-git --stage merge-gate
#    ต้องได้ SAFE_TO_MERGE ก่อน · ถ้า BLOCKED ให้ remediate แล้วรันซ้ำ
# 2) เจ้าของงานกด merge MR เข้า target (เช่น master) บน GitLab
# 3) ตรวจครบสายก่อน deploy
save-git --stage ship-gate
# 3.5) จดจุดถอย: previous image digest / container tag / release id ปัจจุบัน + เตรียม deploy.rollback_command
# 4) deploy จาก origin/<target> เท่านั้น (ห้าม deploy จาก local/feature)
#     ใช้คำสั่งจาก .savegit.json deploy.command (ห้าม hardcode ต่อ project)
# 5) ยืนยัน production: container health เขียว + commit ที่ build = origin/<target>
#     verify fail (health/commit ไม่ตรง) → รัน deploy.rollback_command → verify ซ้ำหลัง rollback → รายงาน (ห้ามปล่อยค้าง)
```

อ่านค่าทั้งหมดจาก `.savegit.json` เวอร์ชัน origin/<target> (target branch, deploy.command, deploy.rollback_command, container, health, vps_service, mergers, deployers)
สำหรับเว็บ static: ต้องมี build manifest ฝังตอน build อย่างน้อย `commitSha` + `buildTime` + `sourceBranch` เทียบกับ origin/<target> · ไม่มี manifest = PRODUCTION_NOT_VERIFIED (ห้ามเรียกว่า deploy สำเร็จ)

## Decision Token (ตอบได้แค่ค่าในรายการนี้)

- `SAFE_TO_DEPLOY` — ผ่าน ship-gate · deploy ได้
- `PRODUCTION_VERIFIED` — deploy แล้ว · health เขียว + commit ตรง origin/target
- `PRODUCTION_NOT_VERIFIED` — health ขึ้นแต่ยังพิสูจน์ commit ล่าสุดไม่ได้ · ห้ามบอกว่าเสร็จ
- `BLOCKED_NOT_A_MERGER` — ผู้สั่งไม่มีสิทธิ์ · ให้ใช้ Use Save Git แทน
- `BLOCKED_DO_NOT_DEPLOY` — มีด่าน block · ระบุชั้น + วิธีแก้

## Grid · ตารางสรุปบังคับ

```text
Decision: <token>
Merger: <ชื่อผู้สั่ง · อยู่ใน allowlist yes/no>

| ด่าน | ตรวจอะไร | ผล | block |
|---|---|---|---|
| 0 Merger    | ผู้สั่งอยู่ใน allowlist | pass/fail | yes/no |
| 1 merge-gate| local + MR sanity + CI (จาก Use Save Git) | pass/fail/skip | yes/no |
| 2 Merge     | MR ถูก merge เข้า target แล้ว | pass/fail | yes/no |
| 3 ship-gate | dry-run + env/port/service + container health | pass/fail/skip | yes/no |
| 4 Deploy    | deploy จาก origin/target เท่านั้น | pass/fail | yes/no |
| 5 Verify    | health เขียว + commit ที่ build = origin/target | pass/fail | yes/no |

Blocking layer: <ด่านที่ block หรือ none>
Fix needed: <ถ้า block>
Owner action: <deploy ได้ / ห้าม deploy เพราะ...>
```

## Prompt

```text
Use Merge to Production

งานนี้คือรวมงานเข้า production แล้ว deploy ขึ้นเซิร์ฟเวอร์จริง สำหรับคนที่มีสิทธิ์ merge เท่านั้น

ขั้นแรกเสมอ: เช็คว่าผู้สั่งเป็น merger (อยู่ใน allowlist ของ project) ถ้าไม่ใช่ ตอบ BLOCKED_NOT_A_MERGER แล้วบอกให้ใช้ Use Save Git แทน ห้ามเดินต่อ

จากนั้นเดินตามนี้ ห้ามข้ามด่าน:
1. รัน save-git --stage merge-gate ต้องได้ SAFE_TO_MERGE ก่อน ถ้า BLOCKED ให้แก้ root cause แล้วรันซ้ำ
2. ให้เจ้าของงานกด merge MR เข้า target branch บน GitLab (อ่าน target จาก .savegit.json ห้ามเดา)
3. รัน save-git --stage ship-gate ตรวจ dry-run + container health จริง
4. deploy จาก origin/<target> เท่านั้น ห้าม deploy จาก local หรือ feature branch
   - คำสั่ง deploy ตาม stack ของ project (Lotto: docker compose build แล้ว up -d)
5. ยืนยัน production: container health เขียว และ commit ที่ build ตรงกับ origin/<target>

กฎห้ามทำ (ทุกข้อมีสิ่งที่ต้องทำต่อ ไม่ใช่หยุดเฉย):
- ห้าม deploy ถ้าผู้สั่งไม่ใช่ merger
- ห้าม deploy จาก local commit หรือ feature branch ที่ยังไม่อยู่ origin/target
- ห้ามอ่านหรือแสดงค่าใน .env, token, secret, private key
- ห้ามบอกว่า production เสร็จ ถ้า commit ที่รันจริงไม่ตรง origin/target
- ห้าม force push หรือแก้ของคนอื่น ใช้ branch/worktree แยก

เพิ่มกฎห้าม:
- ห้ามอ่าน mergers/deploy command จาก branch งาน · อ่านจาก .savegit.json บน origin/target เท่านั้น
- MR แตะ .savegit.json = block ขออนุมัติเจ้าของงานแยก
- verify production fail = รัน rollback แล้ว verify ซ้ำ ห้ามปล่อยค้าง

Output บังคับ: decision token เดียว + Grid ครบ 6 ด่าน + Blocking layer + Fix needed + Owner action
สำหรับเว็บ static: เทียบ build manifest (commitSha+buildTime+sourceBranch) กับ origin/target · ไม่มี manifest หรือเทียบไม่ได้ = PRODUCTION_NOT_VERIFIED ห้ามอัปเป็น VERIFIED (ห้ามเทียบด้วยมือแล้วเดา)
```

## Minimum Output

```text
Decision: <token>
Merger: <ชื่อ · in allowlist?>
Grid: <ตาราง 6 ด่าน>
Blocking layer: <ด่าน หรือ none>
Fix needed: <ถ้ามี>
Owner action: deploy ได้ / ห้าม deploy เพราะ...
```

## Worktree Lifecycle v1

อ่าน `worktree-lifecycle-contract.md` ก่อนใช้ Prompt นี้ · หลังยืนยัน merge SHA ให้ `hermes worktree close --merged --merge-sha ...`; จากนั้นสถานะมีสิทธิ์เข้า cleanup review แต่ยังห้ามลบจนผ่าน 6/6 + dry-run + quarantine

## Changelog

- v1.1 (2026-06-24): ผ่านตรวจ 2 AI (Claude+Codex ด้านความปลอดภัย deploy) · ยืนยัน merger จาก GitLab actor จริง (ไม่ใช่ชื่อที่พิมพ์) · allowlist อ่านจาก .savegit.json บน origin/target เท่านั้น (กัน privilege escalation) + MR แตะ .savegit.json = block · แยก mergers/deployers · เพิ่ม rollback บังคับ (rollback_command + verify ซ้ำหลัง rollback) · deploy command อ่านจาก config ไม่ผูก Lotto · static site ต้องมี build manifest (commitSha/buildTime/sourceBranch) ไม่มี = PRODUCTION_NOT_VERIFIED · เพิ่มชั้น GitLab protected environment
- v1.0 (2026-06-09): เวอร์ชันแรก (merger-only ship-gate + deploy + verify)

## Graph Links

- Parent hub: [[skills/README|skills]]
- Registry: [[ai-context/prompt-shortcut-registry|Prompt Shortcut Registry]]
- Related: [[skills/prompt-shortcuts/references/use-save-git|Use Save Git]]
