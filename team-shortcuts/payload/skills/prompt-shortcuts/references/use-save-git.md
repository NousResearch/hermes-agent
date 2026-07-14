---
title: Use Save Git
aliases:
  - Use Save Git
  - Save Git
  - Save Grid
  - save-grid
  - Use Save Grid
  - save-git
  - use-save-git
  - ใช้ Save Git
  - เซฟ Git
  - ก่อน push
  - ก่อน merge
  - ก่อน deploy
  - Git Safe Flow
  - GitLab Deploy Safe Flow
  - Use GitLab Deploy Safe Flow
  - Use Ship Gate
  - Merge Gate
  - Pre-Merge Gate
  - Pre-Merge Production Gate
tags:
  - prompt-shortcuts
  - git
  - gitlab
  - deploy
  - vps
  - safety-gate
status: active
version: 2.2
updated: 2026-07-14
supersedes: v1.3 (retired 2026-06-08 · do not use old action-based flow)
---

# Use Save Git (v2.2)

## Shortcut

```text
Use Save Git
```

## Purpose

บังคับ flow ปลอดภัยก่อน push, merge, deploy โดย AI ต้องรัน "ด่านตรวจเดียวที่รันได้จริง" แล้วออกใบผ่านเป็น **decision token เดียว + ตาราง Grid เดียว** ก่อน
เจ้าของงานเป็นคนกด merge เอง แต่ AI ต้องตรวจครบ 5 ด่านและบอกชัดว่าชั้นไหนผ่าน ชั้นไหนคือตัว block จริง

`gate` = ด่านตรวจ · `decision token` = คำตัดสินคำเดียว เช่น SAFE_TO_MERGE · `Grid` = ตารางสรุปทุกชั้นในที่เดียว
`SHA` = เลข commit ใช้เทียบว่าแต่ละที่เป็นโค้ดชุดเดียวกัน · `dry-run` = ลองรันบน VPS โดยยังไม่แตะ production จริง

`Save Grid` / `Use Save Grid` เป็น alias ของ `Use Save Git` — หมายถึงให้สร้างตาราง Grid จากด่านตรวจเดียวกัน ไม่ใช่ Shortcut อีกตัว

## Entry Gate · เรียกเมื่อจำเป็นเท่านั้น

- เรียกเมื่อจะ commit, push, merge, deploy หรือกล่าวอ้างว่า Git พร้อมส่ง
- ถ้าแชทไม่มีการเปลี่ยน Git/ไม่มีการส่งโค้ด/เป็นงานอ่านหรือวางแผน ให้คืน `SAVE_GIT_NOT_APPLICABLE` โดยไม่รัน 5 ด่าน
- Close Chat เรียก Save Git เฉพาะเมื่อ Entry Gate เข้าเงื่อนไข ห้ามเรียกทุกครั้งที่ปิดแชท
- ผลสำเร็จต้องออก evidence receipt: project / task_id / branch / HEAD SHA / stage / timestamp / dirty state · ตัวรับใช้ซ้ำได้เฉพาะเมื่อค่าทั้งหมดตรงและไม่มีไฟล์เปลี่ยน

## เปลี่ยนจาก v1.3 (retired) — ทำไมต้องเขียนใหม่

v1.3 เป็นกฎ "ห้าม" เยอะแต่ไปทำงานจริงไม่จบ เลยถูกเลิกใช้ · v2 อุดต้นเหตุจากปัญหาจริง 11 project

| v1.3 พังเพราะ | v2 แก้ด้วย |
|---|---|
| กฎห้ามเยอะ ไม่มีคำสั่งรันจริง 1 เดียว | 1 project = 1 คำสั่ง `save-git` + ไฟล์ตั้งค่า `.savegit.json` ต่อ project |
| ตรวจด้วยการอ่านไฟล์ (tier 1-2) | บังคับ tier 3+ · รัน healthcheck จริงใน container/service จริง |
| `STOP_*` กลายเป็นทางตัน | ทุก block มี "Fix needed" + AI ทำ remediation ต่อเอง |
| ไม่เช็ค scope ของ MR | ด่าน 2 เช็ค commit count + files changed + target ตรง scope |
| ไม่มี `commitSha` ใน health | ด่าน 5 บังคับ health คืน commitSha เทียบ origin/main |
| ไม่รู้จัก stack + ไม่มี dry-run | adapter ต่อ stack (npm/pnpm/python) + ด่าน 4 VPS dry-run |

## Runtime · คำสั่งเดียวที่ AI ต้องรัน

```bash
# ตรวจครบก่อน merge (ด่าน 1-3) แล้วออก decision + Grid
save-git --stage merge-gate

# ตรวจครบทั้งสายก่อน deploy/ship (ด่าน 1-5)
save-git --stage ship-gate --health-url https://example.com/api/health

# ตรวจทีละด่านได้
save-git --stage local        # --fast = ข้าม build/test (ใช้ใน pre-push hook)
save-git --stage mr
save-git --stage ci
save-git --stage dryrun
save-git --stage prod
```

ถ้าเปิด browser/URL อยู่ ให้ใช้ URL นั้น resolve project ก่อน cwd ของ shell เสมอ

```bash
save-git --url http://127.0.0.1:7421/ui --stage merge-gate
```

ถ้า `~/.local/bin` ไม่อยู่ใน PATH ให้เรียกตรง

```bash
python3 $HERMES_OBSIDIAN_ROOT/99-System/scripts/save_git_gate.py --stage merge-gate
```

ติดตั้ง pre-push hook ให้ทุก repo (hook เรียก `--stage local --fast`):

```bash
python3 $HERMES_OBSIDIAN_ROOT/99-System/scripts/install_save_git_hooks.py
```

### ไฟล์ตั้งค่าต่อ project · `.savegit.json` (วางที่ root ของ repo)

นี่คือ adapter ที่ทำให้ด่านตรวจรู้จักเครื่องมือจริงของแต่ละ project · แก้ปัญหา "ตรวจไม่ตรง stack"
gate อ่าน `.savegit.json` ก่อน (ถ้าไม่มีจะลองไฟล์ yaml เมื่อมี pyyaml)

[skip มี 2 แบบ ห้ามปนกัน — แก้ปัญหา "skip เงียบแล้วดูเหมือนผ่าน"]
- `skip-ok` = ตั้งใจไม่มีจริง (เช่นไม่มี CI) ต้องมีเหตุผลใน config เช่น `ci.skip_reason` · ไม่ใช่แค่ไม่ใส่ค่า
- `skip-risk` = field สำคัญของ stage นั้นไม่ถูกตั้ง = ขาดข้อมูลตรวจ · ขึ้นใน Grid ชัด · **ห้ามนับเป็นผ่าน · ห้ามออก SAFE_TO_MERGE/SAFE_TO_DEPLOY**

[field ขั้นต่ำต่อ stage — ขาด = ไม่ปล่อยผ่านเงียบ]
- merge-gate ต้องมี: `remote_must_match` / `default_target` / `scope_guard` · ขาด = OWNER_DECISION_REQUIRED
- ship-gate ต้องมี: `deploy.branch_only` / `deploy.health_url` / `deploy.health_commit_field` · ขาด = BLOCKED
- โปรเจกต์เก่าที่ยังไม่ครบ → รัน `save-git --stage <x> --audit` ก่อน 1 รอบ เพื่อรายงาน field ที่ขาด (audit ไม่ block) แล้วค่อย enforce

```json
{
  "project": "contentfactory",
  "remote_must_match": "contentfactory.git",
  "default_target": "main",
  "stack": "pnpm",
  "checks": {
    "install": "pnpm install --frozen-lockfile",
    "lint": "pnpm lint",
    "typecheck": "pnpm typecheck",
    "build": "pnpm build",
    "test": "pnpm test",
    "audit": "pnpm audit --audit-level high"
  },
  "test_timeout_sec": 600,
  "forbidden_paths": [".env", "*.key", "*.pem"],
  "bundle_globs": ["web/dist/**/*.js"],
  "bundle_must_not_contain": ["localhost:", "127.0.0.1:"],
  "scope_guard": { "max_commits": 5, "max_files": 30 },
  "ci": { "enabled": false },
  "deploy": {
    "branch_only": "origin/main",
    "health_url": "https://aicontentfac.com/api/health",
    "health_commit_field": "commitSha",
    "container_health": { "container": "openclaw", "command": "curl -fsS http://127.0.0.1:18789/health" },
    "vps_service": { "name": "venture-radar-api", "expect_workdir": "/srv/projects/contentfactory/runtime" }
  }
}
```

## Decision Token (ตอบได้แค่ค่าในรายการนี้)

ด่าน merge ·
- `SAFE_TO_MERGE` — เจ้าของงานกด merge ได้
- `BLOCKED_DO_NOT_MERGE` — ห้าม merge · ต้องมีเหตุผล + ชั้นที่ต้องแก้
- `OWNER_DECISION_REQUIRED` — ต้องให้เจ้าของงานตัดสิน เช่น breaking change

ด่าน deploy ·
- `SAFE_TO_DEPLOY` — deploy จาก origin/main ได้
- `PRODUCTION_VERIFIED` — deploy แล้ว · SHA + health + commitSha ตรงครบ
- `PRODUCTION_NOT_VERIFIED` — health 200 แต่ยังไม่ชัวร์ว่า commit ล่าสุด · ห้ามบอกว่าเสร็จ

## Grid · ตารางสรุปบังคับ (อุดปัญหา "ไม่รู้ชั้นไหน block")

gate พิมพ์ตารางนี้ให้อัตโนมัติทุกครั้ง · AI ต้องส่งต่อให้เจ้าของงานครบ

```text
Decision: <token>

| ด่าน | ตรวจอะไร | ผล | block |
|---|---|---|---|
| 1 Local      | git clean, diff in scope, secret, build/test/lint, bundle | pass/fail/skip | yes/no |
| 2 MR sanity  | project/remote, source/target, sync main, commit+files scope, conflict | pass/fail/skip | yes/no |
| 3 CI         | latest-commit pipeline passed, not stuck, migration | pass/fail/skip | yes/no |
| 4 VPS dryrun | build candidate, env/port/service, container health | pass/fail/skip | yes/no |
| 5 Production | deploy from origin/main, SHA match, health commitSha, service path, rollback | pass/fail/skip | yes/no |

Blocking layer: <ด่านที่ block จริง หรือ none>
Fix needed: <สิ่งที่ต้องแก้ ถ้า block>
Owner action: <กด merge ได้ / ห้าม merge เพราะ... / ต้องตัดสินใจเรื่อง...>
```

## Machine Output + Integrity Contract (v2.1)

นอกจาก Grid (คนอ่าน) gate ต้องออก JSON (เครื่องอ่าน · ใช้ต่อใน GitLab merge rule / pre-push hook):
`{ schema_version, decision, exit_code, blocking_layer, per_stage:{result(pass/fail/skip-ok/skip-risk), evidence}, timestamp, shas:{local,origin,deployed,health} }`
- fail-closed: JSON parse ไม่ได้ / field หาย / schema_version ไม่รองรับ = ถือว่า BLOCKED
- `$HERMES_OBSIDIAN_ROOT` ไม่ถูกตั้ง → fail ชัด ห้าม fallback ไป path เดา

ความถูกต้องของ SHA (กัน commitSha หลอก):
- ก่อนเทียบ SHA ต้องเช็ก worktree ไม่ dirty / ไม่มี unpushed commit / ไม่ detached HEAD
- ต้องตรงทั้ง 3 ตัว: deployed SHA = origin/<target> SHA = health commitSha
- health commitSha ต้องมาจาก build จริง (ฝังตอน build/CI artifact) ไม่ใช่ค่าที่ service ตั้งเอง
- health request บังคับ no-store + ตรวจ freshness (timestamp/build id) กัน cache ค้าง

## Prompt

```text
Use Save Git

งานนี้ทำตามมาตรฐาน Git/GitLab/VPS แบบปลอดภัยก่อน push, merge, deploy
หลักใหญ่: ห้ามให้ GitLab merge เป็นที่ลองผิดลองถูก ให้ AI รันด่านตรวจเดียวจนออกใบผ่านก่อน เจ้าของงานค่อยกด merge

เป้าหมาย:
ทำให้ Local, GitLab, VPS ตรงกันอย่างตรวจสอบได้ ไม่เดา ไม่ใช้ความจำ และไม่บอกว่าเสร็จถ้ายังไม่มีหลักฐานจากการรันจริง

ขั้นแรกเสมอ: รัน save-git ตาม stage ที่ตรงกับงาน (merge-gate ก่อน merge, ship-gate ก่อน deploy)
ห้ามตอบเรื่อง merge/deploy จากความจำหรือความรู้สึก ต้องอ่านผล gate จริง

กฎห้ามทำ (ทุกข้อต้องมีสิ่งที่ทำต่อ ไม่ใช่หยุดเฉย ๆ):
- ห้ามอ่านหรือแสดงค่าใน .env, token, password, secret, private key → สแกนชื่อ/รูปแบบแทน
- ห้ามพูดว่า "merge ได้" จาก test local อย่างเดียว → ต้องผ่าน merge-gate ครบ
- ห้าม merge ถ้า worktree dirty/MR ผิด scope → classify แล้ว remediate ก่อน
- ห้าม deploy จาก feature branch หรือ local commit ที่ยังไม่อยู่ GitLab main → deploy จาก origin/main เท่านั้น
- ห้าม force push ยกเว้นเจ้าของงานสั่งชัด + มี backup/tag
- ห้ามบอกว่าเสร็จถ้า Local/GitLab/VPS SHA ไม่ตรง หรือ health commitSha ไม่ตรง commit ล่าสุด
- ห้าม fallback ไป cwd ของ shell ถ้า browser/URL ชี้คนละ project

ด่านตรวจ (gate อ่านค่าจาก .savegit.json):

ด่าน 1 Local (ก่อน push):
- git status clean, diff เฉพาะไฟล์ใน scope
- secret scan ใน diff (ไม่อ่านค่า) + เช็ค forbidden_paths
- รัน build/test/lint/typecheck/audit ตาม stack
- test เกินเวลา test_timeout_sec = "timeout" แยกจาก "fail"
- bundle_must_not_contain กัน frontend ชี้ localhost

ด่าน 2 MR sanity (ก่อน merge):
- อยู่ถูก project/remote (เทียบ remote_must_match)
- MR source = local branch, target = default_target จริง (ห้ามเดา GitLab default)
- branch sync กับ origin/<target> ล่าสุด (กัน conflict/ทับงาน)
- commit count + files changed ตรง scope (scope_guard) → บวมผิดปกติ = BLOCKED จนตรวจ target
- ไม่มี conflict

ด่าน 3 CI:
- pipeline ของ commit ล่าสุดต้อง passed (ไม่ใช่ commit เก่า)
- stuck/pending/runner ไม่รับงาน = BLOCKED (ปัญหา CI ไม่ใช่โค้ด ต้องบอกชัด)
- migration/schema ผ่านถ้ามี database

ด่าน 4 VPS Dry-Run (ก่อนแตะ production):
- build candidate บน worktree แยก, เช็ค env/port/service (ไม่อ่านค่า secret)
- รัน container_health command จริง (docker exec ... curl /health) ไม่ใช่อ่าน yaml
- ห้าม restart production ในด่านนี้

ด่าน 5 Production (หลัง merge + deploy):
- deploy จาก deploy.branch_only (origin/main) เท่านั้น
- deployed SHA = origin/main SHA
- health_url คืน health_commit_field = commit ล่าสุด (กัน health 200 หลอก)
- vps_service ชี้ expect_workdir ถูก path + มี rollback ก่อนบอก PRODUCTION_VERIFIED

Autonomous Remediation Rule:
- BLOCKED ไม่ใช่ final answer เป็นสัญญาณให้หยุด action เสี่ยง แล้วแก้ root cause ต่อ
- dirty files ที่เป็น safety update ของ AI เอง → classify, verify, commit branch แยก, รัน gate ซ้ำ
- งานคนอื่น → ห้ามแตะ ใช้ branch/worktree แยก
- health fail เพราะตรวจผิด target/URL → แก้ target ก่อน ไม่สรุปว่า service พัง
- หยุดถามเจ้าของงานได้เฉพาะ secret / permission / destructive / business decision ที่ AI ตัดสินแทนไม่ได้

Output บังคับ:
- decision token เดียว + Grid ครบ 5 ด่าน + Blocking layer + Fix needed + Owner action
- ห้ามจบด้วย "ตรวจได้อย่างเดียว" หรือ "ไป audit ต่อ" ถ้ายังไม่ลงมือ
- ห้ามบอก merge ได้ถ้า Blocking layer ไม่ใช่ none
```

## Minimum Output

```text
Decision: <token>
Blocking layer: <ด่าน หรือ none>
Grid: <ตาราง 5 ด่าน>
Fix needed: <ถ้ามี>
Remediation performed: <AI ทำอะไรไปแล้ว>
Gate rerun result: <ผลรันซ้ำ>
Owner action: กด merge ได้ / ห้าม merge / ต้องตัดสินใจ
```

## Examples

### ผ่าน

```text
Decision: SAFE_TO_MERGE
Blocking layer: none
Owner action: กด merge MR !12 (codex/feat-x → main) ได้; หลัง merge ยังห้าม deploy จนกว่า ship-gate ผ่าน
```

### ถูก block แล้วระบุชั้นชัด (เคส MR target ผิด)

```text
Decision: BLOCKED_DO_NOT_MERGE
Blocking layer: 2 MR sanity
Fix needed: MR target เป็น master แต่งานจริงอยู่ main · MR บวม 48 commit/217 ไฟล์ ทั้งที่งานจริง 1 commit/7 ไฟล์
Remediation performed: ปิด MR เก่า, เปิด MR ใหม่ target=main
Gate rerun result: ด่าน 2 pass → SAFE_TO_MERGE
Owner action: กด merge MR ใหม่ได้
```

### health หลอก

```text
Decision: PRODUCTION_NOT_VERIFIED
Blocking layer: 5 Production
Fix needed: /api/health 200 แต่ commitSha=abc111 ไม่ตรง origin/main=def222 · production รัน commit เก่า
Remediation performed: redeploy จาก origin/main, รอ service, ยิง health ซ้ำ
Gate rerun result: commitSha=def222 ตรง → PRODUCTION_VERIFIED
Owner action: ตัด local ได้
```

## Worktree Lifecycle v1

อ่าน `worktree-lifecycle-contract.md` ก่อนใช้ Prompt นี้ · ก่อน commit/push ตรวจ `hermes worktree status` ว่า task/path/branch/writer ตรงทะเบียน และตรวจ unpushed/permit จริง · `WTL_BLOCKED` = ห้ามส่ง Git

## Changelog

- v2.2 (2026-07-14): เพิ่ม alias `Save Grid` + Entry Gate ไม่รัน 5 ด่านในแชทที่ไม่มี Git action + evidence receipt ให้ Close/New ใช้ผลเดิมโดยตรวจ project/task/branch/SHA ก่อน
- v2.1 (2026-06-24): ผ่านตรวจ 2 AI (Claude+Codex) · แยก skip เป็น skip-ok/skip-risk (skip-risk ห้ามออก SAFE) + field ขั้นต่ำต่อ stage (ขาด = OWNER_DECISION/BLOCKED ไม่ใช่ skip เงียบ) + โหมด --audit ให้โปรเจกต์เก่า · เพิ่ม machine JSON output (schema_version/exit_code/fail-closed) คู่ Grid · path สคริปต์เป็น $HERMES_OBSIDIAN_ROOT (พกพา) · กัน commitSha หลอก (3-way SHA + build provenance + no-store/freshness + เช็ก dirty/unpushed/detached ก่อนเทียบ)
- v2.0 (2026-06-08): เขียนใหม่เป็นด่านรันได้จริง 5 ชั้น + decision token + Grid (เลิก v1.3)

## Graph Links

- Parent hub: [[skills/README|skills]]
- Registry: [[ai-context/prompt-shortcut-registry|Prompt Shortcut Registry]]
- Evidence: [[95-Inbox-Lab/review/save-git-redesign/evidence-and-review|Redesign Evidence and Review]]
- Related: [[ai-context/worktree-routing-gate|Worktree Routing Gate]]
- Related: [[ai-context/ai-workflow-guardian-policy|AI Workflow Guardian Policy]]
