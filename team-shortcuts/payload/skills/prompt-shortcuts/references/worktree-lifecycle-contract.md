---
title: Hermes Worktree Lifecycle Contract
aliases:
  - Worktree Lifecycle Contract
  - WTL Contract
  - มาตรฐานวงจรชีวิต Worktree
tags:
  - prompt-shortcuts
  - worktree
  - branch
  - team-workflow
  - notebook
  - vps
status: draft-owner-approved
version: "1.0.0"
updated: 2026-07-14
plan_id: WTL
---

# Hermes Worktree Lifecycle Contract v1.0

> สัญญากลางสำหรับทุก Shortcut และเครื่องมือที่สร้าง อ่าน เขียน ส่งต่อ ปิด หรือเก็บกวาด Git worktree บน Notebook และ VPS
>
> หลักสูงสุด: หนึ่งโครงการมีพื้นที่หลักหนึ่งแห่ง · หนึ่งงานเขียนมี task worktree ของตัวเอง · หนึ่ง task มีเครื่องถือสิทธิ์เขียนได้ครั้งละหนึ่งเครื่อง

## 1. ขอบเขตและแหล่งจริง

สัญญานี้คุม:

- การตั้งชื่อและตำแหน่ง Worktree
- การสร้าง branch และ Worktree ต่อ task
- สมุดทะเบียนกลางและสำเนาบนเครื่อง
- สิทธิ์เขียนและการส่งต่อ Notebook ↔ VPS
- พอร์ต คอนเทนเนอร์ ฐานข้อมูลทดสอบ ไฟล์ชั่วคราว และแฟ้มสะสม
- การพัก ปิด รวมงาน กักพัก และเก็บกวาด
- การวัดพื้นที่และ PDCA
- Shortcut ทุกตัวที่อาจนำไปสู่การเขียนไฟล์

ลำดับแหล่งจริง:

```text
สัญญานี้
→ สมุดทะเบียนกลาง
→ ของจริงจาก Git / process / service / filesystem
→ ความจำใน .project/
→ คำบอกในแชท
```

ถ้าข้อมูลขัดกัน ให้เชื่อของจริงและหยุดงานที่เสี่ยงจนกว่าจะซ่อมทะเบียน

## 2. คำเรียกกลาง

| คำ | ความหมาย |
|---|---|
| Canonical repo | พื้นที่หลักหนึ่งแห่งของโครงการ ใช้เป็นฐานและรับงานที่รวมแล้ว ไม่ใช้แก้ฟีเจอร์พร้อมกันหลายแชท |
| Task worktree | พื้นที่ไฟล์แยกของงานหนึ่งงาน เชื่อมกับ branch หนึ่งกิ่ง |
| Task ID | เลขงานกลางที่ไม่ซ้ำ ใช้ผูก branch, path, สิทธิ์, หลักฐาน และ Shortcut |
| Staff ID | รหัสคนรับผิดชอบ เช่น `nat`, `may`, `mind` |
| Machine ID | รหัสเครื่อง เช่น `notebook-nat`, `vps-linux-nat` |
| Writer lease | สิทธิ์เขียนชั่วคราวของ task บนเครื่องเดียว |
| Runtime namespace | ชุดชื่อเฉพาะของ task สำหรับ port/container/database/temp/cache |
| Registry | สมุดทะเบียนกลางที่บอกว่า Worktree ใดอยู่ที่ไหน ใครใช้ และลบได้หรือไม่ |

## 3. โครงสร้างตำแหน่ง

ค่าปริยาย:

```text
Notebook: ~/Documents/Worktrees/<project-id>/<staff-id>/<task-id>-<slug>
VPS:      /home/linux-nat/.worktree/<project-id>/<staff-id>/<task-id>-<slug>
Branch:   task/<staff-id>/<task-id>-<slug>
```

ข้อบังคับ:

1. ทุก path ต้องเป็น absolute path และผ่าน `realpath`
2. ปลายทางต้องอยู่ใต้ root ที่ลงทะเบียนเท่านั้น
3. ห้าม `..`, symlink ออกนอก root, path ซ้อนทับ Worktree อื่น และ path ซ้ำต่างตัวพิมพ์
4. ห้ามสร้างโครงการหลักซ้ำใน `/srv/projects/` หรือ `/home/linux-nat/projects/`
5. Canonical repo กับ task worktree ต้องไม่ใช่ path เดียวกัน
6. AI ห้ามสร้าง path เองจากความจำ ต้องรับจาก Worktree Manager
7. Notebook ทุกเครื่องใช้โฟลเดอร์ `Documents/Worktrees` ใต้ home ของผู้ใช้เป็น root กลาง ห้ามสร้าง root คู่ขนานที่ `~/Worktrees`

โครงการที่มี root พิเศษเก็บไว้ใน project adapter ได้ แต่ต้องคืนค่าเป็น root เดียวต่อ machine และผ่านกฎข้างบน

## 4. รูปแบบรหัส

```text
project_id : ตัวเล็ก + ตัวเลข + ขีดกลาง
staff_id   : ตัวเล็ก + ตัวเลข + ขีดกลาง
task_id    : <PROJECT>-<AREA>-<เลข> หรือ plan issue เช่น WTL-P2-I3
slug       : ตัวเล็ก + ตัวเลข + ขีดกลาง ไม่เกิน 48 ตัวอักษร
machine_id : <kind>-<owner-or-host>
```

กฎ:

- `task_id` ห้ามใช้ซ้ำแม้ย้ายเครื่อง
- branch และ Worktree path ต้องมี `task_id`
- เปลี่ยน slug ได้เฉพาะก่อนเริ่มเขียน ส่วน `task_id` ห้ามเปลี่ยน
- retry หรือรอบแก้ห้ามสร้าง task id ใหม่เพื่อหลบเพดานหรือสิทธิ์

## 5. สมุดทะเบียนกลาง

สมุดทะเบียนกลางบน VPS เป็นแหล่งจริงของทีม สำเนาบน Notebook ใช้เพื่ออ่านและทำงานเดิมขณะขาดการเชื่อมต่อ แต่ไม่มีสิทธิ์สร้าง โอน หรือลบงานใหม่

ข้อมูลขั้นต่ำต่อ task:

```yaml
schema_version: worktree-lifecycle-v1
project_id: hermes-agent
task_id: WTL-P2-I3
staff_id: nat
machine_id: notebook-nat
canonical_repo: /absolute/path/to/project
worktree_path: /absolute/path/to/task-worktree
branch: task/nat/WTL-P2-I3-open
base_branch: main
base_sha: abc123
remote: origin
state: ACTIVE
write_owner: nat
lease_id: opaque-id
lease_expires_at: 2026-07-14T12:00:00Z
runtime:
  port: 8123
  container_project: wtl-p2-i3
  database_name: test_wtl_p2_i3
  temp_dir: /absolute/path
  cache_namespace: wtl-p2-i3
disk:
  code_bytes: 0
  dependency_bytes: 0
  cache_bytes: 0
  build_bytes: 0
  total_bytes: 0
last_seen_at: 2026-07-14T10:00:00Z
created_at: 2026-07-14T09:00:00Z
cleanup:
  eligible: false
  gate_results: {}
  quarantine_until: null
evidence: []
history: []
```

ข้อบังคับ:

- ทุกการเปลี่ยนสถานะต้องเขียน history แบบต่อท้าย ห้ามลบประวัติ
- เขียนผ่าน lock และ temporary file แล้วสลับชื่อเมื่อข้อมูลครบ
- ห้ามเก็บ token, password, `.env` value หรือ secret ในทะเบียน
- แหล่งจริงของทีมใช้ URI `ssh://<user>@<vps>/<absolute-path>/registry.json`; Manager ต้องถือ file lock บน VPS ตลอด transaction และเขียนแบบ atomic
- ทุกครั้งที่อ่าน/เขียน VPS สำเร็จให้ปรับสำเนา cache บน Notebook; cache ใช้ได้เฉพาะ `status --offline` ของ task/machine/lease เดิมที่ยังไม่หมดอายุ
- โหมด offline ห้าม `open`, `handoff`, `accept`, `close`, `abandon`, `import` และ `cleanup`; คำสั่งเปลี่ยนสถานะต้องติดต่อ authority บน VPS ได้
- ถ้าทะเบียนกับ Git ขัดกัน ให้สถานะเป็น `BLOCKED_REGISTRY_DRIFT`
- งานหนึ่งมี active writer lease ได้หนึ่งชุดเท่านั้น

## 6. วงจรสถานะ

```text
CREATED → ACTIVE → PAUSED → ACTIVE
ACTIVE → IN_REVIEW → ACTIVE
IN_REVIEW → MERGED → CLEANUP_READY → QUARANTINED → ARCHIVED
CREATED/ACTIVE/PAUSED/IN_REVIEW → ABANDONED_BY_OWNER → CLEANUP_READY
ทุกสถานะ → BLOCKED
BLOCKED → สถานะเดิม เมื่อแก้สาเหตุและมีหลักฐาน
```

กฎเปลี่ยนสถานะ:

| จาก | ไป | เงื่อนไข |
|---|---|---|
| CREATED | ACTIVE | Worktree/branch/registry/lease ตรงกัน |
| ACTIVE | PAUSED | บันทึก dirty/unpushed/process และปลด writer lease ตามนโยบาย |
| ACTIVE | IN_REVIEW | push แล้วและมีหลักฐาน diff/SHA |
| IN_REVIEW | MERGED | remote ยืนยัน merge SHA |
| MERGED | CLEANUP_READY | ผ่าน cleanup gate 6/6 |
| CLEANUP_READY | QUARANTINED | dry-run report ถูกบันทึก |
| QUARANTINED | ARCHIVED | พ้น 72 ชั่วโมงและตรวจ 6/6 ซ้ำ |

ห้ามข้ามจาก ACTIVE ไป ARCHIVED โดยตรง

## 7. สิทธิ์เขียนและการส่งต่อเครื่อง

กฎ writer lease:

1. หนึ่ง task มี writer machine ได้หนึ่งเครื่อง
2. ผู้ตรวจอ่านได้ แต่ไม่มีสิทธิ์เปลี่ยนไฟล์
3. ก่อนเขียนทุกครั้งตรวจ `task_id + branch + base_sha + allowed_paths + machine_id + lease_id`
4. branch, SHA, path หรือ machine เปลี่ยนโดยไม่มี handoff ให้หยุด
5. lease หมดอายุไม่แปลว่าลบงานได้ ต้องตรวจ process, Git และเจ้าของก่อน

การส่งต่อ Notebook ↔ VPS:

```text
ตรวจ Git/secret/process
→ commit หรือบันทึก WIP ที่อนุญาต
→ push task branch
→ ปลด lease เครื่องเดิม
→ เปิด/ผูก Worktree เครื่องใหม่จาก remote SHA เดียวกัน
→ รับ lease ใหม่
→ ตรวจ SHA และ runtime ใหม่
→ บันทึก history
```

ห้ามใช้ stash เป็นวิธีส่งต่องาน และห้ามเปิด writer เครื่องใหม่ก่อนเครื่องเดิมปลดสิทธิ์สำเร็จ

## 8. กฎเมื่อขาดการเชื่อมต่อ

Notebook ที่ขาดการเชื่อมต่อทำได้เฉพาะ:

- ทำต่อใน task เดิมที่มี local lease และหลักฐานล่าสุด
- commit ในเครื่องโดยไม่รวมเข้า branch กลาง
- อ่านสถานะสำเนาท้องถิ่น

ห้าม:

- สร้าง task หรือ Worktree ใหม่
- โอน writer
- cleanup หรือ delete
- เปลี่ยน branch ของ Worktree งานอื่น
- อ้างว่าสมุดทะเบียนกลางอัปเดตแล้ว

เมื่อเชื่อมต่อกลับ ต้องเทียบ remote SHA และ registry ก่อน push ถ้า diverged ให้ BLOCKED ห้าม force push อัตโนมัติ

## 9. Runtime namespace

Worktree แยกไฟล์อย่างเดียวไม่พอ งานที่รันต้องแยกทรัพยากรด้วย:

- port: จองผ่านตัวจัดสรรกลาง ห้าม hardcode ซ้ำ
- container project: ใช้ task id เป็น prefix
- database: ใช้ฐานทดสอบหรือ schema แยก ห้ามใช้ production
- temp: อยู่ใต้ task worktree หรือ runtime root ของ task
- cache: ใช้คลังดาวน์โหลดร่วมได้ แต่ namespace ผล build ต้องแยก
- service/process: บันทึก PID หรือ service identity ในทะเบียน

ก่อน handoff, close และ cleanup ต้องตรวจว่าทรัพยากรเหล่านี้หยุดแล้ว

## 10. พื้นที่และงบ

วัดอย่างน้อย 5 ส่วน:

1. tracked code
2. untracked/generated files
3. dependencies
4. cache
5. build/output

ระดับเครื่อง:

| การใช้พื้นที่ | การทำงาน |
|---:|---|
| ต่ำกว่า 70% | เปิดงานได้ตามปกติ |
| 70–84% | เตือนและเสนอคืนพื้นที่จาก cache/build |
| 85–89% | หยุดสร้าง Worktree ใหม่ เปิดได้เฉพาะ close/cleanup |
| 90% ขึ้นไป | โหมดกู้พื้นที่ ห้ามเริ่มงานเขียนใหม่ |

ค่าตั้งต้น: หนึ่งคนเปิด ACTIVE/PAUSED ได้ไม่เกิน 3 Worktree ต่อโครงการ เกินต้องมีเหตุผลและการอนุมัติในทะเบียน

## 11. Cleanup gate 6/6

Worktree จะเป็น `CLEANUP_READY` ได้เมื่อผ่านครบ:

1. `git status` ไม่มีไฟล์ค้าง หรือเจ้าของยืนยัน abandoned พร้อม archive evidence
2. ไม่มี commit ที่ยังไม่ push
3. ไม่มี writer lease, process, service, port, container หรือ database session ใช้งาน
4. branch ถูก merge แล้ว หรือเจ้าของยืนยันเลิกทำ
5. remote branch/SHA และหลักฐานกู้คืนมีอยู่
6. dry-run แสดง path, size, branch, reason และผลกระทบให้ตรวจได้

การคืนพื้นที่มีสองชั้น:

- ชั้นปลอดภัยอัตโนมัติ: cache/build/log ชั่วคราวที่ประกาศไว้
- ชั้นเอา Worktree ออก: ต้อง 6/6 + QUARANTINED 72 ชั่วโมง + ตรวจซ้ำ

คำสั่งเอา Worktree ออกต้องใช้ Git worktree API ผ่าน Manager เท่านั้น ห้าม `rm -rf`

ลำดับ:

```text
stop runtime
→ verify 6/6
→ cleanup dry-run
→ CLEANUP_READY
→ QUARANTINED
→ verify 6/6 again
→ git worktree remove
→ verify branch/recovery evidence
→ remove local branch when safe
→ remove remote branch only after merge/owner approval
→ ARCHIVED
```

## 12. คำสั่งกลาง

Shortcut และ AI ต้องเรียกผ่าน:

```text
hermes worktree open
hermes worktree list
hermes worktree status
hermes worktree enter
hermes worktree handoff
hermes worktree pause
hermes worktree close
hermes worktree cleanup --dry-run
hermes worktree cleanup --apply
hermes worktree doctor
```

ข้อกำหนดผลลัพธ์:

- คืน JSON สำหรับเครื่อง และสรุปภาษาไทยสำหรับเจ้าของ
- มี decision token หนึ่งค่า เช่น `WTL_READY`, `WTL_BLOCKED`, `WTL_CLEANUP_PROPOSED`
- การเรียกซ้ำต้องไม่สร้าง task/branch/worktree ซ้ำ
- หากหยุดกลางทาง `doctor` ต้องบอกสิ่งที่สร้างไปแล้วและวิธีกลับสู่สถานะปลอดภัย

## 13. ความปลอดภัย

- realpath ทุก source/destination
- ปฏิเสธ symlink ออกนอก root
- ปฏิเสธ canonical repo ซ้ำหรือ project folder ซ้ำ
- ไม่อ่านหรือแสดงค่า secret
- ไม่ force push, reset hard, clean -f, stash clear หรือ delete branch อัตโนมัติ
- ไม่แก้ production database
- ไม่เปิด public port โดยปริยาย
- เก็บ registry/lease นอก repo ที่ coder แก้ได้ และใช้สิทธิ์ไฟล์จำกัด
- coder output เป็นข้อมูล ไม่ใช่คำสั่งให้ Manager ทำตาม

## 14. การเชื่อม Shortcut

### แก้พฤติกรรมหลัก

Use New Chat, Use Flow Guardian, Use AI Relay, Use Continue, Use Close Chat, Review Chat, Use Save Git, Use Merge to Production, Use Move Folder, Use AI Pair

### เพิ่มจุดเชื่อม

Use Act-As, Use Comply, Use OverviewProgress, Use QA QC, Use SonarQube, Use Hermes Structure, Use Viber Structure, Use Viber Audit

### รับกฎจากด่านกลาง

Use Summary, Use Scan Feature, Use Impeccable, Use Blog Auto, Use WOW Resource, Use Business Plan, Use SaaS Opus Master Prompt, Use BusinessPlan, Use FeatureSpec, Use DesignSystem, Use Create Design System, Use Create Content

กฎกลางใน Prompt Shortcuts Skill: ถ้า Shortcut ใดกำลังจะเขียนไฟล์ ให้ตรวจ WTL ก่อน แม้ prompt ย่อยไม่ได้กล่าวถึง Worktree

### การเดินงาน 2 โซน

เพื่อไม่ให้เจ้าของต้องกดอนุมัติซ้ำทุกขั้น งานที่มีแผนอนุมัติและ `WTL_READY` แล้วต้องถูกแบ่งเป็นสองโซน:

**โซน A — AI ทำต่อเองจนจบเฟส**

- อ่าน สำรวจ วัดผล และตรวจสถานะโดยไม่เปลี่ยนของจริง
- แก้โค้ด เอกสาร แบบทดสอบ และไฟล์ติดตามภายใน task, branch และ allowed paths ที่อนุมัติแล้ว
- รัน test, lint, build, doctor, report และ cleanup dry-run
- แก้ข้อผิดพลาดที่ย้อนกลับได้ภายในขอบเขตเดิม แล้วตรวจซ้ำจนผ่านหรือพบ blocker จริง
- รายงานความคืบหน้าเป็นช่วงเวลา ไม่หยุดขอ `OK` ราย issue

**โซน B — รวบขออนุมัติระดับ Phase ครั้งเดียว**

- เปิด task/Worktree ใหม่ หรือขยาย allowed paths/ownership จากที่อนุมัติ
- commit, push, tag, merge, deploy, production migration หรือเปิด scheduler/service จริง
- ติดตั้ง dependency, เปลี่ยนสิทธิ์, ใช้เงิน, ติดต่อบุคคลภายนอก หรือแตะ secret
- ย้าย ลบ กักพัก หรือเก็บกวาด Worktree/branch/cache ที่อาจเป็นงานของคนอื่น
- ต้องเดา task id, owner, project route หรือข้อมูลสำคัญที่หลักฐานยังไม่พอ

กติกาการขออนุมัติ: ทำโซน A ที่ไม่ติด blocker ให้ครบก่อน แล้วรวมโซน B เป็นแผน Phase เดียวที่บอกคำสั่ง ผลกระทบ วิธีย้อนกลับ และสิ่งที่เจ้าของต้องกด ห้ามทยอยถามทีละนาที ถ้างานโซน B รายการหนึ่งติดอยู่ งานโซน A รายการอื่นต้องเดินต่อได้

## 15. PDCA

Plan:
- สร้าง task id, path, branch, owner, machine, budget, cleanup criteria

Do:
- เขียนใน task worktree และ runtime namespace ที่ได้รับเท่านั้น

Check:
- ก่อนเขียนทุกครั้ง
- ตอน pause/handoff/review/merge/close
- ตรวจเบาทุก 24 ชั่วโมง
- เสนอ cleanup ทุก 168 ชั่วโมง

Act:
- ซ่อม registry drift
- คืน cache/build ที่ปลอดภัย
- กักพักและเอา Worktree ที่ผ่าน 6/6 ออก
- ปรับค่า budget จากข้อมูลจริงของ Notebook/VPS

จบโครงการเป็นการตรวจบัญชีสุดท้าย ไม่ใช่ครั้งแรกที่เริ่ม cleanup

## 16. เหตุการณ์ที่ต้องถูกปฏิเสธ

1. task เดียวมี writer สองเครื่อง
2. branch ไม่มี task id
3. path อยู่นอก registered root
4. Worktree ใหม่ทับ path เดิม
5. handoff โดยยังมี dirty/unpushed/secret
6. cleanup เพราะเก่าอย่างเดียว
7. cleanup ขณะมี process หรือ lease
8. ลบด้วย `rm -rf`
9. remote/registry ใช้ไม่ได้แต่พยายามสร้างหรือโอนงาน
10. พื้นที่เครื่องถึง 85% แล้วยังเปิด Worktree ใหม่
11. Shortcut เขียนไฟล์โดยไม่มี task/worktree/permit
12. AI อ้างผ่านโดยไม่มีผลตัวตรวจ

## 17. เกณฑ์ประกาศใช้

ประกาศ active ได้เมื่อ:

- ตัวตรวจสัญญาปฏิเสธเหตุการณ์เสีย 12/12
- Worktree Manager ผ่านเหตุการณ์ทดสอบ WTL-P5 12/12
- Shortcut visibility เห็นกฎ WTL 30/30
- Pilot Hermes Agent และโครงการไม่ใช้งานจริงลูกค้าผ่าน
- Notebook และ VPS มีหลักฐาน route/registry/disk จริง
- cleanup dry-run ไม่ลบไฟล์และแสดงผลกระทบครบ
- เจ้าของตรวจแผนและผล Pilot

ก่อนครบทุกข้อ สถานะต้องเป็น `draft-owner-approved` ห้ามบอกว่าเปิดใช้ทั้งทีมแล้ว

## 18. Decision tokens

| Token | ความหมาย |
|---|---|
| WTL_READY | task/worktree/branch/lease/runtime ตรงและพร้อมทำงาน |
| WTL_READ_ONLY | อ่านหรือตรวจได้ แต่ไม่มีสิทธิ์เขียน |
| WTL_BLOCKED | มีความขัดแย้งหรือหลักฐานไม่พอ ห้ามเขียน |
| WTL_HANDOFF_READY | เครื่องเดิมปลดสิทธิ์และ remote SHA พร้อมรับต่อ |
| WTL_CLEANUP_PROPOSED | dry-run พร้อมให้เจ้าของตรวจ ยังไม่ลบ |
| WTL_CLEANUP_READY | ผ่าน 6/6 และเข้ากักพักได้ |
| WTL_ARCHIVED | Worktree ถูกเอาออกอย่างปลอดภัยและมีหลักฐานกู้คืน |

## Graph Links

- [[ai-context/worktree-routing-gate|Worktree Routing Gate]]
- [[ai-context/ai-new-chat-startup-gate|AI New Chat Startup Gate]]
- [[skills/prompt-shortcuts/references/use-new-chat|Use New Chat]]
- [[skills/prompt-shortcuts/references/use-ai-relay|Use AI Relay]]
- [[skills/prompt-shortcuts/references/use-close-chat|Use Close Chat]]
- [[skills/prompt-shortcuts/references/use-save-git|Use Save Git]]
