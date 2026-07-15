> memory-schema: v1.2
> plan_id: WTL
> owner-approved: 2026-07-14
> อ่านไฟล์นี้ก่อนทำงาน Worktree Lifecycle ทุกครั้ง แล้วเปิด `.project/OverviewProgress.md` และ `.project/decisions.md` ประกอบ

# Plan WTL — Hermes Worktree Lifecycle Standard v1.0

## เป้าหมาย

สร้างมาตรฐานเดียวสำหรับการสร้าง ใช้งาน ส่งต่อ ปิด และเก็บกวาด Git worktree บน Notebook และ VPS เพื่อให้หลายแชท หลาย AI และหลายคนทำงานร่วมกันโดยไม่สลับ branch หรือเขียนทับไฟล์ของกันและกัน

## กติกาป้องกัน AI หลงทาง

1. ทุกงานในแผนนี้ใช้เลข `WTL-P<phase>-I<issue>` เท่านั้น
2. ก่อนแก้ไฟล์ทุกครั้งต้องเปิดไฟล์นี้และระบุ Phase/Issue ที่กำลังทำ
3. Issue จะเป็น 100% ได้เมื่อช่องเกณฑ์ผ่านครบและมีหลักฐานจริงในช่อง Evidence
4. คำบอกของ AI ไม่นับเป็นหลักฐาน ต้องใช้ผลคำสั่ง test, lint, build, ตัวตรวจเอกสาร หรือการยืนยันของเจ้าของตามชนิดงาน
5. Phase เป็น 100% เมื่อ Issue ใน Phase นั้นครบ 100% ทุกข้อ ไม่มีการปัดเศษขึ้น
6. งานที่ยังไม่มีหลักฐานให้ใช้สถานะ `claimed` หรือ `pending` ห้ามใช้ `verified`
7. หลังคำถามแทรก การย่อบริบท หรือการกลับมาแชทใหม่ ต้องอ่านส่วน “งานถัดไป” ด้านล่างก่อนลงมือ
8. ห้ามแตะไฟล์ค้างของแชทอื่น ห้ามย้าย ห้ามซ่อน และห้ามลบ Worktree ที่ยังไม่ผ่านด่านเก็บกวาด
9. ห้าม merge, push, deploy หรือแก้ production โดยไม่มีคำสั่งอนุมัติแยกจากเจ้าของ

## กติกาเดินงาน 2 โซน

- **โซน A — AI ทำต่อเอง:** อ่าน สำรวจ แก้โค้ด/เอกสาร/แบบทดสอบในขอบเขต WTL ที่อนุมัติแล้ว รันตัวตรวจ ทำรายงานแบบไม่ลบ และแก้ข้อผิดพลาดที่ย้อนกลับได้จนจบเฟส
- **โซน B — ขออนุมัติรวมครั้งเดียว:** เปิดหรือขยาย Worktree/สิทธิ์, commit/push/tag/merge/deploy, เปิด scheduler/service, ติดตั้ง dependency, เปลี่ยนสิทธิ์, ใช้เงิน/คนนอก/secret, ย้ายหรือลบ และกรณีต้องเดา task id/owner
- AI ต้องทำโซน A ที่ไม่ติด blocker ให้ครบก่อน แล้วรวบโซน B เป็น Phase เดียวพร้อมผลกระทบ วิธีย้อนกลับ และสิ่งที่เจ้าของต้องกด ห้ามถามราย issue

## ขอบเขตที่อนุมัติ

- สัญญากลาง Worktree Lifecycle
- Worktree Manager และสมุดทะเบียนกลาง
- การแยก Notebook/VPS, ผู้ถือสิทธิ์เขียน และทรัพยากรตอนรัน
- วงจรเปิด พัก ส่งต่อ ปิด กักพัก และเก็บกวาด
- การวัดพื้นที่และ PDCA
- Shortcut 30/30 ตัว: แก้หลัก 10, เพิ่มจุดเชื่อม 8, รับกฎกลาง 12
- กฎกลาง ชุดแจกทีม ตัวตรวจ และการย้าย Worktree เดิมเข้าทะเบียน

## ข้อห้าม

- ห้ามใช้ `rm -rf` เก็บกวาด Worktree
- ห้ามลบจากอายุเพียงอย่างเดียว
- ห้ามให้สองเครื่องเขียน `task_id` เดียวกันพร้อมกัน
- ห้ามสร้าง Worktree นอก root ที่กำหนด
- ห้ามแก้ไฟล์ `.env*`, secret, production database หรือ deployment ในแผนนี้
- ห้ามเขียนทับ `.project/plan.md` ซึ่งเป็นแผน QAQC/MW ที่ใช้งานอยู่
- ห้ามแตะไฟล์ค้างใน worktree หลักหรือคลัง Obsidian หลัก

## สูตรคำนวณ

- Issue %: 100 = `verified` และมีหลักฐานจริง; สถานะอื่นทั้งหมด = 0 ตาม Use Comply
- Phase % = ผลรวม Issue % / จำนวน Issue
- Project % = จำนวน Issue ที่ 100% / จำนวน Issue ทั้งหมด × 100
- ความพร้อมบางส่วนให้บอกในสถานะและ Evidence แต่ห้ามนำมาบวกเป็นเปอร์เซ็นต์งานเสร็จ

## สรุป Phase

| Phase | เป้าหมาย | Issue | 100% | Phase % | สถานะ |
|---|---|---:|---:|---:|---|
| WTL-P0 | ตั้งพื้นที่และแผนติดตาม | 6 | 6 | 100% | verified |
| WTL-P1 | สัญญากลางและแบบข้อมูล | 9 | 9 | 100% | verified |
| WTL-P2 | Worktree Manager | 10 | 10 | 100% | verified |
| WTL-P3 | เชื่อม Shortcut และกฎกลาง | 20 | 20 | 100% | verified |
| WTL-P4 | สำรวจและกำหนดทางจัดการของเดิม | 6 | 6 | 100% | verified |
| WTL-P5 | ทดสอบเหตุการณ์จริง | 12 | 12 | 100% | verified |
| WTL-P6 | ทดลองใช้ เปิดใช้ และ PDCA | 8 | 8 | 100% | verified |
| **รวม** |  | **71** | **71** | **100%** | verified |

## WTL-P0 — ตั้งพื้นที่และแผนติดตาม

| Issue | งาน | เกณฑ์ผ่าน | % | สถานะ | Evidence |
|---|---|---|---:|---|---|
| WTL-P0-I1 | แยก Worktree โค้ด | branch/path แยกจากพื้นที่งานอื่น | 100 | verified | `/Users/rattanasak/Documents/Viber Project/Tech Tools/HermesAgent-wt-wtl` · branch `codex/wtl-worktree-lifecycle` |
| WTL-P0-I2 | แยก Worktree คลังคำสั่งกลาง | branch/path แยกจากคลังหลักที่ dirty | 100 | verified | `/Users/rattanasak/ObsidianVault/HermesAgent-wt-wtl` · branch `codex/wtl-worktree-lifecycle` |
| WTL-P0-I3 | สร้างไฟล์ติดตาม WTL | ไฟล์มี Phase/Issue/%/Evidence และเข้า Git | 100 | verified | `git check-ignore` ไม่พบกฎซ่อน · `git ls-files` พบ `.project/plan-wtl.md` · 71 task id ไม่ซ้ำ |
| WTL-P0-I4 | บันทึกอนุมัติเจ้าของ | มีข้อความอนุมัติแผนและสั่งดำเนินการ | 100 | verified | ข้อความเจ้าของ 2026-07-14: “ยืนยันอนุมัติแผน...ดำเนินการได้เลย” |
| WTL-P0-I5 | จำแนก Shortcut | Shortcut 30/30 ถูกจัดกลุ่ม | 100 | verified | 10 แก้หลัก + 8 เพิ่มจุดเชื่อม + 12 รับกฎกลาง |
| WTL-P0-I6 | เก็บค่าฐานเครื่องมือและเทสต์ | ระบุคำสั่งตรวจปัจจุบันและผลก่อนแก้ | 100 | verified | 3 แนวทางหยุดก่อนเข้าเทสต์: ไม่มี venv · pytest config ต้องการ xdist · Python ระบบ 3.9 ใช้ `Callable | None` ไม่ได้; บันทึกเป็น environment-blocked ไม่ใช่ test failure |

## WTL-P1 — สัญญากลางและแบบข้อมูล

| Issue | งาน | เกณฑ์ผ่าน | % | สถานะ | Evidence |
|---|---|---|---:|---|---|
| WTL-P1-I1 | นิยาม canonical repo / task worktree | แยกความหมายและข้อห้ามชัด | 100 | verified | Contract §1-2 |
| WTL-P1-I2 | มาตรฐาน path Notebook/VPS | มี root, pattern, realpath guard | 100 | verified | Contract §3 · เจ้าของล็อก Notebook root เป็น `~/Documents/Worktrees` เมื่อ 2026-07-14 |
| WTL-P1-I3 | มาตรฐาน branch/task/staff/machine id | ชื่อไม่ชนและตรวจย้อนกลับได้ | 100 | verified | Contract §4 |
| WTL-P1-I4 | แบบสมุดทะเบียนกลาง | field, authority, local cache, history ครบ | 100 | verified | Contract §5 |
| WTL-P1-I5 | วงจรสถานะ | CREATED ถึง ARCHIVED + BLOCKED transition ครบ | 100 | verified | Contract §6 |
| WTL-P1-I6 | สิทธิ์เขียนและส่งต่อเครื่อง | writer เดียว + handoff + offline rule | 100 | verified | Contract §7-8 |
| WTL-P1-I7 | แยก runtime | port/container/db/temp/cache namespace ครบ | 100 | verified | Contract §9 |
| WTL-P1-I8 | พื้นที่และเก็บกวาด | 6/6 gate, quarantine, 70/85/90 ครบ | 100 | verified | Contract §10-11, §15 |
| WTL-P1-I9 | ตัวตรวจสัญญา | bad fixture ต้องถูกปฏิเสธ | 100 | verified | `contract_check.py` pass · unittest 4/4 |

## WTL-P2 — Worktree Manager

| Issue | งาน | เกณฑ์ผ่าน | % | สถานะ | Evidence |
|---|---|---|---:|---|---|
| WTL-P2-I1 | โครงคำสั่ง `hermes worktree` | help แสดงคำสั่งย่อยครบ | 100 | verified | parser test เห็น 13 subcommands |
| WTL-P2-I2 | สมุดทะเบียนและการล็อก | เขียนแบบปลอดภัยและกันงานซ้ำ | 100 | verified | local/VPS file lock + atomic replace + concurrent `ssh://` authority integration |
| WTL-P2-I3 | `open` | สร้าง task/branch/worktree ตามมาตรฐาน | 100 | verified | dry-run no mutation + apply + idempotency + limit tests |
| WTL-P2-I4 | `list/status/enter/doctor` | อ่านของจริงและรายงานภาษาคน | 100 | verified | branch drift/doctor/parser tests |
| WTL-P2-I5 | `handoff` | โอน Notebook↔VPS โดยไม่เกิดสอง writer | 100 | verified | handoff→accept test ย้าย lease ครั้งเดียวและเก็บ location เดิม |
| WTL-P2-I6 | `pause/close` | ปลด process/permit และบันทึกสถานะ | 100 | verified | pause/close tests ปลด lease |
| WTL-P2-I7 | `cleanup --dry-run` | แสดงรายการและเหตุผลโดยไม่ลบ | 100 | verified | cleanup proposed 6/6 และ Worktree ยังอยู่ |
| WTL-P2-I8 | `cleanup --apply` | ลบผ่าน Git เฉพาะ 6/6 + quarantine | 100 | verified | active ถูกบล็อก; apply แรก quarantine; พ้นเวลาจึง git remove |
| WTL-P2-I9 | runtime allocator | พอร์ต/ชื่อ service/container/db ไม่ชน | 100 | verified | two-task runtime test พอร์ตและ DB ต่างกัน |
| WTL-P2-I10 | รองรับเครื่องมือเดิม | newchat/route/write-permit ใช้ร่วมได้ | 100 | verified | newchat delegate + route registry test + permit cannot bypass WTL lease |

## WTL-P3 — เชื่อม Shortcut และกฎกลาง

### กลุ่มแก้พฤติกรรมหลัก

| Issue | Shortcut | เกณฑ์ผ่าน | % | สถานะ | Evidence |
|---|---|---|---:|---|---|
| WTL-P3-I1 | Use New Chat | เปิด/เลือก task worktree ผ่าน Manager | 100 | verified | direct WTL section + removed fixed-folder conflict |
| WTL-P3-I2 | Use Flow Guardian | ตรวจ task/machine/writer/runtime ก่อนเขียน | 100 | verified | direct WTL section |
| WTL-P3-I3 | Use AI Relay | coder cwd ตรง task worktree; reviewer read-only | 100 | verified | direct WTL section + fixed-workspace rule replaced |
| WTL-P3-I4 | Use Continue | กลับ worktree เดิมจาก task id และเดินโซน A ต่อเอง | 100 | verified | direct WTL section + 2-zone phase approval policy |
| WTL-P3-I5 | Use Close Chat | ปิดสิทธิ์และประเมิน cleanup 6/6 | 100 | verified | direct WTL section |
| WTL-P3-I6 | Review Chat | รายงาน worktree/machine/branch/cleanup | 100 | verified | direct WTL section |
| WTL-P3-I7 | Use Save Git | ตรวจทะเบียน/branch/push/permit ก่อนส่ง | 100 | verified | direct WTL section |
| WTL-P3-I8 | Use Merge to Production | เปลี่ยนเป็น CLEANUP_READY หลัง merge | 100 | verified | direct WTL section |
| WTL-P3-I9 | Use Move Folder | worktree root เป็น no-touch; ส่งให้ Manager | 100 | verified | direct WTL section |
| WTL-P3-I10 | Use AI Pair | ใช้ Manager เดียวกับ Relay | 100 | verified | direct WTL section |

### กลุ่มเพิ่มจุดเชื่อม

| Issue | Shortcut | เกณฑ์ผ่าน | % | สถานะ | Evidence |
|---|---|---|---:|---|---|
| WTL-P3-I11 | Use Act-As | แผนงานเขียนประกาศ task/worktree | 100 | verified | direct WTL section |
| WTL-P3-I12 | Use Comply | issue ผูก task/worktree/evidence และแยกโซน A/B | 100 | verified | direct WTL section + zone column |
| WTL-P3-I13 | Use OverviewProgress | เก็บ machine/worktree/lifecycle | 100 | verified | direct WTL section |
| WTL-P3-I14 | Use QA QC | scan อ่านอย่างเดียว; fix ใช้ worktree แยก | 100 | verified | direct WTL section |
| WTL-P3-I15 | Use SonarQube | ผูกผล scan กับ path/SHA/task | 100 | verified | direct WTL section |
| WTL-P3-I16 | Use Hermes Structure | ติดตั้งสัญญา WTL กลาง | 100 | verified | direct WTL section |
| WTL-P3-I17 | Use Viber Structure | ใส่ lifecycle ในโครงโครงการ | 100 | verified | direct WTL section |
| WTL-P3-I18 | Use Viber Audit | ตรวจความสอดคล้อง WTL | 100 | verified | direct WTL section |

### กลุ่มด่านกลาง

| Issue | งาน | เกณฑ์ผ่าน | % | สถานะ | Evidence |
|---|---|---|---:|---|---|
| WTL-P3-I19 | prompt-shortcuts SKILL | Shortcut ที่จะเขียนต้องผ่าน WTL gate | 100 | verified | central WTL gate + 30/30 visibility checker |
| WTL-P3-I20 | registry + ชุดแจกทีม | แหล่งกลางและสำเนาใช้รุ่นเดียวกัน | 100 | verified | 21 parity files SHA-256 match; visibility tests 2/2 |

## WTL-P4 — สำรวจและกำหนดทางจัดการ Worktree เดิม

| Issue | งาน | เกณฑ์ผ่าน | % | สถานะ | Evidence |
|---|---|---|---:|---|---|
| WTL-P4-I1 | ตัวสแกนแบบอ่านอย่างเดียว | ไม่ลบและไม่เปลี่ยน branch | 100 | verified | scan test + Notebook/VPS read-only commands |
| WTL-P4-I2 | จำแนก managed/unknown/broken | ทุก Worktree มีสถานะ | 100 | verified | inventory: Notebook 4 unknown/0 broken; VPS 4 unknown/0 broken |
| WTL-P4-I3 | กำหนดทางจัดการ Notebook เดิม | มีคำตัดสินเจ้าของโดยไม่เดา owner/task id | 100 | verified | B1 approved 2026-07-15 · เก็บ 4 paths เป็น legacy inventory · ไม่ import/ย้าย/ลบ |
| WTL-P4-I4 | กำหนดทางจัดการ VPS เดิม | มีคำตัดสินเจ้าของโดยไม่เดา route/task id | 100 | verified | B1 approved 2026-07-15 · เก็บ 4 paths เป็น legacy inventory · ไม่ import/ย้าย/ลบ |
| WTL-P4-I5 | วัดพื้นที่ฐาน | แยก code/dependency/cache/build | 100 | verified | `.project/wtl-migration-inventory.md` มี 4 buckets ครบ 8 paths |
| WTL-P4-I6 | รายการ cleanup candidate | ไม่มีการลบ; เจ้าของเห็นเหตุผล | 100 | verified | รายงาน 3 clean-review paths แต่ 0 CLEANUP_READY; ไม่มีการลบ |

## WTL-P5 — ทดสอบเหตุการณ์จริง

| Issue | เหตุการณ์ | เกณฑ์ผ่าน | % | สถานะ | Evidence |
|---|---|---|---:|---|---|
| WTL-P5-I1 | สองงานโครงการเดียวกัน | branch/path ไม่กระทบกัน | 100 | verified | isolation test |
| WTL-P5-I2 | สองคนคนละงาน | registry/permit ไม่ชน | 100 | verified | two-staff test |
| WTL-P5-I3 | task เดียวสองเครื่อง | เครื่องที่สองถูกบล็อก | 100 | verified | duplicate machine/path + accept-before-handoff + VPS authority lock tests |
| WTL-P5-I4 | handoff Notebook→VPS | writer เปลี่ยนครั้งเดียว | 100 | verified | handoff/accept test |
| WTL-P5-I5 | offline Notebook | ทำต่อได้เฉพาะสิทธิ์เดิม | 100 | verified | `status --offline` cache test: machine+unexpired lease+Git must match; transfer blocked |
| WTL-P5-I6 | dirty/unpushed | close/cleanup ถูกบล็อก | 100 | verified | dirty handoff/close test |
| WTL-P5-I7 | merged branch | เข้า CLEANUP_READY ไม่ลบทันที | 100 | verified | close merged + HEAD/state-bound dry-run + quarantine + second dry-run test |
| WTL-P5-I8 | port/container/db ชน | allocator ปฏิเสธหรือจัดค่าใหม่ | 100 | verified | unique runtime test |
| WTL-P5-I9 | พื้นที่ 70/85/90 | เตือน/หยุด/กู้พื้นที่ตรงกฎ | 100 | verified | threshold test 69.9/70/85/90 + real Notebook 96% block |
| WTL-P5-I10 | โปรแกรมหยุดกลาง open/cleanup | กลับมาตรวจต่อได้ ไม่ค้างครึ่งเดียว | 100 | verified | simulated open interruption leaves BLOCKED record/no lease |
| WTL-P5-I11 | พยายามลบงานที่ใช้อยู่ | ถูกปฏิเสธพร้อมเหตุผล | 100 | verified | active cleanup blocked test |
| WTL-P5-I12 | Shortcut visibility 30/30 | ตัวตรวจเห็นกฎ WTL ทุกเส้นทางเขียน | 100 | verified | checker 30/30, direct 18/18, bad fixture blocked |

## WTL-P6 — ทดลองใช้ เปิดใช้ และ PDCA

| Issue | งาน | เกณฑ์ผ่าน | % | สถานะ | Evidence |
|---|---|---|---:|---|---|
| WTL-P6-I1 | Pilot Hermes Agent | เดิน open→close→cleanup dry-run | 100 | verified | temp Git pilot เดิน full lifecycle; ไม่แตะลูกค้าจริง |
| WTL-P6-I2 | Pilot โครงการไม่ใช้งานจริงลูกค้า | เหตุการณ์หลักผ่าน | 100 | verified | isolated bare-origin tests 22/22 |
| WTL-P6-I3 | ตรวจ Notebook จริง | path/registry/disk ถูกต้อง | 100 | verified | 4 worktrees, 96% disk, bucket baseline; no mutation |
| WTL-P6-I4 | ตรวจ VPS จริง | route/service/runtime namespace ถูกต้อง | 100 | verified | SSH scan 4 worktrees/78% + concurrent remote registry lock/atomic write |
| WTL-P6-I5 | ตรวจเบาทุก 24 ชั่วโมง | มี scheduler/report และทดสอบเรียกได้ | 100 | verified | Notebook job `63639d7adcfd` + launchd 86400 วินาที exit 0; VPS job `754c9588d645` + systemd timer active/enabled และรันจริง ok |
| WTL-P6-I6 | เสนอ cleanup ทุก 168 ชั่วโมง | dry-run report ไม่ลบเอง | 100 | verified | Notebook job `5594d40ca72d` + launchd 604800 วินาที exit 0; VPS job `0948314bfe5e` รันจริง ok; คำสั่งมี `--cleanup-review` แต่ไม่มี `cleanup --apply` |
| WTL-P6-I7 | Dashboard/รายงาน PDCA | เห็นจำนวน พื้นที่ unknown/block/ready | 100 | verified | `hermes worktree report` + cadence/state/project/bytes/blocked/candidates test |
| WTL-P6-I8 | Closeout 71/71 | ทุก Issue 100% + หลักฐาน + owner review | 100 | verified | เจ้าของอนุมัติแผนและสั่ง Use Continue จนจบทุก Phase; code PR #39/#40 และ vault MR #2 merged; Notebook/VPS ติดตั้งและรันจริง 4/4 งานผ่าน |

## งานถัดไป

1. งานตามแผน WTL ปิดครบ 71/71 = 100% แล้ว
2. รอบถัดไปให้ปล่อยตัวตั้งเวลาทำงานตามปกติและดูรายงาน PDCA; Worktree เดิม 8 รายการยังเป็น legacy inventory และยังไม่ถูก import/ย้าย/ลบ
3. หากจะนำ Worktree เดิมเข้าทะเบียน ต้องมีเจ้าของ task และรหัสงานจริงก่อน ห้ามเดาข้อมูลย้อนหลัง

### โซน B — แผนอนุมัติรวม

| ลำดับ | Issue เดิม | สิ่งที่ต้องอนุมัติ | ผลกระทบ | วิธีย้อนกลับ |
|---|---|---|---|---|
| B1 | WTL-P4-I3/I4 | อนุมัติแล้ว 2026-07-15 · เก็บ Worktree เดิม 8 รายการเป็น legacy inventory โดยไม่สร้าง task id/owner ปลอม และไม่นำเข้าทะเบียนจนกว่าจะมีเจ้าของมารับรอง | ไม่มีการย้ายหรือลบ; ปิดความเสี่ยงเดาข้อมูล | เปลี่ยนสถานะจาก inventory เป็น import candidate ภายหลังได้ |
| B2 | WTL-P6-I8 | อนุมัติแล้ว 2026-07-15 · commit สองคลัง → push branch → เปิดคำขอรวมงาน → เจ้าของตรวจและกด merge | ส่งโค้ด/กติกาเข้าสู่แหล่งกลาง แต่ยังไม่เปิด scheduler | ยังไม่ merge สามารถปิดคำขอรวมงาน; หลัง merge ใช้ commit ย้อนกลับ |
| B3 | WTL-P6-I5/I6 | อนุมัติจากคำสั่ง Use Continue 2026-07-15 และดำเนินการแล้ว · ติดตั้งรุ่นที่ merge แล้ว เปิดตารางตรวจ 24 ชั่วโมง/เสนอ cleanup 168 ชั่วโมงบน Notebook และ VPS | เริ่มรายงานอัตโนมัติ; cleanup ยังเป็น dry-run และไม่ลบเอง | ปิด launchd jobs / systemd timer และคงข้อมูลทะเบียนเดิม |

ด่านลำดับผ่านแล้ว: B1 → B2 merge/install → B3 activation; ตรวจ `hermes cron run/list/status` จริงทั้ง Notebook/VPS และใช้ launchd/systemd timer แยกเพื่อไม่เปิด Gateway ที่อาจกระทบช่องทางข้อความ

## บันทึกการเปลี่ยนแปลง

| เวลา | Issue | การเปลี่ยนแปลง | หลักฐาน |
|---|---|---|---|
| 2026-07-14 | WTL-P0-I1/I2/I4/I5 | เปิดพื้นที่แยก บันทึกอนุมัติ และจำแนก Shortcut 30/30 | git worktree + ข้อความเจ้าของ + registry scan |
| 2026-07-14 | WTL-P0-I3/I6 | ไฟล์ติดตามเข้า Gitและเก็บค่าฐานการทดสอบ | git check-ignore/ls-files · 3 environment probes |
| 2026-07-14 | WTL-P1-I1..I9 | สร้างสัญญากลางและตัวตรวจ | contract 16 headings/15 phrases/30 shortcuts · unittest 4/4 |
| 2026-07-14 | WTL-P2-I1..I10 | สร้าง Manager + compatibility bridge | Manager unittest 18/18 ก่อนเพิ่ม offline test |
| 2026-07-14 | WTL-P3-I1..I20 | อัปเดต 18 Shortcut + central gate + payload | visibility 30/30 · direct 18/18 · parity 21 files |
| 2026-07-14 | WTL-P4-I1/I2/I5/I6 | สำรวจ Notebook/VPS โดยไม่แก้ Git | 8 paths · 0 broken · 0 cleanup-ready · inventory report |
| 2026-07-14 | WTL-P5-I1..I12 | ทดสอบเหตุการณ์หลัก | tests ครอบคลุม isolation/writer/handoff/offline/cleanup/runtime/disk/recovery |
| 2026-07-14 | WTL-P6-I1..I7 | pilot, Notebook/VPS baseline, remote authority, PDCA | remote lock concurrent pass · scheduler entrypoint tests pass; activation deploy-gated |
| 2026-07-14 | Closeout รอบก่อน owner review | ตรวจ syntax/diff/contract/shortcut/tests | Python 3.12 compile pass · tests 30/30 · contract 16/16 headings, 15/15 phrases, 30/30 shortcuts · direct 18/18 · parity 21 files · diff check pass |
| 2026-07-14 | WTL-P3-I4/I12 | เพิ่มกติกาเดินงาน 2 โซนให้ Continue/Comply/Contract | โซน A ทำต่อเอง · โซน B ขออนุมัติระดับ Phase ครั้งเดียว |
| 2026-07-14 | Closeout โซน A | ตรวจ syntax/contract/shortcut/tests หลังเพิ่ม 2 โซน | Python 3.12 compile pass · tests 32/32 · contract 16/16 headings, 17/17 phrases, 30/30 shortcuts · parity ผ่าน |
| 2026-07-15 | WTL-P4-I3/I4 · B1 | เจ้าของอนุมัติให้เก็บ 8 paths เป็น legacy inventory โดยไม่เดาข้อมูลและไม่ import/ย้าย/ลบ | ข้อความ “อนุมัติ Zone B B1+B2” + `.project/wtl-migration-inventory.md` |
| 2026-07-15 | WTL-P6-I5/I6/I8 · B2/B3 | รวมโค้ดและคลังคำสั่งกลาง ติดตั้งคำสั่ง Worktree เปิดรอบตรวจ Notebook/VPS และรันจริง | GitHub PR #39/#40 merged · GitLab MR #2 merged · cron jobs 4/4 last run ok · launchd 2/2 exit 0 · VPS systemd timer enabled+active · central PDCA timestamps recorded |

## ข้อจำกัดการตรวจทั้ง Repository

- ชุด targeted WTL ใช้ bundled Python 3.12.13 และผ่าน 32/32
- ยังรัน `pytest` ทั้ง Hermes repository ไม่ได้ใน worktree นี้ เพราะไม่มี `.venv`/`venv`; Python ระบบ 3.9 เก่าเกินโค้ด และ bundled Python ไม่มี PyYAML/pytest stack ของโครงการ
- ข้อนี้ไม่ถูกนับเป็น test failure แต่ก่อน merge ควรรัน CI หรือ `scripts/run_tests.sh` ใน environment มาตรฐานของ Hermes
- `hermes cron status` รุ่นปัจจุบันตรวจเฉพาะ Gateway จึงยังพิมพ์คำเตือนแม้ launchd/systemd timer ทำงานอยู่; หลักฐานอัตโนมัติให้อ่านจากสถานะตัวตั้งเวลาของระบบและประวัติ `Last run: ok` ของงานทั้ง 4 รายการ
- Vault หลักบน Notebook ตามหลัง `origin/main` 2 commit และมีไฟล์ค้างของแชทอื่น 5 รายการ จึงไม่ดึงทับ; แหล่งกลางบน GitLab รวม WTL แล้วที่ `faadbd626375a0d039ca611c2a44450907ff7ed3` และตัวติดตั้ง Worktree บน Notebook ไม่ขึ้นกับการดึง Vault รอบนี้
