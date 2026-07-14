> WTL migration inventory · 2026-07-14 Asia/Bangkok · read-only evidence
> owner decision: 2026-07-15 · B1 approved · เก็บทั้ง 8 รายการเป็นบัญชีประวัติเท่านั้น ไม่สร้าง task id/owner แทนคน และยังไม่ import

# รายงานสำรวจ Worktree เดิม — Notebook และ VPS

รายงานนี้เป็นผลจาก `git worktree list --porcelain`, `git status --porcelain`, `git rev-list`, `df`, และตัววัดพื้นที่ WTL เท่านั้น ไม่มีการลบ ย้าย เปลี่ยน branch หรือ import เข้าทะเบียนกลาง

## สรุปสำหรับตัดสินใจ

| เครื่อง | Worktree | managed | unknown | broken | พื้นที่เครื่อง | การตัดสินใจ |
|---|---:|---:|---:|---:|---:|---|
| Notebook | 4 | 0 | 4 | 0 | 96% ใช้แล้ว · เหลือ 38 GiB | `WTL_BLOCKED` สำหรับการสร้างใหม่; สำรวจ/พัฒนาโค้ดในพื้นที่เดิมได้ |
| VPS `linux-nat` | 4 | 0 | 4 | 0 | 78% ใช้แล้ว · เหลือ 213 GiB | ระดับ warning 70%; วางแผน cleanup แต่ห้ามลบอัตโนมัติ |

คำว่า `unknown` แปลว่า Git เห็น Worktree จริง แต่ยังไม่มี `task_id/staff_id/machine_id` ที่ยืนยันในสมุดทะเบียน WTL จึงห้ามเดาและห้าม import อัตโนมัติ

## คำตัดสิน B1 ของเจ้าของ

- ทั้ง 8 รายการคงอยู่ตำแหน่งเดิม ไม่มีการย้าย ลบ หรือเปลี่ยน branch
- เก็บข้อมูลในไฟล์นี้เป็น `legacy inventory` (บัญชีประวัติของพื้นที่เดิม) เท่านั้น
- ห้ามสร้าง `task_id`, `staff_id`, `machine_id` หรือ owner จากการคาดเดา
- ห้ามนำเข้าทะเบียน WTL จนกว่าเจ้าของของ Worktree นั้นจะให้ข้อมูลยืนยัน
- รายการสะอาด 3 รายการยังเป็นเพียง `clean-review` ไม่ใช่ `CLEANUP_READY`

หลักฐานอนุมัติ: ข้อความเจ้าของ 2026-07-15 “อนุมัติ Zone B B1+B2”

## Notebook baseline

| Path | Branch | Dirty | Unpushed | Code | Dependency | Cache | Build | Total | สถานะย้ายทะเบียน |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `/Users/rattanasak/Documents/Viber Project/Tech Tools/Hermes Agent` | `control_webengine_flow` | 2 | 0 | 300,627,581 | 975,604,334 | 35,987,445 | 4,011,292 | 1,316,230,652 | active/dirty · ห้าม cleanup |
| `/Users/rattanasak/Documents/Viber Project/Tech Tools/HermesAgent-wt-stdi2` | `feature/std-i2-project-dir` | 0 | 0 | 99,922,657 | 81,211,332 | 9,499,017 | 261,822 | 190,894,828 | ต้องยืนยัน owner+task id ก่อน import |
| `/Users/rattanasak/Documents/Viber Project/Tech Tools/HermesAgent-wt-wtl` | `codex/wtl-worktree-lifecycle` | 33 | 0 | 102,366,322 | 0 | 808,142 | 261,822 | 103,436,286 | งาน WTL ปัจจุบัน · ห้าม cleanup |
| `/Users/rattanasak/hermes-0170-upgrade` | `upgrade-audit/v0170` | 0 | 0 | 143,813,652 | 321,409,756 | 18,180,129 | 4,134,195 | 487,537,732 | ต้องยืนยัน owner+task id ก่อน import |

## VPS baseline

| Path | Branch | Dirty | Unpushed | Code | Dependency | Cache | Build | Total | สถานะย้ายทะเบียน |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `/home/linux-nat/SynerryTools/hermes-agent/main` | `main` | 48 | 0 | 170,483,390 | 581,685,721 | 13,623,639 | 3,414,824 | 769,207,574 | canonical/dirty · ห้าม cleanup |
| `/home/linux-nat/SynerryTools/hermes-agent/ai-pair-use-ai-pair` | `ai-pair/use-ai-pair` | 17 | 0 | 96,450,681 | 0 | 8,034,079 | 261,006 | 104,745,766 | dirty · ห้าม cleanup |
| `/home/linux-nat/SynerryTools/hermes-agent/nat` | `nat` | 17 | 0 | 111,387,866 | 503,739,374 | 14,990,076 | 3,260,886 | 633,378,202 | dirty · ห้าม cleanup |
| `/tmp/hermes-relay-portal-fix` | `fix/ai-relay-portal` | 0 | 0 | 101,453,072 | 0 | 115,662 | 261,822 | 101,830,556 | ต้องยืนยัน owner+task id และ `/tmp` retention ก่อน import |

## Cleanup candidates (รายงานเท่านั้น)

ยังไม่มีรายการใดเป็น `CLEANUP_READY` เพราะทุก path ขาดสถานะ merged/owner-abandoned, recovery evidence, dry-run record และ quarantine ตาม gate 6/6

รายการที่ “สะอาดและไม่มี commit ค้างส่ง” แต่ยังห้ามลบ:

1. Notebook `HermesAgent-wt-stdi2`
2. Notebook `hermes-0170-upgrade`
3. VPS `/tmp/hermes-relay-portal-fix`

ถ้าจะนำสามรายการนี้กลับมาใช้ในอนาคต ต้องยืนยัน `project_id + task_id + staff_id + machine_id + owner` ก่อนใช้ `hermes worktree import`; หลัง import จะเริ่มที่ `PAUSED` และไม่มี writer lease

## หลักฐานคำสั่ง

- Notebook: `git worktree list --porcelain`, status/unpushed ต่อ path, `df -h .`, `disk_usage()` ของ WTL
- VPS: SSH แบบ read-only ไป `linux-nat@103.142.150.185`, `git worktree list --porcelain`, status/unpushed ต่อ path, `df -h`, ตัววัด bucket แบบเดียวกับ WTL
- route เก่า `staff=nat + project=hermes-agent` ไม่พบ match ในสาม registered roots จึงไม่ fallback ไป path อื่น

## ความเสี่ยงที่เหลือ

- Notebook อยู่ระดับ recovery 90%+; ห้ามสร้าง worktree/dependency/build เพิ่มจนพื้นที่ต่ำกว่า 85%
- VPS อยู่ระดับ warning 70%+; ควรตรวจทุก 24 ชั่วโมงและเสนอรายงาน cleanup ทุก 168 ชั่วโมง
- Worktree ทั้ง 8 ถูกล็อกเป็นบัญชีประวัติตาม B1 แล้ว และจะไม่ถูกนำเข้าทะเบียนด้วยการเดา
