# memory-audit

สคริปต์ตรวจว่าไฟล์ความจำใน `.project/` **อ้างอะไร** ตรงกับ **git จริง** หรือไม่ — กันเคสความจำโกหก (เช่น อ้าง SHA ที่ถูก revert แล้ว หรือไฟล์ถูก `.gitignore` ซ่อน)

## วิธีรัน

จากรากโปรเจกต์:

```bash
python3 scripts/memory-audit/memory_audit.py
```

ระบุ repo อื่น หรือออก JSON:

```bash
python3 scripts/memory-audit/memory_audit.py --repo /path/to/repo
python3 scripts/memory-audit/memory_audit.py --json
```

## สิ่งที่ตรวจ (4 ด่าน)

1. **ป้าย schema** — บรรทัดแรกของ `OverviewProgress.md` มี `> memory-schema:` และ `plan.md` มี `plan_id`
2. **SHA ในความจำ** — ค่า hex ใน backticks ต้องมี commit จริง และยังไม่ถูก revert
3. **ไฟล์ใน git** — ไฟล์ใต้ `.project/` ต้องไม่ถูก ignore และไฟล์หลัก 3 ชิ้นถูก track
4. **เลขงาน relay** — `issue_id` ในสมุด `.hermes/ai-relay/calls-*.md` ต้องขึ้นต้นด้วย `plan_id` ที่รู้จัก (หรือ `jarvis`)

## รหัสออก (exit code)

| ค่า | ความหมาย |
|-----|----------|
| 0 | ผ่านทุกด่าน |
| 1 | พบ fail (ความจำไม่ตรง git / ไฟล์หลุด) |
| 2 | มีแต่เตือน (warn) |
| 3 | ข้อผิดพลาด (ไม่ใช่ git repo / ไม่มี git) |

## เทสต์

```bash
python3 -m pytest scripts/memory-audit/tests/ -q
```