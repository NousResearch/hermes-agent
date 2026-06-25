# ชุดติดตั้ง Prompt Shortcut สำหรับทีม

ชุดนี้ทำให้พนักงานใช้ Shortcut (เช่น `Use Comply`, `Use Continue`, `Review Chat`) ได้บนเครื่องตัวเอง
ทั้งใน **Cursor, Claude Code และ Codex** เหมือนเครื่องเจ้าของระบบ

## ปัญหาที่ชุดนี้แก้

ไฟล์ตัวเชื่อมในแต่ละโปรเจกต์ชี้ไปที่อยู่เต็มของเครื่องเจ้าของระบบ
(`/Users/rattanasak/ObsidianVault/HermesAgent/...`) ซึ่งไม่มีบนเครื่องพนักงาน
พอ AI หาเนื้อ Shortcut ไม่เจอ จึงใช้ไม่ได้ ตัวติดตั้งนี้วางชุด Shortcut ลงเครื่องพนักงาน
แล้วต่อสายให้ทั้ง 3 โปรแกรมมองเห็น

## วิธีติดตั้ง (พนักงานทำครั้งเดียวต่อเครื่อง)

```bash
# 1. ดึงรีโปนี้ (หรือ git pull ถ้ามีอยู่แล้ว)
# 2. เข้าโฟลเดอร์นี้ แล้วรัน:
cd team-shortcuts
bash install-shortcuts.sh           # ต่อ Claude Code + Codex (ไม่ต้องใช้สิทธิ์ผู้ดูแล)
bash install-shortcuts.sh --cursor  # ถ้าใช้ Cursor ด้วย (อาจขอรหัสผู้ดูแล 1 ครั้ง)
```

เสร็จแล้วปิด-เปิดโปรแกรม AI ใหม่ 1 รอบ แล้วลองพิมพ์ `Use Comply` ดู

## ตัวติดตั้งทำอะไรบ้าง

| ขั้น | ทำอะไร | ต้องใช้สิทธิ์ผู้ดูแลไหม |
|---|---|:---:|
| 1 | คัดชุด Shortcut ไป `~/ObsidianVault/HermesAgent/` | ไม่ |
| 2 | ต่อ Claude Code ผ่าน `~/.claude/CLAUDE.md` | ไม่ |
| 3 | ต่อ Codex ผ่าน `~/.codex/skills/prompt-shortcuts` | ไม่ |
| 4 | ต่อ Cursor ผ่านทางลัดชดเชยที่อยู่เดิม (เฉพาะ `--cursor`) | อาจขอ 1 ครั้ง |

ตัวติดตั้งรันซ้ำได้ ไม่พัง (เขียนทับของเดิม)

## สำหรับพนักงานที่ทำงานบน VPS (บัญชี linux-nat)

ไม่ต้องติดตั้ง — VPS มีชุด Shortcut ครบที่ `/home/linux-nat/ObsidianVault/HermesAgent` อยู่แล้ว

## สำหรับเจ้าของระบบ (อัปเดต Shortcut ใหม่)

```bash
bash sync-from-vault.sh   # ดึงของล่าสุดจาก vault มาใส่ payload
git add team-shortcuts/payload && git commit -m "sync shortcuts" && git push
```

แล้วบอกพนักงาน `git pull` + รัน `install-shortcuts.sh` อีกครั้ง
