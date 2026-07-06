# แผนปิดงาน Shortcut / AI Relay ให้จบชัด · 2026-07-06

## เป้าหมาย

ทำให้พนักงานใช้ Prompt Shortcut และ AI Relay ได้จาก GitHub ชุดเดียวกัน พร้อมแยกงานค้างที่ไม่เกี่ยวออกจากงานนี้ให้ชัดเจน

## สถานะที่ตรวจแล้ว

| จุดตรวจ | ผล |
|---|---:|
| GitHub ของเจ้าของ (`rattanasak-ops/hermes-agent`) | มีไฟล์แผนนี้แล้ว · พนักงานติดตั้งผ่าน GitHub โดยไม่ต้องมี repo |
| local repo เทียบ GitHub | 0 ahead / 0 behind |
| จำลองเครื่องพนักงานจาก GitHub สด | ผ่าน |
| Shortcut ในเครื่องพนักงานจำลอง | 28/28 |
| `SKILL.md` ในเครื่องพนักงานจำลอง | 28/28 |
| `Prompt Shortcuts.md` ในเครื่องพนักงานจำลอง | 28/28 |
| prompt ใน `references/` | 32 ไฟล์ |
| local vault เทียบ payload | ต่าง 0 |
| local Codex เทียบ vault | ต่าง 0 |
| VPS vault เทียบ payload | ต่าง 0 |
| VPS Codex เทียบ vault | ต่าง 0 |
| AI Relay test local | 45/45 |
| AI Relay test VPS | 45/45 |
| local `relay-doctor` | ผ่าน 11 · เตือน 0 · ไม่ผ่าน 0 |
| VPS `relay-doctor` | ผ่าน 10 · เตือน 1 · ไม่ผ่าน 0 |
| `Use Opus Plan` / `opus-plan` | ไม่พบ |
| Fable/Faber/Fiber 5 ในตัวเชื่อมมาตรฐาน | 0 |

## คำตัดสิน

งานที่พนักงานต้องใช้ตอนนี้พร้อมแจกแล้ว 100% จากหลักฐาน GitHub, การติดตั้งจำลอง, local, และ VPS

คำสั่งพนักงาน:

```bash
curl -fsSL https://raw.githubusercontent.com/rattanasak-ops/hermes-agent/main/team-shortcuts/install-from-github.sh | bash
```

ถ้าใช้ Cursor:

```bash
curl -fsSL https://raw.githubusercontent.com/rattanasak-ops/hermes-agent/main/team-shortcuts/install-from-github.sh | bash -s -- --cursor
```

ถ้าใช้ AI Relay:

```bash
curl -fsSL https://raw.githubusercontent.com/rattanasak-ops/hermes-agent/main/scripts/ai-relay/relay-setup.sh | bash
relay-doctor
```

## เรื่องค้างที่ยังไม่ใช่ปัญหาการใช้งานพนักงาน

### 1. Local repo มีไฟล์ค้าง 6 รายการ

ไฟล์ค้าง:

```text
.gitignore
.project/decisions.md
.project/plan.md
.project/FeatureSpec-jarvis-voice.md
scripts/jarvis-voice/
tests/scripts/test_jarvis_turn_taking.py
```

ผลกระทบ:

- ไม่กระทบ Shortcut/AI Relay ที่ส่งขึ้น GitHub แล้ว
- ถ้าจะปิดความสะอาดของ local repo ต้องแยกเป็นงาน Jarvis Voice อีก 1 รอบ

เกณฑ์จบ:

- ตัดสินไฟล์ 6 รายการว่าเป็นงานจริงหรือของทดลอง
- ถ้าเป็นงานจริง ให้ทำ branch/commit แยก
- ถ้าเป็นของทดลอง ให้เก็บออกนอก repo หรือทำเป็นไฟล์บันทึกชัดเจน

### 2. VPS repo หลักที่ใช้ตอนนี้มีไฟล์ค้าง 62 รายการ

Path ที่ใช้จริง:

```text
/home/linux-nat/SynerryTools/hermes-agent/main
```

สถานะที่ตรวจพบ:

```text
branch: main
dirty files: 62
remote origin: https://github.com/NousResearch/hermes-agent.git
remote fork: git@github.com:rattanasak-ops/hermes-agent.git
Shortcut release: ใช้ `main` ของ `rattanasak-ops/hermes-agent`
```

ผลกระทบ:

- ห้าม `git pull` ตรงจาก path นี้ เพราะมีไฟล์ค้างและประวัติแยกมาก
- ไม่กระทบการใช้งาน Shortcut ตอนนี้ เพราะไฟล์ `scripts/ai-relay` และ `team-shortcuts` ถูกคัดให้ตรงกับ local แล้ว

เกณฑ์จบ:

- สำรองสถานะ VPS ปัจจุบันเป็น branch หรือ patch ก่อน
- แยกไฟล์ค้าง 62 รายการเป็น 2 กลุ่ม: เก็บจริง / เลิกใช้
- ตั้งชื่อ remote ให้ชัดว่า repo เจ้าของคือ `fork` หรือเปลี่ยนเอกสารให้ใช้ชื่อเดียวกันทุกที่
- หลังจัดกลุ่มแล้วจึงค่อยดึง GitHub ของเจ้าของเข้ามาแบบไม่ทับงานค้าง

### 3. Worktree route ของ Hermes Agent บน VPS ไม่ตรง memory

คำสั่งตรวจ:

```bash
python3 scripts/hermes_worktree_route.py --staff-id nat --project hermes-agent
```

ผล:

```text
ok=false
matches=0
searched_roots=/home/linux-nat/projects, /srv/projects, /home/linux-nat/.worktree
```

ข้อเท็จจริงที่พบ:

```text
/home/linux-nat/projects/hermes-agent/main ไม่ใช่ git worktree
/home/linux-nat/projects/hermes-agent/nat ไม่ใช่ git worktree
/home/linux-nat/SynerryTools/hermes-agent/main เป็น path ที่ใช้จริงในรอบนี้
```

ผลกระทบ:

- AI รอบใหม่อาจหา Hermes Agent worktree ของ VPS ไม่เจอ
- อาจทำให้รายงาน path ผิดหรือไปทำงานผิดจุด

เกณฑ์จบ:

- เลือก source of truth ให้เหลือ 1 path สำหรับ Hermes Agent บน VPS
- อัปเดต route script หรือ route data ให้ `nat + hermes-agent` เจอ path จริง
- อัปเดต Obsidian memory หลังเจ้าของเห็นด้วยกับ path จริง

## แผนปิดงานที่แนะนำ

### เฟส 1 · ปิดงานแจก Shortcut ให้ทีม

สถานะ: เสร็จแล้ว 100%

หลักฐาน:

- GitHub: มีไฟล์แผนนี้แล้ว · พนักงานติดตั้งผ่าน GitHub โดยไม่ต้องมี repo
- จำลองเครื่องพนักงานที่ไม่มี repo Hermes Agent: Shortcut 28/28
- local/VPS: vault, payload, Codex ตรงกัน
- test local/VPS: 45/45

สิ่งที่ต้องทำต่อ:

- แจ้งพนักงานใช้คำสั่ง `curl ... install-from-github.sh | bash` ด้านบน

### เฟส 2 · ปิดความสะอาด local repo

สถานะ: ยังไม่เริ่ม 0/6 ไฟล์

งาน:

1. ตรวจไฟล์ค้าง 6 รายการ
2. แยกงาน Jarvis Voice ออกจากงาน Shortcut
3. ถ้าเป็นงานจริง ให้ทำ commit แยก
4. ถ้าไม่ใช้แล้ว ให้เอาออกจาก repo แบบมีหลักฐาน

ผลที่ได้:

- local repo กลับมา clean 100%
- งาน Shortcut ไม่ปนกับงาน Jarvis Voice

### เฟส 3 · ปิดความสะอาด VPS repo

สถานะ: ยังไม่เริ่ม 0/62 ไฟล์

งาน:

1. สำรองสถานะปัจจุบันของ VPS repo
2. จัดกลุ่มไฟล์ dirty 62 รายการ
3. เก็บเฉพาะงานจริงเข้า branch แยก
4. ทำให้ `team-shortcuts` และ `scripts/ai-relay` อ้างอิง GitHub ของเจ้าของได้ชัดเจน
5. ห้ามดึงจาก upstream ตรงจนกว่าจะสำรองเสร็จ

ผลที่ได้:

- VPS repo ไม่เสี่ยงทับงานค้าง
- รอบถัดไปอัปเดตจาก GitHub เจ้าของได้ชัดเจน

### เฟส 4 · ปิด route memory ของ Hermes Agent

สถานะ: ยังไม่เริ่ม 0/1 route

งาน:

1. ยืนยัน path จริงของ Hermes Agent บน VPS
2. แก้ route ให้ `nat + hermes-agent` เจอ path จริง
3. ตรวจซ้ำให้ `ok=true`
4. อัปเดต memory หลังตรวจผ่าน

ผลที่ได้:

- AI รอบใหม่ไม่หลง path
- งาน VPS รอบต่อไปเริ่มถูกที่ตั้งแต่ต้น

## คำสั่งตรวจซ้ำก่อนปิดทั้งหมด

```bash
git status --branch --short
git ls-remote origin refs/heads/main
scripts/run_tests.sh scripts/ai-relay/tests/test_relay_fixes.py
relay-doctor
```

บน VPS:

```bash
ssh linux-nat@103.142.150.185 'cd /home/linux-nat/SynerryTools/hermes-agent/main && scripts/run_tests.sh scripts/ai-relay/tests/test_relay_fixes.py'
ssh linux-nat@103.142.150.185 'cd /home/linux-nat/SynerryTools/hermes-agent/main && relay-doctor'
```

Shortcut:

```bash
rg -ni 'Use Opus Plan|use-opus-plan|Opus Plan|opus-plan' team-shortcuts/payload/skills/prompt-shortcuts
```

## เกณฑ์ปิดงานสุดท้าย

งานนี้ถือว่าจบทั้งระบบเมื่อครบ 4 ข้อ:

1. พนักงานติดตั้งจาก GitHub แล้วได้ Shortcut 28/28
2. local repo clean หรือมี branch แยกสำหรับงานที่ค้าง
3. VPS repo ไม่ dirty หรือมี branch/patch สำรองครบ
4. route `nat + hermes-agent` ตรวจแล้วได้ `ok=true`
