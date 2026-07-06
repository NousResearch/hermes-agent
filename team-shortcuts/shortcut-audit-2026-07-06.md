# รายงานตรวจ Prompt Shortcut ทั้งชุด · 2026-07-06

## ผลหลังแก้และตรวจซ้ำ

เจ้าของอนุมัติรอบแก้แล้ว และได้ปรับชุด Shortcut ให้ตรงกันทั้งเครื่องเจ้าของ, ชุดติดตั้งทีม, Codex runtime และ VPS runtime แล้ว

ตัวเลขหลังแก้:

| จุดตรวจ | เครื่องเจ้าของ | VPS |
|---|---:|---:|
| Shortcut ในทะเบียนกลาง | 28 | 28 |
| Shortcut ใน `SKILL.md` ที่ Codex โหลดตรง | 28 | 28 |
| Shortcut ในหน้า `Prompt Shortcuts.md` | 28 | 28 |
| prompt `.md` ใน `references/` | 32 | 32 |
| ไฟล์รวมใน skill folder | 37 | 37 |
| ไฟล์ prompt ที่ทะเบียนกลางอ้างแล้วหาไม่เจอ | 0 | 0 |
| hash payload เทียบ vault | ต่าง 0 | ต่าง 0 |
| hash Codex runtime เทียบ vault | ต่าง 0 | ต่าง 0 |

ผลตรวจเฉพาะจุด:

- `Use Opus Plan`, `use-opus-plan`, `Opus Plan`, `opus-plan`: ไม่พบแล้วทั้ง local และ VPS
- Fable/Faber/Fiber 5: ถอดจากทางใช้งานแล้วทั้ง local และ VPS
- ตัวเชื่อมมาตรฐานของ AI Relay เหลือ `codex`, `gemini`, `grok`, `ollama`, `opus`
- `fable` ไม่อยู่ใน `DEFAULT_ADAPTERS` ทั้ง local และ VPS
- `relay-call --tool fable` เหลือเฉพาะข้อความเตือนว่า “ห้ามเรียก” ใน runbook ใหม่
- คำสั่ง `relay-call`, `gate-run`, `relay-doctor`, `relay-report` พบจริงทั้งสองเครื่อง
- test AI Relay ผ่าน local 45/45 และ VPS 45/45
- `relay-doctor` local: ผ่าน 11 · เตือน 0 · ไม่ผ่าน 0
- `relay-doctor` VPS: ผ่าน 10 · เตือน 1 · ไม่ผ่าน 0 โดยเตือนเฉพาะ Hermes xAI ยังไม่ได้ login; Grok CLI และ AI Relay พร้อมใช้

ไฟล์ที่ปรับในรอบแก้:

- `SKILL.md`: เพิ่มแผนที่ Shortcut ให้ครบ 28 ตัว
- `Prompt Shortcuts.md`: เพิ่มแผนที่ Shortcut ให้ครบ 28 ตัว และลบ alias เก่า `Use Opus Plan`
- `agents/openai.yaml`: เพิ่มคำแนะนำ Shortcut ชุดล่าสุด
- `team-shortcuts/README.md`: แก้ให้ VPS ต้องรันตัวติดตั้งหลังอัปเดต payload
- `team-shortcuts/install-shortcuts.sh`: ใช้ `rsync --delete` เพื่อคัดไฟล์ให้ตรงชุดติดตั้งและล้างไฟล์เก่าค้าง
- `team-shortcuts/sync-from-vault.sh`: ใช้ `rsync --delete` เพื่อให้ payload ตรงกับ vault
- `scripts/ai-relay/CLOSE-RUNBOOK.md`: เขียนใหม่เป็น v2.7 และห้ามเรียก Fable

สถานะตอนนี้: local และ VPS ใช้ Shortcut ชุดเดียวกันแล้ว 100% จากหลักฐานไฟล์และ test ข้างบน

ความเสี่ยงที่ยังเหลือ: คำสั่งแจกทีมต้องไม่สมมติว่าพนักงานมี repo Hermes Agent ในเครื่อง

## สรุปสำหรับรีวิว

ส่วนนี้คือผลตรวจรอบแรกก่อนแก้ เก็บไว้เป็นหลักฐานว่าพบปัญหาอะไร

ผลตรวจรอบนี้พบว่า Shortcut ส่วนใหญ่มีไฟล์ prompt อยู่จริงในเครื่องเจ้าของและใน payload สำหรับทีม แต่ยังทำงาน "ไม่เหมือนกันทุกเครื่อง" เพราะแผนที่บางชั้นไม่ทันทะเบียนกลาง และ VPS ยังโหลดไฟล์บางตัวจาก Obsidian ชุดเก่า

ตัวเลขตรวจจริง:

| จุดตรวจ | ผล |
|---|---:|
| Shortcut ในทะเบียนกลาง | 28 ตัว |
| Shortcut ใน `SKILL.md` ที่ Codex โหลดตรง | 19 ตัว |
| Shortcut ในหน้า `Prompt Shortcuts.md` | 18 ตัว |
| ไฟล์ prompt ที่ทะเบียนกลางอ้าง แล้วมีครบบนเครื่องเจ้าของ | 28/28 |
| ไฟล์ prompt ที่ทะเบียนกลางอ้าง แล้วมีครบใน payload ทีม | 28/28 |
| ไฟล์ prompt ที่ทะเบียนกลางอ้าง แล้วมีครบใน Codex runtime เครื่องเจ้าของ | 28/28 |
| ไฟล์ prompt ที่ทะเบียนกลางอ้าง แต่หายใน Codex runtime ของ VPS | 3/28 |

คำตัดสิน: ตอนนี้พนักงานที่ติดตั้งจาก payload ล่าสุดบนเครื่องตัวเองมีโอกาสใช้ได้ครบกว่า VPS แต่ Codex ยังเสี่ยงไม่จับ Shortcut ใหม่บางตัว เพราะ `SKILL.md` ยังมีแผนที่ไม่ครบ

## ขอบเขตที่ตรวจ

ตรวจแบบอ่านอย่างเดียวจากไฟล์จริง:

- เครื่องเจ้าของ: `/Users/rattanasak/ObsidianVault/HermesAgent/skills/prompt-shortcuts`
- ชุดติดตั้งทีม: `team-shortcuts/payload/skills/prompt-shortcuts`
- Codex runtime เครื่องเจ้าของ: `/Users/rattanasak/.codex/skills/prompt-shortcuts`
- VPS Obsidian: `/home/linux-nat/ObsidianVault/HermesAgent/skills/prompt-shortcuts`
- VPS Codex runtime: `/home/linux-nat/.codex/skills/prompt-shortcuts`
- VPS repo payload: `/home/linux-nat/SynerryTools/hermes-agent/main/team-shortcuts/payload`

ไม่ได้รัน workflow จริงของทุก Shortcut เพราะหลายตัวต้องแตะ repo งานจริง, สร้างไฟล์, หรือผ่าน gate ของโปรเจกต์จริงก่อน จึงตรวจระดับ "เรียก Shortcut แล้วหา prompt ถูกไฟล์หรือไม่" เป็นหลัก

## ปัญหาที่พบ

### P1 · `SKILL.md` ไม่ทันทะเบียนกลาง

ผลตรวจ:

`SKILL.md` มีแผนที่ Shortcut 19 ตัว แต่ทะเบียนกลางมี 28 ตัว

Shortcut ที่มีในทะเบียนกลาง แต่ไม่มีใน `SKILL.md`:

| Shortcut | ไฟล์ prompt |
|---|---|
| Use Close Chat | `references/use-close-chat.md` |
| Use Merge to Production | `references/use-merge-to-production.md` |
| Use BusinessPlan | `references/use-businessplan.md` |
| Use OverviewProgress | `references/use-overviewprogress.md` |
| Use FeatureSpec | `references/use-featurespec.md` |
| Use DesignSystem | `references/use-designsystem.md` |
| Use Create Design System | `references/use-create-design-system.md` |
| Use Hermes Structure | `references/use-hermes-structure.md` |
| Use Create Content | `references/use-create-content.md` |

ผลกระทบ:

- พนักงานพิมพ์ Shortcut เหล่านี้แล้ว AI บางตัวอาจไม่รู้ว่าต้องโหลดไฟล์ไหน
- Codex อาจไม่เรียก skill อัตโนมัติ เพราะ description ใน `SKILL.md` ยังไม่ครอบคลุมชื่อใหม่ทุกตัว
- งานปิดแชท, Project OS, คอนเทนต์, ดีไซน์ทั้งระบบ เสี่ยงถูกทำจากความจำแทนไฟล์ prompt จริง

สาเหตุ:

เพิ่ม Shortcut ใหม่เข้า `prompt-shortcut-registry.md` และ payload แล้ว แต่ไม่ได้อัปเดต `SKILL.md` ให้เท่ากัน

วิธีแก้ที่แนะนำ:

ทำให้ `SKILL.md` สร้างจากทะเบียนกลาง หรืออัปเดตด้วยมือให้มี 28 ตัวครบ พร้อมเพิ่มพฤติกรรมสำคัญของ 9 ตัวที่หาย

### P1 · VPS runtime หาย 3 Shortcut

ผลตรวจ:

VPS payload มีไฟล์อยู่ แต่ Obsidian/Codex runtime ของ VPS ไม่มี 3 ไฟล์นี้:

| Shortcut | สถานะบน VPS runtime |
|---|---|
| Use OverviewProgress | หาย |
| Use FeatureSpec | หาย |
| Use DesignSystem | หาย |

ผลกระทบ:

- บน VPS เรียก 3 ตัวนี้แล้วหาไฟล์ prompt ไม่เจอ
- `Use New Chat` และ `Use Close Chat` ที่อ้าง Project OS อาจอ่าน/เขียน memory ไม่ครบตาม Schema v1.2
- พนักงานที่ใช้ VPS จะได้ผลไม่เหมือนเครื่องเจ้าของ

สาเหตุ:

VPS ใช้ symlink จาก Codex ไปที่ Obsidian vault โดยตรง แต่ vault บน VPS ยังไม่ได้รับไฟล์ Project OS 3 ตัว ถึง payload ใน repo จะมีแล้วก็ตาม

วิธีแก้ที่แนะนำ:

หลังคุณอนุมัติ ให้รันตัวติดตั้ง Shortcut บน VPS จาก repo payload เพื่อคัด payload เข้า Obsidian runtime ของ VPS แล้วตรวจซ้ำ

คำสั่งแก้:

```bash
cd /home/linux-nat/SynerryTools/hermes-agent/main/team-shortcuts
bash install-shortcuts.sh
```

### P1 · VPS มี prompt หลายตัวคนละรุ่นกับ payload

ผลตรวจ:

VPS Obsidian กับ VPS payload มี hash ไม่ตรงกัน 16 ตัว เช่น `Use Act-As`, `Use Comply`, `Use Summary`, `Use New Chat`, `Use Close Chat`, `Use Continue`, `Review Chat`, `Use BusinessPlan`, `Use Create Content`

ผลกระทบ:

- ชื่อ Shortcut เดียวกัน แต่ AI บน VPS อาจอ่านเนื้อคนละรุ่นกับพนักงานที่ติดตั้งจาก payload
- งานข้ามเครื่องมีโอกาสสรุปกติกาไม่เหมือนกัน
- บาง Shortcut ที่ผูก Memory Schema v1.2 อาจเขียนไฟล์คนละกติกา

สาเหตุ:

VPS vault ถูกแก้/ซิงก์แบบบางไฟล์มาก่อน และ README เดิมบอกว่า VPS "ไม่ต้องติดตั้ง" ทำให้ไม่มีรอบคัด payload เข้า runtime อย่างครบชุด

วิธีแก้ที่แนะนำ:

ถือ payload เป็นชุดแจกจ่าย แล้วติดตั้งทับเข้า VPS runtime ทุกครั้งหลัง sync จาก vault

### P2 · หน้า `Prompt Shortcuts.md` เก่ากว่าทะเบียนกลาง

ผลตรวจ:

หน้า index มีแผนที่ 18 ตัว แต่ทะเบียนกลางมี 28 ตัว และยังมี `Use Opus Plan` ในคำสั่งตัวอย่าง ทั้งที่ไม่มี prompt file และไม่มีแถวในทะเบียนกลาง

ผลกระทบ:

- คนอ่านเอกสารจะคิดว่าบาง Shortcut ไม่มี ทั้งที่มีในทะเบียนกลาง
- `Use Opus Plan` เป็นชื่อหลอน: เรียกแล้วไม่มีไฟล์ชัดเจนให้เปิด
- พนักงานอาจใช้เอกสารเก่าแทนทะเบียนกลาง

สาเหตุ:

หน้า index ไม่ได้ถูกอัปเดตตามทะเบียนกลางหลังเพิ่ม Shortcut ชุด Project OS, Close Chat, Merge to Production, Create Content, Hermes Structure

วิธีแก้ที่แนะนำ:

สร้างหน้า `Prompt Shortcuts.md` ใหม่จากทะเบียนกลาง และลบหรือแมป `Use Opus Plan` ให้ชัดเจน

### P2 · `openai.yaml` แนะนำ Shortcut ไม่ครบ

ผลตรวจ:

`agents/openai.yaml` ยังแนะนำเฉพาะ Shortcut หลักบางตัว ไม่รวม `Use Close Chat`, `Use Merge to Production`, `Use BusinessPlan`, `Use OverviewProgress`, `Use FeatureSpec`, `Use DesignSystem`, `Use Create Design System`, `Use Hermes Structure`, `Use Create Content`

ผลกระทบ:

- หน้าแนะนำใน adapter (ไฟล์เชื่อมบริบทให้ AI อ่าน) อาจไม่เสนอ Shortcut ที่ควรใช้
- พนักงานอาจไม่รู้ว่ามี Shortcut ใหม่

สาเหตุ:

เพิ่ม Shortcut ใหม่แล้วไม่ได้ปรับ default prompt ของ agent

วิธีแก้ที่แนะนำ:

อัปเดต `openai.yaml` ให้แนะนำจากทะเบียนกลางชุดเดียวกัน

### P2 · ชุดติดตั้งทีมมีข้อความเก่าเรื่องจำนวนไฟล์

ผลตรวจ:

`install-shortcuts.sh` comment ยังบอกว่า "prompt 22 ไฟล์" แต่ payload ตอนนี้มีไฟล์อ้างอิง 32 ไฟล์ และ Shortcut หลัก 28 ตัว

ผลกระทบ:

- ไม่ทำให้ติดตั้งพัง เพราะตัว script นับไฟล์จริงตอนรัน
- แต่ทำให้คนดู code เข้าใจผิดว่าชุดแจกจ่ายยังเก่า

สาเหตุ:

comment ไม่ได้อัปเดตตามจำนวนไฟล์จริง

วิธีแก้ที่แนะนำ:

เปลี่ยนข้อความเป็น "คัดชุด Shortcut ทั้งหมดจาก payload" ไม่ฝังตัวเลข

### P2 · README บอกว่า VPS ไม่ต้องติดตั้ง แต่ผลจริงไม่ครบ

ผลตรวจ:

`team-shortcuts/README.md` ระบุว่าบัญชี `linux-nat` บน VPS ไม่ต้องติดตั้ง เพราะมีชุด Shortcut ครบแล้ว แต่ผลตรวจจริงพบว่าหาย 3 ตัวและหลายไฟล์คนละรุ่นกับ payload

ผลกระทบ:

- พนักงานบน VPS อาจไม่รัน install แล้วใช้ชุดเก่า
- ทีมจะคิดว่า VPS เป็นแหล่งที่พร้อม ทั้งที่ยังต่างจาก payload

สาเหตุ:

README เป็นสมมติฐานเก่า ก่อนมี Project OS recovery และก่อนมีการซิงก์ payload หลายรอบ

วิธีแก้ที่แนะนำ:

แก้ README ให้บอกว่า VPS ต้องรัน `bash install-shortcuts.sh` หลังเจ้าของอัปเดต payload ทุกครั้ง

### P2 · ยังไม่ได้ push ชุดล่าสุดไป GitHub

ผลตรวจ:

repo เครื่องเจ้าของยังมีไฟล์ทีมและ AI Relay ที่แก้แล้วแต่ยังไม่ commit/push

ผลกระทบ:

- พนักงานต้องใช้ตัวติดตั้งจาก GitHub โดยตรง เพราะเครื่องพนักงานไม่มี repo Hermes Agent
- เครื่องเจ้าของและ VPS ที่ผมคัดไฟล์ตรงไว้จะใหม่กว่า GitHub

สาเหตุ:

รอบก่อนเป็นการปรับเครื่องเจ้าของและ VPS โดยตรง ยังไม่ได้ทำรอบส่งขึ้น remote

วิธีแก้ที่แนะนำ:

หลังคุณรีวิวรายงานนี้และอนุมัติรอบแก้ ให้ stage เฉพาะไฟล์ Shortcut/AI Relay ที่เกี่ยวข้อง แล้ว commit/push

## Shortcut ที่ดูพร้อมใช้จากไฟล์ในเครื่องเจ้าของ

กลุ่มนี้มี prompt file ครบทั้ง vault, payload, และ Codex runtime เครื่องเจ้าของ:

- Use Act-As
- Use Comply
- Use Summary
- Use Scan Feature
- Use AI Relay
- Use Viber Structure
- Use Viber Audit
- Use Impeccable
- Use Blog Auto
- Use WOW Resource
- Use Flow Guardian
- Use New Chat
- Use Close Chat
- Use Save Git
- Use Merge to Production
- Use Continue
- Use Move Folder
- Review Chat
- Use AI Pair
- Use Business Plan
- Use SaaS Opus Master Prompt
- Use BusinessPlan
- Use OverviewProgress
- Use FeatureSpec
- Use DesignSystem
- Use Create Design System
- Use Hermes Structure
- Use Create Content

หมายเหตุ: "พร้อมใช้จากไฟล์" แปลว่า AI หา prompt ได้ ไม่ได้แปลว่าทุก Shortcut ผ่าน workflow จริง เพราะบางตัวต้องมี repo งานจริง, git gate, CI, VPS route, หรืออนุมัติจากเจ้าของก่อน

## สิ่งที่แต่ละกลุ่มต้องทำ

### เจ้าของระบบ

1. อนุมัติรอบแก้ Shortcut registry alignment 1 รอบ
2. ให้ AI อัปเดต `SKILL.md`, `Prompt Shortcuts.md`, `agents/openai.yaml`, README และ comment ตัวติดตั้งให้ตรงทะเบียนกลาง 28 ตัว
3. ให้ AI คัด payload เข้า VPS runtime ด้วย `team-shortcuts/install-shortcuts.sh`
4. ให้ AI รัน audit ซ้ำให้ได้:
   - registry 28/28
   - `SKILL.md` 28/28
   - `Prompt Shortcuts.md` 28/28
   - local runtime 28/28
   - VPS runtime 28/28
   - hash vault/payload/runtime ตรงกันในไฟล์ prompt หลัก
5. หลังผลตรวจผ่าน ให้ commit/push เพื่อให้พนักงานดึงจาก GitHub ได้

### พนักงานที่ใช้เครื่องตัวเอง

หลังเจ้าของ push ชุดใหม่แล้ว ให้พนักงานรันจากเครื่องตัวเองได้เลย โดยไม่ต้องมี repo Hermes Agent:

```bash
curl -fsSL https://raw.githubusercontent.com/rattanasak-ops/hermes-agent/main/team-shortcuts/install-from-github.sh | bash
```

ถ้าใช้ Cursor เพิ่ม:

```bash
curl -fsSL https://raw.githubusercontent.com/rattanasak-ops/hermes-agent/main/team-shortcuts/install-from-github.sh | bash -s -- --cursor
```

จากนั้นปิดแล้วเปิดโปรแกรม AI ใหม่ 1 รอบ แล้วลอง:

```text
Use Comply
Use Close Chat
Use OverviewProgress
```

ผลที่ควรได้:

- AI ต้องอ่านไฟล์ prompt จาก `~/ObsidianVault/HermesAgent/skills/prompt-shortcuts/references/...`
- ถ้า AI บอกว่าไม่รู้จัก Shortcut หรือหาไฟล์ไม่เจอ ให้หยุดแล้วแจ้งเจ้าของพร้อมชื่อไฟล์ที่หาย

### พนักงานที่ใช้ AI Relay

ทำเพิ่มเฉพาะเครื่องที่จะให้ AI ตัวอื่นเขียนโค้ด:

```bash
curl -fsSL https://raw.githubusercontent.com/rattanasak-ops/hermes-agent/main/scripts/ai-relay/relay-setup.sh | bash
grok login --oauth
relay-doctor
```

ผลที่ควรได้:

- `relay-call` ต้องพบ
- `gate-run` ต้องพบ
- Grok login ต้องผ่าน
- ถ้า Opus ชน quota ให้รายงานเป็น `status=quota` ไม่ใช่ `crash`

### พนักงานที่ใช้ VPS

หลังเจ้าของคัด payload เข้า VPS runtime แล้ว พนักงานไม่ต้องแก้ไฟล์ Shortcut กลางเอง ให้ใช้งานจาก:

```text
/home/linux-nat/ObsidianVault/HermesAgent/skills/prompt-shortcuts
```

ก่อนทำงานจริงบน VPS ต้องเช็ค worktree route ตามโปรเจกต์และ staff id เสมอ ห้ามใช้ worktree ของคนอื่นแทน

### AI ทุกตัว

เมื่อเจอ Shortcut:

1. อ่านทะเบียนกลางก่อน
2. เปิดไฟล์ prompt ที่แมปไว้เต็มไฟล์
3. ถ้าไฟล์หาย ให้หยุดและรายงานชื่อไฟล์ ไม่เดาจากความจำ
4. ถ้า Shortcut ต้องเขียนไฟล์หรือแตะ production ให้ขออนุมัติก่อนตาม prompt นั้น

## แผนแก้ที่แนะนำหลังรีวิว

ทางที่ดีที่สุดคือทำ "รอบจัดแผนที่ Shortcut ให้ตรงทะเบียนกลาง" เพียงรอบเดียว:

1. ปรับ `SKILL.md` ให้มี 28 ตัวครบ
2. ปรับ `Prompt Shortcuts.md` ให้มี 28 ตัวครบ และลบ/แมป `Use Opus Plan`
3. ปรับ `agents/openai.yaml` ให้แนะนำ Shortcut ชุดล่าสุด
4. ปรับ README ให้ VPS ต้องติดตั้งจาก payload หลังอัปเดต
5. ซิงก์ payload จาก vault
6. รัน install บนเครื่องเจ้าของและ VPS
7. รัน audit ซ้ำ
8. commit/push

คุณค่าของทางนี้:

- ลดปัญหา AI เดาจากความจำ เพราะทุกตัวอ่านไฟล์เดียวกัน
- ลดปัญหาเครื่องเจ้าของ, เครื่องพนักงาน, VPS ได้ผลไม่เหมือนกัน
- ลดเวลาตามแก้รายเครื่อง เพราะพนักงานใช้คำสั่งติดตั้งชุดเดียว

## คำสั่งตรวจที่ใช้

ตรวจจำนวนแถวและไฟล์ในเครื่องเจ้าของ:

```bash
python3 <ตัวตรวจ registry/skill/payload/codex>
```

ตรวจ VPS:

```bash
ssh linux-nat@103.142.150.185 'python3 <ตัวตรวจ registry/skill/payload/codex>'
```

ตรวจ symlink Codex:

```bash
ls -l ~/.codex/skills/prompt-shortcuts
readlink ~/.codex/skills/prompt-shortcuts
```

ตรวจเอกสารทีม:

```bash
sed -n '1,120p' team-shortcuts/README.md
sed -n '1,180p' team-shortcuts/install-shortcuts.sh
```

## สถานะปิดรายงาน

รายงานนี้อัปเดตหลังรอบแก้แล้ว ปัญหา P1/P2 เรื่องแผนที่ Shortcut, VPS runtime, index, `openai.yaml`, README, และไฟล์ค้างบน VPS ถูกแก้และตรวจซ้ำแล้ว

สถานะรวม: 100% พร้อมใช้งานเท่ากันบน local และ VPS ตามหลักฐานไฟล์/test · 0 ไฟล์หาย · 0 hash ต่าง · ยังเหลือเฉพาะขั้น commit/push เพื่อแจกให้พนักงานผ่าน GitHub
