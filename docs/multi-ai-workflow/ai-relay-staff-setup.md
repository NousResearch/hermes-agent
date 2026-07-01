# AI Relay Staff Setup

เอกสารนี้คือขั้นตอนให้พนักงานใช้ `Use AI Relay` กับ Grok ผ่าน Google ID โดยไม่ให้ AI เห็นรหัสลับของพนักงาน

## หลักการ

- พนักงาน login Grok เองในเครื่องของตัวเอง
- AI Relay เรียก `grok` ผ่านคำสั่งในเครื่อง ไม่รับรหัสผ่านจากแชท
- `accounts.yaml` เก็บแค่ป้ายชื่อบัญชี ไม่เก็บรหัสจริง
- `gate-run` เป็นตัวตรวจงานจริง เช่น test หรือ build แล้วจดผลลงสมุดงาน

## เพิ่ม Grok เข้า AI Relay ที่ติดตั้งไว้แล้ว

ถ้าเครื่องมี AI Relay อยู่แล้ว ให้รันคำสั่งนี้จาก root ของ repo:

```bash
relay-add-grok --cwd .
```

คำสั่งนี้จะเพิ่ม `grok` เข้า `.hermes/ai-relay/adapters.yaml` และจัดให้ `grok` อยู่ตัวแรกในสายเขียนโค้ดของ `.hermes/ai-relay/accounts.yaml`

คำสั่งนี้ไม่ login แทนพนักงาน ไม่อ่านรหัส และไม่เขียนรหัสลับลงไฟล์

## ติดตั้งในเครื่องที่ยังไม่มี AI Relay

ให้เปิด Terminal ที่ root ของ repo นี้ แล้วรัน:

```bash
bash scripts/ai-relay/install-local.sh
```

ถ้าเครื่องบอกว่าไม่รู้จัก `relay-doctor` หลังติดตั้ง ให้รัน:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

จากนั้นเช็ก:

```bash
relay-doctor
```

## Login Grok ด้วย Google ID

ให้พนักงานรัน:

```bash
grok login --oauth
```

สิ่งที่จะเกิด:

1. Grok เปิดหน้า login ในเบราว์เซอร์
2. พนักงานกด `Continue with Google`
3. เลือก Google ID ของตัวเอง
4. อนุญาตให้ Grok CLI ใช้งานบัญชี
5. กลับมาที่ Terminal

ตรวจว่า Grok ใช้ได้:

```bash
grok models
```

ผลที่ถือว่าผ่าน:

- ต้องไม่ขึ้น `You are not authenticated`
- ต้องเห็นรายชื่อ model ของ Grok อย่างน้อย 1 ตัว

## Login Hermes xAI

ถ้าต้องการให้ Hermes Agent ใช้ Grok ผ่าน xAI ด้วย ให้รัน:

```bash
hermes auth add xai-oauth
```

สิ่งที่จะเกิดเหมือนกัน:

1. หน้า xAI เปิดในเบราว์เซอร์
2. พนักงานกด `Continue with Google`
3. เลือก Google ID ของตัวเอง
4. กลับมาที่ Terminal

ตรวจ:

```bash
hermes auth status xai-oauth
```

ผลที่ถือว่าผ่าน:

- ต้องไม่ขึ้น `logged out`

## ตรวจความพร้อมก่อนให้ AI ทำงาน

รัน:

```bash
relay-doctor
```

ผลที่ต้องเห็นก่อนใช้ Grok เป็นคนเขียนโค้ด:

- `grok` ต้องพบโปรแกรม
- Grok login ต้องผ่าน
- `relay-call` ต้องพบ
- `gate-run` ต้องพบ
- มีไฟล์ `.hermes/ai-relay/adapters.yaml`
- มีไฟล์ `.hermes/ai-relay/accounts.yaml`

## ใช้งานจริง

สร้าง brief เป็นไฟล์ เช่น:

```bash
mkdir -p .hermes/ai-relay/briefs
printf 'แก้ typo ใน README.md แล้วห้ามแตะไฟล์อื่น\n' > .hermes/ai-relay/briefs/P1-I1.md
```

เรียก Grok:

```bash
relay-call --tool grok --task-id P1-I1 --prompt-file .hermes/ai-relay/briefs/P1-I1.md --cwd .
```

ตรวจงาน:

```bash
gate-run --cwd . --task-id P1-I1
```

ดูสถานะ:

```bash
relay-status
```

## ถ้า Login ไม่ผ่าน

ให้ทำตามลำดับนี้:

1. รัน `grok logout`
2. รัน `grok login --oauth`
3. เลือก `Continue with Google`
4. รัน `grok models`
5. รัน `relay-doctor`

ถ้ายังขึ้น `You are not authenticated` แปลว่ายังไม่ใช่ปัญหา AI Relay แต่เป็น Grok CLI ยังไม่จำบัญชี ต้องให้เจ้าของเครื่อง login ใหม่เอง

## ขอบเขตความพร้อม

หลังติดตั้งแพ็กนี้:

- คำสั่ง Relay พร้อมใช้ในเครื่อง
- Grok พร้อมใช้เมื่อพนักงาน login สำเร็จ
- การตรวจงานมี `gate-run` เป็นหลักฐานจริง

สิ่งที่ AI ยังทำแทนคนไม่ได้:

- กด Google login
- กรอก 2FA
- อนุมัติสิทธิ์ในเบราว์เซอร์
- แก้ปัญหาแพ็กเกจบัญชี xAI ที่จำกัดสิทธิ์จากฝั่งบริการ
