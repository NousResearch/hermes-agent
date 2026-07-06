# Jarvis Voice

ระบบนี้คือ “ปาก” ของจาร์วิสสำหรับ Claude Code: ให้ AI ใส่ marker (ป้ายข้อความที่ hook อ่านได้) แบบนี้ไว้ท้ายคำตอบ

```html
<!-- speak: ข้อความที่ต้องการให้พูด -->
```

เมื่อ Claude Code เรียก Stop hook (สคริปต์ที่ทำงานตอนคำตอบจบ) ไฟล์ `stop-hook.py` จะอ่านคำตอบล่าสุดจาก transcript (บันทึกบทสนทนา), ดึงข้อความใน marker แล้วเรียก `speak.sh` เพื่อพูดออกลำโพงเป็นเสียงไทย

## ทดสอบด้วยมือ

```bash
scripts/jarvis-voice/speak.sh "ทดสอบ"
```

## ตัวแปรที่ตั้งค่าได้

- `JARVIS_VOICE` = เลือกเสียงพูด ค่าเริ่มต้นคือ `th-TH-NiwatNeural`
- `JARVIS_RATE` = ความเร็วเสียง ค่าเริ่มต้นคือ `+10%`
- `JARVIS_VOICE_DISABLED=1` = ปิดเสียงทั้งหมดแบบเงียบ ๆ

## ตัวอย่าง `.claude/settings.json`

ใส่ config (ค่าตั้งค่า) นี้ในไฟล์ `.claude/settings.json` ของโปรเจกต์ โดย command (คำสั่งที่ hook เรียก) ต้องชี้ไปที่ `stop-hook.py` แบบ absolute path (พาธเต็มจากรากเครื่อง):

```json
{
  "hooks": {
    "Stop": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "python3 \"/Users/rattanasak/Documents/Viber Project/Tech Tools/Hermes Agent/scripts/jarvis-voice/stop-hook.py\""
          }
        ]
      }
    ]
  }
}
```

## จาร์วิสเสียงสด

ไฟล์ชุดนี้มีตัวใช้งานเสียงสด 2 แบบ:

```bash
scripts/jarvis-voice/.venv-gemini/bin/python scripts/jarvis-voice/jarvis_live.py
```

คำสั่งนี้เปิดโหมดคุยเสียงกับ Gemini Live โดยตรง เหมาะกับการทดสอบไมค์/ลำโพงแบบสั้น ๆ

```bash
scripts/jarvis-voice/.venv-gemini/bin/python scripts/jarvis-voice/jarvis_app.py
```

คำสั่งนี้เปิดเปลือกใช้งานจริง มีปุ่มลัด `Cmd+Shift+J`, ปุ่มข้างเมาส์สำหรับเปิด/ปิดพูด, กันเสียงย้อน, และปิดไมค์เองเมื่อเงียบนาน

## เปิดเองตอนเข้าเครื่อง

`jarvis-start.command` ใช้ restart งาน launchd ที่เครื่องเจ้าของตั้งไว้แล้ว:

```bash
open scripts/jarvis-voice/jarvis-start.command
```

ไฟล์ `~/Library/LaunchAgents/com.rattanasak.jarvis.plist` เป็นไฟล์ประจำเครื่อง ไม่ได้เก็บใน git เพราะมีพาธและค่าตั้งค่าของเครื่องนั้น ๆ

## ไฟล์ลับและค่าตั้งค่า

- `scripts/jarvis-voice/.env` ต้องอยู่ในเครื่องเท่านั้น และต้องไม่เข้า git
- `GEMINI_API_KEY` อยู่ใน `.env`
- `JARVIS_MODEL` = เปลี่ยนรุ่น Gemini Live
- `JARVIS_HOTKEY` = ปุ่มลัด ค่าเริ่มต้น `<cmd>+<shift>+j`
- `JARVIS_START_ACTIVE=1` = เปิดไมค์ทันทีเมื่อโปรแกรมเริ่ม
- `JARVIS_IN_DEVICE` / `JARVIS_OUT_DEVICE` = ชี้ไมค์/ลำโพงจริงเมื่อเครื่องมีอุปกรณ์เสียงเสมือน
- `JARVIS_MOUSE_TOGGLE=0` = ปิดปุ่มข้างเมาส์
- `JARVIS_MANUAL_ACTIVITY=0` = กลับไปใช้การจับจังหวะพูดจาก Gemini เป็นหลัก
