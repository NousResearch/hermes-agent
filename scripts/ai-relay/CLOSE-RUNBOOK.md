# AI Relay Close Runbook v2.7

อัปเดต: 2026-07-06

เอกสารนี้ใช้ปิดงานหลังถอด Fable/Faber/Fiber 5 ออกจาก AI Relay แล้ว

## สถานะที่ต้องเป็น

- สมองหลักมี Opus 4.8 ตัวเดียว
- coder/reviewer ที่เปิดให้ใช้: `codex`, `grok`, `gemini`, `ollama`
- ห้ามเรียก `relay-call --tool fable`
- ไม่มีบัญชีตัวอย่างหรือไฟล์เชื่อมบริบทให้ Fable/Faber/Fiber 5
- รายงานต้องนับ Fable เป็นเครื่องมือเก่าที่ถูกถอด ไม่ใช่เครื่องมือที่ใช้งานได้

## คำสั่งตรวจหลัก

จากโฟลเดอร์ repo Hermes Agent:

```bash
python3 -m pytest scripts/ai-relay/tests/test_relay_fixes.py
bash scripts/ai-relay/relay-setup.sh
relay-doctor
gate-run --task-id CLOSE-GATE --cwd "$(pwd)"
relay-report --project "$(pwd)"
```

## ตรวจเครื่องมือที่เรียกได้จริง

```bash
relay-call --tool opus --task-id CLOSE-OPUS --prompt-file "ตอบคำว่า OK เท่านั้น" --cwd "$(pwd)"
relay-call --tool grok --task-id CLOSE-GROK --prompt-file "ตอบคำว่า OK เท่านั้น" --cwd "$(pwd)"
relay-call --tool codex --task-id CLOSE-CODEX --prompt-file "ตอบคำว่า OK เท่านั้น" --cwd "$(pwd)"
```

ถ้า Opus ติดโควต้า ให้ถือว่าเครื่องมือทำงานถึงชั้นเรียกจริงแล้วเมื่อ `relay-call` คืนสถานะ `quota` และมีข้อความ reset time ชัดเจน

## หลักฐานว่าถอด Fable แล้ว

```bash
relay-report --project "$(pwd)"
python3 - <<'PY'
import importlib.util
from pathlib import Path

path = Path("scripts/ai-relay/relay-call.py")
spec = importlib.util.spec_from_file_location("relay_call", path)
relay_call = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relay_call)
print("fable" in relay_call.DEFAULT_ADAPTERS)
PY
```

ผลที่ถูกต้อง:

- `relay-report` แสดง Fable/Faber/Fiber อยู่ในกลุ่มเครื่องมือเก่าที่ถูกถอด
- Python ต้องพิมพ์ `False`

## ถ้าพบคำสั่งเก่า

ถ้าเอกสาร งานค้าง หรือสมุดบันทึกผลยังมีคำสั่ง `relay-call --tool fable` ให้ถือว่าเป็นขั้นตอนเก่า ห้ามรัน ให้แก้เป็น Opus 4.8 หรือส่งงานให้ coder ที่ยังเปิดใช้อยู่แทน
