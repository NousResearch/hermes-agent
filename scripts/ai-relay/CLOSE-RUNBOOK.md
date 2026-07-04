# AI Relay VPS Close Runbook

เอกสารนี้ใช้ปิดงาน AI Relay บน VPS (เครื่องเซิร์ฟเวอร์) `myserver` เท่านั้น ไม่แก้ repo (คลังโค้ด) และไม่แตะไฟล์อื่นนอกจากไฟล์ตั้งค่าที่ระบุในขั้นตอนงานคน

ข้อมูลตั้งต้นที่ตรวจแล้ว:
- Hermes Agent `main` และ `origin` อยู่ที่ commit `11db5b37c`
- VPS repo อยู่ที่ `/home/linux-nat/SynerryTools/hermes-agent/main` และ `HEAD` เป็น `11db5b37c`
- `relay-call` และ `gate-run` อยู่ที่ `~/.local/bin` และเป็น symlink ชี้ repo ถูก
- `relay-call` มีเส้นทางสลับจาก Fable ไป Opus แล้ว
- ปัญหาค้างคือ `~/.hermes/.env` มี `CLAUDE_CODE_OAUTH_TOKEN` ของ org ที่ปิดสิทธิ์ Claude Code ทำให้ Fable/Opus เสีย
- ผลตรวจจริงล่าสุด: VPS Grok ล็อกอินแล้ว และใช้ได้จริง
- ผลตรวจจริงล่าสุด: VPS Gemini ยังพัง เพราะ process (โปรแกรมที่รันอยู่) ไม่มี `GEMINI_API_KEY` ตอนรัน
- ห้ามสรุปกลับกันว่า Gemini ใช้ได้แล้วหรือ Grok ยังไม่ได้ล็อกอิน จนกว่าจะมีผลตรวจใหม่กว่านี้

## 1. เกณฑ์ปิดงาน (Definition of Done)

ติ๊กได้ก็ต่อเมื่อทำบน VPS จริง และหลักฐานห้ามมี token (รหัสลับเข้าใช้งาน) แบบเต็ม

- [ ] `relay-doctor` บน VPS จบด้วย `fail=0`
- [ ] Fable auth พังตามปัญหาเดิม แล้ว `relay-call fable` สลับไป Opus 4.8 ได้จริงบน VPS โดยพิสูจน์จาก ledger ช่อง `rotated_from` และ `tried`
- [ ] Opus 4.8 ตอบกลับผ่าน `relay-call` ได้
- [ ] Grok ล็อกอินแล้ว และเรียกผ่าน `relay-call` หรือ `gate-run` ได้ โดยผลล่าสุดบน VPS ผ่านแล้ว
- [ ] Gemini มี `GEMINI_API_KEY` ใน env (ไฟล์ค่าตั้งต้น) ที่ process ใช้จริง และเรียกได้หลัง restart/reload Hermes gateway
- [ ] รุ่น 4.7 เรียกได้ หรือมีบันทึกชัดว่าไม่รองรับ พร้อมเหตุผล วันที่ และชื่อผู้ตรวจ
- [ ] รุ่น 4.6 เรียกได้ หรือมีบันทึกชัดว่าไม่รองรับ พร้อมเหตุผล วันที่ และชื่อผู้ตรวจ
- [ ] ledger (สมุดบันทึกผล) มีแถวครบสำหรับ Fable, Opus 4.8, Grok, Gemini, 4.7, 4.6
- [ ] ledger แต่ละแถวมีเวลา, คำสั่งที่ใช้, ผลลัพธ์, และ error (ข้อผิดพลาด) ที่ปิด token แล้ว
- [ ] ความจำของ Hermes Agent ตรงกับสถานะจริงบน VPS: commit `11db5b37c`, token ถูกจัดการแล้ว, Grok ล็อกอินแล้วและใช้ได้จริง, Gemini เคยพังเพราะไม่มี `GEMINI_API_KEY` ตอนรัน, 4.7/4.6 รองรับหรือไม่
- [ ] ไม่มี token แบบเต็มใน log (บันทึกการทำงาน), แชท, ticket, screenshot, หรือ ledger หลังสแกนคำว่า `TOKEN|KEY|SECRET|PASSWORD|AUTH|CREDENTIAL`
- [ ] มี backup (ไฟล์สำรอง) ของ `~/.hermes/.env` ก่อนแก้, ตั้งสิทธิ์ `600`, เก็บไม่เกิน 7 วัน, และ rollback (ย้อนกลับ) แยกได้ทั้งกรณีกู้ Hermes gateway กับกรณีกู้ relay
- [ ] หลังแก้ `.env` มีการ restart/reload Hermes gateway และ health check (ตรวจสุขภาพ) ยืนยันว่า gateway ยังรันปกติ ไม่ใช่ยืนยันแค่ `claude` หรือ `relay-call`

## 2. Runbook งานคน: แก้ Claude token บน VPS

หลักที่แนะนำ: เริ่มจากคอมเมนต์บรรทัด `CLAUDE_CODE_OAUTH_TOKEN` ใน `~/.hermes/.env` เพราะย้อนกลับง่าย ลดความเสี่ยงกับ Hermes gateway และแก้ตรงสาเหตุที่ทำให้ relay โหลด token เสีย

### 2.0 ตั้งคำสั่งปิด token ก่อนส่งผลทดสอบ

ก่อนส่งผลลัพธ์ของคำสั่งทดสอบทุกคำสั่ง ต้องปิด token ก่อนเสมอ ให้ตั้งฟังก์ชันนี้ใน shell (หน้าต่างคำสั่ง) ของ VPS:

```bash
redact() {
  perl -pe '
    s/(Authorization:\s*).+/${1}[REDACTED]/ig;
    s/\b(Bearer|Basic)\s+\S+/${1} [REDACTED]/ig;
    s/((?:[^[:space:]=:]*(?:TOKEN|KEY|SECRET|PASSWORD|AUTH|CREDENTIAL)[^[:space:]=:]*)(?:\s*[=:]\s*))\S+/${1}[REDACTED]/ig;
  '
}
```

กฎบังคับ:
- ทุกคำสั่งทดสอบที่ต้องส่ง output (ผลลัพธ์) เข้าแชท, ticket, ledger, หรือ screenshot ต้องต่อท้ายด้วย `2>&1 | redact`
- ห้ามส่ง output ดิบจาก `env`, `systemctl status`, `systemctl show-environment`, `journalctl`, หรือ log ของ gateway เพราะอาจโชว์ token ได้
- ถ้าคำสั่งใดพัง ให้ส่งเฉพาะ output ที่ผ่าน `redact` แล้ว

### 2.1 เข้า VPS และ backup ก่อนแก้

```bash
ssh myserver
umask 077
BACKUP_FILE="$HOME/.hermes/.env.backup.$(date +%Y%m%d-%H%M%S)"
cp -p ~/.hermes/.env "$BACKUP_FILE"
chmod 600 "$BACKUP_FILE"
ls -lh "$BACKUP_FILE"
find ~/.hermes -maxdepth 1 -name '.env.backup.*' -type f -mtime +7 -exec rm -f {} \;
```

backup นี้มี secret (รหัสลับ) อยู่ข้างใน จึงห้าม log, ห้ามแนบแชท, ห้ามแชร์ และต้องเก็บไม่เกิน 7 วัน ถ้า token ใน backup เป็น token ของ org ที่ปิดสิทธิ์หรือสงสัยว่าหลุด ให้ revoke/rotate (ปิดสิทธิ์หรือออก token ใหม่) ที่ org หลังระบบกลับมารันได้

### 2.2 ดูว่าบรรทัดไหนคือ token เสียแบบไม่โชว์ token

ห้ามใช้ `cat ~/.hermes/.env` ในแชทหรือ log

```bash
grep -n '^CLAUDE_CODE_OAUTH_TOKEN=' ~/.hermes/.env 2>&1 | redact
```

ถ้าขึ้นเลขบรรทัด แปลว่ามี token ที่ relay โหลดแล้วทำให้ Claude Code เสีย

### 2.3 จัดการ token

ทางหลัก: คอมเมนต์บรรทัดเดิมด้วย editor (โปรแกรมแก้ไฟล์)

```bash
nano ~/.hermes/.env
```

ให้หาบรรทัดนี้:

```bash
CLAUDE_CODE_OAUTH_TOKEN=...
```

แล้วเปลี่ยนเป็น:

```bash
# CLAUDE_CODE_OAUTH_TOKEN=...
```

บันทึกไฟล์แล้วออกจาก editor จากนั้นตรวจอีกครั้งแบบปิด token:

```bash
grep -n 'CLAUDE_CODE_OAUTH_TOKEN' ~/.hermes/.env 2>&1 | redact
```

ถ้าจำเป็นต้องลบแทนคอมเมนต์ ให้ backup ก่อนเสมอ แล้วลบบรรทัดนั้นใน editor เท่านั้น

ถ้าต้องแยก env ของ relay กับ Hermes gateway:
- ให้เก็บ `~/.hermes/.env` สำหรับ Hermes gateway
- ให้สร้าง env ใหม่สำหรับ relay เช่น `~/.hermes/relay.env`
- ใน env ของ relay ห้ามมี `CLAUDE_CODE_OAUTH_TOKEN` ที่เสีย
- ทำวิธีนี้เฉพาะเมื่อผู้ดูแลยืนยันแล้วว่าจุดเริ่ม process ของ relay รองรับ env แยก

### 2.4 Restart/reload Hermes gateway และตรวจ health

หลังแก้ `.env` ต้อง restart/reload Hermes gateway ก่อนทดสอบ `claude` หรือ relay เพราะ gateway อาจยังถือค่า env เก่าอยู่

ถ้า service (บริการระบบ) เป็น systemd ระดับ user:

```bash
GATEWAY_SERVICE="hermes-gateway"
systemctl --user reload-or-restart "$GATEWAY_SERVICE" 2>&1 | redact
systemctl --user is-active "$GATEWAY_SERVICE" 2>&1 | redact
systemctl --user status "$GATEWAY_SERVICE" --no-pager 2>&1 | redact | tail -80
```

ถ้า service เป็น systemd ระดับเครื่อง:

```bash
GATEWAY_SERVICE="hermes-gateway"
sudo systemctl reload-or-restart "$GATEWAY_SERVICE" 2>&1 | redact
sudo systemctl is-active "$GATEWAY_SERVICE" 2>&1 | redact
sudo systemctl status "$GATEWAY_SERVICE" --no-pager 2>&1 | redact | tail -80
```

จากนั้นตรวจ health endpoint (จุดตรวจสุขภาพ) ที่ gateway ใช้จริง:

```bash
# เติมค่าเอง: URL ที่คุณใช้เช็กว่า Hermes dashboard/gateway ยังมีชีวิต
# (AI ยังไม่ยืนยันค่านี้ · VPS มีหลาย service/port เช่น :8099 :3062 :80 · เจ้าของรู้ค่าจริง)
HERMES_GATEWAY_HEALTH_URL="<health URL ของ Hermes gateway ที่ VPS ใช้จริง>"
curl -fsS "$HERMES_GATEWAY_HEALTH_URL" 2>&1 | redact
```

ผลที่ถือว่าผ่าน:
- `is-active` ต้องเป็น `active`
- health endpoint ต้องตอบสำเร็จ
- output ที่ส่งออกนอก VPS ต้องผ่าน `redact`
- ห้ามนับว่า gateway ปกติจากการที่ `claude` หรือ `relay-call` ตอบได้อย่างเดียว

ถ้าไม่รู้ health URL ให้ตรวจ service file ที่ใช้งานจริงแบบปิด token แล้วถามผู้ดูแล ห้ามเดา port:

```bash
systemctl --user cat "$GATEWAY_SERVICE" 2>&1 | redact
```

### 2.5 ทดสอบว่า Claude เรียกได้หลังแก้

ใช้คำสั่งสั้น ๆ ที่ไม่พิมพ์ token

```bash
claude -p "ตอบคำว่า OK เท่านั้น" 2>&1 | redact
```

ผลที่ถือว่าผ่าน:
- มีคำว่า `OK`
- ไม่มี error เรื่อง org ปิดสิทธิ์ Claude Code
- ไม่มี token โผล่ในผลลัพธ์

จากนั้นทดสอบทาง AI Relay:

```bash
MARKER="AUTO_ROTATE_PROOF_$(date +%Y%m%d-%H%M%S)"
relay-call fable "ตอบคำว่า OK เท่านั้น $MARKER" 2>&1 | redact
relay-call opus-4.8 "ตอบคำว่า OK เท่านั้น" 2>&1 | redact
```

ผลที่ถือว่าผ่าน:
- กรณี Fable auth เสีย ต้องเห็นว่า relay สลับไป Opus 4.8 แล้วได้คำตอบ
- Opus 4.8 ต้องตอบ `OK`
- log ต้องปิด token ทุกจุด

### 2.6 พิสูจน์ว่า Fable สลับไป Opus อัตโนมัติจริง

ห้ามใช้ผลจาก `relay-call opus-4.8` ตรง ๆ เป็นหลักฐานว่า Fable สลับอัตโนมัติ เพราะนั่นเป็นการเรียก Opus โดยตรง ให้พิสูจน์จาก ledger ของคำสั่ง `relay-call fable` เท่านั้น

```bash
# ยืนยันแล้ว: relay-call เขียน ledger ที่ <โฟลเดอร์ที่รัน>/.hermes/ai-relay/calls-<branch>.md
# ถ้ารันนอก git หรือไม่มี branch = calls-nobranch.md (เช่น /tmp/relay-vps-test/.hermes/ai-relay/calls-nobranch.md)
# หมายเหตุ: ไฟล์นี้เป็นตาราง markdown ไม่ใช่ JSON — jq ข้างล่างใช้ได้เฉพาะถ้า ledger เป็น json บรรทัดต่อบรรทัด · ถ้าเป็นตาราง ให้ใช้ grep "$MARKER" แทน
LEDGER_FILE="<ledger file ที่ relay-call เขียนจริง · ตามรูปแบบด้านบน>"
tail -n 80 "$LEDGER_FILE" \
  | jq -c --arg marker "$MARKER" 'select(tostring | contains($marker)) | {time, command, rotated_from, tried, result, error}' \
  2>&1 | redact
```

ผลที่ถือว่าผ่าน:
- แถว ledger ต้องเป็นคำสั่งที่เริ่มจาก `relay-call fable`
- ช่อง `rotated_from` ต้องระบุว่าเริ่มจาก `fable`
- ช่อง `tried` ต้องเห็นลำดับที่ลอง `fable` แล้วต่อด้วย `opus-4.8`
- ผลลัพธ์สุดท้ายต้องได้ `OK` หรือสถานะสำเร็จ
- error ที่ส่งเข้าแชทหรือ ledger ต้องผ่าน `redact`

ถ้า ledger ไม่เก็บ marker ใน prompt ให้ใช้เวลาของคำสั่งแทน และส่งเฉพาะแถวล่าสุดช่วงเวลานั้นแบบปิด token:

```bash
tail -n 20 "$LEDGER_FILE" \
  | jq -c '{time, command, rotated_from, tried, result, error}' \
  2>&1 | redact
```

### 2.7 ทางถอยเพื่อกู้ Hermes gateway

ใช้กรณี gateway ไม่ active, health endpoint ไม่ผ่าน, หรือ service start ไม่ขึ้นหลังแก้ `.env` เป้าหมายคือกู้ gateway ก่อน ยังไม่ตัดสินว่า relay ผ่านหรือไม่ผ่าน

หาไฟล์ backup ล่าสุด:

```bash
ls -lt ~/.hermes/.env.backup.* 2>&1 | redact | head
```

คืนไฟล์เดิม โดยแทน `<BACKUP_FILE>` ด้วยชื่อไฟล์ backup ที่ต้องการ:

```bash
cp <BACKUP_FILE> ~/.hermes/.env
chmod 600 ~/.hermes/.env
```

เริ่ม Hermes gateway ใหม่ด้วยคำสั่งที่เครื่องนี้ใช้อยู่ ถ้าเป็น systemd ระดับ user ให้ใช้:

```bash
systemctl --user restart hermes-gateway 2>&1 | redact
systemctl --user is-active hermes-gateway 2>&1 | redact
```

ถ้าเครื่องนี้ใช้ systemd ระดับเครื่อง ให้ใช้:

```bash
sudo systemctl restart hermes-gateway 2>&1 | redact
sudo systemctl is-active hermes-gateway 2>&1 | redact
```

ตรวจว่า gateway กลับมา:

```bash
systemctl --user status hermes-gateway --no-pager 2>&1 | redact | tail -80
```

ถ้าคำสั่ง status ไม่เจอ service ให้หยุดและส่ง error ที่ปิด token แล้วให้ผู้ดูแล ห้ามลองเดาสุ่มชื่อ service ต่อ

### 2.8 ทางถอยเพื่อกู้ relay

ใช้กรณี Hermes gateway health ผ่าน แต่ `relay-call` หรือ `gate-run` ไม่ผ่านหลังแก้ env เป้าหมายคือกู้ relay โดยไม่ทำให้ gateway ที่รันดีอยู่พังซ้ำ

1. ตรวจว่า gateway ยังปกติก่อน:

```bash
systemctl --user is-active hermes-gateway 2>&1 | redact
curl -fsS "$HERMES_GATEWAY_HEALTH_URL" 2>&1 | redact
```

2. ถ้า relay ใช้ env แยก เช่น `~/.hermes/relay.env` ให้ถอยเฉพาะไฟล์ env ของ relay แล้ว restart/reload process ของ relay ตาม service ที่เครื่องนี้ใช้จริง
3. ถ้า relay ใช้ `~/.hermes/.env` ร่วมกับ gateway ให้ restore backup เฉพาะเมื่อจำเป็นต้องกู้บริการทันที และต้องรู้ว่าจะเอา token เสียกลับเข้ามาหรือไม่
4. หลังถอยเพื่อกู้ relay ต้องทดสอบ `relay-call fable`, `relay-call opus-4.8`, `relay-call grok`, และ `relay-call gemini` แบบ `2>&1 | redact`
5. ถ้า backup มี token ของ org ที่ปิดสิทธิ์ ให้ revoke/rotate token ที่ org และสร้าง token ใหม่ที่มีสิทธิ์ถูกต้องหลังระบบกลับมารัน

## 3. Runbook งานคน: ยืนยัน Grok บน VPS

ผลตรวจจริงล่าสุดคือ VPS Grok ล็อกอินแล้วและใช้ได้จริง ขั้นตอนนี้มีไว้เก็บหลักฐานปิดงานแบบปิด token ไม่ใช่ถือว่า Grok ยังไม่ได้ล็อกอิน

### 3.1 ทดสอบ Grok จากสถานะที่ล็อกอินแล้ว

```bash
ssh myserver
relay-call grok "ตอบคำว่า OK เท่านั้น" 2>&1 | redact
gate-run grok "ตอบคำว่า OK เท่านั้น" 2>&1 | redact
```

ผลที่ถือว่าผ่าน:
- ได้คำว่า `OK`
- ไม่มีข้อความให้ล็อกอินซ้ำ
- ไม่มี token หรือ credential ใน log

### 3.2 ล็อกอินใหม่เฉพาะเมื่อผลทดสอบบอกว่าสิทธิ์หาย

device-auth (ล็อกอินด้วยรหัสบนเว็บ) ต้องทำบน VPS เพราะ credential (ข้อมูลเข้าใช้งาน) ต้องอยู่ในเครื่องที่รัน relay

```bash
grok auth login --device 2>&1 | redact
```

ระบบจะให้ URL และรหัส ให้เปิด URL ใน browser (เว็บเบราว์เซอร์) ของคนทำงาน แล้วกรอกรหัสตามที่หน้าจอบอก

ข้อควรระวัง:
- ห้ามส่ง token หรือ credential เข้าแชท
- ส่ง device code ได้เฉพาะตอนกำลังล็อกอิน และไม่ต้องเก็บไว้ใน ticket
- ถ้าคำสั่ง `grok auth login --device` ไม่พบ ให้หยุด แล้วส่งเฉพาะ error ที่ปิด token แล้วให้ผู้ดูแล

## 4. กฎความปลอดภัย

- ห้ามพิมพ์ token แบบเต็มในแชท, log, ticket, screenshot, หรือ ledger
- ห้ามใช้ `cat ~/.hermes/.env` เพื่อส่งผลลัพธ์ให้คนอื่น
- ใช้คำสั่งตรวจแบบปิด token เท่านั้น
- `systemctl status`, `systemctl show-environment`, `env`, และ `journalctl` อาจโชว์ token ผ่าน env หรือ command line ได้ ต้องส่งผ่าน `redact` ก่อนเสมอ
- backup ของ `.env` มี secret อยู่จริง ต้องตั้งสิทธิ์ `600`, ห้ามแชร์, เก็บไม่เกิน 7 วัน, และลบทิ้งเมื่อไม่ต้องใช้ rollback แล้ว
- ถ้า token อยู่ใน org ที่ปิดสิทธิ์ หรือมีโอกาสหลุดจาก log/screenshot ให้ revoke/rotate token ที่ org ไม่ใช่แค่ลบออกจากไฟล์บน VPS

ตัวอย่างคำสั่งตรวจ env แบบปิด token:

```bash
env 2>&1 | redact
systemctl --user show-environment 2>&1 | redact
```

ตัวอย่างคำสั่งตรวจไฟล์ `.env` แบบปิด token:

```bash
grep -nE 'TOKEN|KEY|SECRET|PASSWORD|AUTH|CREDENTIAL' ~/.hermes/.env 2>&1 | redact
```

ตัวอย่างคำสั่งสแกน log ล่าสุดของ Hermes gateway เพื่อยืนยันว่าไม่มี token หลุด:

```bash
SECRET_PATTERN='TOKEN|KEY|SECRET|PASSWORD|AUTH|CREDENTIAL|Bearer[[:space:]]+|Basic[[:space:]]+'
MATCHES="$(journalctl --user -u hermes-gateway --since '1 hour ago' --no-pager 2>&1 | grep -Eini "$SECRET_PATTERN" | redact || true)"
if [ -n "$MATCHES" ]; then
  printf '%s\n' "$MATCHES"
  echo "FAIL: พบข้อความที่ดูเหมือน secret ใน journal ให้ตรวจบน VPS และห้ามส่ง raw log"
else
  echo "OK: ไม่พบข้อความที่ดูเหมือน secret ใน journal ของ gateway ช่วง 1 hour ago"
fi
```

ตัวอย่างคำสั่งสแกน log และ ledger ใต้ `~/.hermes`:

```bash
SECRET_PATTERN='TOKEN|KEY|SECRET|PASSWORD|AUTH|CREDENTIAL|Bearer[[:space:]]+|Basic[[:space:]]+'
MATCHES="$(find ~/.hermes -type f \( -name '*.log' -o -name '*ledger*' \) -print0 | xargs -0 -r grep -Eini "$SECRET_PATTERN" 2>/dev/null | redact || true)"
if [ -n "$MATCHES" ]; then
  printf '%s\n' "$MATCHES"
  echo "FAIL: พบข้อความที่ดูเหมือน secret ในไฟล์ log/ledger ให้ตรวจบน VPS และห้ามส่ง raw file"
else
  echo "OK: ไม่พบข้อความที่ดูเหมือน secret ใน log/ledger ใต้ ~/.hermes"
fi
```

ถ้าต้องส่ง error ให้ส่งเฉพาะส่วนนี้:

```text
command: <คำสั่งที่ใช้>
result: <ผ่าน/ไม่ผ่าน>
error_redacted: <error ที่แทน token ด้วย [REDACTED]>
time: <เวลา VPS>
operator: <ชื่อคนทำ>
```

## 5. ลำดับทำ Phase 2 -> 3 -> 4 -> 5

### Phase 2: แก้ Claude token และยืนยัน Opus 4.8

1. ตั้ง `redact`
2. Backup `~/.hermes/.env` แบบ `chmod 600` และลบ backup ที่เก่ากว่า 7 วัน
3. ตรวจบรรทัด `CLAUDE_CODE_OAUTH_TOKEN` แบบปิด token
4. คอมเมนต์หรือลบ token เสีย
5. Restart/reload Hermes gateway
6. ตรวจ `is-active` และ health endpoint ของ Hermes gateway
7. ทดสอบ `claude -p` แบบ `2>&1 | redact`
8. ทดสอบ `relay-call fable` แบบ `2>&1 | redact`
9. อ่าน ledger เพื่อพิสูจน์การสลับอัตโนมัติจาก `rotated_from` และ `tried`
10. ทดสอบ `relay-call opus-4.8` แบบ `2>&1 | redact`
11. สแกน journal/log/ledger ว่าไม่มี token หลุด
12. บันทึกผลลง ledger

### Phase 3: ยืนยัน Grok

1. ยึดสถานะล่าสุดว่า VPS Grok ล็อกอินแล้วและใช้ได้จริง
2. ทดสอบ `relay-call grok` แบบ `2>&1 | redact`
3. ทดสอบ `gate-run grok` แบบ `2>&1 | redact`
4. ล็อกอินใหม่ด้วย `grok auth login --device` เฉพาะเมื่อผลทดสอบบอกว่าสิทธิ์หาย
5. บันทึกผลลง ledger

### Phase 4: ตรวจครบทั้ง relay

1. รัน `relay-doctor 2>&1 | redact`
2. ตรวจว่า `fail=0`
3. ทดสอบ Gemini อีกครั้งแบบ `2>&1 | redact`
4. ถ้า Gemini ยังพังด้วยข้อความไม่มี `GEMINI_API_KEY` ตอนรัน ให้หยุดปิดงาน, ใส่ `GEMINI_API_KEY` ใน env ที่ process ใช้จริง, restart/reload Hermes gateway, ตรวจ health, แล้วทดสอบใหม่
5. ทดสอบ 4.7 แบบ `2>&1 | redact`
6. ทดสอบ 4.6 แบบ `2>&1 | redact`
7. ถ้า 4.7 หรือ 4.6 ไม่รองรับ ให้จดว่าไม่รองรับ พร้อม error ที่ปิด token แล้ว
8. ตรวจ ledger ว่ามีแถวครบตาม DoD

### Phase 5: ปิดงานและอัปเดตความจำ

1. อัปเดตความจำ Hermes Agent ให้ตรงกับ VPS จริง
2. ระบุว่า repo และ VPS อยู่ที่ commit `11db5b37c`
3. ระบุวิธีจัดการ `CLAUDE_CODE_OAUTH_TOKEN`
4. ระบุว่า Grok บน VPS ล็อกอินแล้วและใช้ได้จริงตามผลตรวจล่าสุด
5. ระบุว่า Gemini บน VPS เคยพังเพราะไม่มี `GEMINI_API_KEY` ตอนรัน และปิดงานได้หลังแก้ env จริงพร้อมทดสอบใหม่ผ่านเท่านั้น
6. ระบุสถานะ Fable, Opus 4.8, Gemini, 4.7, 4.6
7. แนบผล `relay-doctor fail=0` แบบไม่มี token
8. แนบผลสแกน log/ledger ว่าไม่มี token หลุด
9. ปิดงานเมื่อ checklist ใน DoD ถูกติ๊กครบ
