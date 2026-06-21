# Hermes VPS Team Workspace Runbook

เอกสารนี้คือคู่มือสั้นสำหรับเข้าไปทำงานกับ Hermes Agent บน VPS

## เป้าหมาย

ทีมต้องทำงานจาก workspace นี้เท่านั้น:

```text
/home/linux-nat/projects/hermes-agent
```

> หมายเหตุ · สภาพจริงที่ตรวจพบบนเครื่อง (2026-06-21 · อ่านอย่างเดียว) ·
> path ด้านบนยัง **ไม่มีอยู่จริง** บนเครื่อง · ของจริงที่กำลังรันอยู่คือ ·
>
> - Hermes Agent + Dashboard รันจาก `/home/linux-nat/SynerryTools/hermes-agent/main`
>   (บริการ `hermes-dashboard.service` พอร์ต 9119 สถานะ active)
> - ศูนย์ความรู้ทีมรันจาก `/home/linux-nat/projects/hermes-knowledge-portal`
> - บริการ `hermes-knowledge-backup.service` และ `hermes-knowledge-export.service`
>   สถานะ **failed** (น่าจะเพราะดิสก์เต็ม 93%) · ต้องกู้เครื่องก่อน
>
> เจ้าของงานต้องตัดสินใจ · จะย้ายของจริงมาที่ path มาตรฐานด้านบน หรือแก้เอกสาร
> ให้ชี้ path จริง · ผมไม่เดาเจตนา จึงบันทึกสภาพจริงไว้ให้ก่อน

## เข้าเครื่อง

```bash
ssh linux-nat@103.142.150.185
```

## เข้า workspace

```bash
cd /home/linux-nat/projects/hermes-agent
```

## เปิด Dashboard จากเครื่องตัวเอง

เปิดทางเข้าแบบปลอดภัย:

```bash
ssh -L 9119:127.0.0.1:9119 linux-nat@103.142.150.185
```

แล้วเปิดใน browser:

```text
http://127.0.0.1:9119
```

## ตรวจว่า service ยังรันอยู่

```bash
systemctl --user status hermes-dashboard.service --no-pager -l
```

## Restart Dashboard

```bash
systemctl --user restart hermes-dashboard.service
```

## ดู log ล่าสุด

```bash
journalctl --user -u hermes-dashboard.service -n 100 --no-pager
```

## รัน Hermes จาก workspace นี้

```bash
cd /home/linux-nat/projects/hermes-agent
venv/bin/python -m hermes_cli.main --help
```

## อัปเดต workspace จาก Local

จากเครื่อง Local ให้รัน:

```bash
scripts/sync_vps_workspace.sh
```

คำสั่งนี้กันไฟล์ลับและไฟล์หนักออก เช่น `.env`, `.hermes/.env`, `venv`, `node_modules`, cache และ log

## ข้อห้าม

- ห้ามใช้ path อื่นแทน `/home/linux-nat/projects/hermes-agent`
- ห้ามเปิด Dashboard ด้วย `--insecure`
- ห้าม copy `.env` จาก Local เข้า repo workspace
- ถ้าต้องเปิดผ่าน domain ให้ทำผ่าน Nginx พร้อม authentication เท่านั้น
