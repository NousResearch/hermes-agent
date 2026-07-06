#!/bin/bash
# ดับเบิลคลิกครั้งเดียว = ติดตั้งจาร์วิสเป็นระบบเปิดเองตอนเข้าเครื่อง (launchd)
# หลังจากนี้ไม่ต้องเปิด Terminal อีก · เปิดเครื่องมาจาร์วิสตื่นเอง
# ไฟล์นี้ใช้ซ้ำได้ทุกเมื่อ = สั่งรีสตาร์ตจาร์วิส (เช่นหลังอัพโค้ดใหม่)
PLIST="$HOME/Library/LaunchAgents/com.rattanasak.jarvis.plist"
UID_NUM="$(id -u)"

# ปิดตัวเก่าทุกแบบก่อน (ทั้งที่เปิดมือและที่ launchd เปิด)
launchctl bootout "gui/$UID_NUM/com.rattanasak.jarvis" 2>/dev/null
pkill -f "jarvis_app.py" 2>/dev/null
sleep 1

# ลงทะเบียน + เปิดทันที
launchctl bootstrap "gui/$UID_NUM" "$PLIST" && sleep 2

if pgrep -f "jarvis_app.py" >/dev/null; then
  echo "จาร์วิสทำงานแล้ว ✅ และจะเปิดเองอัตโนมัติทุกครั้งที่เข้าเครื่อง"
  echo "ปุ่มลัด Cmd+Shift+J = เปิด/ปิดพูด"
  echo "อยากปิดถาวร: เปิด Terminal พิมพ์  launchctl bootout gui/$UID_NUM/com.rattanasak.jarvis"
else
  echo "❌ เปิดไม่สำเร็จ — ดูสาเหตุที่ /tmp/jarvis-app.log"
  tail -5 /tmp/jarvis-app.log 2>/dev/null
fi
sleep 3
