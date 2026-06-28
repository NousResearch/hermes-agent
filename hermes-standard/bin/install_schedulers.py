#!/usr/bin/env python3
"""
install_schedulers — สร้างไฟล์ตั้งเวลา (req-28, req-43)
สร้างคอนฟิกให้รัน: hermes_scan ทุกวัน + hermes_analyze ทุก 3 ชม.
Mac = launchd plist · Linux = crontab · เขียนไฟล์ไว้ที่ hermes-standard/scheduler/ + บอกคำสั่งติดตั้ง

  python install_schedulers.py <root_projects_dir> <curse_data_dir>
"""
import os
import sys
import platform

BIN = os.path.dirname(os.path.abspath(__file__))
STD_ROOT = os.path.dirname(BIN)
OUT = os.path.join(STD_ROOT, "scheduler")
PY = sys.executable


def main():
    root = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else "<root_projects>"
    data = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else "<curse_data>"
    os.makedirs(OUT, exist_ok=True)
    scan = "%s %s/hermes_scan.py %s --html %s/scan-latest.html" % (PY, BIN, root, OUT)
    ana = "%s %s/hermes_analyze.py --data %s --json %s/analyze-latest.json" % (PY, BIN, data, OUT)

    system = platform.system()
    if system == "Darwin":
        cron = ("# crontab สำหรับ Mac/Linux (รัน: crontab -e แล้ววางบรรทัดล่าง)\n"
                "0 9 * * * %s\n" % scan +
                "0 */3 * * * %s\n" % ana)
        plist = ("<?xml version=\"1.0\"?><!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" "
                 "\"http://www.apple.com/DTDs/PropertyList-1.0.dtd\"><plist version=\"1.0\"><dict>"
                 "<key>Label</key><string>com.hermes.scan</string>"
                 "<key>ProgramArguments</key><array><string>/bin/sh</string><string>-c</string>"
                 "<string>%s</string></array>"
                 "<key>StartCalendarInterval</key><dict><key>Hour</key><integer>9</integer>"
                 "<key>Minute</key><integer>0</integer></dict></dict></plist>\n" % scan)
        open(os.path.join(OUT, "crontab.txt"), "w", encoding="utf-8").write(cron)
        open(os.path.join(OUT, "com.hermes.scan.plist"), "w", encoding="utf-8").write(plist)
        print("สร้างแล้ว: scheduler/crontab.txt + scheduler/com.hermes.scan.plist")
        print("ติดตั้งบน Mac (อย่างใดอย่างหนึ่ง):")
        print("  crontab:  crontab %s/crontab.txt" % OUT)
        print("  launchd:  cp %s/com.hermes.scan.plist ~/Library/LaunchAgents/ && launchctl load ~/Library/LaunchAgents/com.hermes.scan.plist" % OUT)
    else:
        cron = ("0 9 * * * %s\n" % scan + "0 */3 * * * %s\n" % ana)
        open(os.path.join(OUT, "crontab.txt"), "w", encoding="utf-8").write(cron)
        print("สร้างแล้ว: scheduler/crontab.txt (Linux/VPS)")
        print("ติดตั้ง: crontab %s/crontab.txt" % OUT)
    print("\nสแกนทุกวัน 09:00 · วิเคราะห์คำด่าทุก 3 ชม.")


if __name__ == "__main__":
    main()
