---
title: Use SonarQube
shortcut: Use SonarQube
aliases:
  - use-sonarqube
  - SonarQube
  - ใช้ SonarQube
  - สแกน SonarQube
  - ตรวจโค้ดด้วย SonarQube
status: active
version: 1.0.0
updated: 2026-07-11
---

# Use SonarQube

ใช้ Shortcut นี้สำหรับวิเคราะห์โปรเจกต์ด้วย SonarQube ที่ติดตั้งไว้แล้ว ห้ามใช้ Shortcut นี้ติดตั้ง SonarQube ใหม่

## รูปแบบเรียกใช้

```text
Use SonarQube กับโปรเจกต์นี้
```

```text
Use SonarQube กับ /absolute/path/to/project
```

```text
Use SonarQube สรุปผลโปรเจกต์ <project-key>
```

## เป้าหมาย

1. ตรวจโปรเจกต์ปัจจุบันหรือ path ที่ผู้ใช้ระบุ
2. เลือก SonarScanner ให้ตรงกับภาษาและระบบ build
3. ส่งผลไปยัง SonarQube บน VPS ที่ตั้งค่าไว้แล้ว
4. ตรวจว่าผลวิเคราะห์เข้าระบบจริง
5. สรุปปัญหาและลำดับการแก้เป็นภาษาไทย

## กฎบังคับ

1. อ่าน `AGENTS.md` และกฎของโปรเจกต์ก่อนทำงาน
2. ถ้าไม่ทราบเป้าหมาย ให้ถาม path เพียงหนึ่งคำถาม ห้ามเดา
3. ห้ามติดตั้งหรืออัปเกรด SonarQube Server ผ่าน Shortcut นี้
4. ห้ามอ่าน แสดง หรือบันทึก token และ secret
5. อ่านค่าเชื่อมต่อจาก `SONAR_HOST_URL` และ `SONAR_TOKEN` ใน environment ก่อน
6. ถ้าค่าเชื่อมต่อไม่มี ให้บอกวิธีตั้งค่าและหยุด ห้ามเขียน token ลงไฟล์
7. ห้ามส่ง source ไปบริการอื่นนอกเหนือจาก SonarQube ที่เจ้าของกำหนด
8. ห้ามแก้ source code โดยอัตโนมัติ Shortcut นี้มีหน้าที่สแกนและสรุปเท่านั้น หากผู้ใช้ต้องการแก้ ให้เสนอแผนและรออนุมัติ
9. ห้ามกล่าวว่าผ่านถ้ามีเพียง exit code ของ scanner ต้องตรวจผลจาก SonarQube API ด้วย
10. ถ้าใช้เครื่องส่วนตัวและ `SONAR_HOST_URL` เป็น `127.0.0.1:9000` ให้ตรวจ SSH tunnel ก่อน

## ขั้นตอนทำงาน

### 1. ตรวจเป้าหมาย

- หา repo root จาก `git rev-parse --show-toplevel` หรือ path ที่ผู้ใช้ระบุ
- รายงาน path, branch, commit SHA และ dirty status
- อ่านไฟล์กำหนดภาษาและระบบ build เช่น `package.json`, `pom.xml`, `build.gradle`, `.sln`, `pyproject.toml`, `go.mod`
- ตรวจไฟล์ตั้งค่า SonarQube ที่มีอยู่ก่อน ห้ามสร้างซ้ำ
- ตรวจว่ามีผล coverage จากชุดทดสอบหรือไม่ โดยไม่รันชุดทดสอบเองหากผู้ใช้ไม่ได้สั่ง

### 2. ตรวจการเชื่อมต่อ

- ตรวจว่ามี `SONAR_HOST_URL` และ `SONAR_TOKEN` โดยรายงานเฉพาะ `มี/ไม่มี`
- เรียก `/api/system/status` และต้องได้ HTTP 200 พร้อมสถานะ `UP`
- ถ้า server ไม่ตอบ ให้ตรวจ tunnel, DNS หรือบริการตามขอบเขตที่อ่านได้ แต่ห้าม restart VPS หรือ SonarQube เองโดยไม่มีคำสั่งจากผู้ใช้

### 3. ระบุโปรเจกต์

- ใช้ `sonar.projectKey` เดิมถ้ามี
- ถ้ายังไม่มี ให้สร้างจากชื่อ repo ที่ไม่เปิดเผยชื่อลูกค้า ใช้อักษร ตัวเลข จุด ขีด และขีดล่างเท่านั้น
- ตรวจผ่าน SonarQube API ว่า key ชนโปรเจกต์อื่นหรือไม่
- ห้ามลบหรือเขียนทับโปรเจกต์ที่มี key ชน ให้หยุดและรายงาน

### 4. เลือกเครื่องสแกน

- Maven ใช้ SonarScanner for Maven
- Gradle ใช้ SonarScanner for Gradle
- .NET ใช้ SonarScanner for .NET
- JavaScript หรือ TypeScript ที่ไม่มีตัวเชื่อม build ให้ใช้ scanner สำหรับ NPM หรือ SonarScanner CLI ตามเอกสารปัจจุบัน
- Python, Go และโปรเจกต์ทั่วไปใช้ SonarScanner CLI เมื่อไม่มีตัวเชื่อมที่เหมาะกว่า
- ตรวจรุ่น scanner จากเอกสารทางการก่อนติดตั้ง ห้ามตรึงรุ่นจากความจำ
- ถ้าต้องเพิ่ม dependency หรือแก้ไฟล์โปรเจกต์ ให้แสดง diff ที่วางแผนไว้และรออนุมัติก่อน

### 5. ตั้งขอบเขตการสแกน

- รวม source และ test ที่เป็นของโปรเจกต์
- ไม่รวม `.git`, `node_modules`, virtual environment, build output, cache, generated files, vendor, backup, secret และไฟล์ binary
- ใช้ค่า exclusions เดิมของโปรเจกต์ก่อน
- ห้ามซ่อน source ที่มีปัญหาเพียงเพื่อให้คะแนนดีขึ้น

### 6. รันและตรวจผล

- รัน scanner จาก repo root
- จับ command ที่ตัด token ออก, exit code และเวลาจบ
- ตรวจ task id หรือ analysis id จากผล scanner
- เรียก SonarQube API จน background task จบหรือผิดพลาด
- ตรวจวันที่วิเคราะห์และ commit SHA หากระบบรองรับ
- ดึง Quality Gate และตัวเลขล่าสุดจาก API

### 7. สรุปภาษาไทย

รายงานตามรูปแบบนี้:

```text
SonarQube Scan Report
- โปรเจกต์: <name>
- Project key: <key>
- Path: <path>
- Branch / SHA: <branch> / <sha>
- Server: ตอบ HTTP <code> · สถานะ <status>
- Scanner: <name + version>
- Analysis task: <SUCCESS|FAILED|PENDING>
- Quality Gate: <PASSED|FAILED|UNKNOWN>

ผลหลัก
- Reliability: <จำนวนและระดับ>
- Security: <จำนวนและระดับ>
- Security Hotspots: <จำนวนที่ต้องตรวจโดยคน>
- Maintainability: <จำนวนและ technical debt ที่ระบบรายงาน>
- Duplications: <เปอร์เซ็นต์>
- Coverage: <เปอร์เซ็นต์หรือไม่มีข้อมูล>

ปัญหาที่ควรแก้ก่อน
1. <ไฟล์:บรรทัด · ปัญหา · ผลกระทบ>
2. ...

หลักฐาน
- Scanner exit code: <code>
- Server analysis: <status>
- ตรวจสำเร็จ: <N/M>

ความเสี่ยงที่ยังเหลือ
- <รายการ>

ขั้นตอนถัดไปที่แนะนำ
- <หนึ่งข้อ>
```

ถ้า API ให้ข้อมูลไม่ครบ ให้ใช้คำว่า `ไม่ทราบ (ยังไม่ได้ตรวจยืนยัน)` ห้ามใส่ค่าประมาณ

## การเรียกใช้บนเครื่องส่วนตัว

เปิด tunnel ก่อน:

```bash
ssh -N -L 9000:127.0.0.1:9000 <VPS_USER>@<VPS_HOST>
```

อีกหน้าต่างหนึ่งตั้งค่าเฉพาะ session ปัจจุบัน:

```bash
export SONAR_HOST_URL="http://127.0.0.1:9000"
export SONAR_TOKEN="<TOKEN_FROM_SONARQUBE>"
```

จากนั้นเข้า repo แล้วเรียก:

```text
Use SonarQube กับโปรเจกต์นี้
```

## Worktree Lifecycle v1

อ่าน `worktree-lifecycle-contract.md` ก่อนใช้ Prompt นี้ · ผูกผล SonarQube กับ `task_id + worktree path + branch + commit SHA`; scan เป็น read-only ส่วนการแก้ finding ต้องเปิด/ใช้ task worktree ที่ `WTL_READY`

## กรณีที่ต้องหยุด

- SonarQube ไม่ตอบหรือสถานะไม่ใช่ `UP`
- ไม่มี token
- project key ชน
- ต้องแก้ dependency หรือไฟล์โครงการแต่ยังไม่ได้รับอนุมัติ
- scanner ล้มเหลวสามครั้งด้วยสาเหตุชนิดเดิม
- พบว่าคำสั่งอาจส่ง secret หรือ source ไปปลายทางที่ไม่ใช่เซิร์ฟเวอร์ของเจ้าของ
