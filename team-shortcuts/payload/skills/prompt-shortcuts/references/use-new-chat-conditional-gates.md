# Use New Chat · Conditional Gates

อ่านไฟล์นี้เฉพาะเมื่อ Use New Chat พบเงื่อนไขที่ตรงกัน ห้ามโหลดทุกแชต

## Legacy Memory Migration

- ถ้า `.hermes/plan.md`, `.hermes/active.md`, `.hermes/decisions.md` หรือ `handoff.md` ยังมีเนื้อหาจริง ให้ย้ายตาม Memory Schema §1b ทำไฟล์เดิมเป็น stub ชี้ `.project/` และแก้จุดอ้างทางเก่าในไฟล์กฎรากกับ `hermes.project.yaml` ในรอบเดียวกัน
- หลังย้ายหรือสร้างไฟล์ `.project/` รัน `git check-ignore -v .project/<ไฟล์>` ซึ่งต้องไม่พบผล และ `git ls-files .project/` ซึ่งต้องเห็นไฟล์ครบ ถ้าถูกซ่อนให้เพิ่ม `!.project/` กับ `!.project/**` แล้วตรวจใหม่

## Optional Memory Checks

- ถ้ามี `scripts/memory-audit/memory_audit.py` ให้รันหนึ่งครั้ง: exit 1 หยุดแก้ความจำก่อนเริ่มงาน, exit 2 รายงานคำเตือนแล้วทำต่อได้
- ถ้ามี `.project/qaqc-scan.md` ให้อ่านสถานะรายหมวด ปัญหาค้าง แผนแก้ QQF และประวัติรอบล่าสุด แล้วรายงานจำนวนหมวดและงานค้าง

## Team Claim Gate

ใช้เฉพาะโปรเจกต์ที่เปิดงานทีม/ownership routing:

1. ระบุ staff id และหา registered folder ด้วย `hermes_worktree_route.py` ถ้าหายให้หยุด ห้ามสร้างหรือใช้โฟลเดอร์คนอื่น
2. อ่าน `.project/OverviewProgress.md` และ `.project/decisions.md`
3. รัน `claim list` ตรวจ path overlap; claim หมดอายุบนไฟล์เสี่ยงต้องถามเจ้าของ
4. ตรวจ dirty บนโฟลเดอร์ตัวเอง ถ้าไม่ใช่งานเดียวกันให้หยุด
5. ก่อนเขียนทุกงาน รัน `claim acquire` โดยผูก staff/project/folder/branch/task/paths/expires_at
6. หลัง commit หรือส่งต่องาน รัน `claim release` หรือเปลี่ยนเป็น handoff

Claim บอกว่าใครตั้งใจแก้อะไร ส่วน Git status และ branch บอกว่าแก้อะไรไปแล้ว ต้องเทียบทั้งสองเสมอ

## Version History

ประวัติเต็มก่อน v2.4 เก็บใน Git history ของไฟล์หลัก รุ่นที่มีผลต่อการทำงานปัจจุบันให้อ่านจาก frontmatter และ Changelog รุ่นล่าสุดในไฟล์หลัก
