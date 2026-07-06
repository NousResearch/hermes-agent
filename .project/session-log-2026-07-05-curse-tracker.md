# Session Log — Curse Tracker v2 + ปิดตัวเตือนกวน (2026-07-05 → 07-06)

> memory-schema: v1.2 · เขียนแยกไฟล์เพราะ OverviewProgress.md/decisions.md กำลังถูกงานคู่ขนาน (JARVIS เสียง) แก้ค้าง ห้ามทับ

## เป้าหมายรอบนี้
1. ตรวจ + ซ่อมระบบติดตามคำด่า (curse tracker) ที่ก่อนหน้าจับได้แค่ 3 คำ
2. ปิดแบนเนอร์เตือนกฎที่เด้งกวนทุกข้อความ ให้ครอบทุก project

## ทำอะไรไป (ผ่าน Use AI Relay: Opus วางแผน · Codex เขียน · Grok ตรวจ)

### งาน A — Curse Tracker v2 (เสร็จ + merge เข้า main แล้ว)
รากปัญหา: hook `~/.claude/hooks/ai-fail-stats.py` ฝังคำตายตัว 3 คำ ไม่อ่านไฟล์ `curse-keywords.json` → จับพลาด >95%

Changed-files (commit `c649d11e4` บน origin/main · 7 ไฟล์ 1306+ บรรทัด):
| file | changed_by | reason | verification |
|---|---|---|---|
| hermes-standard/learning/hooks/ai-fail-stats-v2.py | Codex | ตัวจับ v2 อ่านสมุดคำด่า (fallback 3 ชั้น) + ห่อ try/except กันแชทพัง | verified: pytest 13/13 |
| hermes-standard/bin/curse_track.py | Codex | เพิ่ม ingest (กันนับซ้ำ) + report แยกโปรเจกต์ + HTML | verified: 18/18 + ingest จริง 56 เหตุการณ์ |
| hermes-standard/learning/curse-keywords.json | Codex | schema v2 (targets/generic/disabled) ย้าย "ห่า"/"มั่ว" เข้า disabled | verified: false positive หาย |
| hermes-standard/bin/install_curse_hook.sh | Codex | ติดตั้งแบบสำรอง+กู้กลับ | verified: test 2/2 |
| tests 3 ไฟล์ | Codex | คลุม detect/ingest/install | verified: รวม 19/19 ผ่าน |

Grok review = verified (ปิดงานได้) · ชี้ 2 จุดสำคัญ (แชทพังตอนเขียนไฟล์ + false positive) → แก้จบในรอบเดียว (CT-FIX)

### งาน B — ปิดตัวเตือนกวน (เสร็จ)
- รากปัญหา: hook `inject-rules-reminder.py` อ่าน `~/.claude/hooks-violations.log` (70,089 บรรทัดสะสมตั้งแต่ 24 เม.ย.) แล้วเด้งแบนเนอร์ "critical 586 ครั้ง" ทุกข้อความ · ไม่ใช่ limit จริง แค่ตัวนับ log เก่าที่ไม่เคยล้าง
- แก้: เพิ่ม `"INJECT_REMINDER_DISABLED": "1"` ใน `env` ของ `~/.claude/settings.json` (ครอบทุก project)
- verified: ยิง hook พร้อมสวิตช์ → output ว่าง exit 0 · ด่านความปลอดภัยจริง (DANGER/PROTECTED gate) ยังอยู่ครบ
- มีผลเมื่อเปิดแชทใหม่ (settings env โหลดตอน start)

## Quality Gate
- pytest (venv ชั่วคราว): 19/19 ผ่าน — `hermes-standard/learning/tests/`
- gate-run ผ่าน relay: CT-I1/I2/I3 บันทึก ledger ครบ (`.hermes/ai-relay/ledger.md`)

## Deploy / Git
- curse work: commit `c649d11e4` → เข้า `origin/main` แล้ว (ยืนยัน: is-ancestor + ไฟล์อยู่ใน tree origin/main)
- main = origin/main (0 ahead / 0 behind)
- ⚠️ branch `feature/curse-tracker-v2` ที่ push ไป fork = **ผิด** (มีแต่ commit P4 a9159ed8b ไม่ใช่งานคำด่า) เพราะ commit จริงไปลงตอน HEAD สลับ branch · ลบทิ้งได้ ไม่กระทบอะไร (งานอยู่ main แล้ว)

## งานค้าง / ส่งต่อ
1. **ติดตั้ง hook v2 ลงเครื่อง** (งานคน/ต้องเจ้าของเห็น): รัน `hermes-standard/bin/install_curse_hook.sh` — hook จริงในเครื่องยังเป็น v1 (สำรองของเดิมให้อัตโนมัติ)
2. ลบ branch fork ที่ผิด: `git push fork --delete feature/curse-tracker-v2` (ถ้าต้องการความสะอาด)
3. แผนรวมระบบคำด่า: นี่คือ "ปัญหาที่ #1" ในแผนมาตรฐานกลาง Hermes 5 ชั้น (ชั้น 5 วงจรเรียนรู้) · ยังเหลือ: ตั้งเวลาวิเคราะห์อัตโนมัติทุก 3 ชม. + dashboard E-Office + แจกข้าม VPS

## ความเสี่ยงที่เหลือ
- คำด่าใหม่ที่ยังไม่เคยพิมพ์ = เติมใน JSON เอง (ไม่ต้องแก้โค้ด)
- VPS ยังไม่ติดตั้ง (รอบนี้ทำเครื่องโน้ตบุ๊กก่อน)
- ตัวจับใหม่ยังไม่ทำงานจริงจนกว่าจะรันตัวติดตั้ง

## Evidence
- timestamp: 2026-07-06 · host: Rattanasaks-MacBook-Pro.local
- cwd: Tech Tools/Hermes Agent
- commands: git merge-base --is-ancestor c649d11e4 origin/main (ใช่) · git ls-tree origin/main (ไฟล์ครบ 3) · pytest 19/19 · hook test exit 0
