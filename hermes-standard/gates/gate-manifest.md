# Gate Manifest — จัดระเบียบ hook 33 ตัว (คลื่น 2 · คำตัดสิน)

> นี่คือ "คำตัดสินบนกระดาษ" ว่า hook เดิมตัวไหนเก็บ/รวม/เข้ากรุ
> การย้ายจริงใน `~/.claude/hooks/` = ขอบเขตกลาง ติดด่าน relay → ทำตอนเจ้าของปลดด่านบนเครื่องนั้น (ยังไม่ทำในรอบนี้)

## กลุ่ม A · ด่านบล็อกจริง → เก็บ เป็นด่านกลางของมาตรฐาน (7)
pre-action-danger-command-gate · pre-action-protected-gate · enforce-codex-review ·
enforce-relay-flow · enforce-merge-permission · enforce-project-scope · enforce-no-guess-pretool

## กลุ่ม B · ตรวจ/เตือน + ฉีดบริบท → เก็บ แต่ "รวม" ให้เหลือชุดเดียว (≈23)
- ฉีดบริบท (รวมเป็น 1): inject-project-map · inject-rules-one-page · inject-rules-reminder
- ตรวจคำตอบ (รวมกลุ่ม): validate-thai-language · validate-response-contract · validate-value-framing ·
  validate-preflight-card · validate-no-pushback · validate-keyword-match · validate-all-stop ·
  validate-ui-task-visual-proof · verify-claims-before-send
- ตรวจกระบวนการ (รวมกลุ่ม): enforce-numerical-gate · enforce-prompt-compile · enforce-comply-tracking ·
  enforce-spec-evidence · enforce-no-guess · enforce-tech-glossary · enforce-module-scope ·
  enforce-design-score · enforce-canary · enforce-continue-mode · repeat-violation-alert · detect-claude-design-bundle

## กลุ่ม C · เก็บแน่นอน (1)
recall-user-memory (ตัวจำความจำ)

## กลุ่ม D · รื้อใหม่ (1)
ai-fail-stats → แทนด้วย `bin/curse_track.py` (คลื่น 4) แล้ว · ตัวเก่าเข้ากรุ

## ข้อเสนอรวมด่าน (ลดจาก 33 → ~12)
- 7 ด่านบล็อกจริง (กลุ่ม A) คงไว้แยก
- กลุ่ม B รวมเป็น 3 ตัว: `inject-context` · `validate-answer` · `enforce-process`
- recall คงไว้ · curse_track มาแทน ai-fail-stats
ผลลัพธ์เป้าหมาย: เหลือ ~12 ตัว ชัดเจน ไม่ซ้ำ · **ต้องอ่านโค้ดแต่ละตัวยืนยันก่อนรวมจริง (ยังไม่ทำ)**
