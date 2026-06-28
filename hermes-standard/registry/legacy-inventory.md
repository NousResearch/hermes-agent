# ทะเบียนของเก่า — เก็บ / รื้อ / เข้ากรุ (สำรวจจริง 2026-06-27 บนเครื่อง Mac)

> หลักการ: ไม่ลบทันที · ตัวที่จะเลิกใช้ให้ "เข้ากรุ (archive)" ก่อน เผื่อมีงานใช้อยู่
> การตัดสินรายตัวแบบละเอียด (โดยเฉพาะ hook ที่ควรรวม) ทำจริงในคลื่น 2 (ต้องอ่านโค้ดทีละตัว ไม่เดา)

## hook ทั้งหมด 33 ตัว (`~/.claude/hooks/`)

### กลุ่ม "ด่านบล็อกจริง" — เก็บ เป็นแกนของด่านกลาง (คลื่น 2/3)
- pre-action-danger-command-gate.py (บล็อกคำสั่งอันตราย 172 ครั้ง)
- pre-action-protected-gate.py (กันแก้ไฟล์ต้องห้าม 23 ครั้ง)
- enforce-codex-review.py (กันปิดงานก่อนรีวิว 28 ครั้ง)
- enforce-relay-flow.py · enforce-merge-permission.py · enforce-project-scope.py · enforce-no-guess-pretool.py

### กลุ่ม "ตรวจ/เตือน + ฉีดบริบท" — เก็บ แต่คลื่น 2 ทบทวนว่าตัวไหนซ้ำ/ควรรวม
- enforce-spec-evidence.py · enforce-design-score.py · enforce-canary.py · enforce-no-guess.py
- enforce-numerical-gate.py · enforce-comply-tracking.py · enforce-continue-mode.py · enforce-module-scope.py
- enforce-prompt-compile.py · enforce-tech-glossary.py
- inject-project-map.py · inject-rules-one-page.py · inject-rules-reminder.py
- validate-all-stop.py · validate-keyword-match.py · validate-preflight-card.py · validate-response-contract.py
- validate-thai-language.py · validate-ui-task-visual-proof.py · validate-value-framing.py
- verify-claims-before-send.py · repeat-violation-alert.py · detect-claude-design-bundle.py
- validate-no-pushback.py (ตรวจซ้ำกับ response-contract? — ทบทวนคลื่น 2)
- recall-user-memory.sh (ตัวจำความจำ — เก็บแน่นอน)

### กลุ่ม "รื้อใหม่"
- ai-fail-stats.py (ตัวจับคำด่าแบบแคบ) → ยกไปทำใหม่ในคลื่น 4 (วงจรเรียนรู้)

## กฎกลาง 9 ไฟล์ (`~/.claude/rules/`)
- cross-check.md · design-score-gate.md · hermes-system.md · language.md · post-deploy-canary.md
- prompt-compliance.md · spec-evidence-gate.md · verification.md · workflow.md
- → หลอมเนื้อหลักเข้า `rules/central-block.md` (โซนกลาง) · ตัวเต็มเก็บอ้างอิงต่อ

## curse-tracker ของเก่า (`~/.claude/ai-fail-stats/`)
- ไฟล์ลอย 7-9 ชิ้น (curse-keywords.json, issues.json, build_dashboard.sh, curse-tracker.html, log.jsonl, view.py, counts.json)
- → รื้อเข้าคลื่น 4 ให้ครบวงจร (เก็บ keywords/issues เดิมมาใช้ต่อ)
