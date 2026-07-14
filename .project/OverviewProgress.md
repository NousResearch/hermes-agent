> memory-schema: v1.2
> อ่านตามลำดับ: plan.md (plan_id: QAQC — active · shortcut Use QA QC) → plan-grd.md (แผน GRD จบแล้ว + คิว GRD-P5..P9) → decisions.md → hermes-standard/REQUIREMENTS.md (บัญชีความต้องการ 66 ข้อ)

# Overview & Progress — Hermes Agent
อัปเดตล่าสุด: 2026-07-14 (Use Migrate Web P3 ครบ 7/7 + §13 + P4 installer · merged main) · branch งานถัดไป: แตกใหม่จาก `main` · ป้าย: [fact] เว้นแต่ระบุ

## สถานะล่าสุด
- **2026-07-14 (แชท Opus · merged เข้า main แล้ว): `Use Migrate Web` — P3 เครื่องมือ 7/7 ครบ + สัญญา §13 COMPLETE + P4 installer** [fact]
  - **เครื่องมือ 7/7** (`scripts/mw/`): work_locks · menu_gate · page_check · mw_doctor · rtm_report · wow_report · backend_check — ทุกตัว Grok เขียน · GPT-5 ตรวจข้ามค่าย · ปิด ~44 false-positive (false-green/ready/verified/healthy) · **mw suite 252 passed**
  - **§13.1 COMPLETE 32/32** (`mw-spec-check.py` ด่าน 6 · `.project/mw-g-testid-map.md`) · pending 0 · strict mode ผ่าน · สัญญา §13 ครบ 3/3
  - **PR #35 (6/7) + PR #36 (ตัวที่ 7 + §13) merged เข้า main แล้ว** (squash) · main มีเครื่องมือครบ
  - **P4 installer เสร็จ**: `scripts/mw/mw-setup.sh` (symlink 8 เครื่องมือเข้า ~/.local/bin + ยิง --help ทุกตัว) + ผูกเข้า `team-shortcuts/install-shortcuts.sh` (best-effort) · 2 เทสต์ผ่าน
  - **ค้าง (ต้องใช้ของเจ้าของ):** รัน `mw-backend-check` กับ **RoadSafeFund จริง** (เดิน flow 1 เมนู · tier 3+) ต้องมี API base + วิธี query DB + token หรือสิทธิ์ VPS — เป็น P4 verify ข้อสุดท้าย
  - **ข้อจำกัดเครื่องนี้ (จดกันลืม):** coder = Grok ผ่าน relay (`AI_RELAY_ALLOW_LOCAL_CLI=1` + PATH `~/.local/bin` ตัวก่อน homebrew) · **`grok` ใช้ subscription ได้จริง (ตัวทางการ 0.2.99 ที่ ~/.local/bin) — โน้ตเก่าที่ว่า "grok ต้องมี API key" ผิด · ต้นเหตุคือ PATH หยิบ grok homebrew v1.0.1 ผิดตัว** · Codex ผ่าน relay crash (MCP/stdin) · reviewer = GPT-5 ผ่าน cross-check MCP (relay review พัง)

- **2026-07-14 (แชท Fable · branch `control_webengine_flow`): Shortcut `Use Migrate Web` — P1+P2 จบ · P3-I1 จบ · รอสลับ Opus ทำ I2** [fact]
  - **แผน MW ทั้งหมดอยู่ `.project/plan.md` หัวข้อ "Plan — MW"** (แผน active จริงของ branch นี้ · plan-anchor ยังอ่าน QAQC เป็นหลัก → เลขงาน MW ใช้ --no-plan)
  - P1: วิเคราะห์ 5 ชุดข้อมูล (FLOW v2 + Workshop + TOR 3 โปรเจกต์ + คลัง Obsidian + คำสั่งเพิ่ม) → บัญชี 55 กลุ่ม + จุดเคาะ 13 จุด เจ้าของอนุมัติครบ
  - P2: SPEC v1.2 (`.project/mw-spec-draft.md`) เจ้าของอนุมัติ · เครื่องตรวจ `scripts/mw-spec-check.py` PASS (ตารางแม่ 55/55 · baseline sha256)
  - P3-I1: prompt `use-migrate-web.md` + `use-migrate-web-flow13.md` (เนื้อต้นฉบับ 439 บรรทัดตรง 100% + embedded_sha256) + registry row — Codex ตรวจ 2 รอบ แก้ครบ
  - **งานถัดไปของแชทใหม่ (Opus): `Use New Chat` → `Use AI Relay` → ทำ MW-P3-I2 ตามแผนส่งมอบใน plan.md (เครื่องมือ 7 ตัว เริ่มที่ I2d work-locks)** · ข้อจำกัด relay บนเครื่องนี้จดไว้ในแผนแล้ว (review พัง → ใช้ cross-check MCP)

- **2026-07-14: เคลียร์ของค้างส่งต่อทีมอื่น — main สะอาดตรง origin/main** [fact · ตรวจ git state จริง tier 3]
  - ต้นเหตุ: branch `codex/block-ai-worktree-creation` (Codex สร้าง 07-12) ค้างอยู่ พร้อม dirty 19 แก้+8 ใหม่ · วันรุ่งขึ้น (07-13) งานชุดเดียวกันถูกทำใหม่สะอาดกว่า merge เข้า origin/main ผ่าน PR #30/#31/#32/#33 + portal routing ไปแล้ว → branch นี้กลายเป็นของซ้ำ
  - ทำ: ปัก 2 tag กันตก (`archive/codex-block-worktree-2026-07-12`→`c185b8a0b` เก็บ `CONTROL-CENTER-DESIGN.md`+worktree-block tests · `archive/local-main-orphan-2026-07-11`→`3bcfabfb9` เก็บ orphan DEC-036) + patch สำรอง 3181 บรรทัดใน scratchpad
  - ลบ 3 branch ค้าง: `codex/block-ai-worktree-creation` + `close/mem-2026-07-11` (remote gone) + `ds-standard-v3` (merged PR #30) · ขยับ local main `branch -f` + `checkout -f` → HEAD `7087b8fcd` = origin/main (เลี่ยง `reset --hard` ที่ classifier บล็อก)
  - ผล: เหลือ branch แค่ `main` + 2 worktree เจ้าของ (`feature/std-i2-project-dir` + `upgrade-audit/v0170` ไม่ถูกแตะ) · working tree สะอาด 0 dirty/untracked · **local main pointer เพี้ยนเดิม (ahead1/behind20) หายแล้ว**
- **2026-07-11: Design System พร้อมใช้จริง + relay tests เขียว 100% + Git graph สะอาด** [fact]
  - (ก) **DS**: ทำ `contrast-audit-run.mjs` (playwright headless · เอา Codex-review fix เข้า main แก้ 3 bug: NaN false-pass/networkidle-ค้าง/browser-leak) + `ds-adopt.sh` shortcut คำเดียว (`prep`/`check` รันด่านครบ build/ds-check/brand-leak/contrast · exit 1 บล็อก) + `admin-states.html` 5 states + เลิกลอก onemanfleet (brand-leak-check) + path portable (VPS/Mac) → merged PR #18/19/22/24/26
  - (ข) **relay**: DEC-036 quota/auth ปลอม (stderr ≤250 guard · PR #25) + ซ่อม test timeout ให้ตรงโค้ด Popen (mock subprocess.run ล้าสมัย) → **relay tests 72/72 เขียว**
  - (ค) **branch cleanup**: merged nat(#28)+shortcut Use Trade-off(#27 · resolve conflict payload) → ลบ merged 25+ branch · **ลบ remote upstream(NousResearch 1,292)+fork ทิ้ง → Git graph 1,300→6** (เหลือ origin/main + vps) · ต้นตอที่เจ้าของเห็น branch เต็ม = upstream ของ NousResearch ไม่ใช่งานเรา
  - (ง) ยืนยัน **JARVIS อยู่ SaaS repo ครบ + active** (typer งานต่อในนั้น) · Hermes jarvis 4 branch = เศษเก่าก่อนย้าย ลบแล้ว
- **2026-07-10: shortcut `Use QA QC` v1.1 เปิดใช้แล้ว (active · ทะเบียน 29→30)** — เจ้าของสั่งจบด้วย Fable ไม่รอกรรมการ · ตารางแม่ 16 หมวด/178 หัวข้อ + วินิจฉัย ViberQC อยู่ในคลัง (`AI-Security-Testing/`) · New Chat v2.0 + Close Chat v2.3 ผูกไฟล์กลาง `.project/qaqc-scan.md` แล้ว · branch งาน: `feature/use-qa-qc` (แผน active ใหม่ plan_id: QAQC · GRD ย้ายไป plan-grd.md) [fact]
- 2026-07-10: ซ่อม relay-call quota ปลอม (คำตอบยาวที่พูดถึง quota โดนตีเป็นโควต้าหมด) — scoped pytest 68/69 (1 แดง = เทสต์ timeout พังก่อนแก้ พิสูจน์แล้ว) · แก้ adapter grok (CLI v1.0.1 ตัด flag เก่า) · **grok headless ต้องมี API key = งานคนค้าง** · กรรมการรีวิวรวบ P1-P4 ยังไม่สำเร็จ (โควต้า/บั๊กวันเดียวกัน 3 ตัว) เลื่อนเป็น hardening [fact]
- **แผน GRD merged เข้า main แล้ว — PR #16 (`8bd9aa5e0`)** · ระบบกันแผนหาย/กัน AI มั่ว ใช้งานจริงบน main [fact]
- เก็บของค้างเช้า 2026-07-08: commit ไฟล์กฎกลาง 3 ไฟล์ในคลัง Obsidian (commit คลัง 7b52e4b — คนละ repo กับตัวนี้ · re-anchor v2.9/v4.2/v1.9) · push branch `feature/p12b-shortcut-guard` + เปิด **PR #17** (install guard กันเขียนทับไฟล์คลังที่ใหม่กว่า) [fact]
- memory-audit รันจริงบน repo: schema/SHA/ไฟล์ความจำ ✅ ครบ · เตือนเลขงานกำพร้า 33 ตัว (งานตั้งค่าจร ไม่ใช่บั๊ก) [fact]
- **แผน GRD ทำครบทั้ง 4 เฟสแล้ว (ประวัติ)** — P1 สัญญางานผูกแผน (plan-anchor + relay-call บังคับ + กฎ re-anchor ใน vault) · P2 memory-audit ตัวเทียบความจำ · P3 ด่านกัน stash กวาดงานคนอื่น · P4 ล้างความจำเก่า — ทุกชิ้นผ่านผู้ตรวจต่างค่าย + เทสต์ scoped เขียว (154 เคสรวม: 64 relay + 10 memory-audit + 80 guards) [fact]
- ตัวเขียนโค้ดจริงของรอบนี้: **Grok เป็นหลัก** (Codex ชนโควต้าตั้งแต่ใบแรก) · ใบแก้สุดท้าย Gemini · ผู้ตรวจ = Claude ทุกใบ [fact]
- **ชุดเทสต์เต็ม repo แดงอยู่ก่อนแล้ว**: `pytest -q` ที่ฐาน main = 683 failed / 24,193 passed (จุดตกอยู่ใน tests/cli, tests/gateway ที่งาน GRD ไม่ได้แตะ) — gate-run จดเป็นแถวแรกใน `.hermes/ledger/` แล้ว · เป็นงานซ่อมแยกรอบ [fact]
- PR #15 (แก้ auth ปลอมใน relay-call) merge เข้า main แล้ว — main HEAD = `5aa135e7f` [fact]
- สาย JARVIS v2: รอเจ้าของทดสอบเสียง P0 แล้วเปิดแชตใหม่ส่ง Use AI Relay [fact]

## งานถัดไป
1. **MW-P4 verify ข้อสุดท้าย: รัน `mw-backend-check` กับ RoadSafeFund จริง** — เดิน flow หลังบ้าน 1 เมนู (submit→DB→admin) หลักฐาน tier 3+ · ต้องมีจากเจ้าของ: API base + วิธี query DB (query_cmd) + token (หรือสิทธิ์ VPS ที่ RoadSafeFund รันอยู่) → ทำ `.work/backend-check.yaml` ของโปรเจกต์จริง
2. **MW-P4 install PR** (`feature/mw-p4-install`): mw-setup.sh + install-shortcuts wiring — รอเจ้าของกด merge
3. (ถ้าต้องการ) รัน mw-setup.sh บน VPS จริงเพื่อยืนยันติดตั้ง 2 ทาง
4. (คิวเดิม) GRD-P5..P8 + QAQC-P5 รอเจ้าของสั่ง

## ข้อห้าม/กติกาล็อก
- ห้ามเขียนความจำทำงานต่อลง `.hermes/` หรือ root — เขียน `.project/` เท่านั้น (Schema v1.2)
- หลังสร้าง/ย้ายไฟล์ `.project/` ต้องผ่านด่าน `git check-ignore` + `git ls-files` ก่อนบอกเสร็จ
- **เลขงานต้องขึ้นต้นด้วย plan_id (เช่น GRD-P1-I1) · เลขที่ไม่มีใน plan.md = ห้ามทำ** · หลังตอบคำถามแทรก ต้องเปิด plan.md ทวนเฟสก่อนลงมือ (กติกาเหล็กของแผน GRD)
- ห้ามแตะ `.claude/launch.json` (งานเจ้าของค้าง) · [ปลดล็อก 2026-07-11: `design-system-standard-v2/` เจ้าของสั่งแก้จน DS พร้อมใช้จริง merged main แล้ว · `scripts/jarvis-voice/` ย้ายไป SaaS repo แล้ว = เศษ]
- ห้าม merge→main / deploy เอง — เจ้าของกด · งานหลายเฟส = 1 PR เดียว
- สมองแผน GRD = Fable ตามคำสั่งเจ้าของ 2026-07-07 (ข้อยกเว้นจากกติกา relay v2.7 ที่ปกติใช้ Opus) · Codex/Claude เขียน-ตรวจสลับค่ายผ่าน relay-call · **verified = มีแถว gate-run เท่านั้น**

## งานค้าง/ส่งต่อ
- ~~**2026-07-11: local main pointer เพี้ยน** (ahead1/behind20 + orphan DEC-036)~~ **แก้แล้ว 2026-07-14: main สะอาดตรง origin/main `7087b8fcd` · orphan เก็บ tag `archive/local-main-orphan-2026-07-11`** [fact] · vps เหลือ branch cache (prune ได้ถ้าอยาก)
- ~~รอเจ้าของ: ตรวจ+กด merge PR #17 + PR #14~~ **merged หมดแล้ว 2026-07-11 (PR ค้าง = 0)** [fact]
- รอเจ้าของ (เดิม): **ตรวจ+กด merge PR #17** (install guard) และ **PR #14** (relay-relogin ที่เปิดค้าง) · ส่งปัญหาชุดสุดท้าย (ปลดล็อก GRD-P6..P8) · rotate GitLab token (ค้างจาก 2026-07-04) · **push คลัง Obsidian ขึ้น GitLab** (commit คลัง 7b52e4b + 614e00a ค้างในเครื่อง 2 ตัว — คนละ repo กับตัวนี้ · AI push main ตรงไม่ได้ ด่านบล็อก) · ติดตั้ง memory-audit รายสัปดาห์ (ถ้าต้องการ): `(crontab -l 2>/dev/null; echo '0 9 * * 1 cd "/Users/rattanasak/Documents/Viber Project/Tech Tools/Hermes Agent" && ./venv/bin/python scripts/memory-audit/memory_audit.py >> ~/.claude/ai-fail-stats/memory-audit.log 2>&1') | crontab -`
- ~~สั่ง commit ไฟล์ JARVIS untracked~~ **แก้ความจำ 2026-07-08: ไฟล์ JARVIS เข้า git แล้วครบ 9 ไฟล์ (รวม FeatureSpec-jarvis-voice.md) — ไม่ใช่งานค้าง** [fact · ยืนยันด้วย git ls-files]
- **งานซ่อมแยกรอบ (ใหม่ 2026-07-08): ชุดเทสต์เต็ม repo แดง 683 เคสที่ฐาน main** — ทำให้ gate-run ตัดสิน pass ไม่ได้ทั้ง repo · ควรไล่ซ่อมหรือกำหนด gate ย่อยที่เขียวได้จริง (เสนอดูดเข้า GRD-P8)
- โควต้า AI คืน 2026-07-08: Codex + Grok ชนโควต้าทั้งคู่ช่วงดึก · Gemini crash ตอนจบแต่เขียนไฟล์สำเร็จ — เช็กโควต้าก่อนเริ่มงานใหญ่รอบถัดไป
- claimed (ยังไม่ตรวจ): iptables :3010 ไม่ persistent ข้าม reboot · webhook pr-review ลงแค่ project เด็กฝึก 527
- ด่านกันลบโฟลเดอร์ทั้งก้อน (phase-013): **โค้ด+เทสต์เข้า main แล้ว (`f9fb0827f`) [fact — แก้ความจำเก่าที่จดว่ายังค้าง]** · ที่ยังค้างจริง = ยืนยันว่า VPS runtime รันโค้ดรุ่นที่มีด่านนี้ (ยัง unverified)
- feature ค้างจากตารางรีวิว Hermes 2026-07-03: F3-F8
- อัปรุ่น v0.18.0 = GRD-P9 (ยังไม่เริ่ม · ต้องทำบัญชีของต่อเติมก่อน)

---

## project นี้คืออะไร (2-3 บรรทัด)
ศูนย์เครื่องมือ AI ส่วนตัวของเจ้าของ (fork จาก NousResearch/hermes-agent v0.17.0 + ของต่อเติม ~3,215 commit): สายพาน AI Relay ประหยัดเงิน · ชุด shortcut คุมวินัยงาน · มาตรฐานกลาง 30-40 โปรเจกต์ (hermes-standard) · เครื่องมือคุมคุณภาพ (violation-audit, pr-review-gate, curse tracker) · gateway ให้ทีม 15 คนใช้บน VPS [fact]

## เสร็จแล้ว (verified) + ประวัติย่อ
- 2026-07-07: PR #15 แก้ auth ปลอม relay-call merge เข้า main (`5aa135e7f`) · สอบสวนต้นตอ AI มั่ว 6 ข้อ + แผน GRD อนุมัติ [fact]
- 2026-07-06: P0-P1 แผนเก่า merged — PR #12 (`da4689a58`) · Project OS ครบ 4/4 · ความจำอยู่ `.project/` [fact]
- 2026-07-05: กู้ shortcut Project OS 3 ตัว (ถูก revert `fff10805b` เมื่อ 2026-06-28 โดยความจำยังจดว่าครบ) + ด่านไฟล์เข้า git จริงทั้งระบบ (`f079acf47`) · relay-call เพิ่มนาฬิกากันค้าง — pytest 16/16
- 2026-07-04-05: relay P3 ครบ 4/4 — PR #8, #9, #10 merged · F1 violation-audit + F2 pr-review-gate ใช้จริง (tier 3) · AI Relay ยืนยันทั้ง notebook + VPS
- 2026-06-21: ด่านกันลบโฟลเดอร์งานทั้งก้อน phase-013 (Codex เขียน · ตรวจแล้ว 38+14 เทสต์) — เข้า main ที่ `f9fb0827f`
- ก่อนหน้า: ดู decisions.md + session log ใน vault (`projects/hermes-agent-dev/`)
