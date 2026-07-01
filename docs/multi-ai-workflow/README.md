# Multi-AI Workflow

เอกสารนี้คือกติกากลางสำหรับให้ AI หลายตัวทำงานร่วมกันในโปรเจกต์เดียวกัน โดยไม่ต้องเล่าบริบทซ้ำทุกครั้ง และไม่ให้ AI เขียนทับงานกันเอง

## เป้าหมาย

- ให้ planner, coder, reviewer, QA, และ context keeper ทำงานต่อกันได้
- ให้ Codex, Claude Code, Qwen, Gemini, Cursor, Antigravity, และ AI ตัวอื่นอ่านคำสั่งชุดเดียวกัน
- ลด token โดยอ่านเฉพาะไฟล์ที่เกี่ยวข้องกับงาน ไม่โหลดทั้ง vault หรือทั้ง repo
- บังคับให้ทุกงานมี issue, owner, worktree, verification, และ handoff
- ใช้ได้ทั้ง local machine, VPS, และทีมที่ SSH เข้ามาทำงาน

## คำศัพท์

- `adapter` = ไฟล์เชื่อมบริบทให้ AI แต่ละตัวอ่าน เช่น `CLAUDE.md`, `QWEN.md`, `GEMINI.md`
- `issue registry` = สมุดทะเบียนติดตามงานใน `.hermes/issues/`
- `handoff` = บันทึกส่งต่องานให้ AI/คนถัดไปทำต่อได้
- `worktree` = โฟลเดอร์ทำงานแยก branch เพื่อไม่ให้หลาย AI แก้ไฟล์ชนกัน
- `verification` = หลักฐานตรวจจริง เช่น test, build, localhost, VPS, หรือ file check

## Source Of Truth

ลำดับแหล่งความจริงสำหรับ AI:

1. คำสั่งล่าสุดจากเจ้าของงาน
2. `AGENTS.md`
3. adapter เฉพาะ tool เช่น `CLAUDE.md`, `QWEN.md`, `GEMINI.md`, `.cursor/rules/*.mdc`
4. `.hermes/context.md`, `.hermes/active.md`, `.hermes/decisions.md`
5. `.hermes/issues/`
6. `.hermes/handoff.md`
7. HermesAgent Obsidian vault เฉพาะไฟล์ context ที่ถูกอ้างถึง

ห้ามโหลดทั้ง vault ถ้าไม่มีเหตุผลชัดเจน

## Workflow

```text
INTAKE -> PLAN -> OWNER APPROVAL -> CLAIM ISSUE -> IMPLEMENT -> VERIFY -> REVIEW -> HANDOFF -> MERGE/ARCHIVE
```

## Role Routing

| Role | เหมาะกับงาน | Output |
|---|---|---|
| Planner | วิเคราะห์ วาง architecture แตก phase | spec, issue list, risk |
| Implementer | เขียนโค้ด แก้ bug ทำ migration | code diff, test notes |
| Reviewer | ตรวจ diff, regression, security, maintainability | review findings |
| Verification | รัน test, build, localhost, VPS | evidence log |
| Context Keeper | สรุป handoff, decisions, active state | `.hermes/handoff.md`, issue update |
| Owner | อนุมัติ scope, phase, merge | decision |

## Issue Rules

ทุกงานที่ให้ AI ทำต้องมี issue ก่อนเริ่ม implementation ยกเว้นงานตอบคำถามสั้นๆ ในแชท

Issue ต้องมี:

- เป้าหมาย
- scope และ out of scope
- AI/คนที่รับผิดชอบ
- branch และ worktree
- ไฟล์ที่อนุญาตให้แตะ
- วิธีตรวจ
- % ทำได้ และ % เหลือ
- หลักฐานตรวจ

ใช้ template: `docs/multi-ai-workflow/templates/issue.md`

## Worktree Rules

รูปแบบชื่อ branch:

```text
ai/<issue_id>-<role>-<short-topic>
```

รูปแบบ path:

```text
../worktrees/<project-slug>/<issue_id>-<role>
```

ตัวอย่าง:

```bash
git worktree add ../worktrees/hermes-agent/phase-001-codex -b ai/phase-001-codex-control-plane
```

กติกา:

- หนึ่ง issue ต่อหนึ่ง worktree
- อย่าให้ AI สองตัวแก้ worktree เดียวกันพร้อมกัน
- ถ้าเป็น web/server ให้แยก port ต่อ worktree
- ห้าม commit `.env`, secret, database runtime, logs, หรือ cache

## Verification Rules

ห้ามบอกว่า 100% ถ้ายังไม่มีหลักฐานตรวจจริง

งานโค้ดควรตรวจอย่างน้อย:

- targeted test
- lint หรือ type check ถ้าโปรเจกต์มี
- build ถ้าแตะ production path
- localhost ถ้าแตะ web/app/server
- VPS ถ้าแตะ deployment, service, proxy, env, หรือ production flow

งานเอกสาร/protocol ควรตรวจ:

- ไฟล์มีอยู่จริง
- template มี field บังคับครบ
- ไม่มี secret
- ไม่มี placeholder ที่ทำให้ AI เดาต่อ
- health checker ผ่าน

## Model Routing

| งาน | แนะนำ |
|---|---|
| วางแผนซับซ้อน | Opus/Gemini reasoning สูง |
| เขียนโค้ดหลายไฟล์ | Codex, Qwen, Cursor |
| Review รอบแรก | Codex หรือ Claude Code |
| ตรวจ architecture ตามแผนเดิม | Planner เดิม เช่น Opus |
| สรุป handoff | รุ่นที่ประหยัดกว่าได้ ถ้ามี template ชัด |
| ตรวจ localhost/VPS | ตัวที่มี shell/browser access |

## Use AI Pair

`Use AI Pair` จับคู่ AI หนึ่งตัวเป็น coder และอีกตัวเป็น reviewer แบบ read-only
โดยมี Hermes หรือ VPS เป็นตัวกลางคุม flow ไม่ให้ AI คุยกันเองแบบไม่มีสมุดงานกลาง

## Use AI Relay

`Use AI Relay` คือโหมดประหยัด token: Hermes หรือ AI ตัวหลักวางแผนและตรวจ ส่วน Grok, Codex, Gemini หรือ Ollama เขียนโค้ดผ่านคำสั่งกลาง `relay-call` แล้วตรวจผลด้วย `gate-run`

คู่มือให้พนักงานเชื่อม Grok ผ่าน Google ID:

```text
docs/multi-ai-workflow/ai-relay-staff-setup.md
```

ติดตั้งคำสั่งในเครื่อง:

```bash
bash scripts/ai-relay/install-local.sh
relay-doctor
```

ถ้าเครื่องมี AI Relay อยู่แล้วและต้องการเพิ่มเฉพาะ Grok:

```bash
relay-add-grok --cwd .
relay-doctor
```

Hermes Agent pilot defaults:

| Role | AI |
|---|---|
| Coder | Codecode |
| Reviewer | Codex |

Required flow:

1. Owner chooses coder and reviewer.
2. Hermes proposes a branch.
3. Owner approves branch.
4. Coder writes a plan before code.
5. Owner approves plan.
6. Coder implements and verifies.
7. Coder writes a reviewer brief.
8. Hermes/VPS prepares a review packet from diff, brief, and evidence.
9. Reviewer checks read-only.
10. Passing review moves to GitLab Merge Request / CI gate.

Review packet hard gate:

- `coder-plan.md` must include `approved_by_owner: yes`.
- `coder-brief.md` must include non-empty `diff_summary`, `files_changed`,
  `commands_run`, `results`, and `review_focus`.
- If these fields are missing, the pair job is blocked as
  `blocked_missing_review_gate` and no reviewer packet may be used.

Private GitLab host:

```text
https://gitlab.dev.jigsawgroups.work/
```

Do not send whole private repositories or token values to reviewer prompts.

Propose a branch:

```bash
python3 scripts/multi_ai_workflow.py ai-pair branch \
  --project . \
  --issue-id pair-001-use-ai-pair \
  --task "Add Use AI Pair"
```

Create pair state:

```bash
python3 scripts/multi_ai_workflow.py ai-pair init \
  --project . \
  --issue-id pair-001-use-ai-pair \
  --task "Add Use AI Pair" \
  --coder-ai Codecode \
  --reviewer-ai Codex \
  --branch ai-pair/pair-001-use-ai-pair \
  --gitlab-host https://gitlab.dev.jigsawgroups.work/
```

Run the coder-plan phase automatically:

```bash
python3 scripts/multi_ai_workflow.py ai-pair run coder-plan \
  --project . \
  --issue-id pair-001-use-ai-pair \
  --execute
```

The selected coder must have a runnable adapter command. For `Codecode`, set:

```bash
export HERMES_AI_PAIR_CODECODE_COMMAND="your-codecode-command --print"
```

If the adapter is missing, the pair job is blocked as
`blocked_missing_coder_runtime`. Do not fall back to manual prompt forwarding.

## Health Check

ตรวจว่าโปรเจกต์พร้อมสำหรับ multi-AI workflow:

```bash
python3 scripts/multi_ai_workflow_check.py --project . --format text
python3 scripts/multi_ai_workflow_check.py --project . --format json
```

ผลลัพธ์ `OK` แปลว่าไฟล์ควบคุมหลักครบ ไม่ได้แปลว่า feature หรือ server ทำงานแล้ว ต้องตรวจตาม issue อีกชั้นเสมอ

## CLI Usage

เริ่มใช้ workflow ในโปรเจกต์ใหม่:

```bash
python3 scripts/multi_ai_workflow.py init --project /path/to/project
python3 scripts/multi_ai_workflow_check.py --project /path/to/project --format text
```

## Opus 4.8 Planner Routing

เมื่อให้ Opus 4.8 เป็นตัววางแผนหลัก ให้ Opus เขียนแผนลงไฟล์ในโปรเจกต์ เช่น:

```text
.hermes/plans/opus-phase-001.md
```

ใช้ template:

```text
docs/multi-ai-workflow/templates/opus-plan.md
```

หลัง Opus เขียนแผนเสร็จ ให้รัน:

```bash
python3 scripts/multi_ai_workflow.py route \
  --project . \
  --plan-file .hermes/plans/opus-phase-001.md \
  --write
```

ระบบจะอ่าน plan แล้วเสนอ AI ถัดไปแบบจัดอันดับ:

| ตัวเลือก | เหมาะกับงาน |
|---|---|
| Codex App | Python, backend, CLI, tests, git, repo changes, code review |
| Qwen on Cursor | React, TypeScript, UI, component, styling, editor-driven work |
| Gemini on Antigravity | browser inspection, UX flow, multimodal, large-context exploration |

ผลลัพธ์จะถูกบันทึกที่:

```text
.hermes/routes/<plan-file-name>.json
```

ไฟล์นี้มี:

- AI ที่แนะนำเป็นอันดับหนึ่ง
- เหตุผลที่เลือก
- เปอร์เซ็นต์ความเหมาะสมของทุกตัวเลือก
- คะแนนดิบและสัญญาณที่ match
- prompt ส่งต่อให้ AI ตัวถัดไป

ตัวอย่าง:

```bash
python3 scripts/multi_ai_workflow.py route \
  --project . \
  --plan-file .hermes/plans/opus-phase-001.md \
  --format text \
  --write
```

ความหมายเชิงใช้งาน:

- Opus = planner หลัก
- router = คนจัดคิวว่าใครควรทำต่อ
- Codex/Qwen/Gemini = executor หรือ reviewer ตามชนิดงาน
- เจ้าของงานไม่ต้องเลือกเองทุกครั้ง แต่ยัง review recommendation ได้จาก `.hermes/routes/`

ตัวอย่างผลลัพธ์:

```text
Recommended: Codex App (codex_app)
Reason: เลือก Codex App เพราะสัญญาณในแผนตรงกับงาน: python, pytest, cli

Ranked options:
- Codex App: 100.0% (score 22; python, pytest, cli, scripts, backend)
- Qwen on Cursor: 36.36% (score 8; frontend, cursor)
- Gemini on Antigravity: 18.18% (score 4; browser, ux)
```

สร้าง issue:

```bash
python3 scripts/multi_ai_workflow.py issue create \
  --project . \
  --issue-id phase-001-example \
  --phase P1 \
  --title "Example implementation" \
  --owner-role "AI Workflow Architect" \
  --assigned-ai Codex \
  --reviewer-ai "Claude Code" \
  --goal "Create a tested example" \
  --scope "One issue file and verification evidence" \
  --out-of-scope "Dashboard and production deploy" \
  --verify-command "scripts/run_tests.sh tests/scripts/test_multi_ai_workflow_cli.py -q" \
  --localhost-check "not applicable" \
  --vps-check "not applicable" \
  --branch "ai/phase-001-example" \
  --worktree-path "../worktrees/project/phase-001-codex"
```

ให้ AI หรือทีม SSH claim งาน:

```bash
python3 scripts/multi_ai_workflow.py issue claim \
  --project . \
  --issue-id phase-001-example \
  --assigned-ai Codex \
  --branch "ai/phase-001-example" \
  --worktree-path "../worktrees/project/phase-001-codex"
```

อัปเดตผลตรวจและปิด issue:

```bash
python3 scripts/multi_ai_workflow.py issue update \
  --project . \
  --issue-id phase-001-example \
  --status verified \
  --done-percent 100 \
  --remaining-percent 0 \
  --evidence "tests passed; checker returned OK"
```

เมื่อ AI executor ทำงานเสร็จและต้องส่งกลับให้ Opus 4.8 review:

```bash
python3 scripts/multi_ai_workflow.py issue complete \
  --project . \
  --issue-id phase-001-example \
  --completed-by "Codex App" \
  --review-ai "Opus 4.8" \
  --evidence "tests passed; localhost verified"
```

คำสั่งนี้จะ:

- ตั้ง issue เป็น `ready_for_opus_review`
- ตั้ง `done_percent: 100` และ `remaining_percent: 0`
- เขียนหลักฐานตรวจจริงลง issue
- สร้าง review request ที่ `.hermes/review-requests/<issue-id>.md`
- ทำให้ `status` API แสดงว่ามีงานรอ Opus review

ดู comply รวมของ issue ทั้งหมด:

```bash
python3 scripts/multi_ai_workflow.py comply --project . --format text
python3 scripts/multi_ai_workflow.py comply --project . --format json
```

สร้าง worktree และ claim issue ในคำสั่งเดียว:

```bash
python3 scripts/multi_ai_workflow.py worktree create \
  --project . \
  --issue-id phase-001-example \
  --assigned-ai Codex \
  --branch "ai/phase-001-example" \
  --worktree-path "../worktrees/project/phase-001-codex" \
  --execute
```

ถ้าไม่ใส่ `--execute` คำสั่งจะเป็น dry-run = แสดงคำสั่งที่จะรัน แต่ยังไม่สร้าง worktree จริง

เขียน handoff:

```bash
python3 scripts/multi_ai_workflow.py handoff write \
  --project . \
  --task "Continue implementation" \
  --issue-id phase-001-example \
  --phase P1 \
  --latest-state "Implementation verified" \
  --next-agent "Reviewer" \
  --next-step "Review diff and evidence" \
  --verification-run "tests passed" \
  --localhost-result "not applicable" \
  --vps-result "not applicable" \
  --remaining-risk "none"
```

เตรียมส่ง issue ไป GitHub:

```bash
python3 scripts/multi_ai_workflow.py github issue \
  --project . \
  --issue-id phase-001-example
```

ค่าเริ่มต้นเป็น dry-run เพื่อไม่ยิง network หรือสร้าง GitHub Issue จริง ถ้าต้องการสร้างจริงให้เพิ่ม `--execute` และเครื่องต้อง login `gh` ไว้แล้ว

ดูสถานะ JSON สำหรับ AI/tool อื่น:

```bash
python3 scripts/multi_ai_workflow.py status --project .
```

ใน `status` จะมี `review_requests.pending_count` ให้คุณดูว่างานไหนรอ Opus 4.8 ตรวจอยู่

เปิด read-only localhost status server:

```bash
python3 scripts/multi_ai_workflow.py serve --project . --host 127.0.0.1 --port 8765
curl http://127.0.0.1:8765/health
curl http://127.0.0.1:8765/comply
curl http://127.0.0.1:8765/status
```

เปิดเฉพาะ `127.0.0.1` เป็นค่าแนะนำ เพราะปลอดภัยกว่า ไม่เปิดให้คนภายนอกยิงเข้ามาโดยตรง

ความหมายของ comply:

- `done_percent` = ทำได้กี่เปอร์เซ็นต์จากหลักฐานที่มี
- `remaining_percent` = เหลือกี่เปอร์เซ็นต์
- ห้ามกรอก `100/0` ถ้ายังไม่ได้รัน verification ตาม issue

## SSH Team Workflow

ทีมที่ SSH เข้าเครื่องหรือ VPS ควรทำตามลำดับนี้:

```bash
cd /path/to/project
python3 scripts/multi_ai_workflow_check.py --project . --format text
python3 scripts/multi_ai_workflow.py comply --project . --format text
git worktree add ../worktrees/<project>/<issue-id>-<role> -b ai/<issue-id>-<role>
python3 scripts/multi_ai_workflow.py issue claim --project . --issue-id <issue-id> --assigned-ai <tool-or-person> --branch ai/<issue-id>-<role> --worktree-path ../worktrees/<project>/<issue-id>-<role>
```

หลังทำงานเสร็จต้องกลับมา update issue พร้อมหลักฐานตรวจจริงก่อนส่งต่อ

## File Clutter Control

ระบบนี้ออกแบบให้ไฟล์ที่ active มีน้อย:

- `.hermes/issues/` = เฉพาะ issue ที่ยัง active หรือเพิ่งจบ
- `.hermes/plans/` = เฉพาะ Opus plan ที่ยังต้อง route หรือเพิ่ง route
- `.hermes/routes/` = เฉพาะ route recommendation ล่าสุดที่ยังต้องใช้
- `.hermes/review-requests/` = เฉพาะงานที่รอ Opus review
- `.hermes/archive/` = รวมงานจบแล้วแบบรายเดือน

ดูว่าจะ archive อะไรบ้างโดยยังไม่ลบไฟล์ต้นทาง:

```bash
python3 scripts/multi_ai_workflow.py compact --project .
```

archive งานที่จบแล้วจริง:

```bash
python3 scripts/multi_ai_workflow.py compact --project . --execute
```

สิ่งที่ compact ทำ:

- รวม issue ที่ `status: verified/reviewed/closed` และ `remaining_percent: 0`
- รวม plan/route ที่ไม่ใช่ README
- เขียนไปรวมที่ `.hermes/archive/workflow-YYYY-MM.md`
- ลบไฟล์ต้นทางที่ archive แล้ว เพื่อลดความรก

สิ่งที่ compact ไม่แตะ:

- `README.md`
- issue ที่ยัง open, claimed, in_progress, changes_requested, หรือ ready_for_opus_review
- secret, `.env`, runtime database, logs, cache

## Phase Complete Criteria

ถือว่า phase พร้อมส่ง review ได้เมื่อ:

- issue ที่เกี่ยวข้องทุกใบมี `status: verified`
- `done_percent: 100`
- `remaining_percent: 0`
- มี `evidence:` เป็นคำสั่งที่รันจริงและผลจริง
- `python3 scripts/multi_ai_workflow.py comply --project . --format text` แสดงภาพรวม `100/0`
- ถ้าเกี่ยวกับ web/server ต้องมี localhost หรือ VPS check จริง ไม่ใช่คำว่า "น่าจะ"
