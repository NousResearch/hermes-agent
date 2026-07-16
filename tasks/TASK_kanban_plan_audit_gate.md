# Kanban Plan-Audit Gate (V2.1 — Slice 1/3)

## Mode

`guarded`

## Goal

- Thêm một "plan-audit gate" vào Kanban: task được đánh dấu `plan_audit_required=true`
  không được dispatcher claim vào `running` cho tới khi có verdict `approved` từ
  auxiliary auditor (`auxiliary.plan_auditor`), tối đa `kanban.plan_audit_max_rounds`
  vòng. Hết số vòng mà chưa approved thì task chuyển `blocked` kèm event
  `plan_audit_exhausted`.
- Không xây orchestrator/DB/workflow-engine mới. Toàn bộ nằm trong schema và
  dispatcher Kanban hiện có (`tasks`, `task_events`, `task_comments`, `claim_task`).
- Đây là slice 1/3 của baseline V2.1 (xem "Ghi Chú Phạm Vi" cuối file). Slice 2
  (budget enforcement cấp task/run) và slice 3 (skill "Kanban Orchestrated Coding"
  + tái dùng `hermes kanban specify`/`auxiliary.triage_specifier`) là runsheet
  riêng, không nằm trong phạm vi này.

## Scope

Allowed:
- Thêm config keys mới dưới `auxiliary.*` (`auxiliary.plan_auditor`) và `kanban.*`
  (`kanban.plan_audit_required` mặc định `false`, `kanban.plan_audit_max_rounds`
  mặc định `2` — khớp `kanban.failure_limit` và Repeated-Blocker Rule) trong
  `DEFAULT_CONFIG` (`hermes_cli/config.py`) + tài liệu trong `cli-config.yaml.example`.
- Thêm gate check bên trong `claim_task()`
  ([hermes_cli/kanban_db.py:3372](hermes_cli/kanban_db.py:3372)), theo đúng pattern
  "structural invariant" đã có cho parent-dependency (dòng 3388-3406: task không
  được `ready -> running` nếu còn parent chưa `done`; gate mới áp dụng cùng cơ chế
  cho điều kiện "plan chưa được audit approve").
- Thêm event kind mới: `plan_audit_requested`, `plan_audit_approved`,
  `plan_audit_rejected`, `plan_audit_exhausted` — **chỉ sau khi** kiểm tra dashboard
  (`plugins/kanban/dashboard/plugin_api.py`), `hermes kanban watch`/`tail` không có
  enum ngầm nào giả định trước danh sách `kind` cố định.
- Helper gọi `auxiliary.plan_auditor` theo đúng pattern resolve của
  `agent/auxiliary_client.py::_resolve_auto` (dòng 4104) — cùng cách
  `auxiliary.kanban_decomposer`/`auxiliary.triage_specifier` đã được đọc.
- Test mới: unit cho gate logic (approved/rejected/exhausted), integration với
  temp `HERMES_HOME` + temp kanban DB.

Not allowed (ngoài slice này):
- Budget/cost enforcement (`kanban.task_budget.*`, dùng `agent/usage_pricing.py`) —
  V2.1 slice 2, runsheet riêng.
- Skill "Kanban Orchestrated Coding", thay đổi `hermes kanban specify` /
  `auxiliary.triage_specifier` — V2.1 slice 3, runsheet riêng.
- OpenClaw, dashboard UI mới, bất kỳ cơ chế filesystem/worktree mới nào (đã có
  `resolve_workspace()` tại [hermes_cli/kanban_db.py:5505](hermes_cli/kanban_db.py:5505),
  không đụng vào).
- Auto-commit, push, deploy.
- Budget/giới hạn cấp board/ngày/account.
- Sửa hành vi `promote_task()` (dòng 4757) hay `decompose_triage_task()` (dòng
  4984) — không liên quan tới ranh giới `ready -> running` mà gate này nhắm tới.

## Source Of Truth

- User request: chuỗi audit trong phiên làm việc này — kế hoạch orchestration v1
  của Codex bị REJECTED (trùng lặp Kanban), v2 (Kanban-first) được APPROVED WITH
  CHANGES, v2.1 chốt baseline cuối gồm đúng 2 phần mới: plan-audit gate + budget
  enforcement, cộng skill/config mỏng nối các phase.
- Repo docs: `AGENTS.md` (mục Kanban, Adding Configuration, Footprint Ladder,
  Prompt Caching Must Not Break), `website/docs/user-guide/features/kanban.md`
  (phần goal-mode — tiền lệ trực tiếp cho "auxiliary judge gate cho một task"),
  skill `kanban-codex-lane` (tiền lệ ownership: bên audit sở hữu quyết định cuối).
- Current state đã xác minh trong phiên này: `claim_task` (kanban_db.py:3372,
  "single enforcement point" cho `ready->running`), `promote_task` (4757),
  `decompose_triage_task` (4984), `dispatch_once`/`_dispatch_once_locked` (6932),
  `resolve_workspace` (5505), `_resolve_auto` (agent/auxiliary_client.py:4104).

## Read Order

1. `AGENTS.md`
2. `PROJECT_RULES.md`
3. `HANDOFF.md`
4. `PROJECT_STRUCTURE.md` (kanban core nằm trong "High-Risk Areas" mục 7 — thay
   đổi cross-file, cần đọc trước khi sửa)
5. `hermes_cli/kanban_db.py` — đọc toàn bộ `claim_task`, `_dispatch_once_locked`,
   và tìm hàm judge-loop của goal-mode hiện có (tên chính xác CHƯA được xác nhận
   trong phiên audit này — việc đầu tiên khi thực thi là `grep` lại, ví dụ
   `judge|goal_mode|_evaluate_goal`, để tái dùng đúng pattern thay vì viết mới)
6. `agent/auxiliary_client.py` — `_resolve_auto` + cách `auxiliary.kanban_decomposer`
   /`auxiliary.triage_specifier` được đọc từ config, dùng làm mẫu cho
   `auxiliary.plan_auditor`
7. `tests/hermes_cli/test_kanban_goal_mode.py`, `test_kanban_promote.py`,
   `test_kanban_db.py`, `test_kanban_reclaim_claim_lock_guard.py` — mẫu test hiện
   có cần soi theo cho style + fixture (temp HERMES_HOME, fake DB)

## Planned Files

- `hermes_cli/config.py` (`DEFAULT_CONFIG`)
- `cli-config.yaml.example` (tài liệu key mới)
- `hermes_cli/kanban_db.py` (gate trong `claim_task`; helper ghi event mới; có
  thể cần thêm cột trên `tasks` để persist round-count nếu không suy ra được đủ
  từ `task_events` — **quyết định lúc investigate, chưa chốt trong runsheet này**)
- `agent/auxiliary_client.py` — CHỈ sửa nếu `_resolve_auto` chưa nhận task-key
  tuỳ ý theo tên (cần xác nhận lúc đọc code; nhiều khả năng không cần sửa vì
  pattern đã generic cho `auxiliary.<task>`)
- `tests/hermes_cli/test_kanban_plan_audit_gate.py` (mới)
- `AGENTS.md` mục Kanban — bổ sung ngắn nếu hành vi public thay đổi (một đoạn,
  không viết lại mục)

## Mini-Plan

- Files to inspect/edit: như "Planned Files"
- Commands to run: `scripts/run_tests.sh tests/hermes_cli/test_kanban_plan_audit_gate.py -q`,
  mở rộng dần sang regression suite liên quan `claim_task`
- Live provider/gateway/cron/browser state touched: no — test dùng fake/mocked
  auxiliary client theo convention ad hoc monkeypatch hiện có (không có fixture
  fake-LLM tập trung trong repo; xem `tests/run_agent/conftest.py` làm mẫu).
  Hành vi runtime thật của tính năng (gọi `auxiliary.plan_auditor` thật khi có
  người dùng bật `plan_audit_required`) nằm ngoài phạm vi validation của slice này.
- Stop conditions: cùng lỗi 2 lần → `BLOCKED` (Repeated-Blocker Rule); nếu vị trí
  hook trong `claim_task` đụng race-condition/lock invariant hiện có (comment RCA
  tại dòng 3395 cho thấy từng có bug thật ở đúng vùng này) → dừng, báo cáo trước
  khi sửa tiếp, không tự ý đổi logic lock hiện hữu.
- Validation plan: unit test gate (approved/rejected/exhausted, mock auxiliary
  client) + integration test với temp `HERMES_HOME`/temp kanban DB xác nhận: (a)
  task `plan_audit_required=true` không thể vào `running` khi chưa có event
  `plan_audit_approved` mới nhất; (b) task vào `blocked` kèm event
  `plan_audit_exhausted` đúng sau `plan_audit_max_rounds` vòng reject liên tiếp;
  (c) task `plan_audit_required=false` (mặc định) hành vi không đổi so với hiện tại
  — regression an toàn cho người dùng không bật tính năng.

## Safety Audit

- No destructive command without approval: đúng, không có lệnh huỷ/xoá
- No live external/provider side effect without approval: đúng — test không gọi
  API thật
- No secrets printed or committed: đúng
- No generated/local/runtime files staged: đúng — chỉ code/test/doc
- No unrelated dirty state hidden: **cần `git status` trước khi bắt đầu** — tại
  thời điểm viết runsheet này, repo có thay đổi chưa liên quan đang pending
  (`.gitignore`, `apps/desktop/src/app/shell/model-menu-panel.tsx`); KHÔNG gộp
  các thay đổi đó vào commit của task này

## Validation Commands

- `scripts/run_tests.sh tests/hermes_cli/test_kanban_plan_audit_gate.py -q`
- `scripts/run_tests.sh tests/hermes_cli/test_kanban_db.py tests/hermes_cli/test_kanban_promote.py tests/hermes_cli/test_kanban_reclaim_claim_lock_guard.py -q`
  (regression quanh `claim_task`)
- `scripts/run_tests.sh tests/hermes_cli/ -q` (trước khi coi là `DONE`)

## Attempt Ledger

_(để trống — điền khi thực thi, không phải lúc lập plan)_

Attempt 1:
- Goal: implement slice 1 plan-audit gate at `claim_task()`.
- Change or action: added per-task DB columns, verdict events, `record_plan_audit_verdict()`, gate enforcement before `ready -> running`, CLI/tool/dashboard opt-in fields, config defaults for `kanban.plan_audit_*` and `auxiliary.plan_auditor`, and focused tests.
- Validation command: `uv run --with pytest python -m pytest tests/hermes_cli/test_kanban_plan_audit_gate.py -q`
- Result: PASS, 8 passed.
- Artifact/log: `tests/hermes_cli/test_kanban_plan_audit_gate.py`.
- Error signature: none.
- Hypothesis after result: gate invariant is covered for default-off, requested, approved, rejected, exhausted, migration, and dispatcher no-spawn behavior.

Attempt 2:
- Goal: focused regression around nearby Kanban behavior.
- Change or action: ran goal-mode/promote/CLI tests plus the new gate tests.
- Validation command: `uv run --with pytest python -m pytest tests/hermes_cli/test_kanban_plan_audit_gate.py tests/hermes_cli/test_kanban_goal_mode.py tests/hermes_cli/test_kanban_promote.py tests/hermes_cli/test_kanban_cli.py -q`
- Result: PASS, 83 passed.
- Artifact/log: terminal output in this session.
- Error signature: none.
- Hypothesis after result: narrow Kanban behavior around claim/promote/CLI remains intact.

Attempt 3:
- Goal: broader regression attempt requested by validation plan.
- Change or action: tried project runner, then native Windows `uv run` fallback.
- Validation command: `bash scripts/run_tests.sh tests/hermes_cli/test_kanban_plan_audit_gate.py -q`
- Result: NOT RUN by script; WSL-side runner could not find a POSIX venv (`no virtualenv found in /mnt/d/OneDrive/Hermes/.venv or .../venv`).
- Artifact/log: terminal output in this session.
- Error signature: Windows `.venv` created by `uv` has `Scripts/`, while `scripts/run_tests.sh` running under WSL expects POSIX venv layout.
- Hypothesis after result: use `uv run --with pytest ...` on this native Windows checkout unless a WSL/POSIX venv is created.

Attempt 4:
- Goal: broad adjacent regression signal.
- Change or action: ran `test_kanban_db.py`, `test_kanban_promote.py`, `test_kanban_reclaim_claim_lock_guard.py` with `uv run --with pytest`.
- Validation command: `uv run --with pytest python -m pytest tests/hermes_cli/test_kanban_db.py tests/hermes_cli/test_kanban_promote.py tests/hermes_cli/test_kanban_reclaim_claim_lock_guard.py -q`
- Result: PARTIAL; 225 passed, 16 failed.
- Artifact/log: terminal output in this session.
- Error signature: failures were native-Windows/environment-sensitive paths (`os.waitpid` zombie tests, POSIX `true`, uv hermes shim resolution, Git worktree slash formatting, rate-limit/protocol-exit classification affected by Windows process semantics).
- Hypothesis after result: failures are not caused by the plan-audit gate; keep them as validation caveat for a Windows test-environment follow-up.

## Execution Log

- Implemented per-task `plan_audit_required` and `plan_audit_max_rounds` in `hermes_cli/kanban_db.py`, including fresh schema, additive migration, `Task.from_row()`, `create_task()`, and event payloads.
- Added `record_plan_audit_verdict()` as the public DB helper for future skill/worker code to record approved/rejected verdicts without touching private event helpers.
- Added gate enforcement in `claim_task()` after parent invariant and before stale-run recovery/claim CAS. The gate never calls an LLM while holding the SQLite write transaction.
- Added CLI/tool/dashboard create fields for opt-in plan audit tasks and registered `auxiliary.plan_auditor` config/menu entry for later slice usage.
- Added explicit CLI/tool warnings that a `plan_audit_required` task remains ready/unclaimed until an auditor caller records verdict events; slice 1 does not run that caller.
- Added `tests/hermes_cli/test_kanban_plan_audit_gate.py`.

## Scope Expansion Note

- The original Planned Files named `hermes_cli/config.py`, `cli-config.yaml.example`,
  `hermes_cli/kanban_db.py`, `agent/auxiliary_client.py` if needed, tests, and
  possibly `AGENTS.md`.
- Implementation deliberately touched CLI/tool/dashboard surfaces too:
  `hermes_cli/kanban.py`, `tools/kanban_tools.py`,
  `plugins/kanban/dashboard/plugin_api.py`, and `hermes_cli/main.py`.
- Rationale: once the DB gate exists, users and orchestrator workers need a
  supported opt-in path and config UI entry, matching the existing `goal_mode`
  exposure pattern. `agent/auxiliary_client.py` did not need changes because
  auxiliary task-key resolution is generic.

## Changed Files

- `hermes_cli/kanban_db.py`
- `tests/hermes_cli/test_kanban_plan_audit_gate.py`
- `hermes_cli/kanban.py`
- `tools/kanban_tools.py`
- `plugins/kanban/dashboard/plugin_api.py`
- `hermes_cli/config.py`
- `hermes_cli/main.py`
- `cli-config.yaml.example`
- `HANDOFF.md`
- `tasks/TASK_kanban_plan_audit_gate.md`

## Final Report Checklist

- Status reported
- Changed files listed
- Commands and validation results listed
- Artifacts listed, or `none`
- Handoff update noted
- Remaining risk noted

---

## Ghi Chú Phạm Vi (ngoài template)

Runsheet này là **slice 1/3** của baseline V2.1 đã chốt trong audit trước:

1. **Slice 1 (runsheet này)** — Plan-audit gate trong `claim_task`.
2. **Slice 2** — Budget enforcement cấp Kanban task/run (`kanban.task_budget.*`,
   dùng `agent/usage_pricing.py` đã có cho cost computation; enforcement là phần
   mới thật sự). Runsheet riêng, phụ thuộc slice 1 chỉ ở việc dùng chung namespace
   config `kanban.*`, không phụ thuộc code.
3. **Slice 3** — Skill "Kanban Orchestrated Coding" nối các phase
   (`intake/spec -> plan -> plan_audit -> execute -> verify -> code_audit ->
   synthesize`), tái dùng `hermes kanban specify`/`auxiliary.triage_specifier` cho
   pha intake/spec. Phụ thuộc slice 1 (cần gate đã tồn tại để skill hướng dẫn model
   dùng đúng).

Không nên gộp cả 3 slice vào một phiên "guarded" duy nhất — vi phạm nguyên tắc
"smallest safe change" trong `PROJECT_RULES.md` và tăng diện tích rủi ro trên một
module đã có RCA lock-race trong lịch sử (`claim_task`).
