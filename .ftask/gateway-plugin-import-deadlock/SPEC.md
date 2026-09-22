# SPEC — gateway-plugin-import-deadlock

> The agent fills this by running the BOUNDED dimensional clarifier (see the
> Spec-first protocol: score Objective/Metric/Target/Scope, ask one question
> per round at the weakest, exit at ambiguity ≤20%), reads it back, and only
> runs `ftask spec gateway-plugin-import-deadlock --approve` once sunke says OK. No code until
> approved. This is the non-coder's real review gate.
>
> The 'How will I know it works' section is the Karpathy gate — `--approve`
> parses it and refuses to flip status if Surface / Acceptance scenarios /
> Regression guards are empty or placeholder. Filling this section honestly
> is what lets the LLM LOOP toward done instead of guessing.

## Done — ONE measurable sentence (fill LAST, after the interview)
> The crisp goal the interview converges to. Must be checkable, not vague —
> it doubles as the future drive-to-green stop condition.
> e.g. 'subscribe-v3 import maps all 7 fields; imported row count = source ±0'.
- Gateway `run` must never start plugin discovery on a background thread before gateway imports complete; the focused launcher regression test and a real local cold start both complete without an import-lock stall, while chat still backgrounds discovery.

## What sunke wants (plain language)  [Objective]
- Eliminate the intermittent startup freeze caused by directory plugins and publish the minimal compatibility fix upstream.

## Out of scope (what we will NOT do)  [Scope]
- Do not change plugin APIs, hook order, TUI behavior, or background discovery for ordinary chat commands.
- Do not include the downstream multitenancy plugin or any local credentials in this PR.

## 任务类型 fix

## 根因 (fix 必填 — 不写根因就会同一个 bug 修两遍)
> 这个缺陷的真实成因是什么?在哪一层进入系统?哪些调用方共享同一个根因?
- `_prepare_agent_startup` backgrounds plugin imports for every agent command, including `gateway`; the main thread subsequently imports gateway modules and synchronously joins discovery, while a directory-plugin discovery thread may wait on modules held by the main thread. This creates a cross-thread Python import-lock cycle. Gateway already owns synchronous discovery on its runtime path, so the eager background start is both redundant and unsafe.

## 拷问(写 Done 前) — 运行 /grilling 与 sunke 对齐到共识;共识即落 Done。(T0/T1 可跳过)

## How will I know it works (Karpathy gate — required to approve)

### Surface (which user-facing surface — pick one or more)
- [ ] web — Interceptor / agent-browser harness
- [x] cli — fresh shell + actual command
- [ ] api — curl against real endpoint
- [ ] lib — 5-line consumer script
- [ ] none — pure doc/config change (no simulate step)

### Visual target (web surface only — 钉死"长成什么样才算对")
> 仅 web surface 任务需填: 参考图路径 / 设计稿 URL / 一句可视判定(如"侧边栏宽 240px、企业名居中")。
> 这是前端视觉验收的对比基准 — 没有它,"看着对"无法机器核验。
- Non-Web task.

### Acceptance scenarios (each = observable user action + observable outcome)
Format: 'user does X → observe Y' (use → to separate action from outcome)
- Start `hermes gateway run` with an enabled directory plugin → observe gateway reaches ready instead of stalling at plugin capability discovery.
- Start ordinary `hermes chat` preparation → observe background plugin discovery still starts once.
- Start the TUI launcher → observe launcher still skips redundant plugin discovery.

### Regression guards (what must NOT break — list things to recheck)
- Gateway runtime still discovers and registers enabled plugins before platform startup.
- Chat startup latency optimization and TUI ownership remain unchanged.
- No plugin-specific special case or new configuration flag is introduced.

### Targeted tests (repo-relative paths; one per bullet, or `full-suite`)
> The direction model lists only tests affected by this task. Invalid/missing targets block; full-suite is CI-only.
- tests/hermes_cli/test_tui_launcher_skips_plugin_discovery.py

## Plan (long tasks only — ordered route + live progress; T1/short may leave empty)
> Steps DERIVED from the Done line (not a chat-plan). Tick `[ ]`→`[x]` as you go.
> This is the compaction-survival anchor: after an auto-compact, read this to see
> exactly which steps are done and what's next — never re-run finished steps.
- [ ] Add a failing gateway-command regression at the public startup-preparation seam.
- [ ] Exclude only gateway from eager background discovery and run focused tests.
- [ ] Rebase through the managed ship gate, obtain independent review, merge the PR, and verify the released branch state.

## Dead ends (filled DURING work — approaches tried & rejected, don't retry)
> Append one line per rejected approach: `approach → why it failed`. Read
> this before each new attempt so the same wrong path isn't tried twice.
- Making only the downstream plugin import-lighter → reduced probability but did not eliminate the core cross-thread import cycle.
