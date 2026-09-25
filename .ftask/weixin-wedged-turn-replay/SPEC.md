# SPEC — weixin-wedged-turn-replay

> The agent fills this by running the BOUNDED dimensional clarifier (see the
> Spec-first protocol: score Objective/Metric/Target/Scope, ask one question
> per round at the weakest, exit at ambiguity ≤20%), reads it back, and only
> runs `ftask spec weixin-wedged-turn-replay --approve` once sunke says OK. No code until
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
- A gateway turn blocked while loading subdirectory hints is bounded and cancelled without leaving a live worker, and its timed-out inbound message is closed exactly once without being re-enqueued or executed as a later fresh turn, as proven by the targeted agent and gateway regression tests.

## What sunke wants (plain language)  [Objective]
- 微信/飞书长任务即使碰到挂死目录，也必须在可控时间内失败并释放会话；超时后不能继续后台执行，不能把旧消息重新当新消息回答。

## Out of scope (what we will NOT do)  [Scope]
- 不调整模型选择、搜索业务逻辑、微信/飞书 UI、消息平台协议或全局超时默认值。
- 不改正常的瞬时 provider 失败重试语义；仅收敛 gateway watchdog 判定的卡死 turn。

## 任务类型 fix

## 根因 (fix 必填 — 不写根因就会同一个 bug 修两遍)
> 这个缺陷的真实成因是什么?在哪一层进入系统?哪些调用方共享同一个根因?
- `agent/subdirectory_hints.py` 在工具调用前同步执行 `Path.is_file()` / `Path.read_text()`，遇到不可响应的挂载目录时没有 I/O 边界，worker 可永久阻塞。
- gateway watchdog 的 hard interrupt 只能设置 agent 中断状态，不能中断卡在 Python 文件 I/O 中的 worker；gateway 已返回超时后旧 worker仍可能恢复并继续产生工具副作用。
- watchdog 超时被统一归为 transient failure，失败路径持久化原 user turn；同时同一会话的待处理消息仍可排队，导致旧请求在超时边界后被再次视作新工作执行。

## 拷问(写 Done 前) — 运行 /grilling 与 sunke 对齐到共识;共识即落 Done。(T0/T1 可跳过)

## How will I know it works (Karpathy gate — required to approve)

### Surface (which user-facing surface — pick one or more)
- [ ] web — Interceptor / agent-browser harness
- [ ] cli — fresh shell + actual command
- [ ] api — curl against real endpoint
- [x] lib — 5-line consumer script
- [ ] none — pure doc/config change (no simulate step)

### Visual target (web surface only — 钉死"长成什么样才算对")
> 仅 web surface 任务需填: 参考图路径 / 设计稿 URL / 一句可视判定(如"侧边栏宽 240px、企业名居中")。
> 这是前端视觉验收的对比基准 — 没有它,"看着对"无法机器核验。
- 

### Acceptance scenarios (each = observable user action + observable outcome)
Format: 'user does X → observe Y' (use → to separate action from outcome)
- agent 在工具调用前遇到一个会阻塞的 subdirectory hint 路径 → hint 读取在短边界内放弃且工具执行链继续/受控失败，不会无限占用 worker。
- gateway watchdog 判定 turn inactivity timeout → 旧 turn 被标记为终止且在稍后解除 I/O 阻塞后也不能继续执行后续工具或覆盖响应。
- 同一 inbound message 因 watchdog timeout 失败 → transcript 形成单一 user/assistant 关闭边界，平台重投或队列 drain 不会把该旧请求重新当作新的 user turn 执行。

### Regression guards (what must NOT break — list things to recheck)
- 正常目录中的 `AGENTS.md` / hint 文件仍完整加载，缓存和父目录顺序不变。
- 普通 provider 429/连接失败仍保留用户输入供下一次显式重试，不改变既有 transient-failure 合同。
- `/stop`、`/new`、新消息排队和后台进程清理的既有 gateway 行为保持通过。

### Targeted tests (repo-relative paths; one per bullet, or `full-suite`)
> The direction model lists only tests affected by this task. Invalid/missing targets block; full-suite is CI-only.
- tests/agent/test_subdirectory_hints.py
- tests/agent/test_tool_call_guardrail_runtime.py
- tests/gateway/test_abandoned_turn_process_cleanup.py
- tests/gateway/test_failure_writer_ownership.py
- tests/gateway/test_session_race_guard.py

## Plan (long tasks only — ordered route + live progress; T1/short may leave empty)
> Steps DERIVED from the Done line (not a chat-plan). Tick `[ ]`→`[x]` as you go.
> This is the compaction-survival anchor: after an auto-compact, read this to see
> exactly which steps are done and what's next — never re-run finished steps.
- [ ] 为 subdirectory hint 文件探测/读取增加可测试的有界 I/O，并覆盖挂死路径。
- [ ] 为 watchdog timeout 建立终止世代/租约合同，阻止迟到 worker 继续产生效果。
- [ ] 区分 watchdog timeout 与可重试 provider failure，关闭一次失败 turn 且不重排旧请求。
- [ ] 运行 targeted tests、ftask review 与 ship gates，读取 PR/合并结果。

## Dead ends (filled DURING work — approaches tried & rejected, don't retry)
> Append one line per rejected approach: `approach → why it failed`. Read
> this before each new attempt so the same wrong path isn't tried twice.
- 
